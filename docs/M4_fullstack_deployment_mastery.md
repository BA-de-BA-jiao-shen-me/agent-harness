# Milestone 4（企业升级版）源码精讲：多 Pod 分布式协调、SSE 断线重连、OTel/Prometheus 可观测与 Vue 3 企业工作台

> **对应最新源码**：
> - [coordination.py](file:///D:/Code/agent-harness/pyharness/infra/coordination.py) | [telemetry.py](file:///D:/Code/agent-harness/pyharness/infra/telemetry.py)
> - [main.py](file:///D:/Code/agent-harness/pyharness/api/main.py) | [deps.py](file:///D:/Code/agent-harness/pyharness/api/deps.py)
> - [App.vue](file:///D:/Code/agent-harness/web/src/App.vue) | [useHarnessStream.ts](file:///D:/Code/agent-harness/web/src/composables/useHarnessStream.ts) | [docker-compose.yml](file:///D:/Code/agent-harness/docker-compose.yml)

---

## 一、 从“单机 Demo”跨向“多 Pod 生产集群”的 5 大基础设施升级

在单机测试时，把对话历史、待审批 `Future` 和取消令牌放在 Python 进程内存字典里一切正常；**但一旦上线用 K8s / Docker 起 3 个 FastAPI Worker Pod，内存态架构会瞬间全线崩溃！**
* Pod A 正在跑用户的 Agent 挂起等审批；
* 主管点“批准”的 HTTP 请求被负载均衡打到了 **Pod B** $\rightarrow$ Pod B 内存里根本没有那个 `approval_id`，直接报 404！

### 1. 分布式协调总线（`CoordinationBackend` & `RedisCoordinationBackend` in `infra/coordination.py`）
我们抽离了 `CoordinationBackend` 抽象层（单机无缝跑 `InMemoryCoordinationBackend`，集群配 `REDIS_URL` 秒切 `RedisCoordinationBackend`），用 Redis 一次性打通 **5 大跨 Pod 分布式能力**：
1. **分布式会话互斥锁（`session_lock`）**：同一 `session_id` 同时发来两个并发请求时，基于 Redis `SET NX EX` 加锁，第二个请求直接返回 `409 Conflict`，杜绝同一会话上下文写乱；
2. **多轮历史跨实例持久化（`save_session_history` / `get_session_history`）**：带 TTL 存在 Redis，第 1 轮打到 Pod A、第 2 轮打到 Pod B 照样丝滑衔接；
3. **跨 Pod 人工审批唤醒（`register_pending_approval` + `resolve_approval` + Redis Pub/Sub）**：审批元数据存 Redis Hash，主管审批请求打到任意 Pod B 时，Pod B 校验完四眼权限后往 Redis Pub/Sub 频道广播决定，瞬间唤醒挂起在 Pod A 上的协程！
4. **跨 Pod 会话取消（`publish_cancel` / `wait_for_cancel`）**：用户点停止按钮打到 Pod C，经 Redis Pub/Sub 广播给 Pod A 的 `_watch_remote_cancel()` 协程，毫秒级触发 `CancellationToken`！
5. **分布式滑动窗口 RPM 限流与 FinOps 配额账本**：支撑多 Pod 精确共享每分钟请求数（`429 Too Many Requests` + `Retry-After`）与部门级每日 Token/USD 记账！

---

## 二、 生产级 Streamable HTTP / SSE 与 `Last-Event-ID` 断线重连（`api/main.py`）

### 1. 核心控制面与流式 API 全览
| 端点 | 方法 | 企业级核心作用 |
| :--- | :---: | :--- |
| `/health` | `GET` | 暴露当前运行 Profile、已挂载 Cordis 插件列表、后端存储/沙箱/向量库类型、各 LLM 路由熔断健康状态 |
| `/metrics` | `GET` | **Prometheus 文本格式指标暴露**（Turn 耗时/状态、Token 与 USD 成本、ACP 策略拦截计数、工具耗时与错误率、安全告警数、待审批 Gauge） |
| `/api/v1/chat/stream` | `POST` | 主对话流式 SSE 接口（全套鉴权 + FinOps 配额 + 分布式会话锁 + 实时推流 + 客户端断连自动取消） |
| `/api/v1/sessions/{id}/stream` | `GET` | **支持 `Last-Event-ID`（或 `?after_seq=`）断线无损重连补发 + 实时续订**！ |
| `/api/v1/sessions/{id}/cancel` | `POST` | 跨 Pod 优雅中断正在运行的会话与子代理 |
| `/api/v1/sessions/{id}/fork` | `POST` | 基于事件溯源从指定 `at_seq` 分叉出平行新会话 |
| `/api/v1/sessions/{id}/audit/verify` | `GET` | 执行 SHA-256 哈希链全量防篡改验签 |
| `/api/v1/approvals/{id}/resolve` | `POST` | 四眼原则（SoD）审批决断接口（强制绑定真实 Principal 身份，防自批防伪造） |
| `/api/v1/admin/usage` | `GET` | 部门/租户级 FinOps Token 与 USD 成本分账报表（需 `admin`/`auditor` 角色） |

### 2. 断线重连是怎么做到“不丢事件、不重复事件”的？
每条 SSE 帧都严格附带单调递增的 `id: {evt.seq}`。浏览器弱网断线重连时，在请求头带上 `Last-Event-ID: 12` 调用 `GET /api/v1/sessions/{id}/stream`：
1. 后端**先订阅**实时队列（防补发窗口漏新事件）；
2. 从 `EventJournal` 回放 `seq > 12` 的持久化历史事件发给前端并记录最大 `last_sent_seq`；
3. 再切入实时队列推送，丢弃 `seq <= last_sent_seq` 的重复事件——实现 **Exactly-Once 有序无损续流**！

---

## 三、 OpenTelemetry GenAI 追踪 + Prometheus 可观测（`infra/telemetry.py`）

* **GenAI 语义约定（Semantic Conventions）**：`TelemetryRecorder.start_span()` 自动把关键属性映射到 OpenTelemetry 官方 GenAI 标准字段（`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.usage.output_tokens`、`gen_ai.tool.name`），直接对接 Jaeger / Grafana Tempo / Langfuse！
* **零依赖优雅降级**：未安装 `opentelemetry-sdk` 时自动退化为 `_NoOpSpan`，同时保留完整的线程安全内存计数器与标准 `/metrics` Prometheus 文本导出，绝不拖累本地开发体验。

---

## 四、 Vue 3 企业工作台 & 6 服务安全加固 Docker Compose

* **Vue 3 企业工作台（`web/src/App.vue` & `useHarnessStream.ts`）**：
  * **身份鉴权栏**：支持一键切换 JWT / API-Key / 匿名开发身份，现场即可演示“Alice 发起高危操作 $\rightarrow$ 切 Bob 主管身份审批放行”的完整四眼闭环！
  * **高阶交互**：思考链折叠、子代理委派卡片、策略拦截/DLP 安全告警徽章、**一键 Fork 会话分叉**、**一键断线重连**、**一键 SHA-256 审计链验签**、主管专属跨用户审批收件箱。
* **Nginx 零缓冲配置（`web/nginx.conf`）**：`proxy_buffering off; proxy_cache off; proxy_read_timeout 300s;` + 后端 `X-Accel-Buffering: no`，彻底消灭流式卡顿。
* **6 服务加固 `docker-compose.yml`**：`api` + `web` + `postgres (16)` + `redis (7)` + `qdrant` + 隔离的 **`docker-socket-proxy`**（不把危险的 `/var/run/docker.sock` 直接裸挂进 API 容器，只通过受限代理暴露最小容器创建接口！）。

---

## 五、 本阶段上海一面高频深挖 5 题

1. **Q：多 Pod 部署时，审批请求和取消请求打到了不同 Pod 怎么处理？**
   * *答*：所有状态（会话锁、对话历史、待审批元数据、配额计数）下沉到 Redis，跨 Pod 审批唤醒与会话取消走 Redis Pub/Sub 广播，实现 API 层彻底无状态水平扩容。
2. **Q：SSE 断线重连时，怎么避免“查库补发完历史后、订阅实时队列前”这几毫秒漏掉一条事件？**
   * *答*：**先订阅实时队列，再查库补发 `after_seq`，最后消费实时队列时按 `seq > last_sent_seq` 去重**——这是分布式流系统经典的“先订阅再回放再去重”范式！
3. **Q：为什么要单独部署一个 `docker-socket-proxy`，而不直接把 `/var/run/docker.sock` 挂载进 FastAPI 容器？**
   * *答*：直接挂载宿主机 `docker.sock` 等同于把宿主机 `root` 权限交给了 API 容器（一旦 API 被攻破立刻宿主机逃逸）；通过代理白名单只开放创建/销毁无网络沙箱容器的最小 API，大幅收敛攻击面。
4. **Q：Prometheus 的 `/metrics` 在多标签下容易出什么线上事故？**
   * *答*：**高基数标签爆炸（High Cardinality）！** 千万不能把 `session_id`、`turn_id` 或 `user_id` 放进 Prometheus Label（几百万会话会撑爆 Prometheus 内存）；只放低基数维度（`tenant_id`, `model`, `status`, `tool_name`, `rule_id`），细粒度单次追踪交给 OpenTelemetry Trace 和 EventJournal！
5. **Q：如果用户短时间内对同一个 `session_id` 狂点两次发送按钮，后端怎么防止对话历史错乱？**
   * *答*：通过 Redis 分布式 `session_lock`（`SET NX EX`）做单会话写互斥，并发冲突直接返回 `409 Conflict`，前端同时置灰发送按钮并切换为“停止中断”。
