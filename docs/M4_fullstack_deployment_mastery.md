# Milestone 4 源码精讲与上海一面深挖手记（FastAPI SSE / Vue 3 工作台 / Docker 生产交付）

## 一、 面试官最爱问的 4 个全栈 Agent 生产工程细节（全部体现在 M4 源码中！）

1. **为什么用 POST Streamable HTTP / SSE，而不用浏览器原生 `EventSource` 或 WebSocket？**
   * 浏览器原生 `EventSource` 只支持 `GET` 请求（无法带复杂 JSON Body 和自定义鉴权 Header）；WebSocket 双向长连接在企业内网防火墙/负载均衡下运维成本高。
   * **最佳实践（`useHarnessStream.ts` + `main.py`）**：用标准 `POST /api/v1/chat/stream` 返回 `text/event-stream`，前端通过 `fetch` + `ReadableStream` 按 `\n\n` 解析事件帧，既支持任意 JSON 请求体与 Token 鉴权，又天然支持服务端单向实时推流（思考链、工具卡、审批卡片、流式 Token），配合 `POST /cancel` 和 `POST /approvals/{id}/resolve` 形成极其优雅的 REST+SSE 混合控制面！
2. **线上部署后“流式输出突然变成憋 10 秒一次性蹦出来”是什么经典大坑？**
   * **Nginx 缓冲坑！** 默认 Nginx 会攒满缓冲区才转发给浏览器。必须在后端响应头加 `X-Accel-Buffering: no`（见 `main.py` 第 139 行），且在 `web/nginx.conf` 中显式配置 `proxy_buffering off; proxy_cache off; proxy_read_timeout 300s;`！
3. **客户端突然关闭浏览器标签页时，后台 Agent 会继续空烧大模型 Token 吗？**
   * 不会！看 `main.py` 第 131 行 `finally` 块：流式生成器检测到连接断开退出时，若 `agent_task` 仍在运行，立即触发 `cancel_token.cancel("client_disconnected")`，秒级回收后台协程并终止工具调用！
4. **为什么 `web/Dockerfile` 要用多阶段构建（Multi-stage Build）？**
   * 阶段 1 用 `node:20-alpine` 编译 TypeScript/Vue 产出纯静态 `dist/`；阶段 2 扔掉几百 MB 的 `node_modules` 和源码，只把 `dist/` 拷进不到 40MB 的 `nginx:1.27-alpine` 镜像——极致精简、减少攻击面、拉取部署秒级完成。
