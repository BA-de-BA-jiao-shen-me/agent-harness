<script setup lang="ts">
import { ref, watch } from 'vue'
import { useHarnessStream } from './composables/useHarnessStream'

const initialSessionId = `web_${Math.random().toString(36).slice(2, 8)}`
const input = ref('')
const sidebarTab = ref<'waterfall' | 'inbox'>('waterfall')
const inboxNotes = ref<Record<string, string>>({})
const selectedQuestionOptions = ref<string[]>([])
const questionFreeText = ref('')
const newBranchInput = ref('')
const showBranchInput = ref(false)

const {
  sessionId,
  authToken,
  lastSeq,
  isRunning,
  answerText,
  reasoningSteps,
  activeSkills,
  events,
  pendingApproval,
  approvalInbox,
  lastTurnUsage,
  auditVerification,
  policyAlerts,
  subagentEvents,
  pendingQuestion,
  artifacts,
  selectedArtifact,
  activeBranch,
  branches,
  steeringNotices,
  sendMessage,
  reconnectStream,
  cancelTurn,
  forkSession,
  resolveApproval,
  fetchApprovalInbox,
  resolveInboxApproval,
  verifyAuditChain,
  sendSteer,
  sendFollowup,
  answerQuestion,
  inspectArtifact,
  closeArtifactViewer,
  fetchSessionTree,
  createBranch,
  switchBranch,
} = useHarnessStream(initialSessionId)

const presets = [
  'SF6 高压断路器紧急闭锁阈值是多少 MPa？',
  '用 SQL 统计各区域故障数和总停机时长',
  '立即向值班群发送 INC-2026-01 高危紧急告警（测试人工审批）',
  '委派 researcher 和 analyst 子代理排查高危隐患并统计停机时长',
]

watch(
  () => pendingQuestion.value?.question_id,
  () => {
    selectedQuestionOptions.value = []
    questionFreeText.value = ''
  }
)

function submit(text?: string) {
  const q = (text ?? input.value).trim()
  if (!q || isRunning.value) return
  input.value = ''
  sendMessage(q)
}

function handleComposerEnter() {
  if (isRunning.value) {
    handleSteer()
  } else {
    submit()
  }
}

async function handleSteer() {
  const msg = input.value.trim()
  if (!msg || !isRunning.value) return
  input.value = ''
  await sendSteer(msg)
}

async function handleFollowup() {
  const msg = input.value.trim()
  if (!msg || !isRunning.value) return
  input.value = ''
  await sendFollowup(msg)
}

function toggleQuestionOption(optionId: string) {
  if (!pendingQuestion.value) return
  if (pendingQuestion.value.multi_select) {
    const idx = selectedQuestionOptions.value.indexOf(optionId)
    if (idx >= 0) {
      selectedQuestionOptions.value.splice(idx, 1)
    } else {
      selectedQuestionOptions.value.push(optionId)
    }
  } else {
    selectedQuestionOptions.value =
      selectedQuestionOptions.value[0] === optionId ? [] : [optionId]
  }
}

async function handleAnswerQuestion() {
  if (!pendingQuestion.value) return
  const freeText = questionFreeText.value.trim()
  const selected = [...selectedQuestionOptions.value]
  if (!selected.length && !freeText) return
  const ok = await answerQuestion(selected, freeText)
  if (ok) {
    selectedQuestionOptions.value = []
    questionFreeText.value = ''
  }
}

async function handleBranchChange(event: Event) {
  const target = event.target as HTMLSelectElement | null
  if (!target || !target.value) return
  await switchBranch(target.value)
}

async function handleCreateBranch() {
  const name = newBranchInput.value.trim()
  if (!name) return
  const ok = await createBranch(name)
  if (ok) {
    newBranchInput.value = ''
    showBranchInput.value = false
  }
}

function switchTab(tab: 'waterfall' | 'inbox') {
  sidebarTab.value = tab
  if (tab === 'inbox') {
    fetchApprovalInbox()
  }
}

async function handleInboxAction(approvalId: string, approved: boolean) {
  const note = inboxNotes.value[approvalId] || (approved ? 'Approved by supervisor' : 'Rejected by supervisor')
  await resolveInboxApproval(approvalId, approved, note)
  delete inboxNotes.value[approvalId]
}

function eventAccent(eventType: string): string {
  if (eventType.includes('approval')) return '#f59e0b'
  if (eventType.includes('question')) return '#06b6d4'
  if (eventType.includes('steered') || eventType.includes('followup')) return '#f97316'
  if (eventType.includes('artifact')) return '#22c55e'
  if (eventType.includes('branch') || eventType.includes('compaction')) return '#8b5cf6'
  if (eventType.includes('security') || eventType.includes('policy') || eventType === 'error') return '#ef4444'
  if (eventType.includes('subagent')) return '#a855f7'
  if (eventType === 'turn_complete') return '#10b981'
  return '#38bdf8'
}
</script>

<template>
  <div class="workbench-shell">
    <!-- 顶部企业治理导航栏 -->
    <header class="topbar">
      <div class="brand-group">
        <h1 class="brand-title">PyHarness — Enterprise Agent Workbench</h1>
        <span class="session-pill" title="Current active session & latest SSE sequence">
          Session: <code>{{ sessionId }}</code>
          <span class="seq-badge">seq #{{ lastSeq }}</span>
        </span>
        <div class="branch-switcher">
          <span class="branch-label" title="当前会话分支">🌿</span>
          <select
            class="branch-select"
            :value="activeBranch"
            :disabled="isRunning"
            title="切换会话分支"
            @change="handleBranchChange"
          >
            <option
              v-for="b in branches"
              :key="b.branch_name"
              :value="b.branch_name"
            >
              {{ b.branch_name }} ({{ b.event_count }})
            </option>
          </select>
          <button
            class="btn btn-slate btn-sm"
            :disabled="isRunning"
            title="创建新会话分支"
            @click="showBranchInput = !showBranchInput"
          >
            + 分支
          </button>
          <template v-if="showBranchInput">
            <input
              v-model="newBranchInput"
              class="branch-input"
              placeholder="分支名称..."
              @keyup.enter="handleCreateBranch()"
            />
            <button
              class="btn btn-indigo btn-sm"
              :disabled="isRunning || !newBranchInput.trim()"
              @click="handleCreateBranch()"
            >
              创建
            </button>
          </template>
          <button
            class="btn btn-slate btn-sm"
            :disabled="isRunning"
            title="刷新分支树"
            @click="fetchSessionTree()"
          >
            ↻ 树
          </button>
        </div>
      </div>

      <div class="controls-group">
        <input
          v-model="authToken"
          type="password"
          autocomplete="off"
          placeholder="Bearer JWT (eyJ...) 或 X-API-Key"
          class="token-input"
          title="JWT 以 eyJ 开头自动使用 Authorization: Bearer，否则使用 X-API-Key"
        />
        <button class="btn btn-slate" :disabled="isRunning" @click="forkSession()">
          Fork 分叉
        </button>
        <button class="btn btn-slate" :disabled="isRunning" @click="reconnectStream()">
          断线重连
        </button>
        <button class="btn btn-indigo" @click="verifyAuditChain()">
          审计验签
        </button>
      </div>
    </header>

    <!-- SHA-256 审计链验签结果栏 -->
    <div
      v-if="auditVerification"
      class="audit-banner"
      :class="auditVerification.valid ? 'audit-valid' : 'audit-invalid'"
    >
      <strong>
        {{ auditVerification.valid ? '✓ SHA-256 审计哈希链验签通过' : '✗ SHA-256 审计哈希链验签失败' }}
      </strong>
      <span>（已校验 {{ auditVerification.event_count }} 条事件）— {{ auditVerification.detail }}</span>
    </div>

    <!-- 主体两栏工作区 -->
    <div class="workspace-grid">
      <!-- 左侧：流式对话、策略告警、子代理与审批工作台 -->
      <section class="panel main-panel">
        <div class="presets-bar">
          <button
            v-for="p in presets"
            :key="p"
            class="preset-btn"
            :disabled="isRunning"
            @click="submit(p)"
          >
            {{ p }}
          </button>
        </div>

        <!-- 激活 Skills 徽章 -->
        <div v-if="activeSkills.length" class="skills-row">
          <span class="row-label">Activated Skills:</span>
          <span v-for="sk in activeSkills" :key="sk" class="skill-chip">{{ sk }}</span>
        </div>

        <!-- Mid-Turn Steering / Follow-Up 通知条 -->
        <div v-if="steeringNotices.length" class="steering-row">
          <span v-for="(notice, idx) in steeringNotices" :key="idx" class="steering-pill">
            {{ notice }}
          </span>
        </div>

        <!-- Offloaded Artifacts 产物芯片条 -->
        <div v-if="artifacts.length" class="artifacts-row">
          <span class="row-label">Artifacts:</span>
          <button
            v-for="art in artifacts"
            :key="art.artifact_id"
            type="button"
            class="artifact-chip"
            :class="{ active: selectedArtifact?.artifact_id === art.artifact_id }"
            :title="art.preview || art.artifact_id"
            @click="inspectArtifact(art.artifact_id)"
          >
            📦 {{ art.tool_name }} ({{ art.char_count }} chars)
          </button>
        </div>

        <!-- 内联可展开 Artifact Inspector 查看面板 -->
        <div v-if="selectedArtifact" class="artifact-inspector">
          <div class="artifact-inspector-head">
            <div class="artifact-meta">
              <strong>📦 {{ selectedArtifact.tool_name }}</strong>
              <code class="artifact-id">{{ selectedArtifact.artifact_id }}</code>
              <span class="artifact-chars">
                {{ selectedArtifact.total_chars }} chars
                <template v-if="selectedArtifact.has_more">
                  （仅显示前 {{ selectedArtifact.limit }} 字符）
                </template>
              </span>
            </div>
            <button type="button" class="btn btn-slate btn-sm" @click="closeArtifactViewer()">
              关闭
            </button>
          </div>
          <pre class="code-block artifact-content">{{ selectedArtifact.content }}</pre>
        </div>

        <!-- Policy / Security Alert 横幅 -->
        <div v-if="policyAlerts.length" class="alerts-stack">
          <div v-for="(alert, idx) in policyAlerts" :key="idx" class="alert-banner">
            <span>🛡️ {{ alert }}</span>
          </div>
        </div>

        <!-- Subagent Delegation 委派徽章 -->
        <div v-if="subagentEvents.length" class="subagents-row">
          <div
            v-for="(sub, idx) in subagentEvents"
            :key="sub.subagent_session_id || idx"
            class="subagent-badge"
            :class="sub.status === 'running' ? 'sub-running' : 'sub-done'"
          >
            <div class="subagent-head">
              <strong>🤖 Subagent: {{ sub.profile || 'worker' }}</strong>
              <span class="sub-tag">{{ sub.mode || 'one_shot' }}</span>
              <span class="sub-tag">depth {{ sub.depth ?? 1 }}</span>
              <span class="sub-status">{{ sub.status || 'completed' }}</span>
            </div>
            <div class="subagent-meta">
              <span v-if="sub.steps !== undefined">steps={{ sub.steps }}</span>
              <span v-if="sub.usage_tokens !== undefined">tokens={{ sub.usage_tokens }}</span>
              <span v-if="sub.subagent_session_id">id={{ sub.subagent_session_id }}</span>
            </div>
            <div v-if="sub.summary || sub.task" class="subagent-text">
              {{ sub.summary || sub.task }}
            </div>
          </div>
        </div>

        <!-- 可折叠 Thinking / Planning 步骤 -->
        <details v-if="reasoningSteps.length" open class="thinking-box">
          <summary class="thinking-summary">
            Thinking / Planning Steps ({{ reasoningSteps.length }})
          </summary>
          <ul class="thinking-list">
            <li v-for="(s, idx) in reasoningSteps" :key="idx">{{ s }}</li>
          </ul>
        </details>

        <!-- Human-in-the-Loop 高危操作内联审批卡片 -->
        <div v-if="pendingApproval" class="approval-card">
          <div class="approval-title">
            ⚠️ 高危操作待人工审批：{{ pendingApproval.tool_name }}
          </div>
          <div class="approval-reason">{{ pendingApproval.reason }}</div>
          <pre class="code-block">{{ JSON.stringify(pendingApproval.arguments, null, 2) }}</pre>
          <div class="action-row">
            <button class="btn btn-green" @click="resolveApproval(true, 'Approved in UI')">
              批准执行
            </button>
            <button class="btn btn-red" @click="resolveApproval(false, 'Rejected by operator')">
              拒绝拦截
            </button>
          </div>
        </div>

        <!-- Interactive Clarification 交互式澄清卡片 -->
        <div v-if="pendingQuestion" class="question-card">
          <div class="question-title">
            ❓ Agent 请求业务澄清：{{ pendingQuestion.question }}
          </div>
          <div v-if="pendingQuestion.options.length" class="question-options">
            <button
              v-for="opt in pendingQuestion.options"
              :key="opt.id"
              type="button"
              class="option-pill"
              :class="{ selected: selectedQuestionOptions.includes(opt.id) }"
              @click="toggleQuestionOption(opt.id)"
            >
              <span class="option-label">{{ opt.label }}</span>
              <span v-if="opt.description" class="option-desc">{{ opt.description }}</span>
            </button>
          </div>
          <div v-if="pendingQuestion.multi_select && pendingQuestion.options.length" class="question-hint">
            支持多选（已选 {{ selectedQuestionOptions.length }} 项）
          </div>
          <input
            v-if="pendingQuestion.allow_free_text"
            v-model="questionFreeText"
            class="question-input"
            placeholder="补充说明或自由输入回答..."
            @keyup.enter="handleAnswerQuestion()"
          />
          <div class="action-row">
            <button
              class="btn btn-cyan"
              :disabled="!selectedQuestionOptions.length && !questionFreeText.trim()"
              @click="handleAnswerQuestion()"
            >
              提交澄清回答
            </button>
          </div>
        </div>

        <!-- 流式输出主区域 -->
        <main class="answer-box">
          {{
            answerText ||
            '请输入业务问题，或点击上方快捷场景体验 Hybrid RAG / SQL 并发沙箱 / 子代理委派 / 高危人工审批与 SHA-256 审计验签。'
          }}
        </main>

        <!-- 输入与发送/纠偏/追加/停止控制栏 -->
        <footer class="composer">
          <input
            v-model="input"
            class="composer-input"
            :placeholder="
              isRunning
                ? '运行中：输入指令可点击「⚡ 纠偏」立即转向，或「📥 追加」轮后自动执行...'
                : '输入业务问题或分析指令...'
            "
            @keyup.enter="handleComposerEnter()"
          />
          <button v-if="!isRunning" class="btn btn-primary" @click="submit()">发送</button>
          <template v-else>
            <button
              class="btn btn-amber"
              :disabled="!input.trim()"
              title="立即中断当前执行计划并按新指令纠偏"
              @click="handleSteer()"
            >
              ⚡ 纠偏
            </button>
            <button
              class="btn btn-indigo"
              :disabled="!input.trim()"
              title="排队等待本轮完成后自动执行追加任务"
              @click="handleFollowup()"
            >
              📥 追加
            </button>
            <button class="btn btn-red" @click="cancelTurn()">停止</button>
          </template>
        </footer>
      </section>

      <!-- 右侧：Event Waterfall 与 主管审批收件箱 (Inbox) 双标签侧栏 -->
      <aside class="panel sidebar-panel">
        <div class="tabs-bar">
          <button
            class="tab-btn"
            :class="{ active: sidebarTab === 'waterfall' }"
            @click="switchTab('waterfall')"
          >
            Event Waterfall ({{ events.length }})
          </button>
          <button
            class="tab-btn"
            :class="{ active: sidebarTab === 'inbox' }"
            @click="switchTab('inbox')"
          >
            主管审批收件箱 (Inbox)
            <span v-if="approvalInbox.length" class="inbox-count">{{ approvalInbox.length }}</span>
          </button>
        </div>

        <!-- Tab 1: Event Waterfall -->
        <div v-if="sidebarTab === 'waterfall'" class="tab-body">
          <div v-if="lastTurnUsage" class="usage-card">
            Tokens: Prompt={{ lastTurnUsage.prompt_tokens }} / Completion={{ lastTurnUsage.completion_tokens }}
            <span v-if="lastTurnUsage.prompt_cache_hit_tokens">
              / Cache={{ lastTurnUsage.prompt_cache_hit_tokens }}
            </span>
            | Est. Cost=${{ lastTurnUsage.estimated_cost_usd }}
          </div>

          <div v-if="!events.length" class="empty-hint">
            暂无事件流。发送消息或点击「断线重连」后，此处将实时展示 Event Journal 瀑布流。
          </div>

          <div
            v-for="ev in events"
            :key="ev.id"
            class="event-card"
            :style="{ borderLeftColor: eventAccent(ev.eventType) }"
          >
            <div class="event-header">
              <strong>
                {{ ev.eventType }}
                <span v-if="ev.seq !== undefined" class="ev-seq">#{{ ev.seq }}</span>
              </strong>
              <span>{{ ev.timestamp }}</span>
            </div>
            <pre class="event-payload">{{ JSON.stringify(ev.payload).slice(0, 260) }}</pre>
          </div>
        </div>

        <!-- Tab 2: 主管审批收件箱 (跨用户 SoD 复核) -->
        <div v-else class="tab-body">
          <div class="inbox-toolbar">
            <span class="inbox-hint">
              租户级待审队列（支持切换主管 Bearer/API-Key 进行跨用户 SoD 审批）
            </span>
            <button class="btn btn-slate btn-sm" @click="fetchApprovalInbox()">刷新 Inbox</button>
          </div>

          <div v-if="!approvalInbox.length" class="empty-hint">
            当前无待审批的高危工具请求。
          </div>

          <div v-for="item in approvalInbox" :key="item.approval_id" class="inbox-card">
            <div class="inbox-card-head">
              <strong>{{ item.tool_name }}</strong>
              <code class="inbox-id">{{ item.approval_id }}</code>
            </div>
            <div class="inbox-meta">
              <span v-if="item.session_id">Session: {{ item.session_id }}</span>
              <span v-if="item.requester_id || item.requester">
                Requester: {{ item.requester_id || item.requester }}
              </span>
            </div>
            <div class="approval-reason">{{ item.reason }}</div>
            <pre class="code-block">{{ JSON.stringify(item.arguments, null, 2) }}</pre>
            <input
              v-model="inboxNotes[item.approval_id]"
              class="note-input"
              placeholder="审批意见（可选）..."
            />
            <div class="action-row">
              <button class="btn btn-green btn-sm" @click="handleInboxAction(item.approval_id, true)">
                批准 (Approve)
              </button>
              <button class="btn btn-red btn-sm" @click="handleInboxAction(item.approval_id, false)">
                拒绝 (Reject)
              </button>
            </div>
          </div>
        </div>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.workbench-shell {
  min-height: 100vh;
  padding: 16px 20px;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  gap: 12px;
  background: #0f172a;
  color: #f8fafc;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.topbar {
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 12px;
  padding: 12px 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}

.brand-group {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.brand-title {
  margin: 0;
  font-size: 18px;
  font-weight: 700;
  color: #f8fafc;
}

.session-pill {
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 12px;
  color: #cbd5e1;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.seq-badge {
  background: #1d4ed8;
  color: #eff6ff;
  border-radius: 999px;
  padding: 1px 7px;
  font-size: 11px;
}

.controls-group {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.token-input {
  width: 240px;
  background: #0f172a;
  color: #e2e8f0;
  border: 1px solid #475569;
  border-radius: 7px;
  padding: 6px 10px;
  font-size: 12px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}

.audit-banner {
  border-radius: 8px;
  padding: 8px 14px;
  font-size: 13px;
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

.audit-valid {
  background: #052e16;
  border: 1px solid #16a34a;
  color: #86efac;
}

.audit-invalid {
  background: #450a0a;
  border: 1px solid #dc2626;
  color: #fca5a5;
}

.workspace-grid {
  flex: 1;
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(320px, 1fr);
  gap: 16px;
  min-height: 0;
}

.panel {
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 12px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.sidebar-panel {
  max-height: calc(100vh - 115px);
  overflow: hidden;
}

.presets-bar {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.preset-btn {
  background: #334155;
  color: #f8fafc;
  border: 1px solid #475569;
  border-radius: 6px;
  padding: 6px 10px;
  cursor: pointer;
  font-size: 12px;
}

.preset-btn:hover:not(:disabled) {
  background: #475569;
}

.skills-row {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  font-size: 12px;
}

.row-label {
  color: #94a3b8;
}

.skill-chip {
  background: #0c4a6e;
  color: #7dd3fc;
  border: 1px solid #0284c7;
  border-radius: 999px;
  padding: 2px 9px;
  font-size: 11px;
}

.alerts-stack {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.alert-banner {
  background: #450a0a;
  border: 1px solid #ef4444;
  color: #fecaca;
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 12px;
}

.subagents-row {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.subagent-badge {
  background: #1e1b4b;
  border: 1px solid #6366f1;
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 12px;
}

.sub-running {
  border-color: #a855f7;
}

.sub-done {
  border-color: #10b981;
}

.subagent-head {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  color: #e0e7ff;
}

.sub-tag {
  background: #312e81;
  color: #c7d2fe;
  border-radius: 4px;
  padding: 1px 6px;
  font-size: 11px;
}

.sub-status {
  margin-left: auto;
  font-size: 11px;
  color: #6ee7b7;
  text-transform: uppercase;
}

.subagent-meta {
  display: flex;
  gap: 10px;
  color: #94a3b8;
  font-size: 11px;
  margin-top: 4px;
}

.subagent-text {
  margin-top: 4px;
  color: #cbd5e1;
  white-space: pre-wrap;
}

.thinking-box {
  background: #0f172a;
  border: 1px solid #334155;
  padding: 10px 12px;
  border-radius: 8px;
}

.thinking-summary {
  cursor: pointer;
  color: #fbbf24;
  font-size: 13px;
  font-weight: 600;
}

.thinking-list {
  margin: 8px 0 0 18px;
  font-size: 13px;
  color: #cbd5e1;
}

.approval-card,
.inbox-card {
  background: #451a03;
  border: 1px solid #f59e0b;
  border-radius: 10px;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.inbox-card {
  background: #0f172a;
  border-color: #d97706;
}

.approval-title {
  font-weight: 700;
  color: #fcd34d;
  font-size: 14px;
}

.approval-reason {
  font-size: 12px;
  color: #fde68a;
}

.code-block {
  margin: 0;
  background: #020617;
  color: #e2e8f0;
  padding: 8px;
  border-radius: 6px;
  font-size: 11px;
  overflow: auto;
}

.answer-box {
  flex: 1;
  min-height: 220px;
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 8px;
  padding: 14px;
  white-space: pre-wrap;
  line-height: 1.65;
  font-size: 14px;
  overflow: auto;
}

.composer {
  display: flex;
  gap: 8px;
}

.composer-input {
  flex: 1;
  background: #0f172a;
  color: #fff;
  border: 1px solid #475569;
  border-radius: 8px;
  padding: 10px 12px;
  font-size: 14px;
}

.tabs-bar {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
}

.tab-btn {
  background: #0f172a;
  color: #94a3b8;
  border: 1px solid #334155;
  border-radius: 8px;
  padding: 8px 10px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
}

.tab-btn.active {
  background: #1d4ed8;
  border-color: #3b82f6;
  color: #ffffff;
}

.inbox-count {
  background: #f59e0b;
  color: #0f172a;
  border-radius: 999px;
  padding: 0 6px;
  font-size: 11px;
  font-weight: 700;
}

.tab-body {
  flex: 1;
  overflow: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding-right: 2px;
}

.usage-card {
  background: #0f172a;
  border: 1px solid #1e3a8a;
  padding: 8px 10px;
  border-radius: 8px;
  font-size: 12px;
  color: #34d399;
}

.event-card {
  background: #0f172a;
  padding: 8px 10px;
  border-radius: 6px;
  font-size: 12px;
  border-left: 3px solid #38bdf8;
}

.event-header {
  display: flex;
  justify-content: space-between;
  color: #93c5fd;
  font-size: 12px;
}

.ev-seq {
  color: #64748b;
  font-weight: 400;
  margin-left: 4px;
}

.event-payload {
  margin: 4px 0 0;
  white-space: pre-wrap;
  word-break: break-word;
  color: #cbd5e1;
  font-size: 11px;
}

.inbox-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
}

.inbox-hint {
  font-size: 11px;
  color: #94a3b8;
}

.inbox-card-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  color: #fbbf24;
  font-size: 13px;
}

.inbox-id {
  font-size: 11px;
  color: #94a3b8;
}

.inbox-meta {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  font-size: 11px;
  color: #94a3b8;
}

.note-input {
  background: #1e293b;
  color: #f8fafc;
  border: 1px solid #475569;
  border-radius: 6px;
  padding: 6px 8px;
  font-size: 12px;
}

.action-row {
  display: flex;
  gap: 8px;
}

.empty-hint {
  color: #64748b;
  font-size: 12px;
  text-align: center;
  padding: 24px 12px;
}

.btn {
  border: none;
  border-radius: 7px;
  padding: 7px 14px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  color: #ffffff;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-sm {
  padding: 5px 10px;
  font-size: 11px;
}

.btn-primary {
  background: #2563eb;
  padding: 10px 18px;
  font-size: 13px;
}

.btn-slate {
  background: #334155;
  border: 1px solid #475569;
}

.btn-indigo {
  background: #4f46e5;
}

.btn-green {
  background: #16a34a;
}

.btn-red {
  background: #dc2626;
}

.btn-amber {
  background: #d97706;
}

.btn-cyan {
  background: #0891b2;
}

.branch-switcher {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 8px;
  padding: 4px 8px;
}

.branch-label {
  font-size: 13px;
}

.branch-select {
  background: #1e293b;
  color: #e2e8f0;
  border: 1px solid #475569;
  border-radius: 6px;
  padding: 4px 8px;
  font-size: 12px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}

.branch-input {
  width: 120px;
  background: #1e293b;
  color: #f8fafc;
  border: 1px solid #475569;
  border-radius: 6px;
  padding: 4px 8px;
  font-size: 12px;
}

.steering-row {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

.steering-pill {
  background: #431407;
  color: #fdba74;
  border: 1px solid #f97316;
  border-radius: 999px;
  padding: 3px 10px;
  font-size: 11px;
  font-weight: 500;
}

.artifacts-row {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  font-size: 12px;
}

.artifact-chip {
  background: #052e16;
  color: #86efac;
  border: 1px solid #22c55e;
  border-radius: 999px;
  padding: 3px 10px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
}

.artifact-chip:hover,
.artifact-chip.active {
  background: #14532d;
  border-color: #4ade80;
  color: #dcfce7;
}

.artifact-inspector {
  background: #022c22;
  border: 1px solid #22c55e;
  border-radius: 10px;
  padding: 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.artifact-inspector-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.artifact-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  font-size: 12px;
  color: #bbf7d0;
}

.artifact-id {
  background: #0f172a;
  color: #86efac;
  border-radius: 4px;
  padding: 2px 6px;
  font-size: 11px;
}

.artifact-chars {
  color: #94a3b8;
  font-size: 11px;
}

.artifact-content {
  max-height: 240px;
  white-space: pre-wrap;
  word-break: break-word;
}

.question-card {
  background: #083344;
  border: 1px solid #06b6d4;
  border-radius: 10px;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.question-title {
  font-weight: 700;
  color: #67e8f9;
  font-size: 14px;
}

.question-options {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.option-pill {
  background: #0f172a;
  color: #e0f2fe;
  border: 1px solid #0891b2;
  border-radius: 8px;
  padding: 6px 12px;
  cursor: pointer;
  font-size: 12px;
  display: inline-flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 2px;
  text-align: left;
}

.option-pill:hover {
  border-color: #22d3ee;
  background: #164e63;
}

.option-pill.selected {
  background: #0e7490;
  border-color: #67e8f9;
  color: #ffffff;
}

.option-label {
  font-weight: 600;
}

.option-desc {
  font-size: 11px;
  color: #bae6fd;
}

.question-hint {
  font-size: 11px;
  color: #a5f3fc;
}

.question-input {
  background: #0f172a;
  color: #f8fafc;
  border: 1px solid #0891b2;
  border-radius: 6px;
  padding: 7px 10px;
  font-size: 12px;
}
</style>
