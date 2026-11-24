<script setup lang="ts">
import { ref } from 'vue'
import { useHarnessStream } from './composables/useHarnessStream'

const sessionId = `web_${Math.random().toString(36).slice(2, 8)}`
const input = ref('')
const {
  isRunning,
  answerText,
  reasoningSteps,
  activeSkills,
  events,
  pendingApproval,
  lastTurnUsage,
  sendMessage,
  cancelTurn,
  resolveApproval,
} = useHarnessStream(sessionId)

const presets = [
  'SF6 高压断路器紧急闭锁阈值是多少 MPa？',
  '用 SQL 统计各区域故障数和总停机时长',
  '立即向值班群发送 INC-2026-01 高危紧急告警（测试人工审批）',
]

function submit(text?: string) {
  const q = (text ?? input.value).trim()
  if (!q || isRunning.value) return
  input.value = ''
  sendMessage(q)
}
</script>

<template>
  <div style="display:grid;grid-template-columns:2fr 1fr;gap:16px;min-height:100vh;padding:20px;box-sizing:border-box;">
    <!-- 左侧：流式对话与审批工作台 -->
    <section style="background:#1e293b;border-radius:12px;padding:20px;display:flex;flex-direction:column;gap:14px;">
      <header style="display:flex;justify-content:space-between;align-items:center;">
        <h2 style="margin:0;font-size:20px;">PyHarness — Enterprise Agent Workbench</h2>
        <span style="font-size:12px;color:#94a3b8;">Session: {{ sessionId }}</span>
      </header>

      <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <button
          v-for="p in presets"
          :key="p"
          @click="submit(p)"
          style="background:#334155;color:#f8fafc;border:none;border-radius:6px;padding:6px 10px;cursor:pointer;font-size:12px;"
        >
          {{ p }}
        </button>
      </div>

      <div v-if="activeSkills.length" style="font-size:12px;color:#38bdf8;">
        Activated Skills: {{ activeSkills.join(', ') }}
      </div>

      <details v-if="reasoningSteps.length" open style="background:#0f172a;padding:10px;border-radius:8px;">
        <summary style="cursor:pointer;color:#fbbf24;font-size:13px;">Thinking / Planning Steps ({{ reasoningSteps.length }})</summary>
        <ul style="margin:8px 0 0 18px;font-size:13px;color:#cbd5e1;">
          <li v-for="(s, idx) in reasoningSteps" :key="idx">{{ s }}</li>
        </ul>
      </details>

      <!-- Human-in-the-Loop 高危操作审批弹窗卡片 -->
      <div
        v-if="pendingApproval"
        style="background:#451a03;border:1px solid #f59e0b;border-radius:10px;padding:14px;"
      >
        <div style="font-weight:700;color:#fcd34d;">⚠️ 高危操作待人工审批：{{ pendingApproval.tool_name }}</div>
        <div style="font-size:12px;color:#fde68a;margin:6px 0;">{{ pendingApproval.reason }}</div>
        <pre style="background:#0f172a;padding:8px;border-radius:6px;font-size:12px;overflow:auto;">{{ JSON.stringify(pendingApproval.arguments, null, 2) }}</pre>
        <div style="display:flex;gap:10px;margin-top:10px;">
          <button @click="resolveApproval(true, 'Approved in UI')" style="background:#16a34a;color:#fff;border:none;padding:6px 14px;border-radius:6px;cursor:pointer;">批准执行</button>
          <button @click="resolveApproval(false, 'Rejected by operator')" style="background:#dc2626;color:#fff;border:none;padding:6px 14px;border-radius:6px;cursor:pointer;">拒绝拦截</button>
        </div>
      </div>

      <main style="flex:1;background:#0f172a;border-radius:8px;padding:14px;white-space:pre-wrap;line-height:1.6;">
        {{ answerText || '请输入问题，或点击上方快捷场景体验 Hybrid RAG / SQL 并发沙箱 / 高危人工审批。' }}
      </main>

      <footer style="display:flex;gap:8px;">
        <input
          v-model="input"
          @keyup.enter="submit()"
          placeholder="输入业务问题或分析指令..."
          style="flex:1;background:#0f172a;color:#fff;border:1px solid #475569;border-radius:8px;padding:10px;"
        />
        <button v-if="!isRunning" @click="submit()" style="background:#2563eb;color:#fff;border:none;border-radius:8px;padding:10px 18px;cursor:pointer;">发送</button>
        <button v-else @click="cancelTurn()" style="background:#ef4444;color:#fff;border:none;border-radius:8px;padding:10px 18px;cursor:pointer;">停止中断</button>
      </footer>
    </section>

    <!-- 右侧：实时 Event Journal 瀑布流与 Token 成本面板 -->
    <aside style="background:#1e293b;border-radius:12px;padding:16px;display:flex;flex-direction:column;gap:12px;max-height:calc(100vh - 40px);overflow:auto;">
      <h3 style="margin:0;font-size:15px;">Event Journal & Trace Waterfall</h3>
      <div v-if="lastTurnUsage" style="background:#0f172a;padding:10px;border-radius:8px;font-size:12px;color:#34d399;">
        Tokens: Prompt={{ lastTurnUsage.prompt_tokens }} / Completion={{ lastTurnUsage.completion_tokens }} | Est. Cost=${{ lastTurnUsage.estimated_cost_usd }}
      </div>
      <div
        v-for="ev in events"
        :key="ev.id"
        style="background:#0f172a;padding:8px 10px;border-radius:6px;font-size:12px;border-left:3px solid #38bdf8;"
      >
        <div style="display:flex;justify-content:space-between;color:#93c5fd;">
          <strong>{{ ev.eventType }}</strong>
          <span>{{ ev.timestamp }}</span>
        </div>
        <pre style="margin:4px 0 0;white-space:pre-wrap;color:#cbd5e1;font-size:11px;">{{ JSON.stringify(ev.payload).slice(0, 220) }}</pre>
      </div>
    </aside>
  </div>
</template>
