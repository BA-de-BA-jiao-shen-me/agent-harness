import { ref } from 'vue'

export interface StreamEventItem {
  id: string
  eventType: string
  payload: Record<string, any>
  timestamp: string
}

export interface PendingApprovalCard {
  approval_id: string
  tool_name: string
  arguments: Record<string, any>
  reason: string
}

/**
 * Composable driving Streamable HTTP / SSE consumption with live cancellation
 * and Human-in-the-Loop approval resolution.
 */
export function useHarnessStream(sessionId: string) {
  const isRunning = ref(false)
  const answerText = ref('')
  const reasoningSteps = ref<string[]>([])
  const activeSkills = ref<string[]>([])
  const events = ref<StreamEventItem[]>([])
  const pendingApproval = ref<PendingApprovalCard | null>(null)
  const lastTurnUsage = ref<Record<string, any> | null>(null)
  let abortController: AbortController | null = null

  async function sendMessage(message: string) {
    isRunning.value = true
    answerText.value = ''
    reasoningSteps.value = []
    activeSkills.value = []
    pendingApproval.value = null
    abortController = new AbortController()

    try {
      const resp = await fetch('/api/v1/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message }),
        signal: abortController.signal,
      })
      if (!resp.ok || !resp.body) {
        throw new Error(`HTTP ${resp.status}`)
      }

      const reader = resp.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const frames = buffer.split('\n\n')
        buffer = frames.pop() || ''

        for (const frame of frames) {
          let eventName = 'message'
          let dataLine = ''
          for (const line of frame.split('\n')) {
            if (line.startsWith('event: ')) eventName = line.slice(7).trim()
            if (line.startsWith('data: ')) dataLine += line.slice(6).trim()
          }
          if (!dataLine) continue
          const parsed = JSON.parse(dataLine)
          const payload = parsed.payload || {}

          events.value.push({
            id: parsed.event_id || String(Date.now()),
            eventType: eventName,
            payload,
            timestamp: new Date().toLocaleTimeString(),
          })

          if (eventName === 'skills_activated') {
            activeSkills.value = payload.skills || []
          } else if (eventName === 'thinking' && payload.reasoning) {
            reasoningSteps.value.push(payload.reasoning)
          } else if (eventName === 'approval_required') {
            pendingApproval.value = {
              approval_id: payload.approval_id,
              tool_name: payload.tool_name,
              arguments: payload.arguments || {},
              reason: payload.reason || '',
            }
          } else if (eventName === 'approval_resolved') {
            pendingApproval.value = null
          } else if (eventName === 'token_delta') {
            answerText.value += payload.content || ''
          } else if (eventName === 'turn_complete') {
            lastTurnUsage.value = payload.usage || null
          }
        }
      }
    } catch (err: any) {
      if (err?.name !== 'AbortError') {
        answerText.value += `\n[Stream Error] ${err?.message || err}`
      }
    } finally {
      isRunning.value = false
      abortController = null
    }
  }

  async function cancelTurn() {
    await fetch(`/api/v1/sessions/${encodeURIComponent(sessionId)}/cancel`, { method: 'POST' })
    abortController?.abort()
    isRunning.value = false
  }

  async function resolveApproval(approved: boolean, note = '') {
    if (!pendingApproval.value) return
    const id = pendingApproval.value.approval_id
    await fetch(`/api/v1/approvals/${encodeURIComponent(id)}/resolve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ approved, reviewer: 'web_operator', note }),
    })
    pendingApproval.value = null
  }

  return {
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
  }
}
