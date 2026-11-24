import { ref } from 'vue'

export interface StreamEventItem {
  id: string
  seq?: number
  eventType: string
  payload: Record<string, any>
  timestamp: string
}

export interface PendingApprovalCard {
  approval_id: string
  session_id?: string
  turn_id?: string
  tool_name: string
  arguments: Record<string, any>
  reason: string
  requester?: string
  requester_id?: string
}

export interface AuditVerificationResult {
  valid: boolean
  detail: string
  event_count: number
}

/**
 * Enterprise composable driving Streamable HTTP / SSE consumption with:
 * - Dynamic JWT Bearer / X-API-Key authentication header injection
 * - Live incremental thinking & token streaming with SSE sequence tracking (`lastSeq`)
 * - Disconnected stream resumption (`reconnectStream`), session forking (`forkSession`)
 * - Tenant-wide supervisor approval inbox (`fetchApprovalInbox` / `resolveInboxApproval`)
 * - Tamper-evident SHA-256 audit hash-chain verification (`verifyAuditChain`)
 * - Policy/security alert and subagent delegation lifecycle tracking
 */
export function useHarnessStream(initialSessionId: string) {
  const sessionId = ref(initialSessionId)
  const authToken = ref('')
  const lastSeq = ref(0)
  const isRunning = ref(false)
  const answerText = ref('')
  const reasoningSteps = ref<string[]>([])
  const activeSkills = ref<string[]>([])
  const events = ref<StreamEventItem[]>([])
  const pendingApproval = ref<PendingApprovalCard | null>(null)
  const approvalInbox = ref<PendingApprovalCard[]>([])
  const lastTurnUsage = ref<Record<string, any> | null>(null)
  const auditVerification = ref<AuditVerificationResult | null>(null)
  const policyAlerts = ref<string[]>([])
  const subagentEvents = ref<Record<string, any>[]>([])

  let abortController: AbortController | null = null
  const seenEventIds = new Set<string>()

  function buildHeaders(extra: Record<string, string> = {}): Record<string, string> {
    const headers: Record<string, string> = { ...extra }
    const token = authToken.value.trim()
    if (token) {
      if (token.startsWith('eyJ')) {
        headers['Authorization'] = `Bearer ${token}`
      } else {
        headers['X-API-Key'] = token
      }
    }
    return headers
  }

  function upsertSubagentEvent(eventName: string, payload: Record<string, any>) {
    const subId = payload.subagent_session_id
    const now = new Date().toLocaleTimeString()
    if (eventName === 'subagent_start') {
      subagentEvents.value.push({
        type: 'start',
        status: 'running',
        ...payload,
        timestamp: now,
      })
      return
    }
    if (eventName === 'subagent_complete') {
      if (subId) {
        const idx = subagentEvents.value.findIndex((item) => item.subagent_session_id === subId)
        if (idx >= 0) {
          subagentEvents.value[idx] = {
            ...subagentEvents.value[idx],
            type: 'complete',
            ...payload,
            completed_at: now,
          }
          return
        }
      }
      subagentEvents.value.push({
        type: 'complete',
        ...payload,
        timestamp: now,
      })
    }
  }

  function handleStreamEvent(
    eventName: string,
    parsed: Record<string, any>,
    seqNum: number | undefined,
    stepIndexMap: Map<number, number>
  ) {
    const payload = parsed.payload || {}
    const eventId = parsed.event_id || (seqNum !== undefined ? `seq_${seqNum}_${eventName}` : `${Date.now()}_${Math.random()}`)

    if (typeof seqNum === 'number' && !Number.isNaN(seqNum) && seqNum > lastSeq.value) {
      lastSeq.value = seqNum
    } else if (typeof parsed.seq === 'number' && parsed.seq > lastSeq.value) {
      lastSeq.value = parsed.seq
    }

    if (parsed.event_id && seenEventIds.has(parsed.event_id)) {
      return
    }
    if (parsed.event_id) {
      seenEventIds.add(parsed.event_id)
    }

    events.value.push({
      id: eventId,
      seq: seqNum ?? parsed.seq,
      eventType: eventName,
      payload,
      timestamp: new Date().toLocaleTimeString(),
    })

    if (eventName === 'skills_activated') {
      activeSkills.value = payload.skills || []
    } else if (eventName === 'thinking' && payload.reasoning) {
      const stepNum = Number(payload.step || 1)
      if (payload.delta && stepIndexMap.has(stepNum)) {
        const arrIdx = stepIndexMap.get(stepNum)!
        reasoningSteps.value[arrIdx] += payload.reasoning
      } else {
        stepIndexMap.set(stepNum, reasoningSteps.value.length)
        reasoningSteps.value.push(payload.reasoning)
      }
    } else if (eventName === 'approval_required') {
      const reqActor = payload.requester_id || payload.requester
      pendingApproval.value = {
        approval_id: payload.approval_id,
        session_id: payload.session_id || sessionId.value,
        turn_id: payload.turn_id,
        tool_name: payload.tool_name,
        arguments: payload.arguments || {},
        reason: payload.reason || '',
        requester: reqActor,
        requester_id: payload.requester_id,
      }
    } else if (eventName === 'approval_resolved') {
      if (!pendingApproval.value || pendingApproval.value.approval_id === payload.approval_id) {
        pendingApproval.value = null
      }
    } else if (eventName === 'token_delta') {
      answerText.value += payload.content || ''
    } else if (eventName === 'turn_complete') {
      lastTurnUsage.value = payload.usage || null
    } else if (eventName === 'policy_evaluated') {
      const decision = String(payload.decision || payload.effect || (payload.allowed === false ? 'deny' : 'allow'))
      const rule = payload.rule_name || payload.rule || payload.policy || ''
      const reason = payload.reason || payload.detail || ''
      if (decision !== 'allow' || payload.allowed === false) {
        const parts = [`[Policy:${decision}]`, rule, reason].filter(Boolean)
        policyAlerts.value.push(parts.join(' '))
      }
    } else if (eventName === 'security_alert') {
      const alertType = payload.alert_type || payload.category || payload.severity || 'security'
      const msg = payload.message || payload.reason || payload.detail || JSON.stringify(payload)
      policyAlerts.value.push(`[Security Alert:${alertType}] ${msg}`)
    } else if (eventName === 'subagent_start' || eventName === 'subagent_complete') {
      upsertSubagentEvent(eventName, payload)
    }
  }

  async function consumeSseResponse(resp: Response, stepIndexMap: Map<number, number>) {
    if (!resp.ok || !resp.body) {
      let detail = `HTTP ${resp.status}`
      try {
        const errBody = await resp.json()
        if (errBody?.detail) detail += `: ${errBody.detail}`
      } catch {
        // Ignore non-JSON error body
      }
      throw new Error(detail)
    }

    const reader = resp.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''

    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const frames = buffer.split(/\r?\n\r?\n/)
      buffer = frames.pop() || ''

      for (const frame of frames) {
        let eventName = 'message'
        let dataLine = ''
        let idLine = ''
        for (const rawLine of frame.split(/\r?\n/)) {
          if (rawLine.startsWith('id:')) idLine = rawLine.slice(3).trim()
          else if (rawLine.startsWith('event:')) eventName = rawLine.slice(6).trim()
          else if (rawLine.startsWith('data:')) {
            const piece = rawLine.slice(5).trimStart()
            dataLine = dataLine ? `${dataLine}\n${piece}` : piece
          }
        }
        if (!dataLine) continue
        try {
          const parsed = JSON.parse(dataLine)
          const parsedSeq = idLine ? Number(idLine) : undefined
          handleStreamEvent(eventName, parsed, parsedSeq, stepIndexMap)
        } catch {
          // Ignore malformed keepalive/comment frames
        }
      }
    }
  }

  async function sendMessage(message: string) {
    isRunning.value = true
    answerText.value = ''
    reasoningSteps.value = []
    activeSkills.value = []
    pendingApproval.value = null
    policyAlerts.value = []
    subagentEvents.value = []
    abortController = new AbortController()
    const stepIndexMap = new Map<number, number>()

    try {
      const resp = await fetch('/api/v1/chat/stream', {
        method: 'POST',
        headers: buildHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ session_id: sessionId.value, message }),
        signal: abortController.signal,
      })
      await consumeSseResponse(resp, stepIndexMap)
    } catch (err: any) {
      if (err?.name !== 'AbortError') {
        answerText.value += `\n[Stream Error] ${err?.message || err}`
      }
    } finally {
      isRunning.value = false
      abortController = null
    }
  }

  async function reconnectStream() {
    if (isRunning.value) return
    isRunning.value = true
    abortController = new AbortController()
    const stepIndexMap = new Map<number, number>()

    try {
      const url = `/api/v1/sessions/${encodeURIComponent(sessionId.value)}/stream?after_seq=${lastSeq.value}`
      const resp = await fetch(url, {
        method: 'GET',
        headers: buildHeaders({ Accept: 'text/event-stream' }),
        signal: abortController.signal,
      })
      await consumeSseResponse(resp, stepIndexMap)
    } catch (err: any) {
      if (err?.name !== 'AbortError') {
        policyAlerts.value.push(`[Reconnect Error] ${err?.message || err}`)
      }
    } finally {
      isRunning.value = false
      abortController = null
    }
  }

  async function cancelTurn() {
    try {
      await fetch(`/api/v1/sessions/${encodeURIComponent(sessionId.value)}/cancel`, {
        method: 'POST',
        headers: buildHeaders(),
      })
    } finally {
      abortController?.abort()
      isRunning.value = false
    }
  }

  async function forkSession(atSeq?: number) {
    const body = atSeq !== undefined ? { at_seq: atSeq } : {}
    const resp = await fetch(`/api/v1/sessions/${encodeURIComponent(sessionId.value)}/fork`, {
      method: 'POST',
      headers: buildHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(body),
    })
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`
      try {
        const err = await resp.json()
        if (err?.detail) detail += `: ${err.detail}`
      } catch {
        // ignore
      }
      policyAlerts.value.push(`[Fork Error] ${detail}`)
      return null
    }
    const data = await resp.json()
    const newSid = data.new_session_id || data.session_id
    if (newSid) {
      sessionId.value = newSid
      lastSeq.value = Number(data.last_seq ?? 0)
      seenEventIds.clear()
      events.value = []
      reasoningSteps.value = []
      activeSkills.value = []
      pendingApproval.value = null
      auditVerification.value = null
      policyAlerts.value = []
      subagentEvents.value = []
      answerText.value = `[Forked] 已切换至新分支会话 ${newSid}（原会话分叉点 seq=${atSeq ?? 'latest'}）。`
    }
    return data
  }

  async function resolveApproval(approved: boolean, note = '') {
    if (!pendingApproval.value) return
    const id = pendingApproval.value.approval_id
    const resp = await fetch(`/api/v1/approvals/${encodeURIComponent(id)}/resolve`, {
      method: 'POST',
      headers: buildHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ approved, reviewer: 'web_operator', note }),
    })
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`
      try {
        const err = await resp.json()
        if (err?.detail) detail += `: ${err.detail}`
      } catch {
        // ignore
      }
      policyAlerts.value.push(`[Approval Blocked] ${detail}`)
      return
    }
    pendingApproval.value = null
  }

  async function fetchApprovalInbox() {
    try {
      const resp = await fetch('/api/v1/approvals', {
        method: 'GET',
        headers: buildHeaders(),
      })
      if (!resp.ok) {
        let detail = `HTTP ${resp.status}`
        try {
          const err = await resp.json()
          if (err?.detail) detail += `: ${err.detail}`
        } catch {
          // ignore
        }
        policyAlerts.value.push(`[Inbox Error] ${detail}`)
        return []
      }
      const data = await resp.json()
      const rawItems: PendingApprovalCard[] = data.pending || data.approvals || []
      approvalInbox.value = rawItems.map((item) => ({
        ...item,
        requester: item.requester_id || item.requester,
        requester_id: item.requester_id || item.requester,
      }))
      return approvalInbox.value
    } catch (err: any) {
      policyAlerts.value.push(`[Inbox Error] ${err?.message || err}`)
      return []
    }
  }

  async function resolveInboxApproval(approvalId: string, approved: boolean, note = '') {
    const resp = await fetch(`/api/v1/approvals/${encodeURIComponent(approvalId)}/resolve`, {
      method: 'POST',
      headers: buildHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ approved, reviewer: 'web_supervisor', note }),
    })
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`
      try {
        const err = await resp.json()
        if (err?.detail) detail += `: ${err.detail}`
      } catch {
        // ignore
      }
      policyAlerts.value.push(`[Inbox Approval Error] ${detail}`)
      return false
    }
    approvalInbox.value = approvalInbox.value.filter((a) => a.approval_id !== approvalId)
    if (pendingApproval.value?.approval_id === approvalId) {
      pendingApproval.value = null
    }
    return true
  }

  async function verifyAuditChain() {
    try {
      const resp = await fetch(
        `/api/v1/sessions/${encodeURIComponent(sessionId.value)}/audit/verify`,
        {
          method: 'GET',
          headers: buildHeaders(),
        }
      )
      if (!resp.ok) {
        let detail = `HTTP ${resp.status}`
        try {
          const err = await resp.json()
          if (err?.detail) detail += `: ${err.detail}`
        } catch {
          // ignore
        }
        auditVerification.value = { valid: false, detail, event_count: 0 }
        return auditVerification.value
      }
      const data = await resp.json()
      auditVerification.value = {
        valid: Boolean(data.valid),
        detail: String(
          data.detail ||
            data.message ||
            (data.valid ? 'SHA-256 hash chain intact' : 'Hash chain mismatch')
        ),
        event_count: Number(data.event_count ?? data.checked_events ?? 0),
      }
      return auditVerification.value
    } catch (err: any) {
      auditVerification.value = {
        valid: false,
        detail: String(err?.message || err),
        event_count: 0,
      }
      return auditVerification.value
    }
  }

  return {
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
    buildHeaders,
    sendMessage,
    reconnectStream,
    cancelTurn,
    forkSession,
    resolveApproval,
    fetchApprovalInbox,
    resolveInboxApproval,
    verifyAuditChain,
  }
}
