// 后端 API 封装：REST + SSE 流式读取
import { store, showToast } from './store.js'

async function request(url, options = {}) {
  const resp = await fetch(url, options)
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`
    try {
      const body = await resp.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body)
    } catch { /* 保留默认信息 */ }
    throw new Error(detail)
  }
  return resp.json()
}

export function fetchMeta() {
  return request('/api/meta')
}

export function fetchRouteHint(text, emotionInstruct, engine) {
  return request('/api/route-hint', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, emotion_instruct: emotionInstruct, tts_engine: engine }),
  }).then((r) => r.text)
}

export function fetchHistory() {
  return request('/api/history')
}

export function fetchFavorites() {
  return request('/api/favorites')
}

export function deleteHistory(path) {
  return request(`/api/history?path=${encodeURIComponent(path)}`, { method: 'DELETE' })
}

export function saveFavorite(payload) {
  return request('/api/favorites', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function deleteFavorite(name) {
  return request(`/api/favorites?name=${encodeURIComponent(name)}`, { method: 'DELETE' })
}

export function savePreset(payload) {
  return request('/api/presets', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function cancelGenerate() {
  return request('/api/cancel', { method: 'POST' })
}

export function openOutputFolder() {
  return request('/api/open-folder', { method: 'POST' })
}

export function previewClip(payload) {
  return request('/api/preview-clip', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function clipUrl(clipId) {
  return request(`/api/clip-url?clip_id=${encodeURIComponent(clipId)}`)
}

export async function uploadReferenceAudio(file) {
  const form = new FormData()
  form.append('file', file)
  const resp = await fetch('/api/reference/upload', { method: 'POST', body: form })
  if (!resp.ok) {
    let detail = `${resp.status}`
    try { detail = (await resp.json()).detail || detail } catch { /* noop */ }
    throw new Error(detail)
  }
  return resp.json()
}

export function transcribeReference(path) {
  return request('/api/reference/transcribe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  }).then((r) => r.text)
}

export function transcribeAny(path) {
  return transcribeReference(path)
}

/**
 * 读取 SSE 流。handlers: { progress, result, error, cancelled, ping }
 * 返回 Promise，在流结束时 resolve；result/error/cancelled 事件通过 handlers 回调。
 */
export async function sseRequest(url, body, handlers = {}) {
  let resp
  try {
    resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  } catch (err) {
    throw new Error('无法连接到后端服务')
  }
  if (!resp.ok || !resp.body) {
    let detail = `${resp.status} ${resp.statusText}`
    try {
      const j = await resp.json()
      detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail ?? j)
    } catch { /* noop */ }
    throw new Error(detail)
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let streamError = null

  const dispatch = (raw) => {
    let event = 'message'
    let data = ''
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim()
      else if (line.startsWith('data:')) data += line.slice(5).trim()
    }
    if (!data) return
    let payload
    try { payload = JSON.parse(data) } catch { return }
    if (event === 'progress') handlers.progress?.(payload)
    else if (event === 'ping') handlers.ping?.(payload)
    else if (event === 'result') handlers.result?.(payload)
    else if (event === 'cancelled') handlers.cancelled?.(payload)
    else if (event === 'error') {
      streamError = new Error(payload.message || '生成失败')
      handlers.error?.(payload)
    }
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let idx
    while ((idx = buffer.indexOf('\n\n')) >= 0) {
      dispatch(buffer.slice(0, idx))
      buffer = buffer.slice(idx + 2)
    }
  }
  if (buffer.trim()) dispatch(buffer)
  if (streamError) throw streamError
}

export { showToast, store }
