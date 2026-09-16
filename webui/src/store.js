// 全局响应式状态
import { reactive } from 'vue'

export const store = reactive({
  meta: null,
  connected: true,
  form: {
    text: '',
    presetKey: '',
    engine: 'qwen3',
    model: '',
    prompt: '',
    calibrationText: '',
    emotionInstruct: '',
    refAudioPath: null,
    refAudioUrl: '',
    refAudioName: '',
    refText: '',
    mode: 'behind_head',
    staticAzimuth: 270,
    staticDistance: 0.3,
    dynamicPath: 'clockwise',
    dynamicCycle: 8.0,
    dynamicStart: 270,
    dynamicDistance: 0.2,
    speed: 1.0,
    pitch: 0,
    seed: 42,
    randomSeed: false,
  },
  routeHint: '',
  // 生成状态机：idle → generating → done / cancelled / error
  gen: {
    phase: 'idle',
    stageText: '',
    progress: 0,
    elapsed: 0,
    startedAt: 0,
  },
  result: null, // { url, filePath, seed, status, clipId, clipUrl }
  clip: null, // { url, clipId, seed, status }
  clipBusy: false,
  batch: {
    running: false,
    stageText: '',
    progress: 0,
    items: [],
    summary: '',
  },
  favorites: [],
  history: [],
  toast: null,
})

let toastTimer = null
export function showToast(text, kind = 'info', duration = 3200) {
  store.toast = { text, kind, id: Date.now() }
  clearTimeout(toastTimer)
  toastTimer = setTimeout(() => { store.toast = null }, duration)
}

// 生成进度百分比映射：模型加载 5%，校准 15%，分块 15-85%，空间 92%，保存 97%
export function progressFromStage(stage, data) {
  switch (stage) {
    case 'model_loading': return 0.04
    case 'voice_lock': return 0.12
    case 'chunk_start': return 0.15 + 0.7 * ((data.index - 1) / Math.max(data.total, 1))
    case 'chunk_done': return 0.15 + 0.7 * (data.index / Math.max(data.total, 1))
    case 'spatial': return 0.92
    case 'saving': return 0.97
    default: return store.gen.progress
  }
}

export function stageTextFromStage(stage, data) {
  switch (stage) {
    case 'model_loading':
      return data.engine === 'cosyvoice2'
        ? '正在加载 CosyVoice2 模型（首次 1-2 分钟）…'
        : `正在加载模型 ${data.model || ''} …`
    case 'voice_lock': return '正在生成校准参考音（锁定音色）…'
    case 'chunk_start': return `正在合成第 ${data.index}/${data.total} 块…`
    case 'chunk_done': return `第 ${data.index}/${data.total} 块完成`
    case 'spatial': return '正在渲染空间音频效果…'
    case 'saving': return '正在保存音频…'
    default: return ''
  }
}
