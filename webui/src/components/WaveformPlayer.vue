<template>
  <div class="wave-player" :class="{ compact }">
    <button class="play-btn" :disabled="!ready" @click="toggle" :title="playing ? '暂停' : '播放'">
      <span v-if="!ready" class="spin">◌</span>
      <svg v-else-if="playing" viewBox="0 0 16 16" width="15" height="15"><rect x="3" y="2.5" width="3.4" height="11" rx="1.2" fill="currentColor"/><rect x="9.6" y="2.5" width="3.4" height="11" rx="1.2" fill="currentColor"/></svg>
      <svg v-else viewBox="0 0 16 16" width="15" height="15"><path d="M4.5 2.8 L13 8 L4.5 13.2 Z" fill="currentColor"/></svg>
    </button>
    <div class="wave-body">
      <canvas ref="canvasEl" @pointerdown="onSeekStart" />
      <div class="times mono">
        <span>{{ fmtTime(current) }}</span>
        <span>{{ fmtTime(duration) }}</span>
      </div>
    </div>
    <audio
      ref="audioEl"
      :src="src"
      preload="metadata"
      @play="onPlay"
      @pause="playing = false"
      @ended="playing = false"
      @timeupdate="onTimeUpdate"
    ></audio>
  </div>
</template>

<script setup>
import { ref, watch, onBeforeUnmount, nextTick } from 'vue'
import { store } from '../store.js'

const props = defineProps({
  src: { type: String, default: '' },
  compact: { type: Boolean, default: false },
  autoplay: { type: Boolean, default: false },
})

const canvasEl = ref(null)
const audioEl = ref(null)
const ready = ref(false)
const playing = ref(false)
const duration = ref(0)
const current = ref(0)

let peaks = []
let rafId = 0
let audioCtx = null

function fmtTime(sec) {
  if (!Number.isFinite(sec) || sec <= 0) return '0:00'
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}

function computePeaks(audioBuffer, buckets) {
  const chCount = audioBuffer.numberOfChannels
  const chData = []
  for (let c = 0; c < chCount; c++) chData.push(audioBuffer.getChannelData(c))
  const size = audioBuffer.length
  const step = Math.max(1, Math.floor(size / buckets))
  const result = new Float32Array(buckets)
  for (let i = 0; i < buckets; i++) {
    const start = i * step
    const end = Math.min(start + step, size)
    let max = 0
    for (let c = 0; c < chCount; c++) {
      const data = chData[c]
      for (let j = start; j < end; j += 3) {
        const v = Math.abs(data[j])
        if (v > max) max = v
      }
    }
    result[i] = max
  }
  return result
}

function draw() {
  const canvas = canvasEl.value
  if (!canvas) return
  const dpr = window.devicePixelRatio || 1
  const cssW = canvas.clientWidth
  const cssH = canvas.clientHeight
  if (cssW === 0) return
  if (canvas.width !== cssW * dpr || canvas.height !== cssH * dpr) {
    canvas.width = cssW * dpr
    canvas.height = cssH * dpr
  }
  const ctx = canvas.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, cssW, cssH)

  // 主题取色：宣纸/墨夜自动适配
  const style = getComputedStyle(document.documentElement)
  const playedColor = style.getPropertyValue('--wave-played').trim() || '#b03a2e'
  const dimColor = style.getPropertyValue('--wave-dim').trim() || 'rgba(55,49,42,0.28)'

  const progress = duration.value > 0 ? current.value / duration.value : 0
  const mid = cssH / 2
  const barW = 2
  const gap = 1.5
  const count = peaks.length

  for (let i = 0; i < count; i++) {
    const x = i * (barW + gap)
    if (x > cssW) break
    const h = Math.max(2, peaks[i] * (cssH * 0.86))
    ctx.fillStyle = i / count <= progress ? playedColor : dimColor
    const y = mid - h / 2
    ctx.fillRect(x, y, barW, h)
  }

  // 播放头
  if (progress > 0) {
    const px = progress * cssW
    ctx.fillStyle = 'rgba(55,49,42,0.55)'
    ctx.fillRect(px - 0.5, 0, 1, cssH)
  }
}

function loop() {
  if (audioEl.value) current.value = audioEl.value.currentTime
  draw()
  if (playing.value) {
    rafId = requestAnimationFrame(loop)
  } else {
    rafId = 0
  }
}

function ensureLoop() {
  if (!rafId && playing.value) rafId = requestAnimationFrame(loop)
}

async function load() {
  ready.value = false
  peaks = []
  duration.value = 0
  current.value = 0
  playing.value = false
  if (!props.src) return
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)()
    const resp = await fetch(props.src)
    if (!resp.ok) throw new Error(`加载失败 ${resp.status}`)
    const buf = await resp.arrayBuffer()
    const audioBuffer = await audioCtx.decodeAudioData(buf)
    await nextTick()
    const cssW = canvasEl.value?.clientWidth || 600
    peaks = computePeaks(audioBuffer, Math.floor(cssW / 3.5))
    duration.value = audioBuffer.duration
    ready.value = true
    draw()
    if (props.autoplay) audioEl.value?.play().catch(() => {})
  } catch (err) {
    console.warn('waveform load failed:', err)
  }
}

function toggle() {
  const audio = audioEl.value
  if (!audio || !ready.value) return
  if (audio.paused) audio.play().catch(() => {})
  else audio.pause()
}

function onPlay() {
  playing.value = true
  ensureLoop()
}

function onTimeUpdate() {
  if (!playing.value && audioEl.value) current.value = audioEl.value.currentTime
}

function onSeekStart(e) {
  const audio = audioEl.value
  const canvas = canvasEl.value
  if (!audio || !canvas || !ready.value) return
  const rect = canvas.getBoundingClientRect()
  const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width))
  audio.currentTime = ratio * (duration.value || 0)
  current.value = audio.currentTime
  draw()
}

watch(() => props.src, (v, old) => {
  if (v !== old) {
    audioEl.value?.pause()
    load()
  }
})

onBeforeUnmount(() => {
  cancelAnimationFrame(rafId)
  rafId = 0
})

// 主题切换时重绘波形
watch(() => store.theme, () => { if (peaks.length) draw() })

// 初始加载 + 首帧绘制
load()
ensureLoop()
</script>

<style scoped>
.wave-player {
  display: flex;
  align-items: center;
  gap: 14px;
  background: var(--input-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 16px;
}
.play-btn {
  width: 42px;
  height: 42px;
  border-radius: 50%;
  border: none;
  background: var(--accent-grad);
  color: #fff8ee;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  flex-shrink: 0;
  box-shadow: 0 3px 12px rgba(176, 58, 46, 0.35);
  transition: transform 0.14s, box-shadow 0.14s;
}
.play-btn:hover:not(:disabled) { transform: scale(1.06); box-shadow: 0 5px 18px rgba(176, 58, 46, 0.5); }
.play-btn:disabled { opacity: 0.45; cursor: default; }
.wave-body { flex: 1; min-width: 0; }
.wave-body canvas { width: 100%; height: 58px; display: block; cursor: pointer; }
.compact .wave-body canvas { height: 38px; }
.compact .play-btn { width: 34px; height: 34px; }
.times { display: flex; justify-content: space-between; font-size: 11px; color: var(--text-faint); margin-top: 4px; }
</style>
