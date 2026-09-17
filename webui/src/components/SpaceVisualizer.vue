<template>
  <div class="space-vis">
    <canvas
      ref="canvasEl"
      :class="{ draggable: mode === 'static_hrir' }"
      @pointerdown="onPointerDown"
      @pointermove="onPointerMove"
      @pointerup="onPointerUp"
    />
    <div class="vis-caption mono">{{ caption }}</div>
  </div>
</template>

<script setup>
// 俯视空间可视化：与后端 hrir.resolve_dynamic_azimuth_for_plan 保持同一角度定义。
// 屏幕映射：0°=正前(上)，90°=右，180°=后，270°=左；r 由距离 0.1-1.0m 映射 0.25R-1R。
import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue'
import { store } from '../store.js'

const props = defineProps({
  mode: { type: String, default: 'mono' },
  azimuth: { type: Number, default: 270 },
  distance: { type: Number, default: 0.3 },
  dynamicPath: { type: String, default: 'clockwise' },
  cycle: { type: Number, default: 8 },
  startAzimuth: { type: Number, default: 270 },
})

const emit = defineEmits(['update:azimuth', 'update:distance'])

const canvasEl = ref(null)
let rafId = 0
let dragging = false
let resizeObs = null

const caption = computed(() => {
  if (props.mode === 'static_hrir') return `方位 ${Math.round(props.azimuth)}° · 距离 ${props.distance.toFixed(1)}m`
  if (props.mode === 'dynamic_hrir') {
    const pathLabel = { clockwise: '顺时针', counterclockwise: '逆时针', side_to_side: '左右摆动' }[props.dynamicPath] || ''
    return `${pathLabel} · 周期 ${props.cycle}s · 距离 ${props.distance.toFixed(1)}m`
  }
  if (props.mode === 'stereo') return '立体声 · 左右声道同相'
  if (props.mode === 'behind_head') return '虚拟脑后 · 660µs 延迟 + 4kHz 陷波'
  return '单声道 · 无空间处理'
})

function distToRadius(dist) {
  const clamped = Math.min(1.0, Math.max(0.1, dist))
  return 0.25 + 0.75 * ((clamped - 0.1) / 0.9)
}

function azimuthToPoint(azDeg, rNorm, cx, cy, R) {
  const theta = (azDeg * Math.PI) / 180
  return [cx + Math.sin(theta) * R * rNorm, cy - Math.cos(theta) * R * rNorm]
}

function dynamicAzimuth(t) {
  const cycle = Math.max(props.cycle, 0.5)
  if (props.dynamicPath === 'clockwise') return mod360(props.startAzimuth - (t / cycle) * 360)
  if (props.dynamicPath === 'counterclockwise') return mod360(props.startAzimuth + (t / cycle) * 360)
  return mod360(180 + Math.cos((2 * Math.PI * t) / cycle) * 90)
}

function mod360(v) { return ((v % 360) + 360) % 360 }

// ---- 主题调色：从 CSS 变量读取，随 宣纸/墨夜 主题自动切换 ----
function parseColor(v) {
  v = (v || '').trim()
  if (v.startsWith('#')) {
    let h = v.slice(1)
    if (h.length === 3) h = h.split('').map((c) => c + c).join('')
    const n = parseInt(h, 16)
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255, 1]
  }
  const m = v.match(/[\d.]+/g)
  return m ? [Number(m[0]), Number(m[1]), Number(m[2]), m[3] !== undefined ? Number(m[3]) : 1] : [0, 0, 0, 1]
}
function cssVar(name) {
  return parseColor(getComputedStyle(document.documentElement).getPropertyValue(name))
}
function withA(c, a) {
  return `rgba(${c[0]},${c[1]},${c[2]},${a !== undefined ? a : c[3]})`
}
function palette() {
  return {
    rings: cssVar('--vis-rings'),
    label: cssVar('--vis-label'),
    headFill: cssVar('--vis-head-fill'),
    headStroke: cssVar('--vis-head-stroke'),
    source: cssVar('--vis-source'),
    sourceCore: cssVar('--vis-source-core'),
    link: cssVar('--vis-link'),
    orbit: cssVar('--vis-orbit'),
    warn: cssVar('--vis-warn'),
  }
}

function draw(now) {
  const canvas = canvasEl.value
  if (!canvas) return
  const dpr = window.devicePixelRatio || 1
  const cssW = canvas.clientWidth
  const cssH = canvas.clientHeight
  if (!cssW) return
  if (canvas.width !== cssW * dpr) canvas.width = cssW * dpr
  if (canvas.height !== cssH * dpr) canvas.height = cssH * dpr
  const ctx = canvas.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, cssW, cssH)
  const P = palette()

  const cx = cssW / 2
  const cy = cssH / 2
  const R = Math.min(cssW, cssH) / 2 - 26

  // 距离参考环
  ctx.strokeStyle = withA(P.rings)
  ctx.lineWidth = 1
  for (const rn of [0.25, 0.475, 0.7, 1.0]) {
    ctx.beginPath()
    ctx.arc(cx, cy, R * rn, 0, Math.PI * 2)
    ctx.stroke()
  }
  // 十字准线
  ctx.beginPath()
  ctx.moveTo(cx, cy - R - 8); ctx.lineTo(cx, cy + R + 8)
  ctx.moveTo(cx - R - 8, cy); ctx.lineTo(cx + R + 8, cy)
  ctx.stroke()

  // 方位标签：前/右/后/左
  ctx.font = `11px ${getComputedStyle(document.documentElement).getPropertyValue('--font-serif') || 'serif'}`
  ctx.fillStyle = withA(P.label)
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText('前', cx, cy - R - 15)
  ctx.fillText('后', cx, cy + R + 15)
  ctx.textAlign = 'left'
  ctx.fillText('右', cx + R + 10, cy)
  ctx.textAlign = 'right'
  ctx.fillText('左', cx - R - 10, cy)

  // 中央头部
  drawHead(ctx, cx, cy, P)

  let az = props.azimuth
  if (props.mode === 'dynamic_hrir') {
    const t = now / 1000
    az = dynamicAzimuth(t)
    // 轨迹（完整轨道虚线）
    ctx.save()
    ctx.setLineDash([4, 5])
    ctx.strokeStyle = withA(P.orbit)
    ctx.lineWidth = 1.2
    ctx.beginPath()
    const rr = R * distToRadius(props.distance)
    if (props.dynamicPath === 'side_to_side') {
      // 摆动弧线：90°-270°
      ctx.arc(cx, cy, rr, (90 * Math.PI) / 180 - Math.PI / 2, (270 * Math.PI) / 180 - Math.PI / 2)
    } else {
      ctx.arc(cx, cy, rr, 0, Math.PI * 2)
    }
    ctx.stroke()
    ctx.restore()
    // 拖尾
    for (let i = 1; i <= 14; i++) {
      const tailAz = dynamicAzimuth(t - i * 0.055)
      const [tx, ty] = azimuthToPoint(tailAz, distToRadius(props.distance), cx, cy, R)
      ctx.fillStyle = withA(P.source, 0.34 * (1 - i / 15))
      ctx.beginPath()
      ctx.arc(tx, ty, 4.5 * (1 - i / 16), 0, Math.PI * 2)
      ctx.fill()
    }
  }

  if (props.mode === 'static_hrir' || props.mode === 'dynamic_hrir') {
    const [sx, sy] = azimuthToPoint(az, distToRadius(props.distance), cx, cy, R)
    // 连线
    ctx.strokeStyle = withA(P.link)
    ctx.lineWidth = 1
    ctx.setLineDash([3, 4])
    ctx.beginPath()
    ctx.moveTo(cx, cy)
    ctx.lineTo(sx, sy)
    ctx.stroke()
    ctx.setLineDash([])
    // 声源点
    const grad = ctx.createRadialGradient(sx, sy, 0, sx, sy, 14)
    grad.addColorStop(0, withA(P.source, 0.9))
    grad.addColorStop(1, withA(P.source, 0))
    ctx.fillStyle = grad
    ctx.beginPath()
    ctx.arc(sx, sy, 14, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = withA(P.sourceCore)
    ctx.beginPath()
    ctx.arc(sx, sy, 5, 0, Math.PI * 2)
    ctx.fill()
    // 角度
    ctx.font = '11px ui-monospace, monospace'
    ctx.fillStyle = withA(P.source, 0.95)
    ctx.textAlign = 'center'
    ctx.fillText(`${Math.round(az)}°`, sx, sy - 16)
  } else if (props.mode === 'stereo') {
    drawSpeaker(ctx, cx - R * 0.72, cy, false, P)
    drawSpeaker(ctx, cx + R * 0.72, cy, true, P)
  } else if (props.mode === 'behind_head') {
    drawSourceBehind(ctx, cx, cy, R, P)
  }
}

function drawHead(ctx, cx, cy, P) {
  // 头部：圆形 + 鼻子（朝上 = 前方）
  ctx.save()
  ctx.fillStyle = withA(P.headFill)
  ctx.strokeStyle = withA(P.headStroke)
  ctx.lineWidth = 1.6
  ctx.beginPath()
  ctx.arc(cx, cy, 19, 0, Math.PI * 2)
  ctx.fill()
  ctx.stroke()
  // 鼻子
  ctx.beginPath()
  ctx.moveTo(cx - 6, cy - 17)
  ctx.quadraticCurveTo(cx, cy - 27, cx + 6, cy - 17)
  ctx.fillStyle = withA(P.headStroke)
  ctx.fill()
  // 耳朵
  ctx.fillStyle = withA(P.source, 0.9)
  ctx.beginPath(); ctx.arc(cx - 19, cy, 3.2, 0, Math.PI * 2); ctx.fill()
  ctx.beginPath(); ctx.arc(cx + 19, cy, 3.2, 0, Math.PI * 2); ctx.fill()
  ctx.restore()
}

function drawSpeaker(ctx, x, y, mirrored, P) {
  ctx.save()
  ctx.strokeStyle = withA(P.link, 0.85)
  ctx.lineWidth = 1.6
  const dir = mirrored ? -1 : 1
  ctx.strokeRect(x - 7, y - 9, 14, 18)
  ctx.beginPath()
  ctx.arc(x + dir * 4, y, 3.2, 0, Math.PI * 2)
  ctx.stroke()
  // 声波
  for (const r of [10, 15]) {
    ctx.strokeStyle = withA(P.link, 0.5 - r / 60)
    ctx.beginPath()
    ctx.arc(x + dir * 7, y, r, -Math.PI / 3.2, Math.PI / 3.2)
    ctx.stroke()
  }
  ctx.restore()
}

function drawSourceBehind(ctx, cx, cy, R, P) {
  const y = cy + R * 0.62
  ctx.save()
  ctx.fillStyle = withA(P.warn, 0.95)
  ctx.beginPath()
  ctx.arc(cx, y, 5, 0, Math.PI * 2)
  ctx.fill()
  for (const r of [11, 17, 23]) {
    ctx.strokeStyle = withA(P.warn, 0.55 - r / 55)
    ctx.lineWidth = 1.4
    ctx.beginPath()
    ctx.arc(cx, y, r, Math.PI + 0.5, -0.5)
    ctx.stroke()
  }
  ctx.restore()
}

function loop(now) {
  draw(now)
  rafId = requestAnimationFrame(loop)
}

function startLoop() {
  if (!rafId) rafId = requestAnimationFrame(loop)
}
function stopLoop() {
  cancelAnimationFrame(rafId)
  rafId = 0
}

function pointerToAzDist(e) {
  const canvas = canvasEl.value
  const rect = canvas.getBoundingClientRect()
  const cx = rect.width / 2
  const cy = rect.height / 2
  const R = Math.min(rect.width, rect.height) / 2 - 26
  const dx = e.clientX - rect.left - cx
  const dy = e.clientY - rect.top - cy
  const az = mod360((Math.atan2(dx, -dy) * 180) / Math.PI)
  const rNorm = Math.sqrt(dx * dx + dy * dy) / R
  const dist = Math.min(1.0, Math.max(0.1, 0.1 + (rNorm - 0.25) / 0.75 * 0.9))
  return { az, dist }
}

function onPointerDown(e) {
  if (props.mode !== 'static_hrir') return
  dragging = true
  canvasEl.value.setPointerCapture(e.pointerId)
  applyPointer(e)
}
function onPointerMove(e) {
  if (!dragging) return
  applyPointer(e)
}
function onPointerUp() {
  dragging = false
}
function applyPointer(e) {
  const { az, dist } = pointerToAzDist(e)
  emit('update:azimuth', Math.round(az / 5) * 5 % 360)
  emit('update:distance', Math.round(dist * 10) / 10)
}

onMounted(() => {
  resizeObs = new ResizeObserver(() => draw(performance.now()))
  resizeObs.observe(canvasEl.value)
  draw(performance.now())
})

onBeforeUnmount(() => {
  stopLoop()
  resizeObs?.disconnect()
})

// 只有动态环绕需要持续动画，其余模式按需重绘
watch(
  () => props.mode === 'dynamic_hrir',
  (needsLoop) => {
    if (needsLoop) startLoop()
    else stopLoop()
  },
  { immediate: true },
)

watch(() => [props.azimuth, props.distance, props.dynamicPath, props.cycle, props.startAzimuth], () => draw(performance.now()))
watch(() => store.gen.phase, () => draw(performance.now()))
// 主题切换（宣纸/墨夜）时重绘
watch(() => store.theme, () => draw(performance.now()))
</script>

<style scoped>
.space-vis { display: flex; flex-direction: column; align-items: center; gap: 6px; }
.space-vis canvas {
  width: 100%;
  max-width: 340px;
  aspect-ratio: 1 / 0.92;
  display: block;
}
.space-vis canvas.draggable { cursor: crosshair; }
.vis-caption { font-size: 11.5px; color: var(--text-faint); }
</style>
