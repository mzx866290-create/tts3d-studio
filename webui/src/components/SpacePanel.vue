<template>
  <div class="panel">
    <h2 class="panel-title">
      🎧 空间引擎
      <span class="hint">拖动声源可调整方位与距离</span>
    </h2>

    <div class="mode-grid">
      <button
        v-for="m in modes"
        :key="m.key"
        class="mode-btn"
        :class="{ active: store.form.mode === m.key, disabled: !m.available }"
        :disabled="!m.available"
        @click="store.form.mode = m.key"
        :title="m.available ? '' : '需要 hrir_spatial_map.npz'"
      >
        <span class="mode-icon">{{ m.icon }}</span>
        <span class="mode-label">{{ m.label }}</span>
      </button>
    </div>

    <div class="vis-wrap">
      <SpaceVisualizer
        :mode="store.form.mode"
        :azimuth="store.form.staticAzimuth"
        :distance="store.form.staticDistance"
        :dynamic-path="store.form.dynamicPath"
        :cycle="store.form.dynamicCycle"
        :start-azimuth="store.form.dynamicStart"
        @update:azimuth="store.form.staticAzimuth = $event"
        @update:distance="store.form.staticDistance = $event"
      />
    </div>

    <div v-if="store.form.mode === 'static_hrir'" class="params">
      <div class="field">
        <label>方位角</label>
        <div class="slider-row">
          <input type="range" min="0" max="360" step="5" v-model.number="store.form.staticAzimuth" />
          <span class="val">{{ store.form.staticAzimuth }}°</span>
        </div>
      </div>
      <div class="field">
        <label>距离</label>
        <div class="slider-row">
          <input type="range" min="0.1" max="1.0" step="0.1" v-model.number="store.form.staticDistance" />
          <span class="val">{{ store.form.staticDistance.toFixed(1) }}m</span>
        </div>
      </div>
    </div>

    <div v-if="store.form.mode === 'dynamic_hrir'" class="params">
      <div class="field">
        <label>移动轨迹</label>
        <div class="chips">
          <button
            v-for="p in store.meta?.paths || []"
            :key="p.key"
            class="chip"
            :class="{ active: store.form.dynamicPath === p.key }"
            @click="store.form.dynamicPath = p.key"
          >{{ p.label }}</button>
        </div>
      </div>
      <div class="field">
        <label>环绕周期</label>
        <div class="slider-row">
          <input type="range" min="2" max="20" step="0.5" v-model.number="store.form.dynamicCycle" />
          <span class="val">{{ store.form.dynamicCycle.toFixed(1) }}s</span>
        </div>
      </div>
      <div class="field">
        <label>起始角度 / 距离</label>
        <div class="slider-row">
          <input type="range" min="0" max="360" step="5" v-model.number="store.form.dynamicStart" />
          <span class="val">{{ store.form.dynamicStart }}°</span>
          <input type="range" min="0.1" max="1.0" step="0.1" v-model.number="store.form.dynamicDistance" style="margin-left: 6px;" />
          <span class="val">{{ store.form.dynamicDistance.toFixed(1) }}m</span>
        </div>
      </div>
    </div>

    <details class="advanced">
      <summary>语速 / 音调</summary>
      <div class="field" style="margin-top: 10px;">
        <label>语速（空间效果前处理）</label>
        <div class="slider-row">
          <input type="range" min="0.5" max="2" step="0.1" v-model.number="store.form.speed" />
          <span class="val">{{ store.form.speed.toFixed(1) }}x</span>
        </div>
      </div>
      <div class="field">
        <label>音调（半音）</label>
        <div class="slider-row">
          <input type="range" min="-12" max="12" step="1" v-model.number="store.form.pitch" />
          <span class="val">{{ store.form.pitch > 0 ? '+' : '' }}{{ store.form.pitch }}st</span>
        </div>
      </div>
    </details>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { store } from '../store.js'
import SpaceVisualizer from './SpaceVisualizer.vue'

const MODE_ICONS = {
  mono: '🎙️',
  stereo: '🔊',
  behind_head: '🧠',
  static_hrir: '📍',
  dynamic_hrir: '🌀',
}

const modes = computed(() => {
  const map = new Map((store.meta?.modes || []).map((m) => [m.key, m]))
  const order = ['mono', 'stereo', 'behind_head', 'static_hrir', 'dynamic_hrir']
  return order
    .filter((k) => map.has(k))
    .map((k) => ({ key: k, label: map.get(k).label, available: map.get(k).available, icon: MODE_ICONS[k] || '·' }))
})

// 确保默认模式可用
if (store.meta) {
  const current = modes.value.find((m) => m.key === store.form.mode)
  if (current && !current.available) {
    const fallback = modes.value.find((m) => m.available)
    if (fallback) store.form.mode = fallback.key
  }
}
</script>

<style scoped>
.mode-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 7px; }
.mode-btn {
  display: flex; flex-direction: column; align-items: center; gap: 4px;
  background: rgba(255, 255, 255, 0.035);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text-dim);
  padding: 9px 4px 7px;
  cursor: pointer;
  font-family: inherit;
  font-size: 11.5px;
  transition: all 0.15s;
}
.mode-btn .mode-icon { font-size: 16px; }
.mode-btn:hover:not(:disabled) { border-color: var(--border-strong); color: var(--text); }
.mode-btn.active {
  border-color: rgba(176, 58, 46, 0.65);
  background: rgba(176, 58, 46, 0.12);
  color: var(--accent-text);
  box-shadow: 0 0 14px rgba(176, 58, 46, 0.18) inset;
}
.mode-btn.disabled { opacity: 0.32; cursor: not-allowed; }
.vis-wrap { margin-top: 8px; display: flex; justify-content: center; }
.params { margin-top: 6px; }
.chips { display: flex; gap: 7px; flex-wrap: wrap; }
.chip.active { border-color: rgba(176, 58, 46, 0.7); background: rgba(176, 58, 46, 0.16); color: var(--accent-text); }
.advanced { margin-top: 10px; border-top: 1px solid var(--border); padding-top: 9px; }
.advanced summary { cursor: pointer; font-size: 12.5px; color: var(--text-dim); user-select: none; }
.advanced summary:hover { color: var(--text); }
@media (max-width: 640px) {
  .mode-grid { grid-template-columns: repeat(3, 1fr); }
}
</style>
