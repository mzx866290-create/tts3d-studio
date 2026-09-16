<template>
  <header class="app-header">
    <div class="brand">
      <div class="logo">
        <svg viewBox="0 0 32 32" width="26" height="26">
          <defs>
            <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stop-color="#8b5cf6" />
              <stop offset="1" stop-color="#06b6d4" />
            </linearGradient>
          </defs>
          <circle cx="16" cy="16" r="14" fill="url(#lg)" />
          <circle cx="16" cy="16" r="8.5" fill="none" stroke="rgba(255,255,255,0.55)" stroke-width="1.6" />
          <circle cx="16" cy="16" r="3" fill="#fff" />
        </svg>
      </div>
      <div>
        <div class="title">TTS 3D Studio</div>
        <div class="subtitle">3D 空间语音工作台</div>
      </div>
    </div>

    <div class="engine-box" v-if="meta">
      <div class="sel-group">
        <label>TTS 引擎</label>
        <select :value="store.form.engine" @change="onEngineChange($event.target.value)">
          <option v-for="e in meta.engines" :key="e.key" :value="e.key">{{ e.label }}</option>
        </select>
      </div>
      <div class="sel-group" v-if="currentEngine && currentEngine.models.length > 1">
        <label>模型</label>
        <select v-model="store.form.model">
          <option v-for="m in currentEngine.models" :key="m.key" :value="m.key">{{ m.label }}</option>
        </select>
      </div>
    </div>

    <div class="spacer"></div>

    <div class="warnings" v-if="meta && !meta.has_hrir">
      <span class="warn-chip" title="把 hrir_spatial_map.npz 放到项目根目录后，静态/动态 HRIR 模式可用">⚠ 未检测到 HRIR 数据</span>
    </div>
    <button class="btn sm" @click="onOpenFolder" title="打开音频输出目录">
      📂 输出目录
    </button>
  </header>
</template>

<script setup>
import { computed } from 'vue'
import { store } from '../store.js'
import { openOutputFolder, showToast } from '../api.js'

const meta = computed(() => store.meta)
const currentEngine = computed(() => store.meta?.engines.find((e) => e.key === store.form.engine))

function onEngineChange(key) {
  store.form.engine = key
  const engine = store.meta?.engines.find((e) => e.key === key)
  if (engine) store.form.model = engine.default_model
}

async function onOpenFolder() {
  try {
    await openOutputFolder()
    showToast('已在访达中打开输出目录', 'ok')
  } catch (err) {
    showToast(err.message, 'error')
  }
}
</script>

<style scoped>
.app-header {
  display: flex;
  align-items: center;
  gap: 26px;
  padding: 14px 26px;
  border-bottom: 1px solid var(--border);
  background: rgba(10, 13, 20, 0.6);
  backdrop-filter: blur(16px);
  position: sticky;
  top: 0;
  z-index: 20;
}
.brand { display: flex; align-items: center; gap: 11px; }
.title { font-size: 16.5px; font-weight: 700; letter-spacing: 0.3px; background: var(--accent-grad); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.subtitle { font-size: 11px; color: var(--text-faint); margin-top: -2px; }
.engine-box { display: flex; gap: 12px; align-items: flex-end; }
.sel-group label { display: block; font-size: 10.5px; color: var(--text-faint); margin-bottom: 3px; letter-spacing: 0.5px; }
.sel-group select { width: auto; min-width: 150px; padding: 6px 30px 6px 10px; font-size: 12.5px; }
.spacer { flex: 1; }
.warn-chip {
  font-size: 12px; color: var(--warn);
  border: 1px solid rgba(251, 191, 36, 0.3);
  background: rgba(251, 191, 36, 0.08);
  padding: 5px 11px; border-radius: 999px;
}
@media (max-width: 900px) {
  .app-header { flex-wrap: wrap; gap: 12px; }
}
</style>
