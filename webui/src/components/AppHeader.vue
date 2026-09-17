<template>
  <header class="app-header">
    <div class="brand">
      <div class="seal">音</div>
      <div>
        <div class="title">TTS 3D Studio</div>
        <div class="subtitle">空间语音工作台 · 三维声境</div>
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
    <button class="btn sm theme-toggle" @click="toggleTheme" :title="store.theme === 'paper' ? '切换到墨夜主题' : '切换到宣纸主题'">
      {{ store.theme === 'paper' ? '☾ 墨夜' : '☀ 宣纸' }}
    </button>
    <button class="btn sm" @click="onOpenFolder" title="打开音频输出目录">
      📂 输出目录
    </button>
  </header>
</template>

<script setup>
import { computed } from 'vue'
import { store, toggleTheme } from '../store.js'
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
  padding: 13px 26px;
  border-bottom: 1px solid var(--border);
  background: var(--panel);
  backdrop-filter: blur(14px);
  position: sticky;
  top: 0;
  z-index: 20;
}
.brand { display: flex; align-items: center; gap: 12px; }
.seal {
  width: 40px;
  height: 40px;
  background: var(--accent);
  color: #fff8ee;
  font-family: var(--font-serif);
  font-size: 22px;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
  box-shadow: inset 0 0 0 1.5px rgba(255, 248, 238, 0.55), inset 0 0 0 3.5px var(--accent), 0 3px 10px rgba(176, 58, 46, 0.35);
  user-select: none;
}
.title {
  font-family: var(--font-serif);
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 1px;
  color: var(--text);
}
.subtitle {
  font-family: var(--font-serif);
  font-size: 11px;
  color: var(--text-faint);
  letter-spacing: 3px;
  margin-top: 1px;
}
.engine-box { display: flex; gap: 12px; align-items: flex-end; }
.sel-group label { display: block; font-size: 10.5px; color: var(--text-faint); margin-bottom: 3px; letter-spacing: 0.5px; }
.sel-group select { width: auto; min-width: 150px; padding: 6px 30px 6px 10px; font-size: 12.5px; }
.spacer { flex: 1; }
.theme-toggle { min-width: 72px; }
.warn-chip {
  font-size: 12px; color: var(--warn);
  border: 1px solid rgba(169, 123, 36, 0.4);
  background: rgba(169, 123, 36, 0.08);
  padding: 5px 11px; border-radius: 4px;
}
@media (max-width: 900px) {
  .app-header { flex-wrap: wrap; gap: 12px; }
}
</style>
