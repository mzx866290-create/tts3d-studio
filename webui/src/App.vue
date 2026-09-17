<template>
  <div class="app-shell">
    <AppHeader />

    <main class="layout" v-if="store.meta">
      <section class="col-left">
        <ScriptPanel />
        <VoicePanel />
      </section>

      <section class="col-right">
        <SpacePanel />
        <GenerateBar @generate="startGenerate" @stop="stopGenerate" />
        <ResultPanel />
      </section>
    </main>

    <div v-else class="loading-screen">
      <span class="spin" style="font-size: 22px;">◌</span>
      <span>正在连接后端服务…</span>
    </div>
    <div v-if="store.meta && !store.connected" class="conn-warn">后端连接已断开，请检查服务是否在运行。</div>

    <nav class="tabs" v-if="store.meta">
      <button
        v-for="t in tabs"
        :key="t.key"
        class="tab"
        :class="{ active: activeTab === t.key }"
        @click="activeTab = t.key"
      >
        {{ t.label }}
        <span class="badge" v-if="t.badge">{{ t.badge }}</span>
      </button>
    </nav>
    <section class="tab-body panel" v-if="store.meta">
      <BatchPanel v-show="activeTab === 'batch'" />
      <FavoritesPanel v-show="activeTab === 'favorites'" />
      <HistoryPanel v-show="activeTab === 'history'" />
    </section>

    <footer class="footer faint mono">
      TTS 3D Studio · 声临其境 · Qwen3-TTS / CosyVoice2 本地渲染
    </footer>

    <div v-if="store.toast" :key="store.toast.id" class="toast" :class="store.toast.kind">
      {{ store.toast.text }}
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import { store, showToast } from './store.js'
import { fetchMeta, fetchRouteHint, fetchHistory, fetchFavorites } from './api.js'
import { startGenerate, stopGenerate } from './generate.js'
import AppHeader from './components/AppHeader.vue'
import ScriptPanel from './components/ScriptPanel.vue'
import VoicePanel from './components/VoicePanel.vue'
import SpacePanel from './components/SpacePanel.vue'
import GenerateBar from './components/GenerateBar.vue'
import ResultPanel from './components/ResultPanel.vue'
import BatchPanel from './components/BatchPanel.vue'
import FavoritesPanel from './components/FavoritesPanel.vue'
import HistoryPanel from './components/HistoryPanel.vue'

const activeTab = ref('batch')

const tabs = computed(() => [
  { key: 'batch', label: '📦 批量生成' },
  { key: 'favorites', label: '⭐ 声线收藏', badge: store.favorites.length || '' },
  { key: 'history', label: '📂 历史记录', badge: store.history.length || '' },
])

onMounted(async () => {
  try {
    const meta = await fetchMeta()
    store.meta = meta
    store.connected = true
  } catch (err) {
    store.connected = false
    showToast(`无法加载服务信息: ${err.message}`, 'error', 8000)
    return
  }
  // 默认引擎/模型/预设
  const firstEngine = store.meta.engines[0]
  store.form.engine = firstEngine.key
  store.form.model = firstEngine.default_model
  const defaultPreset = store.meta.presets.find((p) => p.key === store.meta.default_preset) || store.meta.presets[0]
  if (defaultPreset) {
    store.form.presetKey = defaultPreset.key
    store.form.prompt = defaultPreset.prompt || ''
    store.form.text = defaultPreset.text || ''
    store.form.calibrationText = defaultPreset.calibration_text || ''
  }
  // 无 HRIR 时选一个可用模式
  const currentMode = store.meta.modes.find((m) => m.key === store.form.mode)
  if (!currentMode?.available) {
    const fallback = store.meta.modes.find((m) => m.available)
    if (fallback) store.form.mode = fallback.key
  }
  try {
    store.history = await fetchHistory()
    store.favorites = await fetchFavorites()
  } catch { /* 忽略 */ }
})

// 实时路由提示（防抖）
let hintTimer = null
watch(
  () => [store.form.text, store.form.emotionInstruct, store.form.engine],
  () => {
    clearTimeout(hintTimer)
    hintTimer = setTimeout(async () => {
      try {
        store.routeHint = await fetchRouteHint(store.form.text, store.form.emotionInstruct, store.form.engine)
      } catch { /* 断连时静默 */ }
    }, 250)
  },
  { immediate: true },
)
</script>

<style scoped>
.app-shell {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  max-width: 1460px;
  margin: 0 auto;
}
.layout {
  display: grid;
  grid-template-columns: 1.08fr 1fr;
  gap: 16px;
  padding: 18px 26px 4px;
  align-items: start;
}
.col-left, .col-right {
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-width: 0;
}
.loading-screen {
  display: flex; align-items: center; justify-content: center; gap: 12px;
  padding: 120px 0;
  color: var(--text-dim);
}
.conn-warn {
  margin: 10px 26px;
  border: 1px solid rgba(248, 113, 113, 0.4);
  background: rgba(248, 113, 113, 0.08);
  color: #fca5a5;
  border-radius: var(--radius-sm);
  padding: 10px 14px;
  font-size: 13px;
}
.tabs {
  display: flex;
  gap: 6px;
  padding: 12px 26px 0;
  border-bottom: none;
}
.tab {
  display: inline-flex; align-items: center; gap: 7px;
  background: transparent;
  border: 1px solid transparent;
  border-bottom: none;
  color: var(--text-dim);
  padding: 9px 16px;
  border-radius: 10px 10px 0 0;
  cursor: pointer;
  font-family: inherit;
  font-size: 13px;
  transition: all 0.15s;
}
.tab:hover { color: var(--text); }
.tab.active {
  background: var(--panel);
  border-color: var(--border);
  color: var(--text);
}
.badge {
  background: rgba(176, 58, 46, 0.14);
  color: var(--accent);
  font-size: 10.5px;
  border-radius: 999px;
  padding: 1px 7px;
  font-family: var(--mono);
}
.tab-body { margin: 0 26px; }
.footer {
  text-align: center;
  font-size: 11px;
  padding: 18px 0 22px;
}
@media (max-width: 1024px) {
  .layout { grid-template-columns: 1fr; }
}
</style>
