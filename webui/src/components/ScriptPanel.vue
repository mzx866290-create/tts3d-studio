<template>
  <div class="panel">
    <h2 class="panel-title">
      📝 文稿
      <span class="hint">{{ charCount }} 字 · 预计分 {{ chunkEstimate }} 块</span>
    </h2>

    <textarea
      v-model="store.form.text"
      rows="5"
      placeholder="输入要合成的文本…短文本直出最快；长文本自动分块并锁定音色。"
    ></textarea>

    <div class="route-hint" v-if="store.routeHint" :class="hintClass">
      <span class="icon">{{ hintIcon }}</span>
      <span class="text">{{ hintBody }}</span>
    </div>

    <div class="preset-row">
      <div class="field grow">
        <label>预设</label>
        <select :value="store.form.presetKey" @change="applyPreset($event.target.value)">
          <option v-for="p in store.meta?.presets || []" :key="p.key" :value="p.key">
            {{ p.key }}{{ p.prompt ? ' · ' + p.prompt.slice(0, 26) : '' }}
          </option>
        </select>
      </div>
      <button class="btn sm" @click="savingPreset = !savingPreset" :title="'把当前文稿+提示词存为新预设'">
        {{ savingPreset ? '✕ 取消' : '💾 另存为预设' }}
      </button>
    </div>

    <div class="save-preset-row" v-if="savingPreset">
      <input type="text" v-model="presetName" placeholder="预设名称，如：深夜电台" @keyup.enter="doSavePreset" />
      <button class="btn sm primary" @click="doSavePreset">保存</button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { store, showToast } from '../store.js'
import { savePreset, fetchMeta } from '../api.js'

const savingPreset = ref(false)
const presetName = ref('')

const charCount = computed(() => store.form.text.length)

// 与后端 MAX_TTS_CHUNK_CHARS 一致的粗估
const chunkEstimate = computed(() => {
  const limit = store.meta?.limits?.max_chunk_chars || 400
  const len = store.form.text.trim().length
  if (!len) return 0
  return Math.max(1, Math.ceil(len / limit))
})

// 路由提示文案以 emoji 开头，据此配色
const HINT_STYLES = [
  ['⚡', 'direct', '⚡'],
  ['🔒', 'lock', '🔒'],
  ['🎭', 'emotion', '🎭'],
  ['🎙', 'base', '🎙️'],
  ['⚠', 'warn', '⚠️'],
  ['🔓', 'warn', '🔓'],
  ['⌨', 'idle', '⌨️'],
  ['💡', 'lock', '💡'],
]
const hintStyleKey = computed(() => {
  for (const [prefix, cls] of HINT_STYLES) {
    if (store.routeHint.startsWith(prefix)) return cls
  }
  return 'idle'
})
const hintClass = computed(() => `hint-${hintStyleKey.value}`)
const hintIcon = computed(() => {
  for (const [prefix, , icon] of HINT_STYLES) {
    if (store.routeHint.startsWith(prefix)) return icon
  }
  return '💡'
})
const hintBody = computed(() => store.routeHint.replace(/^(\S+)\s*/, ''))

function applyPreset(key) {
  const preset = store.meta?.presets.find((p) => p.key === key)
  if (!preset) return
  store.form.presetKey = key
  store.form.prompt = preset.prompt || ''
  store.form.text = preset.text || ''
  store.form.calibrationText = preset.calibration_text || ''
}

async function doSavePreset() {
  const name = presetName.value.trim()
  if (!name) {
    showToast('请输入预设名称', 'error')
    return
  }
  try {
    await savePreset({
      name,
      prompt: store.form.prompt,
      text: store.form.text,
      calibration_text: store.form.calibrationText,
    })
    showToast(`已保存预设「${name}」`, 'ok')
    presetName.value = ''
    savingPreset.value = false
    store.meta = await fetchMeta()
    store.form.presetKey = name
  } catch (err) {
    showToast(err.message, 'error')
  }
}
</script>

<style scoped>
textarea { transition: min-height 0.2s; }
.route-hint {
  display: flex;
  gap: 9px;
  align-items: flex-start;
  margin-top: 10px;
  padding: 9px 12px;
  border-radius: var(--radius-sm);
  font-size: 12.5px;
  line-height: 1.5;
  border: 1px solid var(--border);
  background: rgba(255, 255, 255, 0.03);
  color: var(--text-dim);
}
.route-hint .icon { flex-shrink: 0; }
.hint-direct { border-color: rgba(74, 124, 89, 0.4); background: rgba(74, 124, 89, 0.08); }
.hint-lock { border-color: rgba(51, 100, 111, 0.42); background: rgba(51, 100, 111, 0.07); }
.hint-emotion { border-color: rgba(142, 69, 101, 0.45); background: rgba(142, 69, 101, 0.08); }
.hint-warn { border-color: rgba(169, 123, 36, 0.45); background: rgba(169, 123, 36, 0.08); }
.hint-base { border-color: rgba(51, 100, 111, 0.42); }

.preset-row { display: flex; gap: 10px; align-items: flex-end; margin-top: 12px; }
.preset-row .btn { margin-bottom: 1px; }
.save-preset-row { display: flex; gap: 8px; margin-top: 9px; }
</style>
