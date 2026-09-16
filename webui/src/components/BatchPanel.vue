<template>
  <div class="batch-panel">
    <p class="desc dim">
      复用上方的文稿、声音与参数设置，对多个随机 seed × 所选效果类型批量生成，适合挑选最佳变体。
    </p>
    <div class="controls">
      <div class="field" style="margin: 0;">
        <label>变体数量（随机 seed）</label>
        <input type="number" min="1" max="20" v-model.number="count" style="width: 90px;" />
      </div>
      <div class="field grow" style="margin: 0;">
        <label>效果类型</label>
        <div class="chips">
          <button
            v-for="m in availableModes"
            :key="m.key"
            class="chip"
            :class="{ active: effects.includes(m.key) }"
            @click="toggleEffect(m.key)"
          >{{ m.label }}</button>
        </div>
      </div>
      <button class="btn primary" :disabled="store.batch.running" @click="run" style="align-self: flex-end;">
        <span v-if="store.batch.running" class="spin">◌</span>
        {{ store.batch.running ? '生成中…' : '📦 开始批量生成' }}
      </button>
      <button class="btn danger" :disabled="!store.batch.running" @click="stop" style="align-self: flex-end;">⏹ 停止</button>
    </div>

    <div v-if="store.batch.running || store.batch.summary" class="progress-area">
      <div class="progress-meta mono" v-if="store.batch.running">
        <span class="pulse">{{ store.batch.stageText }}</span>
        <span v-if="store.batch.elapsed">{{ store.batch.elapsed.toFixed(0) }}s</span>
      </div>
      <div class="summary" v-if="store.batch.summary">{{ store.batch.summary }}</div>
      <div class="items" v-if="store.batch.items.length">
        <div
          v-for="(item, i) in store.batch.items"
          :key="i"
          class="item"
          :class="{ failed: !item.file_path }"
        >
          <span class="mono">{{ effectLabel(item.effect_type) }}</span>
          <span class="mono dim">seed {{ item.seed }}</span>
          <button class="btn sm" v-if="item.audio_url" @click="play(item)">▶</button>
          <span class="dim" style="font-size: 11.5px;">{{ item.status }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { store, showToast } from '../store.js'
import { startBatch, stopGenerate } from '../generate.js'

const count = ref(3)
const effects = ref(['mono'])

const availableModes = computed(() => (store.meta?.modes || []).filter((m) => m.available))

function toggleEffect(key) {
  const idx = effects.value.indexOf(key)
  if (idx >= 0) effects.value.splice(idx, 1)
  else effects.value.push(key)
}

function effectLabel(key) {
  return store.meta?.modes.find((m) => m.key === key)?.label || key
}

function run() {
  startBatch({ count: Math.min(20, Math.max(1, count.value)), effects: [...effects.value] })
}

function stop() {
  stopGenerate()
}

function play(item) {
  store.result = {
    url: item.audio_url,
    filePath: item.file_path,
    seed: item.seed,
    status: `批量产物：${effectLabel(item.effect_type)} · seed ${item.seed}`,
    clip_id: '',
  }
  showToast('已在上方播放器中载入', 'ok')
}
</script>

<style scoped>
.batch-panel { display: flex; flex-direction: column; gap: 13px; }
.desc { margin: 0; font-size: 12.5px; }
.controls { display: flex; gap: 16px; align-items: flex-end; flex-wrap: wrap; }
.chips { display: flex; gap: 7px; flex-wrap: wrap; }
.chip.active { border-color: rgba(139, 92, 246, 0.7); background: rgba(139, 92, 246, 0.16); color: #c4b5fd; }
.progress-area { border-top: 1px solid var(--border); padding-top: 11px; display: flex; flex-direction: column; gap: 9px; }
.progress-meta { display: flex; justify-content: space-between; font-size: 11.5px; color: var(--text-dim); }
.summary { font-size: 12.5px; color: var(--ok); }
.items { display: flex; flex-direction: column; gap: 5px; }
.item {
  display: flex; align-items: center; gap: 13px;
  font-size: 12px; padding: 5px 9px;
  border: 1px solid var(--border); border-radius: 7px;
}
.item.failed { color: var(--danger); }
</style>
