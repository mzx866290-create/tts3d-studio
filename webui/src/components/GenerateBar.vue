<template>
  <div class="panel gen-panel">
    <div class="seed-row">
      <div class="field" style="margin: 0; flex: 0 0 auto;">
        <label>Seed</label>
        <input type="number" v-model="store.form.seed" style="width: 118px;" :disabled="store.form.randomSeed" />
      </div>
      <label class="switch" style="margin-top: 18px;">
        <input type="checkbox" v-model="store.form.randomSeed" />
        <span class="track"></span>
        <span class="label">随机 seed</span>
      </label>
      <div class="spacer"></div>
      <button
        class="btn lg primary"
        :disabled="generating"
        @click="$emit('generate')"
        style="min-width: 178px;"
      >
        <span v-if="generating" class="spin">◌</span>
        {{ generating ? '生成中…' : '🎵 生成音频' }}
      </button>
      <button class="btn lg danger" :disabled="!generating" @click="$emit('stop')">⏹ 停止</button>
    </div>

    <div class="progress-wrap" v-if="generating">
      <div class="progress-bar">
        <div class="progress-fill" :style="{ width: pct + '%' }"></div>
      </div>
      <div class="progress-meta mono">
        <span class="pulse">{{ store.gen.stageText }}</span>
        <span>{{ store.gen.elapsed.toFixed(0) }}s</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { store } from '../store.js'

defineEmits(['generate', 'stop'])

const generating = computed(() => store.gen.phase === 'generating')
const pct = computed(() => Math.min(99, Math.round(store.gen.progress * 100)))
</script>

<style scoped>
.gen-panel { padding: 14px 18px; }
.seed-row { display: flex; gap: 16px; align-items: flex-start; }
.seed-row input:disabled { opacity: 0.4; }
.spacer { flex: 1; }
.progress-wrap { margin-top: 13px; }
.progress-bar {
  height: 6px;
  border-radius: 3px;
  background: rgba(255, 255, 255, 0.08);
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  background: var(--accent-grad);
  border-radius: 3px;
  transition: width 0.4s ease;
  box-shadow: 0 0 12px rgba(176, 58, 46, 0.6);
}
.progress-meta {
  display: flex;
  justify-content: space-between;
  font-size: 11.5px;
  color: var(--text-dim);
  margin-top: 6px;
}
</style>
