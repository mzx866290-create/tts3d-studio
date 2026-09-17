<template>
  <div class="panel">
    <h2 class="panel-title">
      🎧 生成结果
      <span class="hint" v-if="store.result?.seed !== undefined">seed {{ store.result.seed }}</span>
    </h2>

    <template v-if="store.result?.url">
      <WaveformPlayer :key="store.result.url" :src="store.result.url" autoplay />
      <div class="status mono">{{ store.result.status }}</div>
      <div class="actions">
        <button class="btn sm" @click="download" v-if="downloadUrl">⬇ 下载</button>
        <button class="btn sm" @click="saveOpen = !saveOpen">⭐ 收藏当前声线</button>
      </div>
      <div class="fav-form" v-if="saveOpen">
        <input
          type="text"
          v-model="favName"
          placeholder="给这条声线起个名，如：左耳温柔低语"
          @keyup.enter="doSaveFavorite"
        />
        <button class="btn sm primary" @click="doSaveFavorite">保存</button>
      </div>
      <div class="faint" style="font-size: 11.5px; margin-top: 7px;" v-if="store.result.clip_id">
        参考音 clip {{ store.result.clip_id.slice(0, 10) }}（收藏时会一并保存）
      </div>
    </template>
    <div v-else class="empty-hint" style="padding: 34px 18px;">
      还没有生成结果。写好文稿、选好空间效果，点「🎵 生成音频」试试。
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { store, showToast } from '../store.js'
import { saveFavorite } from '../api.js'
import WaveformPlayer from './WaveformPlayer.vue'

const saveOpen = ref(false)
const favName = ref('')

const downloadUrl = computed(() => {
  if (!store.result?.url) return ''
  return store.result.url + '&download=1'
})

function download() {
  if (!downloadUrl.value) return
  const a = document.createElement('a')
  a.href = downloadUrl.value
  a.download = store.result.filePath?.split('/').pop() || 'voice.wav'
  a.click()
}

async function doSaveFavorite() {
  const name = favName.value.trim()
  if (!name) {
    showToast('请先填入收藏名称', 'error')
    return
  }
  const f = store.form
  try {
    await saveFavorite({
      name,
      prompt: f.prompt,
      text: f.text,
      calibration_text: f.calibrationText,
      emotion_instruct: f.emotionInstruct,
      seed: store.result?.seed ?? f.seed,
      mode_label: modeLabel(f.mode),
      speed_factor: Number(f.speed),
      pitch_semitones: Number(f.pitch),
      clip_id: store.result?.clip_id || store.clip?.clipId || '',
      lock_timbre: !!f.lockTimbre,
    })
    showToast(`已收藏声线「${name}」`, 'ok')
    favName.value = ''
    saveOpen.value = false
  } catch (err) {
    showToast(err.message, 'error')
  }
}

function modeLabel(key) {
  const m = store.meta?.modes.find((m) => m.key === key)
  return m?.label || '虚拟脑后'
}
</script>

<style scoped>
.status {
  font-size: 11px;
  color: var(--text-faint);
  margin-top: 9px;
  word-break: break-all;
  line-height: 1.5;
}
.actions { display: flex; gap: 8px; margin-top: 11px; }
.fav-form { display: flex; gap: 8px; margin-top: 10px; }
</style>
