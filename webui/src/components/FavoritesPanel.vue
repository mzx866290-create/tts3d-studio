<template>
  <div>
    <div v-if="!store.favorites.length" class="empty-hint">
      还没有收藏的声线。生成一段满意的声音后，在「生成结果」里点 ⭐ 收藏，就能一键复用同一把声音。
    </div>
    <div class="fav-grid" v-else>
      <div class="fav-card" v-for="fav in store.favorites" :key="fav.name">
        <div class="fav-head">
          <span class="fav-name">{{ fav.name }}</span>
          <span class="faint mono" style="font-size: 10.5px;" v-if="fav.clip_id">clip</span>
        </div>
        <div class="fav-prompt" :title="fav.prompt">{{ fav.prompt || '（无提示词）' }}</div>
        <div class="fav-meta mono">
          <span>seed {{ fav.seed ?? '—' }}</span>
          <span>{{ fav.mode_label }}</span>
          <span v-if="fav.emotion_instruct" class="emo" :title="fav.emotion_instruct">🎭 情感</span>
          <span v-if="fav.speed_factor != 1">×{{ fav.speed_factor }}</span>
          <span v-if="fav.pitch_semitones">{{ fav.pitch_semitones > 0 ? '+' : '' }}{{ fav.pitch_semitones }}st</span>
        </div>
        <div class="fav-actions">
          <button class="btn sm primary" @click="apply(fav)">应用</button>
          <button class="btn sm" @click="playClip(fav)" v-if="fav.clip_id">试听</button>
          <button class="btn sm danger" @click="remove(fav)">删除</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { store, showToast } from '../store.js'
import { deleteFavorite, clipUrl } from '../api.js'

async function apply(fav) {
  const f = store.form
  f.prompt = fav.prompt || ''
  f.text = fav.text || ''
  f.calibrationText = fav.calibration_text || ''
  f.emotionInstruct = fav.emotion_instruct || ''
  f.seed = fav.seed ?? 42
  f.randomSeed = false
  f.speed = fav.speed_factor ?? 1.0
  f.pitch = fav.pitch_semitones ?? 0
  const mode = store.meta?.modes.find((m) => m.label === fav.mode_label && m.available)
  if (mode) f.mode = mode.key
  if (fav.clip_id) {
    try {
      const r = await clipUrl(fav.clip_id)
      store.clip = { url: r.url, clipId: fav.clip_id, status: '' }
    } catch { /* clip 文件可能已被清理 */ }
  } else {
    store.clip = null
  }
  showToast(`已应用声线「${fav.name}」，seed 固定为 ${f.seed}`, 'ok')
}

async function playClip(fav) {
  try {
    const r = await clipUrl(fav.clip_id)
    store.result = { url: r.url, filePath: '', seed: fav.seed, status: `收藏声线「${fav.name}」的参考音试听`, clip_id: fav.clip_id }
  } catch {
    showToast('参考音文件不存在', 'error')
  }
}

async function remove(fav) {
  try {
    const r = await deleteFavorite(fav.name)
    store.favorites = r.favorites || []
    showToast(`已删除「${fav.name}」`)
  } catch (err) {
    showToast(err.message, 'error')
  }
}
</script>

<style scoped>
.fav-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 12px;
}
.fav-card {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: rgba(255, 255, 255, 0.03);
  padding: 13px 14px;
  display: flex;
  flex-direction: column;
  gap: 7px;
  transition: border-color 0.15s;
}
.fav-card:hover { border-color: rgba(176, 58, 46, 0.4); }
.fav-head { display: flex; align-items: center; justify-content: space-between; }
.fav-name { font-weight: 600; font-size: 13.5px; }
.fav-prompt {
  font-size: 11.5px;
  color: var(--text-dim);
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  min-height: 17px;
}
.fav-meta { display: flex; flex-wrap: wrap; gap: 6px 10px; font-size: 10.5px; color: var(--text-faint); }
.fav-meta .emo { color: var(--accent-text); }
.fav-actions { display: flex; gap: 7px; margin-top: 3px; }
</style>
