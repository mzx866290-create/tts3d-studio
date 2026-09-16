<template>
  <div class="panel">
    <h2 class="panel-title">
      🎨 声音设计
      <span class="hint" v-if="isQwen3">提示词 + seed 决定音色；校准句锁定它</span>
      <span class="hint" v-else>上传参考音频进行克隆</span>
    </h2>

    <template v-if="isQwen3">
      <div class="field">
        <label>声音描述（英文）<span class="tip">描述音色、质感、距离感</span></label>
        <textarea
          v-model="store.form.prompt"
          rows="2"
          placeholder="A warm, gentle female voice, very close to the ear, whispering."
        ></textarea>
      </div>

      <div class="field">
        <label>情绪基调句<span class="tip">决定参考音的语气情绪，修改会更换音色</span></label>
        <input type="text" v-model="store.form.calibrationText" placeholder="留空则使用全局校准句（TTS_VOICE_CALIBRATION_TEXT）" />
      </div>

      <div class="field">
        <label>情感指令（可选）<span class="tip">逐块控制演绎情感，同一音色可换情绪 · 走 CosyVoice2</span></label>
        <input type="text" v-model="store.form.emotionInstruct" placeholder="例如：用激动而悲伤的语气说，声音颤抖" />
        <div class="chips">
          <button
            v-for="ep in store.meta?.emotion_presets || []"
            :key="ep.label"
            class="chip"
            :class="{ active: store.form.emotionInstruct === ep.text }"
            @click="toggleEmotion(ep.text)"
          >{{ ep.label }}</button>
        </div>
      </div>

      <div class="field">
        <label>参考音试听<span class="tip">与正文无关，确认音色满意后再生成正文</span></label>
        <WaveformPlayer v-if="store.clip?.url" :src="store.clip.url" compact />
        <div v-else class="empty-hint" style="padding: 12px;">还没有参考音 — 点下方按钮先生成一句试听</div>
        <div class="row" style="margin-top: 9px;">
          <button class="btn sm" :disabled="store.clipBusy" @click="makeClip(false)">
            <span v-if="store.clipBusy" class="spin">◌</span>
            {{ store.clip?.url ? '▶ 重新生成参考音' : '▶ 生成参考音' }}
          </button>
          <button class="btn sm" :disabled="store.clipBusy || !store.clip?.url" @click="makeClip(true)">
            🎲 换一把声音
          </button>
          <span class="faint mono" style="font-size: 11px;" v-if="store.clip?.clipId">clip {{ store.clip.clipId.slice(0, 8) }}</span>
        </div>
      </div>
    </template>

    <template v-else>
      <div class="field">
        <label>参考音频<span class="tip">上传一段干净的人声（wav/mp3）</span></label>
        <label class="upload-box" :class="{ has: store.form.refAudioUrl }">
          <input type="file" accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a" @change="onUpload" style="display:none" />
          <template v-if="uploading"><span class="spin">◌</span> 上传中…</template>
          <template v-else-if="store.form.refAudioUrl">🎵 {{ store.form.refAudioName || '已上传' }}（点击替换）</template>
          <template v-else>⬆ 点击上传参考音频</template>
        </label>
        <WaveformPlayer v-if="store.form.refAudioUrl" :src="store.form.refAudioUrl" compact style="margin-top: 9px;" />
      </div>
      <div class="field">
        <label>参考文本（可选）<span class="tip">填写准确文本可显著提升克隆质量</span></label>
        <div class="row">
          <textarea v-model="store.form.refText" rows="2" class="grow" placeholder="参考音频里说的话；留空时若安装了 faster-whisper 会自动识别"></textarea>
        </div>
        <button class="btn sm" style="margin-top: 8px;" :disabled="!store.form.refAudioPath || !store.meta?.asr_available || transcribing" @click="doTranscribe">
          <span v-if="transcribing" class="spin">◌</span>
          🎙️ 自动识别参考文本
        </button>
        <div v-if="!store.meta?.asr_available" class="faint" style="font-size: 11.5px; margin-top: 5px;">
          未安装 ASR 后端（pip install faster-whisper 后可用）
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { store, showToast } from '../store.js'
import { previewClip, uploadReferenceAudio, transcribeReference } from '../api.js'
import WaveformPlayer from './WaveformPlayer.vue'

const isQwen3 = computed(() => store.form.engine === 'qwen3')
const uploading = ref(false)
const transcribing = ref(false)

function toggleEmotion(text) {
  store.form.emotionInstruct = store.form.emotionInstruct === text ? '' : text
}

async function makeClip(reroll) {
  store.clipBusy = true
  try {
    const r = await previewClip({
      voice_description: store.form.prompt,
      seed: store.form.seed,
      tts_engine: store.form.engine,
      tts_model_key: store.form.model,
      calibration_text: store.form.calibrationText,
      reroll,
    })
    if (r.clip_url) {
      store.clip = { url: r.clip_url, clipId: r.clip_id, status: r.status }
    }
    if (r.seed !== undefined && r.seed !== null) {
      store.form.seed = r.seed
      store.form.randomSeed = false
    }
    showToast(r.status || '参考音已就绪', 'ok')
  } catch (err) {
    if (String(err.message).includes('进行中')) showToast(err.message, 'error')
    else showToast(`参考音生成失败: ${err.message}`, 'error')
  } finally {
    store.clipBusy = false
  }
}

async function onUpload(e) {
  const file = e.target.files?.[0]
  if (!file) return
  uploading.value = true
  try {
    const r = await uploadReferenceAudio(file)
    store.form.refAudioPath = r.path
    store.form.refAudioUrl = r.url
    store.form.refAudioName = r.name
    showToast('参考音频已上传', 'ok')
  } catch (err) {
    showToast(`上传失败: ${err.message}`, 'error')
  } finally {
    uploading.value = false
    e.target.value = ''
  }
}

async function doTranscribe() {
  if (!store.form.refAudioPath) return
  transcribing.value = true
  try {
    store.form.refText = await transcribeReference(store.form.refAudioPath)
    showToast('识别完成', 'ok')
  } catch (err) {
    showToast(err.message, 'error')
  } finally {
    transcribing.value = false
  }
}
</script>

<style scoped>
.chips { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 8px; }
.chip.active {
  border-color: rgba(139, 92, 246, 0.7);
  background: rgba(139, 92, 246, 0.16);
  color: #c4b5fd;
}
.upload-box {
  display: flex; align-items: center; justify-content: center; gap: 8px;
  border: 1.5px dashed var(--border-strong);
  border-radius: var(--radius-sm);
  padding: 16px;
  color: var(--text-dim);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.15s;
}
.upload-box:hover { border-color: rgba(139, 92, 246, 0.6); color: var(--text); background: rgba(139, 92, 246, 0.05); }
.upload-box.has { border-style: solid; color: var(--text); }
</style>
