import { store, progressFromStage, stageTextFromStage, showToast } from './store.js'
import { sseRequest, cancelGenerate, fetchHistory } from './api.js'

export function buildPayload() {
  const f = store.form
  return {
    text: f.text,
    voice_description: f.prompt,
    preset_key: f.presetKey,
    seed: f.randomSeed ? null : f.seed === '' || f.seed === null ? null : Number(f.seed),
    use_random_seed: !!f.randomSeed,
    render_mode: f.mode,
    static_azimuth_deg: Number(f.staticAzimuth),
    static_distance_m: Number(f.staticDistance),
    dynamic_path: f.dynamicPath,
    dynamic_cycle_time_s: Number(f.dynamicCycle),
    dynamic_start_azimuth_deg: Number(f.dynamicStart),
    dynamic_distance_m: Number(f.dynamicDistance),
    tts_engine: f.engine,
    tts_model_key: f.model,
    reference_audio_path: f.refAudioPath,
    reference_text: f.refText,
    speed_factor: Number(f.speed),
    pitch_semitones: Number(f.pitch),
    calibration_text: f.calibrationText,
    emotion_instruct: f.emotionInstruct,
    lock_timbre: !!f.lockTimbre,
  }
}

export async function stopGenerate() {
  store.gen.cancelRequested = true
  try {
    await cancelGenerate()
  } catch { /* 忽略 */ }
}

// 流式连接意外中断后的兜底：任务很可能仍在服务端执行，
// 轮询历史记录直到出现新文件（最多 ~20 秒），把结果找回来。
async function recoverResult(baselineTop) {
  for (let attempt = 0; attempt < 8; attempt++) {
    if (store.gen.cancelRequested) return false
    await new Promise((r) => setTimeout(r, 2500))
    try {
      const hist = await fetchHistory()
      store.history = hist
      const top = hist[0]
      if (top && top.path && top.path !== baselineTop) {
        store.result = {
          url: top.url,
          filePath: top.path,
          seed: store.form.seed,
          status: `生成完成（连接曾中断，已从历史恢复）· ${top.name}`,
          clip_id: '',
        }
        return true
      }
    } catch { /* 网络抖动，继续重试 */ }
  }
  return false
}

export async function startGenerate() {
  if (store.gen.phase === 'generating') return
  store.gen.phase = 'generating'
  store.gen.progress = 0
  store.gen.stageText = '任务提交中…'
  store.gen.elapsed = 0
  store.gen.cancelRequested = false
  const baselineTop = store.history[0]?.path || ''
  let gotResult = false
  let serverFailed = false
  try {
    await sseRequest('/api/generate', buildPayload(), {
      progress(p) {
        store.gen.progress = progressFromStage(p.stage, p)
        store.gen.stageText = stageTextFromStage(p.stage, p)
        store.gen.elapsed = p.elapsed
      },
      ping(p) {
        store.gen.elapsed = p.elapsed
      },
      result(r) {
        gotResult = true
        // 后端字段是 audio_url/file_path，这里归一化成前端通用字段
        store.result = {
          url: r.audio_url,
          filePath: r.file_path,
          seed: r.seed,
          status: r.status,
          clip_id: r.clip_id || '',
        }
        // 与旧版 UI 一致：生成后回填实际 seed 并关闭随机，便于复现
        store.form.seed = r.seed
        store.form.randomSeed = false
        if (r.clip_id) {
          store.clip = { url: r.clip_url, clipId: r.clip_id, status: '' }
        }
      },
      cancelled() {
        showToast('已中断生成')
      },
      error(p) {
        serverFailed = true
        showToast(p.message || '生成失败', 'error', 6000)
      },
    })
    if (!gotResult && !serverFailed) {
      store.gen.stageText = '连接中断，正在确认生成结果…'
      const recovered = await recoverResult(baselineTop)
      if (recovered) {
        store.form.randomSeed = false
        showToast('生成完成（连接中断后已恢复结果）🎧', 'ok')
      } else if (!store.gen.cancelRequested) {
        showToast('连接中断且未能确认结果，请稍后在历史记录中查看', 'error', 7000)
      }
    } else if (gotResult) {
      showToast('生成完成 🎧', 'ok')
    }
  } catch (err) {
    if (!serverFailed) {
      // fetch 中途断开等异常：任务可能仍在服务端执行，尝试恢复
      store.gen.stageText = '连接中断，正在确认生成结果…'
      const recovered = await recoverResult(baselineTop)
      if (recovered) {
        store.form.randomSeed = false
        showToast('生成完成（连接中断后已恢复结果）🎧', 'ok')
      } else if (!store.gen.cancelRequested) {
        showToast(`${err.message}；未能确认结果，请稍后在历史记录中查看`, 'error', 7000)
      }
    }
  } finally {
    store.gen.phase = 'idle'
    store.gen.stageText = ''
    fetchHistory().then((h) => { store.history = h }).catch(() => {})
  }
}

export async function startBatch({ count, effects }) {
  if (store.batch.running) return
  if (!effects.length) {
    showToast('请至少选择一个效果类型', 'error')
    return
  }
  store.batch.running = true
  store.batch.progress = 0
  store.batch.stageText = '任务提交中…'
  store.batch.items = []
  store.batch.summary = ''
  const payload = { ...buildPayload(), batch_count: count, effect_types: effects }
  try {
    await sseRequest('/api/batch', payload, {
      progress(p) {
        if (p.stage === 'batch_render') {
          store.batch.stageText = `已产出 ${p.index} 个文件…`
        } else if (p.stage === 'batch_seed') {
          store.batch.stageText = `变体 ${p.seed_index}/${p.seed_total}（seed=${p.seed}）…`
        } else {
          store.batch.stageText = stageTextFromStage(p.stage, p) || store.batch.stageText
        }
        store.batch.elapsed = p.elapsed
      },
      ping(p) {
        store.batch.elapsed = p.elapsed
      },
      result(r) {
        store.batch.items = r.items || []
        store.batch.summary = `完成 ${r.succeeded}/${r.total}，失败 ${r.failed}`
        store.history = r.history || store.history
        showToast(`批量生成完成：${r.succeeded}/${r.total}`, 'ok')
      },
      cancelled() {
        showToast('已中断批量生成')
      },
      error() { /* 走 catch */ },
    })
  } catch (err) {
    showToast(err.message, 'error', 5200)
  } finally {
    store.batch.running = false
    store.batch.stageText = ''
    fetchHistory().then((h) => { store.history = h }).catch(() => {})
  }
}
