<template>
  <div>
    <div class="toolbar">
      <input type="text" v-model="search" placeholder="搜索文件名…" class="grow" style="max-width: 260px;" />
      <button class="btn sm" @click="refresh">↻ 刷新</button>
    </div>

    <div v-if="!filtered.length" class="empty-hint" style="margin-top: 10px;">
      还没有生成的音频文件。
    </div>
    <table v-else class="hist-table">
      <thead>
        <tr><th>文件名</th><th>修改时间</th><th style="text-align:right;">大小</th><th style="width: 130px;"></th></tr>
      </thead>
      <tbody>
        <tr v-for="item in filtered" :key="item.path" :class="{ playing: isPlaying(item) }">
          <td class="mono name" :title="item.path">{{ item.name }}</td>
          <td class="mono dim">{{ item.modified }}</td>
          <td class="mono dim" style="text-align:right;">{{ fmtSize(item.size) }}</td>
          <td style="text-align:right;">
            <button class="btn sm" @click="play(item)">▶ 播放</button>
            <button class="btn sm danger" @click="confirmRemove(item)">
              {{ pendingDelete === item.path ? '确认?' : '🗑' }}
            </button>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { store, showToast } from '../store.js'
import { deleteHistory, fetchHistory } from '../api.js'

const search = ref('')
const pendingDelete = ref('')

const filtered = computed(() => {
  const kw = search.value.trim().toLowerCase()
  const items = store.history || []
  if (!kw) return items
  return items.filter((f) => f.name.toLowerCase().includes(kw))
})

function isPlaying(item) {
  return store.result?.filePath === item.path
}

function fmtSize(bytes) {
  if (bytes > 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + ' MB'
  return (bytes / 1024).toFixed(0) + ' KB'
}

function play(item) {
  store.result = { url: item.url, filePath: item.path, seed: undefined, status: `历史回放：${item.name}`, clip_id: '' }
}

async function refresh() {
  try {
    store.history = await fetchHistory()
  } catch { /* noop */ }
}

async function confirmRemove(item) {
  if (pendingDelete.value !== item.path) {
    pendingDelete.value = item.path
    setTimeout(() => {
      if (pendingDelete.value === item.path) pendingDelete.value = ''
    }, 3000)
    return
  }
  try {
    const r = await deleteHistory(item.path)
    store.history = r.history || []
    showToast(`已删除 ${item.name}`)
  } catch (err) {
    showToast(err.message, 'error')
  } finally {
    pendingDelete.value = ''
  }
}
</script>

<style scoped>
.toolbar { display: flex; gap: 9px; margin-bottom: 11px; }
.hist-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.hist-table th {
  text-align: left; color: var(--text-faint); font-weight: 500;
  font-size: 11px; letter-spacing: 0.5px;
  padding: 6px 10px; border-bottom: 1px solid var(--border);
}
.hist-table td { padding: 7px 10px; border-bottom: 1px solid rgba(255, 255, 255, 0.04); }
.hist-table tr:hover td { background: rgba(255, 255, 255, 0.025); }
.hist-table tr.playing td { background: rgba(176, 58, 46, 0.08); }
.hist-table .name { max-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
