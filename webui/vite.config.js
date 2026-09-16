import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 构建产物复制到 tts3d_app/webapp/static 后由 FastAPI 托管；
// dev 模式通过代理连后端：先运行 python app.py run，再 npm run dev。
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:7860',
      '/media': 'http://127.0.0.1:7860',
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
