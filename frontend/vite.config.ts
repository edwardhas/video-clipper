import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'

// The app always calls relative URLs (/api, /media). In development the Vite
// server proxies them to the local backend; the production bundle is served by
// the backend itself, so no host is ever baked into the build.
const DEV_BACKEND = 'http://127.0.0.1:8765'

export default defineConfig({
  base: '/',
  plugins: [vue()],
  server: {
    proxy: {
      '/api': { target: DEV_BACKEND, changeOrigin: true },
      '/media': { target: DEV_BACKEND, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
  },
})
