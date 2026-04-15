import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const djangoTarget = process.env.VITE_DJANGO_PROXY_TARGET || 'http://localhost:8001'
const fastapiTarget = process.env.VITE_FASTAPI_PROXY_TARGET || 'http://localhost:8000'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    watch: {
      usePolling: true,
      interval: 300
    },
    proxy: {
      // Forecast/model-training traffic only.
      '/ml-api': {
        target: fastapiTarget,
        changeOrigin: true,
        ws: true,
        rewrite: (path) => path.replace(/^\/ml-api/, '/api'),
      },
      // Normal application/API traffic goes through Django.
      '/api': {
        target: djangoTarget,
        changeOrigin: true,
      }
    }
  }
})
