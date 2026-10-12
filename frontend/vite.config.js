import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Single FastAPI backend serves /api, /ml-api (alias) and /auth.
const apiTarget = process.env.VITE_API_PROXY_TARGET || 'http://localhost:8000';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    watch: {
      usePolling: true,
      interval: 300,
    },
    proxy: {
      '/ml-api': {
        target: apiTarget,
        changeOrigin: true,
        ws: true,
        rewrite: (path) => path.replace(/^\/ml-api/, '/api'),
      },
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
      '/auth': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
});
