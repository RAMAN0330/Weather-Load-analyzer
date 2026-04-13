/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#16141F',
        sidebar: '#0B0A12',
        card: '#1F1D2B',
        primary: '#7C3AED',
      }
    }
  },
  plugins: [],
}

