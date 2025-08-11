/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'hacker-green': '#00ff00',
        'dark-bg': '#000000',
        'dark-surface': '#111111',
        'dark-border': '#333333',
        'error-red': '#ff0000',
        'warning-orange': '#ff8800'
      },
      fontFamily: {
        'mono': ['JetBrains Mono', 'Consolas', 'monospace'],
        'display': ['UnifrakturCook', 'serif']
      },
      animation: {
        'pulse-green': 'pulse-green 2s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate'
      },
      keyframes: {
        'pulse-green': {
          '0%, 100%': { 
            boxShadow: '0 0 5px #00ff0055'
          },
          '50%': { 
            boxShadow: '0 0 20px #00ff00aa, 0 0 30px #00ff0055' 
          }
        },
        'glow': {
          '0%': { 
            textShadow: '0 0 5px #00ff00, 0 0 10px #00ff00' 
          },
          '100%': { 
            textShadow: '0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 30px #00ff00' 
          }
        }
      }
    },
  },
  plugins: [],
  darkMode: 'class'
}