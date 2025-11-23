import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Primary (Trust & Stability) - Dark Blue
        primary: {
          DEFAULT: '#1a365d',
          light: '#2c5282',
          dark: '#0f1e35',
        },
        // Secondary (Neutral) - Gray
        secondary: {
          DEFAULT: '#718096',
          light: '#a0aec0',
          dark: '#4a5568',
        },
        // Accent (Energy & CTAs) - Orange
        accent: {
          DEFAULT: '#FF6B35',
          light: '#ff8555',
          dark: '#e55a2b',
        },
        // Confidence Levels (Inverted: LOW = Best)
        confidence: {
          low: {
            bg: '#d4edda',
            border: '#28a745',
            text: '#155724',
          },
          medium: {
            bg: '#fff3cd',
            border: '#ffc107',
            text: '#856404',
          },
          high: {
            bg: '#f8d7da',
            border: '#dc3545',
            text: '#721c24',
          },
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Poppins', 'system-ui', 'sans-serif'],
      },
      fontSize: {
        'display': ['40px', { lineHeight: '1.2', fontWeight: '700' }],
        'h1': ['32px', { lineHeight: '1.3', fontWeight: '700' }],
        'h2': ['24px', { lineHeight: '1.4', fontWeight: '600' }],
        'h3': ['18px', { lineHeight: '1.5', fontWeight: '600' }],
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
      },
      boxShadow: {
        'card': '0 2px 8px rgba(0, 0, 0, 0.1)',
        'card-hover': '0 8px 16px rgba(0, 0, 0, 0.15)',
        'button': '0 4px 6px rgba(0, 0, 0, 0.1)',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in',
        'slide-up': 'slideUp 0.5s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}
export default config
