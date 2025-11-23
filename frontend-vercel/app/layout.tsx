import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'NBA Prediction Dashboard v3.0',
  description: 'AI-powered NBA game predictions with 67.1% accuracy. Cutting-edge data visualization and mobile-first design.',
  keywords: ['NBA', 'predictions', 'sports betting', 'machine learning', 'data visualization'],
  authors: [{ name: 'NBA Prediction Team' }],
  viewport: {
    width: 'device-width',
    initialScale: 1,
    maximumScale: 5,
    userScalable: true,
  },
  themeColor: '#1a365d',
  manifest: '/manifest.json',
  icons: {
    icon: '/favicon.ico',
    apple: '/apple-touch-icon.png',
  },
  openGraph: {
    type: 'website',
    title: 'NBA Prediction Dashboard v3.0',
    description: 'AI-powered NBA game predictions with 67.1% accuracy',
    siteName: 'NBA Predictions',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <head>
        {/* PWA meta tags */}
        <meta name="application-name" content="NBA Predictions" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="default" />
        <meta name="apple-mobile-web-app-title" content="NBA Predictions" />
        <meta name="mobile-web-app-capable" content="yes" />

        {/* Preconnect to external domains */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className="min-h-screen flex flex-col">
        {/* Main content */}
        <main className="flex-1">
          {children}
        </main>

        {/* Footer */}
        <footer className="bg-primary text-white py-8 px-4 no-print">
          <div className="max-w-7xl mx-auto text-center">
            <p className="text-sm text-gray-300">
              NBA Prediction Dashboard v3.0 | Model v2.0.0 | 67.1% Validated Accuracy
            </p>
            <p className="text-xs text-gray-400 mt-2">
              ⚠️ For educational purposes only. Bet responsibly.
            </p>
            <p className="text-xs text-gray-400 mt-4">
              Built with Next.js | Deployed on Vercel | Powered by AI
            </p>
          </div>
        </footer>
      </body>
    </html>
  )
}
