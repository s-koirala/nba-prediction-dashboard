'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { TrendingUp, Target, ChevronDown, ChevronUp, Activity, DollarSign, Percent } from 'lucide-react'

// Types
interface Prediction {
  id: string
  homeTeam: string
  awayTeam: string
  predictedWinner: string
  expectedMargin: number
  confidence: 'LOW' | 'MEDIUM' | 'HIGH'
  winProbability: number
  expectedReturn: number
  betAmount: number
  gameTime: string
  modelPredictions: {
    elo: number
    neural: number
    xgboost: number
    ensemble: number
  }
}

// Mock data (will be replaced with API call)
const mockPredictions: Prediction[] = [
  {
    id: '1',
    homeTeam: 'Boston Celtics',
    awayTeam: 'LA Lakers',
    predictedWinner: 'Boston Celtics',
    expectedMargin: 8.5,
    confidence: 'LOW',
    winProbability: 70.8,
    expectedReturn: 127.00,
    betAmount: 140.00,
    gameTime: '7:30 PM ET',
    modelPredictions: {
      elo: 8.2,
      neural: 10.1,
      xgboost: 6.8,
      ensemble: 8.5,
    }
  },
  {
    id: '2',
    homeTeam: 'Golden State Warriors',
    awayTeam: 'Phoenix Suns',
    predictedWinner: 'Phoenix Suns',
    expectedMargin: -5.2,
    confidence: 'MEDIUM',
    winProbability: 63.7,
    expectedReturn: 85.00,
    betAmount: 100.00,
    gameTime: '10:00 PM ET',
    modelPredictions: {
      elo: -5.0,
      neural: -5.5,
      xgboost: -5.1,
      ensemble: -5.2,
    }
  },
  {
    id: '3',
    homeTeam: 'Milwaukee Bucks',
    awayTeam: 'Miami Heat',
    predictedWinner: 'Milwaukee Bucks',
    expectedMargin: 3.1,
    confidence: 'HIGH',
    winProbability: 64.2,
    expectedReturn: 0,
    betAmount: 0,
    gameTime: '8:00 PM ET',
    modelPredictions: {
      elo: 3.5,
      neural: 2.8,
      xgboost: 3.0,
      ensemble: 3.1,
    }
  }
]

export default function Home() {
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [selectedPrediction, setSelectedPrediction] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  // Fetch predictions on mount
  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        // TODO: Replace with actual API call
        // const response = await fetch('/api/predictions')
        // const data = await response.json()
        // setPredictions(data)

        // For now, use mock data
        setTimeout(() => {
          setPredictions(mockPredictions)
          setIsLoading(false)
        }, 500)
      } catch (error) {
        console.error('Failed to fetch predictions:', error)
        setIsLoading(false)
      }
    }

    fetchPredictions()
  }, [])

  // Get confidence styling
  const getConfidenceStyle = (confidence: string) => {
    switch (confidence) {
      case 'LOW':
        return {
          bg: 'bg-confidence-low-bg',
          border: 'border-confidence-low-border',
          text: 'text-confidence-low-text',
          icon: '🟢',
          label: 'BEST',
          winRate: '70.8%'
        }
      case 'MEDIUM':
        return {
          bg: 'bg-confidence-medium-bg',
          border: 'border-confidence-medium-border',
          text: 'text-confidence-medium-text',
          icon: '🟡',
          label: 'MODERATE',
          winRate: '63.7%'
        }
      case 'HIGH':
        return {
          bg: 'bg-confidence-high-bg',
          border: 'border-confidence-high-border',
          text: 'text-confidence-high-text',
          icon: '🔴',
          label: 'SKIP',
          winRate: '64.2%'
        }
      default:
        return {
          bg: 'bg-gray-100',
          border: 'border-gray-300',
          text: 'text-gray-700',
          icon: '⚪',
          label: 'UNKNOWN',
          winRate: 'N/A'
        }
    }
  }

  // Get top pick (highest expected return from LOW confidence)
  const topPick = predictions
    .filter(p => p.confidence === 'LOW')
    .sort((a, b) => b.expectedReturn - a.expectedReturn)[0]

  // Calculate today's stats
  const totalBets = predictions.filter(p => p.betAmount > 0).length
  const totalAmount = predictions.reduce((sum, p) => sum + p.betAmount, 0)
  const totalExpectedReturn = predictions.reduce((sum, p) => sum + p.expectedReturn, 0)
  const expectedROI = totalAmount > 0 ? ((totalExpectedReturn / totalAmount) * 100).toFixed(1) : '0'

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary to-primary-light">
        <div className="text-center text-white">
          <div className="w-16 h-16 border-4 border-white border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-lg font-semibold">Loading predictions...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-primary via-primary-light to-accent text-white py-12 px-4 md:py-20">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-12"
          >
            <h1 className="text-4xl md:text-6xl font-display font-bold mb-4">
              🏀 NBA Predictions
            </h1>
            <p className="text-xl md:text-2xl text-gray-100 mb-2">
              AI-Powered • 67.1% Accuracy • Mobile-First
            </p>
            <div className="flex items-center justify-center gap-4 mt-4 text-sm">
              <span className="badge bg-white/20 text-white border-white/30">
                <Activity className="w-4 h-4 mr-1" />
                Model v2.0.0
              </span>
              <span className="badge bg-white/20 text-white border-white/30">
                <TrendingUp className="w-4 h-4 mr-1" />
                Rolling 4yr Window
              </span>
            </div>
          </motion.div>

          {/* Top Pick Card (Hero) */}
          {topPick && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 }}
              className="max-w-2xl mx-auto"
            >
              <div className="bg-white rounded-2xl shadow-2xl overflow-hidden">
                {/* Header */}
                <div className="bg-gradient-to-r from-confidence-low-border to-green-600 px-6 py-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-white text-sm font-semibold uppercase tracking-wide">
                        🟢 Today's Top Pick
                      </p>
                      <p className="text-white/90 text-xs mt-1">
                        {getConfidenceStyle(topPick.confidence).winRate} Win Rate • Best Performance
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-white text-3xl font-bold">
                        {topPick.winProbability}%
                      </p>
                      <p className="text-white/90 text-xs">Win Probability</p>
                    </div>
                  </div>
                </div>

                {/* Body */}
                <div className="p-6 md:p-8">
                  {/* Matchup */}
                  <div className="mb-6">
                    <div className="flex items-center justify-between gap-4">
                      <div className="flex-1 text-center">
                        <p className="text-2xl md:text-3xl font-bold text-gray-800">
                          {topPick.awayTeam}
                        </p>
                        <p className="text-sm text-gray-500">Away</p>
                      </div>
                      <div className="px-4 py-2 bg-gray-100 rounded-lg">
                        <p className="text-xs text-gray-500 uppercase">vs</p>
                      </div>
                      <div className="flex-1 text-center">
                        <p className="text-2xl md:text-3xl font-bold text-gray-800">
                          {topPick.homeTeam}
                        </p>
                        <p className="text-sm text-gray-500">Home</p>
                      </div>
                    </div>
                    <p className="text-center text-gray-600 mt-2">{topPick.gameTime}</p>
                  </div>

                  {/* Prediction */}
                  <div className="bg-gradient-to-r from-confidence-low-bg to-green-50 rounded-xl p-6 mb-6">
                    <div className="flex items-center justify-between mb-3">
                      <div>
                        <p className="text-sm text-gray-600 font-semibold uppercase">Predicted Winner</p>
                        <p className="text-2xl md:text-3xl font-bold text-gray-900 mt-1">
                          {topPick.predictedWinner}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-sm text-gray-600 font-semibold uppercase">Margin</p>
                        <p className="text-2xl md:text-3xl font-bold text-confidence-low-border mt-1">
                          {topPick.expectedMargin > 0 ? '+' : ''}{topPick.expectedMargin.toFixed(1)}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Betting Info */}
                  <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="text-center p-4 bg-gray-50 rounded-lg">
                      <DollarSign className="w-6 h-6 text-accent mx-auto mb-2" />
                      <p className="text-xs text-gray-600 uppercase">Bet Amount</p>
                      <p className="text-lg font-bold text-gray-900">${topPick.betAmount.toFixed(0)}</p>
                    </div>
                    <div className="text-center p-4 bg-gray-50 rounded-lg">
                      <TrendingUp className="w-6 h-6 text-green-600 mx-auto mb-2" />
                      <p className="text-xs text-gray-600 uppercase">Expected Win</p>
                      <p className="text-lg font-bold text-green-600">${topPick.expectedReturn.toFixed(0)}</p>
                    </div>
                    <div className="text-center p-4 bg-gray-50 rounded-lg">
                      <Percent className="w-6 h-6 text-primary mx-auto mb-2" />
                      <p className="text-xs text-gray-600 uppercase">ROI</p>
                      <p className="text-lg font-bold text-primary">
                        {((topPick.expectedReturn / topPick.betAmount) * 100 - 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>

                  {/* CTA Buttons */}
                  <div className="flex gap-3">
                    <button className="btn btn-primary flex-1 touch-target">
                      <Target className="w-5 h-5 inline mr-2" />
                      Place Bet
                    </button>
                    <button
                      onClick={() => setSelectedPrediction(topPick.id)}
                      className="btn btn-secondary touch-target"
                    >
                      View Details
                    </button>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Today's Summary Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="mt-8 grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto"
          >
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 text-center border border-white/20">
              <p className="text-white/80 text-sm uppercase font-semibold">Games Today</p>
              <p className="text-3xl font-bold text-white mt-1">{predictions.length}</p>
            </div>
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 text-center border border-white/20">
              <p className="text-white/80 text-sm uppercase font-semibold">Recommended Bets</p>
              <p className="text-3xl font-bold text-white mt-1">{totalBets}</p>
            </div>
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 text-center border border-white/20">
              <p className="text-white/80 text-sm uppercase font-semibold">Total to Bet</p>
              <p className="text-3xl font-bold text-white mt-1">${totalAmount.toFixed(0)}</p>
            </div>
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 text-center border border-white/20">
              <p className="text-white/80 text-sm uppercase font-semibold">Expected ROI</p>
              <p className="text-3xl font-bold text-green-300 mt-1">+{expectedROI}%</p>
            </div>
          </motion.div>
        </div>
      </section>

      {/* All Games Section */}
      <section className="py-12 px-4">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-h2 font-display font-bold text-gray-900 mb-8 text-center md:text-left">
            All Games Today
          </h2>

          {/* Games List */}
          <div className="space-y-4">
            {predictions.map((prediction, index) => {
              const style = getConfidenceStyle(prediction.confidence)
              const isExpanded = selectedPrediction === prediction.id

              return (
                <motion.div
                  key={prediction.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className={`${style.bg} border-l-4 ${style.border} rounded-lg overflow-hidden transition-all duration-200`}
                >
                  {/* Card Header (Always Visible) */}
                  <div className="p-4 md:p-6">
                    <div className="flex items-start justify-between gap-4">
                      {/* Left: Matchup */}
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-2xl">{style.icon}</span>
                          <span className={`text-xs font-bold uppercase ${style.text}`}>
                            {style.label}
                          </span>
                          <span className="text-xs text-gray-500">
                            {style.winRate} Win Rate
                          </span>
                        </div>
                        <p className="text-lg font-bold text-gray-900">
                          {prediction.awayTeam} @ {prediction.homeTeam}
                        </p>
                        <p className="text-sm text-gray-600">{prediction.gameTime}</p>
                      </div>

                      {/* Right: Quick Stats */}
                      <div className="text-right">
                        <p className="text-xs text-gray-600 uppercase font-semibold">Predicted</p>
                        <p className="text-lg font-bold text-gray-900">
                          {prediction.predictedWinner.split(' ').pop()}
                        </p>
                        <p className={`text-xl font-bold ${prediction.expectedMargin > 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {prediction.expectedMargin > 0 ? '+' : ''}{prediction.expectedMargin.toFixed(1)}
                        </p>
                      </div>
                    </div>

                    {/* Betting Info (Collapsed State) */}
                    {!isExpanded && (
                      <div className="mt-4 flex items-center justify-between">
                        {prediction.betAmount > 0 ? (
                          <>
                            <div className="flex gap-4">
                              <div>
                                <p className="text-xs text-gray-600">Bet</p>
                                <p className="text-sm font-bold text-gray-900">${prediction.betAmount.toFixed(0)}</p>
                              </div>
                              <div>
                                <p className="text-xs text-gray-600">Expected Win</p>
                                <p className="text-sm font-bold text-green-600">${prediction.expectedReturn.toFixed(0)}</p>
                              </div>
                            </div>
                            <button
                              onClick={() => setSelectedPrediction(prediction.id)}
                              className="btn btn-ghost text-sm py-2 px-4 touch-target flex items-center gap-1"
                            >
                              Details
                              <ChevronDown className="w-4 h-4" />
                            </button>
                          </>
                        ) : (
                          <div className="flex-1 flex items-center justify-between">
                            <p className="text-sm font-semibold text-gray-700">
                              ⏭️ Skip - Optimized strategy avoids {prediction.confidence} confidence
                            </p>
                            <button
                              onClick={() => setSelectedPrediction(prediction.id)}
                              className="btn btn-ghost text-sm py-2 px-4 touch-target flex items-center gap-1"
                            >
                              Why?
                              <ChevronDown className="w-4 h-4" />
                            </button>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Expanded Details (Progressive Disclosure) */}
                    <AnimatePresence>
                      {isExpanded && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.3 }}
                          className="mt-6 pt-6 border-t border-gray-200"
                        >
                          {/* Model Predictions */}
                          <div className="mb-6">
                            <h4 className="text-sm font-bold text-gray-900 uppercase mb-3">
                              Model Predictions Breakdown
                            </h4>
                            <div className="grid grid-cols-2 gap-3">
                              <div className="bg-white rounded-lg p-3">
                                <p className="text-xs text-gray-600">Elo</p>
                                <p className="text-lg font-bold text-gray-900">
                                  {prediction.modelPredictions.elo > 0 ? '+' : ''}{prediction.modelPredictions.elo.toFixed(1)}
                                </p>
                              </div>
                              <div className="bg-white rounded-lg p-3">
                                <p className="text-xs text-gray-600">Neural Network</p>
                                <p className="text-lg font-bold text-gray-900">
                                  {prediction.modelPredictions.neural > 0 ? '+' : ''}{prediction.modelPredictions.neural.toFixed(1)}
                                </p>
                              </div>
                              <div className="bg-white rounded-lg p-3">
                                <p className="text-xs text-gray-600">XGBoost</p>
                                <p className="text-lg font-bold text-gray-900">
                                  {prediction.modelPredictions.xgboost > 0 ? '+' : ''}{prediction.modelPredictions.xgboost.toFixed(1)}
                                </p>
                              </div>
                              <div className="bg-white rounded-lg p-3 border-2 border-primary">
                                <p className="text-xs text-primary font-semibold">Ensemble</p>
                                <p className="text-lg font-bold text-primary">
                                  {prediction.modelPredictions.ensemble > 0 ? '+' : ''}{prediction.modelPredictions.ensemble.toFixed(1)}
                                </p>
                              </div>
                            </div>
                          </div>

                          {/* Model Agreement */}
                          <div className="mb-6 bg-white rounded-lg p-4">
                            <div className="flex items-center justify-between">
                              <div>
                                <p className="text-xs text-gray-600 uppercase font-semibold">Model Agreement</p>
                                <p className="text-sm text-gray-700 mt-1">
                                  {prediction.confidence === 'LOW'
                                    ? 'Models disagree (ensemble strength)'
                                    : prediction.confidence === 'MEDIUM'
                                    ? 'Moderate model agreement'
                                    : 'Models strongly agree'}
                                </p>
                              </div>
                              <div className="text-2xl">
                                {prediction.confidence === 'LOW' ? '🎯' : prediction.confidence === 'MEDIUM' ? '⚖️' : '🤝'}
                              </div>
                            </div>
                          </div>

                          {/* Action Buttons */}
                          <div className="flex gap-3">
                            {prediction.betAmount > 0 ? (
                              <button className="btn btn-primary flex-1 touch-target">
                                <Target className="w-5 h-5 inline mr-2" />
                                Bet ${prediction.betAmount.toFixed(0)}
                              </button>
                            ) : (
                              <div className="flex-1 bg-gray-100 rounded-lg p-4 text-center">
                                <p className="text-sm font-semibold text-gray-700">
                                  ⏭️ Optimized strategy skips {prediction.confidence} confidence games
                                </p>
                                <p className="text-xs text-gray-600 mt-1">
                                  {prediction.confidence} confidence has {style.winRate} win rate (lower expected ROI)
                                </p>
                              </div>
                            )}
                            <button
                              onClick={() => setSelectedPrediction(null)}
                              className="btn btn-ghost touch-target flex items-center gap-1"
                            >
                              <ChevronUp className="w-4 h-4" />
                              Collapse
                            </button>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </motion.div>
              )
            })}
          </div>

          {/* Information Banner */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
            className="mt-12 bg-gradient-to-r from-primary to-primary-light rounded-xl p-6 text-white"
          >
            <div className="flex items-start gap-4">
              <div className="text-4xl">💡</div>
              <div className="flex-1">
                <h3 className="font-bold text-lg mb-2">Why LOW Confidence Performs Best</h3>
                <p className="text-white/90 text-sm leading-relaxed">
                  Our v2.0.0 model achieves its highest win rate (70.8%) on LOW confidence games where individual models disagree.
                  Model disagreement indicates complex matchups where the ensemble method excels, capturing nuances no single model sees.
                  This counterintuitive finding has been validated on 231 out-of-sample games from the 2025-26 season.
                </p>
                <div className="mt-4 flex flex-wrap gap-3">
                  <span className="badge bg-white/20 text-white border-white/30">
                    🟢 LOW: 70.8% win rate
                  </span>
                  <span className="badge bg-white/20 text-white border-white/30">
                    🟡 MEDIUM: 63.7% win rate
                  </span>
                  <span className="badge bg-white/20 text-white border-white/30">
                    🔴 HIGH: 64.2% win rate
                  </span>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  )
}
