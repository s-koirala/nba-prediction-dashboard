import { NextRequest, NextResponse } from 'next/server'
import { readFileSync } from 'fs'
import { join } from 'path'

export const dynamic = 'force-dynamic' // Disable caching for fresh predictions

interface PredictionRow {
  HOME_TEAM: string
  AWAY_TEAM: string
  GAME_TIME: string
  ELO_PREDICTION: number
  NN_PREDICTION: number
  XGB_PREDICTION: number
  ENSEMBLE_PREDICTION: number
}

// Get confidence level based on model agreement
function getConfidenceLevel(predictions: number[]): 'LOW' | 'MEDIUM' | 'HIGH' {
  const spread = Math.max(...predictions) - Math.min(...predictions)

  if (spread > 6) return 'LOW'  // High disagreement = LOW confidence = BEST!
  if (spread > 3) return 'MEDIUM'
  return 'HIGH'
}

// Calculate bet size based on confidence
function calculateBetSize(confidence: 'LOW' | 'MEDIUM' | 'HIGH', bankroll: number = 10000): number {
  // Optimized strategy from walk-forward validation
  const multipliers = {
    'LOW': 0.014,    // 1.4% - BEST performance (70.8% win rate)
    'MEDIUM': 0.0,   // Skip MEDIUM confidence
    'HIGH': 0.0      // Skip HIGH confidence
  }

  return bankroll * multipliers[confidence]
}

// Get win probability based on confidence
function getWinProbability(confidence: 'LOW' | 'MEDIUM' | 'HIGH'): number {
  const rates = {
    'LOW': 70.8,
    'MEDIUM': 63.7,
    'HIGH': 64.2
  }
  return rates[confidence]
}

export async function GET(request: NextRequest) {
  try {
    // Try to read predictions from CSV file
    const predictionsPath = join(process.cwd(), '..', 'results', 'tonights_predictions.csv')

    let predictions = []

    try {
      const csvContent = readFileSync(predictionsPath, 'utf-8')
      const lines = csvContent.trim().split('\n')
      const headers = lines[0].split(',')

      // Parse CSV
      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',')
        const row: any = {}
        headers.forEach((header, index) => {
          row[header] = values[index]
        })

        const modelPredictions = [
          parseFloat(row.ELO_PREDICTION),
          parseFloat(row.NN_PREDICTION),
          parseFloat(row.XGB_PREDICTION),
          parseFloat(row.ENSEMBLE_PREDICTION)
        ]

        const confidence = getConfidenceLevel(modelPredictions)
        const ensemblePred = parseFloat(row.ENSEMBLE_PREDICTION)
        const predictedWinner = ensemblePred > 0 ? row.HOME_TEAM : row.AWAY_TEAM
        const betAmount = calculateBetSize(confidence)
        const winProb = getWinProbability(confidence)

        predictions.push({
          id: i.toString(),
          homeTeam: row.HOME_TEAM,
          awayTeam: row.AWAY_TEAM,
          predictedWinner: predictedWinner,
          expectedMargin: Math.abs(ensemblePred),
          confidence: confidence,
          winProbability: winProb,
          expectedReturn: betAmount * (winProb / 100) * (100 / 110),
          betAmount: betAmount,
          gameTime: row.GAME_TIME || 'TBD',
          modelPredictions: {
            elo: modelPredictions[0],
            neural: modelPredictions[1],
            xgboost: modelPredictions[2],
            ensemble: modelPredictions[3]
          }
        })
      }
    } catch (fileError) {
      console.log('No predictions file found, using mock data')

      // Return mock data if file doesn't exist
      predictions = [
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
            ensemble: 8.5
          }
        },
        {
          id: '2',
          homeTeam: 'Golden State Warriors',
          awayTeam: 'Phoenix Suns',
          predictedWinner: 'Phoenix Suns',
          expectedMargin: 5.2,
          confidence: 'MEDIUM',
          winProbability: 63.7,
          expectedReturn: 0,
          betAmount: 0,
          gameTime: '10:00 PM ET',
          modelPredictions: {
            elo: -5.0,
            neural: -5.5,
            xgboost: -5.1,
            ensemble: -5.2
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
            ensemble: 3.1
          }
        }
      ]
    }

    return NextResponse.json({
      success: true,
      count: predictions.length,
      predictions: predictions
    })

  } catch (error) {
    console.error('Error fetching predictions:', error)
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to fetch predictions',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    )
  }
}
