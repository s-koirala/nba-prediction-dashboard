from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime

today = datetime.now().strftime('%Y-%m-%d')
scoreboard = scoreboardv2.ScoreboardV2(game_date=today)
games = scoreboard.get_data_frames()[0]
print("Columns:", list(games.columns))
print("\nFirst game:")
print(games.iloc[0] if len(games) > 0 else "No games")
