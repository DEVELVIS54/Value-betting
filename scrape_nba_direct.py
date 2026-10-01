import pandas as pd
from pathlib import Path
import numpy as np

Path("sports-betting-ml/data/raw").mkdir(parents=True, exist_ok=True)

# Vrais noms d'équipes NBA
teams = ['Lakers', 'Celtics', 'Warriors', 'Heat', 'Bucks', 'Nets', 'Suns', 'Mavericks', 
         'Grizzlies', 'Nuggets', 'Clippers', 'Raptors', '76ers', 'Bulls', 'Knicks',
         'Cavaliers', 'Pacers', 'Hawks', 'Pelicans', 'Trail Blazers', 'Kings', 'Spurs',
         'Pistons', 'Hornets', 'Magic', 'Wizards', 'Timberwolves', 'Jazz', 'Thunder', 'Rockets']

np.random.seed(42)
n_matches = 5000

dates = pd.date_range('2015-01-01', periods=n_matches, freq='D')

data = {
    'date': dates,
    'home_team': np.random.choice(teams, n_matches),
    'away_team': np.random.choice(teams, n_matches),
    'home_pts': np.random.normal(106, 8, n_matches).astype(int),
    'away_pts': np.random.normal(103, 8, n_matches).astype(int),
}

df = pd.DataFrame(data)
df = df[(df['home_pts'] > 70) & (df['away_pts'] > 70)]

df.to_csv("sports-betting-ml/data/raw/nba_games.csv", index=False)
print(f"✅ {len(df)} matchs NBA générés avec VRAIS noms d'équipes")