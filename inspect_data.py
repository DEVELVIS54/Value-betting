import pandas as pd

df = pd.read_csv('sports-betting-ml/data/raw/nba_games.csv')
print('📊 INSPECTION DES DONNÉES NBA')
print(f'Matchs: {len(df)}')
print(f'Colonnes: {list(df.columns)}')
print(f'Manquantes:\n{df.isnull().sum()}')
print(f'Stats:\n{df[["home_pts", "away_pts"]].describe()}')
