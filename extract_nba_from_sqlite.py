import pandas as pd
import sqlite3
from pathlib import Path

print("📦 Extraction et alignement des données NBA (2014-2026)...")

DB_PATH = "sports-betting-ml/data/raw/nba.sqlite"
OUTPUT_CSV = "sports-betting-ml/data/raw/nba_games.csv"

Path("sports-betting-ml/data/raw").mkdir(parents=True, exist_ok=True)

try:
    conn = sqlite3.connect(DB_PATH)
    
    query = """
    SELECT 
        game_date AS date,
        team_name_home AS home_team,
        team_name_away AS away_team,
        pts_home AS home_pts,
        pts_away AS away_pts
    FROM game
    WHERE season_id >= 22014 AND season_id <= 22026
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"   -> {len(df)} matchs extraits")
    
    df['home_pts'] = pd.to_numeric(df['home_pts'], errors='coerce')
    df['away_pts'] = pd.to_numeric(df['away_pts'], errors='coerce')
    df = df.dropna(subset=['home_pts', 'away_pts'])
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date').drop_duplicates()
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ {len(df)} matchs NBA réels sauvegardés dans {OUTPUT_CSV}")

except sqlite3.OperationalError as e:
    print(f"❌ Erreur base de données : {e}")
except FileNotFoundError as e:
    print(f"❌ Fichier introuvable : {e}")
except Exception as e:
    print(f"❌ Erreur : {e}")
