"""check_pkl.py : affiche le schema reel du pkl (avant et apres update_history)."""
import pandas as pd

P = r"data\processed\nba_with_features.pkl"
d = pd.read_pickle(P)

print("shape:", d.shape)
print("colonnes:", list(d.columns))
print(d.dtypes.to_string())
print("dates:", d["date"].min(), "->", d["date"].max())
print("game_id:", repr(d["game_id"].iloc[0]), repr(d["game_id"].iloc[-1]), "unique:", d["game_id"].is_unique)
print("equipes:", len(set(d["home_team"]) | set(d["away_team"])))
for c in d.columns:
    if c.endswith(("tov", "tov_pct", "reb", "reb_pct", "pace")):
        print(c, "mediane:", round(float(d[c].median()), 4))
print(d.tail(2).T.to_string())
