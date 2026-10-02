"""compare_features.py : features de update_history.py vs training_matrix.pkl (feature_engineering.py)."""
import numpy as np
import pandas as pd

from update_history import FEATURES_19

a = pd.read_pickle(r"data\processed\nba_with_features.pkl")
t = pd.read_pickle(r"data\processed\training_matrix.pkl")
print("training_matrix:", t.shape)
print("colonnes:", list(t.columns))

if "game_id" in t.columns:
    key = ["game_id"]
elif {"date", "home_team", "away_team"} <= set(t.columns):
    key = ["date", "home_team", "away_team"]
else:
    raise SystemExit("[STOP] aucune cle de jointure dans training_matrix.pkl, colle-moi les colonnes ci-dessus")

miss = [c for c in FEATURES_19 if c not in t.columns]
if miss:
    raise SystemExit(f"[STOP] colonnes absentes de training_matrix.pkl: {miss}")

m = a[key + FEATURES_19].merge(t[key + FEATURES_19], on=key, suffixes=("_moi", "_fe"))
print("cle:", key, "| lignes jointes:", len(m), "| dans mon pkl:", len(a), "| dans la matrice:", len(t))

rows = []
for c in FEATURES_19:
    x, y = m[c + "_moi"].to_numpy(float), m[c + "_fe"].to_numpy(float)
    both = ~(np.isnan(x) | np.isnan(y))
    d = np.abs(x[both] - y[both])
    rows.append((c, float(d.max()) if len(d) else float("nan"), int((d > 1e-6).sum()),
                 int((np.isnan(x) != np.isnan(y)).sum())))
print(pd.DataFrame(rows, columns=["feature", "ecart_max", "n_ecarts>1e-6", "n_NaN_differents"]).to_string(index=False))
