"""diag_test_totals.py : CONFIRMATION sur la periode de test (2023-10 -> 2026-04) du biais Totals en debut de saison.

Hypothese formee en walk-forward (train uniquement) : le modele Totals surestime les totaux sur les 10 premiers matchs.
Ce script ne regle rien : il confirme ou infirme l'hypothese hors echantillon, pour 2 variantes
(v1 = 19 colonnes, v2 = 19 colonnes + matchs_joues_cette_saison), meme decroissance 0.75, memes reglages.
Entrainement sur le train seul (jusqu'a 2023-06), comme model_train.py. Aucun fichier modifie.

Lancer depuis la racine du projet : python diag_test_totals.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import model_train as mt  # noqa: E402
from feature_engineering import FEATURE_COLUMNS, GAMES_PLAYED_COL, TOTALS_FEATURE_COLUMNS  # noqa: E402

TARGET, DECAY = "total_pts", 0.75
BINS = [-1, 9, 19, 40, 82]
LABELS = ["0-9", "10-19", "20-40", "41-82"]


def fit_predict(train, test, feats):
    is_valid = (train["date"] >= mt.VALID_START).to_numpy()
    w = mt.recency_weights(train["season"], DECAY)
    probe = mt.make_model("regressor", mt.MAX_ESTIMATORS, mt.EARLY_STOPPING_ROUNDS)
    probe.fit(train.loc[~is_valid, feats], train.loc[~is_valid, TARGET], sample_weight=w[~is_valid],
              eval_set=[(train.loc[is_valid, feats], train.loc[is_valid, TARGET])], verbose=False)
    best_n = max(int(probe.best_iteration) + 1, mt.MIN_FINAL_ESTIMATORS)
    model = mt.make_model("regressor", best_n)
    model.fit(train[feats], train[TARGET], sample_weight=w, verbose=False)
    return model.predict(test[feats]), best_n


def main():
    df = mt.load_training_matrix()
    train, test = mt.split_data(df)
    y = test[TARGET].to_numpy(float)
    bucket = pd.cut(test[GAMES_PLAYED_COL], bins=BINS, labels=LABELS)
    print(f"train {len(train)} | test {len(test)} ({test['date'].min().date()} -> {test['date'].max().date()})")

    preds, rows = {}, []
    for label, feats in (("v1 (19 colonnes)", list(FEATURE_COLUMNS)), ("v2 (+ matchs_joues)", list(TOTALS_FEATURE_COLUMNS))):
        p, n_trees = fit_predict(train, test, feats)
        preds[label] = p
        err = p - y
        row = {"variante": label, "arbres": n_trees, "MAE": float(np.abs(err).mean()), "corr": float(np.corrcoef(p, y)[0, 1])}
        for b in LABELS:
            m = (bucket == b).to_numpy()
            row[f"biais {b}"] = float(err[m].mean())
        row["biais TOUS"] = float(err.mean())
        rows.append(row)
    print()
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\nn par tranche :", {b: int((bucket == b).sum()) for b in LABELS})
    for b in LABELS:
        m = (bucket == b).to_numpy()
        print(f"  erreur-type du biais, tranche {b} : +/-{np.std(preds['v1 (19 colonnes)'][m] - y[m], ddof=1) / np.sqrt(m.sum()):.2f}")

    a1, a2 = np.abs(preds["v1 (19 colonnes)"] - y), np.abs(preds["v2 (+ matchs_joues)"] - y)
    d = a2 - a1
    print(f"\nMAE v2 - v1 (apparie, tous matchs) : {d.mean():+.3f} +/- {d.std(ddof=1) / np.sqrt(len(d)):.3f}")
    for b in LABELS:
        m = (bucket == b).to_numpy()
        print(f"  tranche {b:6s} : {d[m].mean():+.3f} +/- {d[m].std(ddof=1) / np.sqrt(m.sum()):.3f}")


if __name__ == "__main__":
    main()
