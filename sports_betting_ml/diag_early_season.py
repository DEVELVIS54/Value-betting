"""diag_early_season.py : erreur et calibration selon le nombre de matchs joues dans la saison.

Mesure, sur les folds walk-forward 2018-19 a 2022-23 (AUCUN test, aucun modele modifie),
si le modele est moins bon en debut de saison (forme heritee de la saison precedente).
Configs : h2h et spreads actuels (mcw=10, decay=0.90), totals retenu (mcw=10, decay=0.90).

Lancer depuis la racine du projet : python diag_early_season.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from experiment_walkforward import FEATURE_COLUMNS, FOLDS, MJ, load_matrix, mt, slope

MARKETS = [("h2h", "classifier", "home_win", 10.0, 0.90),
           ("spreads", "regressor", "margin", 10.0, 0.90),
           ("totals", "regressor", "total_pts", 10.0, 0.90)]
BINS = [-1, 9, 19, 40, 82]
LABELS = ["0-9", "10-19", "20-40", "41-82"]


def fold_preds(df, kind, target, mcw, decay, s):
    feats = list(FEATURE_COLUMNS)
    train = df[(df["date"] >= mt.TRAIN_START) & (df["season"] < s)]
    valid = df[df["season"] == s].copy()
    assert valid["date"].max() <= mt.TRAIN_END, "fold dans la periode de test : interdit"
    inner_tr, inner_va = train[train["season"] < s - 1], train[train["season"] == s - 1]
    probe = mt.make_model(kind, mt.MAX_ESTIMATORS, mt.EARLY_STOPPING_ROUNDS)
    probe.set_params(min_child_weight=mcw)
    probe.fit(inner_tr[feats], inner_tr[target], sample_weight=mt.recency_weights(inner_tr["season"], decay),
              eval_set=[(inner_va[feats], inner_va[target])], verbose=False)
    best_n = max(int(probe.best_iteration) + 1, mt.MIN_FINAL_ESTIMATORS)
    model = mt.make_model(kind, best_n)
    model.set_params(min_child_weight=mcw)
    model.fit(train[feats], train[target], sample_weight=mt.recency_weights(train["season"], decay), verbose=False)
    if kind == "classifier":
        valid["pred"] = np.clip(model.predict_proba(valid[feats])[:, 1], 1e-4, 1 - 1e-4)
    else:
        valid["pred"] = model.predict(valid[feats])
    return valid


def main():
    df = load_matrix()
    print(f"matrice: {len(df)} matchs | folds {FOLDS} | test non utilise")
    for market, kind, target, mcw, decay in MARKETS:
        v = pd.concat([fold_preds(df, kind, target, mcw, decay, s) for s in FOLDS], ignore_index=True)
        v["bucket"] = pd.cut(v[MJ], bins=BINS, labels=LABELS)
        rows = []
        for b, g in list(v.groupby("bucket", observed=True)) + [("TOUS", v)]:
            y, p = g[target].to_numpy(float), g["pred"].to_numpy(float)
            if kind == "classifier":
                rows.append(dict(matchs_joues=b, n=len(g), logloss=log_loss(y, p), precision=float(np.mean((p >= .5) == (y == 1))),
                                 p_moyen=float(p.mean()), taux_reel=float(y.mean()),
                                 pente=slope(p, y) if len(g) >= 300 else float("nan")))
            else:
                rows.append(dict(matchs_joues=b, n=len(g), MAE=float(np.mean(np.abs(y - p))), biais=float(np.mean(p - y)),
                                 sigma=float(np.std(y - p))))
        print(f"\n=== {market.upper()} ===")
        print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\nLecture : tranche '0-9' nettement pire que 'TOUS' (logloss > +0.03, MAE > +0.5) ou pente H2H < 0.8 "
          "=> le modele est trop confiant en debut de saison : reduire W_MODEL a l'inference.")


if __name__ == "__main__":
    main()
