"""diag_totals_bias.py : biais du modele Totals par tranche de matchs joues, selon 4 variantes.

Folds walk-forward 2018-19 a 2022-23 (AUCUN test, aucun modele modifie).
Variantes : decroissance des poids {0.75, 0.90} x feature matchs_joues_cette_saison {non, oui}.
biais = prediction - realite (> 0 : le modele surestime le total => faux signaux Over).

Regle fixee AVANT de voir les resultats : on retient la variante dont le biais reste le plus proche de 0
dans les 4 tranches (ecart max |biais|), a MAE comparable (+/- 0.10).

Lancer depuis la racine du projet : python diag_totals_bias.py
"""
import numpy as np
import pandas as pd

from experiment_walkforward import FEATURE_COLUMNS, FOLDS, MJ, load_matrix, mt

TARGET, MCW = "total_pts", 10.0
BINS = [-1, 9, 19, 40, 82]
LABELS = ["0-9", "10-19", "20-40", "41-82"]
VARIANTS = [("decay 0.75", 0.75, False), ("decay 0.90", 0.90, False),
            ("decay 0.75 + matchs_joues", 0.75, True), ("decay 0.90 + matchs_joues", 0.90, True)]


def fold_preds(df, decay, use_mj, s):
    feats = list(FEATURE_COLUMNS) + ([MJ] if use_mj else [])
    train = df[(df["date"] >= mt.TRAIN_START) & (df["season"] < s)]
    valid = df[df["season"] == s].copy()
    assert valid["date"].max() <= mt.TRAIN_END, "fold dans la periode de test : interdit"
    inner_tr, inner_va = train[train["season"] < s - 1], train[train["season"] == s - 1]
    probe = mt.make_model("regressor", mt.MAX_ESTIMATORS, mt.EARLY_STOPPING_ROUNDS)
    probe.set_params(min_child_weight=MCW)
    probe.fit(inner_tr[feats], inner_tr[TARGET], sample_weight=mt.recency_weights(inner_tr["season"], decay),
              eval_set=[(inner_va[feats], inner_va[TARGET])], verbose=False)
    best_n = max(int(probe.best_iteration) + 1, mt.MIN_FINAL_ESTIMATORS)
    model = mt.make_model("regressor", best_n)
    model.set_params(min_child_weight=MCW)
    model.fit(train[feats], train[TARGET], sample_weight=mt.recency_weights(train["season"], decay), verbose=False)
    valid["pred"] = model.predict(valid[feats])
    return valid


def main():
    df = load_matrix()
    print(f"matrice: {len(df)} matchs | folds {FOLDS} | test non utilise | biais = prediction - realite")
    rows = []
    for label, decay, use_mj in VARIANTS:
        v = pd.concat([fold_preds(df, decay, use_mj, s) for s in FOLDS], ignore_index=True)
        v["bucket"] = pd.cut(v[MJ], bins=BINS, labels=LABELS)
        err = v["pred"] - v[TARGET]
        row = {"variante": label, "MAE": float(err.abs().mean())}
        for b in LABELS:
            row[f"biais {b}"] = float(err[v["bucket"] == b].mean())
        row["biais TOUS"] = float(err.mean())
        row["ecart max |biais|"] = max(abs(row[f"biais {b}"]) for b in LABELS)
        rows.append(row)
        print(f"  {label} : fait", flush=True)
    print()
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\nEffet sur un signal : un biais b decale la moyenne ancree de 0.2*b points (W_MODEL totals = 0.2) ;"
          "\n1 point de decalage de la moyenne = ~4% d'EV (sigma ~19, cote ~1.91), donc ~0.8% d'EV par point de biais.")


if __name__ == "__main__":
    main()
