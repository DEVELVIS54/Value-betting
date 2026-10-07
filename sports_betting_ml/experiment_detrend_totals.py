"""experiment_detrend_totals.py : Totals sur cible detendancee, evalue en walk-forward (AUCUN test, aucun modele modifie).

Idee (ISLP 10.5.2, series non stationnaires) : le niveau de scoring de la ligue monte plus vite que le modele ne s'adapte
(biais ~ -3 pts). On predit donc  total - niveau(t)  puis on rajoute niveau(t), ou
niveau(t) = moyenne de total_pts des N matchs de la ligue strictement ANTERIEURS a la date du match.

Protocole fixe AVANT les resultats (variante principale : N = 300) :
  retenue si (1) ecart max |biais| par tranche de matchs joues <= 50% de la reference ET (2) MAE <= reference + 0.10.
  Les autres N (150, 600) sont des controles de sensibilite, pas des candidats.
Reference = Totals actuel (cible brute, 19 colonnes, decroissance 0.75, min_child_weight 10).
Les ~100 premiers matchs (pas assez d'historique pour le niveau) sont exclus de TOUTES les variantes.

Lancer depuis la racine du projet : python experiment_detrend_totals.py
"""
import numpy as np
import pandas as pd

from experiment_walkforward import FEATURE_COLUMNS, FOLDS, MJ, load_matrix, mt

TARGET, DECAY, MCW = "total_pts", 0.75, 10.0
MIN_PRIOR = 100
BINS = [-1, 9, 19, 40, 82]
LABELS = ["0-9", "10-19", "20-40", "41-82"]
VARIANTS = [("reference (cible brute)", None), ("detendancee N=150", 150),
            ("detendancee N=300 (principale)", 300), ("detendancee N=600", 600)]


def trailing_level(df: pd.DataFrame, n: int) -> np.ndarray:
    """Moyenne de total_pts des n matchs precedents (date STRICTEMENT anterieure, toutes equipes). NaN si < MIN_PRIOR."""
    tot = df[TARGET].to_numpy(float)
    cs = np.concatenate([[0.0], np.cumsum(tot)])
    first = pd.Series(np.arange(len(df))).groupby(df["date"].to_numpy()).transform("min").to_numpy()
    lo = np.maximum(first - n, 0)
    level = (cs[first] - cs[lo]) / np.maximum(first - lo, 1)
    return np.where(first >= MIN_PRIOR, level, np.nan)


def fold_preds(df, n, s):
    feats = list(FEATURE_COLUMNS)
    level = df["_lvl"].to_numpy() if n is None else df[f"_lvl{n}"].to_numpy()
    shift = np.zeros(len(df)) if n is None else level
    y_adj = df[TARGET].to_numpy(float) - shift
    is_tr = ((df["date"] >= mt.TRAIN_START) & (df["season"] < s)).to_numpy()
    is_va = (df["season"] == s).to_numpy()
    assert df.loc[is_va, "date"].max() <= mt.TRAIN_END, "fold dans la periode de test : interdit"
    inner_tr, inner_va = is_tr & (df["season"] < s - 1).to_numpy(), is_tr & (df["season"] == s - 1).to_numpy()

    w_in = mt.recency_weights(df.loc[inner_tr, "season"], DECAY)
    probe = mt.make_model("regressor", mt.MAX_ESTIMATORS, mt.EARLY_STOPPING_ROUNDS)
    probe.set_params(min_child_weight=MCW)
    probe.fit(df.loc[inner_tr, feats], y_adj[inner_tr], sample_weight=w_in,
              eval_set=[(df.loc[inner_va, feats], y_adj[inner_va])], verbose=False)
    best_n = max(int(probe.best_iteration) + 1, mt.MIN_FINAL_ESTIMATORS)
    model = mt.make_model("regressor", best_n)
    model.set_params(min_child_weight=MCW)
    model.fit(df.loc[is_tr, feats], y_adj[is_tr], sample_weight=mt.recency_weights(df.loc[is_tr, "season"], DECAY), verbose=False)

    out = df.loc[is_va, ["date", "season", MJ, TARGET]].copy()
    out["pred"] = model.predict(df.loc[is_va, feats]) + shift[is_va]      # on rajoute le niveau : retour a l'echelle des totaux
    return out


def main():
    df = load_matrix().sort_values("date", kind="mergesort").reset_index(drop=True)
    for n in (150, 300, 600):
        df[f"_lvl{n}"] = trailing_level(df, n)
    df["_lvl"] = 0.0
    df = df[df["_lvl300"].notna() & df["_lvl600"].notna() & df["_lvl150"].notna()].reset_index(drop=True)
    print(f"matrice: {len(df)} matchs (apres exclusion des {MIN_PRIOR} premiers) | folds {FOLDS} | test non utilise")

    rows, preds = [], {}
    for label, n in VARIANTS:
        v = pd.concat([fold_preds(df, n, s) for s in FOLDS], ignore_index=True)
        preds[label] = v
        err = v["pred"] - v[TARGET]
        bucket = pd.cut(v[MJ], bins=BINS, labels=LABELS)
        row = {"variante": label, "MAE": float(err.abs().mean())}
        for b in LABELS:
            row[f"biais {b}"] = float(err[bucket == b].mean())
        row["biais TOUS"] = float(err.mean())
        row["ecart max |biais|"] = max(abs(row[f"biais {b}"]) for b in LABELS)
        rows.append(row)
        print(f"  {label} : fait", flush=True)

    print()
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    ref = np.abs(preds[VARIANTS[0][0]]["pred"] - preds[VARIANTS[0][0]][TARGET]).to_numpy()
    print("\nMAE - reference (apparie, tous matchs) :")
    for label, _ in VARIANTS[1:]:
        v = preds[label]
        d = np.abs(v["pred"] - v[TARGET]).to_numpy() - ref
        print(f"  {label:32s} {d.mean():+.3f} +/- {d.std(ddof=1) / np.sqrt(len(d)):.3f}")

    base, main_row = rows[0], rows[2]
    ok_bias = main_row["ecart max |biais|"] <= 0.5 * base["ecart max |biais|"]
    ok_mae = main_row["MAE"] <= base["MAE"] + 0.10
    print(f"\nProtocole (N=300) : ecart max |biais| {main_row['ecart max |biais|']:.2f} vs {base['ecart max |biais|']:.2f} "
          f"(<= 50% : {'OUI' if ok_bias else 'NON'}) | MAE {main_row['MAE']:.2f} vs {base['MAE']:.2f} "
          f"(<= +0.10 : {'OUI' if ok_mae else 'NON'})")
    print("=> " + ("CANDIDAT : confirmer UNE fois sur le test 2023-26." if ok_bias and ok_mae
                   else "NON RETENU : garder le Totals actuel + recentrage live."))


if __name__ == "__main__":
    main()
