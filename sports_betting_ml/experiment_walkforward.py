"""experiment_walkforward.py : compare des configurations XGBoost en walk-forward.

- Periode d'entrainement uniquement (saisons 2018-19 a 2022-23). AUCUNE donnee de test (>= 2023-10-01).
- Aucun modele, aucun pkl n'est modifie.
- Procedure par fold (saison s), identique a model_train.py :
    train = saisons < s | early stopping sur la saison s-1 | re-entrainement sur train | score sur la saison s.

Lancer depuis la racine du projet, venv actif :
    python experiment_walkforward.py            # tout (3 marches)
    python experiment_walkforward.py h2h        # un seul marche
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import model_train as mt  # noqa: E402
from feature_engineering import FEATURE_COLUMNS, PKL_PATH, load_training_matrix  # noqa: E402

MJ = "matchs_joues_cette_saison"
FOLDS = [2018, 2019, 2020, 2021, 2022]          # saison = annee de debut (2022 = 2022-23)

# (libelle, min_child_weight, decroissance, + matchs_joues)
# h2h : h = p(1-p) <= 0.25  => mcw=10 ~ 40 matchs/feuille ; mcw=2.5 ~ 10 matchs (poids = 1)
# regression : h = 1        => mcw=N ~ N matchs/feuille (poids = 1)
CONFIGS = {
    "h2h": ("classifier", "home_win", [
        ("actuel  mcw=10  (~40 matchs)", 10.0, 0.90, False),
        ("mcw=5   (~20 matchs)", 5.0, 0.90, False),
        ("mcw=2.5 (~10 matchs)", 2.5, 0.90, False),
        ("mcw=1   (~4 matchs)", 1.0, 0.90, False),
        ("mcw=2.5 + matchs_joues", 2.5, 0.90, True),
        ("mcw=10  + matchs_joues", 10.0, 0.90, True),
    ]),
    "spreads": ("regressor", "margin", [
        ("actuel  mcw=10", 10.0, 0.90, False),
        ("mcw=5", 5.0, 0.90, False),
        ("mcw=20", 20.0, 0.90, False),
        ("mcw=10 + matchs_joues", 10.0, 0.90, True),
    ]),
    "totals": ("regressor", "total_pts", [
        ("actuel  mcw=10 decay=0.75", 10.0, 0.75, False),
        ("mcw=10 decay=0.90", 10.0, 0.90, False),
        ("mcw=5  decay=0.90", 5.0, 0.90, False),
        ("mcw=5  decay=0.75", 5.0, 0.75, False),
        ("mcw=10 decay=0.90 + matchs_joues", 10.0, 0.90, True),
    ]),
}


def load_matrix() -> pd.DataFrame:
    df = load_training_matrix()
    raw = pd.read_pickle(PKL_PATH)
    if MJ not in raw.columns:
        sys.exit(f"[STOP] {MJ} absente du pkl : lancer d'abord python update_history.py")
    mj = raw[["date", "home_team", "away_team", MJ]].copy()
    mj["date"] = pd.to_datetime(mj["date"]).astype("datetime64[ns]")
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    out = df.merge(mj, on=["date", "home_team", "away_team"], how="left", validate="one_to_one")
    if out[MJ].isna().any():
        sys.exit(f"[STOP] {int(out[MJ].isna().sum())} matchs sans {MJ} apres jointure")
    return out


def slope(p, y) -> float:
    z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))).reshape(-1, 1)
    return float(LogisticRegression(C=1e6).fit(z, y).coef_[0][0])


def run_fold(df, kind, target, mcw, decay, use_mj, s):
    feats = list(FEATURE_COLUMNS) + ([MJ] if use_mj else [])
    train = df[(df["date"] >= mt.TRAIN_START) & (df["season"] < s)]
    valid = df[df["season"] == s]
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

    y = valid[target].to_numpy(dtype=float)
    if kind == "classifier":
        p = np.clip(model.predict_proba(valid[feats])[:, 1], 1e-4, 1 - 1e-4)
        return dict(n=best_n, main=log_loss(y, p), acc=accuracy_score(y, p >= 0.5), p=p, y=y)
    pred = model.predict(valid[feats])
    direction = float(np.mean((pred > 0) == (y > 0))) if target == "margin" else float("nan")
    return dict(n=best_n, main=float(np.mean(np.abs(y - pred))), acc=direction)


def run_market(df, market):
    kind, target, configs = CONFIGS[market]
    main_name = "logloss" if kind == "classifier" else "MAE"
    print(f"\n=== {market.upper()} | walk-forward {FOLDS[0]}-{FOLDS[-1]} | metrique principale : {main_name} ===")
    base = None
    rows = []
    for label, mcw, decay, use_mj in configs:
        t0 = time.time()
        res = [run_fold(df, kind, target, mcw, decay, use_mj, s) for s in FOLDS]
        mains = np.array([r["main"] for r in res])
        row = dict(label=label, trees=float(np.mean([r["n"] for r in res])),
                   main=float(mains.mean()), acc=float(np.mean([r["acc"] for r in res])), mains=mains)
        if kind == "classifier":
            row["slope"] = slope(np.concatenate([r["p"] for r in res]), np.concatenate([r["y"] for r in res]))
        if base is None:
            base = mains
        d = mains - base
        row["delta"] = float(d.mean())
        row["se"] = float(d.std(ddof=1) / np.sqrt(len(d))) if np.any(d != 0) else 0.0
        row["better"] = int((d < 0).sum())
        row["sec"] = time.time() - t0
        rows.append(row)
        extra = f" | pente {row['slope']:.2f}" if kind == "classifier" else ""
        acc_txt = "-" if np.isnan(row["acc"]) else f"{row['acc']:.4f}"
        print(f"{label:36s} arbres={row['trees']:5.0f} | {main_name}={row['main']:.4f} | "
              f"acc/direction={acc_txt}{extra} | delta={row['delta']:+.4f} (+/-{row['se']:.4f}) "
              f"| meilleur sur {row['better']}/{len(FOLDS)} folds | {row['sec']:.0f}s", flush=True)
    return rows


def main():
    wanted = [a for a in sys.argv[1:] if a in CONFIGS] or list(CONFIGS)
    df = load_matrix()
    print(f"matrice: {len(df)} matchs | folds {FOLDS} | test (>= 2023-10-01) non utilise")
    for market in wanted:
        run_market(df, market)
    print("\nLecture : un changement n'est retenu que si delta < 0, |delta| > 2x l'erreur (+/-) et meilleur sur >= 4/5 folds.")


if __name__ == "__main__":
    main()
