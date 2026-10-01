# sports_betting_ml/src/model_train.py
"""
Entraînement NBA — 3 modèles XGBoost, séparation temporelle stricte.

- h2h     : classificateur (victoire domicile) + calibration de Platt optionnelle
- spreads : régression de la marge (pts domicile - pts extérieur)
- totals  : régression du total de points (pondération forte des saisons récentes)

Validation : train oct 2014 -> juin 2023 | test oct 2023 -> juin 2026 (jamais vu)
Portes qualité (exception si non respectées, aucun fichier sauvegardé) :
  h2h accuracy >= 54% | spreads précision de direction >= 54% | totals MAE < baseline
Production : après validation, chaque modèle est ré-entraîné sur TOUTES les données
jusqu'en avril 2026 (nombre d'arbres x1.15).

Calibration de Platt (H2H uniquement) :
  - ajustée sur des prédictions hors échantillon par saison (fenêtre expansive), JAMAIS sur le test
  - évaluation propre : calibrateur ajusté sur les saisons <= fin du train, mesuré sur le test
  - activée seulement si le logloss du test baisse d'au moins PLATT_MIN_GAIN
  - version de production : ajustée sur toutes les saisons hors échantillon disponibles
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from xgboost import XGBClassifier, XGBRegressor

try:
    from feature_engineering import FEATURE_COLUMNS, load_training_matrix
except ImportError:  # exécution depuis la racine du projet
    from src.feature_engineering import FEATURE_COLUMNS, load_training_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_START = pd.Timestamp("2014-10-01")
TRAIN_END = pd.Timestamp("2023-06-30")
TEST_START = pd.Timestamp("2023-10-01")
TEST_END = pd.Timestamp("2026-06-30")
VALID_START = pd.Timestamp("2022-10-01")  # dernière saison du train = validation early stopping

MIN_ACCURACY = 0.54
RANDOM_STATE = 42
MAX_ESTIMATORS = 1500
EARLY_STOPPING_ROUNDS = 60
MIN_FINAL_ESTIMATORS = 60
PROD_TREE_FACTOR = 1.15

PLATT_FIRST_SEASON = 2018   # première saison validée hors échantillon pour le calibrateur
PLATT_TREES = 150           # arbres des modèles hors échantillon (léger : ~2 s chacun)
PLATT_MIN_GAIN = 0.002      # gain de logloss minimal sur le test pour activer le calibrateur

# marché -> (type, colonne cible, décroissance par saison d'ancienneté)
MARKETS: Dict[str, Tuple[str, str, float]] = {
    "h2h": ("classifier", "home_win", 0.90),
    "spreads": ("regressor", "margin", 0.90),
    "totals": ("regressor", "total_pts", 0.75),
}


class ModelQualityError(RuntimeError):
    """Qualité de test insuffisante."""


def make_model(kind: str, n_estimators: int, early_stopping_rounds: int = None):
    common = dict(
        max_depth=3,
        learning_rate=0.03,
        n_estimators=n_estimators,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        gamma=1.0,
        reg_lambda=5.0,
        reg_alpha=0.5,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        early_stopping_rounds=early_stopping_rounds,
    )
    if kind == "classifier":
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    return XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common)


def recency_weights(seasons: pd.Series, decay: float) -> np.ndarray:
    last = seasons.max()
    return (decay ** (last - seasons)).to_numpy(dtype=float)


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].copy()
    test = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)].copy()
    if train.empty or test.empty:
        raise ValueError(f"Split vide : train={len(train)}, test={len(test)}. Vérifie les dates.")
    if train["date"].max() >= test["date"].min():
        raise AssertionError("Chevauchement train/test : data leakage temporel.")
    return train, test


# --------------------------------------------------------------------------- #
# Calibration de Platt (H2H)
# --------------------------------------------------------------------------- #
def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def apply_platt(p, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * _logit(p) + b)))


def fit_platt(pairs) -> Tuple[float, float]:
    """pairs : liste de (logits, y). Retourne (pente a, ordonnée b)."""
    z = np.concatenate([p[0] for p in pairs]).reshape(-1, 1)
    y = np.concatenate([p[1] for p in pairs])
    lr = LogisticRegression(C=1e6).fit(z, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def oof_h2h(everything: pd.DataFrame) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Prédictions hors échantillon par saison (fenêtre expansive) : {saison: (logits, y)}."""
    decay = MARKETS["h2h"][2]
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for s in range(PLATT_FIRST_SEASON, int(everything["season"].max()) + 1):
        tr = everything[everything["season"] < s]
        va = everything[everything["season"] == s]
        if tr.empty or va.empty:
            continue
        m = make_model("classifier", PLATT_TREES)
        m.fit(tr[FEATURE_COLUMNS], tr["home_win"], sample_weight=recency_weights(tr["season"], decay), verbose=False)
        out[s] = (_logit(m.predict_proba(va[FEATURE_COLUMNS])[:, 1]), va["home_win"].to_numpy())
    return out


# --------------------------------------------------------------------------- #
# Évaluation
# --------------------------------------------------------------------------- #
def evaluate(market: str, kind: str, model, Xte, yte, ytr, wtr) -> Dict[str, Any]:
    if kind == "classifier":
        proba = np.clip(model.predict_proba(Xte)[:, 1], 1e-4, 1 - 1e-4)
        pred = (proba >= 0.5).astype(int)
        rate = float(yte.mean())
        return {
            "n_test": int(len(yte)),
            "accuracy": float(accuracy_score(yte, pred)),
            "log_loss": float(log_loss(yte, proba)),
            "brier": float(brier_score_loss(yte, proba)),
            "baseline_accuracy": float(max(rate, 1 - rate)),
            "baseline_log_loss": float(log_loss(yte, np.full(len(yte), np.clip(rate, 1e-4, 1 - 1e-4)))),
        }
    pred = model.predict(Xte)
    resid = yte.to_numpy(dtype=float) - pred
    base = float(np.average(ytr.to_numpy(dtype=float), weights=wtr))
    m = {
        "n_test": int(len(yte)),
        "mae": float(np.mean(np.abs(resid))),
        "baseline_mae": float(np.mean(np.abs(yte.to_numpy(dtype=float) - base))),
        "bias": float(np.mean(pred - yte.to_numpy(dtype=float))),
        "sigma": float(np.std(resid)),
        "corr": float(np.corrcoef(pred, yte)[0, 1]),
    }
    if market == "spreads":
        m["direction_accuracy"] = float(np.mean((pred > 0) == (yte.to_numpy() > 0)))
    return m


def train_market(market: str, kind: str, target: str, decay: float,
                 train: pd.DataFrame, test: pd.DataFrame, everything: pd.DataFrame) -> Dict[str, Any]:
    Xtr, ytr = train[FEATURE_COLUMNS], train[target]
    Xte, yte = test[FEATURE_COLUMNS], test[target]
    wtr = recency_weights(train["season"], decay)

    # 1) Early stopping sur la dernière saison du train
    is_valid = (train["date"] >= VALID_START).to_numpy()
    if is_valid.sum() < 100 or (~is_valid).sum() < 500:
        raise ValueError(f"[{market}] Données insuffisantes pour la validation temporelle.")
    probe = make_model(kind, MAX_ESTIMATORS, EARLY_STOPPING_ROUNDS)
    probe.fit(
        Xtr[~is_valid], ytr[~is_valid],
        sample_weight=wtr[~is_valid],
        eval_set=[(Xtr[is_valid], ytr[is_valid])],
        verbose=False,
    )
    best_n = max(int(getattr(probe, "best_iteration", MIN_FINAL_ESTIMATORS)) + 1, MIN_FINAL_ESTIMATORS)

    # 2) Modèle de validation (train seul) -> métriques hors échantillon
    model = make_model(kind, best_n)
    model.fit(Xtr, ytr, sample_weight=wtr, verbose=False)
    metrics = evaluate(market, kind, model, Xte, yte, ytr, wtr)
    metrics.update({"n_train": int(len(train)), "n_estimators_validation": best_n})

    # 2b) Calibration de Platt (H2H) : évaluation propre puis décision
    platt = None
    if kind == "classifier":
        oof = oof_h2h(everything)
        last_train_season = int(train["season"].max())
        eval_pairs = [v for s, v in oof.items() if s <= last_train_season]
        if eval_pairs:
            a_e, b_e = fit_platt(eval_pairs)
            p_raw = np.clip(model.predict_proba(Xte)[:, 1], 1e-4, 1 - 1e-4)
            ll_cal = float(log_loss(yte, np.clip(apply_platt(p_raw, a_e, b_e), 1e-4, 1 - 1e-4)))
            slope_test = float(LogisticRegression(C=1e6).fit(_logit(p_raw).reshape(-1, 1), yte).coef_[0][0])
            gain = metrics["log_loss"] - ll_cal
            metrics.update({
                "test_slope": slope_test,
                "platt_eval_params": [a_e, b_e],
                "platt_logloss": ll_cal,
                "platt_gain": float(gain),
            })
            if gain >= PLATT_MIN_GAIN:
                platt = fit_platt(list(oof.values()))  # version de production : toutes les saisons OOF
        metrics["platt_active"] = platt is not None

    # 3) Modèle de production : toutes les données jusqu'à la fin de la période de test
    prod = make_model(kind, int(best_n * PROD_TREE_FACTOR))
    prod.fit(
        everything[FEATURE_COLUMNS], everything[target],
        sample_weight=recency_weights(everything["season"], decay),
        verbose=False,
    )
    metrics["n_production"] = int(len(everything))

    return {
        "model": prod,
        "kind": kind,
        "sigma": metrics.get("sigma"),
        "platt": platt,
        "metrics": metrics,
    }


def quality_failure(market: str, m: Dict[str, Any]) -> str:
    if market == "h2h" and m["accuracy"] < MIN_ACCURACY:
        return f"h2h accuracy {m['accuracy']:.4f} < {MIN_ACCURACY}"
    if market == "spreads" and m["direction_accuracy"] < MIN_ACCURACY:
        return f"spreads direction {m['direction_accuracy']:.4f} < {MIN_ACCURACY}"
    if market == "totals" and m["mae"] >= m["baseline_mae"]:
        return f"totals MAE {m['mae']:.2f} >= baseline {m['baseline_mae']:.2f}"
    return ""


def main() -> Dict[str, Any]:
    print("Chargement et construction de la matrice de features...")
    df = load_training_matrix()
    train, test = split_data(df)
    everything = df[(df["date"] >= TRAIN_START) & (df["date"] <= TEST_END)].copy()
    print(
        f"Train : {len(train)} matchs ({train['date'].min().date()} -> {train['date'].max().date()}) | "
        f"Test : {len(test)} matchs ({test['date'].min().date()} -> {test['date'].max().date()}) | "
        f"Production : {len(everything)} matchs"
    )

    results: Dict[str, Dict[str, Any]] = {}
    failures = []
    for market, (kind, target, decay) in MARKETS.items():
        res = train_market(market, kind, target, decay, train, test, everything)
        results[market] = res
        m = res["metrics"]
        if kind == "classifier":
            line = (
                f"[{market:7s}] acc={m['accuracy']:.4f} (base {m['baseline_accuracy']:.4f}) | "
                f"logloss={m['log_loss']:.4f} (base {m['baseline_log_loss']:.4f}) | brier={m['brier']:.4f}"
            )
            if "platt_gain" in m:
                line += (
                    f" | pente test={m['test_slope']:.2f} | Platt gain={m['platt_gain']:+.4f} -> "
                    f"{'ACTIF' if m['platt_active'] else 'inactif'}"
                )
            print(line)
        else:
            extra = f" | direction={m['direction_accuracy']:.4f}" if "direction_accuracy" in m else ""
            print(
                f"[{market:7s}] MAE={m['mae']:.2f} (base {m['baseline_mae']:.2f}) | biais={m['bias']:+.2f} | "
                f"sigma={m['sigma']:.2f} | corr={m['corr']:.3f}{extra}"
            )
        fail = quality_failure(market, m)
        if fail:
            failures.append(fail)

    if failures:
        raise ModelQualityError(" | ".join(failures) + ". Aucun modèle sauvegardé.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for market, res in results.items():
        joblib.dump(
            {
                "model": res["model"],
                "kind": res["kind"],
                "sigma": res["sigma"],
                "platt": list(res["platt"]) if res["platt"] else None,
                "metrics": res["metrics"],
            },
            MODELS_DIR / f"basketball_{market}.joblib",
        )
    joblib.dump(list(FEATURE_COLUMNS), MODELS_DIR / "basketball_features.joblib")

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_period": [str(train["date"].min().date()), str(train["date"].max().date())],
        "test_period": [str(test["date"].min().date()), str(test["date"].max().date())],
        "features": list(FEATURE_COLUMNS),
        "models": {k: v["metrics"] for k, v in results.items()},
    }
    with open(MODELS_DIR / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Modèles de production sauvegardés dans {MODELS_DIR}")
    return report


if __name__ == "__main__":
    try:
        main()
    except ModelQualityError as exc:
        print(f"ERREUR QUALITÉ : {exc}")
        sys.exit(2)