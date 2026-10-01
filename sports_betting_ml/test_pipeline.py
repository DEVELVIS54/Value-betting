# sports_betting_ml/test_pipeline.py
"""
Tests de validation du pipeline NBA (sans appel API, sans écriture dans le tracker).
Usage : python test_pipeline.py
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import feature_engineering as fe  # noqa: E402
import predictor as pr  # noqa: E402

RESULTS = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        RESULTS.append((name, True, detail or ""))
        print(f"[OK]   {name} {('- ' + str(detail)) if detail else ''}")
    except Exception as exc:
        RESULTS.append((name, False, str(exc)))
        print(f"[FAIL] {name} - {exc}")


# --------------------------------------------------------------------------- #
def t_pkl_exists():
    assert fe.PKL_PATH.exists(), f"Introuvable : {fe.PKL_PATH}"
    raw = pd.read_pickle(fe.PKL_PATH)
    return f"{len(raw)} matchs"


def t_feature_columns():
    assert len(fe.FEATURE_COLUMNS) == 19
    assert len(set(fe.FEATURE_COLUMNS)) == 19


def t_training_matrix():
    m = fe.load_training_matrix()
    assert list(m[fe.FEATURE_COLUMNS].columns) == fe.FEATURE_COLUMNS
    assert m[fe.FEATURE_COLUMNS].isna().sum().sum() == 0, "NaN dans les features"
    assert np.isfinite(m[fe.FEATURE_COLUMNS].to_numpy()).all(), "Valeurs infinies"
    assert m["pace_10"].between(85, 115).all(), "Pace hors plage réaliste (85-115)"
    assert m["home_efg_10"].between(0.35, 0.65).all(), "eFG% hors plage"
    assert m["home_rest_days"].between(0, fe.REST_CAP).all()
    return f"{m.shape[0]} lignes, {m['date'].min().date()} -> {m['date'].max().date()}"


def t_leakage():
    tg = fe.add_rolling_features(fe.build_team_games(pd.read_pickle(fe.PKL_PATH)))
    fe.verify_no_leakage(tg)


def t_live_equals_train():
    """La forme calculée en live doit être identique à celle de l'entraînement."""
    raw = pd.read_pickle(fe.PKL_PATH)
    tg_long = fe.build_team_games(raw)
    m = fe.build_training_matrix(raw)
    row = m.iloc[len(m) // 2]
    live = fe.build_live_features(tg_long, row["home_team"], row["away_team"], row["date"])
    diff = np.abs(live.iloc[0].to_numpy() - row[fe.FEATURE_COLUMNS].to_numpy(dtype=float)).max()
    assert diff < 1e-6, f"Écart train/live = {diff}"
    return f"écart max {diff:.2e}"


def t_models():
    for market, path in pr.MODEL_PATHS.items():
        assert path.exists(), f"Modèle manquant : {path}"
    assert pr.FEATURES_PATH.exists(), "basketball_features.joblib manquant"
    p = pr.BasketballPredictor()
    return f"{len(p.known_teams)} équipes connues"


def t_predict_not_constant():
    p = pr.BasketballPredictor()
    teams = p.known_teams
    date = pd.Timestamp.today().normalize()
    probs = set()
    for i in range(min(8, len(teams) - 1)):
        r = p.predict_game(teams[i], teams[i + 1], date)
        probs.add(round(r["p_home_win"], 4))
    assert len(probs) > 3, f"Probabilités quasi constantes : {probs}"
    assert probs != {0.55}, "Retour aux probabilités par défaut (55%)"
    return f"{len(probs)} valeurs distinctes"


def t_adaptive_threshold():
    assert pr.min_ev_threshold(1.50) == 0.75
    assert pr.min_ev_threshold(1.80) == 0.75
    assert pr.min_ev_threshold(2.60) == 2.0
    assert 0.75 < pr.min_ev_threshold(2.15) < 2.0


def t_one_alert_per_match():
    p = pr.BasketballPredictor()
    home, away = p.known_teams[0], p.known_teams[1]
    start = (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat()
    event = {
        "id": "TEST_EVENT",
        "commence_time": start,
        "home_team": home,
        "away_team": away,
        "bookmakers": [{
            "key": "bk", "title": "BookTest",
            "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": home, "price": 6.0}, {"name": away, "price": 6.0}]},
                {"key": "totals", "outcomes": [
                    {"name": "Over", "point": 200.5, "price": 6.0},
                    {"name": "Under", "point": 200.5, "price": 6.0}]},
                {"key": "spreads", "outcomes": [
                    {"name": home, "point": -3.5, "price": 6.0},
                    {"name": away, "point": 3.5, "price": 6.0}]},
            ],
        }],
    }
    pr.MAX_EV_PCT = 1e9; pr.MIN_BOOKS = 1; pr.MAX_EV_PCT = 1e9; pr.MIN_BOOKS = 1; pr.MAX_EV_PCT = 1e9; pr.MIN_BOOKS = 1; alert = p.evaluate_event(event, log=False)
    assert alert is not None, "Cotes extrêmes (6.0) : au moins un +EV attendu"
    assert isinstance(alert, dict) and alert["market"] in ("h2h", "spreads", "totals")
    return f"1 alerte retenue : {alert['market']} | {alert['pick']} | EV {alert['ev_pct']}%"


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    check("Fichier historique .pkl", t_pkl_exists)
    check("19 colonnes uniques", t_feature_columns)
    check("Matrice d'entraînement propre", t_training_matrix)
    check("Anti data-leakage (shift(1))", t_leakage)
    check("Cohérence features train vs live", t_live_equals_train)
    check("Modèles + features joblib", t_models)
    check("Probabilités non constantes", t_predict_not_constant)
    check("Seuil EV adaptatif", t_adaptive_threshold)
    check("1 seule alerte par match", t_one_alert_per_match)

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} tests OK")
    sys.exit(1 if failed else 0)