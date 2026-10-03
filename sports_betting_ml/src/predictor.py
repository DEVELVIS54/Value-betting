# sports_betting_ml/src/predictor.py
"""
Prédicteur NBA temps réel — Mode Signal Pur, ancré sur le marché.

- 3 modèles : h2h (probabilité), spreads (marge attendue), totals (total attendu)
- Ancrage : valeur finale = marché + W * clip(modèle - marché, +/- écart max)
  (le marché = consensus des bookmakers, H2H sans marge)
- Probabilités : H2H directe ; spreads/totals via loi normale (sigma mesurée hors échantillon)
- EV% avec seuil adaptatif (0.75% cotes <= 1.80, 2.0% cotes > 2.50) et plafond MAX_EV_PCT
  (au-dessus = erreur probable du modèle, signal écarté)
- MAX 1 alerte par match (EV% la plus élevée)
- Log de chaque prédiction : data/processed/live_predictions_tracker.csv
- Alertes : data/alerts/value_bets.json (lu en polling par telegram_alerts.py)
"""

import csv
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from difflib import get_close_matches
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

try:
    from feature_engineering import (
        FEATURE_COLUMNS,
        InsufficientHistoryError,
        build_live_features,
        load_team_games,
    )
except ImportError:  # exécution depuis la racine du projet
    from src.feature_engineering import (
        FEATURE_COLUMNS,
        InsufficientHistoryError,
        build_live_features,
        load_team_games,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
ALERTS_PATH = PROJECT_ROOT / "data" / "alerts" / "value_bets.json"
TRACKER_PATH = PROJECT_ROOT / "data" / "processed" / "live_predictions_tracker.csv"

MODEL_PATHS = {
    "h2h": MODELS_DIR / "basketball_h2h.joblib",
    "spreads": MODELS_DIR / "basketball_spreads.joblib",
    "totals": MODELS_DIR / "basketball_totals.joblib",
}
FEATURES_PATH = MODELS_DIR / "basketball_features.joblib"
EXPECTED_KIND = {"h2h": "classifier", "spreads": "regressor", "totals": "regressor"}
SIGMA_BOUNDS = {"spreads": (10.0, 16.0), "totals": (15.0, 24.0)}

# Ancrage marché : poids accordé au modèle et écart maximal pris en compte
W_MODEL = {"h2h": 0.30, "spreads": 0.30, "totals": 0.20}
MAX_DEV = {"h2h": 0.10, "spreads": 4.0, "totals": 5.0}   # proba / points / points
MAX_LINE_GAP = 2.0        # ignore les lignes trop éloignées du consensus (pts)
MIN_BOOKS = 2             # nb minimal de bookmakers pour former un consensus
MAX_HOURS_AHEAD = 72      # n'évalue que les matchs à moins de 72 h (forme et cotes à jour)
MAX_EV_PCT = 12.0         # au-dessus : erreur probable, signal écarté

PROB_MIN, PROB_MAX = 0.02, 0.98

# Totals : mode observation tant que le recentrage live n'est pas estime (biais de niveau du modele)
TOTALS_MIN_EVENTS = 100   # matchs observes (total modele vs marche) avant d'emettre des alertes Totals

# Seuil d'EV adaptatif (en %)
EV_MIN_FAVORITE = 0.75   # cote <= 1.80
EV_MIN_OUTSIDER = 2.00   # cote > 2.50
ODDS_FAV, ODDS_DOG = 1.80, 2.50

TRACKER_FIELDS = [
    "logged_at", "event_id", "game_date", "home_team", "away_team", "market",
    "pick", "line", "model_prob", "fair_odds", "odds", "bookmaker", "ev_pct",
    "model_value", "market_value",
    "alerted", "result", "correct",
]

_ND = NormalDist()


def _clip(p: float) -> float:
    return min(max(float(p), PROB_MIN), PROB_MAX)


def _anchor(model_value: float, market_value: float, weight: float, max_dev: float) -> float:
    dev = float(np.clip(model_value - market_value, -max_dev, max_dev))
    return market_value + weight * dev


def min_ev_threshold(odds: float) -> float:
    """Seuil d'EV% minimal adaptatif selon la cote."""
    if odds <= ODDS_FAV:
        return EV_MIN_FAVORITE
    if odds > ODDS_DOG:
        return EV_MIN_OUTSIDER
    ratio = (odds - ODDS_FAV) / (ODDS_DOG - ODDS_FAV)
    return EV_MIN_FAVORITE + ratio * (EV_MIN_OUTSIDER - EV_MIN_FAVORITE)


def _to_us_local_date(commence_time: str) -> pd.Timestamp:
    """Heure UTC de l'API -> date locale US (Eastern), alignée sur l'historique."""
    dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        local = dt.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        local = dt - timedelta(hours=5)
    return pd.Timestamp(local.date())


def _load_artifact(path: Path, market: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Modèle {market} introuvable : {path}")
    obj = joblib.load(path)
    if not isinstance(obj, dict) or obj.get("kind") != EXPECTED_KIND[market]:
        raise ValueError(f"Modèle {market} : ancien format. Relance : python src\\model_train.py")
    model = obj["model"]
    n_in = getattr(model, "n_features_in_", None)
    if n_in is not None and int(n_in) != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Modèle {market} : {n_in} features attendues, {len(FEATURE_COLUMNS)} fournies. "
            f"Ré-entraîne avec model_train.py."
        )
    sigma = None
    if market in SIGMA_BOUNDS:
        lo, hi = SIGMA_BOUNDS[market]
        sigma = min(max(float(obj.get("sigma") or lo), lo), hi)
    return {"model": model, "sigma": sigma, "platt": obj.get("platt")}


def _platt(params, p):
    """Calibration de Platt (a, b) si disponible, sinon probabilite inchangee."""
    if not params:
        return float(p)
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    z = np.log(p / (1 - p))
    return float(1.0 / (1.0 + np.exp(-(params[0] * z + params[1]))))


class BasketballPredictor:
    def __init__(self, team_games: Optional[pd.DataFrame] = None):
        self.team_games = team_games if team_games is not None else load_team_games()
        self.known_teams = sorted(self.team_games["team"].unique())
        self._team_cache: Dict[str, str] = {}

        self.models: Dict[str, Dict[str, Any]] = {
            market: _load_artifact(path, market) for market, path in MODEL_PATHS.items()
        }

        if FEATURES_PATH.exists():
            saved = list(joblib.load(FEATURES_PATH))
            if saved != FEATURE_COLUMNS:
                missing = [c for c in FEATURE_COLUMNS if c not in saved]
                extra = [c for c in saved if c not in FEATURE_COLUMNS]
                raise ValueError(
                    "basketball_features.joblib incompatible avec feature_engineering.py. "
                    f"Manquantes : {missing} | En trop : {extra}. Relance model_train.py."
                )

        self._migrate_tracker()
        self._logged = self._load_logged_keys()

    # ----------------------------- équipes ---------------------------------- #
    def resolve_team(self, name: str) -> str:
        name = name.strip()
        if name in self._team_cache:
            return self._team_cache[name]

        resolved = None
        if name in self.known_teams:
            resolved = name
        if resolved is None:
            lower = {t.casefold(): t for t in self.known_teams}
            resolved = lower.get(name.casefold())
        if resolved is None:  # "Los Angeles Clippers" -> "LA Clippers"
            words = name.split()
            for nick in (" ".join(words[-2:]), words[-1]):
                hits = [t for t in self.known_teams if t.casefold().endswith(nick.casefold())]
                if len(hits) == 1:
                    resolved = hits[0]
                    break
        if resolved is None:
            close = get_close_matches(name, self.known_teams, n=1, cutoff=0.6)
            resolved = close[0] if close else None
        if resolved is None:
            raise KeyError(f"Équipe non reconnue : '{name}'. Équipes connues : {self.known_teams}")

        self._team_cache[name] = resolved
        return resolved

    # ---------------------------- prédiction -------------------------------- #
    def predict_game(self, home_team: str, away_team: str, game_date) -> Dict[str, Any]:
        home = self.resolve_team(home_team)
        away = self.resolve_team(away_team)
        X = build_live_features(self.team_games, home, away, game_date)[FEATURE_COLUMNS]
        return {
            "home": home,
            "away": away,
            "p_home_win": _clip(_platt(self.models["h2h"]["platt"], self.models["h2h"]["model"].predict_proba(X)[0, 1])),
            "mu_margin": float(self.models["spreads"]["model"].predict(X)[0]),
            "mu_total": float(self.models["totals"]["model"].predict(X)[0]),
        }

    # ---------------------------- marché ------------------------------------ #
    @staticmethod
    def _consensus(event: Dict[str, Any]) -> Dict[str, Any]:
        home, away = event["home_team"], event["away_team"]
        ph, margins, totals = [], [], []
        for bk in event.get("bookmakers", []):
            for mk in bk.get("markets", []):
                key, outs = mk.get("key"), mk.get("outcomes", [])
                if key == "h2h":
                    o = {x.get("name"): x.get("price") for x in outs}
                    ch, ca = o.get(home), o.get(away)
                    if ch and ca and float(ch) > 1.01 and float(ca) > 1.01:
                        a, b = 1.0 / float(ch), 1.0 / float(ca)
                        ph.append(a / (a + b))
                elif key == "spreads":
                    for x in outs:
                        if x.get("name") == home and x.get("point") is not None:
                            margins.append(-float(x["point"]))
                elif key == "totals":
                    for x in outs:
                        if x.get("name") == "Over" and x.get("point") is not None:
                            totals.append(float(x["point"]))
        return {
            "p_home": float(np.mean(ph)) if len(ph) >= MIN_BOOKS else None,
            "margin": float(np.median(margins)) if len(margins) >= MIN_BOOKS else None,
            "total": float(np.median(totals)) if len(totals) >= MIN_BOOKS else None,
        }

    # ---------------------------- événement API ----------------------------- #
    def _candidates(self, event: Dict[str, Any], pred: Dict[str, Any], totals_bias: float = 0.0) -> List[Dict[str, Any]]:
        api_home, api_away = event["home_team"], event["away_team"]
        cons = self._consensus(event)

        p_home = mu_m = mu_t = None
        if cons["p_home"] is not None:
            p_home = _clip(_anchor(pred["p_home_win"], cons["p_home"], W_MODEL["h2h"], MAX_DEV["h2h"]))
        if cons["margin"] is not None:
            mu_m = _anchor(pred["mu_margin"], cons["margin"], W_MODEL["spreads"], MAX_DEV["spreads"])
        if cons["total"] is not None:
            mu_t = _anchor(pred["mu_total"] - totals_bias, cons["total"], W_MODEL["totals"], MAX_DEV["totals"])
        sig_m, sig_t = self.models["spreads"]["sigma"], self.models["totals"]["sigma"]

        out: List[Dict[str, Any]] = []
        for bk in event.get("bookmakers", []):
            bk_name = bk.get("title") or bk.get("key", "")
            for mk in bk.get("markets", []):
                key = mk.get("key")
                for o in mk.get("outcomes", []):
                    odds = o.get("price")
                    if odds is None or float(odds) <= 1.01:
                        continue
                    odds = float(odds)
                    name, point = o.get("name"), o.get("point")
                    prob, pick, line = None, None, None

                    if key == "h2h" and p_home is not None:
                        if name == api_home:
                            prob, pick = p_home, f"{api_home} ML"
                        elif name == api_away:
                            prob, pick = 1.0 - p_home, f"{api_away} ML"

                    elif key == "spreads" and mu_m is not None and point is not None:
                        s = float(point)
                        if name == api_home and abs(-s - cons["margin"]) <= MAX_LINE_GAP:
                            prob, pick, line = _ND.cdf((mu_m + s) / sig_m), f"{api_home} {s:+g}", s
                        elif name == api_away and abs(s - cons["margin"]) <= MAX_LINE_GAP:
                            prob, pick, line = _ND.cdf((s - mu_m) / sig_m), f"{api_away} {s:+g}", s

                    elif key == "totals" and mu_t is not None and point is not None:
                        line_v = float(point)
                        if abs(line_v - cons["total"]) <= MAX_LINE_GAP:
                            p_over = 1.0 - _ND.cdf((line_v - mu_t) / sig_t)
                            if name == "Over":
                                prob, pick, line = p_over, f"Over {line_v:g}", line_v
                            elif name == "Under":
                                prob, pick, line = 1.0 - p_over, f"Under {line_v:g}", line_v

                    if prob is None:
                        continue
                    prob = _clip(prob)
                    out.append({
                        "market": key,
                        "pick": pick,
                        "line": line,
                        "model_prob": prob,
                        "fair_odds": 1.0 / prob,
                        "odds": odds,
                        "bookmaker": bk_name,
                        "ev_pct": (prob * odds - 1.0) * 100.0,
                    })
        return out

    def evaluate_event(self, event: Dict[str, Any], log: bool = True) -> Optional[Dict[str, Any]]:
        """Retourne l'unique meilleure alerte +EV du match, ou None."""
        try:
            hours_ahead = (datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
                           - datetime.now(timezone.utc)).total_seconds() / 3600.0
        except (KeyError, ValueError):
            hours_ahead = 0.0
        if hours_ahead > MAX_HOURS_AHEAD:
            return None
        try:
            game_date = _to_us_local_date(event["commence_time"])
            pred = self.predict_game(event["home_team"], event["away_team"], game_date)
        except (KeyError, InsufficientHistoryError) as exc:
            print(f"[predictor] Match ignoré ({event.get('home_team')} vs {event.get('away_team')}) : {exc}")
            return None

        cons = self._consensus(event)
        obs = {
            "h2h": (pred["p_home_win"], cons["p_home"]),
            "spreads": (pred["mu_margin"], cons["margin"]),
            "totals": (pred["mu_total"], cons["total"]),
        }
        n_tot, bias_tot = self._totals_live_bias()
        totals_active = n_tot >= TOTALS_MIN_EVENTS
        candidates = self._candidates(event, pred, totals_bias=bias_tot if totals_active else 0.0)
        if not candidates:
            return None

        qualified = [
            c for c in candidates
            if min_ev_threshold(c["odds"]) <= c["ev_pct"] <= MAX_EV_PCT
            and (c["market"] != "totals" or totals_active)
        ]
        best_alert = max(qualified, key=lambda c: c["ev_pct"]) if qualified else None

        if log:
            for market in ("h2h", "spreads", "totals"):
                pool = [c for c in candidates if c["market"] == market]
                if not pool:
                    continue
                top = max(pool, key=lambda c: c["ev_pct"])
                alerted = int(best_alert is not None and top is best_alert)
                self._log(event, game_date, top, alerted, *obs[market])

        if best_alert is None:
            return None

        return {
            "alert_id": f"{event.get('id', '')}|{best_alert['market']}|{best_alert['pick']}",
            "match": f"{event['home_team']} vs {event['away_team']}",
            "commence_time": event["commence_time"],
            "market": best_alert["market"],
            "pick": best_alert["pick"],
            "model_prob_pct": round(best_alert["model_prob"] * 100.0, 2),
            "fair_odds": round(best_alert["fair_odds"], 3),
            "odds": round(best_alert["odds"], 3),
            "bookmaker": best_alert["bookmaker"],
            "ev_pct": round(best_alert["ev_pct"], 2),
        }

    def evaluate_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        n_tot, bias_tot = self._totals_live_bias()
        state = f"ACTIF (recentre de {bias_tot:+.2f} pts)" if n_tot >= TOTALS_MIN_EVENTS else "OBSERVATION (aucune alerte Totals)"
        print(f"[predictor] Totals : {n_tot}/{TOTALS_MIN_EVENTS} matchs observes -> {state}")
        alerts = []
        for ev in events:
            alert = self.evaluate_event(ev)
            if alert:
                alerts.append(alert)
        alerts.sort(key=lambda a: a["ev_pct"], reverse=True)
        return alerts

    # ------------------------------ sorties --------------------------------- #
    @staticmethod
    def save_alerts(alerts: List[Dict[str, Any]], path: Path = ALERTS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(alerts),
            "alerts": alerts,
        }
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)  # écriture atomique : le polling ne lit jamais un fichier partiel

    # ------------------------------ tracker --------------------------------- #
    @staticmethod
    def _migrate_tracker() -> None:
        """Archive un tracker a l'ancien format (sans model_value / market_value) au lieu de melanger les schemas."""
        if not TRACKER_PATH.exists() or TRACKER_PATH.stat().st_size == 0:
            return
        with open(TRACKER_PATH, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != TRACKER_FIELDS:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            TRACKER_PATH.rename(TRACKER_PATH.with_name(f"live_predictions_tracker_old_{stamp}.csv"))
            print(f"[predictor] Ancien tracker archive : live_predictions_tracker_old_{stamp}.csv")

    @staticmethod
    def _totals_live_bias() -> tuple:
        """(nb de matchs observes, moyenne de (total modele - total marche)) lue dans le tracker."""
        if not TRACKER_PATH.exists():
            return 0, 0.0
        try:
            df = pd.read_csv(TRACKER_PATH)
        except Exception:
            return 0, 0.0
        if not {"market", "event_id", "model_value", "market_value"} <= set(df.columns):
            return 0, 0.0
        d = df[df["market"] == "totals"].drop_duplicates("event_id")
        dev = (pd.to_numeric(d["model_value"], errors="coerce")
               - pd.to_numeric(d["market_value"], errors="coerce")).dropna()
        return int(len(dev)), (float(dev.mean()) if len(dev) else 0.0)

    @staticmethod
    def _load_logged_keys() -> set:
        keys = set()
        if TRACKER_PATH.exists():
            with open(TRACKER_PATH, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    keys.add((row.get("event_id"), row.get("market")))
        return keys

    def _log(self, event: Dict[str, Any], game_date: pd.Timestamp, c: Dict[str, Any], alerted: int,
             model_value: Optional[float] = None, market_value: Optional[float] = None) -> None:
        key = (str(event.get("id", "")), c["market"])
        if key in self._logged:
            return
        TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
        is_new = not TRACKER_PATH.exists()
        with open(TRACKER_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=TRACKER_FIELDS)
            if is_new:
                w.writeheader()
            w.writerow({
                "logged_at": datetime.now(timezone.utc).isoformat(),
                "event_id": key[0],
                "game_date": game_date.date().isoformat(),
                "home_team": event["home_team"],
                "away_team": event["away_team"],
                "market": c["market"],
                "pick": c["pick"],
                "line": "" if c["line"] is None else c["line"],
                "model_prob": round(c["model_prob"], 4),
                "fair_odds": round(c["fair_odds"], 3),
                "odds": round(c["odds"], 3),
                "bookmaker": c["bookmaker"],
                "ev_pct": round(c["ev_pct"], 2),
                "model_value": "" if model_value is None else round(float(model_value), 4),
                "market_value": "" if market_value is None else round(float(market_value), 4),
                "alerted": alerted,
                "result": "",
                "correct": "",
            })
        self._logged.add(key)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage : python src/predictor.py "Équipe domicile" "Équipe extérieur"')
        sys.exit(1)
    predictor = BasketballPredictor()
    result = predictor.predict_game(sys.argv[1], sys.argv[2], pd.Timestamp.today().normalize())
    print(json.dumps(result, indent=2, ensure_ascii=False))