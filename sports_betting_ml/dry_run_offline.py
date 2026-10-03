"""dry_run_offline.py : essai a blanc hors ligne du predictor (0 credit Odds API).

Construit 2 evenements factices au format The Odds API, les passe dans le VRAI BasketballPredictor
(vrais modeles, vrai historique) avec un tracker de test temporaire. Aucun fichier reel n'est touche.

Verifie : 3 marches logges par match avec model_value / market_value, plafond d'EV a 12%, Totals en observation.

Lancer depuis la racine du projet : python dry_run_offline.py
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
import src.predictor as P  # noqa: E402

TEST_TRACKER = Path("data/processed/_dry_run_tracker.csv")
HOME, AWAY = "Boston Celtics", "New York Knicks"


def book(title, spread_price=1.91, extra=None):
    markets = [
        {"key": "h2h", "outcomes": [{"name": HOME, "price": 1.55}, {"name": AWAY, "price": 2.45}]},
        {"key": "spreads", "outcomes": [{"name": HOME, "price": spread_price, "point": -4.5},
                                        {"name": AWAY, "price": 1.91, "point": 4.5}]},
        {"key": "totals", "outcomes": [{"name": "Over", "price": 1.91, "point": 225.5},
                                       {"name": "Under", "price": 1.91, "point": 225.5}]},
    ]
    return {"key": title.lower(), "title": title, "markets": markets}


def event(eid, books):
    t = (datetime.now(timezone.utc) + timedelta(hours=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"id": eid, "home_team": HOME, "away_team": AWAY, "commence_time": t, "bookmakers": books}


def main():
    if TEST_TRACKER.exists():
        TEST_TRACKER.unlink()
    P.TRACKER_PATH = TEST_TRACKER          # tracker de test uniquement
    bp = P.BasketballPredictor()

    ev_normal = event("dry_normal", [book("BookA"), book("BookB"), book("BookC")])
    # un bookmaker aberrant : cote 6.0 sur le spread domicile => EV enorme, doit etre rejete (plafond 12%)
    ev_outlier = event("dry_outlier", [book("BookA"), book("BookB"), book("Outlier", spread_price=6.0)])

    alerts = bp.evaluate_events([ev_normal, ev_outlier])
    print("\nalertes emises :")
    for a in alerts:
        print(f"  {a['match']} | {a['market']} | {a['pick']} | cote {a['odds']} | EV {a['ev_pct']}%")
    if not alerts:
        print("  (aucune)")

    df = pd.read_csv(TEST_TRACKER)
    cols = ["event_id", "market", "pick", "odds", "ev_pct", "model_value", "market_value", "alerted"]
    print("\ntracker de test :")
    print(df[cols].to_string(index=False))

    assert list(df.columns) == P.TRACKER_FIELDS, "en-tete du tracker inattendu"
    assert (df.groupby("event_id")["market"].nunique() == 3).all(), "3 marches attendus par match"
    assert df[["model_value", "market_value"]].notna().all().all(), "model_value / market_value manquants"
    assert not ((df["alerted"] == 1) & (df["ev_pct"] > P.MAX_EV_PCT)).any(), "alerte au-dessus du plafond d'EV"
    assert not any(a["market"] == "totals" for a in alerts), "alerte Totals en observation"
    out = df[df["event_id"] == "dry_outlier"]
    spreads_top = out[out["market"] == "spreads"].iloc[0]
    print(f"\nspreads, evenement aberrant : EV logge {spreads_top['ev_pct']}% | alerte = {int(spreads_top['alerted'])} "
          f"(plafond {P.MAX_EV_PCT}%)")
    print("OK : tous les controles passent")
    TEST_TRACKER.unlink()


if __name__ == "__main__":
    main()
