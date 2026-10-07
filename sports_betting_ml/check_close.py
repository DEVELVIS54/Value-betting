"""check_close.py : verifie sur le VRAI predictor que la cloture est mise a jour d'un scan a l'autre (0 credit Odds API).

Deux scans factices du meme match : dans 10 h, puis dans 90 min avec un marche qui a bouge.
Tracker de test temporaire ; aucun fichier reel n'est touche.

Lancer depuis la racine du projet : python check_close.py
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
import src.predictor as P  # noqa: E402

TEST_TRACKER = Path("data/processed/_check_close_tracker.csv")
HOME, AWAY = "Boston Celtics", "New York Knicks"


def event(minutes_ahead, total, point, home_price):
    t = (datetime.now(timezone.utc) + timedelta(minutes=minutes_ahead)).strftime("%Y-%m-%dT%H:%M:%SZ")
    books = []
    for name in ("BookA", "BookB", "BookC"):
        books.append({"key": name.lower(), "title": name, "markets": [
            {"key": "h2h", "outcomes": [{"name": HOME, "price": home_price}, {"name": AWAY, "price": 2.45}]},
            {"key": "spreads", "outcomes": [{"name": HOME, "price": 1.91, "point": point},
                                            {"name": AWAY, "price": 1.91, "point": -point}]},
            {"key": "totals", "outcomes": [{"name": "Over", "price": 1.91, "point": total},
                                           {"name": "Under", "price": 1.91, "point": total}]},
        ]})
    return {"id": "check_close", "home_team": HOME, "away_team": AWAY, "commence_time": t, "bookmakers": books}


def main():
    if TEST_TRACKER.exists():
        TEST_TRACKER.unlink()
    P.TRACKER_PATH = TEST_TRACKER
    bp = P.BasketballPredictor()

    bp.evaluate_events([event(600, total=225.5, point=-4.5, home_price=1.55)])
    d1 = pd.read_csv(TEST_TRACKER)
    print("\nscan 1 (match dans 10 h) :")
    print(d1[["market", "market_value", "close_market_value", "close_lead_min"]].to_string(index=False))

    bp.evaluate_events([event(90, total=227.0, point=-5.0, home_price=1.50)])
    d2 = pd.read_csv(TEST_TRACKER).set_index("market")
    print("\nscan 2 (match dans 90 min, marche bouge) :")
    print(d2[["market_value", "close_market_value", "close_lead_min"]].to_string())

    assert len(d2) == 3, "pas de doublon attendu"
    assert abs(d2.loc["totals", "close_market_value"] - 227.0) < 1e-6, "cloture Totals non mise a jour"
    assert abs(d2.loc["spreads", "close_market_value"] - 5.0) < 1e-6, "cloture Spreads non mise a jour"
    assert d2.loc["h2h", "close_market_value"] > d2.loc["h2h", "market_value"], "cloture H2H non mise a jour"
    assert d2["close_lead_min"].between(80, 91).all(), "delai avant le match inattendu"
    assert list(pd.read_csv(TEST_TRACKER).columns) == P.TRACKER_FIELDS, "en-tete inattendu"
    print("\nOK : la cloture suit le marche, sans doublon, delai ~ 90 min")
    TEST_TRACKER.unlink()


if __name__ == "__main__":
    main()
