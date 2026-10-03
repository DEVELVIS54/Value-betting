"""
Orchestrateur NBA — Mode Signal Pur.

Commandes :
  python main.py train            entraîne les 3 modèles (feature_engineering -> model_train)
  python main.py scan             1 scan des cotes -> prédictions -> data/alerts/value_bets.json
  python main.py watch            scans répétés (défaut 300 min : ~432 crédits/mois sur 500)
  python main.py settle           récupère les scores finaux et remplit le tracker (2 crédits)
  python main.py report           accuracy réelle du tracker (objectif : 100 paris à blanc)
  python main.py full             train + scan

Le poller Telegram tourne à part : python src/telegram_alerts.py
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import odds_fetcher  # noqa: E402
import model_train  # noqa: E402
from predictor import BasketballPredictor, TRACKER_PATH, TRACKER_FIELDS  # noqa: E402

SCORES_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/scores/"
DEFAULT_WATCH_INTERVAL_MIN = 300
TARGET_PAPER_BETS = 100


def cmd_train() -> None:
    try:
        model_train.main()
    except model_train.ModelQualityError as exc:
        print(f"ERREUR QUALITÉ : {exc}")
        sys.exit(2)


def cmd_scan() -> int:
    events = odds_fetcher.fetch_nba_odds()
    if not events:
        print("[scan] Aucun match NBA à venir.")
        BasketballPredictor.save_alerts([])
        return 0
    predictor = BasketballPredictor()
    alerts = predictor.evaluate_events(events)
    predictor.save_alerts(alerts)
    print(f"[scan] {len(events)} matchs analysés -> {len(alerts)} signal(s) +EV (max 1 par match).")
    for a in alerts:
        print(
            f"  {a['match']} | {a['market']} | {a['pick']} | IA {a['model_prob_pct']}% | "
            f"théo {a['fair_odds']} | réelle {a['odds']} ({a['bookmaker']}) | EV {a['ev_pct']:+.2f}%"
        )
    return len(alerts)


def cmd_watch(interval_min: int) -> None:
    print(f"[watch] Scan toutes les {interval_min} min. Ctrl+C pour arrêter.")
    while True:
        try:
            cmd_scan()
        except odds_fetcher.QuotaError as exc:
            print(f"[watch] Quota : {exc} — scans suspendus, nouvel essai dans {interval_min} min.")
        except Exception as exc:
            print(f"[watch] Erreur : {exc}")
        try:
            time.sleep(interval_min * 60)
        except KeyboardInterrupt:
            print("\n[watch] Arrêt.")
            return


def _fetch_scores(days_from: int = 3) -> Dict[str, Dict[str, float]]:
    key = odds_fetcher.get_api_key()
    resp = requests.get(
        SCORES_URL,
        params={"apiKey": key, "daysFrom": min(max(days_from, 1), 3)},
        timeout=odds_fetcher.REQUEST_TIMEOUT_S,
    )
    remaining = odds_fetcher._header_int(resp, "x-requests-remaining")
    used = odds_fetcher._header_int(resp, "x-requests-used")
    last = odds_fetcher._header_int(resp, "x-requests-last")
    if remaining is not None or used is not None:
        odds_fetcher.save_quota(remaining, used, last)
    resp.raise_for_status()

    finals: Dict[str, Dict[str, float]] = {}
    for ev in resp.json():
        if not ev.get("completed") or not ev.get("scores"):
            continue
        try:
            finals[ev["id"]] = {s["name"]: float(s["score"]) for s in ev["scores"]}
        except (KeyError, ValueError, TypeError):
            continue
    return finals


def _grade(row: pd.Series, scores: Dict[str, float]) -> Optional[str]:
    """Retourne 'win' / 'loss' / 'push' ou None si non évaluable."""
    home, away = row["home_team"], row["away_team"]
    if home not in scores or away not in scores:
        return None
    h, a = scores[home], scores[away]
    market, pick = row["market"], str(row["pick"])

    if market == "h2h":
        winner = home if h > a else away
        return "win" if pick.startswith(winner) else "loss"

    line = float(row["line"])
    if market == "totals":
        total = h + a
        if total == line:
            return "push"
        over = total > line
        return "win" if (pick.startswith("Over") == over) else "loss"

    if market == "spreads":
        if pick.startswith(home):
            diff = (h - a) + line
        elif pick.startswith(away):
            diff = (a - h) + line
        else:
            return None
        if diff == 0:
            return "push"
        return "win" if diff > 0 else "loss"
    return None


def cmd_settle() -> None:
    if not TRACKER_PATH.exists():
        print("[settle] Tracker vide.")
        return
    df = pd.read_csv(TRACKER_PATH, dtype=str).fillna("")
    for col in TRACKER_FIELDS:   # tracker a l'ancien format : colonnes ajoutees vides
        if col not in df.columns:
            df[col] = ""
    pending = df["result"] == ""
    if not pending.any():
        print("[settle] Rien à régler.")
        return

    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    game_day = pd.to_datetime(df["game_date"], errors="coerce")
    # l'API de scores ne couvre que 3 jours : au-dela, la ligne ne peut plus etre reglee (pas d'appel paye pour elle)
    due = pending & (game_day <= today) & (game_day >= today - pd.Timedelta(days=3))
    if not due.any():
        print("[settle] Aucun match encore joué : pas d'appel API.")
        return
    finals = _fetch_scores()
    settled = 0
    for idx in df[pending].index:
        row = df.loc[idx]
        scores = finals.get(row["event_id"])
        if not scores:
            continue
        res = _grade(row, scores)
        if res is None:
            continue
        df.at[idx, "result"] = res
        df.at[idx, "correct"] = "" if res == "push" else str(int(res == "win"))
        settled += 1

    tmp = TRACKER_PATH.with_suffix(".tmp")
    df[TRACKER_FIELDS].to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, TRACKER_PATH)
    print(f"[settle] {settled} prédiction(s) réglée(s) sur {int(pending.sum())} en attente.")


def cmd_report() -> None:
    if not TRACKER_PATH.exists():
        print("[report] Aucun tracker.")
        return
    df = pd.read_csv(TRACKER_PATH)
    total = len(df)
    done = df[df["correct"].notna() & (df["correct"].astype(str) != "")].copy()
    done["correct"] = done["correct"].astype(int)

    print(f"Prédictions loggées : {total} | réglées : {len(done)}")
    if done.empty:
        return

    def block(title: str, d: pd.DataFrame) -> None:
        if d.empty:
            return
        print(
            f"{title:<22} n={len(d):<4} accuracy={d['correct'].mean():.2%} "
            f"| proba IA moy={d['model_prob'].mean():.2%} | EV moy={d['ev_pct'].mean():+.2f}%"
        )

    block("GLOBAL", done)
    for m in ("h2h", "spreads", "totals"):
        block(f"Marché {m}", done[done["market"] == m])

    alerted = done[done["alerted"] == 1]
    block("Alertes envoyées", alerted)
    n_alerted = len(alerted)
    print(f"Progression paris à blanc (alertes) : {n_alerted}/{TARGET_PAPER_BETS}")
    if n_alerted >= TARGET_PAPER_BETS:
        last = alerted.tail(TARGET_PAPER_BETS)
        print(f"Accuracy sur les {TARGET_PAPER_BETS} derniers signaux : {last['correct'].mean():.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NBA Signal Pur — orchestrateur")
    parser.add_argument("command", choices=["train", "scan", "watch", "settle", "report", "full"])
    parser.add_argument("--interval", type=int, default=DEFAULT_WATCH_INTERVAL_MIN, help="minutes entre scans (watch)")
    args = parser.parse_args()

    try:
        if args.command == "train":
            cmd_train()
        elif args.command == "scan":
            cmd_scan()
        elif args.command == "watch":
            cmd_watch(args.interval)
        elif args.command == "settle":
            cmd_settle()
        elif args.command == "report":
            cmd_report()
        elif args.command == "full":
            cmd_train()
            cmd_scan()
    except odds_fetcher.QuotaError as exc:
        print(f"QUOTA : {exc}")
        sys.exit(3)
    except (RuntimeError, FileNotFoundError, requests.RequestException) as exc:
        print(f"ERREUR : {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()