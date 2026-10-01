# sports_betting_ml/src/telegram_alerts.py
"""
Poller Telegram — Mode Signal Pur.

- Lit data/alerts/value_bets.json toutes les 60 s (uniquement si le fichier a changé)
- Envoie chaque signal une seule fois (déduplication persistante par match : 1 alerte max par match)
- Format : Match | Marché | Pari Proposé | Probabilité IA (%) | Cote Théorique (1/p) | Cote Réelle | EV%
- Aucun calcul financier (pas de Kelly, pas de mise)

Config (variables d'environnement, ou .env, ou config.py) :
  TELEGRAM_BOT_TOKEN   token du bot (obligatoire)
  TELEGRAM_CHAT_ID     par défaut 6423161858

Lancement : python src/telegram_alerts.py          (boucle infinie)
            python src/telegram_alerts.py --once   (un seul passage)
            python src/telegram_alerts.py --test   (message de test)
"""

import argparse
import html
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALERTS_PATH = PROJECT_ROOT / "data" / "alerts" / "value_bets.json"
SENT_PATH = PROJECT_ROOT / "data" / "alerts" / "sent_alerts.json"

POLL_INTERVAL_S = 60
DEFAULT_CHAT_ID = "6423161858"
SENT_RETENTION_DAYS = 4
MAX_ALERT_AGE_H = 36           # ignore un fichier d'alertes trop ancien
REQUEST_TIMEOUT_S = 20
SEND_PAUSE_S = 1.2             # limite Telegram : ~1 msg/s par chat

MARKET_LABELS = {"h2h": "H2H", "spreads": "Spread", "totals": "Totals"}


def _load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _get_credentials() -> Dict[str, str]:
    _load_env_file()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            import config  # type: ignore
            token = token or getattr(config, "TELEGRAM_BOT_TOKEN", "")
            chat_id = chat_id or str(getattr(config, "TELEGRAM_CHAT_ID", ""))
        except Exception:
            pass
    chat_id = chat_id or DEFAULT_CHAT_ID
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN introuvable (variable d'environnement, .env ou config.py)."
        )
    return {"token": token, "chat_id": chat_id}


# --------------------------------------------------------------------------- #
# État persistant (déduplication)
# --------------------------------------------------------------------------- #
def _now() -> datetime:
    return datetime.now(timezone.utc)


def load_sent() -> Dict[str, str]:
    if not SENT_PATH.exists():
        return {}
    try:
        data = json.loads(SENT_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_sent(sent: Dict[str, str]) -> None:
    cutoff = _now() - timedelta(days=SENT_RETENTION_DAYS)
    kept = {}
    for k, ts in sent.items():
        try:
            if datetime.fromisoformat(ts) >= cutoff:
                kept[k] = ts
        except ValueError:
            continue
    SENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SENT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, SENT_PATH)


def match_key(alert: Dict[str, Any]) -> str:
    """Clé de déduplication : 1 seule alerte par match (id événement, sinon nom du match)."""
    alert_id = str(alert.get("alert_id", ""))
    event_id = alert_id.split("|")[0] if alert_id else ""
    return event_id or f"{alert.get('match', '')}|{alert.get('commence_time', '')}"


# --------------------------------------------------------------------------- #
# Lecture des alertes
# --------------------------------------------------------------------------- #
def read_alerts() -> List[Dict[str, Any]]:
    if not ALERTS_PATH.exists():
        return []
    try:
        payload = json.loads(ALERTS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[telegram] Lecture impossible ({exc}) — nouvel essai au prochain cycle.")
        return []

    if isinstance(payload, list):
        return payload
    generated = payload.get("generated_at")
    if generated:
        try:
            age = _now() - datetime.fromisoformat(generated)
            if age > timedelta(hours=MAX_ALERT_AGE_H):
                print("[telegram] value_bets.json obsolète — ignoré.")
                return []
        except ValueError:
            pass
    return payload.get("alerts", [])


def is_started(alert: Dict[str, Any]) -> bool:
    ct = alert.get("commence_time")
    if not ct:
        return False
    try:
        return datetime.fromisoformat(ct.replace("Z", "+00:00")) <= _now()
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# Format & envoi
# --------------------------------------------------------------------------- #
def format_alert(alert: Dict[str, Any]) -> str:
    esc = lambda v: html.escape(str(v))
    market = MARKET_LABELS.get(alert.get("market", ""), str(alert.get("market", "")).upper())
    ct = alert.get("commence_time", "")
    try:
        start = datetime.fromisoformat(ct.replace("Z", "+00:00")).strftime("%d/%m %H:%M UTC")
    except ValueError:
        start = ct
    return (
        f"🏀 <b>SIGNAL +EV</b>\n"
        f"<b>Match</b> : {esc(alert.get('match', ''))}\n"
        f"<b>Coup d'envoi</b> : {esc(start)}\n"
        f"<b>Marché</b> : {esc(market)}\n"
        f"<b>Pari proposé</b> : {esc(alert.get('pick', ''))}\n"
        f"<b>Probabilité IA</b> : {float(alert.get('model_prob_pct', 0)):.2f}%\n"
        f"<b>Cote théorique (1/p)</b> : {float(alert.get('fair_odds', 0)):.2f}\n"
        f"<b>Cote réelle</b> : {float(alert.get('odds', 0)):.2f} ({esc(alert.get('bookmaker', ''))})\n"
        f"<b>EV</b> : {float(alert.get('ev_pct', 0)):+.2f}%"
    )


def send_message(creds: Dict[str, str], text: str, retries: int = 3) -> bool:
    url = f"https://api.telegram.org/bot{creds['token']}/sendMessage"
    body = {
        "chat_id": creds["chat_id"],
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, json=body, timeout=REQUEST_TIMEOUT_S)
            if r.status_code == 200:
                return True
            if r.status_code == 429:
                wait = int(r.json().get("parameters", {}).get("retry_after", 5))
                print(f"[telegram] Rate limit — pause {wait}s")
                time.sleep(wait + 1)
                continue
            print(f"[telegram] Échec HTTP {r.status_code} : {r.text[:200]}")
            if 400 <= r.status_code < 500:
                return False  # erreur définitive (token/chat invalide)
        except requests.RequestException as exc:
            print(f"[telegram] Erreur réseau (essai {attempt}/{retries}) : {exc}")
        time.sleep(2 * attempt)
    return False


# --------------------------------------------------------------------------- #
# Cycle de polling
# --------------------------------------------------------------------------- #
def process_once(creds: Dict[str, str]) -> int:
    alerts = read_alerts()
    if not alerts:
        return 0

    sent = load_sent()

    # Sécurité : 1 seule alerte par match dans le lot lu (la plus forte EV%)
    best: Dict[str, Dict[str, Any]] = {}
    for a in alerts:
        k = match_key(a)
        if k not in best or float(a.get("ev_pct", 0)) > float(best[k].get("ev_pct", 0)):
            best[k] = a

    n_sent = 0
    for key, alert in sorted(best.items(), key=lambda kv: -float(kv[1].get("ev_pct", 0))):
        if key in sent or is_started(alert):
            continue
        if send_message(creds, format_alert(alert)):
            sent[key] = _now().isoformat()
            save_sent(sent)  # persisté immédiatement : aucun doublon après un crash
            n_sent += 1
            print(f"[telegram] Envoyé : {alert.get('match')} | {alert.get('pick')} | EV {alert.get('ev_pct')}%")
            time.sleep(SEND_PAUSE_S)
    return n_sent


def run_forever(creds: Dict[str, str]) -> None:
    print(f"[telegram] Polling toutes les {POLL_INTERVAL_S}s sur {ALERTS_PATH}")
    last_mtime: Optional[float] = None
    while True:
        try:
            mtime = ALERTS_PATH.stat().st_mtime if ALERTS_PATH.exists() else None
            if mtime is not None and mtime != last_mtime:
                process_once(creds)
                last_mtime = mtime
        except Exception as exc:  # le poller ne doit jamais s'arrêter
            print(f"[telegram] Erreur cycle : {exc}")
        time.sleep(POLL_INTERVAL_S)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="un seul passage puis quitte")
    parser.add_argument("--test", action="store_true", help="envoie un message de test")
    args = parser.parse_args()

    creds = _get_credentials()

    if args.test:
        ok = send_message(creds, "✅ Bot signaux NBA opérationnel (mode Signal Pur).")
        print("Test envoyé." if ok else "Échec de l'envoi du test.")
        sys.exit(0 if ok else 1)
    if args.once:
        print(f"{process_once(creds)} signal(s) envoyé(s).")
        return
    try:
        run_forever(creds)
    except KeyboardInterrupt:
        print("\n[telegram] Arrêt.")


if __name__ == "__main__":
    main()