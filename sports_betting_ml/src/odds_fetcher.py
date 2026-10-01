# sports_betting_ml/src/odds_fetcher.py
"""
Récupération des cotes NBA via The Odds API (h2h, spreads, totals) en UN seul appel groupé.

- Endpoint : /v4/sports/basketball_nba/odds (pré-match uniquement)
- Coût API = nb de marchés x nb de régions (3 marchés x 1 région = 3 crédits par scan)
- Suivi du quota via les en-têtes x-requests-remaining / x-requests-used
- Garde-fou : refuse le scan si le quota restant < réserve
- Snapshot JSON horodaté dans data/odds_snapshots/ (+ latest.json)
- Ne garde que les matchs à venir ; nettoie les cotes invalides

Config (variables d'environnement, .env ou config.py) :
  ODDS_API_KEY        clé The Odds API (obligatoire)
  ODDS_REGIONS        défaut "us"  (ex: "us,eu" double le coût)
  ODDS_BOOKMAKERS     optionnel, liste séparée par des virgules (remplace regions)

Usage : python src/odds_fetcher.py           (scan + affichage)
        python src/odds_fetcher.py --quota   (affiche le quota connu, sans appel API)
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "odds_snapshots"
QUOTA_PATH = PROJECT_ROOT / "data" / "odds_quota.json"

API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
MARKETS = "h2h,spreads,totals"
ODDS_FORMAT = "decimal"
DATE_FORMAT = "iso"
REQUEST_TIMEOUT_S = 30
QUOTA_RESERVE = 10          # crédits minimum à conserver
MONTHLY_QUOTA = 500
MAX_SNAPSHOTS = 60          # rotation des fichiers d'historique


class QuotaError(RuntimeError):
    """Quota The Odds API insuffisant ou clé refusée."""


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


def get_api_key() -> str:
    _load_env_file()
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            import config  # type: ignore
            key = getattr(config, "ODDS_API_KEY", "")
        except Exception:
            pass
    if not key or key.upper().startswith(("YOUR", "VOTRE", "XXX")):
        raise RuntimeError("ODDS_API_KEY introuvable ou factice (variable d'environnement, .env ou config.py).")
    return key


# --------------------------------------------------------------------------- #
# Quota
# --------------------------------------------------------------------------- #
def load_quota() -> Dict[str, Any]:
    if QUOTA_PATH.exists():
        try:
            return json.loads(QUOTA_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_quota(remaining: Optional[int], used: Optional[int], last_cost: Optional[int]) -> None:
    QUOTA_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "remaining": remaining,
        "used": used,
        "last_call_cost": last_cost,
        "monthly_quota": MONTHLY_QUOTA,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = QUOTA_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, QUOTA_PATH)


def _header_int(resp: requests.Response, name: str) -> Optional[int]:
    val = resp.headers.get(name)
    try:
        return int(float(val)) if val is not None else None
    except ValueError:
        return None


def estimated_cost() -> int:
    n_markets = len(MARKETS.split(","))
    if os.environ.get("ODDS_BOOKMAKERS"):
        n_books = len([b for b in os.environ["ODDS_BOOKMAKERS"].split(",") if b.strip()])
        return n_markets * max(1, -(-n_books // 10))  # 1 région équivalente / 10 bookmakers
    n_regions = len([r for r in os.environ.get("ODDS_REGIONS", "us").split(",") if r.strip()])
    return n_markets * max(1, n_regions)


# --------------------------------------------------------------------------- #
# Nettoyage
# --------------------------------------------------------------------------- #
def _clean_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        commence = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return None
    if commence <= datetime.now(timezone.utc):
        return None  # pré-match uniquement

    books = []
    for bk in event.get("bookmakers", []):
        markets = []
        for mk in bk.get("markets", []):
            outcomes = [
                o for o in mk.get("outcomes", [])
                if isinstance(o.get("price"), (int, float)) and float(o["price"]) > 1.01
            ]
            if outcomes and mk.get("key") in ("h2h", "spreads", "totals"):
                markets.append({"key": mk["key"], "outcomes": outcomes})
        if markets:
            books.append({
                "key": bk.get("key"),
                "title": bk.get("title"),
                "last_update": bk.get("last_update"),
                "markets": markets,
            })
    if not books:
        return None
    return {
        "id": event.get("id"),
        "commence_time": event["commence_time"],
        "home_team": event.get("home_team"),
        "away_team": event.get("away_team"),
        "bookmakers": books,
    }


# --------------------------------------------------------------------------- #
# Snapshots
# --------------------------------------------------------------------------- #
def save_snapshot(events: List[Dict[str, Any]]) -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count": len(events),
        "events": events,
    }
    path = SNAPSHOT_DIR / f"nba_odds_{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    (SNAPSHOT_DIR / "latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    snaps = sorted(SNAPSHOT_DIR.glob("nba_odds_*.json"))
    for old in snaps[:-MAX_SNAPSHOTS]:
        old.unlink(missing_ok=True)
    return path


def load_latest_snapshot() -> List[Dict[str, Any]]:
    path = SNAPSHOT_DIR / "latest.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("events", [])


# --------------------------------------------------------------------------- #
# Appel API
# --------------------------------------------------------------------------- #
def fetch_nba_odds(save: bool = True) -> List[Dict[str, Any]]:
    """Un appel groupé h2h+spreads+totals. Retourne la liste d'événements nettoyés."""
    api_key = get_api_key()

    known = load_quota().get("remaining")
    cost = estimated_cost()
    if known is not None and known < max(QUOTA_RESERVE, cost):
        raise QuotaError(
            f"Quota insuffisant : {known} crédits restants (coût scan {cost}, réserve {QUOTA_RESERVE})."
        )

    params: Dict[str, str] = {
        "apiKey": api_key,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT,
        "dateFormat": DATE_FORMAT,
    }
    if os.environ.get("ODDS_BOOKMAKERS"):
        params["bookmakers"] = os.environ["ODDS_BOOKMAKERS"]
    else:
        params["regions"] = os.environ.get("ODDS_REGIONS", "us")

    resp = requests.get(API_URL, params=params, timeout=REQUEST_TIMEOUT_S)

    remaining = _header_int(resp, "x-requests-remaining")
    used = _header_int(resp, "x-requests-used")
    last = _header_int(resp, "x-requests-last")
    if remaining is not None or used is not None:
        save_quota(remaining, used, last)

    if resp.status_code == 401:
        raise QuotaError("Clé API refusée (401) : vérifie ODDS_API_KEY.")
    if resp.status_code == 429:
        raise QuotaError("Limite de requêtes atteinte (429).")
    resp.raise_for_status()

    raw = resp.json()
    events = [e for e in (_clean_event(ev) for ev in raw) if e]
    events.sort(key=lambda e: e["commence_time"])

    if save:
        save_snapshot(events)

    print(
        f"[odds] {len(events)} matchs NBA à venir | coût {last} | "
        f"quota restant {remaining} / utilisé {used}"
    )
    return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quota", action="store_true", help="affiche le quota connu sans appel API")
    args = parser.parse_args()

    if args.quota:
        print(json.dumps(load_quota(), indent=2))
        sys.exit(0)

    try:
        evts = fetch_nba_odds()
    except (QuotaError, RuntimeError, requests.RequestException) as exc:
        print(f"ERREUR : {exc}")
        sys.exit(1)

    for e in evts:
        n_books = len(e["bookmakers"])
        print(f"{e['commence_time']} | {e['away_team']} @ {e['home_team']} | {n_books} bookmakers")