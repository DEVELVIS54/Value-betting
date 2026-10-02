"""net_check4.py : isole le User-Agent (4 requetes, plages distinctes, pause 2 s)."""
import time

from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.library import http as nba_http

from update_history import nba_headers

base = (getattr(nba_http, "STATS_HEADERS", None)
        or getattr(getattr(nba_http, "NBAStatsHTTP", None), "headers", None))
print("UA par defaut nba_api:", (base or {}).get("User-Agent"))
mine = nba_headers()


def llg(d_from, headers):
    kw = dict(season="2025-26", season_type_all_star="Regular Season",
              date_from_nullable=d_from, date_to_nullable="04/12/2026", timeout=20)
    if headers is not None:
        kw["headers"] = headers
    return len(leaguegamelog.LeagueGameLog(**kw).get_data_frames()[0])


def timed(label, d_from, headers):
    t = time.time()
    try:
        print(f"[OK ] {label}: {llg(d_from, headers)} lignes ({time.time() - t:.1f}s)")
    except Exception as e:
        print(f"[ERR] {label}: {type(e).__name__} ({time.time() - t:.1f}s)")
    time.sleep(2)


timed("headers=None (defaut)      | 04/03", "04/03/2026", None)
timed("nba_headers() (mon UA)     | 04/04", "04/04/2026", mine)
timed("copie exacte du defaut     | 04/05", "04/05/2026", dict(base) if base else None)
timed("headers=None (defaut)      | 04/07", "04/07/2026", None)
