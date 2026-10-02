"""net_check2.py : isole l'effet des en-tetes et de la plage de dates (4 requetes, pause 2 s)."""
import time

from nba_api.stats.endpoints import leaguegamelog

from update_history import nba_headers


def llg(d_from, d_to, headers=None):
    kw = dict(season="2025-26", season_type_all_star="Regular Season",
              date_from_nullable=d_from, date_to_nullable=d_to, timeout=20)
    if headers:
        kw["headers"] = headers
    return f"{len(leaguegamelog.LeagueGameLog(**kw).get_data_frames()[0])} lignes"


def timed(label, fn):
    t = time.time()
    try:
        print(f"[OK ] {label}: {fn()} ({time.time() - t:.1f}s)")
    except Exception as e:
        print(f"[ERR] {label}: {type(e).__name__} ({time.time() - t:.1f}s)")
    time.sleep(2)


h = nba_headers()
print("nba_headers():", "None (en-tetes librairie)" if h is None else f"{len(h)} cles")
timed("defaut   | 04/02-04/12 (court)", lambda: llg("04/02/2026", "04/12/2026"))
timed("nba_hdrs | 04/10-04/12 (court)", lambda: llg("04/10/2026", "04/12/2026", h))
timed("defaut   | 04/02-09/30 (long)", lambda: llg("04/02/2026", "09/30/2026"))
timed("nba_hdrs | 04/02-09/30 (long)", lambda: llg("04/02/2026", "09/30/2026", h))
