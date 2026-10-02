"""net_check.py : diagnostic reseau minimal (5 requetes, pause 2 s)."""
import time

import requests
from nba_api.stats.endpoints import leaguegamelog

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
MY_HEADERS = {
    "User-Agent": UA,
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


def timed(label, fn):
    t = time.time()
    try:
        print(f"[OK ] {label}: {fn()} ({time.time() - t:.1f}s)")
    except Exception as e:
        print(f"[ERR] {label}: {type(e).__name__}: {str(e)[:110]} ({time.time() - t:.1f}s)")
    time.sleep(2)


def get(url, **kw):
    r = requests.get(url, timeout=15, **kw)
    return r.status_code, len(r.content)


def llg(headers=None):
    kw = dict(season="2025-26", season_type_all_star="Regular Season",
              date_from_nullable="04/10/2026", date_to_nullable="04/12/2026", timeout=30)
    if headers:
        kw["headers"] = headers
    return f"{len(leaguegamelog.LeagueGameLog(**kw).get_data_frames()[0])} lignes"


timed("stats.nba.com racine", lambda: get("https://stats.nba.com"))
timed("cdn.nba.com scoreboard", lambda: get(
    "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json",
    headers={"User-Agent": UA}))
timed("cdn.nba.com calendrier", lambda: get(
    "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json",
    headers={"User-Agent": UA}))
timed("LeagueGameLog en-tetes par defaut nba_api", llg)
timed("LeagueGameLog en-tetes personnalises", lambda: llg(MY_HEADERS))
