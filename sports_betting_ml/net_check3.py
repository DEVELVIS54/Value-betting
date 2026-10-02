"""net_check3.py : meme appel que update_history.py, avec et sans les parametres explicites."""
import time

from nba_api.stats.endpoints import leaguegamelog

from update_history import nba_headers

h = nba_headers()
base = dict(season="2025-26", season_type_all_star="Regular Season",
            date_from_nullable="04/06/2026", date_to_nullable="04/12/2026",
            headers=h, timeout=20)
for label, extra in (("sans params explicites", {}),
                     ("avec league_id/player_or_team", dict(league_id="00", player_or_team_abbreviation="T"))):
    t = time.time()
    try:
        n = len(leaguegamelog.LeagueGameLog(**base, **extra).get_data_frames()[0])
        print(f"[OK ] {label}: {n} lignes ({time.time() - t:.1f}s)")
    except Exception as e:
        print(f"[ERR] {label}: {type(e).__name__} ({time.time() - t:.1f}s)")
    time.sleep(2)
