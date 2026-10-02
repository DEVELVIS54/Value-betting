"""patch_headers.py : corrige les en-tetes HTTP et le concat pandas 3 dans update_history.py."""
from pathlib import Path

p = Path("update_history.py")
s = p.read_text(encoding="utf-8")

assert "headers=HEADERS," in s, "patch deja applique ou fichier different"
s = s.replace("headers=HEADERS,", "headers=nba_headers(),")

func = '''def nba_headers():
    """En-tetes par defaut de nba_api (complets) + User-Agent realiste.
    Un dict partiel remplace celui de la librairie et provoque des timeouts."""
    try:
        from nba_api.stats.library import http as nba_http
    except Exception:
        return None
    base = (getattr(nba_http, "STATS_HEADERS", None)
            or getattr(getattr(nba_http, "NBAStatsHTTP", None), "headers", None))
    if not base:
        log.warning("En-tetes par defaut de nba_api introuvables : en-tetes de la librairie utilises")
        return None
    h = dict(base)
    h["User-Agent"] = HEADERS["User-Agent"]
    return h


'''
s = s.replace("def fetch_range(", func + "def fetch_range(", 1)

old = "new_base = pd.concat([old_base, fresh], ignore_index=True)"
new = "new_base = old_base.copy() if fresh.empty else pd.concat([old_base, fresh], ignore_index=True)"
if old in s:
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print("patch OK")
