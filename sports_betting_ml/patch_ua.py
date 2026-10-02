"""patch_ua.py : utilise les en-tetes par defaut (coherents) de nba_api."""
from pathlib import Path

p = Path("update_history.py")
s = p.read_text(encoding="utf-8")
assert "headers=nba_headers()," in s, "patch deja applique ou fichier different"
s = s.replace("headers=nba_headers(),", "headers=None,  # en-tetes par defaut de nba_api (UA de navigateur coherent)")
p.write_text(s, encoding="utf-8")
print("patch OK")
