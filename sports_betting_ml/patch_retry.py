"""patch_retry.py : plus d'essais, attente plus longue entre deux echecs."""
from pathlib import Path

p = Path("update_history.py")
s = p.read_text(encoding="utf-8")
assert "RETRIES = 4" in s and "time.sleep(PAUSE * (2 ** k))" in s, "patch deja applique ou fichier different"
s = s.replace("RETRIES = 4", "RETRIES = 6")
s = s.replace("time.sleep(PAUSE * (2 ** k))", "time.sleep(10 * k)")
p.write_text(s, encoding="utf-8")
print("patch OK")
