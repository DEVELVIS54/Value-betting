"""patch_totals_decay.py : decroissance des poids Totals 0.75 -> 0.90 (retenu en walk-forward)."""
from pathlib import Path

p = Path("src/model_train.py")
s = p.read_text(encoding="utf-8")
old = '"totals": ("regressor", "total_pts", 0.75),'
assert old in s, "patch deja applique ou fichier different"
p.write_text(s.replace(old, '"totals": ("regressor", "total_pts", 0.90),'), encoding="utf-8")
print("patch OK")
