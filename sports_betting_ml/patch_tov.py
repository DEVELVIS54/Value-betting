"""patch_tov.py : tov_pct_10 en pourcentage (x100), comme feature_engineering.py."""
from pathlib import Path

p = Path("update_history.py")
s = p.read_text(encoding="utf-8")
old = '        F[f"{s}_efg_10"], F[f"{s}_tov_pct_10"] = T["efg_10"], T["tov_10"]'
new = '        F[f"{s}_efg_10"], F[f"{s}_tov_pct_10"] = T["efg_10"], T["tov_10"] * tov_scale'
anchor = "    F = pd.DataFrame(index=H.index)\n"
assert old in s and anchor in s and "tov_scale" not in s, "patch deja applique ou fichier different"
s = s.replace(old, new)
s = s.replace(anchor, anchor
              + "    # feature_engineering.py exprime tov_pct en pourcentage (x100), le pkl stocke tov en fraction\n"
              + '    tov_scale = 100.0 if np.nanmedian(df[c[("home", "tov")]].to_numpy(float)) < 1 else 1.0\n', 1)
p.write_text(s, encoding="utf-8")
print("patch OK")
