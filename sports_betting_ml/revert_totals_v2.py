"""revert_totals_v2.py : retour du Totals a la configuration d'origine (decroissance 0.75, 19 colonnes).

Restaure les 4 fichiers depuis les sauvegardes *.bak_totals creees par patch_totals_v2.py, puis remet
la decroissance Totals a 0.75 dans src/model_train.py. Les sauvegardes sont conservees.
"""
import shutil
import sys
from pathlib import Path

FILES = ["src/feature_engineering.py", "src/model_train.py", "src/predictor.py", "experiment_walkforward.py"]
OLD = '"totals": ("regressor", "total_pts", 0.90),'
NEW = '"totals": ("regressor", "total_pts", 0.75),'


def main() -> int:
    for f in FILES:
        if not Path(f + ".bak_totals").exists():
            sys.exit(f"[STOP] sauvegarde introuvable : {f}.bak_totals (lancer depuis la racine du projet)")
    saved = Path("src/model_train.py.bak_totals").read_text(encoding="utf-8")
    if saved.count(OLD) != 1:
        sys.exit("[STOP] ancre de decroissance 0.90 introuvable dans src/model_train.py.bak_totals")
    for f in FILES:
        shutil.copy2(f + ".bak_totals", f)
    p = Path("src/model_train.py")
    p.write_text(p.read_text(encoding="utf-8").replace(OLD, NEW), encoding="utf-8")
    print("revert OK : Totals = decroissance 0.75, 19 colonnes (fichiers restaures depuis *.bak_totals)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
