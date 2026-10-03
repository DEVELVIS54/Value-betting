"""patch_totals_v2.py : Totals v2 = decroissance 0.75 + feature matchs_joues_cette_saison (walk-forward : biais de debut de saison).

Tout ou rien : toutes les ancres sont verifiees AVANT d'ecrire quoi que ce soit ; sauvegardes *.bak_totals.
Fichiers : src/feature_engineering.py, src/model_train.py, src/predictor.py, experiment_walkforward.py
"""
import sys
from pathlib import Path

EDITS = {
    "src/feature_engineering.py": [
        ("assert len(FEATURE_COLUMNS) == 19",
         "assert len(FEATURE_COLUMNS) == 19\n\n"
         "# Totals : 19 colonnes + matchs deja joues dans la saison (corrige le biais de debut de saison)\n"
         "GAMES_PLAYED_COL = \"matchs_joues_cette_saison\"\n"
         "TOTALS_FEATURE_COLUMNS: List[str] = FEATURE_COLUMNS + [GAMES_PLAYED_COL]"),
        ('    tg["b2b"] = (tg["rest_days"] == 1).astype(int)',
         '    tg["b2b"] = (tg["rest_days"] == 1).astype(int)\n\n'
         "    # Matchs deja joues par l'equipe dans la saison courante (octobre = nouvelle saison), AVANT ce match\n"
         '    tg["_season_start"] = np.where(tg["date"].dt.month >= 10, tg["date"].dt.year, tg["date"].dt.year - 1)\n'
         '    tg["games_played"] = tg.groupby(["team", "_season_start"]).cumcount()'),
        ('    keep += [f"pace_{STABLE_WINDOW}", "rest_days", "b2b"]',
         '    keep += [f"pace_{STABLE_WINDOW}", "rest_days", "b2b", "games_played"]'),
        ('    out["away_b2b"] = away.loc[out.index, "b2b"]',
         '    out["away_b2b"] = away.loc[out.index, "b2b"]\n'
         '    out[GAMES_PLAYED_COL] = np.minimum(home.loc[out.index, "games_played"], away.loc[out.index, "games_played"])'),
        ("    return pd.DataFrame([row])[FEATURE_COLUMNS].astype(float)",
         "    return pd.DataFrame([row])[FEATURE_COLUMNS].astype(float)\n\n\n"
         "def live_games_played(team_games: pd.DataFrame, home_team: str, away_team: str, game_date) -> float:\n"
         '    """min(matchs deja joues dans la saison courante par chaque equipe) avant game_date.\n'
         '    Meme definition que la colonne matchs_joues_cette_saison de la matrice d\'entrainement."""\n'
         "    gd = pd.Timestamp(game_date)\n"
         "    if gd.tzinfo is not None:\n"
         "        gd = gd.tz_convert(None)\n"
         "    gd = gd.normalize()\n"
         "    start_year = gd.year if gd.month >= 10 else gd.year - 1\n"
         "    start = pd.Timestamp(year=start_year, month=10, day=1)\n"
         "    counts = []\n"
         "    for t in (home_team.strip(), away_team.strip()):\n"
         '        mask = (team_games["team"] == t) & (team_games["date"] >= start) & (team_games["date"] < gd)\n'
         "        counts.append(int(mask.sum()))\n"
         "    return float(min(counts))"),
    ],
    "src/model_train.py": [
        ("    from feature_engineering import FEATURE_COLUMNS, load_training_matrix",
         "    from feature_engineering import FEATURE_COLUMNS, TOTALS_FEATURE_COLUMNS, load_training_matrix"),
        ("    from src.feature_engineering import FEATURE_COLUMNS, load_training_matrix",
         "    from src.feature_engineering import FEATURE_COLUMNS, TOTALS_FEATURE_COLUMNS, load_training_matrix"),
        ('"totals": ("regressor", "total_pts", 0.90),',
         '"totals": ("regressor", "total_pts", 0.75),'),
        ("class ModelQualityError(RuntimeError):",
         "# Colonnes par marche : Totals recoit en plus matchs_joues_cette_saison (biais de debut de saison)\n"
         "MARKET_FEATURES: Dict[str, list] = {\n"
         '    "h2h": list(FEATURE_COLUMNS),\n'
         '    "spreads": list(FEATURE_COLUMNS),\n'
         '    "totals": list(TOTALS_FEATURE_COLUMNS),\n'
         "}\n\n\n"
         "class ModelQualityError(RuntimeError):"),
        ("    Xtr, ytr = train[FEATURE_COLUMNS], train[target]",
         "    cols = MARKET_FEATURES.get(market, list(FEATURE_COLUMNS))\n"
         "    Xtr, ytr = train[cols], train[target]"),
        ("    Xte, yte = test[FEATURE_COLUMNS], test[target]",
         "    Xte, yte = test[cols], test[target]"),
        ("        everything[FEATURE_COLUMNS], everything[target],",
         "        everything[cols], everything[target],"),
        ('"model": res["model"],',
         '"model": res["model"],\n'
         '                "features": list(MARKET_FEATURES.get(market, FEATURE_COLUMNS)),'),
    ],
    "src/predictor.py": [
        ("    from feature_engineering import (\n        FEATURE_COLUMNS,",
         "    from feature_engineering import (\n        FEATURE_COLUMNS,\n        GAMES_PLAYED_COL,\n"
         "        TOTALS_FEATURE_COLUMNS,\n        live_games_played,"),
        ("    from src.feature_engineering import (\n        FEATURE_COLUMNS,",
         "    from src.feature_engineering import (\n        FEATURE_COLUMNS,\n        GAMES_PLAYED_COL,\n"
         "        TOTALS_FEATURE_COLUMNS,\n        live_games_played,"),
        ("    if n_in is not None and int(n_in) != len(FEATURE_COLUMNS):",
         "    expected = len(TOTALS_FEATURE_COLUMNS) if market == \"totals\" else len(FEATURE_COLUMNS)\n"
         "    if n_in is not None and int(n_in) != expected:"),
        ("{n_in} features attendues, {len(FEATURE_COLUMNS)} fournies.",
         "{n_in} features attendues, {expected} fournies."),
        ("        X = build_live_features(self.team_games, home, away, game_date)[FEATURE_COLUMNS]",
         "        X = build_live_features(self.team_games, home, away, game_date)[FEATURE_COLUMNS]\n"
         "        X_tot = X.copy()\n"
         "        X_tot[GAMES_PLAYED_COL] = live_games_played(self.team_games, home, away, game_date)\n"
         "        X_tot = X_tot[TOTALS_FEATURE_COLUMNS]"),
        ('self.models["totals"]["model"].predict(X)[0]',
         'self.models["totals"]["model"].predict(X_tot)[0]'),
    ],
    "experiment_walkforward.py": [
        ("    df = load_training_matrix()",
         "    df = load_training_matrix()\n"
         "    if MJ in df.columns:   # la matrice contient deja la colonne (feature_engineering patche)\n"
         "        return df"),
    ],
}


def main() -> int:
    texts = {}
    for name, edits in EDITS.items():
        p = Path(name)
        if not p.exists():
            sys.exit(f"[STOP] fichier introuvable : {name} (lancer depuis la racine du projet)")
        s = p.read_text(encoding="utf-8")
        if "TOTALS_FEATURE_COLUMNS" in s or (name == "experiment_walkforward.py" and "feature_engineering patche" in s):
            sys.exit(f"[STOP] patch deja applique sur {name}")
        for old, _ in edits:
            n = s.count(old)
            if n != 1:
                sys.exit(f"[STOP] {name} : ancre trouvee {n} fois (1 attendue) :\n{old}")
        texts[name] = s
    for name, edits in EDITS.items():
        s = texts[name]
        for old, new in edits:
            s = s.replace(old, new)
        Path(name + ".bak_totals").write_text(texts[name], encoding="utf-8")
        Path(name).write_text(s, encoding="utf-8")
        print(f"patch OK : {name} ({len(edits)} modifications, sauvegarde {name}.bak_totals)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
