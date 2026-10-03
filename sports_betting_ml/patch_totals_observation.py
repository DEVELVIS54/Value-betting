"""patch_totals_observation.py : Totals en mode observation + recentrage live du biais de niveau.

- Le tracker enregistre, pour chaque match et chaque marche, la valeur du modele (avant ancrage) et celle du marche
  (colonnes model_value / market_value). Aucun appel API supplementaire.
- Tant que moins de TOTALS_MIN_EVENTS matchs sont observes, AUCUNE alerte Totals n'est emise (H2H et Spreads inchanges).
- Ensuite, le total du modele est recentre de l'ecart moyen (modele - marche) observe en direct avant l'ancrage.
- Un tracker a l'ancien format est archive (live_predictions_tracker_old_<date>.csv), jamais ecrase.

Tout ou rien : toutes les ancres sont verifiees avant d'ecrire ; sauvegarde src/predictor.py.bak_obs.
"""
import sys
from pathlib import Path

TARGET = "src/predictor.py"

EDITS = [
    # 1) constante
    ("PROB_MIN, PROB_MAX = 0.02, 0.98",
     "PROB_MIN, PROB_MAX = 0.02, 0.98\n\n"
     "# Totals : mode observation tant que le recentrage live n'est pas estime (biais de niveau du modele)\n"
     "TOTALS_MIN_EVENTS = 100   # matchs observes (total modele vs marche) avant d'emettre des alertes Totals"),
    # 2) colonnes du tracker
    ('    "pick", "line", "model_prob", "fair_odds", "odds", "bookmaker", "ev_pct",',
     '    "pick", "line", "model_prob", "fair_odds", "odds", "bookmaker", "ev_pct",\n'
     '    "model_value", "market_value",'),
    # 3) migration au demarrage
    ("        self._logged = self._load_logged_keys()",
     "        self._migrate_tracker()\n"
     "        self._logged = self._load_logged_keys()"),
    # 4) nouvelles methodes (avant _load_logged_keys)
    ("    @staticmethod\n    def _load_logged_keys() -> set:",
     "    @staticmethod\n"
     "    def _migrate_tracker() -> None:\n"
     '        """Archive un tracker a l\'ancien format (sans model_value / market_value) au lieu de melanger les schemas."""\n'
     "        if not TRACKER_PATH.exists() or TRACKER_PATH.stat().st_size == 0:\n"
     "            return\n"
     '        with open(TRACKER_PATH, newline="", encoding="utf-8") as f:\n'
     "            header = next(csv.reader(f), [])\n"
     "        if header != TRACKER_FIELDS:\n"
     '            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")\n'
     '            TRACKER_PATH.rename(TRACKER_PATH.with_name(f"live_predictions_tracker_old_{stamp}.csv"))\n'
     '            print(f"[predictor] Ancien tracker archive : live_predictions_tracker_old_{stamp}.csv")\n'
     "\n"
     "    @staticmethod\n"
     "    def _totals_live_bias() -> tuple:\n"
     '        """(nb de matchs observes, moyenne de (total modele - total marche)) lue dans le tracker."""\n'
     "        if not TRACKER_PATH.exists():\n"
     "            return 0, 0.0\n"
     "        try:\n"
     "            df = pd.read_csv(TRACKER_PATH)\n"
     "        except Exception:\n"
     "            return 0, 0.0\n"
     '        if not {"market", "event_id", "model_value", "market_value"} <= set(df.columns):\n'
     "            return 0, 0.0\n"
     '        d = df[df["market"] == "totals"].drop_duplicates("event_id")\n'
     '        dev = (pd.to_numeric(d["model_value"], errors="coerce")\n'
     '               - pd.to_numeric(d["market_value"], errors="coerce")).dropna()\n'
     "        return int(len(dev)), (float(dev.mean()) if len(dev) else 0.0)\n"
     "\n"
     "    @staticmethod\n"
     "    def _load_logged_keys() -> set:"),
    # 5) signature _candidates
    ("    def _candidates(self, event: Dict[str, Any], pred: Dict[str, Any]) -> List[Dict[str, Any]]:",
     "    def _candidates(self, event: Dict[str, Any], pred: Dict[str, Any], totals_bias: float = 0.0) -> List[Dict[str, Any]]:"),
    # 6) recentrage du total du modele
    ('_anchor(pred["mu_total"], cons["total"], W_MODEL["totals"], MAX_DEV["totals"])',
     '_anchor(pred["mu_total"] - totals_bias, cons["total"], W_MODEL["totals"], MAX_DEV["totals"])'),
    # 7) etat Totals + valeurs d'observation, avant le calcul des candidats
    ("        candidates = self._candidates(event, pred)",
     "        cons = self._consensus(event)\n"
     "        obs = {\n"
     '            "h2h": (pred["p_home_win"], cons["p_home"]),\n'
     '            "spreads": (pred["mu_margin"], cons["margin"]),\n'
     '            "totals": (pred["mu_total"], cons["total"]),\n'
     "        }\n"
     "        n_tot, bias_tot = self._totals_live_bias()\n"
     "        totals_active = n_tot >= TOTALS_MIN_EVENTS\n"
     "        candidates = self._candidates(event, pred, totals_bias=bias_tot if totals_active else 0.0)"),
    # 8) pas d'alerte Totals en observation
    ('            if min_ev_threshold(c["odds"]) <= c["ev_pct"] <= MAX_EV_PCT',
     '            if min_ev_threshold(c["odds"]) <= c["ev_pct"] <= MAX_EV_PCT\n'
     '            and (c["market"] != "totals" or totals_active)'),
    # 9) appel du log avec les valeurs d'observation
    ("                self._log(event, game_date, top, alerted)",
     "                self._log(event, game_date, top, alerted, *obs[market])"),
    # 10) signature _log
    ("    def _log(self, event: Dict[str, Any], game_date: pd.Timestamp, c: Dict[str, Any], alerted: int) -> None:",
     "    def _log(self, event: Dict[str, Any], game_date: pd.Timestamp, c: Dict[str, Any], alerted: int,\n"
     "             model_value: Optional[float] = None, market_value: Optional[float] = None) -> None:"),
    # 11) champs ecrits
    ('                "alerted": alerted,',
     '                "model_value": "" if model_value is None else round(float(model_value), 4),\n'
     '                "market_value": "" if market_value is None else round(float(market_value), 4),\n'
     '                "alerted": alerted,'),
    # 12) message d'etat a chaque scan
    ("    def evaluate_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:",
     "    def evaluate_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:\n"
     "        n_tot, bias_tot = self._totals_live_bias()\n"
     '        state = f"ACTIF (recentre de {bias_tot:+.2f} pts)" if n_tot >= TOTALS_MIN_EVENTS else "OBSERVATION (aucune alerte Totals)"\n'
     '        print(f"[predictor] Totals : {n_tot}/{TOTALS_MIN_EVENTS} matchs observes -> {state}")'),
]


def main() -> int:
    p = Path(TARGET)
    if not p.exists():
        sys.exit(f"[STOP] fichier introuvable : {TARGET} (lancer depuis la racine du projet)")
    s = p.read_text(encoding="utf-8")
    if "TOTALS_MIN_EVENTS" in s:
        sys.exit("[STOP] patch deja applique")
    for old, _ in EDITS:
        n = s.count(old)
        if n != 1:
            sys.exit(f"[STOP] ancre trouvee {n} fois (1 attendue) :\n{old}")
    Path(TARGET + ".bak_obs").write_text(s, encoding="utf-8")
    for old, new in EDITS:
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")
    print(f"patch OK : {TARGET} ({len(EDITS)} modifications, sauvegarde {TARGET}.bak_obs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
