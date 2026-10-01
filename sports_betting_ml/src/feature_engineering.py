# sports_betting_ml/src/feature_engineering.py
"""
Feature engineering NBA : Four Factors (Dean Oliver) standardisés par le Pace,
jours de repos / back-to-back, formes flash (3) et stable (10), shift(1) strict.

Entrée  : data/processed/nba_with_features.pkl (1 ligne par match, format "wide")
          colonnes utilisées : date, home_team, away_team, home_pts, away_pts, pace,
          {home,away}_{efg, tov, reb, ft_rate}
Sortie  : matrice d'entraînement (19 colonnes) + fonctions live pour predictor.py
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PKL_PATH = PROJECT_ROOT / "data" / "processed" / "nba_with_features.pkl"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "training_matrix.pkl"

FLASH_WINDOW = 3
STABLE_WINDOW = 10
MIN_PERIODS = 3          # identique en entraînement et en live (cohérence)
REST_CAP = 7             # plafond des jours de repos (coupure inter-saison)

# Métriques par match (par équipe), toutes normalisées base 100 possessions
METRICS = ["efg", "tov_pct", "reb_pct", "ft_rate", "net_rtg", "pace"]

# Ordre EXACT des 19 colonnes attendues par le modèle
FEATURE_COLUMNS: List[str] = [
    # Forme stable (10 matchs) : Four Factors + Net Rating
    "home_efg_10", "away_efg_10",
    "home_tov_pct_10", "away_tov_pct_10",
    "home_reb_pct_10", "away_reb_pct_10",
    "home_ft_rate_10", "away_ft_rate_10",
    "home_net_rtg_10", "away_net_rtg_10",
    # Forme flash (3 matchs)
    "home_efg_3", "away_efg_3",
    "home_net_rtg_3", "away_net_rtg_3",
    # Fatigue
    "home_rest_days", "away_rest_days",
    "home_b2b", "away_b2b",
    # Rythme attendu du match
    "pace_10",
]
assert len(FEATURE_COLUMNS) == 19

STABLE_METRICS = ["efg", "tov_pct", "reb_pct", "ft_rate", "net_rtg"]
FLASH_METRICS = ["efg", "net_rtg"]

DATE_ALIASES = ["date", "game_date", "GAME_DATE", "Date"]
HOME_ALIASES = ["home_team", "team_name_home", "team_home"]
AWAY_ALIASES = ["away_team", "team_name_away", "team_away"]
PTS_ALIASES = {"home": ["home_pts", "pts_home"], "away": ["away_pts", "pts_away"]}
PACE_ALIASES = ["pace", "possessions", "poss"]


class InsufficientHistoryError(ValueError):
    """Pas assez de matchs passés pour calculer la forme d'une équipe."""


# --------------------------------------------------------------------------- #
# Utilitaires de colonnes
# --------------------------------------------------------------------------- #
def _resolve(df: pd.DataFrame, candidates: List[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Colonne introuvable pour '{what}'. Essayé : {candidates}. "
        f"Colonnes disponibles : {list(df.columns)}"
    )


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _as_ratio(s: pd.Series) -> pd.Series:
    """Ramène un taux exprimé en % (ex: 54.2) vers une fraction (0.542)."""
    return s / 100.0 if s.median() > 1.5 else s


def _as_percent(s: pd.Series) -> pd.Series:
    """Ramène un taux exprimé en fraction (0.14) vers un pourcentage (14.0)."""
    return s * 100.0 if s.median() < 1.0 else s


# --------------------------------------------------------------------------- #
# Construction du format long (1 ligne = 1 équipe dans 1 match)
# --------------------------------------------------------------------------- #
def build_team_games(df: pd.DataFrame) -> pd.DataFrame:
    """Format long + métriques par match, base 100 possessions. Aucun rolling ici."""
    df = df.copy().reset_index(drop=True)
    gid = pd.Series(df.index, index=df.index)

    date_col = _resolve(df, DATE_ALIASES, "date")
    dates = pd.to_datetime(df[date_col])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    dates = dates.dt.normalize()

    home_col = _resolve(df, HOME_ALIASES, "équipe domicile")
    away_col = _resolve(df, AWAY_ALIASES, "équipe extérieur")
    pace = _num(df, _resolve(df, PACE_ALIASES, "pace")).where(lambda s: s > 0)

    pts = {s: _num(df, _resolve(df, PTS_ALIASES[s], f"points {s}")) for s in ("home", "away")}

    # Four Factors par match (colonnes pré-calculées) ; les *_ma* du pkl sont ignorés
    efg = {s: _as_ratio(_num(df, f"{s}_efg")) for s in ("home", "away")}
    ftr = {s: _as_ratio(_num(df, f"{s}_ft_rate")) for s in ("home", "away")}
    tov = {s: _as_percent(_num(df, f"{s}_tov")) for s in ("home", "away")}

    reb_raw = {s: _num(df, f"{s}_reb") for s in ("home", "away")}
    if reb_raw["home"].median() > 1.5:  # rebonds bruts -> part de rebonds du match
        tot = (reb_raw["home"] + reb_raw["away"]).replace(0, np.nan)
        reb = {"home": reb_raw["home"] / tot, "away": reb_raw["away"] / tot}
    else:
        reb = reb_raw

    frames = []
    for side, opp, is_home, tcol, ocol in (
        ("home", "away", 1, home_col, away_col),
        ("away", "home", 0, away_col, home_col),
    ):
        frames.append(pd.DataFrame({
            "gid": gid,
            "date": dates,
            "team": df[tcol].astype(str).str.strip(),
            "opp_team": df[ocol].astype(str).str.strip(),
            "is_home": is_home,
            "pts": pts[side],
            "opp_pts": pts[opp],
            "efg": efg[side],
            "tov_pct": tov[side],
            "reb_pct": reb[side],
            "ft_rate": ftr[side],
            "pace": pace,
            # Net rating : points marqués - encaissés pour 100 possessions (pace = possessions)
            "net_rtg": 100.0 * (pts[side] - pts[opp]) / pace,
        }))

    tg = pd.concat(frames, ignore_index=True)
    tg = tg.replace([np.inf, -np.inf], np.nan)
    tg = tg.sort_values(["team", "date", "gid"]).reset_index(drop=True)
    return tg


# --------------------------------------------------------------------------- #
# Rolling + repos (anti data-leakage : shift(1) obligatoire)
# --------------------------------------------------------------------------- #
def add_rolling_features(tg: pd.DataFrame) -> pd.DataFrame:
    tg = tg.sort_values(["team", "date", "gid"]).reset_index(drop=True)
    by_team = tg["team"]

    for m in METRICS:
        shifted = tg.groupby("team")[m].shift(1)  # exclut le match courant
        for w in (FLASH_WINDOW, STABLE_WINDOW):
            tg[f"{m}_{w}"] = (
                shifted.groupby(by_team)
                .rolling(w, min_periods=MIN_PERIODS)
                .mean()
                .reset_index(level=0, drop=True)
            )

    # Repos : écart avec le match précédent de la même équipe (info connue avant le match)
    rest = tg.groupby("team")["date"].diff().dt.days
    tg["rest_days"] = rest.clip(upper=REST_CAP).fillna(REST_CAP)
    tg["b2b"] = (tg["rest_days"] == 1).astype(int)
    return tg


# --------------------------------------------------------------------------- #
# Format wide pour l'entraînement
# --------------------------------------------------------------------------- #
def build_training_matrix(df: pd.DataFrame) -> pd.DataFrame:
    tg = add_rolling_features(build_team_games(df))

    keep = ["gid", "date", "team", "pts", "opp_pts"]
    for m in STABLE_METRICS:
        keep.append(f"{m}_{STABLE_WINDOW}")
    for m in FLASH_METRICS:
        keep.append(f"{m}_{FLASH_WINDOW}")
    keep += [f"pace_{STABLE_WINDOW}", "rest_days", "b2b"]

    home = tg[tg["is_home"] == 1][keep].set_index("gid")
    away = tg[tg["is_home"] == 0][keep].set_index("gid")

    out = pd.DataFrame(index=home.index.intersection(away.index))
    out["date"] = home.loc[out.index, "date"]
    out["home_team"] = home.loc[out.index, "team"]
    out["away_team"] = away.loc[out.index, "team"]

    for m in STABLE_METRICS:
        c = f"{m}_{STABLE_WINDOW}"
        out[f"home_{c}"] = home.loc[out.index, c]
        out[f"away_{c}"] = away.loc[out.index, c]
    for m in FLASH_METRICS:
        c = f"{m}_{FLASH_WINDOW}"
        out[f"home_{c}"] = home.loc[out.index, c]
        out[f"away_{c}"] = away.loc[out.index, c]

    out["home_rest_days"] = home.loc[out.index, "rest_days"]
    out["away_rest_days"] = away.loc[out.index, "rest_days"]
    out["home_b2b"] = home.loc[out.index, "b2b"]
    out["away_b2b"] = away.loc[out.index, "b2b"]
    out["pace_10"] = (
        home.loc[out.index, f"pace_{STABLE_WINDOW}"]
        + away.loc[out.index, f"pace_{STABLE_WINDOW}"]
    ) / 2.0

    # Cibles (jamais utilisées comme features)
    out["home_pts"] = home.loc[out.index, "pts"]
    out["away_pts"] = away.loc[out.index, "pts"]
    out["home_win"] = (out["home_pts"] > out["away_pts"]).astype(int)
    out["margin"] = out["home_pts"] - out["away_pts"]
    out["total_pts"] = out["home_pts"] + out["away_pts"]

    # Saison NBA : octobre -> juin
    out["season"] = np.where(out["date"].dt.month >= 9, out["date"].dt.year, out["date"].dt.year - 1)

    out = out.sort_values("date").reset_index(drop=True)
    out = out.dropna(subset=FEATURE_COLUMNS + ["home_pts", "away_pts"]).reset_index(drop=True)
    return out


def load_training_matrix(pkl_path: Optional[Path] = None) -> pd.DataFrame:
    path = Path(pkl_path) if pkl_path else PKL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    return build_training_matrix(pd.read_pickle(path))


# --------------------------------------------------------------------------- #
# Fonctions live (utilisées par predictor.py)
# --------------------------------------------------------------------------- #
def load_team_games(pkl_path: Optional[Path] = None) -> pd.DataFrame:
    """Historique long avec métriques par match, à charger une seule fois."""
    path = Path(pkl_path) if pkl_path else PKL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    return build_team_games(pd.read_pickle(path))


def _team_form(tg: pd.DataFrame, team: str, game_date: pd.Timestamp) -> Dict[str, float]:
    hist = tg[(tg["team"] == team) & (tg["date"] < game_date)].sort_values(["date", "gid"])
    if len(hist) < MIN_PERIODS:
        raise InsufficientHistoryError(
            f"{team} : {len(hist)} match(s) avant {game_date.date()} (min {MIN_PERIODS})."
        )
    last10 = hist.tail(STABLE_WINDOW)
    last3 = hist.tail(FLASH_WINDOW)
    form = {f"{m}_{STABLE_WINDOW}": float(last10[m].mean()) for m in METRICS}
    form.update({f"{m}_{FLASH_WINDOW}": float(last3[m].mean()) for m in METRICS})
    rest = (game_date - hist["date"].iloc[-1]).days
    form["rest_days"] = float(min(rest, REST_CAP))
    form["b2b"] = float(rest == 1)
    return form


def build_live_features(
    team_games: pd.DataFrame,
    home_team: str,
    away_team: str,
    game_date,
) -> pd.DataFrame:
    """Vecteur de 19 colonnes (ordre exact FEATURE_COLUMNS) pour un match à venir.
    Utilise uniquement les matchs strictement antérieurs à game_date."""
    gd = pd.Timestamp(game_date)
    if gd.tzinfo is not None:
        gd = gd.tz_convert(None)
    gd = gd.normalize()

    h = _team_form(team_games, home_team.strip(), gd)
    a = _team_form(team_games, away_team.strip(), gd)

    row = {}
    for m in STABLE_METRICS:
        c = f"{m}_{STABLE_WINDOW}"
        row[f"home_{c}"], row[f"away_{c}"] = h[c], a[c]
    for m in FLASH_METRICS:
        c = f"{m}_{FLASH_WINDOW}"
        row[f"home_{c}"], row[f"away_{c}"] = h[c], a[c]
    row["home_rest_days"], row["away_rest_days"] = h["rest_days"], a["rest_days"]
    row["home_b2b"], row["away_b2b"] = h["b2b"], a["b2b"]
    row["pace_10"] = (h[f"pace_{STABLE_WINDOW}"] + a[f"pace_{STABLE_WINDOW}"]) / 2.0

    return pd.DataFrame([row])[FEATURE_COLUMNS].astype(float)


# --------------------------------------------------------------------------- #
# Contrôle anti-leakage
# --------------------------------------------------------------------------- #
def verify_no_leakage(tg_rolled: pd.DataFrame) -> None:
    """Le 1er match de chaque équipe ne doit avoir aucune forme calculée, et la
    forme d'un match ne doit jamais inclure ce match."""
    first = tg_rolled.groupby("team").head(1)
    if first[f"efg_{STABLE_WINDOW}"].notna().any():
        raise AssertionError("Data leakage : forme non nulle sur le 1er match d'une équipe.")

    sample_team = tg_rolled["team"].iloc[len(tg_rolled) // 2]
    t = tg_rolled[tg_rolled["team"] == sample_team].reset_index(drop=True)
    for i in range(MIN_PERIODS + 5, min(len(t), MIN_PERIODS + 40)):
        expected = t.loc[max(0, i - STABLE_WINDOW):i - 1, "efg"].mean()
        got = t.loc[i, f"efg_{STABLE_WINDOW}"]
        if not np.isclose(expected, got, equal_nan=True):
            raise AssertionError(f"Data leakage détecté ({sample_team}, ligne {i}).")


# --------------------------------------------------------------------------- #
# Features differentielles (domicile - exterieur) pour H2H et Spreads
# --------------------------------------------------------------------------- #
DELTA_COLUMNS: List[str] = [
    "d_efg_10", "d_tov_pct_10", "d_reb_pct_10", "d_ft_rate_10", "d_net_rtg_10",
    "d_efg_3", "d_net_rtg_3", "d_rest_days", "d_b2b", "pace_10",
]
_DELTA_BASES = ["efg_10", "tov_pct_10", "reb_pct_10", "ft_rate_10", "net_rtg_10",
                "efg_3", "net_rtg_3", "rest_days", "b2b"]


def to_deltas(X: pd.DataFrame) -> pd.DataFrame:
    """19 colonnes brutes -> 10 features differentielles (domicile - exterieur)."""
    out = pd.DataFrame(index=X.index)
    for base in _DELTA_BASES:
        out[f"d_{base}"] = X[f"home_{base}"] - X[f"away_{base}"]
    out["pace_10"] = X["pace_10"]
    return out[DELTA_COLUMNS]


if __name__ == "__main__":
    raw = pd.read_pickle(PKL_PATH)
    print(f"Matchs bruts : {len(raw)}")

    tg_ = add_rolling_features(build_team_games(raw))
    verify_no_leakage(tg_)
    print("Contrôle anti-leakage : OK")

    matrix = build_training_matrix(raw)
    matrix.to_pickle(OUT_PATH)
    print(f"Matrice d'entraînement : {matrix.shape} -> {OUT_PATH}")
    print(f"Colonnes features : {len(FEATURE_COLUMNS)}")
    print(f"Période : {matrix['date'].min().date()} -> {matrix['date'].max().date()}")
    print(matrix[FEATURE_COLUMNS].describe().T[["mean", "min", "max"]].round(3).to_string())