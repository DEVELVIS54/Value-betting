"""update_history.py : mise a jour quotidienne de l'historique NBA (anti-leakage strict).

Lancer depuis la racine du projet, venv actif :
    python update_history.py                      # production
    python update_history.py --dry-run            # aucune ecriture
    python update_history.py --season 2025-26 --refetch-days 10 --dry-run
        # test : refetch de matchs deja presents -> reconcilie unites et formules

Pipeline : pkl -> fetch veille (LeagueGameLog) -> lignes brutes -> wide -> 19 features (t-1)
           + matchs_joues_cette_saison -> audit -> ecriture atomique.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.static import teams as static_teams

# ------------------------------- CONFIG -------------------------------
ROOT = Path(__file__).resolve().parent
PKL = ROOT / "data" / "processed" / "nba_with_features.pkl"
LEDGER = ROOT / "data" / "processed" / "raw_boxscores_2026_27.pkl"
LOCK = ROOT / "data" / "processed" / ".update_history.lock"
LOG_FILE = ROOT / "logs" / "update_history.log"

SEASON = "2026-27"
SEASON_TYPE = "Regular Season"
TIMEOUT = 30          # secondes, impose
PAUSE = 2             # secondes, impose apres chaque appel HTTP
RETRIES = 6
OVERLAP_DAYS = 3      # recouvre les matchs termines apres le dernier run
ATOL = 1e-9

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

FEATURES_19 = [
    "home_efg_10", "away_efg_10", "home_tov_pct_10", "away_tov_pct_10",
    "home_reb_pct_10", "away_reb_pct_10", "home_ft_rate_10", "away_ft_rate_10",
    "home_net_rtg_10", "away_net_rtg_10", "home_efg_3", "away_efg_3",
    "home_net_rtg_3", "away_net_rtg_3", "home_rest_days", "away_rest_days",
    "home_b2b", "away_b2b", "pace_10",
]
NEW_COLS = ["home_matchs_joues_cette_saison", "away_matchs_joues_cette_saison",
            "matchs_joues_cette_saison"]
BASE_COLS = ["game_id", "date", "home_team", "away_team", "home_pts", "away_pts", "pace"]
STAT_ALTS = {"efg": ["efg"], "tov": ["tov", "tov_pct"], "reb": ["reb", "reb_pct"], "ft": ["ft_rate"]}
RAW_NEEDED = ["GAME_ID", "TEAM_ID", "GAME_DATE", "MATCHUP", "WL", "PTS",
              "FGM", "FGA", "FG3M", "FTM", "FTA", "OREB", "DREB", "TOV"]

log = logging.getLogger("update_history")


# ------------------------------- OUTILS -------------------------------
def setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


def acquire_lock() -> bool:
    """Evite deux instances simultanees (run_all.py + lancement manuel)."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists() and time.time() - LOCK.stat().st_mtime > 3600:
        LOCK.unlink()  # verrou perime
    try:
        fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release_lock() -> None:
    try:
        LOCK.unlink()
    except FileNotFoundError:
        pass


def atomic_pickle(obj: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    obj.to_pickle(tmp)
    os.replace(tmp, path)


def norm(s: str) -> str:
    s = str(s).lower().replace("los angeles", "la")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def build_id2name(old: pd.DataFrame) -> dict:
    """TEAM_ID -> nom d'equipe du pkl, via le jeu statique nba_api (0 requete HTTP)."""
    by_norm = {norm(n): n for n in pd.unique(pd.concat([old["home_team"], old["away_team"]]))}
    id2name, missing = {}, []
    for t in static_teams.get_teams():
        keys = [norm(t["full_name"]), norm(f'{t["city"]} {t["nickname"]}'),
                norm(t["nickname"]), t["abbreviation"].lower()]
        hit = next((by_norm[k] for k in keys if k in by_norm), None)
        if hit is None:
            missing.append(t["full_name"])
        else:
            id2name[int(t["id"])] = hit
    if missing:
        sys.exit(f"[FATAL] Equipes sans correspondance dans le pkl: {missing} | "
                 f"noms pkl: {sorted(by_norm.values())}")
    return id2name


def resolve_stat_cols(df: pd.DataFrame) -> dict:
    c = {}
    for s in ("home", "away"):
        for k, alts in STAT_ALTS.items():
            hit = [f"{s}_{a}" for a in alts if f"{s}_{a}" in df.columns]
            if not hit:
                sys.exit(f"[FATAL] Colonne {s}_{k} absente. Colonnes pkl: {list(df.columns)}")
            c[(s, k)] = hit[0]
    miss = [x for x in BASE_COLS if x not in df.columns]
    if miss:
        sys.exit(f"[FATAL] Colonnes absentes du pkl: {miss}")
    extra = set(df.columns) - set(BASE_COLS) - set(c.values()) - set(FEATURES_19) - set(NEW_COLS)
    if extra:
        sys.exit(f"[FATAL] Colonnes pkl non calculables pour les nouveaux matchs: {sorted(extra)}")
    return c


# ------------------------------- RESEAU -------------------------------
def nba_headers():
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


def fetch_range(season: str, d_from, d_to) -> pd.DataFrame:
    """1 appel LeagueGameLog (30 equipes). timeout=30, sleep(2) strict apres chaque appel."""
    last_err = None
    for k in range(1, RETRIES + 1):
        try:
            frames = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=SEASON_TYPE,
                league_id="00",
                player_or_team_abbreviation="T",
                date_from_nullable=d_from.strftime("%m/%d/%Y"),
                date_to_nullable=d_to.strftime("%m/%d/%Y"),
                headers=None,  # en-tetes par defaut de nba_api (UA de navigateur coherent)
                timeout=TIMEOUT,
            ).get_data_frames()
            time.sleep(PAUSE)
            return frames[0]
        except Exception as e:
            last_err = e
            log.warning("Essai %d/%d %s->%s: %s", k, RETRIES, d_from, d_to, e)
            time.sleep(10 * k)
    raise RuntimeError(f"nba_api indisponible: {last_err}")


# ------------------------------- BRUT -> WIDE -------------------------
def to_wide(raw: pd.DataFrame, old: pd.DataFrame, c: dict, id2name: dict) -> pd.DataFrame:
    """Lignes equipe -> lignes match au format du pkl (unites alignees sur l'existant)."""
    raw = raw[raw["WL"].notna()].copy()
    raw["is_home"] = raw["MATCHUP"].str.contains("vs.", regex=False)
    raw["poss"] = raw["FGA"] - raw["OREB"] + raw["TOV"] + 0.44 * raw["FTA"]
    H = raw[raw["is_home"]].drop_duplicates("GAME_ID").set_index("GAME_ID")
    A = raw[~raw["is_home"]].drop_duplicates("GAME_ID").set_index("GAME_ID")
    ids = H.index.intersection(A.index)
    H, A = H.loc[ids], A.loc[ids]
    if len(ids) == 0:
        return pd.DataFrame(columns=list(old.columns) + ["api_id"])

    unknown = (set(H["TEAM_ID"]) | set(A["TEAM_ID"])) - set(id2name)
    if unknown:
        sys.exit(f"[FATAL] TEAM_ID inconnus du jeu statique: {sorted(unknown)}")

    pct_tov = pd.to_numeric(old[c[("home", "tov")]]).median() > 1
    pct_reb = pd.to_numeric(old[c[("home", "reb")]]).median() > 1
    ktov = 1.0 if pct_tov else 0.01
    kreb = 100.0 if pct_reb else 1.0
    tot_reb = H["OREB"] + H["DREB"] + A["OREB"] + A["DREB"]

    out = pd.DataFrame({
        "api_id": ids.to_numpy(),
        "date": pd.to_datetime(H["GAME_DATE"]).to_numpy(),
        "home_team": [id2name[int(t)] for t in H["TEAM_ID"]],
        "away_team": [id2name[int(t)] for t in A["TEAM_ID"]],
        "home_pts": H["PTS"].to_numpy(), "away_pts": A["PTS"].to_numpy(),
        "pace": ((H["poss"] + A["poss"]) / 2).to_numpy(),
    })
    for s, T in (("home", H), ("away", A)):
        out[c[(s, "efg")]] = ((T["FGM"] + 0.5 * T["FG3M"]) / T["FGA"]).to_numpy()
        out[c[(s, "tov")]] = (100 * T["TOV"] / T["poss"] * ktov).to_numpy()
        out[c[(s, "reb")]] = ((T["OREB"] + T["DREB"]) / tot_reb * kreb).to_numpy()
        out[c[(s, "ft")]] = (T["FTM"] / T["FGA"]).to_numpy()

    # alignement des dtypes sur le pkl
    if pd.api.types.is_numeric_dtype(old["game_id"]):
        out["game_id"] = out["api_id"].astype("int64")
    else:
        width = len(str(old["game_id"].iloc[0]))
        out["game_id"] = out["api_id"].astype(str).str.zfill(width)
    if not np.issubdtype(old["date"].dtype, np.datetime64):
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out[list(old.columns) + ["api_id"]]


def reconcile(wide: pd.DataFrame, old: pd.DataFrame, c: dict) -> None:
    """Matchs deja presents : valeurs recalculees == valeurs stockees (valide formules et unites)."""
    m = wide.merge(old, on="game_id", suffixes=("_n", "_o"))
    if m.empty:
        return
    cols = ["pace"] + [c[(s, k)] for s in ("home", "away") for k in STAT_ALTS]
    bad = [x for x in cols if not np.allclose(m[f"{x}_n"].astype(float), m[f"{x}_o"].astype(float),
                                              rtol=1e-3, atol=1e-6)]
    if bad:
        sys.exit(f"[FATAL] Reconciliation echouee ({len(m)} matchs), colonnes divergentes: {bad}")
    log.info("Reconciliation OK sur %d matchs existants", len(m))


# ------------------------------- FEATURES (t-1) -----------------------
def compute_features(df: pd.DataFrame, c: dict) -> pd.DataFrame:
    """19 features + matchs joues. Moyennes mobiles des matchs PRECEDENTS uniquement :
    shift(1) intra-equipe PUIS rolling (min_periods=3), historique continu entre saisons."""
    parts = []
    for s, o in (("home", "away"), ("away", "home")):
        parts.append(pd.DataFrame({
            "game_id": df["game_id"].to_numpy(),
            "date": pd.to_datetime(df["date"]).to_numpy(),
            "side": s,
            "team": df[f"{s}_team"].to_numpy(),
            "efg": df[c[(s, "efg")]].to_numpy(float),
            "tov": df[c[(s, "tov")]].to_numpy(float),
            "reb": df[c[(s, "reb")]].to_numpy(float),
            "ft": df[c[(s, "ft")]].to_numpy(float),
            "net": (100 * (df[f"{s}_pts"] - df[f"{o}_pts"]) / df["pace"]).to_numpy(float),
            "pace": df["pace"].to_numpy(float),
        }))
    L = pd.concat(parts, ignore_index=True)
    L = L.sort_values(["team", "date", "game_id"], kind="mergesort").reset_index(drop=True)

    g = L.groupby("team", sort=False)
    for col in ("efg", "tov", "reb", "ft", "net", "pace"):
        L[f"{col}_prev"] = g[col].shift(1)                      # decalage strict
    g = L.groupby("team", sort=False)
    roll = lambda col, w: g[f"{col}_prev"].transform(lambda x: x.rolling(w, min_periods=3).mean())
    for col in ("efg", "tov", "reb", "ft", "net", "pace"):
        L[f"{col}_10"] = roll(col, 10)
    L["efg_3"], L["net_3"] = roll("efg", 3), roll("net", 3)
    L["rest"] = (L["date"] - g["date"].shift(1)).dt.days.fillna(7).clip(upper=7)
    L["b2b"] = (L["rest"] == 1).astype(int)

    # matchs joues AVANT ce match dans la saison courante (octobre+ = nouvelle saison)
    L["season"] = np.where(L["date"].dt.month >= 10, L["date"].dt.year, L["date"].dt.year - 1)
    L["n_played"] = L.groupby(["team", "season"], sort=False).cumcount()

    H = L[L["side"] == "home"].set_index("game_id")
    A = L[L["side"] == "away"].set_index("game_id").reindex(H.index)
    F = pd.DataFrame(index=H.index)
    # feature_engineering.py exprime tov_pct en pourcentage (x100), le pkl stocke tov en fraction
    tov_scale = 100.0 if np.nanmedian(df[c[("home", "tov")]].to_numpy(float)) < 1 else 1.0
    for s, T in (("home", H), ("away", A)):
        F[f"{s}_efg_10"], F[f"{s}_tov_pct_10"] = T["efg_10"], T["tov_10"] * tov_scale
        F[f"{s}_reb_pct_10"], F[f"{s}_ft_rate_10"] = T["reb_10"], T["ft_10"]
        F[f"{s}_net_rtg_10"] = T["net_10"]
        F[f"{s}_efg_3"], F[f"{s}_net_rtg_3"] = T["efg_3"], T["net_3"]
        F[f"{s}_rest_days"], F[f"{s}_b2b"] = T["rest"], T["b2b"]
    F["pace_10"] = (H["pace_10"] + A["pace_10"]) / 2
    F["home_matchs_joues_cette_saison"] = H["n_played"]
    F["away_matchs_joues_cette_saison"] = A["n_played"]
    F["matchs_joues_cette_saison"] = np.minimum(H["n_played"], A["n_played"])
    return F[FEATURES_19 + NEW_COLS]


# ------------------------------- AUDIT --------------------------------
def audit(old_base: pd.DataFrame, new_base: pd.DataFrame, F_new: pd.DataFrame,
          added_ids: list, c: dict, old_stored: pd.DataFrame, rebase: bool) -> None:
    assert new_base["game_id"].is_unique, "game_id dupliques"
    teams = set(new_base["home_team"]) | set(new_base["away_team"])
    assert len(teams) == 30, f"{len(teams)} equipes (30 attendues)"
    assert len(new_base) >= 14000, f"{len(new_base)} matchs (>= 14000 attendus)"

    # (a) le futur ne modifie pas le passe
    F_old = compute_features(old_base, c)
    a = F_old[FEATURES_19].to_numpy(float)
    b = F_new.reindex(F_old.index)[FEATURES_19].to_numpy(float)
    ok = np.isclose(a, b, atol=ATOL, equal_nan=True)
    assert ok.all(), f"FUITE: passe modifie, features: {list(np.array(FEATURES_19)[~ok.all(axis=0)])}"

    # (b) un match n'influence pas ses propres features
    d = pd.to_datetime(new_base["date"])
    mask = (d == d.max()).to_numpy()
    num = ["home_pts", "away_pts", "pace"] + [c[(s, k)] for s in ("home", "away") for k in STAT_ALTS]
    P = new_base.copy()
    P.loc[mask, num] = P.loc[mask, num].to_numpy(float) * 1.37
    ids = new_base.loc[mask, "game_id"]
    ok2 = np.isclose(F_new.loc[ids, FEATURES_19].to_numpy(float),
                     compute_features(P, c).loc[ids, FEATURES_19].to_numpy(float),
                     atol=ATOL, equal_nan=True)
    assert ok2.all(), "FUITE: features d'un match dependantes de son propre box score"

    # (c) derive de definition vs features deja stockees dans le pkl
    if all(x in old_stored.columns for x in FEATURES_19) and not rebase:
        stored = old_stored.set_index("game_id")[FEATURES_19].reindex(F_old.index).to_numpy(float)
        same = np.isclose(stored, a, rtol=1e-6, atol=1e-6, equal_nan=True)
        assert same.all(), ("DERIVE: features stockees != recalculees "
                            f"({(~same).any(axis=1).sum()} matchs). Relancer avec --rebase pour valider.")

    # (d) nouveaux matchs et colonne de warm-up
    if added_ids:
        vals = F_new.loc[added_ids, FEATURES_19].to_numpy(float)
        assert np.isfinite(vals).all(), "NaN/inf dans les features des nouveaux matchs"
    mj = F_new[NEW_COLS]
    assert (mj >= 0).all().all() and (mj <= 82).all().all(), "matchs_joues hors [0,82]"
    log.info("AUDIT OK | %d nouveaux matchs | passe inchange | independance verifiee", len(added_ids))


# ------------------------------- MAIN ---------------------------------
def run(args) -> int:
    old = pd.read_pickle(PKL)
    c = resolve_stat_cols(old)
    id2name = build_id2name(old)
    feat_cols = [x for x in FEATURES_19 + NEW_COLS if x in old.columns]
    old_base = old.drop(columns=feat_cols).copy()

    last = pd.to_datetime(old_base["date"]).max().date()
    target = datetime.now(timezone.utc).date() - timedelta(days=1)   # veille (heure NBA)
    start = last - timedelta(days=args.refetch_days)
    log.info("pkl: %d matchs, dernier match %s | cible %s | saison %s",
             len(old_base), last, target, args.season)

    fresh, raw_new = pd.DataFrame(columns=list(old_base.columns)), None
    if start <= target:
        raw = fetch_range(args.season, start, target)
        if not raw.empty:
            miss = [x for x in RAW_NEEDED if x not in raw.columns]
            if miss:
                sys.exit(f"[FATAL] Colonnes absentes de LeagueGameLog: {miss} | recues: {list(raw.columns)}")
            wide = to_wide(raw, old_base, c, id2name)
            reconcile(wide, old_base, c)
            fresh_w = wide[~wide["game_id"].isin(old_base["game_id"])]
            raw_new = raw[raw["GAME_ID"].isin(fresh_w["api_id"])]
            fresh = fresh_w.drop(columns=["api_id"])
    else:
        log.info("Rien a recuperer (pkl plus recent que la cible).")

    need_features = len(fresh) > 0 or not all(x in old.columns for x in FEATURES_19 + NEW_COLS) or args.rebase
    if not need_features:
        log.info("A jour : aucun nouveau match, colonnes deja presentes.")
        return 0

    new_base = old_base.copy() if fresh.empty else pd.concat([old_base, fresh], ignore_index=True)
    new_base = new_base.sort_values(["date", "game_id"], kind="mergesort").reset_index(drop=True)
    F_new = compute_features(new_base, c)
    added_ids = fresh["game_id"].tolist()

    try:
        audit(old_base, new_base, F_new, added_ids, c, old, args.rebase)
    except AssertionError as e:
        log.error("%s | aucun fichier modifie.", e)
        return 1

    out = new_base.copy()
    aligned = F_new.reindex(out["game_id"])
    for col in FEATURES_19 + NEW_COLS:
        out[col] = aligned[col].to_numpy()

    if args.dry_run:
        log.info("[DRY-RUN] %d nouveaux matchs, %d -> %d lignes, rien ecrit.", len(fresh), len(old), len(out))
        return 0

    if raw_new is not None and len(raw_new):
        ledger = pd.read_pickle(LEDGER) if LEDGER.exists() else pd.DataFrame()
        ledger = pd.concat([ledger, raw_new], ignore_index=True).drop_duplicates(["GAME_ID", "TEAM_ID"])
        atomic_pickle(ledger, LEDGER)
        log.info("Ledger brut: %d lignes equipe", len(ledger))

    shutil.copy2(PKL, str(PKL) + ".bak")
    atomic_pickle(out, PKL)
    log.info("OK %s: %d -> %d matchs, %d colonnes", PKL.name, len(old), len(out), out.shape[1])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=SEASON)
    ap.add_argument("--refetch-days", type=int, default=OVERLAP_DAYS)
    ap.add_argument("--rebase", action="store_true", help="accepte la derive vs features stockees")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    setup_logging()
    if not acquire_lock():
        log.error("Autre instance en cours (verrou %s).", LOCK)
        return 2
    try:
        return run(args)
    except Exception as e:
        log.exception("Echec: %s", e)
        return 1
    finally:
        release_lock()


if __name__ == "__main__":
    sys.exit(main())
