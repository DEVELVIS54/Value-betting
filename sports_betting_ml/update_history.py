# sports_betting_ml/update_history.py
"""
Met a jour l'historique des matchs pendant la saison (sans credit Odds API) :
  1. recupere les matchs joues de la saison en cours via nba_api
  2. les ajoute a data/nba_ext.sqlite (sans doublon)
  3. reconstruit data/processed/nba_with_features.pkl (nba.sqlite + nba_ext.sqlite, 2014 -> aujourd'hui)
Le predictor relit ce fichier a chaque scan : formes, repos et back-to-back sont donc a jour.

Usage : python update_history.py                    (saison en cours)
        python update_history.py --season 2025-26   (test de l'API sur une saison deja connue)
        python update_history.py --force            (reconstruit meme sans nouveau match)
"""

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
NBA_DB = os.environ.get("NBA_DB", r"C:\Users\elvis\Desktop\archive\archive\nba.sqlite")
EXT_DB = ROOT / "data" / "nba_ext.sqlite"
PKL = ROOT / "data" / "processed" / "nba_with_features.pkl"
STATS = ["fgm", "fga", "fg3m", "ftm", "fta", "oreb", "dreb", "tov"]
SEL = ", ".join(f"{s}_{x}" for s in STATS for x in ("home", "away"))


def current_season() -> str:
    now = datetime.now()
    y = now.year if now.month >= 9 else now.year - 1
    return f"{y}-{str(y + 1)[-2:]}"


def load_team_names() -> dict:
    """Un seul nom par franchise (via l'ID d'equipe) : evite les doublons type Clippers."""
    con = sqlite3.connect("file:" + Path(NBA_DB).as_posix() + "?mode=ro", uri=True)
    df = pd.read_sql_query(
        "SELECT team_id_home AS id, team_name_home AS name FROM game "
        "WHERE game_date >= '2019-01-01' ORDER BY game_date DESC", con)
    con.close()
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    return df.dropna().drop_duplicates("id").set_index("id")["name"].to_dict()


def fetch_season(season: str, name_of: dict):
    """Retourne un DataFrame 'wide' (1 ligne par match) ou None si l'API est indisponible."""
    from nba_api.stats.endpoints import leaguegamelog

    raw = None
    for attempt in range(4):
        try:
            raw = leaguegamelog.LeagueGameLog(
                season=season, season_type_all_star="Regular Season", timeout=60
            ).get_data_frames()[0]
            break
        except Exception as exc:
            print(f"[update] essai {attempt + 1}/4 echoue : {exc}")
            time.sleep(5 * (attempt + 1))
    if raw is None:
        return None
    if raw.empty:
        return pd.DataFrame()

    raw["is_home"] = raw["MATCHUP"].str.contains("vs.", regex=False)
    h = raw[raw["is_home"]].drop_duplicates("GAME_ID").set_index("GAME_ID")
    a = raw[~raw["is_home"]].drop_duplicates("GAME_ID").set_index("GAME_ID")
    idx = h.index.intersection(a.index)
    if len(idx) == 0:
        return pd.DataFrame()

    out = pd.DataFrame({
        "game_id": [str(i) for i in idx],
        "game_date": pd.to_datetime(h.loc[idx, "GAME_DATE"]).dt.strftime("%Y-%m-%d").values,
        "team_name_home": h.loc[idx, "TEAM_ID"].map(name_of).fillna(h.loc[idx, "TEAM_NAME"]).values,
        "team_name_away": a.loc[idx, "TEAM_ID"].map(name_of).fillna(a.loc[idx, "TEAM_NAME"]).values,
    })
    for c in ["PTS"] + [s.upper() for s in STATS]:
        out[f"{c.lower()}_home"] = h.loc[idx, c].values
        out[f"{c.lower()}_away"] = a.loc[idx, c].values
    return out


def upsert(new: pd.DataFrame) -> int:
    """Ajoute les nouveaux matchs a nba_ext.sqlite (sans doublon). Retourne le nombre ajoute."""
    EXT_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(EXT_DB)
    try:
        old = pd.read_sql_query("SELECT * FROM game", con)
    except Exception:
        old = pd.DataFrame()
    n_before = len(old)
    merged = pd.concat([old, new], ignore_index=True)
    merged["game_id"] = merged["game_id"].astype(str)
    merged = merged.drop_duplicates("game_id", keep="last")
    merged.to_sql("game", con, if_exists="replace", index=False)
    con.close()
    return len(merged) - n_before


def rebuild(name_of: dict) -> pd.DataFrame:
    con = sqlite3.connect("file:" + Path(NBA_DB).as_posix() + "?mode=ro", uri=True)
    d1 = pd.read_sql_query(
        "SELECT game_id, game_date, team_id_home, team_id_away, pts_home, pts_away, "
        f"{SEL} FROM game WHERE season_type = 'Regular Season' AND game_date >= '2014-10-01'", con)
    con.close()
    for c in ("team_id_home", "team_id_away"):
        d1[c] = pd.to_numeric(d1[c], errors="coerce")
    d1["home_team"] = d1["team_id_home"].map(name_of)
    d1["away_team"] = d1["team_id_away"].map(name_of)
    d1 = d1.drop(columns=["team_id_home", "team_id_away"])

    con = sqlite3.connect(EXT_DB)
    d2 = pd.read_sql_query(
        "SELECT game_id, game_date, team_name_home AS home_team, team_name_away AS away_team, "
        f"pts_home, pts_away, {SEL} FROM game", con)
    con.close()

    d = pd.concat([d1, d2], ignore_index=True)
    d["game_id"] = d["game_id"].astype(str)
    num = [c for c in d.columns if c not in ("game_id", "game_date", "home_team", "away_team")]
    d[num] = d[num].apply(pd.to_numeric, errors="coerce")
    d = d.dropna().drop_duplicates("game_id")
    d = d[(d["fga_home"] > 0) & (d["fga_away"] > 0)]

    poss_h = d["fga_home"] - d["oreb_home"] + d["tov_home"] + 0.44 * d["fta_home"]
    poss_a = d["fga_away"] - d["oreb_away"] + d["tov_away"] + 0.44 * d["fta_away"]
    reb_h = d["oreb_home"] + d["dreb_home"]
    reb_a = d["oreb_away"] + d["dreb_away"]

    out = pd.DataFrame({
        "game_id": d["game_id"],
        "date": pd.to_datetime(d["game_date"], format="mixed").dt.normalize(),
        "home_team": d["home_team"].str.strip(),
        "away_team": d["away_team"].str.strip(),
        "home_pts": d["pts_home"],
        "away_pts": d["pts_away"],
        "pace": (poss_h + poss_a) / 2.0,
        "home_efg": (d["fgm_home"] + 0.5 * d["fg3m_home"]) / d["fga_home"],
        "away_efg": (d["fgm_away"] + 0.5 * d["fg3m_away"]) / d["fga_away"],
        "home_tov": d["tov_home"] / poss_h,
        "away_tov": d["tov_away"] / poss_a,
        "home_reb": reb_h / (reb_h + reb_a),
        "away_reb": reb_a / (reb_h + reb_a),
        "home_ft_rate": d["ftm_home"] / d["fga_home"],
        "away_ft_rate": d["ftm_away"] / d["fga_away"],
    }).replace([np.inf, -np.inf], np.nan).dropna().sort_values("date").reset_index(drop=True)

    if out["home_team"].nunique() != 30 or len(out) < 14000 or out["pace"].std() < 1:
        raise RuntimeError(
            f"Dataset incoherent ({len(out)} matchs, {out['home_team'].nunique()} equipes) : pkl inchange."
        )
    tmp = PKL.with_name("nba_with_features.tmp.pkl")
    out.to_pickle(tmp)
    os.replace(tmp, PKL)  # remplacement atomique
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=None, help="ex: 2025-26 (defaut : saison en cours)")
    ap.add_argument("--force", action="store_true", help="reconstruit le pkl meme sans nouveau match")
    args = ap.parse_args()
    season = args.season or current_season()

    name_of = load_team_names()
    new = fetch_season(season, name_of)
    if new is None:
        print("[update] API indisponible : historique inchange.")
        sys.exit(1)

    added = upsert(new) if not new.empty else 0
    print(f"[update] saison {season} : {len(new)} matchs recuperes, {added} nouveau(x) dans nba_ext.sqlite")

    if added > 0 or args.force:
        out = rebuild(name_of)
        print(
            f"[update] pkl reconstruit : {len(out)} matchs | {out['date'].min().date()} -> "
            f"{out['date'].max().date()} | {out['home_team'].nunique()} equipes"
        )
    else:
        print("[update] Aucun nouveau match : pkl inchange.")


if __name__ == "__main__":
    main()