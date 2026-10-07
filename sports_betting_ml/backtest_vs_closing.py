"""backtest_vs_closing.py : le modele apporte-t-il de l'information en plus de la cote de CLOTURE ? (+ ROI simule)

Les predictions sont faites en walk-forward : chaque saison est predite par des modeles entraines UNIQUEMENT sur les
saisons precedentes (memes reglages que la production). Aucun fichier du systeme n'est modifie, aucun reglage n'est
ajuste d'apres ce backtest.

Lancer depuis la racine du projet :
    python backtest_vs_closing.py fichier1.csv [fichier2.xlsx ...] --inspect   # structure detectee, rien d'autre
    python backtest_vs_closing.py fichier1.csv [fichier2.xlsx ...]             # backtest complet

Formats de cotes acceptes (cotes de CLOTURE) :
  - "large" : une ligne par match avec date, equipe domicile, equipe exterieure, et parmi : spread, total, moneylines ;
  - archive Sportsbook Reviews Online : deux lignes par match (colonnes VH, Team, Final, Close, ML), un fichier par saison
    dont le nom contient l'annee (ex. nba odds 2018-19.xlsx).
Si la structure n'est pas reconnue, le script affiche les colonnes et s'arrete : colle-moi cet affichage.

Regles de lecture fixees AVANT les resultats :
  Information : l'ecart apparie (modele ancre - marche) est < 0 (logloss ou MAE), > 2 erreurs-types en valeur absolue,
                et negatif dans au moins 70% des saisons.   Sinon : NEUTRE (ou NUIT si > 0 et > 2 erreurs-types).
  ROI : on ne conclut a un avantage que si ROI/erreur-type > 2 avec au moins 500 paris.
  Les prix des spreads/totaux sont supposes a 1.909 (-110) : les fichiers n'ont en general pas ces prix.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import model_train as mt  # noqa: E402
import predictor as pr  # noqa: E402
from feature_engineering import FEATURE_COLUMNS, load_training_matrix  # noqa: E402

PRICE_SPREAD_TOTAL = 1.909
SIGMA_BOUNDS = {"spreads": (10.0, 16.0), "totals": (15.0, 24.0)}
SIGMA_DEFAULT = {"spreads": 14.0, "totals": 19.0}
TOTALS_MIN_OBS = getattr(pr, "TOTALS_MIN_EVENTS", 100)
MIN_ROWS_FOR_SEASON = 300

WIDE = {
    "date": ["date", "game_date", "gamedate"],
    "home": ["home", "home_team", "team_home", "hometeam", "home_name"],
    "away": ["away", "away_team", "team_away", "awayteam", "visitor", "away_name"],
    "spread": ["spread", "close_spread", "closing_spread", "home_spread", "spread_home", "point_spread", "line"],
    "total": ["total", "close_total", "closing_total", "over_under", "ou", "total_line", "overunder"],
    "ml_home": ["ml_home", "moneyline_home", "home_ml", "home_moneyline", "homeml", "home_odds"],
    "ml_away": ["ml_away", "moneyline_away", "away_ml", "away_moneyline", "awayml", "away_odds"],
}


# ------------------------------------------------------------------ lecture des cotes
def _ncol(c) -> str:
    return re.sub(r"[^a-z0-9_]", "", str(c).strip().lower().replace(" ", "_"))


def read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"[STOP] fichier introuvable : {path}")
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        df = None
        for enc in ("utf-8", "latin-1"):
            try:
                df = pd.read_csv(p, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            sys.exit(f"[STOP] encodage illisible : {path}")
    df.columns = [_ncol(c) for c in df.columns]
    return df


def american_to_decimal(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    dec = np.where(x >= 100, 1 + x / 100.0, np.where(x <= -100, 1 + 100.0 / np.abs(x), x))
    return pd.Series(dec, index=x.index).where(lambda s: s > 1.0)


def parse_sbro(df: pd.DataFrame, path: str) -> pd.DataFrame:
    m = re.search(r"(20\d{2})[-_ ]?(?:20)?\d{2}", Path(path).name)
    if not m:
        sys.exit(f"[STOP] annee de saison introuvable dans le nom du fichier : {Path(path).name} (attendu ex. 'nba odds 2018-19.xlsx')")
    season_start = int(m.group(1))
    d = df.reset_index(drop=True)
    if len(d) % 2:
        d = d.iloc[:-1]
    v, h = d.iloc[0::2].reset_index(drop=True), d.iloc[1::2].reset_index(drop=True)
    if not (v["vh"].astype(str).str.upper().str.strip().eq("V").all() and h["vh"].astype(str).str.upper().str.strip().eq("H").all()):
        sys.exit("[STOP] format deux-lignes : l'alternance V/H n'est pas respectee")

    def num(s):
        s = s.astype(str).str.strip().str.lower().replace({"pk": "0", "nl": "nan", "none": "nan"})
        return pd.to_numeric(s, errors="coerce")

    cv, ch = num(v["close"]), num(h["close"])
    lo, hi = np.minimum(cv, ch), np.maximum(cv, ch)
    ml_v = pd.to_numeric(v.get("ml"), errors="coerce") if "ml" in v else pd.Series(np.nan, index=v.index)
    ml_h = pd.to_numeric(h.get("ml"), errors="coerce") if "ml" in h else pd.Series(np.nan, index=h.index)
    fav_by_row = ch <= cv                       # la ligne qui porte le spread (la plus petite valeur) = favori
    fav_by_ml = ml_h < ml_v
    both = ml_h.notna() & ml_v.notna()
    if both.any():
        print(f"  [{Path(path).name}] favori deduit du spread == favori deduit de la moneyline : {(fav_by_row[both] == fav_by_ml[both]).mean():.1%}")
    fav_home = np.where(both, fav_by_ml, fav_by_row)
    raw = pd.to_numeric(v["date"], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(4)
    month, day = raw.str[:-2].astype(int), raw.str[-2:].astype(int)
    year = np.where(month >= 9, season_start, season_start + 1)
    date = pd.to_datetime(dict(year=year, month=month, day=day), errors="coerce")
    return pd.DataFrame({
        "date": date, "home": h["team"].astype(str).str.strip(), "away": v["team"].astype(str).str.strip(),
        "spread": np.where(fav_home, -lo, lo), "total": hi, "ml_home": ml_h, "ml_away": ml_v,
    })


def to_canonical(df: pd.DataFrame, path: str, inspect: bool) -> pd.DataFrame:
    cols = set(df.columns)
    if {"vh", "team", "final", "close", "date"} <= cols:
        if inspect:
            print(f"  {Path(path).name} : format archive deux-lignes (VH / Team / Close / ML)")
        return parse_sbro(df, path)
    pick = {}
    for key, cands in WIDE.items():
        hit = [c for c in cands if c in cols]
        if hit:
            pick[key] = hit[0]
    needed = {"date", "home", "away"}
    has_market = "spread" in pick or "total" in pick or ("ml_home" in pick and "ml_away" in pick)
    if not (needed <= set(pick) and has_market):
        print(f"[STOP] structure non reconnue pour {Path(path).name}.\nColonnes : {list(df.columns)}\nPremieres lignes :\n{df.head(5).to_string()}")
        sys.exit(1)
    if inspect:
        print(f"  {Path(path).name} : format large, colonnes retenues : {pick}")
    out = pd.DataFrame({k: df[v] for k, v in pick.items()})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for k in ("spread", "total", "ml_home", "ml_away"):
        if k not in out:
            out[k] = np.nan
    return out[["date", "home", "away", "spread", "total", "ml_home", "ml_away"]]


# ------------------------------------------------------------------ equipes
def _key(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower().replace("los angeles", "la"))


def map_teams(names, targets) -> dict:
    tk = {t: _key(t) for t in targets}
    out, bad = {}, []
    for n in sorted(set(names)):
        k = _key(n)
        for stage in (lambda v: v == k, lambda v: v.startswith(k), lambda v: k.startswith(v), lambda v: v.endswith(k)):
            cand = [t for t, v in tk.items() if k and stage(v)]
            if len(cand) == 1:
                out[n] = cand[0]
                break
        else:
            bad.append((n, cand))
    if bad:
        sys.exit(f"[STOP] equipes sans correspondance unique : {bad}\nEquipes du systeme : {sorted(targets)}")
    return out


# ------------------------------------------------------------------ walk-forward
def wf_preds(mx: pd.DataFrame, market: str, seasons) -> pd.Series:
    kind, target, decay = mt.MARKETS[market]
    feats, out = list(FEATURE_COLUMNS), []
    for s in seasons:
        train = mx[(mx["date"] >= mt.TRAIN_START) & (mx["season"] < s)]
        valid = mx[mx["season"] == s]
        inner_tr, inner_va = train[train["season"] < s - 1], train[train["season"] == s - 1]
        if len(inner_tr) < 500 or len(inner_va) < 100 or valid.empty:
            continue
        probe = mt.make_model(kind, mt.MAX_ESTIMATORS, mt.EARLY_STOPPING_ROUNDS)
        probe.fit(inner_tr[feats], inner_tr[target], sample_weight=mt.recency_weights(inner_tr["season"], decay),
                  eval_set=[(inner_va[feats], inner_va[target])], verbose=False)
        best_n = max(int(probe.best_iteration) + 1, mt.MIN_FINAL_ESTIMATORS)
        model = mt.make_model(kind, best_n)
        model.fit(train[feats], train[target], sample_weight=mt.recency_weights(train["season"], decay), verbose=False)
        p = model.predict_proba(valid[feats])[:, 1] if kind == "classifier" else model.predict(valid[feats])
        out.append(pd.Series(p, index=valid.index))
        print(f"    {market:8s} saison {s}-{str(s + 1)[-2:]} : {len(valid)} matchs predits ({best_n} arbres)", flush=True)
    return pd.concat(out) if out else pd.Series(dtype=float)


# ------------------------------------------------------------------ statistiques
def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    return (float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 else (float("nan"), float("nan"))


def verdict(per_season_delta, delta, se):
    if np.isnan(delta):
        return "n/a"
    neg = float(np.mean([x < 0 for x in per_season_delta])) if per_season_delta else 0.0
    if delta < 0 and abs(delta) > 2 * se and neg >= 0.7:
        return f"INFORMATION (negatif dans {neg:.0%} des saisons)"
    if delta > 0 and abs(delta) > 2 * se:
        return "NUIT (le melange est moins bon que le marche seul)"
    return "NEUTRE (ecart dans le bruit)"


def expanding_prior_mean(values: np.ndarray, dates: pd.Series, min_prior: int) -> np.ndarray:
    """Moyenne des valeurs des matchs a date STRICTEMENT anterieure ; NaN si moins de min_prior matchs."""
    cs = np.concatenate([[0.0], np.cumsum(values)])
    first = pd.Series(np.arange(len(values))).groupby(dates.to_numpy()).transform("min").to_numpy()
    mean = cs[first] / np.maximum(first, 1)
    return np.where(first >= min_prior, mean, np.nan)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    mx = load_training_matrix().reset_index(drop=True)
    mx["date"] = pd.to_datetime(mx["date"]).astype("datetime64[ns]")
    teams = sorted(set(mx["home_team"]) | set(mx["away_team"]))

    print("Lecture des cotes :")
    odds = pd.concat([to_canonical(read_any(f), f, args.inspect) for f in args.files], ignore_index=True)
    odds = odds.dropna(subset=["date"])
    mapping = map_teams(set(odds["home"]) | set(odds["away"]), teams)
    odds["home_team"], odds["away_team"] = odds["home"].map(mapping), odds["away"].map(mapping)
    odds["date"] = pd.to_datetime(odds["date"]).astype("datetime64[ns]")
    print(f"  {len(odds)} lignes de cotes | {odds['date'].min().date()} -> {odds['date'].max().date()} | equipes mappees : {len(mapping)}")
    if args.inspect:
        print(odds.head(5).to_string())
        print("\n--inspect : aucune autre operation.")
        return

    # --- jointure avec la matrice (tolerance +/- 1 jour sur la date : fuseaux horaires)
    keys = ["date", "home_team", "away_team"]
    mxk = mx[keys].assign(_mx=mx.index)
    remaining = odds.copy()
    remaining["_oid"] = np.arange(len(remaining))
    parts = []
    for shift in (0, -1, 1):
        t = remaining.copy()
        t["date"] = t["date"] + pd.Timedelta(days=shift)
        m = t.merge(mxk, on=keys, how="inner").drop_duplicates(subset=["_oid"]).drop_duplicates(subset=["_mx"])
        parts.append(m)
        remaining = remaining[~remaining["_oid"].isin(m["_oid"])]
    jd = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["_mx"])
    jd = jd.merge(mx[["home_pts", "away_pts", "season"]], left_on="_mx", right_index=True, how="left").reset_index(drop=True)
    print(f"  jointure : {len(jd)} matchs apparies sur {len(odds)} lignes de cotes ({len(jd) / max(len(odds), 1):.1%})")
    if len(jd) < 0.9 * len(odds):
        print("  ATTENTION : moins de 90% de correspondance (dates ou noms d'equipes ?)")

    jd["act_margin"] = jd["home_pts"] - jd["away_pts"]
    jd["act_total"] = jd["home_pts"] + jd["away_pts"]
    jd["home_win"] = (jd["act_margin"] > 0).astype(int)

    # --- controles d'orientation (signe) a partir des resultats reels
    if jd["spread"].notna().any():
        c = np.corrcoef((-jd["spread"]).dropna(), jd.loc[jd["spread"].notna(), "act_margin"])[0, 1]
        if c < 0:
            jd["spread"] = -jd["spread"]
            print(f"  spread : correlation negative ({c:+.2f}) -> signe inverse")
        else:
            print(f"  spread : correlation marge de marche / marge reelle = {c:+.2f} (orientation OK)")
    jd["mkt_margin"] = -jd["spread"]
    dh, da = american_to_decimal(jd["ml_home"]), american_to_decimal(jd["ml_away"])
    jd["dec_home"], jd["dec_away"] = dh, da
    jd["p_mkt"] = (1 / dh) / (1 / dh + 1 / da)
    if jd["p_mkt"].notna().any():
        c = np.corrcoef(jd["p_mkt"].dropna(), jd.loc[jd["p_mkt"].notna(), "home_win"])[0, 1]
        if c < 0:
            jd["p_mkt"] = 1 - jd["p_mkt"]
            jd["dec_home"], jd["dec_away"] = da, dh
            print(f"  moneyline : correlation negative ({c:+.2f}) -> domicile/exterieur inverses")
        else:
            print(f"  moneyline : correlation proba de marche / victoire = {c:+.2f} (orientation OK)")

    jd = jd.sort_values(["date", "_mx"]).reset_index(drop=True)
    seasons = sorted(s for s, g in jd.groupby("season") if len(g) >= MIN_ROWS_FOR_SEASON and s >= mx["season"].min() + 2)
    print(f"\nSaisons evaluees (walk-forward, entrainement = saisons precedentes) : {seasons}")
    if not seasons:
        sys.exit("[STOP] aucune saison evaluable (il faut >= 2 saisons d'historique avant la premiere saison de cotes)")

    print("Predictions walk-forward :")
    pred = {m: wf_preds(mx, m, seasons) for m in ("h2h", "spreads", "totals")}
    jd["m_p"] = jd["_mx"].map(pred["h2h"])
    jd["m_margin"] = jd["_mx"].map(pred["spreads"])
    jd["m_total"] = jd["_mx"].map(pred["totals"])
    jd = jd[jd["season"].isin(seasons)].dropna(subset=["m_p", "m_margin", "m_total"]).reset_index(drop=True)

    W, MD = pr.W_MODEL, pr.MAX_DEV
    results = {}

    # ================= H2H : logloss
    h = jd.dropna(subset=["p_mkt"]).copy()
    if len(h):
        pm = h["p_mkt"].clip(pr.PROB_MIN, pr.PROB_MAX)
        pmod = h["m_p"].clip(pr.PROB_MIN, pr.PROB_MAX)
        pan = (pm + W["h2h"] * (pmod - pm).clip(-MD["h2h"], MD["h2h"])).clip(pr.PROB_MIN, pr.PROB_MAX)
        ll = lambda p: -(h["home_win"] * np.log(p) + (1 - h["home_win"]) * np.log(1 - p))
        h["ll_mkt"], h["ll_mod"], h["ll_anc"] = ll(pm), ll(pmod), ll(pan)
        print("\n=== H2H : logloss (plus bas = mieux) ===")
        rows = []
        for s, g in h.groupby("season"):
            d, se = paired(g["ll_anc"], g["ll_mkt"])
            rows.append(dict(saison=s, n=len(g), marche=g["ll_mkt"].mean(), modele=g["ll_mod"].mean(), ancre=g["ll_anc"].mean(), d_ancre_marche=d, se=se))
        print(pd.DataFrame(rows).round(4).to_string(index=False))
        dm, sm = paired(h["ll_mod"], h["ll_mkt"])
        da_, sa = paired(h["ll_anc"], h["ll_mkt"])
        print(f"TOUS : modele - marche = {dm:+.4f} +/- {sm:.4f} | ancre - marche = {da_:+.4f} +/- {sa:.4f}")
        print("Verdict H2H :", verdict([r["d_ancre_marche"] for r in rows], da_, sa))
        print(f"Biais moyen (modele - marche) : {(h['m_p'] - h['p_mkt']).mean() * 100:+.2f} pts de proba")

    # ================= SPREADS : MAE de la marge
    sp = jd.dropna(subset=["mkt_margin"]).copy()
    if len(sp):
        mkt, mod = sp["mkt_margin"], sp["m_margin"]
        anc = mkt + W["spreads"] * (mod - mkt).clip(-MD["spreads"], MD["spreads"])
        sp["e_mkt"], sp["e_mod"], sp["e_anc"] = (sp["act_margin"] - mkt).abs(), (sp["act_margin"] - mod).abs(), (sp["act_margin"] - anc).abs()
        sp["m_anc"] = anc
        print("\n=== SPREADS : MAE de la marge (plus bas = mieux) ===")
        rows = []
        for s, g in sp.groupby("season"):
            d, se = paired(g["e_anc"], g["e_mkt"])
            rows.append(dict(saison=s, n=len(g), marche=g["e_mkt"].mean(), modele=g["e_mod"].mean(), ancre=g["e_anc"].mean(),
                             d_ancre_marche=d, se=se, biais_mod_moins_marche=(g["m_margin"] - g["mkt_margin"]).mean()))
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        dm, sm = paired(sp["e_mod"], sp["e_mkt"])
        da_, sa = paired(sp["e_anc"], sp["e_mkt"])
        print(f"TOUS : modele - marche = {dm:+.3f} +/- {sm:.3f} | ancre - marche = {da_:+.3f} +/- {sa:.3f}")
        print("Verdict Spreads :", verdict([r["d_ancre_marche"] for r in rows], da_, sa))

    # ================= TOTALS : MAE, avec recentrage live (moyenne des ecarts modele - marche sur les matchs precedents)
    tt = jd.dropna(subset=["total"]).sort_values(["date", "_mx"]).reset_index(drop=True)
    if len(tt):
        dev = (tt["m_total"] - tt["total"]).to_numpy()
        bias = expanding_prior_mean(dev, tt["date"], TOTALS_MIN_OBS)
        tt["bias_prior"] = bias
        cen = tt["m_total"] - np.nan_to_num(bias, nan=0.0)
        tt["m_total_c"] = cen
        anc = tt["total"] + W["totals"] * (cen - tt["total"]).clip(-MD["totals"], MD["totals"])
        tt["m_tot_anc"] = anc
        tt["e_mkt"], tt["e_raw"] = (tt["act_total"] - tt["total"]).abs(), (tt["act_total"] - tt["m_total"]).abs()
        tt["e_cen"], tt["e_anc"] = (tt["act_total"] - cen).abs(), (tt["act_total"] - anc).abs()
        obs = tt[tt["bias_prior"].notna()]
        print(f"\n=== TOTALS : MAE du total, apres les {TOTALS_MIN_OBS} premiers matchs (recentrage live actif) : {len(obs)} matchs ===")
        rows = []
        for s, g in obs.groupby("season"):
            d, se = paired(g["e_anc"], g["e_mkt"])
            rows.append(dict(saison=s, n=len(g), marche=g["e_mkt"].mean(), modele_brut=g["e_raw"].mean(), recentre=g["e_cen"].mean(),
                             ancre=g["e_anc"].mean(), d_ancre_marche=d, se=se, biais_brut=(g["m_total"] - g["total"]).mean(),
                             biais_recentre=(g["m_total_c"] - g["total"]).mean()))
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        da_, sa = paired(obs["e_anc"], obs["e_mkt"])
        print(f"TOUS : ancre - marche = {da_:+.3f} +/- {sa:.3f}")
        print("Verdict Totals :", verdict([r["d_ancre_marche"] for r in rows], da_, sa))

    # ================= ROI simule aux prix de cloture (un pari par marche et par match, regles de production)
    print("\n=== ROI simule aux prix de cloture (mise 1 ; spreads/totaux a 1.909 suppose) ===")
    bets = []
    sig = {"spreads": SIGMA_DEFAULT["spreads"], "totals": SIGMA_DEFAULT["totals"]}
    sigma_by_season = {}
    for s in seasons:                          # sigma = ecart-type des residus des saisons PRECEDENTES uniquement
        prev = jd[jd["season"] < s]
        sigma_by_season[s] = {
            "spreads": float(np.clip((prev["act_margin"] - prev["m_margin"]).std() if len(prev) > 300 else SIGMA_DEFAULT["spreads"], *SIGMA_BOUNDS["spreads"])),
            "totals": float(np.clip((prev["act_total"] - prev["m_total"]).std() if len(prev) > 300 else SIGMA_DEFAULT["totals"], *SIGMA_BOUNDS["totals"])),
        }

    def add(market, season, date, prob, odds, outcome):
        ev = (prob * odds - 1) * 100
        if pr.min_ev_threshold(odds) <= ev <= pr.MAX_EV_PCT:
            profit = 0.0 if outcome == "push" else (odds - 1 if outcome == "win" else -1.0)
            bets.append(dict(market=market, season=season, date=date, ev=ev, profit=profit))

    for r in h.itertuples() if len(h) else []:
        pm_ = min(max(r.p_mkt, pr.PROB_MIN), pr.PROB_MAX)
        p = min(max(pm_ + W["h2h"] * float(np.clip(r.m_p - pm_, -MD["h2h"], MD["h2h"])), pr.PROB_MIN), pr.PROB_MAX)
        cands = [(p, r.dec_home, "win" if r.home_win == 1 else "loss"), (1 - p, r.dec_away, "win" if r.home_win == 0 else "loss")]
        p_, o_, out_ = max(cands, key=lambda c: c[0] * c[1])
        add("h2h", r.season, r.date, p_, o_, out_)
    for r in sp.itertuples() if len(sp) else []:
        sg = sigma_by_season[r.season]["spreads"]
        mu = r.m_anc
        s_home = r.spread
        cands = [(float(norm.cdf((mu + s_home) / sg)), r.act_margin + s_home),
                 (float(norm.cdf((-s_home - mu) / sg)), -(r.act_margin + s_home))]
        best = max(cands, key=lambda c: c[0])
        p_, margin_ = best
        outcome = "push" if margin_ == 0 else ("win" if margin_ > 0 else "loss")
        add("spreads", r.season, r.date, min(max(p_, pr.PROB_MIN), pr.PROB_MAX), PRICE_SPREAD_TOTAL, outcome)
    for r in tt[tt["bias_prior"].notna()].itertuples() if len(tt) else []:
        sg = sigma_by_season[r.season]["totals"]
        p_over = float(1 - norm.cdf((r.total - r.m_tot_anc) / sg))
        if p_over >= 0.5:
            p_, d_ = p_over, r.act_total - r.total
        else:
            p_, d_ = 1 - p_over, r.total - r.act_total
        outcome = "push" if d_ == 0 else ("win" if d_ > 0 else "loss")
        add("totals", r.season, r.date, min(max(p_, pr.PROB_MIN), pr.PROB_MAX), PRICE_SPREAD_TOTAL, outcome)

    if not bets:
        print("Aucun pari ne passe les regles de production (seuil d'EV, plafond 12%).")
        return
    b = pd.DataFrame(bets)
    n_games = len(jd)
    rows = []
    for name, g in list(b.groupby("market")) + [("TOUS", b)]:
        roi = g["profit"].mean() * 100
        se = g["profit"].std(ddof=1) / np.sqrt(len(g)) * 100 if len(g) > 2 else float("nan")
        rows.append(dict(marche=name, paris=len(g), part_des_matchs=f"{len(g) / n_games:.1%}", ROI_pct=roi, erreur_type=se,
                         t=roi / se if se and se > 0 else float("nan"), EV_moy_pct=g["ev"].mean()))
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    by_s = b.groupby("season")["profit"].agg(["count", "mean"]).rename(columns={"count": "paris", "mean": "ROI"})
    by_s["ROI"] = (by_s["ROI"] * 100).round(2)
    print("\nROI par saison (%) :\n" + by_s.to_string())
    tot = rows[-1]
    print(f"\nLecture : avantage seulement si t > 2 avec >= 500 paris. Ici : {tot['paris']} paris, ROI {tot['ROI_pct']:+.2f}% (t = {tot['t']:.2f}).")
    print("Rappel : le meilleur prix parmi ~10 bookmakers (non simule ici) et l'information tardive (blessures) ne sont pas representes.")


if __name__ == "__main__":
    main()
