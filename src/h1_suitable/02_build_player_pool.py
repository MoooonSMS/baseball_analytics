"""Phase 1: 선수-팀-시즌 지표(wOBA/wRC+), 후보 풀, 적합 인재 S, N (사후 + ex-ante + 민감도 변형).

입력
  KBO  data/raw/kbo_relay_events_{y}.csv (2020~2025) -> 타석 테이블(h1_kbo_pa_events.pkl 캐시)
       data/raw/kbo_boxscore_batters_{y}.csv          -> 선수 식별(playerCode)·팀·득점·파크팩터
       (KBO 공식기록실은 자동수집 금지라 비규정 포함 전체 타자 기록을 relay에서 재구성)
  MLB  data/raw/mlb30_pbp_{y}.csv, mlb30_boxscore_batters_{y}.csv, mlb30_players_hitting_all_{y}.csv
지표: 두 리그 모두 RE24 선형가중치(1~9회) -> 리그 wOBA=리그 OBP로 scale -> wRC+(파크팩터 없음/있음)

출력
  data/processed/h1_players.csv, h1_team_n.csv, h1_league_constants.csv
  outputs/h1_suitable/phase1_*.csv, phase1_results.json, phase1_n_hist.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import h1_common as H  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KBO_YEARS = [C.KBO_EXANTE_PRIOR] + C.SEASONS
MLB_YEARS = [y for y in C.SEASONS if (C.RAW / f"mlb30_pbp_{y}.csv").exists()]
RES: dict = {}
COUNT_COLS = ["pa", "ab", "h", "d2", "d3", "hr", "bb", "ibb", "hbp", "sf"]


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ======================================================================
# 데이터 적재
# ======================================================================
def kbo_pa() -> pd.DataFrame:
    cache = C.PROC / "h1_kbo_pa_events.pkl"
    pa = pd.read_pickle(cache) if cache.exists() else pd.DataFrame(columns=["year"])
    have = set(pa["year"].unique())
    mtime = cache.stat().st_mtime if cache.exists() else 0
    # 캐시에 없거나, relay 파일이 캐시보다 새로우면(수집 중이던 시즌 등) 그 시즌만 다시 만든다
    need = [y for y in KBO_YEARS if (C.RAW / f"kbo_relay_events_{y}.csv").exists()
            and (y not in have or (C.RAW / f"kbo_relay_events_{y}.csv").stat().st_mtime > mtime)]
    if need:
        pa = pa[~pa["year"].isin(need)]
        box = H.load_kbo_box(need)
        new = H.attach_kbo_player(pd.concat([H.kbo_pa_events(y) for y in need], ignore_index=True), box)
        pa = pd.concat([pa, new], ignore_index=True) if len(pa) else new
        pa.to_pickle(cache)
    return pa[pa["year"].isin(KBO_YEARS)]


def totals_from_pa(pa: pd.DataFrame, runs: pd.Series) -> pd.DataFrame:
    """타석 테이블 -> 리그-시즌 합계(표준 컬럼). runs: year별 리그 득점."""
    lines = H.kbo_lines_from_relay(pa.assign(playerCode=pa["playerCode"].fillna(-1),
                                             team=pa["team"].fillna("?"), name=pa["name"].fillna("?"),
                                             batter=pa["batter"].fillna("?")) if "batter" in pa
                                   else pa)
    t = lines.groupby("year")[COUNT_COLS + ["so"]].sum().reset_index()
    t["r"] = t["year"].map(runs)
    return t


def mlb_pa_frame() -> pd.DataFrame:
    return pd.concat([H.load_mlb_pa(y) for y in MLB_YEARS], ignore_index=True)


def mlb_players() -> pd.DataFrame:
    d = pd.concat([pd.read_csv(C.RAW / f"mlb30_players_hitting_all_{y}.csv", encoding="utf-8-sig")
                   for y in MLB_YEARS], ignore_index=True)
    d = d.fillna({c: 0 for c in ["plateAppearances", "atBats", "hits", "doubles", "triples", "homeRuns",
                                 "baseOnBalls", "intentionalWalks", "hitByPitch", "sacFlies", "strikeOuts"]})
    d = d.rename(columns={"playerId": "pid", "plateAppearances": "pa", "atBats": "ab", "hits": "h",
                          "doubles": "d2", "triples": "d3", "homeRuns": "hr", "baseOnBalls": "bb",
                          "intentionalWalks": "ibb", "hitByPitch": "hbp", "sacFlies": "sf",
                          "strikeOuts": "so"})
    return d[["year", "team", "pid", "name"] + COUNT_COLS + ["so"]]


def mlb_box() -> pd.DataFrame:
    return pd.concat([pd.read_csv(C.RAW / f"mlb30_boxscore_batters_{y}.csv", encoding="utf-8-sig")
                      .drop_duplicates(["gameId", "playerCode", "batOrder"]).assign(year=y)
                      for y in MLB_YEARS], ignore_index=True)


def park_factor(box: pd.DataFrame, years) -> pd.Series:
    """팀별(홈구장) 단순 파크팩터: (홈 R+RA)/(원정 R+RA), 홈 절반 보정, 지정 시즌 통합."""
    g = box[box["year"].isin(years)].drop_duplicates(["gameId", "team"]).copy()
    g["rs_ra"] = g["teamScore"] + g["oppScore"]
    t = g.groupby(["team", "homeAway"])["rs_ra"].mean().unstack()
    return 1 + (t["home"] / t["away"] - 1) / 2


# ======================================================================
# 지표 계산
# ======================================================================
def league_constants(name: str, pa: pd.DataFrame, totals: pd.DataFrame):
    lw = H.re24_runvalues(pa[pa["outcome"].notna()], C.RE24_MAX_INNING)
    w, lg = H.rescale_from_runvalues(lw, totals)
    out = pd.concat([lw.add_prefix("lw_"), w.add_prefix("w_"), lg.drop(columns="lw_out")], axis=1)
    out["league"] = name
    return w, lg, out


def add_metrics(d: pd.DataFrame, w: pd.DataFrame, lg: pd.DataFrame, pf: pd.Series | None,
                suffix: str = "") -> pd.DataFrame:
    d = H.add_basic(d)
    d[f"woba{suffix}"] = H.woba(d, w)
    d[f"wrc{suffix}"] = H.wrc_plus(d[f"woba{suffix}"], d["year"], lg)
    if pf is not None:
        rppa = d["year"].map(lg["rppa"])
        p = d["team"].map(pf).fillna(1.0)
        wraa_pa = (d[f"woba{suffix}"] - d["year"].map(lg["lg_woba"])) / d["year"].map(lg["scale"])
        d[f"wrc_pf{suffix}"] = (wraa_pa + rppa + (rppa - p * rppa)) / rppa * 100
    return d


def league_rates(d: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    """리그-시즌 PA가중 평균 OBP·BB%·(OBP-AVG) — 전 타석 기준."""
    t = H.add_basic(totals.set_index("year", drop=False))
    return pd.DataFrame({"obp": t["obp"], "bb_pct": t["bb_pct"], "obp_minus_avg": t["obp_minus_avg"]})


# ======================================================================
# N 판정
# ======================================================================
MEMBER_COLS = ["year", "team", "pid", "name", "pa", "rank", "S"]


def compute_n(d: pd.DataFrame, lgr: pd.DataFrame, pool_pa=C.POOL_MIN_PA, prod="wrc",
              onbase="obp", rule="league_mean", q=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """d: 선수-팀-시즌. 반환: (팀-시즌 N, 선수 S 플래그가 붙은 후보 풀)."""
    p = d[d["pa"] >= pool_pa].copy()
    p["rank"] = p.groupby(["year", "team"])[prod].rank(ascending=False, method="first")
    if rule == "league_mean":
        thr = p["year"].map(lgr[onbase])
    else:  # 리그-시즌 후보 풀 분위수
        thr = p.groupby("year")[onbase].transform(lambda s: s.quantile(q))
    p["S"] = (p["rank"] <= C.TOP_K) & (p[onbase] >= thr)
    teams = d.groupby(["year", "team"]).size().index.to_frame(index=False)
    t = p.groupby(["year", "team"]).agg(pool=("pid", "size"), N=("S", "sum")).reset_index()
    t = teams.merge(t, on=["year", "team"], how="left").fillna({"pool": 0, "N": 0})
    return t, p


def compute_n_exante(d: pd.DataFrame, lgr: pd.DataFrame, years) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """전년도 PA>=300(전 팀 합산) 선수의 전년도 wRC+/OBP로 S 판정, 해당 시즌 팀 구성원 기준.
    반환: (팀-시즌 N, 제외 비율, 선수 S 플래그)"""
    if not years:
        return (pd.DataFrame(columns=["year", "team", "pool", "N"]),
                pd.DataFrame(columns=["year", "pool_players", "no_prior", "no_prior_share", "no_prior_pa_share"]),
                pd.DataFrame(columns=MEMBER_COLS))
    prior = d.groupby(["year", "pid"]).apply(
        lambda g: pd.Series({"pa_prev": g["pa"].sum(),
                             "wrc_prev": np.average(g["wrc"], weights=g["pa"]) if g["pa"].sum() else np.nan,
                             "obp_prev": np.average(g["obp"].fillna(0), weights=g["pa"]) if g["pa"].sum() else np.nan}),
        include_groups=False).reset_index()
    prior["year"] += 1
    prior = prior[prior["pa_prev"] >= C.EXANTE_MIN_PA]
    cur = d[d["year"].isin(years) & (d["pa"] >= C.EXANTE_MEMBER_MIN_PA)].merge(prior, on=["year", "pid"], how="left")
    cur["lg_obp_prev"] = (cur["year"] - 1).map(lgr["obp"])
    e = cur[cur["wrc_prev"].notna()].copy()
    e["rank"] = e.groupby(["year", "team"])["wrc_prev"].rank(ascending=False, method="first")
    e["S"] = (e["rank"] <= C.TOP_K) & (e["obp_prev"] >= e["lg_obp_prev"])
    teams = cur.groupby(["year", "team"]).size().index.to_frame(index=False)
    t = e.groupby(["year", "team"]).agg(pool=("pid", "size"), N=("S", "sum")).reset_index()
    t = teams.merge(t, on=["year", "team"], how="left").fillna({"pool": 0, "N": 0})
    # 제외 비율: 해당 시즌 후보풀(PA>=300) 중 전년도 PA>=300 기록이 없는 선수
    pool = cur[cur["pa"] >= C.POOL_MIN_PA]
    excl = pool.groupby("year").apply(lambda g: pd.Series({
        "pool_players": len(g), "no_prior": int(g["wrc_prev"].isna().sum()),
        "no_prior_share": g["wrc_prev"].isna().mean(),
        "no_prior_pa_share": g.loc[g["wrc_prev"].isna(), "pa"].sum() / g["pa"].sum()}),
        include_groups=False).reset_index()
    return t, excl, e


def variants(league: str, d: pd.DataFrame, lgr: pd.DataFrame, years) -> dict:
    """{변형명: (팀-시즌 N, 선수 S 플래그)}"""
    d = d[d["year"].isin(years)]
    v = {"primary": compute_n(d, lgr)}
    for pa_ in C.POOL_MIN_PA_SENS:
        v[f"pool_pa{pa_}"] = compute_n(d, lgr, pool_pa=pa_)
    for k, q in C.OBP_RULE_SENS.items():
        v[f"obp_{k}"] = compute_n(d, lgr, rule="quantile", q=q)
    for ob in C.ONBASE_ALT:
        v[f"onbase_{ob}"] = compute_n(d, lgr, onbase=ob)
    v["prod_woba"] = compute_n(d, lgr, prod="woba")
    v["prod_wrc_pf"] = compute_n(d, lgr, prod="wrc_pf")
    alt = "wrc_alt"
    if alt in d:
        v["weights_alt"] = compute_n(d.assign(wrc=d[alt]), lgr)
    return v


def n_table(v: dict, league: str) -> pd.DataFrame:
    rows = []
    for k, t in v.items():
        vc = t["N"].astype(int).value_counts().reindex(range(4), fill_value=0)
        rows.append({"league": league, "variant": k, "team_seasons": len(t),
                     **{f"N={i}": int(vc[i]) for i in range(4)}, "mean_N": t["N"].mean(),
                     "pool_lt3": int((t["pool"] < 3).sum())})
    return pd.DataFrame(rows)


# ======================================================================
# 검증
# ======================================================================
def validate_kbo_lines(lines: pd.DataFrame):
    off = H.load_kbo_hitters()
    cols = ["pa", "ab", "h", "d2", "d3", "hr", "bb", "ibb", "hbp", "sf", "so"]
    lp = lines.groupby(["year", "team", "name_full"])[cols].sum().reset_index()
    m = off.merge(lp, left_on=["year", "team", "name"], right_on=["year", "team", "name_full"],
                  suffixes=("_off", ""))
    traded = (m["pa"] - m["pa_off"]).abs() > 30  # 공식기록은 최종 소속팀에 합산
    out = {"official_qualified": len(off), "matched": len(m), "traded_or_split_excluded": int(traded.sum())}
    mm = m[~traded]
    for c in cols:
        dif = mm[c] - mm[f"{c}_off"]
        out[c] = {"exact": float((dif == 0).mean()), "within1": float((dif.abs() <= 1).mean()),
                  "mean_diff": float(dif.mean())}
    return out, m


def main():
    C.OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ KBO
    section("KBO: relay -> 전체 타자 기록, RE24 가중치")
    pa = kbo_pa()
    yrs = sorted(pa["year"].unique())
    box = H.load_kbo_box(yrs)
    runs = box.drop_duplicates(["gameId", "team"]).groupby("year")["teamScore"].sum()
    RES["kbo_years"] = [int(y) for y in yrs]
    RES["kbo_match"] = pa.groupby("year")["match"].value_counts(normalize=True).unstack().round(4).to_dict("index")
    relay_r = pa.groupby("year")["runs"].sum()
    RES["kbo_relay_runs_vs_box"] = pd.DataFrame({"relay": relay_r, "box": runs}).to_dict("index")
    print("매칭:", RES["kbo_match"])
    print("득점 relay vs box:", RES["kbo_relay_runs_vs_box"])
    lines = H.kbo_lines_from_relay(pa)
    lines = lines.rename(columns={"playerCode": "pid"})
    tot = totals_from_pa(pa, runs)
    w, lg, kconst = league_constants("KBO", pa, tot)
    pf = park_factor(box, C.SEASONS)
    k = add_metrics(lines, w, lg, pf)
    # 교차검증: (a) statiz 상수 재스케일 (2021~2025만)
    lw_a = H.statiz_runvalues(tot[tot["year"].isin(C.SEASONS)])
    w_a, lg_a = H.rescale_from_runvalues(lw_a, tot[tot["year"].isin(C.SEASONS)])
    ka = k[k["year"].isin(C.SEASONS)]
    k.loc[ka.index, "woba_alt"] = H.woba(ka, w_a)
    k.loc[ka.index, "wrc_alt"] = H.wrc_plus(k.loc[ka.index, "woba_alt"], ka["year"], lg_a)
    k["league"] = "KBO"
    val, _ = validate_kbo_lines(lines)
    RES["kbo_lines_vs_official"] = val
    print("공식기록(규정타석) 대조:", {c: round(v["within1"], 3) for c, v in val.items() if isinstance(v, dict)})
    kq = k[k["pa"] >= 300]
    RES["kbo_rank_corr_re24_vs_statiz"] = float(stats.spearmanr(kq["woba"], kq["woba_alt"], nan_policy="omit")[0])
    kl = league_rates(k, tot)

    # ------------------------------------------------------------ MLB
    section("MLB: pbp RE24 가중치, 30팀 전체 타자")
    mpa = mlb_pa_frame()
    mbox = mlb_box()
    mruns = mbox.drop_duplicates(["gameId", "team"]).groupby("year")["teamScore"].sum()
    games = mpa.groupby("year")["gameId"].nunique()
    RES["mlb_games"] = games.to_dict()
    pr = mpa.groupby(["gameId", "batTeam"])["runs"].sum().rename_axis(["gameId", "team"])
    br = mbox.drop_duplicates(["gameId", "team"]).set_index(["gameId", "team"])["teamScore"]
    RES["mlb_pbp_runs_match_share"] = float(((pr - br).dropna() == 0).mean())
    mp = mlb_players()
    mtot = mp.groupby("year")[COUNT_COLS + ["so"]].sum().reset_index()
    mtot["r"] = mtot["year"].map(mruns)
    mw, mlg, mconst = league_constants("MLB", mpa, mtot)
    mpf = park_factor(mbox, MLB_YEARS)
    m = add_metrics(mp, mw, mlg, mpf)
    fg = H.mlb_weights()
    m["woba_alt"] = H.woba(m, fg)
    m["wrc_alt"] = H.wrc_plus(m["woba_alt"], m["year"],
                              mlg.assign(lg_woba=H.woba(mtot, fg).values))
    m["league"] = "MLB"
    mq = m[m["pa"] >= 300]
    RES["mlb_rank_corr_re24_vs_fangraphs"] = float(stats.spearmanr(mq["woba"], mq["woba_alt"])[0])
    ml = league_rates(m, mtot)

    const = pd.concat([kconst, mconst]).reset_index(names="year")
    const.to_csv(C.PROC / "h1_league_constants.csv", index=False, encoding="utf-8-sig")
    print(const[["league", "year", "lw_bb", "lw_1b", "lw_2b", "lw_hr", "lw_out", "scale", "lg_woba", "rppa"]]
          .round(3).to_string(index=False))

    # ------------------------------------------------------------ 검증 기준(§3-2)
    section("§3-2 검증")
    vrows = []
    for name, d, lgc, tt in [("KBO", k, lg, tot), ("MLB", m, mlg, mtot)]:
        for y, g in d.groupby("year"):
            g = g.dropna(subset=["wrc"])
            q = g[g["pa"] >= 300]
            vrows.append({"league": name, "year": y, "all_pa_weighted_wrc": np.average(g["wrc"], weights=g["pa"]),
                          "pool300_pa_weighted_wrc": np.average(q["wrc"], weights=q["pa"]),
                          "lg_woba": lgc.loc[y, "lg_woba"], "lg_obp": lgc.loc[y, "lg_obp"],
                          "scale": lgc.loc[y, "scale"], "pool300_n": len(q)})
    vt = pd.DataFrame(vrows)
    print(vt.round(3).to_string(index=False))
    stars = k.merge(pd.DataFrame(C.STAR_CHECKS, columns=["name_full", "year"]), on=["name_full", "year"])
    print(stars[["year", "name_full", "team", "pa", "obp", "slg", "woba", "wrc", "wrc_alt"]].round(3).to_string(index=False))
    RES["validation"] = {
        "pa_weighted_wrc": vt.round(3).to_dict("records"),
        "pass_100pm3": bool((vt["all_pa_weighted_wrc"] - 100).abs().le(C.WRC_PLUS_TOL).all()),
        "stars": stars[["year", "name_full", "wrc", "wrc_alt"]].round(1).to_dict("records"),
        "rank_corr_kbo_methods": RES["kbo_rank_corr_re24_vs_statiz"],
        "rank_corr_mlb_methods": RES["mlb_rank_corr_re24_vs_fangraphs"]}
    vt.to_csv(C.OUT / "phase1_validation_wrc.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------ 선수 테이블
    keep = ["league", "year", "team", "pid", "name", "pa", "ab", "h", "d2", "d3", "hr", "bb", "ibb", "hbp", "sf",
            "obp", "avg", "slg", "iso", "bb_pct", "obp_minus_avg", "woba", "wrc", "wrc_pf", "woba_alt", "wrc_alt"]
    k["name"] = k["name_full"]
    players = pd.concat([k[keep], m[keep]], ignore_index=True)
    players.to_csv(C.PROC / "h1_players.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------ N
    section("N 분포 (사후 + 민감도 + ex-ante)")
    kv = variants("KBO", k, kl, C.SEASONS)
    mv = variants("MLB", m, ml, MLB_YEARS)
    kx, kex, kxm = compute_n_exante(k, kl, C.SEASONS)
    mx, mex, mxm = compute_n_exante(m, ml, [y for y in MLB_YEARS if y not in C.MLB_EXANTE_SKIP])
    kv["exante"], mv["exante"] = (kx, kxm), (mx, mxm)
    # 선수 단위 S 플래그(Phase 2 배치 분석용): 변형별 팀 내 순위 상위 K명
    members = pd.concat([p.loc[p["rank"] <= C.TOP_K, MEMBER_COLS].assign(league=lgn, variant=var)
                         for lgn, vv in [("KBO", kv), ("MLB", mv)] for var, (_, p) in vv.items()],
                        ignore_index=True)
    members.to_csv(C.PROC / "h1_s_members.csv", index=False, encoding="utf-8-sig")
    kv = {var: t for var, (t, _) in kv.items()}
    mv = {var: t for var, (t, _) in mv.items()}
    nt = pd.concat([n_table(kv, "KBO"), n_table(mv, "MLB")], ignore_index=True)
    print(nt.round(2).to_string(index=False))
    nt.to_csv(C.OUT / "phase1_n_sensitivity.csv", index=False, encoding="utf-8-sig")
    ex = pd.concat([kex.assign(league="KBO"), mex.assign(league="MLB")])
    print("ex-ante 제외 비율:\n", ex.round(3).to_string(index=False))
    ex.to_csv(C.OUT / "phase1_exante_exclusion.csv", index=False, encoding="utf-8-sig")
    team_n = pd.concat([t.assign(league=lgn, variant=var) for lgn, vv in [("KBO", kv), ("MLB", mv)]
                        for var, t in vv.items()], ignore_index=True)
    team_n.to_csv(C.PROC / "h1_team_n.csv", index=False, encoding="utf-8-sig")
    prim = nt[nt["variant"] == "primary"]
    by_year = team_n[team_n["variant"] == "primary"].groupby(["league", "year"])["N"].value_counts().unstack(fill_value=0)
    print("연도별 primary N:\n", by_year.to_string())
    by_year.reset_index().to_csv(C.OUT / "phase1_n_by_year.csv", index=False, encoding="utf-8-sig")

    # 게이트: 구간 팀-시즌 < 10
    gate = {}
    for _, r in nt.iterrows():
        small = [i for i in range(4) if r[f"N={i}"] < 10]
        gate[f"{r['league']}|{r['variant']}"] = small
    RES["gate_bins_lt10"] = gate
    RES["n_primary"] = prim.round(3).to_dict("records")
    RES["n_sensitivity"] = nt.round(3).to_dict("records")
    RES["exante_exclusion"] = ex.round(4).to_dict("records")
    with open(C.OUT / "phase1_results.json", "w", encoding="utf-8") as f:
        json.dump(RES, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    chart(team_n)
    print("\n저장 완료")


# ======================================================================
def wilson(x, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def chart(team_n: pd.DataFrame):
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    prim = team_n[team_n["variant"] == "primary"]
    width = 0.38
    for i, lgn in enumerate(["KBO", "MLB"]):
        t = prim[prim["league"] == lgn]
        n = len(t)
        cnt = t["N"].astype(int).value_counts().reindex(range(4), fill_value=0)
        share = cnt / n
        ci = np.array([wilson(c, n) for c in cnt])
        xs = np.arange(4) + (i - 0.5) * (width + 0.02)
        ax.bar(xs, share, width=width, color=C.COLOR[lgn], label=f"{lgn} (팀-시즌 {n})", zorder=2)
        ax.errorbar(xs, share, yerr=[share - ci[:, 0], ci[:, 1] - share], fmt="none",
                    ecolor="#444444", elinewidth=1, capsize=3, zorder=3)
        for x_, s_, c_ in zip(xs, share, cnt):
            ax.text(x_, s_ + 0.03, f"{c_}", ha="center", va="bottom", fontsize=8, color="#333333")
    ax.set_xticks(range(4), [f"N={i}" for i in range(4)])
    ax.set_ylabel("팀-시즌 비율")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, color="#e6e6e6", zorder=0)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("팀별 '2번 적합 인재' 수(N) 분포, 2021~2025 (오차막대: Wilson 95% CI, 막대 위 숫자: 팀-시즌 수)",
                 fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(C.OUT / "phase1_n_hist.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
