"""Phase 0: 사전 점검 재현, wOBA/wRC+ 재계산·검증, 기존 H1/H2 영향, 커버리지, 예비 N 분포.

실행: python src/h1_suitable/00_audit_metrics.py
출력: outputs/h1_suitable/phase0_*.csv, phase0_results.json
      data/processed/h1_kbo_players_woba_v2.csv
기존 산출물(outputs/leadoff_analysis/*, data/processed/kbo_*)은 읽기만 한다.
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

RES: dict = {}
ALL_KBO_YEARS = [2020] + C.SEASONS


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def save(df: pd.DataFrame, name: str):
    df.to_csv(C.OUT / f"phase0_{name}.csv", index=False, encoding="utf-8-sig")


# ======================================================================
# 1. §2 문제 재현
# ======================================================================
def reproduce_old_metrics():
    section("#1 기존 KBO wOBA/wRC+ 재현 (data/processed/kbo_players_woba.csv)")
    p = pd.read_csv(C.PROC / "kbo_players_woba.csv", encoding="utf-8-sig")
    stars = []
    for name, y in C.STAR_CHECKS:
        r = p[(p["name"] == name) & (p["year"] == y)]
        stars.append({"name": name, "year": y, "in_file": len(r) > 0,
                      "obp": r["OBP"].iloc[0] if len(r) else None,
                      "slg": r["SLG"].iloc[0] if len(r) else None,
                      "woba_old": r["woba"].iloc[0] if len(r) else None,
                      "wrc_old": r["wrc_plus"].iloc[0] if len(r) else None})
    print(pd.DataFrame(stars).round(3).to_string(index=False))
    pw = p.groupby("year").apply(lambda g: np.average(g["wrc_plus"], weights=g["pa"]),
                                 include_groups=False)
    print("연도별 PA가중 평균 wRC+(규정타석):", pw.round(1).to_dict())
    rho = {"woba~wrc_plus": stats.spearmanr(p["woba"], p["wrc_plus"])[0],
           "woba~SLG": stats.spearmanr(p["woba"], p["SLG"])[0],
           "woba~OPS": stats.spearmanr(p["woba"], p["ops"])[0]}
    print("순위상관:", {k: round(v, 3) for k, v in rho.items()})
    g = pd.read_csv(C.PROC / "kbo_no2_games.csv", encoding="utf-8-sig")
    neg = float((g["season_wrc"] < 0).mean())
    print(f"kbo_no2_games season_wrc 음수 비율: {neg:.1%} (min {g['season_wrc'].min():.1f})")
    RES["old_metrics"] = {"stars": stars, "pa_weighted_wrc_by_year": pw.to_dict(),
                          "rank_corr": rho, "no2_games_neg_wrc_share": neg}


def reproduce_coverage_issues():
    section("#2 KBO 선수 데이터 = 규정타석뿐 / #3 MLB 10팀 / #4 올스타·포스트시즌 / #5 스키마")
    h = H.load_kbo_hitters()
    kc = h.groupby("year").agg(n=("name", "size"), min_pa=("pa", "min"),
                               teams=("team", "nunique")).reset_index()
    print("KBO hitters_*.csv:\n", kc.to_string(index=False))
    print("2025 팀별 선수 수:", h[h["year"] == 2025]["team"].value_counts().to_dict())
    RES["kbo_hitters_coverage"] = kc.to_dict("records")

    mlb = []
    for y in C.SEASONS:
        bx = pd.read_csv(C.RAW / f"mlb_boxscore_batters_{y}.csv", encoding="utf-8-sig")
        al = pd.read_csv(C.RAW / f"mlb_players_hitting_all_{y}.csv", encoding="utf-8-sig")
        q = pd.read_csv(C.RAW / f"mlb_player_hitting_{y}.csv", encoding="utf-8-sig")
        mlb.append({"year": y, "box_teams": bx["team"].nunique(), "box_games": bx["gameId"].nunique(),
                    "all_teams": al["team"].nunique(), "all_players": len(al),
                    "qual_teams": q["team"].nunique(), "qual_players": len(q), "qual_min_pa": q["pa"].min()})
    mlb = pd.DataFrame(mlb)
    print("MLB:\n", mlb.to_string(index=False))
    print("MLB 박스스코어 팀:", sorted(bx["team"].unique()))
    RES["mlb_coverage"] = mlb.to_dict("records")

    b = H.load_kbo_box(ALL_KBO_YEARS, regular_only=False)
    gm = b.drop_duplicates(["gameId", "team"])
    ex = gm.groupby("year").agg(games=("gameId", "nunique"),
                                allstar_games=("gameId", lambda s: s[gm.loc[s.index, "is_allstar"]].nunique()),
                                post_games=("gameId", lambda s: s[gm.loc[s.index, "is_post"]].nunique())).reset_index()
    reg = gm[~gm["is_allstar"] & ~gm["is_post"]]
    ex["regular_games"] = ex["year"].map(reg.groupby("year")["gameId"].nunique())
    ex["team_games_min"] = ex["year"].map(reg.groupby(["year", "team"]).size().groupby("year").min())
    ex["team_games_max"] = ex["year"].map(reg.groupby(["year", "team"]).size().groupby("year").max())
    print("KBO 박스스코어 경기 구성:\n", ex.to_string(index=False))
    RES["kbo_box_composition"] = ex.to_dict("records")
    RES["box_schema"] = list(pd.read_csv(C.RAW / "kbo_boxscore_batters_2025.csv", nrows=1,
                                         encoding="utf-8-sig").columns)
    print("박스스코어 스키마:", RES["box_schema"])
    save(ex, "kbo_box_composition")

    # 기존 build_leadoff_dataset.load_boxscores는 올스타만 제외하고 포스트시즌은 포함한다
    old_post = gm[(gm["year"] >= 2021) & gm["is_post"]].groupby("year")["gameId"].nunique()
    RES["old_pipeline_postseason_games_included"] = old_post.to_dict()
    print("기존 파이프라인(2021~)에 섞인 포스트시즌 경기 수:", old_post.to_dict())


def park_factors():
    section("#6 단순 파크팩터 (박스스코어 홈/원정 득실, 정규시즌)")
    b = H.load_kbo_box(ALL_KBO_YEARS)
    g = b.drop_duplicates(["gameId", "team"]).copy()
    g["rs_ra"] = g["teamScore"] + g["oppScore"]
    t = g.groupby(["year", "team", "homeAway"])["rs_ra"].mean().unstack()
    t["pf_raw"] = t["home"] / t["away"]
    t["pf_half"] = 1 + (t["pf_raw"] - 1) / 2  # 홈경기가 절반이라는 단순 보정
    t = t.reset_index()
    pooled = g[g["year"] >= 2021].groupby(["team", "homeAway"])["rs_ra"].mean().unstack()
    pooled["pf_raw_5yr"] = pooled["home"] / pooled["away"]
    pooled["pf_half_5yr"] = 1 + (pooled["pf_raw_5yr"] - 1) / 2
    print(pooled.round(3).sort_values("pf_half_5yr").to_string())
    save(t, "kbo_park_factor_by_season")
    save(pooled.reset_index(), "kbo_park_factor_pooled")
    RES["kbo_pf_pooled"] = pooled["pf_half_5yr"].round(3).to_dict()
    # MLB는 10팀 박스스코어뿐 -> 그 팀들의 홈/원정만 계산 가능(30팀 확보 후 재계산)
    m = H.load_mlb_box(C.SEASONS).drop_duplicates(["gameId", "team"]).copy()
    m["rs_ra"] = m["teamScore"] + m["oppScore"]
    mp = m.groupby(["team", "homeAway"])["rs_ra"].mean().unstack()
    mp["pf_half_5yr"] = 1 + (mp["home"] / mp["away"] - 1) / 2
    RES["mlb_pf_pooled_10teams"] = mp["pf_half_5yr"].round(3).to_dict()
    print("MLB(10팀):", RES["mlb_pf_pooled_10teams"])
    return t


def exante_availability():
    section("#7 ex-ante 준비 상태 / #8 Statiz")
    have_2020_box = (C.RAW / "kbo_boxscore_batters_2020.csv").exists()
    have_2020_hit = (C.RAW / "hitters_2020.csv").exists()
    RES["exante"] = {"kbo_box_2020": have_2020_box, "kbo_hitters_2020": have_2020_hit,
                     "kbo_relay_2020": (C.RAW / "kbo_relay_events_2020.csv").exists(),
                     "statiz_constants_2020": 2020 in pd.read_csv(C.RAW / "statiz_woba_constants.csv")["Year"].values}
    print(RES["exante"])
    print("Statiz: 접근하지 않음. statiz_*.csv는 읽기 전용으로만 사용.")


# ======================================================================
# 2. wOBA / wRC+ 재계산
# ======================================================================
def recompute():
    section("§3-2 wOBA/wRC+ 재계산: (a) statiz 상수 재스케일 vs (b) relay RE24")
    tot = H.kbo_league_totals()
    lw_a = H.statiz_runvalues(tot)
    w_a, lg_a = H.rescale_from_runvalues(lw_a, tot)

    cache = C.PROC / "h1_kbo_pa_events.pkl"
    if cache.exists():
        pa = pd.read_pickle(cache)
    else:
        pa = pd.concat([H.kbo_pa_events(y) for y in C.SEASONS], ignore_index=True)
        pa.to_pickle(cache)
    # relay 득점 = 박스스코어 득점 검증
    box = H.load_kbo_box(C.SEASONS)
    box_r = box.drop_duplicates(["gameId", "team"]).groupby("year")["teamScore"].sum()
    relay_r = pa.groupby("year")["runs"].sum()
    RES["relay_runs_check"] = pd.DataFrame({"relay": relay_r, "box": box_r}).to_dict("index")
    print("relay 득점 vs 박스스코어 득점:", RES["relay_runs_check"])
    lw_b = H.re24_runvalues(pa)
    w_b, lg_b = H.rescale_from_runvalues(lw_b, tot)

    statiz = pd.read_csv(C.RAW / "statiz_woba_constants.csv").set_index("Year")
    const = pd.concat({"a_lw": lw_a, "b_lw": lw_b, "a_w": w_a, "b_w": w_b}, axis=1)
    lgcmp = pd.DataFrame({"statiz_lgwOBA": statiz["wOBA"], "statiz_Scale": statiz["Scale"],
                          "a_lg_woba": lg_a["lg_woba"], "a_scale": lg_a["scale"],
                          "b_scale": lg_b["scale"], "lw_out_a": lg_a["lw_out"],
                          "lw_out_b": lg_b["lw_out"], "rppa": lg_a["rppa"]}).loc[C.SEASONS]
    # 재스케일 전 'statiz 가중치 그대로'의 리그 wOBA(-lw_out 반영 전)도 같이
    t = H.add_basic(tot.set_index("year"))
    n = H.event_counts(t)
    old_raw = sum(lw_a[e] * n[e] for e in H.EVENTS) / t["epa"]
    lgcmp["statiz_wOBA_reconstructed"] = (sum((lw_a[e] - lw_a["out"]) * n[e] for e in H.EVENTS)
                                          / t["epa"] * statiz["Scale"])
    lgcmp["old_method_lg_woba"] = old_raw
    print("리그 상수 비교:\n", lgcmp.round(4).to_string())
    print("가중치:\n", const.round(3).to_string())
    save(lgcmp.reset_index(names="year"), "kbo_league_constants")
    save(const.reset_index(names="year"), "kbo_weights")
    RES["kbo_league_constants"] = lgcmp.round(4).to_dict("index")
    return tot, (w_a, lg_a), (w_b, lg_b)


def kbo_players_v2(wa, wb):
    h = H.load_kbo_hitters()
    h = H.add_basic(h)
    h["woba_a"] = H.woba(h, wa[0])
    h["wrc_a"] = H.wrc_plus(h["woba_a"], h["year"], wa[1])
    h["woba_b"] = H.woba(h, wb[0])
    h["wrc_b"] = H.wrc_plus(h["woba_b"], h["year"], wb[1])
    old = pd.read_csv(C.PROC / "kbo_players_woba.csv", encoding="utf-8-sig")
    h = h.merge(old[["year", "name", "team", "woba", "wrc_plus"]].rename(
        columns={"woba": "woba_old", "wrc_plus": "wrc_old"}), on=["year", "name", "team"], how="left")
    return h


def mlb_players(lg_w, lg):
    """MLB 규정타석(30팀) + 전체 타자(10팀)."""
    q = pd.concat([pd.read_csv(C.RAW / f"mlb_player_hitting_{y}.csv", encoding="utf-8-sig")
                   for y in C.SEASONS], ignore_index=True)
    q = q.rename(columns={"season": "year", "2b": "d2", "3b": "d3"})
    a = pd.concat([pd.read_csv(C.RAW / f"mlb_players_hitting_all_{y}.csv", encoding="utf-8-sig")
                   for y in C.SEASONS], ignore_index=True)
    a = a.rename(columns={"plateAppearances": "pa", "atBats": "ab", "hits": "h", "doubles": "d2",
                          "triples": "d3", "homeRuns": "hr", "baseOnBalls": "bb",
                          "intentionalWalks": "ibb", "hitByPitch": "hbp", "sacFlies": "sf"})
    out = []
    for d, src in [(q, "qualified_30"), (a, "all_10")]:
        d = H.add_basic(d)
        d["woba"] = H.woba(d, lg_w)
        d["wrc"] = H.wrc_plus(d["woba"], d["year"], lg)
        d["src"] = src
        out.append(d)
    return out


def validate(h, mlbq, mlba, lg_a, lg_mlb, tot):
    section("§3-2 검증 기준")
    v = {}
    # (1) 리그-시즌 PA가중 평균 wRC+ = 100 (전 리그 타석 기준; 팀 합계로 계산)
    team = []
    for y in C.SEASONS:
        a = pd.read_csv(C.RAW / f"statiz_slot_agg_{y}.csv").groupby("code").sum(numeric_only=True)
        team.append(pd.DataFrame({"year": y, "team": a.index, "pa": a["PA"], "ab": a["AB"], "h": a["H"],
                                  "d2": a["2B"], "d3": a["3B"], "hr": a["HR"], "bb": a["BB"],
                                  "ibb": a["IB"], "hbp": a["HP"], "sf": a["SF"]}))
    team = pd.concat(team, ignore_index=True)
    rows = []
    for meth, (w, lg) in {"a": (WA[0], WA[1]), "b": (WB[0], WB[1])}.items():
        team[f"wrc_{meth}"] = H.wrc_plus(H.woba(team, w), team["year"], lg)
    for y, g in team.groupby("year"):
        q = h[h["year"] == y]
        rows.append({"league": "KBO", "year": y,
                     "full_league_wrc_a": np.average(g["wrc_a"], weights=g["pa"]),
                     "full_league_wrc_b": np.average(g["wrc_b"], weights=g["pa"]),
                     "qualified_wrc_a": np.average(q["wrc_a"], weights=q["pa"]),
                     "qualified_wrc_old": np.average(q["wrc_old"], weights=q["pa"]),
                     "qualified_n": len(q)})
    mteam = H.load_mlb_slot_totals()
    mteam = mteam[mteam["year"].isin(C.SEASONS)]
    mteam["wrc"] = H.wrc_plus(H.woba(mteam, H.mlb_weights()), mteam["year"], lg_mlb)
    for y, g in mteam.groupby("year"):
        q = mlbq[mlbq["year"] == y]
        a = mlba[(mlba["year"] == y)]
        rows.append({"league": "MLB", "year": y,
                     "full_league_wrc_a": np.average(g["wrc"], weights=g["pa"]),
                     "qualified_wrc_a": np.average(q["wrc"], weights=q["pa"]),
                     "all10_wrc": np.average(a.dropna(subset=["wrc"])["wrc"], weights=a.dropna(subset=["wrc"])["pa"]),
                     "qualified_n": len(q)})
    pw = pd.DataFrame(rows)
    print("PA가중 평균 wRC+:\n", pw.round(1).to_string(index=False))
    save(pw, "wrc_league_mean")
    v["pa_weighted_wrc"] = pw.round(2).to_dict("records")
    v["pass_league_100pm3"] = bool((pw["full_league_wrc_a"].sub(100).abs() <= C.WRC_PLUS_TOL).all())

    # (2) 스타 선수
    stars = []
    for name, y in C.STAR_CHECKS:
        r = h[(h["name"] == name) & (h["year"] == y)]
        stars.append({"name": name, "year": y, "in_qualified_file": len(r) > 0,
                      **({k: float(r[k].iloc[0]) for k in ["pa", "obp", "slg", "woba_old", "wrc_old",
                                                            "woba_a", "wrc_a", "woba_b", "wrc_b"]}
                         if len(r) else {})})
    st = pd.DataFrame(stars)
    print("스타 선수:\n", st.round(3).to_string(index=False))
    top = h.sort_values("wrc_a", ascending=False).groupby("year").head(3)[
        ["year", "name", "team", "pa", "obp", "slg", "woba_a", "wrc_a"]]
    print("연도별 wRC+(a) 상위 3:\n", top.round(3).to_string(index=False))
    v["stars"] = stars
    v["top3_by_year"] = top.round(3).to_dict("records")

    # (3) 순위상관
    rc = {"a~b (woba)": stats.spearmanr(h["woba_a"], h["woba_b"])[0],
          "a~old (woba)": stats.spearmanr(h["woba_a"], h["woba_old"])[0],
          "a~OPS": stats.spearmanr(h["woba_a"], h["obp"] + h["slg"])[0],
          "a~b (wrc, 연도 내)": float(np.mean([stats.spearmanr(g["wrc_a"], g["wrc_b"])[0]
                                              for _, g in h.groupby("year")]))}
    # statiz_season_batters_2021: wRC+ 컬럼이 없어 같은 공식으로 그 카운팅 스탯에서 wOBA 계산 후 비교
    s = pd.read_csv(C.RAW / "statiz_season_batters_2021.csv").rename(columns={
        "PA": "pa", "AB": "ab", "H": "h", "2B": "d2", "3B": "d3", "HR": "hr", "BB": "bb",
        "IB": "ibb", "HP": "hbp", "SF": "sf"})
    s["woba_statiz_counts"] = H.woba(s, WA[0])
    m = h[h["year"] == 2021].merge(s[["name", "pa", "woba_statiz_counts"]], on="name",
                                   suffixes=("", "_statiz"))
    rc["official~statiz_counts 2021 (a)"] = stats.spearmanr(m["woba_a"], m["woba_statiz_counts"])[0]
    rc["n_matched_statiz_2021"] = len(m)
    rc["pa_exact_match_share"] = float((m["pa"] == m["pa_statiz"]).mean())
    v["rank_corr"] = rc
    print("순위상관:", {k: round(x, 4) if isinstance(x, float) else x for k, x in rc.items()})
    print("statiz_season_batters_2021 컬럼:", list(s.columns))

    # (4) 같은 눈금: 리그 평균 wOBA = 리그 OBP, 규정타석 분포 비교
    dist = []
    for lgname, d, lgc in [("KBO", h.assign(woba=h["woba_a"], wrc=h["wrc_a"]), WA[1]), ("MLB", mlbq, lg_mlb)]:
        d = d.copy()
        d["woba_minus_lg"] = d["woba"] - d["year"].map(lgc["lg_woba"])
        dist.append({"league": lgname, "n": len(d),
                     **{f"woba_minus_lg_q{int(q*100)}": d["woba_minus_lg"].quantile(q) for q in [.1, .5, .9]},
                     **{f"wrc_q{int(q*100)}": d["wrc"].quantile(q) for q in [.1, .5, .9]},
                     "wrc_max": d["wrc"].max()})
    dist = pd.DataFrame(dist)
    lgo = pd.DataFrame({"KBO_lg_woba": WA[1]["lg_woba"], "KBO_lg_obp": WA[1]["lg_obp"],
                        "MLB_lg_woba": lg_mlb["lg_woba"], "MLB_lg_obp": lg_mlb["lg_obp"],
                        "KBO_scale": WA[1]["scale"], "MLB_scale": lg_mlb["scale"],
                        "KBO_rppa": WA[1]["rppa"], "MLB_rppa": lg_mlb["rppa"]}).loc[C.SEASONS]
    print("리그 wOBA vs OBP:\n", lgo.round(4).to_string())
    print("규정타석 분포:\n", dist.round(3).to_string(index=False))
    v["scale_check"] = lgo.round(4).to_dict("index")
    v["qualified_distribution"] = dist.round(4).to_dict("records")
    RES["validation"] = v
    save(h[["year", "name", "team", "pa", "obp", "slg", "iso", "woba_old", "wrc_old",
            "woba_a", "wrc_a", "woba_b", "wrc_b"]], "kbo_players_before_after")


# ======================================================================
# 3. 기존 H1/H2 영향
# ======================================================================
def old_results_impact(h):
    section("기존 H1/H2 결과에 대한 wRC+ 수정 영향")
    sys.path.insert(0, str(C.ROOT / "src" / "leadoff"))
    import build_leadoff_dataset as bld
    import leadoff_hypotheses as lh
    old = json.load(open(C.ROOT / "outputs" / "leadoff_analysis" / "hypothesis_results.json", encoding="utf-8"))

    p = bld.load_players()
    p = p.merge(h[["year", "name", "team", "woba_a", "wrc_a"]], on=["year", "name", "team"], how="left")
    assert p["woba_a"].notna().all()
    p["woba"], p["wrc_plus"] = p["woba_a"], p["wrc_a"]
    b = bld.load_boxscores()  # 기존 로직 그대로(포스트시즌 포함) — wRC+ 효과만 분리
    games = bld.no2_games(b, p)
    kbo_team = bld.team_no2(games, b, p)
    kbo_slot = bld.kbo_team_slot(b, p)
    mlb_slot = pd.read_csv(C.PROC / "mlb_team_slot.csv", encoding="utf-8-sig")

    kbo_p, mlb_p = lh.load_kbo_players(), lh.load_mlb_players()
    allp = lh.add_combo_metrics(pd.concat([kbo_p, mlb_p], ignore_index=True).dropna(subset=["obp", "iso"]))
    new_prop = lh.h1_proportions(allp)
    new_depth = lh.h1_roster_depth_vs_realized(allp, kbo_team, mlb_slot)
    new_trade = lh.h2_slot2_vs_slot4(kbo_slot, mlb_slot)["result"]
    new_flat = lh.h2_flatness(kbo_slot, mlb_slot)["result"]

    def row(item, key, old_v, new_v):
        return {"item": item, "stat": key, "old": old_v, "new": new_v}
    rows = []
    for o, nrow in zip(old["h1_proportions"], new_prop.to_dict("records")):
        for k in ["kbo_prop", "mlb_prop", "p"]:
            rows.append(row(f"H1 비율({o['threshold']})", k, o[k], nrow[k]))
    for k in ["kbo_r", "kbo_p", "mlb_r", "mlb_p"]:
        rows.append(row("H1 보조: 팀 내 콤보 수 ~ 2번 wOBA", k, old["h1_roster_depth_vs_realized"][k], new_depth[k]))
    for k in ["r_kbo", "p_kbo", "r_mlb", "z_diff", "p_diff"]:
        rows.append(row("H2 2번-4번 상관", k, old["h2_tradeoff"][k], new_trade[k]))
    for k in ["kbo_mean", "mlb_mean", "t", "p"]:
        rows.append(row("H2 평탄도", k, old["h2_flatness"][k], new_flat[k]))
    imp = pd.DataFrame(rows)
    imp["old"], imp["new"] = imp["old"].astype(float), imp["new"].astype(float)
    print(imp.round(4).to_string(index=False))
    save(imp, "old_results_impact")
    RES["old_results_impact"] = imp.round(5).to_dict("records")

    # 참고: 같은 경로를 쓰는 다른 기존 산출물(leadoff_analysis 03/05)의 2번 wRC+ 중앙값 분할
    old_team = pd.read_csv(C.PROC / "kbo_team_no2.csv", encoding="utf-8-sig")
    cmp_ = old_team[["year", "team", "no2_wrc"]].merge(kbo_team[["year", "team", "no2_wrc"]],
                                                        on=["year", "team"], suffixes=("_old", "_new"))
    hi_old = cmp_["no2_wrc_old"] >= cmp_["no2_wrc_old"].median()
    hi_new = cmp_["no2_wrc_new"] >= cmp_["no2_wrc_new"].median()
    RES["leadoff05_median_split_reclassified"] = int((hi_old != hi_new).sum())
    RES["no2_wrc_rank_corr"] = float(stats.spearmanr(cmp_["no2_wrc_old"], cmp_["no2_wrc_new"])[0])
    print("leadoff_analysis 05(2번 wRC+ 중앙값 분할) 재분류 팀-시즌:",
          RES["leadoff05_median_split_reclassified"], "/", len(cmp_),
          "| no2_wrc 순위상관", round(RES["no2_wrc_rank_corr"], 4))
    print("참고 H1 보조 실제 2번 콤보 체크(기존): n_combo_no2 =", old["h1_actual_no2_check"]["n_combo_no2"])


# ======================================================================
# 4. 커버리지 표 + 예비 N
# ======================================================================
def match_kbo(box: pd.DataFrame, players: pd.DataFrame) -> pd.Series:
    """박스 (year, team, name) -> players 인덱스. 정확 매칭 후 4자 접두어 폴백(기존 방식)."""
    key = players.set_index(["year", "team", "name"]).index
    pos = pd.Series(range(len(players)), index=key)
    pos = pos[~pos.index.duplicated()]
    k = pd.MultiIndex.from_frame(box[["year", "team", "name"]])
    m = pd.Series(pos.reindex(k).values, index=box.index)
    long = players[players["name"].str.len() > 4].copy()
    long["n4"] = long["name"].str[:4]
    long = long.drop_duplicates(["year", "team", "n4"])
    p4 = pd.Series(long.index.map(lambda i: players.index.get_loc(i)),
                   index=pd.MultiIndex.from_frame(long[["year", "team", "n4"]]))
    miss = m.isna() & (box["name"].str.len() == 4)
    if miss.any():
        m[miss] = p4.reindex(pd.MultiIndex.from_frame(box.loc[miss, ["year", "team", "name"]])).values
    return m


def coverage(h, mlbq, mlba):
    section("커버리지 표: 리그·시즌·팀별 선수 수와 경기 수, 박스스코어 PA 근사 커버율")
    rows = []
    kb = H.load_kbo_box(C.SEASONS)
    kb["pa_approx"] = kb["ab"] + kb["bb"]
    kb["m"] = match_kbo(kb, h.reset_index(drop=True))
    for (y, t), g in kb.groupby(["year", "team"]):
        rows.append({"league": "KBO", "year": y, "team": t, "games": g["gameId"].nunique(),
                     "box_players": g["name"].nunique(),
                     "season_file_players": int(((h["year"] == y) & (h["team"] == t)).sum()),
                     "season_file_pa_ge300": int(((h["year"] == y) & (h["team"] == t) & (h["pa"] >= 300)).sum()),
                     "box_pa_covered_share": g.loc[g["m"].notna(), "pa_approx"].sum() / g["pa_approx"].sum()})
    mb = H.load_mlb_box(C.SEASONS)
    mb["pa_approx"] = mb["ab"] + mb["bb"]
    for (y, t), g in mb.groupby(["year", "team"]):
        names = set(mlba.loc[(mlba["year"] == y) & (mlba["team"] == t), "name"])
        rows.append({"league": "MLB", "year": y, "team": t, "games": g["gameId"].nunique(),
                     "box_players": g["playerCode"].nunique(),
                     "season_file_players": len(names),
                     "season_file_pa_ge300": int(((mlba["year"] == y) & (mlba["team"] == t) & (mlba["pa"] >= 300)).sum()),
                     "box_pa_covered_share": g.loc[g["name"].isin(names), "pa_approx"].sum() / g["pa_approx"].sum()})
    cov = pd.DataFrame(rows)
    save(cov, "coverage")
    summ = cov.groupby(["league", "year"]).agg(teams=("team", "nunique"), games_min=("games", "min"),
                                               players_per_team=("season_file_players", "mean"),
                                               pool300_per_team=("season_file_pa_ge300", "mean"),
                                               box_pa_covered=("box_pa_covered_share", "mean")).reset_index()
    print(summ.round(3).to_string(index=False))
    RES["coverage_summary"] = summ.round(3).to_dict("records")


def n_of_team(d: pd.DataFrame, prod: str, lg_obp: pd.Series, pool_pa=C.POOL_MIN_PA) -> pd.DataFrame:
    d = d[d["pa"] >= pool_pa].copy()
    d["rank"] = d.groupby(["year", "team"])[prod].rank(ascending=False, method="first")
    d["S"] = (d["rank"] <= C.TOP_K) & (d["obp"] >= d["year"].map(lg_obp))
    t = d.groupby(["year", "team"]).agg(pool=("name", "size"), N=("S", "sum")).reset_index()
    return t


def prelim_n(h, mlbq, mlba, lg_mlb):
    section("예비 N 분포 (현재 보유 데이터, 커버리지 한계 있음)")
    kh = h.assign(wrc=h["wrc_a"])
    sets = {
        "KBO 규정타석(50팀-시즌)": n_of_team(kh, "wrc", WA[1]["lg_obp"]),
        "MLB 규정타석(30팀, 150팀-시즌)": n_of_team(mlbq, "wrc", lg_mlb["lg_obp"]),
        "MLB 전체타자(10팀, 50팀-시즌)": n_of_team(mlba, "wrc", lg_mlb["lg_obp"]),
    }
    # 10팀만 놓고 규정타석으로 판정했을 때와 전체로 판정했을 때 N 일치율 (규정타석 한계의 크기)
    ten = mlba["team"].unique()
    full_name = {"ATL": "Atlanta Braves", "BOS": "Boston Red Sox", "CHC": "Chicago Cubs",
                 "HOU": "Houston Astros", "LAD": "Los Angeles Dodgers", "NYY": "New York Yankees",
                 "SEA": "Seattle Mariners", "SF": "San Francisco Giants", "STL": "St. Louis Cardinals",
                 "TOR": "Toronto Blue Jays"}
    q10 = mlbq[mlbq["team"].isin(full_name.values())].copy()
    q10["team"] = q10["team"].map({v: k for k, v in full_name.items()})
    a = n_of_team(mlba, "wrc", lg_mlb["lg_obp"]).merge(
        n_of_team(q10, "wrc", lg_mlb["lg_obp"]), on=["year", "team"], how="left", suffixes=("_all", "_qual"))
    a["N_qual"] = a["N_qual"].fillna(0)
    agree = float((a["N_all"] == a["N_qual"]).mean())
    sets["MLB 10팀을 규정타석만으로 판정"] = a.rename(columns={"N_qual": "N"})[["year", "team", "N"]].assign(pool=a["pool_qual"])

    rows = []
    for k, t in sets.items():
        vc = t["N"].value_counts().reindex(range(4), fill_value=0)
        rows.append({"set": k, "team_seasons": len(t), **{f"N={i}": int(vc[i]) for i in range(4)},
                     "pool_lt3": int((t["pool"].fillna(0) < 3).sum()), "mean_N": t["N"].mean()})
    nd = pd.DataFrame(rows)
    print(nd.round(2).to_string(index=False))
    print(f"MLB 10팀: 규정타석 판정 N과 전체타자 판정 N 일치율 {agree:.0%}")
    save(nd, "prelim_n_distribution")
    RES["prelim_n"] = nd.round(3).to_dict("records")
    RES["mlb10_qual_vs_all_N_agreement"] = agree


# ======================================================================
def main():
    global WA, WB
    C.OUT.mkdir(parents=True, exist_ok=True)
    reproduce_old_metrics()
    reproduce_coverage_issues()
    park_factors()
    exante_availability()
    tot, WA, WB = recompute()
    h = kbo_players_v2(WA, WB)
    w_mlb, lg_mlb, reg = H.mlb_league_constants(C.SEASONS)
    kt = pd.concat([pd.read_csv(C.RAW / f"statiz_slot_agg_{y}.csv").groupby("code").sum(numeric_only=True)
                    .reset_index().assign(year=y) for y in C.SEASONS]).rename(columns={
                        "PA": "pa", "AB": "ab", "H": "h", "2B": "d2", "3B": "d3", "HR": "hr", "BB": "bb",
                        "IB": "ibb", "HP": "hbp", "SF": "sf", "code": "team"})
    # statiz 팀 코드는 팀명과 바로 매핑되지 않아, 팀 R은 statiz R에 연도별 (박스/statiz) 비율을 곱해 보정
    ratio = tot.set_index("year")["r"] / tot.set_index("year")["r_statiz"]
    kt["r"] = kt["R"] * kt["year"].map(ratio)
    kreg = H.team_regression_scale(kt, WA[0])
    RES["scale_by_team_regression"] = {"MLB_2010_2025_ex2020": reg, "KBO_2021_2025": kreg}
    print("\n팀 득점 회귀로 추정한 scale:", RES["scale_by_team_regression"])
    mlbq, mlba = mlb_players(w_mlb, lg_mlb)
    validate(h, mlbq, mlba, WA[1], lg_mlb, tot)
    h.to_csv(C.PROC / "h1_kbo_players_woba_v2.csv", index=False, encoding="utf-8-sig")
    old_results_impact(h)
    coverage(h, mlbq, mlba)
    prelim_n(h, mlbq, mlba, lg_mlb)
    with open(C.OUT / "phase0_results.json", "w", encoding="utf-8") as f:
        json.dump(RES, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("\n저장:", C.OUT)


if __name__ == "__main__":
    main()
