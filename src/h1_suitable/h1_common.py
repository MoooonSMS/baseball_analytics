"""H1 재검증 공용 유틸: 리그 합계, wOBA 가중치(두 방법), wOBA/wRC+ 계산, 박스스코어 로드.

카운팅 스탯 표준 컬럼명: year, pa, ab, h, d2, d3, hr, bb, ibb, hbp, sf (+ r)

wOBA 정의(FanGraphs 관행, 두 리그 동일):
  w_i = scale * (lw_i - lw_out)        lw: 평균 타석 대비 득점가치
  wOBA = Σ w_i n_i / (AB + BB - IBB + HBP + SF)
  scale: 리그 PA가중 평균 wOBA = 리그 OBP 가 되도록 정한다
  wRC+ = ((wOBA - lgwOBA)/scale + lgR/PA) / (lgR/PA) * 100      (파크팩터 없음)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

sys.path.insert(0, str(C.ROOT / "src" / "leadoff"))
from build_situational_probs import classify_outcome  # noqa: E402

EVENTS = ["bb", "hbp", "1b", "2b", "3b", "hr"]


# ---------------------------------------------------------------- 공통 계산
def add_basic(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ubb"] = d["bb"] - d["ibb"]
    d["s1"] = d["h"] - d["d2"] - d["d3"] - d["hr"]
    d["epa"] = d["ab"] + d["ubb"] + d["hbp"] + d["sf"]
    obp_den = d["ab"] + d["bb"] + d["hbp"] + d["sf"]
    d["obp"] = (d["h"] + d["bb"] + d["hbp"]) / obp_den.replace(0, np.nan)
    d["avg"] = d["h"] / d["ab"].replace(0, np.nan)
    d["slg"] = (d["s1"] + 2 * d["d2"] + 3 * d["d3"] + 4 * d["hr"]) / d["ab"].replace(0, np.nan)
    d["iso"] = d["slg"] - d["avg"]
    d["bb_pct"] = d["bb"] / d["pa"].replace(0, np.nan)
    d["obp_minus_avg"] = d["obp"] - d["avg"]
    return d


def event_counts(d: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"bb": d["ubb"], "hbp": d["hbp"], "1b": d["s1"], "2b": d["d2"],
                         "3b": d["d3"], "hr": d["hr"]})


def woba(df: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """weights: index=year, columns=EVENTS (wOBA 스케일)."""
    d = add_basic(df)
    n = event_counts(d)
    num = sum(n[e] * d["year"].map(weights[e]) for e in EVENTS)
    return num / d["epa"].replace(0, np.nan)


def wrc_plus(woba_s: pd.Series, year: pd.Series, lg: pd.DataFrame) -> pd.Series:
    """lg: index=year, columns lg_woba, scale, rppa."""
    rppa = year.map(lg["rppa"])
    return ((woba_s - year.map(lg["lg_woba"])) / year.map(lg["scale"]) + rppa) / rppa * 100


def rescale_from_runvalues(lw: pd.DataFrame, totals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """lw: index=year, columns=EVENTS + ['out'] (평균 타석 대비 득점가치, out 포함).
    totals: 리그 합계(add_basic 적용 전 표준 컬럼).
    반환: (wOBA 가중치, 리그 상수 lg_woba/scale/rppa/lw_out/lg_obp)."""
    t = add_basic(totals.set_index("year", drop=False))
    n = event_counts(t)
    rel = lw[EVENTS].sub(lw["out"], axis=0)
    raw = sum(rel[e] * n[e] for e in EVENTS) / t["epa"]
    scale = t["obp"] / raw
    w = rel.mul(scale, axis=0)
    lg = pd.DataFrame({"lg_woba": t["obp"], "scale": scale, "rppa": t["r"] / t["pa"],
                       "lw_out": lw["out"], "lg_obp": t["obp"]})
    return w, lg


# ---------------------------------------------------------------- KBO 리그 합계
def kbo_league_totals(years=C.SEASONS) -> pd.DataFrame:
    """statiz_slot_agg(팀×타순 집계, 읽기 전용)의 합 = 리그 전체 타석 합계."""
    rows = []
    for y in years:
        p = C.RAW / f"statiz_slot_agg_{y}.csv"
        if not p.exists():
            continue
        a = pd.read_csv(p).sum(numeric_only=True)
        rows.append({"year": y, "pa": a["PA"], "ab": a["AB"], "h": a["H"], "d2": a["2B"],
                     "d3": a["3B"], "hr": a["HR"], "bb": a["BB"], "ibb": a["IB"],
                     "hbp": a["HP"], "sf": a["SF"], "r_statiz": a["R"], "so": a["SO"], "sh": a["SH"]})
    t = pd.DataFrame(rows)
    # 리그 득점은 박스스코어 정규시즌 팀 득점 합을 쓴다. statiz 타순집계 R은 대주자 득점 등이
    # 빠져 약 5% 적다(Phase 0 실측: 2021 6,549 vs 6,896).
    box = load_kbo_box(list(t["year"]))
    r = box.drop_duplicates(["gameId", "team"]).groupby("year")["teamScore"].sum()
    t["r"] = t["year"].map(r)
    return t


# ---------------------------------------------------------------- 방법 (a): statiz 상수 재스케일
def statiz_runvalues(totals: pd.DataFrame) -> pd.DataFrame:
    """statiz_woba_constants의 eBB/1B/2B/3B/HR은 '평균 타석 대비 득점가치'(합이 0이 되는 선형가중치)
    스케일이다. 리그 전체 Σ lw_i n_i + lw_out n_out = 0 으로 아웃 가치를 복원한다.
    n_out = AB - H + SF (ePA 안의 아웃; 실책출루는 AB-H에 포함되어 아웃 취급)."""
    c = pd.read_csv(C.RAW / "statiz_woba_constants.csv").set_index("Year")
    lw = pd.DataFrame({"bb": c["eBB"], "hbp": c["eBB"], "1b": c["1B"], "2b": c["2B"],
                       "3b": c["3B"], "hr": c["HR"]})
    t = add_basic(totals.set_index("year", drop=False))
    lw = lw.loc[t.index]
    n = event_counts(t)
    on = sum(lw[e] * n[e] for e in EVENTS)
    n_out = t["ab"] - t["h"] + t["sf"]
    lw["out"] = -on / n_out
    return lw


# ---------------------------------------------------------------- 방법 (b): relay RE24
def kbo_pa_events(year: int) -> pd.DataFrame:
    """relay 이벤트 -> 타석 단위(before/after 상태, 결과, 득점, 이닝).
    (gameId, no) = 타석 1개. before = 직전 그룹의 마지막 행 상태. 득점 = 그룹 안 '홈인' 행 수
    (+ 홈런이면 타자 본인 1점; 타자 '홈인' 행은 따로 찍히지 않는 것을 박스스코어 득점 합으로 검증)."""
    df = pd.read_csv(C.RAW / f"kbo_relay_events_{year}.csv", encoding="utf-8-sig")
    df = df[df["gameId"].str[:4] == str(year)]  # 포스트시즌 제외
    df = df.sort_values(["gameId", "seqno"]).reset_index(drop=True)
    df["is_end"] = df["type"].isin([13, 23])
    df["homein"] = (df["type"] == 24) & df["text"].str.contains("홈인", na=False)
    key = ["gameId", "no"]
    g = df.groupby(key, sort=False)
    last = g.tail(1).set_index(key)[["inn", "homeOrAway", "base1", "base2", "base3", "out"]]
    ends = df[df["is_end"]].groupby(key, sort=False)["text"].first()
    runs = g["homein"].sum()
    pa = last.copy()
    pa["runs_homein"] = runs
    end_text = ends.reindex(pa.index)
    pa["outcome"] = end_text.where(end_text.isna(), end_text.map(classify_outcome))
    pa.loc[end_text.str.contains("고의4구", na=False).values, "outcome"] = "IBB"
    pa = pa.reset_index()
    # before 상태 = 같은 경기 직전 그룹의 after
    for c_ in ["base1", "base2", "base3", "out"]:
        pa[f"{c_}_b"] = pa.groupby("gameId")[c_].shift(1).fillna(0).astype(int)
    pa = pa[pa["outcome"].notna()].copy()
    pa["runs"] = pa["runs_homein"] + (pa["outcome"] == "HR").astype(int)
    # 이닝이 바뀌는 첫 타석은 before가 0아웃·주자없음(이닝시작 표시행이 그룹으로 끼어 있음)
    pa["state_b"] = ((pa["base1_b"] > 0).astype(int).astype(str) + (pa["base2_b"] > 0).astype(int).astype(str)
                     + (pa["base3_b"] > 0).astype(int).astype(str) + pa["out_b"].astype(str))
    pa["state_a"] = ((pa["base1"] > 0).astype(int).astype(str) + (pa["base2"] > 0).astype(int).astype(str)
                     + (pa["base3"] > 0).astype(int).astype(str) + pa["out"].astype(str))
    pa["year"] = year
    return pa


OUT_OUTCOMES = {"OUT", "K", "GIDP", "FC", "SAC_FLY", "ERROR"}
RE_EVENT_MAP = {"BB": "bb", "HBP": "hbp", "1B": "1b", "2B": "2b", "3B": "3b", "HR": "hr"}


def re24_runvalues(pa: pd.DataFrame) -> pd.DataFrame:
    """시즌별 RE24 선형가중치. 완결 이닝(3아웃 도달)만 RE 행렬 추정에 사용.
    out 가치 = OUT_OUTCOMES 평균 RE24 (실책출루는 ePA상 아웃이므로 포함)."""
    rows = []
    for y, d in pa.groupby("year"):
        d = d.copy()
        half = ["gameId", "inn", "homeOrAway"]
        d["inning_runs_after"] = d.groupby(half)["runs"].transform(lambda s: s[::-1].cumsum()[::-1]) - d["runs"]
        complete = d.groupby(half)["out"].transform("max") >= 3
        d["ro_rest"] = d["inning_runs_after"] + d["runs"]  # 이 타석 시작부터 이닝 끝까지 득점
        re = d[complete].groupby("state_b")["ro_rest"].mean()
        d["re_b"] = d["state_b"].map(re)
        d["re_a"] = np.where(d["out"] >= 3, 0.0, d["state_a"].map(re))
        d["re24"] = d["re_a"] - d["re_b"] + d["runs"]
        lw = {RE_EVENT_MAP[k]: d.loc[d["outcome"] == k, "re24"].mean() for k in RE_EVENT_MAP}
        lw["out"] = d.loc[d["outcome"].isin(OUT_OUTCOMES), "re24"].mean()
        lw["year"] = y
        rows.append(lw)
    return pd.DataFrame(rows).set_index("year")


# ---------------------------------------------------------------- KBO 선수 시즌 기록
def load_kbo_hitters(years=C.SEASONS) -> pd.DataFrame:
    dfs = []
    for y in years:
        p = C.RAW / f"hitters_{y}.csv"
        if p.exists():
            dfs.append(pd.read_csv(p, encoding="utf-8-sig"))
    h = pd.concat(dfs, ignore_index=True).rename(columns={
        "선수명": "name", "팀명": "team", "PA": "pa", "AB": "ab", "R": "r", "H": "h", "2B": "d2",
        "3B": "d3", "HR": "hr", "BB": "bb", "IBB": "ibb", "HBP": "hbp", "SF": "sf", "SO": "so",
        "OBP": "obp_official", "SLG": "slg_official"})
    return h


# ---------------------------------------------------------------- MLB
def load_mlb_slot_totals() -> pd.DataFrame:
    m = pd.read_csv(C.RAW / "mlb_slot_splits.csv")
    team = m.groupby(["season", "team"], as_index=False)[
        ["plateAppearances", "atBats", "hits", "doubles", "triples", "homeRuns",
         "baseOnBalls", "intentionalWalks", "hitByPitch", "sacFlies"]].sum()
    # 타순 스플릿의 runs는 전부 0(API가 스플릿 득점을 안 줌) -> 팀 득점은 mlb_team_runs.csv
    tr = pd.read_csv(C.RAW / "mlb_team_runs.csv")[["season", "team", "runs"]]
    team = team.merge(tr, on=["season", "team"], how="left")
    return team.rename(columns={"season": "year", "plateAppearances": "pa", "atBats": "ab",
                                "runs": "r", "hits": "h", "doubles": "d2", "triples": "d3",
                                "homeRuns": "hr", "baseOnBalls": "bb", "intentionalWalks": "ibb",
                                "hitByPitch": "hbp", "sacFlies": "sf"})


def mlb_weights() -> pd.DataFrame:
    c = pd.read_csv(C.RAW / "mlb_woba_constants.csv").set_index("season")
    return pd.DataFrame({"bb": c["wBB"], "hbp": c["wHBP"], "1b": c["w1B"], "2b": c["w2B"],
                         "3b": c["w3B"], "hr": c["wHR"]})


def team_regression_scale(team: pd.DataFrame, weights: pd.DataFrame) -> dict:
    """팀-시즌 R/PA ~ 팀 wOBA (시즌 고정효과) 기울기 = 1/scale. 두 리그에 같은 방법으로 쓰는
    scale 교차검증용. 반환: {'scale', 'se', 'n'}."""
    t = team.copy()
    t["woba"] = woba(t, weights)
    t["rppa"] = t["r"] / t["pa"]
    t = t.dropna(subset=["woba"])
    x = t["woba"] - t.groupby("year")["woba"].transform("mean")  # 시즌 고정효과 = 연도 내 demean
    y = t["rppa"] - t.groupby("year")["rppa"].transform("mean")
    b = (x * y).sum() / (x * x).sum()
    dof = len(t) - t["year"].nunique() - 1
    se = np.sqrt(((y - b * x) ** 2).sum() / dof / (x * x).sum())
    return {"scale": 1 / b, "scale_lo": 1 / (b + 1.96 * se), "scale_hi": 1 / (b - 1.96 * se),
            "n": int(len(t))}


def mlb_league_constants(years) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """MLB: FanGraphs 가중치(이미 wOBA 스케일). lgwOBA·lgR/PA는 30팀 타순 스플릿 합으로,
    scale은 Guts 접근 불가(403)라 팀 득점 회귀로 추정(전 시즌 공통 1개)."""
    w = mlb_weights()
    team = load_mlb_slot_totals()
    lgt = team.groupby("year", as_index=False).sum(numeric_only=True)
    lgt = lgt[lgt["year"].isin(w.index)]
    lgt["lg_woba"] = woba(lgt, w).values
    reg = team_regression_scale(team[team["year"].isin(w.index) & (team["year"] != 2020)], w)
    lg = pd.DataFrame({"lg_woba": lgt.set_index("year")["lg_woba"],
                       "scale": reg["scale"],
                       "rppa": (lgt["r"] / lgt["pa"]).values,
                       "lg_obp": add_basic(lgt)["obp"].values}, index=lgt["year"])
    return w, lg.loc[[y for y in years if y in lg.index]], reg


# ---------------------------------------------------------------- 박스스코어
def load_kbo_box(years, regular_only=True) -> pd.DataFrame:
    dfs = []
    for y in years:
        b = pd.read_csv(C.RAW / f"kbo_boxscore_batters_{y}.csv", encoding="utf-8-sig")
        b["year"] = y
        dfs.append(b)
    b = pd.concat(dfs, ignore_index=True)
    b["is_allstar"] = b["team"].isin(C.KBO_ALLSTAR_TEAMS) | b["opponent"].isin(C.KBO_ALLSTAR_TEAMS)
    b["is_post"] = b["gameId"].str[:4] != b["year"].astype(str)
    if regular_only:
        b = b[~b["is_allstar"] & ~b["is_post"]].copy()
    b["team"] = b["team"].replace(C.KBO_TEAM_ALIAS)
    b["opponent"] = b["opponent"].replace(C.KBO_TEAM_ALIAS)
    return b


def load_mlb_box(years) -> pd.DataFrame:
    dfs = []
    for y in years:
        b = pd.read_csv(C.RAW / f"mlb_boxscore_batters_{y}.csv", encoding="utf-8-sig")
        b["year"] = y
        dfs.append(b)
    return pd.concat(dfs, ignore_index=True)
