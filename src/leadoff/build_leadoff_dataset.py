"""'강한 1번' 분석용 데이터셋 구축.

입력
  data/raw/hitters_{y}.csv              KBO 공식 시즌 선수 기록 (2021-2025)
  data/raw/kbo_boxscore_batters_{y}.csv 네이버 경기별 타자 박스스코어 (batOrder 포함)
  data/raw/statiz_slot_agg_{y}.csv      스탯티즈 팀×타순 집계 (리그 프로필용)
  data/raw/mlb_slot_splits.csv          MLB 팀×타순 스플릿 (2010-2025)

출력 (data/processed/)
  kbo_players_woba.csv     선수-시즌 wOBA/wRC+
  kbo_leadoff_games.csv    게임 단위: 팀, 그날 1번 선발, 그 선수의 시즌 wRC+, 팀 득점
  kbo_team_leadoff.csv     팀-시즌: 리드오프 질 지수, 팀 R/G, 나머지 라인업 wOBA
  kbo_slot_profile.csv     KBO 리그 시즌×타순 프로필 (wOBA, ISO, BB%, K%)
  mlb_slot_profile.csv     MLB 리그 시즌×타순 프로필 (+ 팀 단위 상대 생산성)
"""
from pathlib import Path

import numpy as np
import pandas as pd

import metrics

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
YEARS = [2021, 2022, 2023, 2024, 2025]

# 네이버 팀명 -> KBO 공식 팀명 (hitters csv의 팀명 표기와 일치)
TEAM_MAP = {"LG": "LG", "KT": "KT", "SSG": "SSG", "NC": "NC", "KIA": "KIA",
            "두산": "두산", "롯데": "롯데", "삼성": "삼성", "한화": "한화", "키움": "키움"}


def load_players() -> pd.DataFrame:
    dfs = []
    for y in YEARS:
        df = pd.read_csv(RAW / f"hitters_{y}.csv")
        df = df.rename(columns={
            "선수명": "name", "팀명": "team", "PA": "pa", "AB": "ab", "R": "runs",
            "H": "hits", "2B": "doubles", "3B": "triples", "HR": "hr", "BB": "bb",
            "IBB": "ibb", "HBP": "hbp", "SF": "sf", "SO": "so", "TB": "tb",
        })
        dfs.append(df)
    p = pd.concat(dfs, ignore_index=True)
    p["woba"] = metrics.add_woba(p, bb="bb", ibb="ibb", hbp="hbp", h="hits",
                                 d2="doubles", d3="triples", hr="hr", ab="ab", sf="sf")
    # 리그 R/PA (시즌별, 전체 타자 합)
    lg = p.groupby("year").apply(lambda g: g["runs"].sum() / g["pa"].sum(), include_groups=False)
    lg = lg.rename("rppa").to_frame()
    p["wrc_plus"] = metrics.add_wrc_plus(p, woba_col="woba", lg=lg)
    p["iso"] = (p["tb"] - p["hits"]) / p["ab"].replace(0, np.nan)
    p["bb_pct"] = p["bb"] / p["pa"].replace(0, np.nan)
    return p


ALLSTAR_TEAMS = {"나눔", "드림"}  # KBO 올스타전 특별 편성팀 (정규시즌 팀 아님, 분석 제외)


def load_boxscores() -> pd.DataFrame:
    dfs = [pd.read_csv(RAW / f"kbo_boxscore_batters_{y}.csv", encoding="utf-8-sig") for y in YEARS]
    b = pd.concat(dfs, ignore_index=True)
    b = b[~b["team"].isin(ALLSTAR_TEAMS) & ~b["opponent"].isin(ALLSTAR_TEAMS)]
    b["year"] = b["gameDate"].str[:4].astype(int)
    b["team"] = b["team"].map(TEAM_MAP).fillna(b["team"])
    b["opponent"] = b["opponent"].map(TEAM_MAP).fillna(b["opponent"])
    return b


def leadoff_games(b: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """게임×팀: 그날 슬롯1 '선발'(같은 batOrder 중 첫 번째 행)과 팀 득점."""
    slot1 = b[b["batOrder"] == 1].copy()
    # CSV는 API 순서 유지: 같은 게임·팀·타순에서 첫 행이 선발
    starters = slot1.groupby(["gameId", "team"], as_index=False).first()
    q = p[["year", "name", "team", "pa", "woba", "wrc_plus", "iso", "bb_pct"]].rename(
        columns={"pa": "season_pa", "woba": "season_woba", "wrc_plus": "season_wrc"})
    g = starters.merge(q, on=["year", "name", "team"], how="left")

    # 폴백: 네이버 박스스코어는 외국인 선수명을 4자로 잘라 표기하는 경우가 있어
    # (예: hitters.csv의 "소크라테스" -> boxscore "소크라테") 정확매칭이 실패한다.
    # 박스 이름이 정확히 4자이고 미매칭인 경우, (연도,팀)에서 그 이름을 접두어로
    # 갖는 5자 이상 선수명으로 보정 매칭한다.
    fill_cols = ["season_pa", "season_woba", "season_wrc", "iso", "bb_pct"]
    miss = g["season_woba"].isna() & (g["name"].str.len() == 4)
    if miss.any():
        q_long = q[q["name"].str.len() > 4].copy()
        q_long["name4"] = q_long["name"].str[:4]
        q_long = q_long.drop_duplicates(subset=["year", "team", "name4"])
        fb = g.loc[miss, ["gameId", "team", "year", "name"]].merge(
            q_long[["year", "team", "name4"] + fill_cols],
            left_on=["year", "team", "name"], right_on=["year", "team", "name4"],
            how="left").set_index("gameId")
        for col in fill_cols:
            g.loc[miss, col] = g.loc[miss, "gameId"].map(fb[col])

    g["home"] = (g["homeAway"] == "home").astype(int)
    return g[["gameId", "gameDate", "year", "team", "opponent", "home", "teamScore",
              "oppScore", "name", "playerCode", "season_pa", "season_woba",
              "season_wrc", "iso", "bb_pct"]]


def team_leadoff(games: pd.DataFrame, b: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """팀-시즌: 리드오프 질 지수(선발 경기수 가중 시즌 wOBA/wRC+), 팀 R/G,
    나머지 라인업(2~9번 선발들)의 가중 wOBA."""
    rows = []
    q = p.set_index(["year", "name", "team"])
    for (y, team), g in games.groupby(["year", "team"]):
        n_games = g["gameId"].nunique()
        rg = g["teamScore"].sum() / n_games
        lead_woba = np.average(g["season_woba"].dropna(),
                               weights=None) if g["season_woba"].notna().any() else np.nan
        # 가중 평균: 각 경기 선발 리드오프의 시즌 wOBA 평균(=경기수 가중)
        lead_woba = g["season_woba"].mean()
        lead_wrc = g["season_wrc"].mean()
        # 나머지 라인업: 슬롯 2~9 선발들의 시즌 wOBA 경기 가중 평균
        sub = b[(b["year"] == y) & (b["team"] == team) & (b["batOrder"].between(2, 9))]
        starters = sub.groupby(["gameId", "batOrder"], as_index=False).first()
        merged = starters.merge(
            p[["year", "name", "team", "woba"]], on=["name", "team"], how="left",
            suffixes=("", "_p"))
        merged = merged[merged["year_p"] == y] if "year_p" in merged else merged
        rest_woba = merged["woba"].mean()
        rows.append({"year": y, "team": team, "games": n_games, "runs_pg": rg,
                     "leadoff_woba": lead_woba, "leadoff_wrc": lead_wrc,
                     "rest_woba": rest_woba})
    return pd.DataFrame(rows)


def kbo_team_slot(b: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """팀-시즌-타순(1~9): 해당 슬롯 선발들의 시즌 wOBA 평균과 선발 경기수.

    mlb_team_slot.csv와 같은 그레인(year/team/slot)으로 맞춰 슬롯1-슬롯4
    트레이드오프, 라인업 평탄도(뎁스) 분석에 쓴다."""
    starters = b[b["batOrder"].between(1, 9)].groupby(
        ["gameId", "team", "batOrder"], as_index=False).first()
    merged = starters.merge(
        p[["year", "name", "team", "woba"]], on=["name", "team"], how="left",
        suffixes=("", "_p"))
    merged = merged[merged["year_p"] == merged["year"]]
    out = merged.groupby(["year", "team", "batOrder"], as_index=False).agg(
        games=("gameId", "nunique"), woba=("woba", "mean"))
    return out.rename(columns={"batOrder": "slot"})


def kbo_slot_profile() -> pd.DataFrame:
    dfs = [pd.read_csv(RAW / f"statiz_slot_agg_{y}.csv") for y in YEARS]
    a = pd.concat(dfs, ignore_index=True)
    lg = a.groupby(["year", "bo"], as_index=False).sum(numeric_only=True)
    lg["1B"] = lg["H"] - lg["2B"] - lg["3B"] - lg["HR"]
    c = metrics.constants()
    w = lg["year"].map(c["eBB"])
    w1, w2, w3, w4 = (lg["year"].map(c[k]) for k in ["1B", "2B", "3B", "HR"])
    ubb = lg["BB"] - lg["IB"]
    num = w * (ubb + lg["HP"]) + w1 * lg["1B"] + w2 * lg["2B"] + w3 * lg["3B"] + w4 * lg["HR"]
    den = lg["AB"] + ubb + lg["HP"] + lg["SF"]
    lg["woba"] = num / den
    lg["iso"] = (lg["TB"] - lg["H"]) / lg["AB"]
    lg["bb_pct"] = lg["BB"] / lg["PA"]
    lg["k_pct"] = lg["SO"] / lg["PA"]
    lg["obp"] = (lg["H"] + lg["BB"] + lg["HP"]) / (lg["AB"] + lg["BB"] + lg["HP"] + lg["SF"])
    lg["slg"] = lg["TB"] / lg["AB"]
    lg["ops"] = lg["obp"] + lg["slg"]
    lg["sh_pct"] = lg["SH"] / lg["PA"]
    # 시즌별 전 슬롯 평균 대비 상대 wOBA/OPS
    for col in ["woba", "ops"]:
        season_avg = lg.groupby("year").apply(
            lambda g: (g[col] * g["PA"]).sum() / g["PA"].sum(), include_groups=False)
        lg[f"rel_{col}"] = lg[col] / lg["year"].map(season_avg)
    return lg[["year", "bo", "PA", "woba", "obp", "slg", "ops", "iso", "bb_pct",
               "k_pct", "sh_pct", "rel_woba", "rel_ops"]]


def mlb_slot_profile() -> pd.DataFrame:
    m = pd.read_csv(RAW / "mlb_slot_splits.csv")
    for col in ["obp", "slg", "ops", "avg"]:
        m[col] = pd.to_numeric(m[col], errors="coerce")
    lg = m.groupby(["season", "slot"], as_index=False).agg(
        pa=("plateAppearances", "sum"), ab=("atBats", "sum"), h=("hits", "sum"),
        d2=("doubles", "sum"), d3=("triples", "sum"), hr=("homeRuns", "sum"),
        bb=("baseOnBalls", "sum"), ibb=("intentionalWalks", "sum"),
        hbp=("hitByPitch", "sum"), sf=("sacFlies", "sum"), so=("strikeOuts", "sum"),
        sb=("stolenBases", "sum"), tb=("totalBases", "sum"), r=("runs", "sum"))
    lg["obp"] = (lg["h"] + lg["bb"] + lg["hbp"]) / (lg["ab"] + lg["bb"] + lg["hbp"] + lg["sf"])
    lg["slg"] = lg["tb"] / lg["ab"]
    lg["ops"] = lg["obp"] + lg["slg"]
    lg["iso"] = (lg["tb"] - lg["h"]) / lg["ab"]
    lg["bb_pct"] = lg["bb"] / lg["pa"]
    lg["k_pct"] = lg["so"] / lg["pa"]
    for col in ["ops"]:
        season_avg = lg.groupby("season").apply(
            lambda g: (g[col] * g["pa"]).sum() / g["pa"].sum(), include_groups=False)
        lg[f"rel_{col}"] = lg[col] / lg["season"].map(season_avg)
    # 팀 단위 상대 OPS (검정용 분포)
    team = m.copy()
    team_avg = team.groupby(["season", "teamId"]).apply(
        lambda g: (g["ops"] * g["plateAppearances"]).sum() / g["plateAppearances"].sum(),
        include_groups=False).rename("team_ops")
    team = team.merge(team_avg, on=["season", "teamId"])
    team["rel_ops_team"] = team["ops"] / team["team_ops"]
    team_out = team[["season", "slot", "team", "teamId", "plateAppearances", "ops",
                     "rel_ops_team"]]
    team_out.to_csv(OUT / "mlb_team_slot.csv", index=False, encoding="utf-8-sig")
    return lg


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = load_players()
    p.to_csv(OUT / "kbo_players_woba.csv", index=False, encoding="utf-8-sig")
    b = load_boxscores()
    games = leadoff_games(b, p)
    games.to_csv(OUT / "kbo_leadoff_games.csv", index=False, encoding="utf-8-sig")
    tl = team_leadoff(games, b, p)
    tl.to_csv(OUT / "kbo_team_leadoff.csv", index=False, encoding="utf-8-sig")
    tslot = kbo_team_slot(b, p)
    tslot.to_csv(OUT / "kbo_team_slot.csv", index=False, encoding="utf-8-sig")
    ks = kbo_slot_profile()
    ks.to_csv(OUT / "kbo_slot_profile.csv", index=False, encoding="utf-8-sig")
    ms = mlb_slot_profile()
    ms.to_csv(OUT / "mlb_slot_profile.csv", index=False, encoding="utf-8-sig")
    print("players:", len(p), "| leadoff games:", len(games),
          "| join miss:", games["season_woba"].isna().mean().round(3))
    print("team-seasons:", len(tl))
    print("team-season-slots:", len(tslot))
    print(ks[ks["year"] == 2024][["bo", "woba", "ops", "rel_ops"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
