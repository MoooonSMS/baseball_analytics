"""KBO의 '강한 2번' 반사실 시뮬레이션(best_hitter_move_sim.py)을 MLB에 동일하게 적용한다.

KBO 50개 팀-시즌에서는 최고타자를 2번으로 옮겨도 평균 이득이 유의하지 않았다
(-0.2점/144경기). 이게 KBO만의 현상인지 확인하기 위해 같은 방법론·같은 표본
규모(10팀x5년=50팀-시즌)를 MLB에 적용한다.

KBO와의 차이(공용 파이프라인은 best_hitter_move_core.py):
  - 데이터: data/raw/mlb_boxscore_batters_{year}.csv (scrape_mlb_boxscores.py),
    data/raw/mlb_players_hitting_all_{year}.csv (scrape_mlb_players_all.py)
  - 포지션 코드가 이미 영문 단일값(C/1B/.../DH)이라 KBO처럼 첫글자 추출이나
    이름잘림 보정이 필요 없음
  - 최고타자 랭킹 지표는 wOBA 대신 OPS 사용(팀-연도 내 랭킹 목적이라 지표
    선택이 결과에 영향 없음 — KBO 분석에서 이미 확인)
  - 2021년 국내내셔널리그(ATL/STL/CHC/LAD/SF) 팀은 유니버설 DH 이전이라
    투수가 타석에 서는 경기가 섞여 있음 — DH 포지션 코드가 없는 경우
    select_position_starters의 폴백(잔여 포지션은 타석수 최댓값 미배정
    선수로 채움)이 작동한다.

출력: outputs/leadoff_analysis/15_best_hitter_move_all_teams_mlb.png,
      best_hitter_move_sim_mlb.json, best_hitter_move_table_mlb.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import lineup_sim as ls
from best_hitter_move_core import LeagueConfig
from best_hitter_move_core import main as core_main

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw"
YEARS = [2021, 2022, 2023, 2024, 2025]
TEAMS = ["SEA", "SF", "STL", "TOR", "ATL", "NYY", "BOS", "CHC", "HOU", "LAD"]
VALID_POS = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"}


def load_players() -> pd.DataFrame:
    dfs = [pd.read_csv(RAW / f"mlb_players_hitting_all_{y}.csv", encoding="utf-8-sig") for y in YEARS]
    return pd.concat(dfs, ignore_index=True)


def load_box(year: int) -> pd.DataFrame:
    return pd.read_csv(RAW / f"mlb_boxscore_batters_{year}.csv", encoding="utf-8-sig")


def filter_team_box(b: pd.DataFrame, team: str) -> pd.DataFrame:
    return b[b["team"] == team]


def pos_key(b: pd.DataFrame) -> pd.Series:
    return b["pos"]


def build_events(row_player, r: pd.Series):
    if row_player is not None and pd.notna(row_player.get("plateAppearances")):
        ev = ls.events_from_full(
            pa=row_player["plateAppearances"], ab=row_player["atBats"], h=row_player["hits"],
            d2=row_player["doubles"], d3=row_player["triples"], hr=row_player["homeRuns"],
            bb=row_player["baseOnBalls"], hbp=row_player["hitByPitch"], sf=row_player["sacFlies"])
        ops = float(row_player["ops"]) if pd.notna(row_player["ops"]) else np.nan
        return ev, ops, "full"
    ev = ls.events_from_boxscore(pa_est=r["pa_est"], ab=r["ab"], h=r["hit"], hr=r["hr"], bb=r["bb"])
    return ev, np.nan, "boxscore"


def real_team_rpg(year: int, team: str) -> float:
    b_all = pd.read_csv(RAW / f"mlb_boxscore_batters_{year}.csv", encoding="utf-8-sig")
    g = b_all[b_all["team"] == team].drop_duplicates("gameId")
    return float(g["teamScore"].mean()) if len(g) else np.nan


CFG = LeagueConfig(
    league="MLB", teams=TEAMS, years=YEARS, seed_base=2000,
    valid_pos=VALID_POS, foreigner_fallback=False, metric_key="ops",
    load_players=load_players, load_box=load_box, filter_team_box=filter_team_box,
    pos_key=pos_key, build_events=build_events, real_team_rpg=real_team_rpg,
    png_name="15_best_hitter_move_all_teams_mlb.png",
    json_name="best_hitter_move_sim_mlb.json",
    csv_name="best_hitter_move_table_mlb.csv",
    chart_title_prefix="MLB 50개 팀-시즌: 최고타자 2번 이동 시뮬레이션 이득",
    pilot_team="TOR", pilot_year=2024,
)


if __name__ == "__main__":
    core_main(CFG)
