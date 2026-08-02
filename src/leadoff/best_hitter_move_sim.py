"""'강한 2번'의 이득을 KIA 김도영 1건이 아니라 KBO 50개 팀-시즌 전체로 검증한다.

`kim_doyoung_analysis.py`는 KIA 2024·김도영 한 사례에 대해서만 반사실(counterfactual)
몬테카를로 시뮬레이션(로스터·실력 고정, 타순만 이동)을 했다. 이 스크립트는 같은 방법을
5시즌(2021-2025) × 10구단 = 50개 팀-시즌 전체로 일반화해, "각 팀의 실제 최고타자를
2번으로 옮기면 대체로 득점에 이득인가"를 대응표본(paired) 통계로 직접 검정한다.

방법:
  1. 팀-시즌별 실제 타석 상위 9인('주전')을 boxscore에서 추출, 정규시즌 wOBA와
     이름매칭(외국인 선수 이름잘림 보정 포함)해 이벤트벡터 부여.
  2. 9인 중 시즌 wOBA 최댓값 선수 = '최고타자'로 지정, 그 선수의 실제 최빈 타순 기록.
  3. 나머지 8인 고정, 최고타자만 '실제 슬롯'과 '2번 슬롯'에 각각 삽입해 시뮬레이션
     (계산량을 9슬롯 전체 스윕 대비 1/4.5로 줄임 — 공통시드로 노이즈 상쇄).
  4. 50개 팀-시즌의 gain=(rpg_slot2-rpg_actual)*144를 대응표본 t-test/Wilcoxon으로 집계.

결과가 어느 방향이든(이득 있음/없음/혼재) 그대로 보고한다 — 결론을 미리 정하지 않는다.

공용 파이프라인은 best_hitter_move_core.py에 있고, 이 파일은 KBO 데이터 스키마
(포지션 코드 첫글자, wOBA 랭킹, 이름잘림 폴백 등)만 정의한다.

출력: outputs/leadoff_analysis/14_best_hitter_move_all_teams.png,
      best_hitter_move_sim.json, best_hitter_move_table.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import lineup_sim as ls
from best_hitter_move_core import LeagueConfig
from best_hitter_move_core import main as core_main

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
YEARS = [2021, 2022, 2023, 2024, 2025]
TEAMS = ["LG", "KT", "SSG", "NC", "KIA", "두산", "롯데", "삼성", "한화", "키움"]
ALLSTAR = {"나눔", "드림"}
VALID_POS = set("포一二三유좌중우지")  # C,1B,2B,3B,SS,LF,CF,RF,DH — 교/주/타/투는 수비위치 특정 불가라 제외


def load_players() -> pd.DataFrame:
    return pd.read_csv(PROC / "kbo_players_woba.csv", encoding="utf-8-sig")


def load_box(year: int) -> pd.DataFrame:
    return pd.read_csv(RAW / f"kbo_boxscore_batters_{year}.csv", encoding="utf-8-sig")


def filter_team_box(b: pd.DataFrame, team: str) -> pd.DataFrame:
    return b[(b["team"] == team) & (~b["opponent"].isin(ALLSTAR))]


def pos_key(b: pd.DataFrame) -> pd.Series:
    return b["pos"].fillna("").str[:1]


def build_events(row_player, r: pd.Series):
    if row_player is not None:
        ev = ls.events_from_full(pa=row_player["pa"], ab=row_player["ab"], h=row_player["hits"],
                                 d2=row_player["doubles"], d3=row_player["triples"], hr=row_player["hr"],
                                 bb=row_player["bb"], hbp=row_player["hbp"], sf=row_player["sf"])
        return ev, float(row_player["woba"]), "full"
    ev = ls.events_from_boxscore(pa_est=r["pa_est"], ab=r["ab"], h=r["hit"], hr=r["hr"], bb=r["bb"])
    return ev, np.nan, "boxscore"


def real_team_rpg(year: int, team: str) -> float:
    real_rpg = pd.read_csv(PROC / "kbo_team_no2.csv", encoding="utf-8-sig")
    real = real_rpg[(real_rpg["year"] == year) & (real_rpg["team"] == team)]
    return float(real["runs_pg"].iloc[0]) if len(real) else np.nan


CFG = LeagueConfig(
    league="KBO", teams=TEAMS, years=YEARS, seed_base=1000,
    valid_pos=VALID_POS, foreigner_fallback=True, metric_key="woba",
    load_players=load_players, load_box=load_box, filter_team_box=filter_team_box,
    pos_key=pos_key, build_events=build_events, real_team_rpg=real_team_rpg,
    png_name="14_best_hitter_move_all_teams.png",
    json_name="best_hitter_move_sim.json",
    csv_name="best_hitter_move_table.csv",
    chart_title_prefix="KBO 50개 팀-시즌: 최고타자 2번 이동 시뮬레이션 이득",
    pilot_team="KIA", pilot_year=2024,
)


if __name__ == "__main__":
    core_main(CFG)
