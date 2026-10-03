"""H1 재검증 사전 등록 파라미터 (H1_ANALYSIS_INSTRUCTION.md §3).

결과를 보기 전에 고정한다. 바꿔야 하면 아래 CHANGELOG에 날짜·사유·변경 전후 값을
남기고, 변경 전후 결과를 모두 보고한다.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "h1_suitable"

# ---------------------------------------------------------------- §3-1 단위와 기간
SEASONS = [2021, 2022, 2023, 2024, 2025]
KBO_EXANTE_PRIOR = 2020          # KBO 2021 ex-ante 판정용 전년도
MLB_EXANTE_SKIP = {2021}         # MLB 2020(60경기)은 신뢰 낮음 -> MLB 2021 ex-ante 제외
KBO_ALLSTAR_TEAMS = {"나눔", "드림"}
# KBO 네이버 gameId는 정규시즌이 'YYYYMMDD...'로 시작하고, 포스트시즌은
# '3333'/'4444'/'5555'/'7777' 같은 접두어를 쓴다(Phase 0에서 실측 확인).
KBO_TEAM_ALIAS = {"SK": "SSG"}   # 2020 SK -> 2021~ SSG (동일 프랜차이즈)

# ---------------------------------------------------------------- §3-2 지표
# 선형가중치 방법: "re24" (b, 주) | "statiz_rescaled" (a, KBO 교차검증).
# MLB도 StatsAPI play-by-play로 같은 RE24 방법을 적용한다(FanGraphs 가중치는 교차검증).
KBO_LW_METHOD = "re24"
MLB_LW_METHOD = "re24"
WRC_PLUS_TOL = 3.0               # 리그-시즌 PA가중 평균 wRC+ 100±3
STAR_CHECKS = [("이정후", 2021), ("김도영", 2024), ("안현민", 2025)]
STAR_TARGET = 150
# statiz_season_batters_2021에는 wRC+ 컬럼이 없어 기준을 대체:
#   (a)-(b) 방법 간 순위상관 >= 0.95  +  statiz 카운팅으로 계산한 wOBA와 순위상관 >= 0.95
STATIZ_RANK_CORR_MIN = 0.95

# ---------------------------------------------------------------- §3-3 후보 풀과 S
POOL_MIN_PA = 300
POOL_MIN_PA_SENS = [200, 400]
TOP_K = 3                        # 팀 내 wRC+ 상위 K명
OBP_RULE = "league_mean"         # OBP >= 리그-시즌 PA가중 평균 OBP
OBP_RULE_SENS = ["team_top50", "team_top40", "team_top33"]  # 민감도: 상위 50/40/33%
ONBASE_ALT = ["bb_pct", "obp_minus_avg"]
PRODUCTION_ALT = ["woba", "wrc_plus_pf"]
EXANTE_MIN_PA = 300

# ---------------------------------------------------------------- §3-4 슬롯
SLOT_GROUPS = {"top": [1, 2], "middle": [3, 4, 5], "bottom": [6, 7, 8, 9]}
SINGLE_SLOTS = [2, 4]
# 박스스코어에 HBP/SF가 없어 슬롯 PA는 ab+bb로 근사(두 리그 동일)

# ---------------------------------------------------------------- §3-5 효과 크기
EQUIV_MARGIN = 0.10              # 슬롯 점유 비율 차이 ±10%p (TOST)
BOOTSTRAP_B = 10_000
SIM_N_GAMES = 120_000

# ---------------------------------------------------------------- 차트
COLOR = {"KBO": "#2a78d6", "MLB": "#eb6834"}

CHANGELOG = [
    # ("YYYY-MM-DD", "변경 내용", "사유"),
    ("2026-10-03", "KBO_LW_METHOD statiz_rescaled -> re24, MLB_LW_METHOD=re24 추가",
     "Phase 0 승인: 두 리그 동일 방법·2020 ex-ante 커버. Phase 0에서 (a)-(b) 순위상관 0.9999로 결론 영향 없음"),
    ("2026-10-03", "statiz 순위상관 기준을 방법 간 + statiz 카운팅 일치성으로 대체",
     "statiz_season_batters_2021에 wRC+ 컬럼 없음(Phase 0 확인)"),
    ("2026-10-03", "EQUIV_MARGIN 0.10 확정", "Phase 0 승인"),
]
