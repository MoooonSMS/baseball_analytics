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
RE24_MAX_INNING = 9              # RE 행렬·선형가중치는 1~9회만(MLB 연장 승부치기 주자 때문에 두 리그 동일 적용)
EXANTE_MEMBER_MIN_PA = 1         # ex-ante: 해당 시즌 그 팀에서 1타석 이상이면 팀 구성원
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
OBP_RULE_SENS = {"pool_top50": 0.50, "pool_top40": 0.60, "pool_top33": 0.67}  # 민감도: 리그-시즌 후보풀(PA>=기준) OBP 분위수 이상
ONBASE_ALT = ["bb_pct", "obp_minus_avg"]
PRODUCTION_ALT = ["woba", "wrc_plus_pf"]
EXANTE_MIN_PA = 300
# 보조 정의(설계 단계 조정, 2026-10-04 — 배치 결과를 보기 전): N이 primary에서 N=3에 쏠려
# 공급 변동이 거의 없으므로, OBP >= 리그-시즌 후보풀 OBP 상위 33%(=OBP_RULE_SENS["pool_top33"])를
# 보조 정의로 사전 등록하고 두 리그 N 분포를 함께 보고한다. 기본 정의(OBP_RULE)는 그대로 유지.
OBP_RULE_AUX = "pool_top33"

# ---------------------------------------------------------------- Phase 2 설계(2026-10-04 조정)
# (a) 상위 3명 선수 단위 within-team 분석
#     유형 점수 = z(OBP) - z(ISO). z는 리그-시즌 후보풀(PA >= POOL_MIN_PA) 평균·표준편차 기준(PA 비가중)
#     종속변수 = 그 선수가 해당 팀에서 소화한 PA 중 1~2번 슬롯 비율
#     1단계: 팀-시즌별 (유형 점수, 1~2번 비율) Spearman 상관의 리그 평균
#     2단계: 1~2번 비율 ~ 유형 점수 x 리그 + wRC+ 순위 더미 x 리그 + 팀-시즌 고정효과, 팀 클러스터 SE
# (b) 층화: K = 상위 3명 중 출루형(유형 점수 > TYPE_K_THRESHOLD) 수 (0~3)
#     리그별 어떤 K 구간이 팀-시즌 MIN_BIN 미만이면 K<=1 / K>=2 로 합친다
# 용량-반응: KBO는 primary N 변동이 없어 N 기준 검정 불가 -> K와 연속 유형 점수로 대체(KBO 검정력 낮음 명시)
# primary N=3 팀-시즌은 Phase 3 낭비 상한 분석의 주 표본
TYPE_SCORE = ("obp", "iso")      # z(OBP) - z(ISO)
TYPE_K_THRESHOLD = 0.0
MIN_BIN = 10

# ---------------------------------------------------------------- §3-4 슬롯
SLOT_GROUPS = {"top": [1, 2], "middle": [3, 4, 5], "bottom": [6, 7, 8, 9]}
SINGLE_SLOTS = [2, 4]
# 박스스코어에 HBP/SF가 없어 슬롯 PA는 ab+bb로 근사(두 리그 동일)

# ---------------------------------------------------------------- §3-5 효과 크기
EQUIV_MARGIN = 0.10              # 슬롯 점유 비율 차이 ±10%p (TOST)
BOOTSTRAP_B = 10_000
SIM_N_GAMES = 120_000

# ---------------------------------------------------------------- §5 Phase 3 (2026-10-04, 시뮬레이션 실행 전 확정)
# 주전 9인: 팀-시즌 박스스코어에서 포지션별 최다 타석(select_position_starters 로직, pid 기준)
# 실제 타순: 주전 9인을 그 팀 PA가중 평균 슬롯 순으로 정렬. 이벤트: 그 팀 시즌 기록(HBP·SF 포함)
# 반사실 (주전 중 wRC+ 1위 = 'best'; 팀 1위가 주전이 아니면 주전 중 최고):
#   A  (주)   best <-> 실제 2번 주전 자리 교환(swap). best가 이미 2번이면 0
#   A' (민감도) best를 2번에 삽입, 나머지 상대 순서 유지(insert) — 기존 best_hitter_move와 비교용
#   B  상위 3명(primary rank<=3) 중 주전 각자를 1번 또는 2번과 swap한 최대 6가지 중 최대
#   C  9인 전체 재배열(두 칸 교환 국소 탐색). 시험 실행 후 범위 결정
# 선택 편향 방지: B·C는 선택용 시드(SIM_SEARCH_GAMES)로 고르고, 고른 타순을 독립 평가 시드로
#   SIM_N_GAMES 재평가. 실제 vs 반사실은 같은 평가 시드(대응 난수)
# 엔진: 리그 간 비교의 주 결과는 두 리그 같은 기본 엔진, KBO는 상황확률 엔진 병기
# 표본: 주 = primary N=3 팀-시즌, 보조 = 전체 팀-시즌. 보고 = ΔR/144(팀-시즌·팀 클러스터 부트스트랩 CI)
SIM_CF_MAIN = "A_swap"
SIM_CF = ["A_swap", "A_insert", "B_top3_best6", "C_full_opt"]
SIM_ENGINE_MAIN = "basic"
SIM_SEARCH_GAMES = 8_000
SIM_SEED_BASE = 7000
SIM_GAMES_PER_SEASON = 144       # ΔR/144

# ---------------------------------------------------------------- 차트
COLOR = {"KBO": "#2a78d6", "MLB": "#eb6834"}

CHANGELOG = [
    # ("YYYY-MM-DD", "변경 내용", "사유"),
    ("2026-10-03", "KBO_LW_METHOD statiz_rescaled -> re24, MLB_LW_METHOD=re24 추가",
     "Phase 0 승인: 두 리그 동일 방법·2020 ex-ante 커버. Phase 0에서 (a)-(b) 순위상관 0.9999로 결론 영향 없음"),
    ("2026-10-03", "statiz 순위상관 기준을 방법 간 + statiz 카운팅 일치성으로 대체",
     "statiz_season_batters_2021에 wRC+ 컬럼 없음(Phase 0 확인)"),
    ("2026-10-03", "EQUIV_MARGIN 0.10 확정", "Phase 0 승인"),
    ("2026-10-03", "OBP 민감도 기준을 리그-시즌 후보풀 분위수로 구체화, RE24_MAX_INNING=9, EXANTE_MEMBER_MIN_PA=1 추가",
     "Phase 1 구현 전(N 결과 보기 전) 정의 구체화. KBO 공식기록실은 robots.txt/고지로 자동수집 금지라 "
     "KBO 전체 타자 기록은 relay 이벤트에서 재구성"),
    ("2026-10-04", "OBP_RULE_AUX=pool_top33 보조 정의 등록; Phase 2를 N 구간 비교에서 상위 3명 within-team "
     "유형 점수 분석 + K(출루형 수) 층화로 교체; 용량-반응은 K·연속 변수로 대체; N=3은 Phase 3 주 표본",
     "N 분포가 두 리그 모두 N=3에 쏠림(KBO N<=2 6개). 배치 결과는 보기 전 설계 단계 조정. 기본 정의 유지. "
     "H1_ANALYSIS_INSTRUCTION.md §10 변경 이력"),
    ("2026-10-04", "Phase 3 반사실 설계 확정: A swap(주), A' insert, B 상위 3명 1~2번 최대, C 전체 최적화(시험 후 범위 결정); "
     "선택/평가 시드 분리", "Phase 2.5 결과(1위 타자 2번 배치 8% vs 27%) 근거, 사용자 승인. 시뮬레이션 실행 전"),
]
