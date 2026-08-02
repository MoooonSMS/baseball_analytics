# KBO Baseball Analytics

KBO(한국프로야구) 공식기록실 데이터를 스크래핑해서 정리하고, 탐색적 분석과 간단한 모델링, 팀 간 상대전적 통계 검정까지 해보는 개인 프로젝트입니다.

## 구성

```
src/
  hitting_eda/               # 선수기록 EDA: 개인 타자/투수 기본기록 스크래핑 + 탐색적 분석 + 예측 모델
    scrape_kbo.py             # 타자/투수 기본기록 스크래핑 (2021-2025)
    eda.py                     # 데이터 정제 + 탐색적 분석 차트 생성
    model.py                   # 올해 성적으로 다음 시즌 3할타자 여부를 예측하는 baseline 모델
  h2h/                        # 상대전적 검정: 팀 간 상대전적표 스크래핑 + 이항검정
    scrape_h2h.py              # 팀 간 상대전적표(승-패-무) 스크래핑
    h2h_test.py                # Log5 기대 승률 대비 실제 상대전적이 통계적으로 이례적인지 이항검정
  leadoff/                    # '강한 2번' 분석 파이프라인
    scrape_naver_boxscores.py  # 네이버 스포츠 API에서 경기별 타순(batOrder) 포함 박스스코어 수집
    scrape_mlb_splits.py       # MLB StatsAPI 팀×타순 스플릿 수집 (2010-2025)
    scrape_mlb_team_runs.py    # MLB StatsAPI 팀-시즌 총 득점/PA 수집
    metrics.py                 # Statiz 선형가중치 기반 wOBA/wRC+ 계산 유틸
    build_leadoff_dataset.py   # '강한 2번' 분석용 데이터셋 구축 (KBO+MLB)
    leadoff_analysis.py        # KBO vs MLB 2번타자 전략 회귀분석·가설검정·차트
    scrape_mlb_player_hitting.py  # MLB StatsAPI 규정타석 개인 타자 시즌기록 수집 (2021-2025)
    leadoff_hypotheses.py      # '강한 2번'이 KBO에서 안 통하는 이유 3가설 검증
    lineup_sim.py               # 타순(라인업) 몬테카를로 시뮬레이터
    kim_doyoung_analysis.py    # 김도영 라인업 최적화 사례 분석
    best_hitter_move_core.py   # 최고타자 2번 이동 반사실 검증 공용 파이프라인 (KBO/MLB 공유)
    best_hitter_move_sim.py    # best_hitter_move_core.py의 KBO 설정 — 김도영 사례를 50개 팀-시즌으로 일반화
    scrape_mlb_boxscores.py    # MLB StatsAPI 경기별 타자 박스스코어(타순/포지션) 수집
    scrape_mlb_players_all.py  # MLB StatsAPI 팀x시즌 전체 타자 시즌기록(규정타석 무관) 수집
    best_hitter_move_sim_mlb.py # best_hitter_move_core.py의 MLB 설정 — 같은 검증을 MLB 50개 팀-시즌에 적용
    slot_slg_trend.py          # 타순별(1-9번) 장타율 연도 추이 보조 차트
    scrape_kbo_relay.py        # Naver relay API에서 KBO 경기 매 플레이 베이스-아웃 상태 원본 수집
    build_situational_probs.py # relay 원본 -> 상황조건부 진루·병살·도루 확률표 추출
    lineup_optimize_all_teams.py # 업그레이드된 lineup_sim으로 '최적 vs 실제 타순' 이득을 KBO 50개 팀-시즌에서 재검증

data/                    # 세 분석이 공유하는 데이터 저장소 (leadoff가 hitting_eda의 원본 데이터도 씀)
  raw/              # 스크래핑 원본 CSV (KBO 공식기록, Statiz, 네이버 박스스코어, MLB StatsAPI)
  processed/        # 정제된 데이터 (hitters.csv, kbo/mlb 타순·리드오프 데이터셋)

outputs/
  hitting_eda/            # 선수기록 EDA 차트(PNG)
  h2h/                     # 상대전적 이항검정 결과 엑셀
  leadoff_analysis/        # 리드오프 분석 차트, 회귀결과, 사례 데이터
```

## 스크립트 설명

### `scrape_kbo.py`
KBO 공식기록실(koreabaseball.com)에서 타자/투수 기본기록을 가져옵니다. 사이트가 ASP.NET WebForms 기반이라 연도 변경이나 페이지 이동이 `__doPostBack()` postback으로 처리되는데, 매 요청마다 `__VIEWSTATE` 등 hidden 필드를 유지하며 이를 흉내냅니다. 타자는 Basic1(AVG, PA, HR 등)과 Basic2(BB, SO, SLG/OBP/OPS 등) 두 탭을 합쳐야 전체 지표가 나옵니다.

### `scrape_h2h.py`
팀 순위 페이지의 상대전적표(팀별 맞대결 승-패-무 매트릭스)를 스크래핑합니다.

### `eda.py`
수집한 타자 기록을 정제해 `data/processed/hitters.csv`로 저장하고, OPS 분포/팀별 평균 OPS/홈런-타율 관계/주요 지표 상관관계 히트맵을 생성합니다.

### `model.py`
올해 시점에 관측 가능한 지표(AVG, OBP, SLG, OPS, HR, RBI, SO, BB, PA)로 "다음 시즌 3할 달성 여부"를 예측하는 로지스틱 회귀 baseline입니다. 규정타석 근처(PA≥100) 선수만 사용하고, ROC-AUC/confusion matrix/classification report로 평가합니다.

### `h2h_test.py`
특정 팀이 각 상대 팀에게 보인 승률이 통계적으로 유의하게 이례적인지를 이항검정으로 확인합니다.

- 기대 승률은 Log5 공식(Bill James)으로 계산하며, 이때 두 팀의 '평소 실력'은 서로를 제외한 나머지 팀 상대 승률(leave-one-out)을 씁니다. 자기 자신과의 대결 기록을 기준(baseline)에 섞으면 안 되기 때문입니다.
- KBO는 팀당 상대 1팀과 16경기 내외만 치르기 때문에 표본이 작아 통계적 검정력이 낮다는 점을 함께 봐야 합니다.
- 결과는 콘솔 표와 함께 `outputs/h2h_{year}_{team}.xlsx`로 저장됩니다.

### `scrape_naver_boxscores.py` / `scrape_mlb_splits.py` / `scrape_mlb_team_runs.py`
'강한 2번' 전략 분석용 원천 데이터를 모으는 스크립트입니다. KBO 공식기록실은 시즌 누적 기록만 제공하고 경기별 타순이 없어, 네이버 스포츠 API에서 `batOrder`(타순) 필드가 포함된 경기별 박스스코어를 따로 수집합니다(2021-2025, 중단 후 재실행 시 이미 저장된 gameId는 건너뜀). MLB는 StatsAPI의 `sitCodes=b1-b9` 스플릿으로 팀×타순 스탯을, 타순 스플릿에는 없는 팀 시즌 총득점은 별도 엔드포인트로 수집합니다(2010-2025).

KBO 세이버매트릭스(팀×타순 집계, wOBA 선형가중치 상수, 타순별 선수 개인기록)는 [Statiz](https://www.statiz.co.kr)에서 받아 `data/raw/statiz_*.csv`로 저장해 사용합니다. (2026-07 기준 Statiz가 사이트 전체 크롤링을 금지 공지했기 때문에, 이 데이터는 2021-2025년치까지만 있고 이후 확장하지 않습니다.)

### `metrics.py` / `build_leadoff_dataset.py`
Statiz 연도별 선형가중치로 wOBA·wRC+를 계산하는 유틸(`metrics.py`)과, 위 원본들을 조인해 분석용 데이터셋(`data/processed/kbo_*`, `mlb_*`)을 만드는 스크립트입니다. 파크팩터는 반영하지 않습니다. 핵심 산출물은 게임별 2번타자 기록(`kbo_no2_games.csv`)과 팀-시즌 2번타자 질 지수(`kbo_team_no2.csv`)입니다.

### `leadoff_analysis.py`
왜 2번인가: 1번은 항상 주자 없는 이닝 첫 타석이 껴 있는 반면, 2번은 PA를 거의 그대로 유지하면서도 주자가 있는 상황에 더 자주 들어선다는 것이 현대 세이버매트릭스(*The Book*)의 핵심 결론입니다. KBO가 MLB만큼 '강한 2번'(출루형·파워형 2번타자) 전략을 적극적으로 쓰지 않는지를 데이터로 검증합니다.
- KBO(50팀-시즌)·MLB(480팀-시즌)에서 팀 득점을 2번타자 질 + 나머지 라인업 질로 회귀(OLS)해 두 리그의 효과 크기를 비교
- 2번타자 wRC+ 상/하위 그룹 간 팀 득점 차이 검정(Welch t-test, Mann-Whitney U)
- KBO vs MLB 2번타순의 리그 평균 대비 생산성 추이, 타순별 희생번트 비율, ISO/BB%/K% 프로필 비교 차트 생성
- 2024년 KBO 외국인 타자의 2번 기용 실측 성적 등 사례 탐색(원래 이 선수들은 리드오프 실험 사례로 알려져 있어 표본이 작을 수 있음)

결과는 `outputs/leadoff_analysis/`에 저장됩니다.

### `scrape_mlb_player_hitting.py`
MLB StatsAPI에서 규정타석(Qualified) 타자의 개인 시즌 기록을 수집합니다(2021-2025, KBO 데이터 존재 연도와 표본 정의를 맞춤). `leadoff_hypotheses.py`의 OBP+ISO 희소성 비교에 사용됩니다.

### `leadoff_hypotheses.py`
`leadoff_analysis.py`가 KBO와 MLB의 2번타자 효과 크기가 다르다는 것까지는 보였지만, *왜* 다른지는 설명하지 않습니다. 이 스크립트는 그 이유에 대한 세 가설을 사전에 방법론을 정한 뒤 검증합니다(데이터가 가설을 지지하지 않아도 그대로 보고).

- **H1 (희소성)**: MLB 2번타자가 통하는 건 OBP와 파워(ISO)를 동시에 갖춘 선수가 있어서다. KBO는 그런 선수가 상대적으로 희소하다.
- **H2 (라인업 뎁스)**: 2번에 최고 타자를 배치하면 4번에 쓸 선수가 부족해지는 트레이드오프가 KBO에서 더 크다.
- **H3 (득점 환경)**: 타고투저/투고투저 여부가 2번타자 효과를 조절한다.

결과는 `outputs/leadoff_analysis/06-09_*.png`, `hypothesis_results.json`에 저장됩니다.

### `lineup_sim.py` / `kim_doyoung_analysis.py`
`lineup_sim.py`는 타자 시즌 비율 스탯(BB/HBP, 1B, 2B, 3B, HR, out)으로부터 base-out 상태를 몬테카를로로 근사하는 라인업 시뮬레이터입니다(The Book 계열의 단순화 모델). 기본 모드는 고정 확률 상수 2개로 주루를 결정하는 단순 모델(도루/병살/희생타 없음)이지만, `situational_probs` 인자로 `build_situational_probs.py`의 KBO 실측 확률표를 넘기면 아웃카운트별 조건부 진루확률 + 병살 + 도루 로직이 추가로 적용됩니다(아래 `scrape_kbo_relay.py` 참고). 이 인자를 넘기지 않는 모든 호출(MLB 등)은 기존 단순 모델 그대로 동작해 영향받지 않습니다. 절대 득점 수준보다 슬롯 간 상대 비교 용도로 리그 R/G에 맞춰 캘리브레이션합니다.

`kim_doyoung_analysis.py`는 이를 이용해 "기아는 왜 2도영(2번 김도영)을 안 하는가"를 분석합니다. 시뮬레이션이 실제로 1-9번 전 슬롯을 탐색하기 때문에, '강한 2번' 가설과 무관하게 결과가 어느 슬롯을 가리키든 그대로 나옵니다.
- KBO가 MLB보다 팀 최고타자를 3-4번에 두는 관행이 강한지(팀-시즌별 최고 생산 슬롯 분포, MLB의 1-2번 이동 추세)
- KIA 2024 실제 라인업에서 김도영을 1-9번 어디에 두는 것이 팀 득점에 유리한지 시뮬레이션으로 검증

결과는 `outputs/leadoff_analysis/10-11_*.png`, `kim_doyoung_sim.json`에 저장됩니다.

### `best_hitter_move_sim.py` / `best_hitter_move_sim_mlb.py` (공용 코어: `best_hitter_move_core.py`)
KBO·MLB 두 스크립트는 데이터 스키마(포지션 코드, 최고타자 랭킹 지표, 이름 매칭 방식)만 다르고 파이프라인은 동일해, 공용 로직을 `best_hitter_move_core.py`에 두고 각 스크립트는 리그별 설정(`LeagueConfig`)만 정의하는 얇은 엔트리포인트입니다.

김도영 사례는 KIA 2024 1개 팀-시즌에 불과해 일반화할 수 없습니다. 이 스크립트는 같은 반사실(counterfactual) 방법 — 실제 로스터·실력은 고정하고 대상 선수만 타순 이동 — 을 KBO 5시즌×10구단 = 50개 팀-시즌 전체에 적용합니다. '주전 9인'은 boxscore의 포지션 필드(포/一/二/三/유/좌/중/우/지)를 이용해 **포지션별로 그 자리 타석수가 가장 많은 선수**를 그리디로 배정해 구성합니다(한 선수가 2개 이상 포지션에서 동시에 1위인 경우가 50개 팀-시즌 중 40%로 흔해, 타석수가 가장 높은 주포지션에 배정하고 밀려난 자리는 차순위 선수로 채움). 이 9인 중 시즌 wOBA가 가장 높은 선수를 그 팀의 최고타자로 지정하고, 나머지 8인을 실제 순서에 고정한 채 그 선수를 실제 슬롯과 2번 슬롯에 각각 넣어 시뮬레이션한 팀 득점을 비교합니다(n_games=120,000/슬롯, 팀-시즌별로 독립적이라 병렬 실행). 대응표본(paired) t-test·Wilcoxon으로 50개 팀-시즌 전체의 평균 이득을 검정합니다.

결과는 `outputs/leadoff_analysis/14_best_hitter_move_all_teams.png`, `best_hitter_move_sim.json`, `best_hitter_move_table.csv`에 저장됩니다.

### `scrape_mlb_boxscores.py` / `scrape_mlb_players_all.py` / `best_hitter_move_sim_mlb.py`
KBO 50개 팀-시즌에서는 최고타자를 2번으로 옮겨도 평균 이득이 유의하지 않았습니다(-0.2점/144경기). 이게 KBO만의 현상인지, 아니면 반사실 시뮬레이션 자체의 한계인지 확인하기 위해 같은 방법론·같은 표본 크기(10팀×5년=50팀-시즌)를 MLB에 그대로 적용합니다.

`scrape_mlb_boxscores.py`는 KBO의 `scrape_naver_boxscores.py`에 대응하는 스크립트로, StatsAPI의 schedule + live feed 엔드포인트에서 경기별 타순·포지션·타격기록을 모아 `mlb_boxscore_batters_{year}.csv`로 저장합니다(`kbo_boxscore_batters_{year}.csv`와 동일 스키마라 `best_hitter_move_sim.py`의 로직을 그대로 재사용 가능). `scrape_mlb_players_all.py`는 기존 `scrape_mlb_player_hitting.py`(규정타석 이상만)와 달리 부분출전 선수까지 포함한 전체 타자 시즌기록을 모아 '포지션별 최다타석' 주전 선정과 최고타자 랭킹(OPS 기준)에 씁니다. KBO와 다른 점은 포지션 코드가 이미 영문 단일값(C/1B/.../DH)이라 첫글자 추출이 불필요하고, 최고타자 랭킹에 wOBA 대신 OPS를 쓴다는 것(팀-연도 내 랭킹 목적이라 지표 선택이 결과에 영향 없음), 2021년 유니버설 DH 이전 내셔널리그 팀은 투수가 타석에 서는 경기가 섞여 있어 포지션 배정에 폴백 로직이 필요하다는 것입니다.

결과는 `outputs/leadoff_analysis/15_best_hitter_move_all_teams_mlb.png`, `best_hitter_move_sim_mlb.json`, `best_hitter_move_table_mlb.csv`에 저장됩니다.

### `slot_slg_trend.py`
'강한 2번' 보조자료로, 타순별(1-9번) 장타율(SLG) 연도 추이를 선그래프로 그립니다. MLB는 2015-2025년(`data/raw/mlb_slot_splits.csv`), KBO는 Statiz 크롤링 금지 공지 이전에 확보한 2021-2025년(`data/raw/statiz_slot_agg_*.csv`)만 다룹니다.

결과는 `outputs/leadoff_analysis/12_mlb_slot_slg_trend.png`, `13_kbo_slot_slg_trend.png`에 저장됩니다.

### `scrape_kbo_relay.py` / `build_situational_probs.py` / `lineup_optimize_all_teams.py`
`best_hitter_move_sim.py`(KBO)와 `kim_doyoung_analysis.py`의 라인업 전체 재배열 실험 모두 이득이 The Book이 인용하는 시즌 5~15점(144경기 환산 약 4.4~13.3점)보다 훨씬 작았습니다. 원인 후보는 `lineup_sim.py`가 주루 진루를 고정 확률 상수 2개로만 결정하고 병살·도루를 아예 반영하지 않는다는 점이었습니다. 이 세 스크립트는 실제 KBO 데이터로 상황조건부 확률표를 만들어 엔진을 업그레이드하고, 같은 "최적 vs 실제" 비교를 재실행합니다.

`scrape_kbo_relay.py`는 Naver Sports의 비공식 relay API(`/schedule/games/{gameId}/relay?inning=N`, `scrape_naver_boxscores.py`가 쓰는 것과 같은 API 계열)에서 KBO 경기 매 플레이의 베이스-아웃 상태를 원본 그대로 수집합니다(`inning=`이 실제 페이지네이션 파라미터임을 실측으로 확인 — `no`/`startNo` 등은 무시됨). `base1/base2/base3` 필드는 불리언이 아니라 그 루에 있는 주자의 타순번호(0=비어있음)라, 이름 매칭 없이도 연속 이벤트를 비교해 주자의 실제 이동 경로를 추적할 수 있습니다. 5개 시즌(2021-2025) 3,615경기, 약 197만 행을 수집했으며 안타 총계가 boxscore 합계와 정확히 일치함을 검증했습니다(`data/raw/kbo_relay_events_{year}.csv`).

`build_situational_probs.py`는 이 원본에서 타석(PA) 단위로 이전/이후 베이스-아웃 상태와 결과 유형을 추출해, 아웃카운트별 1루주자 진루확률(단타/2루타 시), 병살확률(`lineup_sim.py`의 뭉뚱그린 'out' 이벤트 정의에 맞춘 분모 사용 — 11.7~12.1%로 통상치와 일치), 도루 시도·성공률을 계산합니다(`data/processed/kbo_situational_probs.json`). 새로 뽑은 진루확률(단타 시 1루주자→3루 26.9%, 2루타 시 득점 44.7%)이 기존 고정 상수(0.28, 0.45)와 거의 일치해 교차검증도 됐습니다.

`lineup_optimize_all_teams.py`는 `kim_doyoung_analysis.py`의 `optimize_order`(pairwise-swap 언덕등반 탐색)를 팀 무관하게 일반화하고 이 확률표를 적용해, KBO 50개 팀-시즌 전체에서 "최적 타순 vs 실제 타순" 이득을 재측정합니다.

결과는 `outputs/leadoff_analysis/16_lineup_optimize_all_teams.png`, `lineup_optimize_sim.json`, `lineup_optimize_table.csv`에 저장됩니다.

## 결과 요약

**기본 회귀 (`leadoff_analysis.py`)** — MLB에서는 2번타자 질이 팀 득점에 뚜렷한 독립 효과를 갖지만(표준화 β=0.300, p<0.001, R²=0.835), KBO에서는 나머지 라인업(rest_woba, β=0.534, p=0.003)을 통제하면 2번타자 자체의 효과는 유의하지 않습니다(β=0.106, p=0.537, R²=0.379, n=50). 다만 통제 없이 KBO 2번타자 wRC+ 상/하위 50% 그룹만 비교하면 팀 득점 차이는 유의합니다(4.96 vs 4.65 R/G, Welch t p=0.021) — 즉 2번타자 질과 팀 전력은 함께 움직이지만, KBO 표본(n=50)에서는 순수 독립효과를 통계적으로 분리해내기 어렵습니다.

**'강한 2번' 가설 검증 (H1-H3)**
- H1 (희소성): OBP+ISO를 동시에 갖춘 '콤보' 선수 비율은 KBO 13.7% vs MLB 16.1%로 통계적으로 유의한 차이가 아님(p=0.36). 절대 수준에서는 KBO가 OBP는 유의하게 높고(0.363 vs 0.332, p<0.001) ISO는 유의하게 낮아(0.146 vs 0.181, p<0.001), 콘택트 중심·저삼진(K% 16.1% vs 20.6%) 리그 환경 차이가 원인에 더 가까워 보입니다. 특히 KBO 50개 팀-시즌 중 주전 2번타자가 '콤보-엘리트'였던 경우는 **0건**이었습니다.
- H2 (라인업 뎁스): 2번-4번 타순 간 인재 트레이드오프 상관관계는 KBO(r=0.47, p<0.001)가 MLB(r=0.13, p=0.004)보다 뚜렷이 크고, 그 차이도 통계적으로 유의합니다(p=0.015) — 리드오프(1번) 기준으로는 보이지 않았던 뎁스 제약이 2번 기준에서는 나타납니다. 다만 라인업 전체의 '평탄도'(9개 슬롯 생산력의 표준편차) 자체는 KBO와 MLB 간 유의한 차이가 없습니다(p=0.124).
- H3 (득점 환경): KBO 회귀에서는 2번타자 계수가 유의하지 않았고(p=0.72) MLB는 강하게 유의(p<1e-31)해, 리그 자체의 구조적 차이가 득점 환경보다 커 보입니다. MLB를 2015년 전/후로 나누면 2번타자 효과가 더 커졌습니다(β 0.21→0.26), 세이버매트릭스 확산 시점과 대략 맞물립니다.

**2번타자 프로필**: KBO 2번은 ISO 0.109·BB% 9.4%·K% 16.6%로, MLB 2번(ISO 0.162·BB% 8.4%·K% 19.3%)보다 파워는 낮고 삼진은 적은 콘택트형입니다.

**김도영 라인업 사례** — KBO는 최고타자를 3-4번에 두는 비중이 78%로 MLB(59%)보다 뚜렷이 높습니다(평균 최고생산 슬롯 KBO 3.18번 vs MLB 3.37번). 김도영만 슬롯을 옮겨보는 시뮬레이션에서는 실제 사용 타순(3번, 팀 득점 6.21 R/G)이 아닌 **2번**(6.24 R/G)이 최적으로 나와 '강한 2번' 가설과 일치합니다(144경기 환산 +3.7점). 다만 라인업 전체를 재배열하는 최적화에서는 김도영이 오히려 1번으로 가는 해가 나왔고, 그때 이득은 경기당 0.015점(144경기 +2.1점) 수준으로 사실상 잡음에 가깝습니다 — 두 반사실 실험이 서로 다른 슬롯을 가리키는 만큼, "2번이 압도적으로 낫다"기보다는 "실제 관행(3번)보다는 1~2번 쪽이 근소하게 낫다" 정도로 해석하는 게 안전합니다.

**50개 팀-시즌 전체 반사실 검증 (`best_hitter_move_sim.py`)** — 김도영의 +3.7점/144경기가 리그 전체에 일반화되는지 확인하기 위해, 같은 방법(로스터 고정, 타순만 이동)을 KBO 5시즌×10구단 전체 팀-시즌으로 확장했습니다. 주전 9인은 포지션별 최다타석 선수로 구성했고(총 타석 상위 9명 방식과 비교해도 결론은 동일 — 아래 참고), 각 팀-시즌의 실제 최고타자(시즌 wOBA 기준)만 2번으로 옮겼을 때 평균 이득은 **-0.2점/144경기**(95% CI [-0.7, +0.3], n_games=120,000/슬롯), 대응표본 t-test p=0.483·Wilcoxon p=0.498로 **0과 통계적으로 구별되지 않았습니다**. 이득을 본 팀-시즌은 50개 중 23개(46%)였습니다(시뮬레이션 캘리브레이션 오차는 실제 팀 R/G 대비 평균 절대 0.22점 수준으로 양호). 주전 선정 방식을 "팀 전체 타석 상위 9명"으로 바꿔 돌려도 평균 -0.2점/144경기로 사실상 동일한 결과였습니다 — 방법론에 민감하지 않은 결론입니다. 김도영 본인은 이 정밀도에서 재확인해도 양수(+0.8~+2.5점, 선정 방식에 따라 소폭 변동)였지만, 리그 전체 평균은 정확히 0 근처에 걸려 있습니다 — 즉 김도영 사례가 완전히 우연은 아니었어도, 리그 평균으로 일반화되진 않습니다.

이는 처음 질문 — "KBO 2번타자 효과가 안 보이는 건 이득이 없어서인가, 아니면 이득은 있는데 관행 때문에 못 보는 것인가" — 에 실제로 답을 줍니다. 회귀분석은 관찰 데이터의 교란(강한 2번 기용팀=원래 전력 좋은 팀) 때문에 효과를 못 봤다고 해도, 그 교란을 원천적으로 우회하는 반사실 시뮬레이션조차 뚜렷한 평균 이득을 보여주지 못했습니다. H1(콤보-엘리트 선수의 2번 기용 0/50건)과 H2(2-4번 타순 트레이드오프 KBO 유의하게 강함)가 보여주는 강한 관행 자체는 사실이지만, 이 단순화된 시뮬레이션 안에서는 그 관행이 "명백한 득점 손실을 감수하면서" 유지되는 것으로는 보이지 않습니다 — "이득이 있는데 관행 때문에 안 한다"는 주장은 이 데이터로는 뒷받침되지 않습니다. (다만 `lineup_sim.py`는 병살·희생타·라인업 보호효과·투수 대응을 반영하지 않는 단순화 모델이라는 한계는 남아 있습니다.)

**MLB 50개 팀-시즌 반사실 검증 (`best_hitter_move_sim_mlb.py`)** — KBO의 null 결과가 KBO만의 현상인지 확인하기 위해 같은 방법·같은 표본 규모(10팀×5년=50팀-시즌, 2021-2025)를 MLB에 적용했습니다. 평균 이득은 **+0.5점/144경기**(95% CI [+0.1, +0.9], n_games=120,000/슬롯), 대응표본 t-test p=0.020·Wilcoxon p=0.040으로 **0과 통계적으로 구별됩니다** — 다만 The Book이 인용하는 라인업 최적화 전체 이득(144경기 환산 약 4.4~13.3점)의 1/10 이하 수준으로 실질적 크기는 작습니다. 이득을 본 팀-시즌은 50개 중 24개(48%)로 절반에 못 미치지만(소수의 큰 양수 사례가 평균을 끌어올리는 비대칭 분포), 시뮬레이션 캘리브레이션 오차는 평균 절대 0.26점으로 KBO와 비슷한 수준입니다. 즉 "최고타자를 2번에 세우면 이득이 있다"는 세이버매트릭스 통념은 MLB에서 통계적으로는 확인되지만 크기는 미미합니다 — KBO(-0.2점, 비유의)와 MLB(+0.5점, 유의하지만 작음)의 차이 자체도 실질적으로는 크지 않고, "최고타자만 옮기는" 국소적 개입이 라인업 최적화에서 큰 효과는 아니라는 쪽에 여전히 무게가 실립니다(`kim_doyoung_analysis.py`의 라인업 전체 재배열 실험에서 이득이 잡음 수준이었던 것과 일관됩니다).

(참고: 이전 버전은 시드를 `hash((year, team))`로 생성해 `team` 문자열의 파이썬 해시 솔팅 때문에 실행할 때마다 시드가 달라졌다 — 그때 MLB 결과는 +0.3점·p=0.119(비유의)였다. `zlib.crc32` 기반 결정론적 시드로 고친 뒤 재실행한 결과가 위 수치이며, 점추정치 자체는 크게 안 바뀌었지만(+0.3→+0.5점) 유의성 판정이 뒤집힐 만큼 유의확률 경계에 가까운 결과라는 뜻이다 — n_games=120,000에서도 몬테카를로 노이즈가 완전히 무시할 수준은 아니다.)

**엔진 업그레이드 후 재검증 (`scrape_kbo_relay.py` / `build_situational_probs.py` / `lineup_optimize_all_teams.py`)** — 위 두 반사실 검증(최고타자만 2번 이동)은 모두 이득이 거의 0이었지만, `lineup_sim.py` 자체가 주루 진루를 고정 확률 상수 2개로만 결정하고 병살·도루를 아예 반영하지 않는 단순화 모델이라는 한계가 남아 있었습니다. KBO 5개 시즌(2021-2025) 3,615경기의 Naver relay 매 플레이 데이터로 아웃카운트별 조건부 진루확률·병살확률(11.7~12.1%, 통상치와 일치)·도루확률(성공률 75.3%)을 실측하고, `lineup_sim.py`를 이 확률표로 업그레이드했습니다(30개 팀-시즌 캘리브레이션: 평균절대오차 0.226점/경기로 기존 모델과 비슷한 수준, 다만 병살·도루로 아웃이 늘어 평균 -0.16점 정도 낮게 나오는 방향으로 편향).

이 업그레이드된 엔진으로 `kim_doyoung_analysis.py`의 `optimize_order`(라인업 전체 재배열, pairwise-swap 언덕등반 탐색)를 KBO 50개 팀-시즌 전체로 확장해 재실행한 결과, 평균 이득이 **+3.0점/144경기**(95% CI [+1.9, +4.0])로 커졌고, 대응표본 t-test·Wilcoxon 모두 **p<0.00001로 뚜렷하게 유의**해졌습니다(기존 단순 모델은 김도영 1개 팀-시즌만으로 +2.1점, 유의성 검정 자체를 안 했음). 이득을 본 팀-시즌은 50개 중 39개(78%)였습니다. 즉 **"KBO 라인업 최적화 이득이 거의 없다"는 이전 결론은 상당 부분 시뮬레이터의 단순화(병살·도루 미반영) 때문이었다는 게 확인됐습니다** — 실제 데이터 기반 상황조건부 확률을 반영하니 이득이 커지고 통계적으로도 명확해졌습니다. 다만 The Book이 인용하는 시즌 5~15점(144경기 환산 약 4.4~13.3점) 범위에는 CI 상단(4.0)이 근접했을 뿐 완전히 들어오지는 않았습니다 — 남은 차이는 이 모델에 아직 없는 요소(희생번트, 선수별 GIDP/도루 성향, 구장효과) 때문일 수도 있고, The Book의 범위 자체가 MLB 기준이라 KBO에 그대로 적용되지 않을 수도 있습니다.

전체 수치는 `outputs/leadoff_analysis/hypothesis_results.json`, `stats_summary.json`, `kim_doyoung_sim.json`, `best_hitter_move_sim.json`, `best_hitter_move_sim_mlb.json`, `lineup_optimize_sim.json` 참고.

## 실행 방법

```bash
pip install -r requirements.txt

python src/hitting_eda/scrape_kbo.py       # 타자/투수 기록 수집
python src/h2h/scrape_h2h.py               # 상대전적 수집
python src/hitting_eda/eda.py              # 정제 + 차트 생성
python src/hitting_eda/model.py            # baseline 모델 평가
python src/h2h/h2h_test.py                 # 상대전적 이항검정 (연도/팀은 파일 내 main() 인자로 지정)

python src/leadoff/scrape_naver_boxscores.py   # KBO 경기별 타순 박스스코어 수집
python src/leadoff/scrape_mlb_splits.py        # MLB 팀×타순 스플릿 수집
python src/leadoff/scrape_mlb_team_runs.py     # MLB 팀-시즌 득점 수집
python src/leadoff/build_leadoff_dataset.py    # 리드오프 분석용 데이터셋 구축
python src/leadoff/leadoff_analysis.py         # KBO vs MLB 리드오프 전략 회귀분석 + 차트

python src/leadoff/scrape_mlb_player_hitting.py  # MLB 개인 타자 시즌기록 수집
python src/leadoff/leadoff_hypotheses.py         # 리드오프 3가설 검증 + 차트
python src/leadoff/kim_doyoung_analysis.py       # 라인업 관행 분석 + 김도영 시뮬레이션
python src/leadoff/best_hitter_move_sim.py       # 최고타자 2번 이동 반사실 검증 (KBO 50개 팀-시즌)

python src/leadoff/scrape_mlb_boxscores.py       # MLB 경기별 타순/포지션 박스스코어 수집
python src/leadoff/scrape_mlb_players_all.py     # MLB 팀x시즌 전체 타자 시즌기록 수집
python src/leadoff/best_hitter_move_sim_mlb.py   # 최고타자 2번 이동 반사실 검증 (MLB 50개 팀-시즌)
```
