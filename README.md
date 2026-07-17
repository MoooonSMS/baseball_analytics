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
    slot_slg_trend.py          # 타순별(1-9번) 장타율 연도 추이 보조 차트

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
`lineup_sim.py`는 타자 시즌 비율 스탯(BB/HBP, 1B, 2B, 3B, HR, out)으로부터 base-out 상태를 몬테카를로로 근사하는 라인업 시뮬레이터입니다(The Book 계열의 단순화 모델). 도루/병살/희생타/투수교체 등은 반영하지 않으며, 절대 득점 수준보다 슬롯 간 상대 비교 용도로 리그 R/G에 맞춰 캘리브레이션합니다.

`kim_doyoung_analysis.py`는 이를 이용해 "기아는 왜 2도영(2번 김도영)을 안 하는가"를 분석합니다. 시뮬레이션이 실제로 1-9번 전 슬롯을 탐색하기 때문에, '강한 2번' 가설과 무관하게 결과가 어느 슬롯을 가리키든 그대로 나옵니다.
- KBO가 MLB보다 팀 최고타자를 3-4번에 두는 관행이 강한지(팀-시즌별 최고 생산 슬롯 분포, MLB의 1-2번 이동 추세)
- KIA 2024 실제 라인업에서 김도영을 1-9번 어디에 두는 것이 팀 득점에 유리한지 시뮬레이션으로 검증

결과는 `outputs/leadoff_analysis/10-11_*.png`, `kim_doyoung_sim.json`에 저장됩니다.

### `slot_slg_trend.py`
'강한 2번' 보조자료로, 타순별(1-9번) 장타율(SLG) 연도 추이를 선그래프로 그립니다. MLB는 2015-2025년(`data/raw/mlb_slot_splits.csv`), KBO는 Statiz 크롤링 금지 공지 이전에 확보한 2021-2025년(`data/raw/statiz_slot_agg_*.csv`)만 다룹니다.

결과는 `outputs/leadoff_analysis/12_mlb_slot_slg_trend.png`, `13_kbo_slot_slg_trend.png`에 저장됩니다.

## 결과 요약

**기본 회귀 (`leadoff_analysis.py`)** — MLB에서는 2번타자 질이 팀 득점에 뚜렷한 독립 효과를 갖지만(표준화 β=0.300, p<0.001, R²=0.835), KBO에서는 나머지 라인업(rest_woba, β=0.534, p=0.003)을 통제하면 2번타자 자체의 효과는 유의하지 않습니다(β=0.106, p=0.537, R²=0.379, n=50). 다만 통제 없이 KBO 2번타자 wRC+ 상/하위 50% 그룹만 비교하면 팀 득점 차이는 유의합니다(4.96 vs 4.65 R/G, Welch t p=0.021) — 즉 2번타자 질과 팀 전력은 함께 움직이지만, KBO 표본(n=50)에서는 순수 독립효과를 통계적으로 분리해내기 어렵습니다.

**'강한 2번' 가설 검증 (H1-H3)**
- H1 (희소성): OBP+ISO를 동시에 갖춘 '콤보' 선수 비율은 KBO 13.7% vs MLB 16.1%로 통계적으로 유의한 차이가 아님(p=0.36). 절대 수준에서는 KBO가 OBP는 유의하게 높고(0.363 vs 0.332, p<0.001) ISO는 유의하게 낮아(0.146 vs 0.181, p<0.001), 콘택트 중심·저삼진(K% 16.1% vs 20.6%) 리그 환경 차이가 원인에 더 가까워 보입니다. 특히 KBO 50개 팀-시즌 중 주전 2번타자가 '콤보-엘리트'였던 경우는 **0건**이었습니다.
- H2 (라인업 뎁스): 2번-4번 타순 간 인재 트레이드오프 상관관계는 KBO(r=0.47, p<0.001)가 MLB(r=0.13, p=0.004)보다 뚜렷이 크고, 그 차이도 통계적으로 유의합니다(p=0.015) — 리드오프(1번) 기준으로는 보이지 않았던 뎁스 제약이 2번 기준에서는 나타납니다. 다만 라인업 전체의 '평탄도'(9개 슬롯 생산력의 표준편차) 자체는 KBO와 MLB 간 유의한 차이가 없습니다(p=0.124).
- H3 (득점 환경): KBO 회귀에서는 2번타자 계수가 유의하지 않았고(p=0.72) MLB는 강하게 유의(p<1e-31)해, 리그 자체의 구조적 차이가 득점 환경보다 커 보입니다. MLB를 2015년 전/후로 나누면 2번타자 효과가 더 커졌습니다(β 0.21→0.26), 세이버매트릭스 확산 시점과 대략 맞물립니다.

**2번타자 프로필**: KBO 2번은 ISO 0.109·BB% 9.4%·K% 16.6%로, MLB 2번(ISO 0.162·BB% 8.4%·K% 19.3%)보다 파워는 낮고 삼진은 적은 콘택트형입니다.

**김도영 라인업 사례** — KBO는 최고타자를 3-4번에 두는 비중이 78%로 MLB(59%)보다 뚜렷이 높습니다(평균 최고생산 슬롯 KBO 3.18번 vs MLB 3.37번). 김도영만 슬롯을 옮겨보는 시뮬레이션에서는 실제 사용 타순(3번, 팀 득점 6.21 R/G)이 아닌 **2번**(6.24 R/G)이 최적으로 나와 '강한 2번' 가설과 일치합니다(144경기 환산 +3.7점). 다만 라인업 전체를 재배열하는 최적화에서는 김도영이 오히려 1번으로 가는 해가 나왔고, 그때 이득은 경기당 0.015점(144경기 +2.1점) 수준으로 사실상 잡음에 가깝습니다 — 두 반사실 실험이 서로 다른 슬롯을 가리키는 만큼, "2번이 압도적으로 낫다"기보다는 "실제 관행(3번)보다는 1~2번 쪽이 근소하게 낫다" 정도로 해석하는 게 안전합니다.

전체 수치는 `outputs/leadoff_analysis/hypothesis_results.json`, `stats_summary.json`, `kim_doyoung_sim.json` 참고.

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
```
