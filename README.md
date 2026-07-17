# KBO Baseball Analytics

KBO(한국프로야구) 공식기록실 데이터를 스크래핑해서 정리하고, 탐색적 분석과 간단한 모델링, 팀 간 상대전적 통계 검정까지 해보는 개인 프로젝트입니다.

## 구성

```
src/
  scrape_kbo.py             # 타자/투수 기본기록 스크래핑 (2021~2025)
  scrape_h2h.py              # 팀 간 상대전적표(승-패-무) 스크래핑
  eda.py                     # 데이터 정제 + 탐색적 분석 차트 생성
  model.py                   # 올해 성적으로 다음 시즌 3할타자 여부를 예측하는 baseline 모델
  h2h_test.py                # Log5 기대 승률 대비 실제 상대전적이 통계적으로 이례적인지 이항검정
  scrape_naver_boxscores.py  # 네이버 스포츠 API에서 경기별 타순(batOrder) 포함 박스스코어 수집
  scrape_mlb_splits.py       # MLB StatsAPI 팀×타순 스플릿 수집 (2010~2025)
  scrape_mlb_team_runs.py    # MLB StatsAPI 팀-시즌 총 득점/PA 수집
  metrics.py                 # Statiz 선형가중치 기반 wOBA/wRC+ 계산 유틸
  build_leadoff_dataset.py   # '강한 1번' 분석용 데이터셋 구축 (KBO+MLB)
  leadoff_analysis.py        # KBO vs MLB 리드오프 전략 회귀분석·가설검정·차트
  scrape_mlb_player_hitting.py  # MLB StatsAPI 규정타석 개인 타자 시즌기록 수집 (2021~2025)
  leadoff_hypotheses.py      # '강한 1번'이 KBO에서 안 통하는 이유 3가설 검증
  lineup_sim.py               # 타순(라인업) 몬테카를로 시뮬레이터
  kim_doyoung_analysis.py    # 김도영 라인업 최적화 사례 분석

data/
  raw/              # 스크래핑 원본 CSV (KBO 공식기록, Statiz, 네이버 박스스코어, MLB StatsAPI)
  processed/        # 정제된 데이터 (hitters.csv, kbo/mlb 타순·리드오프 데이터셋)

outputs/                # 차트(PNG) 및 상대전적 분석 엑셀
  leadoff_analysis/      # 리드오프 분석 차트, 회귀결과, 사례 데이터
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
'강한 1번' 전략 분석용 원천 데이터를 모으는 스크립트입니다. KBO 공식기록실은 시즌 누적 기록만 제공하고 경기별 타순이 없어, 네이버 스포츠 API에서 `batOrder`(타순) 필드가 포함된 경기별 박스스코어를 따로 수집합니다(2021~2025, 중단 후 재실행 시 이미 저장된 gameId는 건너뜀). MLB는 StatsAPI의 `sitCodes=b1~b9` 스플릿으로 팀×타순 스탯을, 타순 스플릿에는 없는 팀 시즌 총득점은 별도 엔드포인트로 수집합니다(2010~2025).

KBO 세이버매트릭스(팀×타순 집계, wOBA 선형가중치 상수, 타순별 선수 개인기록)는 [Statiz](https://www.statiz.co.kr)에서 받아 `data/raw/statiz_*.csv`로 저장해 사용합니다.

### `metrics.py` / `build_leadoff_dataset.py`
Statiz 연도별 선형가중치로 wOBA·wRC+를 계산하는 유틸(`metrics.py`)과, 위 원본들을 조인해 분석용 데이터셋(`data/processed/kbo_*`, `mlb_*`)을 만드는 스크립트입니다. 파크팩터는 반영하지 않습니다.

### `leadoff_analysis.py`
KBO가 MLB만큼 '강한 1번'(출루형·파워형 리드오프) 전략을 적극적으로 쓰지 않는 이유를 데이터로 검증합니다.
- KBO(50팀-시즌)·MLB(480팀-시즌)에서 팀 득점을 리드오프 타자 질 + 나머지 라인업 질로 회귀(OLS)해 두 리그의 효과 크기를 비교
- 리드오프 wRC+ 상/하위 그룹 간 팀 득점 차이 검정(Welch t-test, Mann-Whitney U)
- KBO vs MLB 1번타순의 리그 평균 대비 생산성 추이, 타순별 희생번트 비율, ISO/BB%/K% 프로필 비교 차트 생성
- 2024년 KBO 외국인 타자 리드오프 실험(KIA·KT·한화) 실측 성적 등 사례 탐색

결과는 `outputs/leadoff_analysis/`에 저장됩니다.

### `scrape_mlb_player_hitting.py`
MLB StatsAPI에서 규정타석(Qualified) 타자의 개인 시즌 기록을 수집합니다(2021~2025, KBO 데이터 존재 연도와 표본 정의를 맞춤). `leadoff_hypotheses.py`의 OBP+ISO 희소성 비교에 사용됩니다.

### `leadoff_hypotheses.py`
`leadoff_analysis.py`가 KBO와 MLB의 리드오프 효과 크기가 다르다는 것까지는 보였지만, *왜* 다른지는 설명하지 않습니다. 이 스크립트는 그 이유에 대한 세 가설을 사전에 방법론을 정한 뒤 검증합니다(데이터가 가설을 지지하지 않아도 그대로 보고).

- **H1 (희소성)**: MLB 리드오프가 통하는 건 OBP와 파워(ISO)를 동시에 갖춘 선수가 있어서다. KBO는 그런 선수가 상대적으로 희소하다.
- **H2 (라인업 뎁스)**: 1번에 최고 타자를 배치하면 4번에 쓸 선수가 부족해지는 트레이드오프가 KBO에서 더 크다.
- **H3 (득점 환경)**: 타고투저/투고투저 여부가 리드오프 효과를 조절한다.

결과는 `outputs/leadoff_analysis/06~09_*.png`, `hypothesis_results.json`에 저장됩니다.

### `lineup_sim.py` / `kim_doyoung_analysis.py`
`lineup_sim.py`는 타자 시즌 비율 스탯(BB/HBP, 1B, 2B, 3B, HR, out)으로부터 base-out 상태를 몬테카를로로 근사하는 라인업 시뮬레이터입니다(The Book 계열의 단순화 모델). 도루/병살/희생타/투수교체 등은 반영하지 않으며, 절대 득점 수준보다 슬롯 간 상대 비교 용도로 리그 R/G에 맞춰 캘리브레이션합니다.

`kim_doyoung_analysis.py`는 이를 이용해 "기아는 왜 1도영(리드오프 김도영)을 안 하는가"를 분석합니다.
- KBO가 MLB보다 팀 최고타자를 3~4번에 두는 관행이 강한지(팀-시즌별 최고 생산 슬롯 분포, MLB의 1~2번 이동 추세)
- KIA 2024 실제 라인업에서 김도영을 1~9번 어디에 두는 것이 팀 득점에 유리한지 시뮬레이션으로 검증

결과는 `outputs/leadoff_analysis/10~11_*.png`, `kim_doyoung_sim.json`에 저장됩니다.

## 결과 요약

**리드오프 가설 검증 (H1~H3)** — 세 가설 모두 KBO의 '약한 1번' 관행을 뚜렷하게 설명하지 못했습니다.
- H1: OBP+ISO를 동시에 갖춘 선수 비율은 KBO 13.7% vs MLB 16.1%로 통계적으로 유의한 차이가 아님(p=0.36). 다만 절대 수준에서는 KBO가 OBP는 유의하게 높고(0.363 vs 0.332, p<0.001) ISO는 유의하게 낮아(0.146 vs 0.181, p<0.001), 콘택트 중심·저삼진(K% 16.1% vs 20.6%) 리그 환경 차이가 원인에 더 가까워 보입니다. 실제로 KBO에서 '컴보' 선수를 리드오프로 쓴 팀-시즌은 50개 중 3개뿐이라 검정력 자체가 낮습니다.
- H2: 1번-4번 타순 간 인재 트레이드오프 상관관계는 KBO(r=0.31)와 MLB(r=0.29)가 비슷해(차이 유의하지 않음, p=0.91), 뎁스 차이로 보기 어렵습니다.
- H3: KBO 회귀에서는 리드오프 계수가 유의하지 않았고(p=0.83) MLB는 강하게 유의(p<1e-28)해, 리그 자체의 구조적 차이(계수 크기)가 득점 환경보다 커 보입니다.

**김도영 라인업 사례** — KBO는 최고타자를 3~4번에 두는 비중이 78%로 MLB(59%)보다 뚜렷이 높습니다(MLB는 2010년대 이후 1~2번으로 이동하는 추세, 최근 2023~2025년 26~57% 수준까지 상승). 시뮬레이션상 김도영을 실제 사용 타순(3번, 팀 득점 6.21 R/G)이 아닌 최적 타순(2번, 6.24 R/G)에 두면 144경기 환산 약 +3.7득점의 이득이 있지만, 라인업 전체를 재배열한 최적해와 실제 라인업의 차이는 경기당 0.015점 수준으로 사실상 미미합니다 — 즉 관행이 통계적으로 크게 손해를 보는 선택은 아니라는 결론입니다.

전체 수치는 `outputs/leadoff_analysis/hypothesis_results.json`, `kim_doyoung_sim.json` 참고.

## 실행 방법

```bash
pip install -r requirements.txt

python src/scrape_kbo.py       # 타자/투수 기록 수집
python src/scrape_h2h.py       # 상대전적 수집
python src/eda.py              # 정제 + 차트 생성
python src/model.py            # baseline 모델 평가
python src/h2h_test.py         # 상대전적 이항검정 (연도/팀은 파일 내 main() 인자로 지정)

python src/scrape_naver_boxscores.py   # KBO 경기별 타순 박스스코어 수집
python src/scrape_mlb_splits.py        # MLB 팀×타순 스플릿 수집
python src/scrape_mlb_team_runs.py     # MLB 팀-시즌 득점 수집
python src/build_leadoff_dataset.py    # 리드오프 분석용 데이터셋 구축
python src/leadoff_analysis.py         # KBO vs MLB 리드오프 전략 회귀분석 + 차트

python src/scrape_mlb_player_hitting.py  # MLB 개인 타자 시즌기록 수집
python src/leadoff_hypotheses.py         # 리드오프 3가설 검증 + 차트
python src/kim_doyoung_analysis.py       # 라인업 관행 분석 + 김도영 시뮬레이션
```
