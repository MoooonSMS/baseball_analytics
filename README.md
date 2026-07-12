# KBO Baseball Analytics

KBO(한국프로야구) 공식기록실 데이터를 스크래핑해서 정리하고, 탐색적 분석과 간단한 모델링, 팀 간 상대전적 통계 검정까지 해보는 개인 프로젝트입니다.

## 구성

```
src/
  scrape_kbo.py    # 타자/투수 기본기록 스크래핑 (2021~2025)
  scrape_h2h.py     # 팀 간 상대전적표(승-패-무) 스크래핑
  eda.py            # 데이터 정제 + 탐색적 분석 차트 생성
  model.py          # 올해 성적으로 다음 시즌 3할타자 여부를 예측하는 baseline 모델
  h2h_test.py       # Log5 기대 승률 대비 실제 상대전적이 통계적으로 이례적인지 이항검정

data/
  raw/              # 스크래핑 원본 CSV
  processed/        # 정제된 데이터 (hitters.csv)

outputs/            # 차트(PNG) 및 상대전적 분석 엑셀
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

## 실행 방법

```bash
pip install -r requirements.txt

python src/scrape_kbo.py       # 타자/투수 기록 수집
python src/scrape_h2h.py       # 상대전적 수집
python src/eda.py              # 정제 + 차트 생성
python src/model.py            # baseline 모델 평가
python src/h2h_test.py         # 상대전적 이항검정 (연도/팀은 파일 내 main() 인자로 지정)
```
