"""아주 단순한 베이스라인 모델: 올해 성적으로 '다음 시즌 3할타자 여부'를 예측.

리스크모형팀에서 하는 PD(부도확률) 모델링과 구조가 비슷하게 맞춰봤다:
- 피처: 올해 시점에 관측 가능한 지표들
- 타깃: 다음 시즌에 실현되는 이진 결과 (3할 달성 여부)
- 로지스틱 회귀 baseline + ROC-AUC, confusion matrix로 평가
"""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

FEATURES = ["avg", "obp", "slg", "ops", "hr", "rbi", "so", "bb", "pa"]
MIN_PA = 100  # 규정타석 근처로 표본을 어느 정도 신뢰할 수 있는 선수만 사용


def build_dataset() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "hitters.csv", encoding="utf-8-sig")
    df = df[df["pa"] >= MIN_PA].copy()

    df = df.sort_values(["name", "year"])
    df["next_year"] = df["year"] + 1
    next_avg = df[["name", "year", "avg"]].rename(
        columns={"year": "next_year", "avg": "next_avg"}
    )
    merged = df.merge(next_avg, on=["name", "next_year"], how="inner")
    merged["target_300"] = (merged["next_avg"] >= 0.300).astype(int)
    return merged


def main():
    data = build_dataset()
    print(f"학습 가능 표본 수: {len(data)}")
    print("타깃 비율(다음 시즌 3할 달성):", round(data["target_300"].mean(), 3))

    X = data[FEATURES]
    y = data["target_300"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_s, y_train)

    proba = model.predict_proba(X_test_s)[:, 1]
    pred = model.predict(X_test_s)

    print("\nROC-AUC:", round(roc_auc_score(y_test, proba), 3))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
    print("\nClassification report:\n", classification_report(y_test, pred))

    coef = pd.Series(model.coef_[0], index=FEATURES).sort_values()
    print("\n피처 계수(표준화 후):\n", coef)


if __name__ == "__main__":
    main()
