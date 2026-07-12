"""수집한 KBO 타자 기록을 정제하고 기본적인 EDA 차트를 뽑는다."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

# KBO 공식기록실 표기 -> 다루기 쉬운 영문 컬럼명
HITTER_COLS = {
    "선수명": "name",
    "팀명": "team",
    "AVG": "avg",
    "G": "games",
    "PA": "pa",
    "AB": "ab",
    "R": "runs",
    "H": "hits",
    "2B": "doubles",
    "3B": "triples",
    "HR": "hr",
    "TB": "tb",
    "RBI": "rbi",
    "BB": "bb",
    "HBP": "hbp",
    "SO": "so",
    "GDP": "gdp",
    "SLG": "slg",
    "OBP": "obp",
    "OPS": "ops",
    "year": "year",
}


def load_hitters() -> pd.DataFrame:
    frames = []
    for path in sorted(RAW_DIR.glob("hitters_*.csv")):
        frames.append(pd.read_csv(path, encoding="utf-8-sig"))
    df = pd.concat(frames, ignore_index=True)

    keep = [c for c in HITTER_COLS if c in df.columns]
    df = df[keep].rename(columns=HITTER_COLS)

    numeric_cols = [c for c in df.columns if c not in ("name", "team")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["pa"])
    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_hitters()
    df.to_csv(PROCESSED_DIR / "hitters.csv", index=False, encoding="utf-8-sig")

    print(df.describe())
    print("\n연도별 선수 수:\n", df.groupby("year")["name"].count())

    sns.set_theme(style="whitegrid", font="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False

    # 1) OPS 분포
    plt.figure(figsize=(8, 5))
    sns.histplot(x=df["ops"].dropna(), bins=30)
    plt.title("KBO 타자 OPS 분포")
    plt.savefig(OUTPUT_DIR / "ops_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2) 팀별 평균 OPS
    plt.figure(figsize=(9, 5))
    team_ops = df.groupby("team")["ops"].mean().sort_values(ascending=False)
    sns.barplot(x=team_ops.values, y=team_ops.index)
    plt.title("팀별 평균 OPS")
    plt.xlabel("OPS")
    plt.savefig(OUTPUT_DIR / "team_avg_ops.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3) 홈런-타율 관계
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=df, x="hr", y="avg", hue="year", palette="viridis")
    plt.title("홈런 vs 타율")
    plt.savefig(OUTPUT_DIR / "hr_vs_avg.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4) 주요 지표 상관관계
    corr_cols = ["avg", "obp", "slg", "ops", "hr", "rbi", "bb", "so"]
    corr = df[corr_cols].corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("타자 지표 상관관계")
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n차트 저장 완료: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
