"""'강한 2번' 보조자료: 타순별(1~9번) 장타율(SLG) 연도 추이.

입력: data/raw/mlb_slot_splits.csv (MLB, 2015-2025)
      data/raw/statiz_slot_agg_{y}.csv (KBO, 2021-2025 — Statiz 크롤링 금지 공지로 그 이전 연도는 미포함)
출력: outputs/leadoff_analysis/12_mlb_slot_slg_trend.png
      outputs/leadoff_analysis/13_kbo_slot_slg_trend.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "leadoff_analysis"

MLB_YEAR_MIN, MLB_YEAR_MAX = 2015, 2025
KBO_YEARS = [2021, 2022, 2023, 2024, 2025]

# tab10류 정성적 팔레트: 인접 타순끼리도 색상이 뚜렷이 구분되도록
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#17becf", "#bcbd22"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]


def load_mlb_slot_slg() -> pd.DataFrame:
    m = pd.read_csv(RAW / "mlb_slot_splits.csv")
    m = m[(m["season"] >= MLB_YEAR_MIN) & (m["season"] <= MLB_YEAR_MAX) & m["slot"].between(1, 9)]
    agg = m.groupby(["season", "slot"], as_index=False).agg(
        ab=("atBats", "sum"), tb=("totalBases", "sum"))
    agg["slg"] = agg["tb"] / agg["ab"]
    return agg.rename(columns={"season": "year", "slot": "slot"})


def load_kbo_slot_slg() -> pd.DataFrame:
    dfs = [pd.read_csv(RAW / f"statiz_slot_agg_{y}.csv") for y in KBO_YEARS]
    a = pd.concat(dfs, ignore_index=True)
    agg = a.groupby(["year", "bo"], as_index=False).agg(
        ab=("AB", "sum"), tb=("TB", "sum"))
    agg["slg"] = agg["tb"] / agg["ab"]
    return agg.rename(columns={"bo": "slot"})


def plot(df: pd.DataFrame, year_min: int, year_max: int, title: str, out_name: str):
    sns.set_theme(style="whitegrid", font="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for slot, color, marker in zip(range(1, 10), COLORS, MARKERS):
        s = df[df["slot"] == slot].sort_values("year")
        ax.plot(s["year"], s["slg"], color=color, linewidth=2.2, marker=marker,
                markersize=5, label=f"{slot}번")

    ax.set_xlabel("연도")
    ax.set_ylabel("장타율 (SLG)", labelpad=10)
    ax.set_title(title)
    ax.set_xticks(range(year_min, year_max + 1))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="타순", loc="center left", bbox_to_anchor=(1.02, 0.5),
               frameon=False)
    ax.margins(x=0.03)
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"저장 완료: {out_path}")


def main():
    mlb = load_mlb_slot_slg()
    plot(mlb, MLB_YEAR_MIN, MLB_YEAR_MAX,
         f"MLB 타순별(1~9번) 장타율 추이 ({MLB_YEAR_MIN}-{MLB_YEAR_MAX})",
         "12_mlb_slot_slg_trend.png")

    kbo = load_kbo_slot_slg()
    plot(kbo, KBO_YEARS[0], KBO_YEARS[-1],
         f"KBO 타순별(1~9번) 장타율 추이 ({KBO_YEARS[0]}-{KBO_YEARS[-1]})",
         "13_kbo_slot_slg_trend.png")


if __name__ == "__main__":
    main()
