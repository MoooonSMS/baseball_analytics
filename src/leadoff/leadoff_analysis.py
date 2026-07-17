"""'강한 2번' 전략: KBO vs MLB 비교 분석.

세이버매트릭스 통찰: 2번 타순은 1번과 거의 같은 PA를 받으면서도(1번은 항상
주자 없이 시작하는 첫 타석이 있는 반면, 2번은 그 비중이 낮아 주자 있는
상황에 더 자주 타석에 들어선다) 팀 최고 타자를 배치하기에 1번보다 유리하다는
것이 현대 세이버매트릭스(The Book 등)의 핵심 결론이다. 이 분석은 그 관점에서
'강한 2번' 전략을 KBO/MLB 데이터로 검증한다.

가설
  H1  KBO 2번 타자의 질(wOBA)이 높을수록 팀 득점(R/G)이 늘어난다
      (나머지 라인업 질 통제).
  H2  MLB에서도 동일 관계가 성립하며, 최근으로 올수록 2번 슬롯의 리그 평균
      대비 상대생산성이 다른 슬롯보다 뚜렷하게 개선되어 왔다.
  H3  KBO 2번 슬롯은 MLB 2번 슬롯보다 '작전 야구'(희생번트) 비중이 높고
      파워/출루 지표가 낮아, 전통적 '작전형 2번' 프로필에 가깝다.

출력: outputs/leadoff_analysis/*.png, 콘솔에 회귀·검정 결과 요약
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "leadoff_analysis"

BLUE = "#2a78d6"    # KBO
ORANGE = "#eb6834"  # MLB
GRAY = "#898781"    # 기준선/보조


def setup_style():
    sns.set_theme(style="whitegrid", font="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------- 데이터 로드
def load_all():
    kbo_slot = pd.read_csv(PROC / "kbo_slot_profile.csv", encoding="utf-8-sig")
    mlb_slot = pd.read_csv(PROC / "mlb_slot_profile.csv", encoding="utf-8-sig")
    kbo_team = pd.read_csv(PROC / "kbo_team_no2.csv", encoding="utf-8-sig")
    mlb_split = pd.read_csv(RAW / "mlb_slot_splits.csv", encoding="utf-8-sig")
    mlb_runs = pd.read_csv(RAW / "mlb_team_runs.csv", encoding="utf-8-sig")
    return kbo_slot, mlb_slot, kbo_team, mlb_split, mlb_runs


def build_mlb_team_no2(mlb_split: pd.DataFrame, mlb_runs: pd.DataFrame) -> pd.DataFrame:
    """MLB 팀-시즌: 2번 슬롯 OPS, 나머지(1, 3~9번) PA가중 OPS, 팀 R/PA."""
    s = mlb_split.copy()
    for c in ["ops", "plateAppearances"]:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    slot2 = s[s["slot"] == 2][["season", "teamId", "team", "ops"]].rename(
        columns={"ops": "no2_ops"})
    rest = s[s["slot"] != 2]
    rest_ops = rest.groupby(["season", "teamId"]).apply(
        lambda g: np.average(g["ops"], weights=g["plateAppearances"]),
        include_groups=False).rename("rest_ops").reset_index()
    r = mlb_runs.copy()
    r["r_per_pa"] = r["runs"] / r["plateAppearances"]
    m = slot2.merge(rest_ops, on=["season", "teamId"]).merge(
        r[["season", "teamId", "r_per_pa", "gamesPlayed", "runs"]], on=["season", "teamId"])
    m["runs_pg"] = m["runs"] / m["gamesPlayed"]
    return m


# ---------------------------------------------------------------- (a) 추세
def chart_trend(kbo_slot, mlb_slot):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    mlb2 = mlb_slot[mlb_slot["slot"] == 2].sort_values("season")
    axes[0].plot(mlb2["season"], mlb2["rel_ops"], color=ORANGE, marker="o", markersize=4, linewidth=2)
    axes[0].axhline(1.0, color=GRAY, linewidth=1, linestyle="--")
    axes[0].set_title("MLB 2번타순 상대생산성 (rel. OPS, 2010-2025)")
    axes[0].set_xlabel("시즌")
    axes[0].set_ylabel("리그 평균 대비 (=1.0)")

    kbo2 = kbo_slot[kbo_slot["bo"] == 2].sort_values("year")
    axes[1].plot(kbo2["year"], kbo2["rel_woba"], color=BLUE, marker="o", markersize=4, linewidth=2)
    axes[1].axhline(1.0, color=GRAY, linewidth=1, linestyle="--")
    axes[1].set_title("KBO 2번타순 상대생산성 (rel. wOBA, 2021-2025)")
    axes[1].set_xlabel("시즌")
    axes[1].set_xticks(kbo2["year"])

    fig.suptitle("2번 슬롯의 리그 평균 대비 생산성 추이", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "01_trend_slot2_relative.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- (b) 희생번트%
def chart_sac_bunt(kbo_slot):
    g = kbo_slot.groupby("bo")["sh_pct"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [BLUE if bo == 2 else GRAY for bo in g["bo"]]
    ax.bar(g["bo"].astype(str), g["sh_pct"] * 100, color=colors)
    ax.set_xlabel("타순")
    ax.set_ylabel("희생번트 비율 (%, 타석당)")
    ax.set_title("KBO 타순별 희생번트 비율 (2021-2025 평균)")
    fig.tight_layout()
    fig.savefig(OUT / "02_kbo_sac_bunt_by_slot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return g


# ---------------------------------------------------------------- (c) KBO 산점도+회귀
def analyze_kbo_regression(kbo_team: pd.DataFrame):
    d = kbo_team.dropna(subset=["no2_woba", "rest_woba", "runs_pg"]).copy()
    X = sm.add_constant(d[["no2_woba", "rest_woba"]])
    model = sm.OLS(d["runs_pg"], X).fit()

    # 표준화 계수 (z-score) - 효과 크기 비교용
    dz = d.copy()
    for c in ["no2_woba", "rest_woba", "runs_pg"]:
        dz[c] = (dz[c] - dz[c].mean()) / dz[c].std()
    Xz = sm.add_constant(dz[["no2_woba", "rest_woba"]])
    modelz = sm.OLS(dz["runs_pg"], Xz).fit()

    # 산점도 + 단순회귀선(2번타자 wOBA vs 득점)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.regplot(data=d, x="no2_woba", y="runs_pg", ax=ax,
                scatter_kws={"color": BLUE, "alpha": 0.7, "s": 40},
                line_kws={"color": BLUE})
    ax.set_xlabel("2번타자 시즌 wOBA")
    ax.set_ylabel("팀 득점/경기 (R/G)")
    ax.set_title("KBO: 2번타자 질 vs 팀 득점 (2021-2025, 팀-시즌)")
    fig.tight_layout()
    fig.savefig(OUT / "03_kbo_no2_vs_runs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return model, modelz, d


def analyze_mlb_regression(mlb_team: pd.DataFrame):
    d = mlb_team.dropna(subset=["no2_ops", "rest_ops", "r_per_pa"]).copy()
    X = sm.add_constant(d[["no2_ops", "rest_ops"]])
    model = sm.OLS(d["r_per_pa"], X).fit()

    dz = d.copy()
    for c in ["no2_ops", "rest_ops", "r_per_pa"]:
        dz[c] = (dz[c] - dz[c].mean()) / dz[c].std()
    Xz = sm.add_constant(dz[["no2_ops", "rest_ops"]])
    modelz = sm.OLS(dz["r_per_pa"], Xz).fit()
    return model, modelz, d


# ---------------------------------------------------------------- (d) 프로필 비교
def chart_profile_compare(kbo_slot, mlb_slot):
    k2 = kbo_slot[kbo_slot["bo"] == 2][["iso", "bb_pct", "k_pct"]].mean()
    m2 = mlb_slot[mlb_slot["slot"] == 2][["iso", "bb_pct", "k_pct"]].mean()

    labels = ["ISO(순장타율)", "BB%(볼넷)", "K%(삼진)"]
    kbo_vals = [k2["iso"], k2["bb_pct"], k2["k_pct"]]
    mlb_vals = [m2["iso"], m2["bb_pct"], m2["k_pct"]]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, kbo_vals, width, label="KBO 2번 (2021-2025)", color=BLUE)
    ax.bar(x + width / 2, mlb_vals, width, label="MLB 2번 (2010-2025)", color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("비율")
    ax.set_title("2번 타자 프로필: KBO vs MLB")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "04_profile_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return k2, m2


# ---------------------------------------------------------------- (e) 상/하위 그룹
def chart_high_low_group(kbo_team: pd.DataFrame):
    d = kbo_team.dropna(subset=["no2_wrc", "runs_pg"]).copy()
    med = d["no2_wrc"].median()
    high = d[d["no2_wrc"] >= med]["runs_pg"]
    low = d[d["no2_wrc"] < med]["runs_pg"]
    tstat, pval = stats.ttest_ind(high, low, equal_var=False)
    u, pval_mw = stats.mannwhitneyu(high, low, alternative="two-sided")

    fig, ax = plt.subplots(figsize=(6, 5))
    means = [low.mean(), high.mean()]
    ses = [low.std(ddof=1) / np.sqrt(len(low)), high.std(ddof=1) / np.sqrt(len(high))]
    ax.bar(["2번타자 wRC+\n하위 50%", "2번타자 wRC+\n상위 50%"], means,
           yerr=ses, capsize=6, color=[GRAY, BLUE])
    ax.set_ylabel("팀 득점/경기 (R/G)")
    ax.set_title(f"2번타자 질 상/하위 그룹별 팀 득점\n(Welch t-test p={pval:.3f}, Mann-Whitney p={pval_mw:.3f})")
    fig.tight_layout()
    fig.savefig(OUT / "05_high_low_no2_group.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"n_high": len(high), "n_low": len(low), "mean_high": high.mean(),
            "mean_low": low.mean(), "t": tstat, "p_t": pval, "p_mw": pval_mw}


# ---------------------------------------------------------------- 사례 탐색
def find_strong_no2_cases(kbo_team: pd.DataFrame, kbo_players: pd.DataFrame):
    d = kbo_team.dropna(subset=["no2_wrc"]).copy()
    d["lg_avg_rpg"] = d.groupby("year")["runs_pg"].transform("mean")
    d["rpg_vs_lg"] = d["runs_pg"] - d["lg_avg_rpg"]
    top = d.sort_values("no2_wrc", ascending=False).head(8)
    return top[["year", "team", "no2_wrc", "runs_pg", "lg_avg_rpg", "rpg_vs_lg"]]


def check_2024_import_no2(games: pd.DataFrame):
    """2024년 외국인 타자(소크라테스/로하스/페라자)가 2번으로 나선 경기 실측치.

    원래 이 세 선수는 리드오프(1번) 실험 사례로 알려져 있어, 2번 타순
    기준으로 걸러내면 표본이 적거나 없을 수 있다."""
    names = ["소크라테", "로하스", "페라자"]  # 네이버 박스스코어 외국인명 4자 절삭 표기
    g = games[(games["year"] == 2024)]
    rows = []
    for n in names:
        sub = g[g["name"].str.contains(n, na=False)]
        if len(sub) == 0:
            continue
        rows.append({
            "name": n, "team": sub["team"].iloc[0], "starts": len(sub),
            "team_runs_pg_when_no2": sub["teamScore"].mean(),
            "season_wrc": sub["season_wrc"].iloc[0],
        })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    kbo_slot, mlb_slot, kbo_team, mlb_split, mlb_runs = load_all()
    kbo_games = pd.read_csv(PROC / "kbo_no2_games.csv", encoding="utf-8-sig")
    kbo_players = pd.read_csv(PROC / "kbo_players_woba.csv", encoding="utf-8-sig")
    mlb_team = build_mlb_team_no2(mlb_split, mlb_runs)

    chart_trend(kbo_slot, mlb_slot)
    sac = chart_sac_bunt(kbo_slot)

    kbo_model, kbo_modelz, kbo_d = analyze_kbo_regression(kbo_team)
    mlb_model, mlb_modelz, mlb_d = analyze_mlb_regression(mlb_team)

    k2, m2 = chart_profile_compare(kbo_slot, mlb_slot)
    grp = chart_high_low_group(kbo_team)
    top_cases = find_strong_no2_cases(kbo_team, kbo_players)
    imports_2024 = check_2024_import_no2(kbo_games)

    print("=" * 70)
    print("[H3] KBO 타순별 희생번트% (2번이 3~6번보다 높은가?)")
    print(sac.round(4).to_string(index=False))

    print("\n[H1] KBO OLS: runs_pg ~ no2_woba + rest_woba  (n={})".format(len(kbo_d)))
    print(kbo_model.summary().tables[1])
    print("표준화 계수(beta):\n", kbo_modelz.params.round(3))

    print("\n[H2] MLB OLS: r_per_pa ~ no2_ops + rest_ops  (n={})".format(len(mlb_d)))
    print(mlb_model.summary().tables[1])
    print("표준화 계수(beta):\n", mlb_modelz.params.round(3))

    print("\n[H1 보조] 2번타자 wRC+ 상/하위 그룹 R/G 비교:", grp)

    print("\n[사례] KBO 2번타자 wRC+ 상위 8개 팀-시즌:")
    print(top_cases.round(2).to_string(index=False))

    print("\n[사례] 2024년 외국인 타자의 2번 기용 실측:")
    print(imports_2024.round(2).to_string(index=False))

    print("\n[프로필] KBO 2번:", k2.round(4).to_dict())
    print("[프로필] MLB 2번:", m2.round(4).to_dict())

    # 결과 요약을 리포트용으로 저장
    summary = {
        "kbo_model_params": kbo_model.params.to_dict(),
        "kbo_model_pvalues": kbo_model.pvalues.to_dict(),
        "kbo_model_rsq": kbo_model.rsquared,
        "kbo_modelz_params": kbo_modelz.params.to_dict(),
        "mlb_model_params": mlb_model.params.to_dict(),
        "mlb_model_pvalues": mlb_model.pvalues.to_dict(),
        "mlb_model_rsq": mlb_model.rsquared,
        "mlb_modelz_params": mlb_modelz.params.to_dict(),
        "group_test": grp,
    }
    import json
    with open(OUT / "stats_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    top_cases.to_csv(OUT / "top_no2_cases.csv", index=False, encoding="utf-8-sig")
    imports_2024.to_csv(OUT / "imports_2024.csv", index=False, encoding="utf-8-sig")
    print(f"\n차트/요약 저장 완료: {OUT}")


if __name__ == "__main__":
    main()
