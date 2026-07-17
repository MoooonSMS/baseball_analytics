"""'강한 1번'이 KBO에서 안 통하는 이유에 대한 세 가지 가설을 검증한다.

H1  파워+출루 겸비 희소성: MLB 리드오프가 통하는 건 OBP와 파워(ISO)를 동시에
    갖춘 선수가 있어서다. KBO는 그런 선수(상대적으로도)가 희소하다.
H2  라인업 뎁스: 1번에 최고 타자를 배치해도 4번(클린업)에 쓸 선수가 남아있어야
    하는데, KBO는 뎁스가 얕아 1번-4번 사이에 트레이드오프가 있다.
H3  득점 환경(타고투저/투고투저)이 리드오프 효과를 조절한다.

사전에 방법론(임계값, 검정 방식)을 확정한 뒤 결과를 본다 — 데이터가 가설을
지지하지 않아도 그대로 보고한다.

출력: outputs/leadoff_analysis/06~08_*.png, hypothesis_results.json
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

import leadoff_analysis as la

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "leadoff_analysis"
YEARS = [2021, 2022, 2023, 2024, 2025]

BLUE = "#2a78d6"    # KBO
ORANGE = "#eb6834"  # MLB
GRAY = "#898781"


def setup_style():
    sns.set_theme(style="whitegrid", font="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False


def fisher_r_to_z_diff(r1: float, n1: int, r2: float, n2: int) -> tuple[float, float]:
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z = (z1 - z2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


# ---------------------------------------------------------------- 데이터 로드
COLS = ["league", "year", "name", "team", "pa", "obp", "slg", "iso", "so", "bb", "woba", "wrc_plus"]


def load_kbo_players() -> pd.DataFrame:
    p = pd.read_csv(PROC / "kbo_players_woba.csv", encoding="utf-8-sig")
    p = p.rename(columns={"OBP": "obp", "SLG": "slg"})  # so, bb는 이미 소문자
    p["league"] = "KBO"
    return p[COLS]


def load_mlb_players() -> pd.DataFrame:
    dfs = [pd.read_csv(RAW / f"mlb_player_hitting_{y}.csv") for y in YEARS]
    m = pd.concat(dfs, ignore_index=True)
    m = m.rename(columns={"season": "year"})
    for c in ["obp", "slg", "avg"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["iso"] = m["slg"] - m["avg"]
    m["league"] = "MLB"
    m["woba"] = np.nan
    m["wrc_plus"] = np.nan
    return m[COLS]


def load_all():
    kbo_p = load_kbo_players()
    mlb_p = load_mlb_players()
    kbo_team = pd.read_csv(PROC / "kbo_team_leadoff.csv", encoding="utf-8-sig")
    kbo_games = pd.read_csv(PROC / "kbo_leadoff_games.csv", encoding="utf-8-sig")
    kbo_slot = pd.read_csv(PROC / "kbo_team_slot.csv", encoding="utf-8-sig")
    mlb_slot = pd.read_csv(PROC / "mlb_team_slot.csv", encoding="utf-8-sig")
    mlb_split = pd.read_csv(RAW / "mlb_slot_splits.csv", encoding="utf-8-sig")
    mlb_runs = pd.read_csv(RAW / "mlb_team_runs.csv", encoding="utf-8-sig")
    return kbo_p, mlb_p, kbo_team, kbo_games, kbo_slot, mlb_slot, mlb_split, mlb_runs


# ======================================================================
# H1. 파워+출루 겸비 희소성
# ======================================================================
def add_combo_metrics(allp: pd.DataFrame) -> pd.DataFrame:
    """리그-연도 내부에서 표준화(z-score) + 상위 구간 플래그. '상대적으로' 희소한지를 본다."""
    d = allp.copy()
    g = d.groupby(["league", "year"])
    d["z_obp"] = g["obp"].transform(lambda s: (s - s.mean()) / s.std())
    d["z_iso"] = g["iso"].transform(lambda s: (s - s.mean()) / s.std())
    d["combo_z"] = d[["z_obp", "z_iso"]].min(axis=1)  # 둘 다 좋아야 높은 점수
    for q, label in [(0.67, "p33"), (0.75, "p25"), (0.60, "p40")]:
        thr_obp = g["obp"].transform(lambda s: s.quantile(q))
        thr_iso = g["iso"].transform(lambda s: s.quantile(q))
        d[f"combo_{label}"] = (d["obp"] >= thr_obp) & (d["iso"] >= thr_iso)
    return d


def h1_proportions(allp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in ["p33", "p25", "p40"]:
        kbo = allp[allp["league"] == "KBO"]
        mlb = allp[allp["league"] == "MLB"]
        kbo_x, kbo_n = int(kbo[f"combo_{label}"].sum()), len(kbo)
        mlb_x, mlb_n = int(mlb[f"combo_{label}"].sum()), len(mlb)
        z, p = proportions_ztest([kbo_x, mlb_x], [kbo_n, mlb_n])
        kbo_teams = kbo.groupby(["year", "team"]).ngroups
        mlb_teams = mlb.groupby(["year", "team"]).ngroups
        rows.append({
            "threshold": label,
            "kbo_prop": kbo_x / kbo_n, "kbo_n": kbo_n, "kbo_count": kbo_x,
            "kbo_per_team_season": kbo_x / kbo_teams,
            "mlb_prop": mlb_x / mlb_n, "mlb_n": mlb_n, "mlb_count": mlb_x,
            "mlb_per_team_season": mlb_x / mlb_teams,
            "z": z, "p": p,
        })
    return pd.DataFrame(rows)


def h1_actual_leadoff_check(allp: pd.DataFrame, kbo_games: pd.DataFrame, kbo_team: pd.DataFrame) -> dict:
    """KBO: 실제 선발 리드오프가 콤보-엘리트였는지가 팀 득점 초과분과 관련 있는가."""
    primary = (kbo_games.groupby(["year", "team", "name"]).size()
               .reset_index(name="starts")
               .sort_values("starts", ascending=False)
               .drop_duplicates(["year", "team"]))
    kbo_p = allp[allp["league"] == "KBO"]
    merged = primary.merge(kbo_p[["year", "team", "name", "combo_p33"]],
                            on=["year", "team", "name"], how="left")
    merged["combo_p33"] = np.where(merged["combo_p33"].isna(), False, merged["combo_p33"]).astype(bool)

    d = kbo_team.copy()
    d["lg_avg_rpg"] = d.groupby("year")["runs_pg"].transform("mean")
    d["rpg_vs_lg"] = d["runs_pg"] - d["lg_avg_rpg"]
    merged = merged.merge(d[["year", "team", "rpg_vs_lg"]], on=["year", "team"], how="left")

    yes = merged.loc[merged["combo_p33"], "rpg_vs_lg"].dropna()
    no = merged.loc[~merged["combo_p33"], "rpg_vs_lg"].dropna()
    t, p = stats.ttest_ind(yes, no, equal_var=False) if len(yes) >= 2 and len(no) >= 2 else (np.nan, np.nan)
    return {
        "n_combo_leadoff": int(len(yes)), "n_noncombo_leadoff": int(len(no)),
        "mean_rpg_vs_lg_combo": float(yes.mean()) if len(yes) else None,
        "mean_rpg_vs_lg_noncombo": float(no.mean()) if len(no) else None,
        "t": float(t) if pd.notna(t) else None, "p": float(p) if pd.notna(p) else None,
    }


def h1_roster_depth_vs_realized(allp: pd.DataFrame, kbo_team: pd.DataFrame, mlb_slot: pd.DataFrame) -> dict:
    """팀에 콤보-엘리트 선수가 많을수록 실제 리드오프 생산력도 높아지는가 (양 리그 비교)."""
    kbo_cnt = (allp[allp["league"] == "KBO"].groupby(["year", "team"])["combo_p33"]
               .sum().reset_index(name="n_combo"))
    kk = kbo_cnt.merge(kbo_team[["year", "team", "leadoff_woba"]], on=["year", "team"], how="inner").dropna()
    r_kbo, p_kbo = stats.pearsonr(kk["n_combo"], kk["leadoff_woba"]) if len(kk) >= 3 else (np.nan, np.nan)

    mlb_cnt = (allp[allp["league"] == "MLB"].groupby(["year", "team"])["combo_p33"]
               .sum().reset_index(name="n_combo"))
    m1 = mlb_slot[mlb_slot["slot"] == 1][["season", "team", "ops"]].rename(columns={"season": "year"})
    mm = mlb_cnt.merge(m1, on=["year", "team"], how="inner").dropna()
    r_mlb, p_mlb = stats.pearsonr(mm["n_combo"], mm["ops"]) if len(mm) >= 3 else (np.nan, np.nan)

    return {"kbo_r": r_kbo, "kbo_p": p_kbo, "kbo_n": len(kk),
            "mlb_r": r_mlb, "mlb_p": p_mlb, "mlb_n": len(mm)}


def chart_h1(prop_df: pd.DataFrame):
    row = prop_df[prop_df["threshold"] == "p33"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(["KBO", "MLB"], [row["kbo_prop"] * 100, row["mlb_prop"] * 100],
                color=[BLUE, ORANGE])
    axes[0].set_ylabel("콤보-엘리트 비율 (%, 규정타석 선수 중)")
    axes[0].set_title(f"OBP·ISO 동시 상위 33% 선수 비율\n(2-proportion z-test p={row['p']:.4f})")

    axes[1].bar(["KBO", "MLB"], [row["kbo_per_team_season"], row["mlb_per_team_season"]],
                color=[BLUE, ORANGE])
    axes[1].set_ylabel("팀-시즌당 평균 콤보-엘리트 선수 수")
    axes[1].set_title("팀당 '파워+출루 겸비' 선수 보유 수")

    fig.suptitle("H1. 파워+출루 겸비 선수의 희소성: KBO vs MLB (2021-2025)", y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "06_combo_scarcity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- H1 심화 (절대수준)
def h1_absolute_level(allp: pd.DataFrame) -> dict:
    """상대적 순위가 아니라 '절대 수치' 격차를 본다.

    - KBO vs MLB 원자료 OBP·ISO 평균/중앙값 Welch t-test.
    - KBO 콤보-엘리트(리그 내 상위33%) 선수들의 원자료 OBP·ISO가 MLB 전체
      분포에서 몇 백분위인지(percentileofscore).
    - KBO 선수 중 MLB '절대' 상위33% 기준선(OBP·ISO)을 동시에 넘는 비율(역방향도).
    """
    kbo = allp[allp["league"] == "KBO"]
    mlb = allp[allp["league"] == "MLB"]

    out = {"means": {}}
    for col in ["obp", "iso"]:
        t, p = stats.ttest_ind(kbo[col].dropna(), mlb[col].dropna(), equal_var=False)
        out["means"][col] = {
            "kbo_mean": float(kbo[col].mean()), "mlb_mean": float(mlb[col].mean()),
            "kbo_median": float(kbo[col].median()), "mlb_median": float(mlb[col].median()),
            "diff": float(kbo[col].mean() - mlb[col].mean()), "t": float(t), "p": float(p),
        }

    # KBO 콤보-엘리트의 원자료가 MLB 분포에서 몇 백분위인가 (중앙값 기준)
    kbo_combo = kbo[kbo["combo_p33"]]
    out["kbo_combo_percentile_in_mlb"] = {
        col: float(stats.percentileofscore(mlb[col].dropna(), kbo_combo[col].median(), kind="mean"))
        for col in ["obp", "iso"]
    }

    # 절대 기준선: MLB(그리고 KBO) 전체 상위33% 분위값을 동시 통과하는 비율
    mlb_obp_bar, mlb_iso_bar = mlb["obp"].quantile(0.67), mlb["iso"].quantile(0.67)
    kbo_obp_bar, kbo_iso_bar = kbo["obp"].quantile(0.67), kbo["iso"].quantile(0.67)
    out["absolute_bar"] = {
        "mlb_obp_bar": float(mlb_obp_bar), "mlb_iso_bar": float(mlb_iso_bar),
        "kbo_obp_bar": float(kbo_obp_bar), "kbo_iso_bar": float(kbo_iso_bar),
        # KBO 선수가 MLB의 절대 잣대(상위33% 기준선)를 동시 통과하는 비율
        "kbo_pass_mlb_bar": float(((kbo["obp"] >= mlb_obp_bar) & (kbo["iso"] >= mlb_iso_bar)).mean()),
        # 참고 역방향: MLB 선수가 KBO 절대 잣대를 통과하는 비율
        "mlb_pass_kbo_bar": float(((mlb["obp"] >= kbo_obp_bar) & (mlb["iso"] >= kbo_iso_bar)).mean()),
        "kbo_combo_count": int(((kbo["obp"] >= mlb_obp_bar) & (kbo["iso"] >= mlb_iso_bar)).sum()),
        "kbo_n": int(len(kbo)),
    }
    return out


# ---------------------------------------------------------------- H1 심화 (투수 수준 간접)
def h1_pitching_proxy(allp: pd.DataFrame) -> dict:
    """리그 '경쟁 수준'의 간접지표로 규정타석 타자들의 K%·BB%를 비교.

    한계: 이는 타자 접근법과 투수력이 혼재된 지표이며, 순수 투수 실력을 분리하려면
    두 리그를 모두 경험한 동일 선수의 성적 변환(translation) 데이터가 필요하다.
    그런 표본은 대부분 대체선수급 용병이라 대표성이 낮아 여기서는 다루지 않는다.
    따라서 아래 수치는 '투수 수준이 낮다'의 확증이 아니라 정황 근거로만 해석한다.
    """
    d = allp.copy()
    d["k_pct"] = d["so"] / d["pa"]
    d["bb_pct"] = d["bb"] / d["pa"]
    kbo, mlb = d[d["league"] == "KBO"], d[d["league"] == "MLB"]
    res = {}
    for col in ["k_pct", "bb_pct"]:
        t, p = stats.ttest_ind(kbo[col].dropna(), mlb[col].dropna(), equal_var=False)
        res[col] = {"kbo_mean": float(kbo[col].mean()), "mlb_mean": float(mlb[col].mean()),
                    "t": float(t), "p": float(p)}
    return res


def chart_h1_absolute(allp: pd.DataFrame, absres: dict, pitch: dict):
    kbo = allp[allp["league"] == "KBO"]
    mlb = allp[allp["league"] == "MLB"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) OBP 분포
    ax = axes[0, 0]
    ax.hist(mlb["obp"], bins=25, alpha=0.55, color=ORANGE, density=True, label="MLB")
    ax.hist(kbo["obp"], bins=25, alpha=0.55, color=BLUE, density=True, label="KBO")
    ax.axvline(absres["absolute_bar"]["mlb_obp_bar"], color=ORANGE, ls="--", lw=1.5)
    ax.axvline(absres["absolute_bar"]["kbo_obp_bar"], color=BLUE, ls="--", lw=1.5)
    ax.set_title("OBP 분포 (점선=각 리그 상위33% 기준선)")
    ax.set_xlabel("OBP")
    ax.legend()

    # (b) ISO 분포
    ax = axes[0, 1]
    ax.hist(mlb["iso"], bins=25, alpha=0.55, color=ORANGE, density=True, label="MLB")
    ax.hist(kbo["iso"], bins=25, alpha=0.55, color=BLUE, density=True, label="KBO")
    ax.axvline(absres["absolute_bar"]["mlb_iso_bar"], color=ORANGE, ls="--", lw=1.5)
    ax.axvline(absres["absolute_bar"]["kbo_iso_bar"], color=BLUE, ls="--", lw=1.5)
    ax.set_title("ISO 분포 (점선=각 리그 상위33% 기준선)")
    ax.set_xlabel("ISO")
    ax.legend()

    # (c) K% / BB% 리그 비교
    ax = axes[1, 0]
    x = np.arange(2)
    ax.bar(x - 0.2, [pitch["k_pct"]["kbo_mean"] * 100, pitch["bb_pct"]["kbo_mean"] * 100],
           0.4, color=BLUE, label="KBO")
    ax.bar(x + 0.2, [pitch["k_pct"]["mlb_mean"] * 100, pitch["bb_pct"]["mlb_mean"] * 100],
           0.4, color=ORANGE, label="MLB")
    ax.set_xticks(x)
    ax.set_xticklabels(["삼진 K%", "볼넷 BB%"])
    ax.set_ylabel("% (규정타석 타자)")
    ax.set_title("리그 경쟁수준 간접지표 (낮은 K% = 타자 우위 정황)")
    ax.legend()

    # (d) 상대 리그 절대 기준 동시 통과율
    ax = axes[1, 1]
    ab = absres["absolute_bar"]
    ax.bar(["KBO 선수가\nMLB 절대기준 통과", "MLB 선수가\nKBO 절대기준 통과"],
           [ab["kbo_pass_mlb_bar"] * 100, ab["mlb_pass_kbo_bar"] * 100], color=[BLUE, ORANGE])
    ax.set_ylabel("OBP·ISO 동시 통과율 (%)")
    ax.set_title("절대 잣대로 본 '콤보형' 통과율")

    fig.suptitle("H1 심화. 콤보형의 절대 수준 격차 & 투수 수준 정황", y=1.0, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "09_absolute_level.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# H2. 라인업 뎁스 (1번-4번 트레이드오프)
# ======================================================================
def h2_slot1_vs_slot4(kbo_slot: pd.DataFrame, mlb_slot: pd.DataFrame) -> dict:
    k1 = kbo_slot[kbo_slot["slot"] == 1][["year", "team", "woba"]].rename(columns={"woba": "slot1"})
    k4 = kbo_slot[kbo_slot["slot"] == 4][["year", "team", "woba"]].rename(columns={"woba": "slot4"})
    kk = k1.merge(k4, on=["year", "team"]).dropna()
    r_kbo, p_kbo = stats.pearsonr(kk["slot1"], kk["slot4"])

    m1 = mlb_slot[mlb_slot["slot"] == 1][["season", "team", "ops"]].rename(columns={"ops": "slot1"})
    m4 = mlb_slot[mlb_slot["slot"] == 4][["season", "team", "ops"]].rename(columns={"ops": "slot4"})
    mm = m1.merge(m4, on=["season", "team"]).dropna()
    r_mlb, p_mlb = stats.pearsonr(mm["slot1"], mm["slot4"])

    z_diff, p_diff = fisher_r_to_z_diff(r_kbo, len(kk), r_mlb, len(mm))
    return {"kbo": kk, "mlb": mm,
            "result": {"r_kbo": r_kbo, "p_kbo": p_kbo, "n_kbo": len(kk),
                       "r_mlb": r_mlb, "p_mlb": p_mlb, "n_mlb": len(mm),
                       "z_diff": z_diff, "p_diff": p_diff}}


def team_flatness(slot_df: pd.DataFrame, val_col: str, year_col: str, team_col: str) -> pd.DataFrame:
    """팀-시즌 내 9개 타순 생산력의 변동성(연도 내 표준화 후 표준편차) = 라인업 평탄도(뎁스 proxy)."""
    d = slot_df.copy()
    d["z"] = d.groupby(year_col)[val_col].transform(lambda s: (s - s.mean()) / s.std())
    return d.groupby([year_col, team_col])["z"].std().reset_index(name="flatness_std").dropna()


def h2_flatness(kbo_slot: pd.DataFrame, mlb_slot: pd.DataFrame) -> dict:
    kbo_flat = team_flatness(kbo_slot, "woba", "year", "team")
    mlb_flat = team_flatness(mlb_slot.rename(columns={"season": "year"}), "ops", "year", "team")
    t, p = stats.ttest_ind(kbo_flat["flatness_std"], mlb_flat["flatness_std"], equal_var=False)
    return {"kbo_flat": kbo_flat, "mlb_flat": mlb_flat,
            "result": {"kbo_mean": kbo_flat["flatness_std"].mean(), "kbo_n": len(kbo_flat),
                       "mlb_mean": mlb_flat["flatness_std"].mean(), "mlb_n": len(mlb_flat),
                       "t": t, "p": p}}


def chart_h2(tradeoff: dict, flatness: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    kk, mm = tradeoff["kbo"], tradeoff["mlb"]
    axes[0].scatter(kk["slot1"], kk["slot4"], color=BLUE, alpha=0.6, s=35, label="KBO (wOBA)")
    sns.regplot(x=kk["slot1"], y=kk["slot4"], ax=axes[0], scatter=False, color=BLUE, ci=None)
    ax2 = axes[0].twiny().twinx()
    ax2.scatter(mm["slot1"], mm["slot4"], color=ORANGE, alpha=0.5, s=35, marker="^", label="MLB (OPS)")
    sns.regplot(x=mm["slot1"], y=mm["slot4"], ax=ax2, scatter=False, color=ORANGE, ci=None)
    axes[0].set_xlabel("1번타순 생산력 (KBO: wOBA)")
    axes[0].set_ylabel("4번타순 생산력 (KBO: wOBA)", color=BLUE)
    ax2.set_xlabel("1번타순 생산력 (MLB: OPS)")
    ax2.set_ylabel("4번타순 생산력 (MLB: OPS)", color=ORANGE)
    r = tradeoff["result"]
    axes[0].set_title(f"1번 vs 4번 타순 생산력\nKBO r={r['r_kbo']:.3f}(p={r['p_kbo']:.3f}) | "
                       f"MLB r={r['r_mlb']:.3f}(p={r['p_mlb']:.3f})", fontsize=10)

    kf, mf = flatness["kbo_flat"], flatness["mlb_flat"]
    bp = axes[1].boxplot([kf["flatness_std"], mf["flatness_std"]], tick_labels=["KBO", "MLB"],
                          patch_artist=True, medianprops=dict(color="black"))
    for patch, color in zip(bp["boxes"], [BLUE, ORANGE]):
        patch.set_facecolor(color)
    fr = flatness["result"]
    axes[1].set_ylabel("팀 내 9개 타순 생산력의 표준편차 (연도 내 표준화)")
    axes[1].set_title(f"라인업 '평탄도'(뎁스 proxy)\nWelch t-test p={fr['p']:.4f}", fontsize=10)

    fig.suptitle("H2. 라인업 뎁스: 1번-4번 트레이드오프 & 평탄도", y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "07_slot1_vs_slot4_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# H3. 득점 환경(타고투저/투고투저) 조절효과
# ======================================================================
def kbo_league_era() -> pd.DataFrame:
    rows = []
    kbo_team = pd.read_csv(PROC / "kbo_team_leadoff.csv", encoding="utf-8-sig")
    for y in YEARS:
        b = pd.read_csv(RAW / f"kbo_boxscore_batters_{y}.csv", encoding="utf-8-sig")
        hr_rate = b["hr"].sum() / (b["ab"].sum() + b["bb"].sum())  # PA 근사(HBP/SF 미포함)
        runs_pg = kbo_team.loc[kbo_team["year"] == y, "runs_pg"].mean()
        rows.append({"year": y, "hr_rate": hr_rate, "runs_pg": runs_pg})
    return pd.DataFrame(rows)


def mlb_league_era(mlb_split: pd.DataFrame, mlb_runs: pd.DataFrame) -> pd.DataFrame:
    hr = mlb_split.groupby("season").apply(
        lambda g: g["homeRuns"].sum() / g["plateAppearances"].sum(), include_groups=False
    ).rename("hr_rate").reset_index()
    rpg = mlb_runs.groupby("season").apply(
        lambda g: g["runs"].sum() / g["gamesPlayed"].sum(), include_groups=False
    ).rename("runs_pg").reset_index()
    return hr.merge(rpg, on="season")


def h3_interaction(kbo_team: pd.DataFrame, kbo_era: pd.DataFrame,
                    mlb_team: pd.DataFrame, mlb_era: pd.DataFrame) -> dict:
    kd = kbo_team.dropna(subset=["leadoff_woba", "rest_woba", "runs_pg"]).merge(
        kbo_era[["year", "hr_rate"]], on="year")
    kd["leadoff_c"] = kd["leadoff_woba"] - kd["leadoff_woba"].mean()
    kd["hr_c"] = kd["hr_rate"] - kd["hr_rate"].mean()
    kbo_model = smf.ols("runs_pg ~ leadoff_c * hr_c + rest_woba", data=kd).fit()

    md = mlb_team.dropna(subset=["leadoff_ops", "rest_ops", "r_per_pa"]).merge(
        mlb_era[["season", "hr_rate"]], on="season")
    md["leadoff_c"] = md["leadoff_ops"] - md["leadoff_ops"].mean()
    md["hr_c"] = md["hr_rate"] - md["hr_rate"].mean()
    mlb_model = smf.ols("r_per_pa ~ leadoff_c * hr_c + rest_ops", data=md).fit()

    return {
        "kbo_n": len(kd), "kbo_params": kbo_model.params.to_dict(), "kbo_pvalues": kbo_model.pvalues.to_dict(),
        "mlb_n": len(md), "mlb_params": mlb_model.params.to_dict(), "mlb_pvalues": mlb_model.pvalues.to_dict(),
    }


def h3_era_split(mlb_team: pd.DataFrame, split_year: int = 2015) -> dict:
    """참고용: MLB를 2015년 기준 전/후로 나눠 리드오프 계수 크기 비교(단일 컷포인트라 보조 지표)."""
    out = {}
    for label, sub in [("pre", mlb_team[mlb_team["season"] < split_year]),
                        ("post", mlb_team[mlb_team["season"] >= split_year])]:
        d = sub.dropna(subset=["leadoff_ops", "rest_ops", "r_per_pa"]).copy()
        for c in ["leadoff_ops", "rest_ops", "r_per_pa"]:
            d[c] = (d[c] - d[c].mean()) / d[c].std()
        X = sm.add_constant(d[["leadoff_ops", "rest_ops"]])
        model = sm.OLS(d["r_per_pa"], X).fit()
        out[label] = {"n": len(d), "beta_leadoff": model.params["leadoff_ops"],
                       "p_leadoff": model.pvalues["leadoff_ops"]}
    return out


def chart_h3(kbo_era: pd.DataFrame, mlb_era: pd.DataFrame, interaction: dict, era_split: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(mlb_era["season"], mlb_era["hr_rate"] * 100, color=ORANGE, marker="o",
                 markersize=4, label="MLB")
    axes[0].plot(kbo_era["year"], kbo_era["hr_rate"] * 100, color=BLUE, marker="o",
                 markersize=4, label="KBO")
    axes[0].axvline(2015, color=GRAY, linestyle="--", linewidth=1)
    axes[0].set_xlabel("시즌")
    axes[0].set_ylabel("HR / (AB+BB) 근사 (%)")
    axes[0].set_title("리그 홈런 비율 추이")
    axes[0].legend()

    labels = ["MLB\n(2010-2014)", "MLB\n(2015-2025)"]
    betas = [era_split["pre"]["beta_leadoff"], era_split["post"]["beta_leadoff"]]
    pvals = [era_split["pre"]["p_leadoff"], era_split["post"]["p_leadoff"]]
    colors = [ORANGE if p < 0.05 else GRAY for p in pvals]
    axes[1].bar(labels, betas, color=colors)
    for i, (b, p) in enumerate(zip(betas, pvals)):
        axes[1].text(i, b, f"p={p:.3f}", ha="center", va="bottom" if b >= 0 else "top", fontsize=9)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("표준화 리드오프 계수 (β)")
    mi = interaction["mlb_pvalues"].get("leadoff_c:hr_c", float("nan"))
    axes[1].set_title(f"시대 구분별 MLB 리드오프 효과\n(상호작용항 p={mi:.4f})", fontsize=10)

    fig.suptitle("H3. 득점 환경(타고투저) 추이와 리드오프 효과 조절", y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "08_era_hr_trend.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()

    kbo_p, mlb_p, kbo_team, kbo_games, kbo_slot, mlb_slot, mlb_split, mlb_runs = load_all()
    mlb_team = la.build_mlb_team_leadoff(mlb_split, mlb_runs)

    # ---------- H1 ----------
    allp = pd.concat([kbo_p, mlb_p], ignore_index=True)
    allp = allp.dropna(subset=["obp", "iso"])
    allp = add_combo_metrics(allp)
    allp.to_csv(PROC / "combo_elite_players.csv", index=False, encoding="utf-8-sig")

    h1_prop = h1_proportions(allp)
    h1_actual = h1_actual_leadoff_check(allp, kbo_games, kbo_team)
    h1_depth = h1_roster_depth_vs_realized(allp, kbo_team, mlb_slot)
    chart_h1(h1_prop)

    # ---------- H1 심화 (절대수준 + 투수 정황) ----------
    h1_abs = h1_absolute_level(allp)
    h1_pitch = h1_pitching_proxy(allp)
    chart_h1_absolute(allp, h1_abs, h1_pitch)

    # ---------- H2 ----------
    h2_tradeoff = h2_slot1_vs_slot4(kbo_slot, mlb_slot)
    h2_flat = h2_flatness(kbo_slot, mlb_slot)
    chart_h2(h2_tradeoff, h2_flat)

    # ---------- H3 ----------
    kbo_era = kbo_league_era()
    mlb_era = mlb_league_era(mlb_split, mlb_runs)
    h3_inter = h3_interaction(kbo_team, kbo_era, mlb_team, mlb_era)
    h3_split = h3_era_split(mlb_team)
    chart_h3(kbo_era, mlb_era, h3_inter, h3_split)

    print("=" * 70)
    print("[H1] 콤보-엘리트(OBP·ISO 동시 상위) 비율, KBO vs MLB")
    print(h1_prop.round(4).to_string(index=False))
    print("\n[H1 보조] KBO 실제 리드오프가 콤보-엘리트였는지 vs 득점 초과분:", h1_actual)
    print("[H1 보조] 팀 내 콤보-엘리트 수 vs 실제 리드오프 생산력 상관:", h1_depth)

    print("\n[H1 심화] 절대 수준 격차 (OBP/ISO 원자료):")
    for col, v in h1_abs["means"].items():
        print(f"  {col}: KBO {v['kbo_mean']:.3f} vs MLB {v['mlb_mean']:.3f} "
              f"(diff {v['diff']:+.3f}, p={v['p']:.2e})")
    print("  KBO 콤보엘리트의 MLB분포 내 백분위:", {k: round(v, 1) for k, v in h1_abs["kbo_combo_percentile_in_mlb"].items()})
    print("  KBO 선수가 MLB 절대기준 동시통과:", f"{h1_abs['absolute_bar']['kbo_pass_mlb_bar']*100:.1f}% "
          f"({h1_abs['absolute_bar']['kbo_combo_count']}/{h1_abs['absolute_bar']['kbo_n']})")
    print("  (역방향) MLB 선수가 KBO 절대기준 동시통과:", f"{h1_abs['absolute_bar']['mlb_pass_kbo_bar']*100:.1f}%")
    print("[H1 심화] 투수 정황(K%/BB%):",
          f"K% KBO {h1_pitch['k_pct']['kbo_mean']*100:.1f} vs MLB {h1_pitch['k_pct']['mlb_mean']*100:.1f} (p={h1_pitch['k_pct']['p']:.2e})")

    print("\n[H2] 1번-4번 트레이드오프:", h2_tradeoff["result"])
    print("[H2] 라인업 평탄도(뎁스):", h2_flat["result"])

    print("\n[H3] 리드오프×HR비율 상호작용 회귀:")
    print(" KBO:", {k: round(v, 4) for k, v in h3_inter["kbo_pvalues"].items()})
    print(" MLB:", {k: round(v, 4) for k, v in h3_inter["mlb_pvalues"].items()})
    print("[H3 참고] MLB 시대 구분 리드오프 베타:", h3_split)

    summary = {
        "h1_proportions": h1_prop.to_dict("records"),
        "h1_actual_leadoff_check": h1_actual,
        "h1_roster_depth_vs_realized": h1_depth,
        "h1_absolute_level": h1_abs,
        "h1_pitching_proxy": h1_pitch,
        "h2_tradeoff": h2_tradeoff["result"],
        "h2_flatness": h2_flat["result"],
        "h3_interaction": h3_inter,
        "h3_era_split": h3_split,
    }
    with open(OUT / "hypothesis_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n결과 저장 완료: {OUT / 'hypothesis_results.json'}")


if __name__ == "__main__":
    main()
