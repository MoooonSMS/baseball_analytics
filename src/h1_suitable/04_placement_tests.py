"""Phase 2: 상위 3명 배치 분석(2026-10-04 설계 조정, H1_ANALYSIS_INSTRUCTION.md §10).

(a) within-team: 팀 내 wRC+ 상위 3명의 유형 점수 z(OBP)-z(ISO)와 1~2번 타석 점유의 관계
    1단계 팀-시즌별 Spearman 상관의 리그 평균, 2단계 wRC+ 순위를 통제한 팀-시즌 고정효과 모형
(b) K(상위 3명 중 출루형 수) 층화: §3-4 1차 지표 (a)(b)(c)의 리그 간 차이(CI + TOST)
용량-반응: K·평균 유형 점수 x 리그(리그-시즌·팀 고정효과, 팀 클러스터 SE). N은 KBO 변동이 없어 제외
배치 함수(§5 Phase 2-4): 평균 슬롯 ~ wRC+ 순위 + z(OBP) + z(ISO), 리그별, 팀-시즌 고정효과
견고성: ex-ante·보조 정의·민감도 변형 전부에서 핵심량 재계산

표본 단위는 팀-시즌. 부트스트랩은 리그 안에서 팀-시즌을 복원 추출(B=C.BOOTSTRAP_B),
회귀 SE는 팀(프랜차이즈) 클러스터. 시드는 zlib.crc32(라벨) 기반.

입력: data/processed/h1_players.csv, h1_s_members.csv(02), h1_slot_player.csv, h1_slot_team.csv(03),
      h1_team_n.csv
출력: outputs/h1_suitable/phase2_*.csv, phase2_results.json, phase2_*.png

실행: python src/h1_suitable/04_placement_tests.py
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

LEAGUES = ["KBO", "MLB"]
SLOTS = [f"s{i}" for i in range(1, 10)]
MAIN_VARIANTS = ["primary", "obp_pool_top33", "exante"]   # primary / 보조 정의 / ex-ante
VARIANT_LABEL = {"primary": "기본 정의", "obp_pool_top33": "보조 정의(OBP 상위 33%)", "exante": "ex-ante(전년도)"}
RES: dict = {}


def section(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def rng(label: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(label.encode("utf-8")))


# ======================================================================
# 데이터
# ======================================================================
def rates(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    den = d["ab"] + d["bb"] + d["hbp"] + d["sf"]
    d["obp"] = (d["h"] + d["bb"] + d["hbp"]) / den.where(den > 0)
    d["iso"] = (d["d2"] + 2 * d["d3"] + 3 * d["hr"]) / d["ab"].where(d["ab"] > 0)
    return d


def type_reference(p: pd.DataFrame) -> pd.DataFrame:
    """리그-시즌 후보풀(PA>=POOL_MIN_PA, 선수-팀 행) OBP·ISO 평균·표준편차."""
    q = p[p["pa"] >= C.POOL_MIN_PA]
    return q.groupby(["league", "year"]).agg(obp_m=("obp", "mean"), obp_s=("obp", "std"),
                                            iso_m=("iso", "mean"), iso_s=("iso", "std"))


def add_type(d: pd.DataFrame, ref: pd.DataFrame, ref_year: str = "year") -> pd.DataFrame:
    r = ref.reindex(pd.MultiIndex.from_frame(d[["league", ref_year]].set_axis(["league", "year"], axis=1)))
    d = d.copy()
    d["z_obp"] = (d["obp"].values - r["obp_m"].values) / r["obp_s"].values
    d["z_iso"] = (d["iso"].values - r["iso_m"].values) / r["iso_s"].values
    d["type"] = d["z_obp"] - d["z_iso"]
    return d


def load():
    p = pd.read_csv(C.PROC / "h1_players.csv", encoding="utf-8-sig").assign(pid=lambda d: d["pid"].astype("int64"))
    mem = pd.read_csv(C.PROC / "h1_s_members.csv", encoding="utf-8-sig").assign(pid=lambda d: d["pid"].astype("int64"))
    sp = pd.read_csv(C.PROC / "h1_slot_player.csv", encoding="utf-8-sig")
    st = pd.read_csv(C.PROC / "h1_slot_team.csv", encoding="utf-8-sig")
    tn = pd.read_csv(C.PROC / "h1_team_n.csv", encoding="utf-8-sig")
    return p, mem, sp, st, tn


def top3_frame(variant: str, p, mem, sp, ref) -> pd.DataFrame:
    """변형별 팀 내 상위 3명 + 유형 점수 + 슬롯 점유. ex-ante는 전년도 기록으로 유형 점수를 매긴다."""
    m = mem[(mem["variant"] == variant) & (mem["rank"] <= C.TOP_K)][["league", "year", "team", "pid", "name",
                                                                     "rank", "S"]]
    key = ["league", "year", "team", "pid"]
    if variant == "exante":
        cnt = ["pa", "ab", "h", "d2", "d3", "hr", "bb", "hbp", "sf"]
        prev = rates(p.groupby(["league", "year", "pid"])[cnt].sum().reset_index())
        prev["year"] += 1
        m = m.merge(prev[["league", "year", "pid", "obp", "iso"]], on=["league", "year", "pid"], how="left")
        m["ref_year"] = m["year"] - 1
        m = add_type(m, ref, "ref_year").drop(columns="ref_year")
    else:
        m = m.merge(p[key + ["obp", "iso"]], on=key, how="left")
        m = add_type(m, ref)
    m = m.merge(sp[key + ["slot_pa"] + SLOTS], on=key, how="left")
    m[["slot_pa"] + SLOTS] = m[["slot_pa"] + SLOTS].fillna(0)
    m["top_pa"] = m["s1"] + m["s2"]
    m["top_share"] = m["top_pa"] / m["slot_pa"].where(m["slot_pa"] > 0)
    m["avg_slot"] = (m[SLOTS] * np.arange(1, 10)).sum(axis=1) / m["slot_pa"].where(m["slot_pa"] > 0)
    m["ts"] = m["league"] + "_" + m["year"].astype(str) + "_" + m["team"]
    m["club"] = m["league"] + "_" + m["team"]
    return m


def team_frame(m: pd.DataFrame, st: pd.DataFrame, tn: pd.DataFrame, variant: str) -> pd.DataFrame:
    """팀-시즌 단위: K, 평균 유형 점수, §3-4 1차 지표 (a)(b)(c), 상위 3명의 1~2번 자리 점유."""
    t = st.set_index(["league", "year", "team"])
    rows = []
    for (lg, y, tm), g in m.groupby(["league", "year", "team"]):
        tt = t.loc[(lg, y, tm)]
        s = g[g["S"]]
        rows.append({
            "league": lg, "year": y, "team": tm, "n_top": len(g),
            "K": int((g["type"] > C.TYPE_K_THRESHOLD).sum()), "type_mean": g["type"].mean(),
            "N_s": int(g["S"].sum()),
            "a_S_top_share": s["top_pa"].sum() / s["slot_pa"].sum() if s["slot_pa"].sum() > 0 else np.nan,
            "b_S_share_of_top": s["top_pa"].sum() / (tt["s1"] + tt["s2"]),
            "c_S_share_of_2": s["s2"].sum() / tt["s2"],
            "top3_share_of_top": g["top_pa"].sum() / (tt["s1"] + tt["s2"]),
            "top3_top_share": g["top_pa"].sum() / g["slot_pa"].sum() if g["slot_pa"].sum() > 0 else np.nan,
            "S_slot4_share": s["s4"].sum() / s["slot_pa"].sum() if s["slot_pa"].sum() > 0 else np.nan,
            "S_mid_share": s[["s3", "s4", "s5"]].sum().sum() / s["slot_pa"].sum() if s["slot_pa"].sum() > 0 else np.nan,
            "S_bottom_share": s[["s6", "s7", "s8", "s9"]].sum().sum() / s["slot_pa"].sum() if s["slot_pa"].sum() > 0 else np.nan,
        })
    d = pd.DataFrame(rows)
    n = tn[tn["variant"] == ("primary" if variant not in set(tn["variant"]) else variant)]
    d = d.merge(n[["league", "year", "team", "N"]], on=["league", "year", "team"], how="left")
    d["ts"] = d["league"] + "_" + d["year"].astype(str) + "_" + d["team"]
    d["club"] = d["league"] + "_" + d["team"]
    return d


# ======================================================================
# 통계 도구
# ======================================================================
def boot_mean_diff(x_kbo: np.ndarray, x_mlb: np.ndarray, label: str, B=C.BOOTSTRAP_B) -> dict:
    """리그별 팀-시즌 값의 평균과 KBO-MLB 차이: 팀-시즌 복원 추출 부트스트랩 + TOST(±EQUIV_MARGIN)."""
    a, b = x_kbo[~np.isnan(x_kbo)], x_mlb[~np.isnan(x_mlb)]
    out = {"n_kbo": len(a), "n_mlb": len(b)}
    if len(a) < 2 or len(b) < 2:
        return out | {"kbo": np.nan, "mlb": np.nan, "diff": np.nan}
    g = rng(label)
    ba = a[g.integers(0, len(a), (B, len(a)))].mean(axis=1)
    bb = b[g.integers(0, len(b), (B, len(b)))].mean(axis=1)
    bd = ba - bb
    d = a.mean() - b.mean()
    se = bd.std(ddof=1)
    m = C.EQUIV_MARGIN
    p_tost = max(1 - stats.norm.cdf((d + m) / se), stats.norm.cdf((d - m) / se)) if se > 0 else np.nan
    lo90, hi90 = np.percentile(bd, [5, 95])
    return out | {
        "kbo": a.mean(), "kbo_lo": np.percentile(ba, 2.5), "kbo_hi": np.percentile(ba, 97.5),
        "mlb": b.mean(), "mlb_lo": np.percentile(bb, 2.5), "mlb_hi": np.percentile(bb, 97.5),
        "diff": d, "diff_lo": np.percentile(bd, 2.5), "diff_hi": np.percentile(bd, 97.5),
        "p_diff": float(min(1.0, 2 * min((bd <= 0).mean(), (bd >= 0).mean()))),
        "p_tost": p_tost, "equivalent": bool(lo90 > -m and hi90 < m),
    }


def within_corr(m: pd.DataFrame) -> pd.DataFrame:
    """팀-시즌별 상위 3명의 Spearman(유형 점수, 1~2번 점유). 1~2번 점유가 모두 같으면 NaN."""
    def f(g):
        g = g.dropna(subset=["type", "top_share"])
        if len(g) < 3 or g["top_share"].nunique() < 2 or g["type"].nunique() < 2:
            return np.nan
        return stats.spearmanr(g["type"], g["top_share"])[0]
    r = m.groupby(["league", "ts"]).apply(f, include_groups=False).rename("rho").reset_index()
    return r


def demean(d: pd.DataFrame, cols: list[str], by: str) -> pd.DataFrame:
    return d[cols] - d.groupby(by)[cols].transform("mean")


def fe_ols(d: pd.DataFrame, y: str, xs: list[str], fe: str, cluster: str) -> dict:
    """팀-시즌 고정효과(within 변환) OLS, 클러스터 강건 SE(CR1, 자유도 보정에 FE 수 포함)."""
    d = d.dropna(subset=[y] + xs).copy()
    Y = demean(d, [y], fe)[y].to_numpy()
    X = demean(d, xs, fe).to_numpy()
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    e = Y - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((len(xs), len(xs)))
    for _, idx in d.groupby(cluster).indices.items():
        s = X[idx].T @ e[idx]
        meat += np.outer(s, s)
    n, k, G = len(Y), len(xs) + d[fe].nunique(), d[cluster].nunique()
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    return {"beta": dict(zip(xs, beta)), "V": V, "xs": xs, "n": n, "G": G, "n_fe": d[fe].nunique(),
            "X": X, "Y": Y, "groups_ts": d[fe].to_numpy(), "league": d["league"].to_numpy()}


def lincomb(fit: dict, w: dict) -> tuple[float, float, float, float]:
    """선형결합 추정치, SE, 95% CI(t, df=G-1)."""
    c = np.array([w.get(x, 0.0) for x in fit["xs"]])
    est = float(c @ np.array([fit["beta"][x] for x in fit["xs"]]))
    se = float(np.sqrt(c @ fit["V"] @ c))
    tq = stats.t.ppf(0.975, fit["G"] - 1)
    return est, se, est - tq * se, est + tq * se


def fe_boot(fit: dict, w: dict, label: str, B=C.BOOTSTRAP_B) -> tuple[float, float]:
    """팀-시즌 블록 부트스트랩(리그 안에서 복원 추출). within 변환은 블록 안에서 끝나므로 재사용."""
    c = np.array([w.get(x, 0.0) for x in fit["xs"]])
    ts = fit["groups_ts"]
    blocks = {lg: [np.where(ts == t)[0] for t in pd.unique(ts[fit["league"] == lg])] for lg in LEAGUES}
    g = rng(label)
    est = np.empty(B)
    for b in range(B):
        idx = np.concatenate([blk[i] for lg in LEAGUES for blk in [blocks[lg]]
                              for i in g.integers(0, len(blk), len(blk))])
        beta, *_ = np.linalg.lstsq(fit["X"][idx], fit["Y"][idx], rcond=None)
        est[b] = c @ beta
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


# ======================================================================
# 분석
# ======================================================================
def step1(m: pd.DataFrame, label: str) -> dict:
    r = within_corr(m)
    out = boot_mean_diff(r.loc[r["league"] == "KBO", "rho"].to_numpy(),
                         r.loc[r["league"] == "MLB", "rho"].to_numpy(), f"step1|{label}")
    out["undefined_kbo"] = int(r.loc[r["league"] == "KBO", "rho"].isna().sum())
    out["undefined_mlb"] = int(r.loc[r["league"] == "MLB", "rho"].isna().sum())
    return out


def step2(m: pd.DataFrame, label: str, boot=True) -> dict:
    d = m.dropna(subset=["type", "top_share"]).copy()
    d["kbo"] = (d["league"] == "KBO").astype(float)
    d["type_kbo"] = d["type"] * d["kbo"]
    for r in [2, 3]:
        d[f"r{r}"] = (d["rank"] == r).astype(float)
        d[f"r{r}_kbo"] = d[f"r{r}"] * d["kbo"]
    xs = ["type", "type_kbo", "r2", "r3", "r2_kbo", "r3_kbo"]
    fit = fe_ols(d, "top_share", xs, "ts", "club")
    out = {"n_players": fit["n"], "n_team_seasons": fit["n_fe"], "n_clusters": fit["G"]}
    for name, w in [("mlb_slope", {"type": 1}), ("kbo_slope", {"type": 1, "type_kbo": 1}),
                    ("diff", {"type_kbo": 1})]:
        est, se, lo, hi = lincomb(fit, w)
        out |= {name: est, f"{name}_se": se, f"{name}_lo": lo, f"{name}_hi": hi}
    if boot:
        out["diff_boot_lo"], out["diff_boot_hi"] = fe_boot(fit, {"type_kbo": 1}, f"step2|{label}")
    # 리그별 클러스터 수가 다르므로 KBO 단독 모형(클러스터 10)도 보고
    for lg in LEAGUES:
        f1 = fe_ols(d[d["league"] == lg], "top_share", ["type", "r2", "r3"], "ts", "club")
        est, se, lo, hi = lincomb(f1, {"type": 1})
        out |= {f"{lg.lower()}_only_slope": est, f"{lg.lower()}_only_lo": lo, f"{lg.lower()}_only_hi": hi,
                f"{lg.lower()}_only_G": f1["G"]}
    return out


def k_bins(t: pd.DataFrame) -> tuple[pd.Series, str]:
    cnt = t.groupby(["league", "K"]).size().unstack(fill_value=0).reindex(columns=range(4), fill_value=0)
    if (cnt < C.MIN_BIN).any().any():
        return t["K"].map(lambda k: "K≤1" if k <= 1 else "K≥2"), "merged"
    return t["K"].map(lambda k: f"K={k}"), "full"


def k_strata(t: pd.DataFrame, label: str) -> tuple[pd.DataFrame, str]:
    t = t.copy()
    t["kbin"], mode = k_bins(t)
    rows = []
    for kb in sorted(t["kbin"].unique()) + ["전체"]:
        g = t if kb == "전체" else t[t["kbin"] == kb]
        for ind in ["a_S_top_share", "b_S_share_of_top", "c_S_share_of_2", "top3_share_of_top"]:
            r = boot_mean_diff(g.loc[g["league"] == "KBO", ind].to_numpy(dtype=float),
                               g.loc[g["league"] == "MLB", ind].to_numpy(dtype=float), f"k|{label}|{kb}|{ind}")
            rows.append({"kbin": kb, "indicator": ind,
                         "small_bin": min(r["n_kbo"], r["n_mlb"]) < C.MIN_BIN} | r)
    return pd.DataFrame(rows), mode


def dose_response(t: pd.DataFrame) -> pd.DataFrame:
    """y ~ x + x:KBO + 리그-시즌 FE + 팀 FE, 팀 클러스터 SE. x = K 또는 평균 유형 점수."""
    rows = []
    d = t.copy()
    d["kbo"] = (d["league"] == "KBO").astype(int)
    d["lgyr"] = d["league"] + d["year"].astype(str)
    for y in ["top3_share_of_top", "top3_top_share", "a_S_top_share"]:
        for x in ["K", "type_mean"]:
            dd = d.dropna(subset=[y, x])
            f = smf.ols(f"{y} ~ {x} + {x}:kbo + lgyr + club", data=dd).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(dd["club"])[0]})
            ci = f.conf_int()
            kb = f.t_test(f"{x} + {x}:kbo = 0")
            rows.append({"y": y, "x": x, "n": int(f.nobs),
                         "mlb_slope": f.params[x], "mlb_lo": ci.loc[x, 0], "mlb_hi": ci.loc[x, 1],
                         "kbo_slope": float(kb.effect[0]), "kbo_lo": float(kb.conf_int()[0, 0]),
                         "kbo_hi": float(kb.conf_int()[0, 1]),
                         "diff": f.params[f"{x}:kbo"], "diff_lo": ci.loc[f"{x}:kbo", 0],
                         "diff_hi": ci.loc[f"{x}:kbo", 1], "diff_p": f.pvalues[f"{x}:kbo"],
                         "kbo_x_sd": dd.loc[dd["kbo"] == 1, x].std(), "mlb_x_sd": dd.loc[dd["kbo"] == 0, x].std()})
    return pd.DataFrame(rows)


def placement_function(m: pd.DataFrame) -> pd.DataFrame:
    """평균 슬롯 ~ 순위 더미 + z(OBP) + z(ISO), 리그별 팀-시즌 FE. 음수 계수 = 앞 타순."""
    rows = []
    d = m.dropna(subset=["avg_slot", "z_obp", "z_iso"]).copy()
    d["kbo"] = (d["league"] == "KBO").astype(float)
    for r in [2, 3]:
        d[f"r{r}"] = (d["rank"] == r).astype(float)
    base = ["r2", "r3", "z_obp", "z_iso"]
    for x in base:
        d[f"{x}_kbo"] = d[x] * d["kbo"]
    fit = fe_ols(d, "avg_slot", base + [f"{x}_kbo" for x in base], "ts", "club")
    for x in base:
        for name, w in [("MLB", {x: 1}), ("KBO", {x: 1, f"{x}_kbo": 1}), ("KBO-MLB", {f"{x}_kbo": 1})]:
            est, se, lo, hi = lincomb(fit, w)
            rows.append({"term": x, "league": name, "coef": est, "se": se, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


# ======================================================================
# 차트
# ======================================================================
def setup_plot():
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def style(ax):
    ax.grid(True, axis="x", color="#e6e6e6", zorder=0)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def chart_step(rob: pd.DataFrame):
    plt = setup_plot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    for ax, (pre, title, xl) in zip(axes, [
            ("s1", "1단계: 팀 내 순위상관 ρ의 평균", "평균 Spearman ρ(유형 점수, 1~2번 점유)"),
            ("s2", "2단계: 팀-시즌 FE 기울기(리그별 모형)", "유형 점수 1 증가당 1~2번 점유 변화")]):
        r = rob[rob["variant"].isin(MAIN_VARIANTS)].set_index("variant").loc[MAIN_VARIANTS]
        ys = np.arange(len(r))[::-1]
        for j, lg in enumerate(LEAGUES):
            if pre == "s1":
                est, lo, hi = r[f"s1_{lg.lower()}"], r[f"s1_{lg.lower()}_lo"], r[f"s1_{lg.lower()}_hi"]
            else:
                est, lo, hi = (r[f"s2_{lg.lower()}_only_slope"], r[f"s2_{lg.lower()}_only_lo"],
                               r[f"s2_{lg.lower()}_only_hi"])
            yy = ys + (0.15 if j == 0 else -0.15)
            ax.errorbar(est, yy, xerr=[est - lo, hi - est], fmt="o", ms=6, color=C.COLOR[lg],
                        ecolor=C.COLOR[lg], elinewidth=2, capsize=0, label=lg, zorder=3)
        ax.axvline(0, color="#888888", lw=1, zorder=1)
        ax.set_yticks(ys, [VARIANT_LABEL[v] for v in r.index])
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_xlabel(xl, fontsize=9)
        style(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper right", fontsize=9, ncol=2)
    fig.suptitle("상위 3명 중 출루형일수록 1~2번에 서는가 (점: 추정치, 선: 95% CI)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(C.OUT / "phase2_within_team.png")
    plt.close(fig)


def chart_k(ks: pd.DataFrame, mode: str):
    plt = setup_plot()
    inds = [("a_S_top_share", "(a) S 선수 타석 중 1~2번 비율"), ("b_S_share_of_top", "(b) 1~2번 타석 중 S 비율"),
            ("c_S_share_of_2", "(c) 2번 타석 중 S 비율")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=150, sharey=True)
    bins = [b for b in ks["kbin"].unique() if b != "전체"] + ["전체"]
    for ax, (ind, title) in zip(axes, inds):
        g = ks[ks["indicator"] == ind].set_index("kbin").loc[bins]
        x = np.arange(len(bins))
        for j, lg in enumerate(LEAGUES):
            l = lg.lower()
            est, lo, hi = g[l], g[f"{l}_lo"], g[f"{l}_hi"]
            xx = x + (-0.12 if j == 0 else 0.12)
            ax.errorbar(xx, est, yerr=[est - lo, hi - est], fmt="o", ms=6, color=C.COLOR[lg],
                        ecolor=C.COLOR[lg], elinewidth=2, capsize=0, label=lg)
        labels = [f"{b}\n(KBO {int(g.loc[b, 'n_kbo'])}, MLB {int(g.loc[b, 'n_mlb'])})" for b in bins]
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", color="#e6e6e6", zorder=0)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"K(상위 3명 중 출루형 수)별 §3-4 1차 지표, 기본 정의 ({'K≤1/K≥2로 합침' if mode == 'merged' else 'K 구간 그대로'}; "
                 "괄호: 팀-시즌 수, 선: 95% CI)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(C.OUT / "phase2_k_strata.png")
    plt.close(fig)


def chart_scatter(m: pd.DataFrame):
    plt = setup_plot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150, sharey=True)
    for ax, lg in zip(axes, LEAGUES):
        d = m[(m["league"] == lg)].dropna(subset=["type", "top_share"])
        ax.scatter(d["type"], d["top_share"], s=14, color=C.COLOR[lg], alpha=0.45, edgecolors="none")
        bins = pd.cut(d["type"], [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf])
        b = d.groupby(bins, observed=True).agg(x=("type", "mean"), y=("top_share", "mean"), n=("type", "size"))
        ax.plot(b["x"], b["y"], color="#333333", lw=2, marker="o", ms=5, label="구간 평균")
        ax.set_title(f"{lg}: 상위 3명 {len(d)}명", fontsize=9.5, loc="left")
        ax.set_xlabel("유형 점수 z(OBP) - z(ISO)  (← 장타형 · 출루형 →)", fontsize=9)
        ax.grid(True, color="#eeeeee", zorder=0)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("1~2번 타석 점유")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("팀 내 wRC+ 상위 3명의 유형과 1~2번 배치, 기본 정의 2021~2025", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(C.OUT / "phase2_type_scatter.png")
    plt.close(fig)


# ======================================================================
def main():
    C.OUT.mkdir(parents=True, exist_ok=True)
    p, mem, sp, st, tn = load()
    ref = type_reference(p)

    rob_rows, frames = [], {}
    variants = ["primary", "obp_pool_top33", "exante"] + [v for v in mem["variant"].unique()
                                                          if v not in MAIN_VARIANTS]
    for v in variants:
        m = top3_frame(v, p, mem, sp, ref)
        t = team_frame(m, st, tn, v)
        frames[v] = (m, t)
        main_v = v in MAIN_VARIANTS
        s1 = step1(m, v)
        s2 = step2(m, v, boot=main_v)
        ka = boot_mean_diff(t.loc[t["league"] == "KBO", "a_S_top_share"].to_numpy(dtype=float),
                            t.loc[t["league"] == "MLB", "a_S_top_share"].to_numpy(dtype=float), f"ind_a|{v}")
        row = {"variant": v, "team_seasons_kbo": int((t["league"] == "KBO").sum()),
               "team_seasons_mlb": int((t["league"] == "MLB").sum()),
               "missing_type": int(m["type"].isna().sum()), "missing_slot": int(m["top_share"].isna().sum())}
        row |= {f"s1_{k}": v_ for k, v_ in s1.items()}
        row |= {f"s2_{k}": v_ for k, v_ in s2.items()}
        row |= {f"ind_a_{k}": v_ for k, v_ in ka.items()}
        rob_rows.append(row)
    rob = pd.DataFrame(rob_rows)

    # ------------------------------------------------------------ 기본 정의 상세
    m, t = frames["primary"]
    section("기본 정의: 1단계 팀 내 순위상관")
    r = rob.set_index("variant")
    for v in MAIN_VARIANTS:
        x = r.loc[v]
        print(f"[{v}] KBO ρ={x.s1_kbo:.3f} [{x.s1_kbo_lo:.3f},{x.s1_kbo_hi:.3f}] (n={x.s1_n_kbo}, 정의불가 {x.s1_undefined_kbo}) | "
              f"MLB ρ={x.s1_mlb:.3f} [{x.s1_mlb_lo:.3f},{x.s1_mlb_hi:.3f}] (n={x.s1_n_mlb}) | "
              f"차이 {x.s1_diff:+.3f} [{x.s1_diff_lo:+.3f},{x.s1_diff_hi:+.3f}]")
    section("2단계: 팀-시즌 FE, wRC+ 순위 통제")
    for v in MAIN_VARIANTS:
        x = r.loc[v]
        print(f"[{v}] MLB 기울기 {x.s2_mlb_slope:+.3f} [{x.s2_mlb_slope_lo:+.3f},{x.s2_mlb_slope_hi:+.3f}] | "
              f"KBO 기울기 {x.s2_kbo_slope:+.3f} [{x.s2_kbo_slope_lo:+.3f},{x.s2_kbo_slope_hi:+.3f}] | "
              f"차이 {x.s2_diff:+.3f} 클러스터CI [{x.s2_diff_lo:+.3f},{x.s2_diff_hi:+.3f}] "
              f"부트CI [{x.s2_diff_boot_lo:+.3f},{x.s2_diff_boot_hi:+.3f}] | KBO 단독 {x.s2_kbo_only_slope:+.3f} "
              f"[{x.s2_kbo_only_lo:+.3f},{x.s2_kbo_only_hi:+.3f}] (G={x.s2_kbo_only_G})")

    section("K 층화 (기본 정의)")
    kc = t.groupby(["league", "K"]).size().unstack(fill_value=0).reindex(columns=range(4), fill_value=0)
    print("K 분포:\n", kc.to_string())
    ks, mode = k_strata(t, "primary")
    print("구간 방식:", mode)
    print(ks[["kbin", "indicator", "n_kbo", "n_mlb", "kbo", "mlb", "diff", "diff_lo", "diff_hi", "p_tost",
              "equivalent", "small_bin"]].round(3).to_string(index=False))
    ks_aux, mode_aux = k_strata(frames["obp_pool_top33"][1], "obp_pool_top33")
    ks_aux = ks_aux.assign(variant="obp_pool_top33")

    section("용량-반응 (K, 평균 유형 점수)")
    dr = pd.concat([dose_response(frames[v][1]).assign(variant=v) for v in MAIN_VARIANTS], ignore_index=True)
    print(dr[dr["variant"] == "primary"].round(3).to_string(index=False))

    section("배치 함수 (평균 슬롯, 음수 = 앞 타순)")
    pf = placement_function(m)
    print(pf.round(3).to_string(index=False))

    section("N=3 팀-시즌(Phase 3 주 표본) 서술")
    n3 = t[t["N"] == 3]
    desc = n3.groupby("league")[["a_S_top_share", "b_S_share_of_top", "c_S_share_of_2", "S_slot4_share",
                                 "S_mid_share", "S_bottom_share"]].mean()
    print(f"N=3 팀-시즌: KBO {int((n3.league == 'KBO').sum())}, MLB {int((n3.league == 'MLB').sum())}")
    print(desc.round(3).to_string())
    n3_a = boot_mean_diff(n3.loc[n3.league == "KBO", "a_S_top_share"].to_numpy(dtype=float),
                          n3.loc[n3.league == "MLB", "a_S_top_share"].to_numpy(dtype=float), "n3|a")

    section("견고성: 변형별 핵심량")
    cols = ["variant", "team_seasons_kbo", "team_seasons_mlb", "s1_kbo", "s1_mlb", "s1_diff", "s1_diff_lo",
            "s1_diff_hi", "s2_kbo_slope", "s2_mlb_slope", "s2_diff", "s2_diff_lo", "s2_diff_hi",
            "ind_a_kbo", "ind_a_mlb", "ind_a_diff", "ind_a_diff_lo", "ind_a_diff_hi", "ind_a_equivalent"]
    print(rob[cols].round(3).to_string(index=False))

    # ------------------------------------------------------------ 저장
    rob.to_csv(C.OUT / "phase2_robustness.csv", index=False, encoding="utf-8-sig")
    pd.concat([ks.assign(variant="primary"), ks_aux]).to_csv(C.OUT / "phase2_k_strata.csv", index=False,
                                                             encoding="utf-8-sig")
    dr.to_csv(C.OUT / "phase2_dose_response.csv", index=False, encoding="utf-8-sig")
    pf.to_csv(C.OUT / "phase2_placement_function.csv", index=False, encoding="utf-8-sig")
    t.to_csv(C.OUT / "phase2_team_seasons_primary.csv", index=False, encoding="utf-8-sig")
    m.drop(columns=["ts", "club"]).to_csv(C.OUT / "phase2_top3_players_primary.csv", index=False,
                                          encoding="utf-8-sig")
    RES["k_distribution_primary"] = {lg: kc.loc[lg].to_dict() for lg in kc.index}
    RES["k_bin_mode"] = {"primary": mode, "obp_pool_top33": mode_aux}
    RES["main"] = rob[rob["variant"].isin(MAIN_VARIANTS)].round(4).to_dict("records")
    RES["n3_descriptive"] = desc.round(4).to_dict("index")
    RES["n3_indicator_a"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in n3_a.items()}
    RES["notes"] = ["표본 단위 팀-시즌(KBO 50, MLB 150). KBO 클러스터 10개라 KBO 단독 SE는 불안정",
                    "슬롯 PA = ab+bb 근사(두 리그 동일)", "다중 비교: 주 결과는 기본 정의의 1·2단계와 지표 (a)(b)(c)"]
    with open(C.OUT / "phase2_results.json", "w", encoding="utf-8") as f:
        json.dump(RES, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    chart_step(rob)
    chart_k(ks, mode)
    chart_scatter(m)
    print("\n저장 완료")


if __name__ == "__main__":
    main()
