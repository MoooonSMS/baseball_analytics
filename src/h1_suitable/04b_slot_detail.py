"""Phase 2.5: 슬롯 분포 보강(2026-10-04 추가, Phase 3 반사실 설계 근거).

(a) 상위 3명(S, 상위 3명 전체)의 타석을 슬롯 1~9 개별로 KBO vs MLB 비교
(b) 상위 3명 안에서 wRC+ 1위·2위·3위 타자별 슬롯 분포
(c) 1번·2번 타석을 소화한 선수의 팀 내 wRC+ 순위 분포(1, 2, 3, 4~6, 7위 이하, 후보풀 밖 PA<300)

값은 팀-시즌 단위 비율(팀-시즌 동일 가중)의 리그 평균. 불확실성은 세 가지를 병기:
  - 팀-시즌 부트스트랩 95% CI(리그 안에서 팀-시즌 복원 추출)
  - 팀 클러스터 부트스트랩 95% CI(프랜차이즈 단위 복원 추출; KBO 10, MLB 30)
  - KBO-MLB 차이의 wild cluster bootstrap p값(팀 클러스터, Webb 6점 가중치, 귀무가설 부과, CR1 t)
B = C.BOOTSTRAP_B, 시드 zlib.crc32(라벨).

입력: 02~04 산출물(h1_players, h1_s_members, h1_slot_player, h1_slot_team, h1_team_n)
출력: outputs/h1_suitable/phase25_*.csv, phase25_results.json, phase25_*.png

실행: python src/h1_suitable/04b_slot_detail.py
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

P2 = importlib.import_module("04_placement_tests")
LEAGUES = P2.LEAGUES
SLOTS = P2.SLOTS
RANK_CATS = ["1위", "2위", "3위", "4~6위", "7위 이하", "풀 밖"]
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


# ======================================================================
# 불확실성
# ======================================================================
def compare(df: pd.DataFrame, cols: list[str], label: str, B=C.BOOTSTRAP_B) -> pd.DataFrame:
    """df: 팀-시즌 행(league, club, cols). 열마다 리그 평균, 두 종류 CI, 차이 CI, wild p."""
    rows = []
    for col in cols:
        d = df[["league", "club", col]].dropna()
        g = P2.rng(f"{label}|{col}")
        out = {"col": col}
        boots_ts, boots_cl = {}, {}
        for lg in LEAGUES:
            x = d.loc[d["league"] == lg, col].to_numpy(float)
            cl = d.loc[d["league"] == lg, "club"].to_numpy()
            out[f"{lg.lower()}"] = x.mean()
            out[f"n_{lg.lower()}"] = len(x)
            boots_ts[lg] = x[g.integers(0, len(x), (B, len(x)))].mean(axis=1)
            codes, uniq = pd.factorize(cl)
            s = np.bincount(codes, weights=x)
            n = np.bincount(codes).astype(float)
            pick = g.integers(0, len(uniq), (B, len(uniq)))
            boots_cl[lg] = s[pick].sum(axis=1) / n[pick].sum(axis=1)
            for kind, bt in [("ts", boots_ts[lg]), ("cl", boots_cl[lg])]:
                out[f"{lg.lower()}_{kind}_lo"], out[f"{lg.lower()}_{kind}_hi"] = np.percentile(bt, [2.5, 97.5])
        out["diff"] = out["kbo"] - out["mlb"]
        for kind, bt in [("ts", boots_ts), ("cl", boots_cl)]:
            bd = bt["KBO"] - bt["MLB"]
            out[f"diff_{kind}_lo"], out[f"diff_{kind}_hi"] = np.percentile(bd, [2.5, 97.5])
        out["wild_p"] = wild_cluster_p(d[col].to_numpy(float), (d["league"] == "KBO").to_numpy(),
                                       d["club"].to_numpy(), f"wild|{label}|{col}", B)
        rows.append(out)
    return pd.DataFrame(rows)


def _diff_t(Y: np.ndarray, kbo: np.ndarray, codes: np.ndarray, G: int) -> tuple[np.ndarray, np.ndarray]:
    """Y: (B, n). y ~ 1 + kbo 의 kbo 계수(=그룹 평균 차이)와 CR1 클러스터 SE."""
    nk, nm = kbo.sum(), (~kbo).sum()
    mk = Y[:, kbo].mean(axis=1)
    mm = Y[:, ~kbo].mean(axis=1)
    e = Y - np.where(kbo, mk[:, None], mm[:, None])
    w = np.where(kbo, 1 / nk, -1 / nm)            # 차이 추정치의 영향 함수 계수
    sc = np.zeros((Y.shape[0], G))
    np.add.at(sc.T, codes, (e * w).T)
    n = Y.shape[1]
    v = (sc ** 2).sum(axis=1) * G / (G - 1) * (n - 1) / (n - 2)
    return mk - mm, np.sqrt(v)


def wild_cluster_p(y: np.ndarray, kbo: np.ndarray, club: np.ndarray, label: str, B: int) -> float:
    codes, uniq = pd.factorize(club)
    G = len(uniq)
    d0, se0 = _diff_t(y[None, :], kbo, codes, G)
    t0 = d0[0] / se0[0]
    e_r = y - y.mean()                            # 귀무가설(차이 0) 부과 잔차
    wts = P2.rng(label).choice(WEBB, size=(B, G))
    Y = y.mean() + wts[:, codes] * e_r[None, :]
    d, se = _diff_t(Y, kbo, codes, G)
    return float((np.abs(d / se) >= abs(t0)).mean())


# ======================================================================
# 데이터
# ======================================================================
def share_rows(m: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """팀-시즌별 (선택된 선수들의 슬롯 PA 합) / (선택된 선수들의 총 PA)."""
    g = m[mask].groupby(["league", "year", "team", "club"])[SLOTS + ["slot_pa"]].sum()
    g = g[g["slot_pa"] > 0]
    out = g[SLOTS].div(g["slot_pa"], axis=0)
    out["s1_2"] = out["s1"] + out["s2"]
    out["s3_5"] = out[["s3", "s4", "s5"]].sum(axis=1)
    out["s6_9"] = out[["s6", "s7", "s8", "s9"]].sum(axis=1)
    return out.reset_index()


def rank_all(p: pd.DataFrame) -> pd.DataFrame:
    """전 선수의 팀 내 wRC+ 순위 범주(후보풀 PA>=POOL_MIN_PA 안에서 순위, 기본 정의와 같은 방식)."""
    p = p[p["year"].isin(C.SEASONS)].copy()
    pool = p["pa"] >= C.POOL_MIN_PA
    p.loc[pool, "rank"] = p[pool].groupby(["league", "year", "team"])["wrc"].rank(ascending=False, method="first")
    p["rank_cat"] = pd.cut(p["rank"], [0, 1, 2, 3, 6, np.inf], labels=RANK_CATS[:5]).astype(object)
    p["rank_cat"] = p["rank_cat"].fillna("풀 밖")
    return p[["league", "year", "team", "pid", "rank_cat"]]


def slot_rank_shares(sp: pd.DataFrame, rk: pd.DataFrame, slot: str) -> pd.DataFrame:
    d = sp.merge(rk, on=["league", "year", "team", "pid"], how="left")
    d["rank_cat"] = d["rank_cat"].fillna("풀 밖")      # 시즌 기록에 없는 선수(극소수)도 풀 밖
    g = d.pivot_table(index=["league", "year", "team"], columns="rank_cat", values=slot, aggfunc="sum",
                      fill_value=0).reindex(columns=RANK_CATS, fill_value=0)
    g = g.div(g.sum(axis=1), axis=0).reset_index()
    g["club"] = g["league"] + "_" + g["team"]
    return g


# ======================================================================
# 차트
# ======================================================================
def dot_ci(ax, res: pd.DataFrame, cols: list[str], labels: list[str], kind="ts"):
    x = np.arange(len(cols))
    r = res.set_index("col").loc[cols]
    for j, lg in enumerate(LEAGUES):
        l = lg.lower()
        est, lo, hi = r[l], r[f"{l}_{kind}_lo"], r[f"{l}_{kind}_hi"]
        ax.errorbar(x + (-0.14 if j == 0 else 0.14), est, yerr=[est - lo, hi - est], fmt="o", ms=5,
                    color=C.COLOR[lg], ecolor=C.COLOR[lg], elinewidth=2, capsize=0, label=lg)
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.grid(True, axis="y", color="#e6e6e6", zorder=0)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def chart_a(res_s, res_all):
    plt = P2.setup_plot()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150, sharey=True)
    for ax, res, t in [(axes[0], res_s, "S 선수(기본 정의)"), (axes[1], res_all, "팀 내 wRC+ 상위 3명 전체")]:
        dot_ci(ax, res, SLOTS, [f"{i}번" for i in range(1, 10)])
        ax.set_title(t, fontsize=9.5, loc="left")
    axes[0].set_ylabel("팀-시즌 평균 타석 비율")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper right", fontsize=9, ncol=2, bbox_to_anchor=(0.985, 1.0))
    fig.suptitle("(a) 상위 타자의 타석이 어느 슬롯에서 나오는가 (선: 팀-시즌 부트스트랩 95% CI)", fontsize=10.5,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(C.OUT / "phase25_a_slot_dist.png")
    plt.close(fig)


def chart_b(res_rank):
    plt = P2.setup_plot()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=150, sharey=True)
    for ax, rk in zip(axes, [1, 2, 3]):
        dot_ci(ax, res_rank[rk], SLOTS, [str(i) for i in range(1, 10)])
        ax.set_title(f"팀 내 wRC+ {rk}위", fontsize=9.5, loc="left")
        ax.set_xlabel("타순", fontsize=9)
    axes[0].set_ylabel("팀-시즌 평균 타석 비율")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper right", fontsize=9, ncol=2, bbox_to_anchor=(0.985, 1.0))
    fig.suptitle("(b) wRC+ 1~3위 타자별 슬롯 분포 (선: 팀-시즌 부트스트랩 95% CI)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(C.OUT / "phase25_b_rank_slots.png")
    plt.close(fig)


def chart_c(res_c):
    plt = P2.setup_plot()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150, sharey=True)
    for ax, slot in zip(axes, ["s1", "s2"]):
        dot_ci(ax, res_c[slot], RANK_CATS, RANK_CATS)
        ax.set_title(f"{slot[1]}번 타석을 소화한 타자의 팀 내 wRC+ 순위", fontsize=9.5, loc="left")
    axes[0].set_ylabel("팀-시즌 평균 타석 비율")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper right", fontsize=9, ncol=2, bbox_to_anchor=(0.985, 1.0))
    fig.suptitle("(c) 1·2번은 팀에서 몇 번째 타자가 맡는가 (순위: PA≥300 후보풀 안, 선: 팀-시즌 부트스트랩 95% CI)",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(C.OUT / "phase25_c_slot_rank.png")
    plt.close(fig)


# ======================================================================
def main():
    p, mem, sp, st, tn = P2.load()
    ref = P2.type_reference(p)
    m = P2.top3_frame("primary", p, mem, sp, ref)
    show = ["col", "n_kbo", "n_mlb", "kbo", "mlb", "diff", "diff_ts_lo", "diff_ts_hi", "diff_cl_lo", "diff_cl_hi",
            "wild_p"]
    grp = ["s1_2", "s3_5", "s6_9"]

    # (a)
    res_s = compare(share_rows(m, m["S"]), SLOTS + grp, "a|S")
    res_all = compare(share_rows(m, m["rank"] <= 3), SLOTS + grp, "a|top3")
    P2.section("(a) 슬롯별 분포: S 선수")
    print(res_s[show].round(3).to_string(index=False))
    P2.section("(a) 슬롯별 분포: 상위 3명 전체")
    print(res_all[show].round(3).to_string(index=False))

    # (b)
    res_rank = {}
    for rk in [1, 2, 3]:
        res_rank[rk] = compare(share_rows(m, m["rank"] == rk), SLOTS + grp, f"b|{rk}")
        P2.section(f"(b) wRC+ {rk}위")
        print(res_rank[rk][show].round(3).to_string(index=False))

    # (c)
    rk = rank_all(p)
    sp_ = sp.assign(pid=sp["pid"].astype("int64"))
    res_c = {}
    for slot in ["s1", "s2"]:
        res_c[slot] = compare(slot_rank_shares(sp_, rk, slot), RANK_CATS, f"c|{slot}")
        P2.section(f"(c) {slot[1]}번 타석의 팀 내 wRC+ 순위")
        print(res_c[slot][show].round(3).to_string(index=False))

    # 저장
    out = pd.concat([res_s.assign(part="a", group="S"), res_all.assign(part="a", group="top3")]
                    + [res_rank[k].assign(part="b", group=f"rank{k}") for k in [1, 2, 3]]
                    + [res_c[s].assign(part="c", group=s) for s in ["s1", "s2"]], ignore_index=True)
    out.to_csv(C.OUT / "phase25_slot_detail.csv", index=False, encoding="utf-8-sig")
    res = {"notes": ["팀-시즌 동일 가중 평균", "ts=팀-시즌 부트스트랩, cl=팀(프랜차이즈) 클러스터 부트스트랩, "
                     "wild_p=wild cluster bootstrap(Webb, 귀무 부과) p", "KBO 클러스터 10개"],
           "rows": out.round(4).to_dict("records")}
    with open(C.OUT / "phase25_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    chart_a(res_s, res_all)
    chart_b(res_rank)
    chart_c(res_c)
    print("\n저장 완료")


if __name__ == "__main__":
    main()
