"""'기아는 왜 1도영(리드오프 김도영)을 안 하는가' — 라인업 최적화 사례 분석.

두 축:
  (B) '관행' 증거: KBO는 팀 최고타자를 3~4번에 두는 경향이 MLB보다 강한가
      (팀-시즌별 최고 생산 슬롯 분포, MLB의 1~2번 이동 추세).
  (C) 김도영 사례: KIA 2024 라인업에서 김도영을 1~9번 어디에 두는 것이 팀 득점에
      유리한가를 몬테카를로 시뮬(lineup_sim)로 검증. 실제 배치(주로 3번)의 손해도 정량화.

출력: outputs/leadoff_analysis/10_best_hitter_slot.png, 11_kim_doyoung_slot_value.png,
      kim_doyoung_sim.json
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import lineup_sim as ls

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "leadoff_analysis"
YEARS = [2021, 2022, 2023, 2024, 2025]

BLUE = "#2a78d6"    # KBO
ORANGE = "#eb6834"  # MLB
GRAY = "#898781"
GREEN = "#0b7d2b"

ALLSTAR = {"나눔", "드림"}


def setup_style():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sns.set_theme(style="whitegrid", font="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False


# ======================================================================
# (B) '관행' 증거: 팀 최고타자는 몇 번에 서는가
# ======================================================================
def best_hitter_slot() -> dict:
    kbo = pd.read_csv(PROC / "kbo_team_slot.csv", encoding="utf-8-sig")  # year,team,slot,woba
    mlb = pd.read_csv(PROC / "mlb_team_slot.csv", encoding="utf-8-sig")  # season,slot,team,ops

    # 팀-시즌별 최고 생산 슬롯
    kbo_best = kbo.dropna(subset=["woba"]).loc[
        kbo.dropna(subset=["woba"]).groupby(["year", "team"])["woba"].idxmax()]
    mlb_best = mlb.dropna(subset=["ops"]).loc[
        mlb.dropna(subset=["ops"]).groupby(["season", "team"])["ops"].idxmax()]

    kbo_dist = kbo_best["slot"].value_counts(normalize=True).sort_index()
    mlb_dist = mlb_best["slot"].value_counts(normalize=True).sort_index()

    # MLB: '최고 슬롯 ≤ 2' 비율의 연도 추세
    mlb_best_g = mlb_best.copy()
    mlb_best_g["top2"] = mlb_best_g["slot"] <= 2
    mlb_trend = mlb_best_g.groupby("season")["top2"].mean()

    return {
        "kbo_dist": kbo_dist, "mlb_dist": mlb_dist,
        "kbo_share_slot34": float(kbo_best["slot"].isin([3, 4]).mean()),
        "mlb_share_slot34": float(mlb_best["slot"].isin([3, 4]).mean()),
        "kbo_share_slot12": float(kbo_best["slot"].isin([1, 2]).mean()),
        "mlb_share_slot12": float(mlb_best["slot"].isin([1, 2]).mean()),
        "mlb_trend": mlb_trend,
        "kbo_mean_best_slot": float(kbo_best["slot"].mean()),
        "mlb_mean_best_slot": float(mlb_best["slot"].mean()),
    }


def chart_best_hitter_slot(bh: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    slots = np.arange(1, 10)
    kd = [bh["kbo_dist"].get(s, 0) * 100 for s in slots]
    md = [bh["mlb_dist"].get(s, 0) * 100 for s in slots]
    ax.bar(slots - 0.2, kd, 0.4, color=BLUE, label="KBO (2021-2025)")
    ax.bar(slots + 0.2, md, 0.4, color=ORANGE, label="MLB (2010-2025)")
    ax.set_xlabel("타순")
    ax.set_ylabel("팀 최고타자가 이 슬롯인 비율 (%)")
    ax.set_xticks(slots)
    ax.set_title("팀 최고 생산 타자는 몇 번에 서는가")
    ax.legend()

    ax = axes[1]
    tr = bh["mlb_trend"]
    ax.plot(tr.index, tr.values * 100, color=ORANGE, marker="o", markersize=4)
    ax.axhline(bh["kbo_share_slot12"] * 100, color=BLUE, ls="--", lw=1.5,
               label=f"KBO 평균 {bh['kbo_share_slot12']*100:.0f}%")
    ax.set_xlabel("시즌")
    ax.set_ylabel("최고타자가 1~2번인 팀 비율 (%)")
    ax.set_title("MLB: 최고타자를 상위타순에 두는 흐름")
    ax.legend()

    fig.suptitle("관행 증거: KBO는 최고타자를 3~4번에, MLB는 점점 1~2번으로", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "10_best_hitter_slot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# (C) 김도영 사례: KIA 2024 라인업 구축
# ======================================================================
KIM = "김도영"


def load_kia_lineup(year: int = 2024):
    """KIA 그 시즌 주전 9인(총 타석 상위)과 각자의 타석당 이벤트벡터, 실제 평균 타순."""
    b = pd.read_csv(RAW / f"kbo_boxscore_batters_{year}.csv", encoding="utf-8-sig")
    b = b[(b["team"] == "KIA") & (~b["opponent"].isin(ALLSTAR))].copy()
    p = pd.read_csv(PROC / "kbo_players_woba.csv", encoding="utf-8-sig")
    pq = p[(p["team"] == "KIA") & (p["year"] == year)].set_index("name")

    # 선수별 boxscore 집계 + 평균/최빈 타순
    modal = (b.groupby(["name", "batOrder"]).size().reset_index(name="n")
             .sort_values("n", ascending=False).drop_duplicates("name")
             .set_index("name")["batOrder"])
    agg = b.groupby("name").agg(
        ab=("ab", "sum"), hit=("hit", "sum"), hr=("hr", "sum"), bb=("bb", "sum"),
        rbi=("rbi", "sum"), run=("run", "sum"),
        pa_games=("gameId", "nunique"), mean_order=("batOrder", "mean")).reset_index()
    agg["modal_order"] = agg["name"].map(modal)
    # 타석수 추정: (AB + BB) 보정 — HBP/SF/SH(≈3%) 반영해 PA 근사(과소추정 방지)
    agg["pa_est"] = (agg["ab"] + agg["bb"]) / 0.97
    regulars = agg.sort_values("pa_est", ascending=False).head(9).copy()

    lineup = []
    for _, r in regulars.iterrows():
        name = r["name"]
        if name in pq.index:  # 규정타석: 풀슬래시 사용
            q = pq.loc[name]
            ev = ls.events_from_full(pa=q["pa"], ab=q["ab"], h=q["hits"],
                                     d2=q["doubles"], d3=q["triples"], hr=q["hr"],
                                     bb=q["bb"], hbp=q["hbp"], sf=q["sf"])
            src = "full"
        else:  # 미규정: boxscore + 리그분할 보정
            ev = ls.events_from_boxscore(pa_est=r["pa_est"], ab=r["ab"], h=r["hit"],
                                         hr=r["hr"], bb=r["bb"])
            src = "boxscore"
        lineup.append({"name": name, "events": ev, "mean_order": r["mean_order"],
                       "modal_order": int(r["modal_order"]), "pa_est": int(r["pa_est"]),
                       "src": src})

    # 실제 상대 순서 = 평균 타순으로 정렬
    lineup = sorted(lineup, key=lambda d: d["mean_order"])
    return lineup


def actual_order_events(lineup):
    """실제(평균타순) 순서의 events 배열(9,6)과 김도영의 실제 인덱스."""
    ev = np.array([d["events"] for d in lineup])
    kim_idx = next(i for i, d in enumerate(lineup) if d["name"] == KIM)
    return ev, kim_idx


# ---- 반사실 ① 김도영만 이동 ----
def counterfactual_move_kim(lineup, n_games=40000, seed=0):
    """나머지 8인 실제 순서 고정, 김도영을 슬롯 1~9에 삽입하며 팀 R/G와 김도영 개인 지표."""
    others = [d for d in lineup if d["name"] != KIM]
    others_ev = [d["events"] for d in others]
    kim_ev = next(d["events"] for d in lineup if d["name"] == KIM)

    rows = []
    for slot in range(9):  # 0-index
        order = others_ev[:slot] + [kim_ev] + others_ev[slot:]
        res = ls.simulate_lineup(np.array(order), n_games=n_games, seed=seed)
        rows.append({
            "slot": slot + 1,
            "team_rpg": res["runs_per_game"],
            "kim_rbi_pg": res["rbi_per_game"][slot],
            "kim_runs_pg": res["runs_by_slot"][slot],
        })
    return pd.DataFrame(rows)


# ---- 반사실 ② 근사 최적 순서 (pairwise-swap hill climbing, 공통난수) ----
def optimize_order(lineup, n_search=8000, n_final=40000, seed=0):
    ev = [d["events"] for d in lineup]
    names = [d["name"] for d in lineup]
    order = list(range(9))  # 실제(평균타순) 순서에서 출발

    def rpg(idx_order, n, s=seed):
        arr = np.array([ev[i] for i in idx_order])
        return ls.simulate_lineup(arr, n_games=n, seed=s)["runs_per_game"]

    best_val = rpg(order, n_search)
    improved = True
    while improved:
        improved = False
        for i in range(9):
            for j in range(i + 1, 9):
                cand = order.copy()
                cand[i], cand[j] = cand[j], cand[i]
                v = rpg(cand, n_search)
                if v > best_val + 1e-4:
                    order, best_val, improved = cand, v, True
    # 최종 고정밀 평가(공통 시드)
    actual = list(range(9))
    actual_rpg = rpg(actual, n_final)
    opt_rpg = rpg(order, n_final)
    opt_names = [names[i] for i in order]
    kim_opt_slot = opt_names.index(KIM) + 1
    return {
        "actual_order": names, "actual_rpg": actual_rpg,
        "optimal_order": opt_names, "optimal_rpg": opt_rpg,
        "gain_rpg": opt_rpg - actual_rpg, "kim_optimal_slot": kim_opt_slot,
    }


# ---- 경험적 슬롯 맥락 (볼륨 vs 타점맥락) ----
def empirical_slot_context(year_range=YEARS):
    dfs = [pd.read_csv(RAW / f"kbo_boxscore_batters_{y}.csv", encoding="utf-8-sig") for y in year_range]
    b = pd.concat(dfs, ignore_index=True)
    b = b[~b["team"].isin(ALLSTAR) & ~b["opponent"].isin(ALLSTAR)].copy()
    b["pa_row"] = b["ab"] + b["bb"]  # 타석 근사(HBP/SF 미포함)
    # 팀-게임당 슬롯별 타석/타점/득점 (선발+교체 합산)
    per = b.groupby(["gameId", "team", "batOrder"]).agg(
        pa=("pa_row", "sum"), rbi=("rbi", "sum"), run=("run", "sum")).reset_index()
    slot = per.groupby("batOrder").agg(
        pa_pg=("pa", "mean"), rbi_pg=("rbi", "mean"), run_pg=("run", "mean")).reset_index()
    return slot[slot["batOrder"].between(1, 9)]


# ======================================================================
def chart_kim(move_df: pd.DataFrame, opt: dict, slot_ctx: pd.DataFrame, kim_actual_order: int,
              se: float = 0.0):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (좌) 김도영 슬롯별 팀 R/G
    ax = axes[0]
    top = move_df["team_rpg"].max()
    # ±2·SE '통계적 동률' 밴드: 이 안의 슬롯들은 최적과 사실상 구별되지 않음
    if se > 0:
        ax.axhspan(top - 2 * se, top, color=GREEN, alpha=0.10,
                   label=f"최적과 통계적 동률 (±2·SE={2*se:.3f})")
    ax.plot(move_df["slot"], move_df["team_rpg"], color=BLUE, marker="o", lw=2)
    best_slot = move_df.loc[move_df["team_rpg"].idxmax(), "slot"]
    ax.scatter([best_slot], [top], color=GREEN, s=120, zorder=5, label=f"최댓값 {int(best_slot)}번")
    ax.axvline(kim_actual_order, color=GRAY, ls="--", lw=1.5,
               label=f"실제(최빈) {int(kim_actual_order)}번")
    ax.set_xlabel("김도영을 두는 타순")
    ax.set_ylabel("팀 득점/경기 (시뮬)")
    ax.set_xticks(range(1, 10))
    ax.set_title("김도영 타순별 팀 득점 (나머지 8인 고정)")
    ax.legend(fontsize=9)

    # (우) 김도영 개인 타점기대값 vs 득점 + 슬롯 타석볼륨
    ax = axes[1]
    ax.plot(move_df["slot"], move_df["kim_rbi_pg"], color=ORANGE, marker="s", lw=2, label="김도영 타점/경기")
    ax.plot(move_df["slot"], move_df["kim_runs_pg"], color=BLUE, marker="o", lw=2, label="김도영 득점/경기")
    ax.set_xlabel("김도영을 두는 타순")
    ax.set_ylabel("경기당 기대값 (시뮬)")
    ax.set_xticks(range(1, 10))
    ax.set_title("타순별 김도영 개인 타점·득점 기대값")
    ax.legend(loc="upper right")

    rpg_modal = float(move_df.loc[move_df["slot"] == kim_actual_order, "team_rpg"].iloc[0])
    gain144 = (top - rpg_modal) * 144
    fig.suptitle(f"김도영 라인업 시뮬레이션 (KIA 2024) — 위로 올릴수록 타점↓·득점↑이 상쇄돼 팀 득점은 거의 평평 "
                 f"(실제 {int(kim_actual_order)}번→최댓값 {int(best_slot)}번 이득 {gain144:+.1f}점/144경기)",
                 y=1.02, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "11_kim_doyoung_slot_value.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()

    # ---- (B) 관행 증거 ----
    bh = best_hitter_slot()
    chart_best_hitter_slot(bh)

    # ---- (C) 김도영 ----
    lineup = load_kia_lineup(2024)
    kim = next(d for d in lineup if d["name"] == KIM)
    _, kim_idx = actual_order_events(lineup)

    N_MOVE = 120000  # 슬롯 간 미세차 해석을 위해 표본↑(SE 절반)
    move_df = counterfactual_move_kim(lineup, n_games=N_MOVE, seed=7)
    opt = optimize_order(lineup, n_search=10000, n_final=N_MOVE, seed=7)
    slot_ctx = empirical_slot_context()

    # 캘리브레이션: 실제 순서 팀 R/G vs 실제 KIA 2024 R/G
    kia_actual_rpg_real = pd.read_csv(PROC / "kbo_team_leadoff.csv", encoding="utf-8-sig")
    kia_real = kia_actual_rpg_real[(kia_actual_rpg_real["team"] == "KIA") &
                                   (kia_actual_rpg_real["year"] == 2024)]["runs_pg"].iloc[0]

    # 몬테카를로 표준오차(팀 R/G): 경기 득점 표준편차(~3)/sqrt(n). 슬롯 간 차이 해석의 잣대.
    team_rpg_se = 3.0 / np.sqrt(N_MOVE)
    chart_kim(move_df, opt, slot_ctx, kim_actual_order=kim["modal_order"], se=team_rpg_se)

    best_slot = int(move_df.loc[move_df["team_rpg"].idxmax(), "slot"])
    kim_modal = kim["modal_order"]
    rpg_at_modal = float(move_df.loc[move_df["slot"] == kim_modal, "team_rpg"].iloc[0])
    rpg_at_best = float(move_df["team_rpg"].max())
    print("=" * 70)
    print("[관행] 팀 최고타자 슬롯: KBO 3~4번 비율 {:.0%} vs MLB {:.0%} | "
          "1~2번 비율 KBO {:.0%} vs MLB {:.0%}".format(
              bh["kbo_share_slot34"], bh["mlb_share_slot34"],
              bh["kbo_share_slot12"], bh["mlb_share_slot12"]))
    print("  평균 최고슬롯: KBO {:.2f} vs MLB {:.2f}".format(
        bh["kbo_mean_best_slot"], bh["mlb_mean_best_slot"]))

    print("\n[김도영] KIA 2024 주전 9인 (실제 평균타순 순):")
    for d in lineup:
        print(f"  {d['mean_order']:.1f}번  {d['name']}  ({d['src']})")
    print(f"\n  캘리브레이션: 시뮬 실제순서 R/G {opt['actual_rpg']:.2f} vs 실제 KIA 2024 {kia_real:.2f}")
    print("\n[김도영 이동] 슬롯별 팀 R/G:")
    print(move_df.round(3).to_string(index=False))
    print(f"\n  → 팀 득점 최대 슬롯: {best_slot}번 (실제 최빈 {kim_modal}번, 평균 {kim['mean_order']:.1f}번)")
    print(f"  몬테카를로 SE ~ +-{team_rpg_se:.3f} R/G. 실제(최빈 {kim_modal}번) {rpg_at_modal:.3f} "
          f"vs 최적({best_slot}번) {rpg_at_best:.3f} → 차이 {rpg_at_best-rpg_at_modal:+.3f} "
          f"({(rpg_at_best-rpg_at_modal)*144:+.1f}점/144경기)")
    print(f"\n[최적화] 실제 {opt['actual_rpg']:.3f} → 근사최적 {opt['optimal_rpg']:.3f} "
          f"R/G (+{opt['gain_rpg']:.3f}), 최적 순서에서 김도영 {opt['kim_optimal_slot']}번")
    print("  최적 순서:", " - ".join(f"{i+1}.{n}" for i, n in enumerate(opt["optimal_order"])))
    print("\n[경험적] KBO 슬롯별 경기당 타석/타점/득점 (2021-2025):")
    print(slot_ctx.round(3).to_string(index=False))

    result = {
        "convention_best_slot": {k: (v if not isinstance(v, pd.Series) else v.to_dict())
                                 for k, v in bh.items()},
        "kim_lineup": [{"name": d["name"], "mean_order": d["mean_order"], "src": d["src"]}
                       for d in lineup],
        "calibration_sim_vs_real_rpg": {"sim": opt["actual_rpg"], "real": float(kia_real)},
        "kim_move": move_df.to_dict("records"),
        "kim_best_slot": best_slot,
        "kim_modal_order": kim_modal,
        "kim_actual_mean_order": kim["mean_order"],
        "montecarlo_se_rpg": float(team_rpg_se),
        "rpg_at_modal": rpg_at_modal,
        "rpg_at_best": rpg_at_best,
        "gain_best_vs_modal_per144": float((rpg_at_best - rpg_at_modal) * 144),
        "optimize": {k: v for k, v in opt.items()},
        "empirical_slot_context": slot_ctx.to_dict("records"),
    }
    with open(OUT / "kim_doyoung_sim.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n저장 완료: {OUT / 'kim_doyoung_sim.json'}")


if __name__ == "__main__":
    main()
