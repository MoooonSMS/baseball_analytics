"""업그레이드된 lineup_sim 엔진으로 '최적 타순 vs 실제 타순' 이득을 KBO 50개 팀-시즌
전체에서 재검증한다 (Phase 4 — task#2 계획의 최종 검증 단계).

배경: `kim_doyoung_analysis.py`의 `optimize_order`(KIA 2024 1개 팀-시즌)와
`best_hitter_move_sim.py`(50개 팀-시즌, '최고타자만 2번 이동')는 모두 구엔진(고정
확률 상수 2개, 병살·도루 없음)으로 이득이 The Book이 인용하는 시즌 5~15점
(144경기 환산 약 4.4~13.3점)보다 훨씬 작다는 결론을 냈다. `lineup_sim.py`를
KBO 실측 상황조건부 확률(병살·도루 포함)로 업그레이드한 뒤(Phase 3), 같은 '최적 vs
실제' 비교를 50개 팀-시즌 전체로 확장해 결과가 The Book 범위에 가까워지는지 확인한다.

방법: `best_hitter_move_core.load_team_lineup`으로 팀-시즌별 주전 9인 이벤트벡터를
구성한다(best_hitter_move_sim.py의 KBO Config 재사용 — 로스터 선정 방식이 기존
분석들과 동일해 결과가 서로 비교 가능하다). `kim_doyoung_analysis.optimize_order`의
pairwise-swap hill climbing 로직을 그대로 쓰되, `situational_probs`(Phase 3 업그레이드
엔진)를 적용한다. 실제(평균타순) 순서 대비 근사 최적 순서의 이득을 대응표본(paired)
t-test/Wilcoxon으로 50개 팀-시즌 전체에서 집계한다(집계 로직은 best_hitter_move_core의
aggregate_stats를 그대로 재사용).

출력: outputs/leadoff_analysis/16_lineup_optimize_all_teams.png,
      lineup_optimize_sim.json, lineup_optimize_table.csv
"""
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import best_hitter_move_core as core
import best_hitter_move_sim as bhs
import lineup_sim as ls

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs" / "leadoff_analysis"

BLUE = "#2a78d6"
GREEN = "#0b7d2b"
WARN = "#c0392b"

# The Book이 인용하는 "최적 vs 실제 라인업" 이득: 시즌 5~15점 -> 144경기 기준 그대로
# (KBO 정규시즌도 144경기라 환산 불필요, 다만 원문 인용이 MLB 162경기 기준일 가능성
# 있어 참고용 범위로만 표시)
THE_BOOK_LOW, THE_BOOK_HIGH = 4.4, 13.3


def optimize_order(events_list: list, situational_probs: dict,
                   n_search: int = 8000, n_final: int = 40000, seed: int = 0) -> dict:
    """kim_doyoung_analysis.optimize_order를 팀 무관·situational_probs 지원으로 일반화."""
    order = list(range(9))  # 실제(평균타순) 순서에서 출발

    def rpg(idx_order, n):
        arr = np.array([events_list[i] for i in idx_order])
        return ls.simulate_lineup(arr, n_games=n, seed=seed,
                                  situational_probs=situational_probs)["runs_per_game"]

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
    actual = list(range(9))
    actual_rpg = rpg(actual, n_final)
    opt_rpg = rpg(order, n_final)
    return {"actual_rpg": actual_rpg, "optimal_rpg": opt_rpg, "optimal_order_idx": order}


def _run_one_team_season(year: int, team: str, n_search: int, n_final: int, seed_base: int):
    """워커 프로세스에서 팀-시즌 1건을 처리. 다른 팀-시즌과 완전히 독립적이라 병렬화 대상."""
    players = bhs.CFG.load_players()
    lineup = core.load_team_lineup(bhs.CFG, year, team, players)
    if len(lineup) < 9:
        return None, f"  [skip] {year} {team}: 주전 9인 미달({len(lineup)}명)"

    probs = ls.load_situational_probs()
    events_list = [d["events"] for d in lineup]
    names = [d["name"] for d in lineup]
    seed = seed_base + zlib.crc32(f"{year}{team}".encode()) % 10000

    res = optimize_order(events_list, probs, n_search=n_search, n_final=n_final, seed=seed)
    gain144 = (res["optimal_rpg"] - res["actual_rpg"]) * 144
    opt_names = [names[i] for i in res["optimal_order_idx"]]
    real_val = bhs.real_team_rpg(year, team)
    row = {
        "year": year, "team": team, "rpg_actual": res["actual_rpg"],
        "rpg_optimal": res["optimal_rpg"], "gain_144g": gain144,
        "real_team_rpg": real_val, "optimal_order": " - ".join(opt_names),
    }
    msg = (f"  {year} {team:2s}  실제 R/G={res['actual_rpg']:.3f} -> "
           f"최적 R/G={res['optimal_rpg']:.3f}  gain={gain144:+.1f}점/144경기")
    return row, msg


def run_all(teams, years, n_search: int, n_final: int, seed_base: int = 3000,
           n_workers: int = 1) -> pd.DataFrame:
    tasks = [(year, team) for year in years for team in teams]

    if n_workers <= 1:
        results = [_run_one_team_season(year, team, n_search, n_final, seed_base)
                  for year, team in tasks]
    else:
        results = [None] * len(tasks)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            fut_to_idx = {
                ex.submit(_run_one_team_season, year, team, n_search, n_final, seed_base): i
                for i, (year, team) in enumerate(tasks)
            }
            for fut in as_completed(fut_to_idx):
                results[fut_to_idx[fut]] = fut.result()

    rows = []
    for row, msg in results:
        print(msg, flush=True)
        if row is not None:
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["year", "team"]).reset_index(drop=True)
    return df


def chart(df: pd.DataFrame, agg: dict):
    d = df.sort_values("gain_144g").reset_index(drop=True)
    colors = [GREEN if g > 0 else WARN for g in d["gain_144g"]]
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(np.arange(len(d)), d["gain_144g"], color=colors)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels([f"{r.year} {r.team}" for r in d.itertuples()], fontsize=8)
    ax.axvline(0, color="black", lw=1)
    ax.axvspan(THE_BOOK_LOW, THE_BOOK_HIGH, color=BLUE, alpha=0.08,
              label=f"The Book 인용 범위 ({THE_BOOK_LOW:.1f}~{THE_BOOK_HIGH:.1f}점/144경기)")
    ax.axvline(agg["mean_gain_144g"], color=BLUE, ls="--", lw=2,
               label=f"평균 {agg['mean_gain_144g']:+.1f}점/144경기 "
                     f"(95% CI [{agg['ci95_low']:+.1f}, {agg['ci95_high']:+.1f}])")
    ax.set_xlabel("최적 타순(언덕등반 탐색) vs 실제 타순 득점 이득 (점/144경기)")
    ax.set_title(f"KBO 50개 팀-시즌: 업그레이드 엔진(병살·도루·조건부 진루) 기준 라인업 최적화 이득\n"
                f"대응표본 t-test p={agg['p_ttest']:.4f} | Wilcoxon p={agg['p_wilcoxon']:.4f} | "
                f"이득 본 팀-시즌 {agg['pct_positive']:.0f}%", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "16_lineup_optimize_all_teams.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse
    import json
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-search", type=int, default=8000)
    ap.add_argument("--n-final", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=max(1, min(12, mp.cpu_count() - 1)))
    ap.add_argument("--pilot", action="store_true", help="KIA 2024만 빠르게 실행")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    core.setup_style()

    teams = ["KIA"] if args.pilot else bhs.TEAMS
    years = [2024] if args.pilot else bhs.YEARS
    n_workers = 1 if args.pilot else args.workers

    print(f"n_search={args.n_search}, n_final={args.n_final}, teams={len(teams)}, "
          f"years={len(years)}, workers={n_workers}")
    df = run_all(teams, years, n_search=args.n_search, n_final=args.n_final, n_workers=n_workers)

    if args.pilot:
        print(df.to_string(index=False))
        return

    agg = core.aggregate_stats(df)
    print("=" * 70)
    print(f"[집계] n={agg['n']}  평균 gain={agg['mean_gain_144g']:+.1f}점/144경기  "
          f"95% CI [{agg['ci95_low']:+.1f}, {agg['ci95_high']:+.1f}]")
    print(f"  paired t-test p={agg['p_ttest']:.4f} | Wilcoxon p={agg['p_wilcoxon']:.4f} | "
          f"이득 팀-시즌 비율={agg['pct_positive']:.1f}%")
    print(f"  캘리브레이션(시뮬 실제순서 R/G - 진짜 팀 R/G): "
          f"평균차={agg['calibration_mean_diff_rpg']:+.3f}  평균절대차={agg['calibration_mean_abs_diff_rpg']:.3f}")
    print(f"  The Book 인용 범위: {THE_BOOK_LOW:.1f}~{THE_BOOK_HIGH:.1f}점/144경기")

    chart(df, agg)
    df.to_csv(OUT / "lineup_optimize_table.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "lineup_optimize_sim.json", "w", encoding="utf-8") as f:
        json.dump({"records": df.to_dict("records"), "aggregate": agg,
                  "the_book_range": [THE_BOOK_LOW, THE_BOOK_HIGH]}, f,
                 ensure_ascii=False, indent=2, default=float)
    print(f"\n저장 완료: {OUT / 'lineup_optimize_sim.json'}, "
          f"{OUT / 'lineup_optimize_table.csv'}, {OUT / '16_lineup_optimize_all_teams.png'}")


if __name__ == "__main__":
    main()
