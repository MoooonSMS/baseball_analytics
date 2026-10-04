"""Phase 3: 배치 차이의 득점 가치(반사실 타순 시뮬레이션). 설계는 config.py §5 Phase 3(2026-10-04 사전 등록).

팀-시즌마다 주전 9인(포지션별 최다 타석)을 실제 PA가중 평균 슬롯 순으로 세우고, 반사실과의 R/G 차이를
ΔR/144로 보고한다.
  A_swap    주전 중 wRC+ 1위(best) <-> 실제 2번 주전 자리 교환 (주 결과)
  A_insert  best를 2번에 삽입, 나머지 상대 순서 유지
  B_top3_best6  상위 3명(primary rank<=3) 중 주전 각자를 1번 또는 2번과 교환한 후보 중 최대
  C_full_opt    두 칸 교환 국소 탐색으로 9인 전체 재배열 (--with-c, 범위는 시험 실행 후 결정)
B·C는 선택용 시드(SIM_SEARCH_GAMES)로 고르고, 고른 타순을 독립 평가 시드로 SIM_N_GAMES 재평가한다.
실제 vs 반사실은 같은 평가 시드(대응 난수). 엔진: 두 리그 기본 엔진(주), KBO는 상황확률 엔진 병기.

입력: data/processed/h1_players.csv, h1_s_members.csv, h1_slot_player.csv, h1_team_n.csv, 박스스코어
출력: outputs/h1_suitable/phase3_sim_rows.csv, phase3_summary.csv, phase3_results.json, phase3_gain.png

실행: python src/h1_suitable/05_run_value.py --pilot          # 1팀씩 시험 + C 소요 시간
      python src/h1_suitable/05_run_value.py [--with-c] [--workers 15]
"""
import argparse
import importlib
import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "leadoff"))
import config as C  # noqa: E402
import h1_common as H  # noqa: E402
import lineup_sim as ls  # noqa: E402
from best_hitter_move_core import select_position_starters  # noqa: E402

# 포지션 코드: src/leadoff/best_hitter_move_sim.py, best_hitter_move_sim_mlb.py와 같음
POS_CFG = {
    "KBO": SimpleNamespace(valid_pos=set("포一二三유좌중우지"), pos_key=lambda b: b["pos"].fillna("").str[:1]),
    "MLB": SimpleNamespace(valid_pos={"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"},
                           pos_key=lambda b: b["pos"]),
}
SLOTS = [f"s{i}" for i in range(1, 10)]
CF_LABEL = {"A_swap": "A: 1위 타자 ↔ 2번 교환", "A_insert": "A': 1위 타자 2번 삽입",
            "B_top3_best6": "B: 상위 3명 1~2번 최적", "C_full_opt": "C: 전체 재배열"}


# ======================================================================
# 라인업 구성
# ======================================================================
def load_boxes() -> pd.DataFrame:
    k = H.load_kbo_box(C.SEASONS).assign(league="KBO")
    m = pd.concat([pd.read_csv(C.RAW / f"mlb30_boxscore_batters_{y}.csv", encoding="utf-8-sig")
                   .drop_duplicates(["gameId", "playerCode", "batOrder"]).assign(year=y)
                   for y in C.SEASONS], ignore_index=True).assign(league="MLB")
    cols = ["league", "year", "gameId", "team", "batOrder", "playerCode", "pos", "ab", "bb", "teamScore"]
    b = pd.concat([k[cols], m[cols]], ignore_index=True)
    b["pid"] = b["playerCode"].astype("int64")
    return b


def build_lineups() -> tuple[list[dict], list[str]]:
    p = pd.read_csv(C.PROC / "h1_players.csv", encoding="utf-8-sig").assign(pid=lambda d: d["pid"].astype("int64"))
    p = p.set_index(["league", "year", "team", "pid"])
    mem = pd.read_csv(C.PROC / "h1_s_members.csv", encoding="utf-8-sig")
    top3 = (mem[(mem["variant"] == "primary") & (mem["rank"] <= C.TOP_K)]
            .assign(pid=lambda d: d["pid"].astype("int64"))
            .groupby(["league", "year", "team"])["pid"].apply(set))
    sp = pd.read_csv(C.PROC / "h1_slot_player.csv", encoding="utf-8-sig").set_index(["league", "year", "team", "pid"])
    mean_slot = (sp[SLOTS] * np.arange(1, 10)).sum(axis=1) / sp["slot_pa"]
    tn = pd.read_csv(C.PROC / "h1_team_n.csv", encoding="utf-8-sig")
    nmap = tn[tn["variant"] == "primary"].set_index(["league", "year", "team"])["N"]
    box = load_boxes()
    real = box.drop_duplicates(["league", "gameId", "team"]).groupby(["league", "year", "team"])["teamScore"].mean()

    tasks, skipped = [], []
    for (lg, y, tm), b in box.groupby(["league", "year", "team"]):
        starters = select_position_starters(POS_CFG[lg], b.assign(name=b["pid"]))
        rows = []
        for pid in starters:
            key = (lg, y, tm, pid)
            if key not in p.index:
                continue
            r = p.loc[key]
            ev = ls.events_from_full(pa=r["pa"], ab=r["ab"], h=r["h"], d2=r["d2"], d3=r["d3"], hr=r["hr"],
                                     bb=r["bb"], hbp=r["hbp"], sf=r["sf"])
            rows.append({"pid": int(pid), "name": r["name"], "wrc": float(r["wrc"]), "pa": int(r["pa"]),
                         "mean_slot": float(mean_slot.get(key, np.nan)), "events": ev})
        if len(rows) < 9:
            skipped.append(f"{lg} {y} {tm}: 주전 {len(rows)}명")
            continue
        rows.sort(key=lambda d: d["mean_slot"])
        t3 = top3.get((lg, y, tm), set())
        tasks.append({"league": lg, "year": int(y), "team": tm, "N": int(nmap.get((lg, y, tm), -1)),
                      "real_rpg": float(real[(lg, y, tm)]), "lineup": rows,
                      "top3_idx": [i for i, d in enumerate(rows) if d["pid"] in t3]})
    return tasks, skipped


# ======================================================================
# 시뮬레이션 (워커)
# ======================================================================
def seed_of(task: dict, tag: str) -> int:
    return C.SIM_SEED_BASE + zlib.crc32(f"{task['league']}|{task['year']}|{task['team']}|{tag}".encode()) % 100_000


def swap(order: list, i: int, j: int) -> list:
    o = order.copy()
    o[i], o[j] = o[j], o[i]
    return o


def local_search(rpg, start: list, n: int, seed: int) -> list:
    """두 칸 교환 국소 탐색(lineup_optimize_all_teams.optimize_order와 같은 절차)."""
    order, best = start, rpg(start, n, seed)
    improved = True
    while improved:
        improved = False
        for i in range(9):
            for j in range(i + 1, 9):
                cand = swap(order, i, j)
                v = rpg(cand, n, seed)
                if v > best + 1e-4:
                    order, best, improved = cand, v, True
    return order


def run_task(task: dict, with_c: bool) -> list[dict]:
    lineup = task["lineup"]
    ev = [d["events"] for d in lineup]
    engines = {"basic": None}
    if task["league"] == "KBO":
        engines["situational"] = ls.load_situational_probs()
    best = max(range(9), key=lambda i: lineup[i]["wrc"])
    base = list(range(9))
    out = []
    for eng, sp in engines.items():
        def rpg(order, n, seed, sp=sp):
            return ls.simulate_lineup(np.array([ev[i] for i in order]), n_games=n, seed=seed,
                                      situational_probs=sp)["runs_per_game"]
        es, ss = seed_of(task, "eval"), seed_of(task, "search")
        t0 = time.time()
        actual = rpg(base, C.SIM_N_GAMES, es)
        cf = {}
        cf["A_swap"] = swap(base, best, 1)
        a_ins = [i for i in base if i != best]
        cf["A_insert"] = a_ins[:1] + [best] + a_ins[1:]
        cands = [swap(base, i, t) for i in task["top3_idx"] for t in (0, 1)]
        if cands:
            vals = [rpg(o, C.SIM_SEARCH_GAMES, ss) for o in cands]
            cf["B_top3_best6"] = cands[int(np.argmax(vals))]
        if with_c:
            cf["C_full_opt"] = local_search(rpg, base, C.SIM_SEARCH_GAMES, ss)
        cache = {tuple(base): actual}
        row = {k: task[k] for k in ["league", "year", "team", "N", "real_rpg"]}
        row |= {"engine": eng, "rpg_actual": actual, "best_name": lineup[best]["name"],
                "best_actual_slot": best + 1, "n_top3_starters": len(task["top3_idx"])}
        for k, o in cf.items():
            if tuple(o) not in cache:
                cache[tuple(o)] = rpg(o, C.SIM_N_GAMES, es)
            row[f"rpg_{k}"] = cache[tuple(o)]
            row[f"gain_{k}"] = (cache[tuple(o)] - actual) * C.SIM_GAMES_PER_SEASON
            row[f"order_{k}"] = "-".join(str(lineup[i]["pid"]) for i in o)
        row["order_actual"] = "-".join(str(d["pid"]) for d in lineup)
        row["names_actual"] = "-".join(str(d["name"]) for d in lineup)
        row["seconds"] = time.time() - t0
        out.append(row)
    return out


# ======================================================================
# 집계
# ======================================================================
def summarize(df: pd.DataFrame, cfs: list[str]) -> pd.DataFrame:
    P25 = importlib.import_module("04b_slot_detail")
    df = df.assign(club=df["league"] + "_" + df["team"])
    rows = []
    for sample, d in [("N3", df[df["N"] == 3]), ("all", df)]:
        for eng in ["basic", "situational"]:
            e = d[d["engine"] == eng]
            if eng == "situational":       # KBO만: 리그 비교 없이 KBO 값만
                for cf in cfs:
                    x = e[f"gain_{cf}"].to_numpy(float)
                    g = np.random.default_rng(zlib.crc32(f"sit|{sample}|{cf}".encode()))
                    bt = x[g.integers(0, len(x), (C.BOOTSTRAP_B, len(x)))].mean(axis=1)
                    rows.append({"sample": sample, "engine": eng, "col": f"gain_{cf}", "kbo": x.mean(),
                                 "n_kbo": len(x), "kbo_ts_lo": np.percentile(bt, 2.5),
                                 "kbo_ts_hi": np.percentile(bt, 97.5)})
                continue
            r = P25.compare(e, [f"gain_{cf}" for cf in cfs], f"phase3|{sample}")
            rows += [x | {"sample": sample, "engine": eng} for x in r.to_dict("records")]
    return pd.DataFrame(rows)


def calibration(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(diff=df["rpg_actual"] - df["real_rpg"])
    return d.groupby(["league", "engine"]).agg(team_seasons=("diff", "size"), sim_rpg=("rpg_actual", "mean"),
                                               real_rpg=("real_rpg", "mean"), mean_diff=("diff", "mean"),
                                               mae=("diff", lambda s: s.abs().mean())).reset_index()


def chart(summ: pd.DataFrame, cfs: list[str]):
    P2 = importlib.import_module("04_placement_tests")
    plt = P2.setup_plot()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), dpi=150, sharex=True)
    for ax, sample, title in [(axes[0], "N3", "주 표본: N=3 팀-시즌"), (axes[1], "all", "전체 팀-시즌")]:
        s = summ[(summ["sample"] == sample) & (summ["engine"] == "basic")].set_index("col")
        ys = np.arange(len(cfs))[::-1]
        for j, lg in enumerate(["KBO", "MLB"]):
            l = lg.lower()
            r = s.loc[[f"gain_{c}" for c in cfs]]
            yy = ys + (0.13 if j == 0 else -0.13)
            ax.errorbar(r[l], yy, xerr=[r[l] - r[f"{l}_ts_lo"], r[f"{l}_ts_hi"] - r[l]], fmt="o", ms=6,
                        color=C.COLOR[lg], ecolor=C.COLOR[lg], elinewidth=2, capsize=0, label=lg)
        ax.axvline(0, color="#888888", lw=1)
        ax.set_yticks(ys, [CF_LABEL[c] for c in cfs], fontsize=8.5)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_xlabel("실제 타순 대비 득점 변화 (점/144경기)", fontsize=9)
        P2.style(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper right", fontsize=9, ncol=2, bbox_to_anchor=(0.985, 1.0))
    fig.suptitle("반사실 타순의 득점 가치, 기본 엔진 (선: 팀-시즌 부트스트랩 95% CI)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(C.OUT / "phase3_gain.png")
    plt.close(fig)


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="KBO·MLB 1팀씩, C 포함, 소요 시간 측정")
    ap.add_argument("--with-c", action="store_true")
    ap.add_argument("--workers", type=int, default=15)
    args = ap.parse_args()

    tasks, skipped = build_lineups()
    print(f"팀-시즌 {len(tasks)}개 구성, 제외 {len(skipped)}개", *skipped, sep="\n  ")
    if args.pilot:
        pick = [next(t for t in tasks if t["league"] == lg and t["N"] == 3) for lg in ["KBO", "MLB"]]
        for t in pick:
            print(f"\n[{t['league']} {t['year']} {t['team']}] 실제 타순:",
                  ", ".join(f"{i + 1}.{d['name']}({d['wrc']:.0f})" for i, d in enumerate(t["lineup"])))
            for r in run_task(t, with_c=True):
                print(f"  {r['engine']:11s} {r['seconds']:6.1f}s  실제 {r['rpg_actual']:.3f} R/G (실측 {r['real_rpg']:.3f})  "
                      + "  ".join(f"{k}={r[f'gain_{k}']:+.1f}" for k in C.SIM_CF if f"gain_{k}" in r))
        return

    cfs = [c for c in C.SIM_CF if c != "C_full_opt" or args.with_c]
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_task, t, args.with_c): t for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            rows += f.result()
            if i % 20 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows).sort_values(["league", "year", "team", "engine"])
    df.to_csv(C.OUT / "phase3_sim_rows.csv", index=False, encoding="utf-8-sig")

    cal = calibration(df)
    summ = summarize(df, cfs)
    print("\n캘리브레이션:\n", cal.round(3).to_string(index=False))
    show = ["sample", "engine", "col", "n_kbo", "n_mlb", "kbo", "kbo_ts_lo", "kbo_ts_hi", "mlb", "mlb_ts_lo",
            "mlb_ts_hi", "diff", "diff_ts_lo", "diff_ts_hi", "diff_cl_lo", "diff_cl_hi", "wild_p"]
    print("\nΔR/144:\n", summ.reindex(columns=show).round(2).to_string(index=False))
    summ.to_csv(C.OUT / "phase3_summary.csv", index=False, encoding="utf-8-sig")
    cal.to_csv(C.OUT / "phase3_calibration.csv", index=False, encoding="utf-8-sig")
    with open(C.OUT / "phase3_results.json", "w", encoding="utf-8") as f:
        json.dump({"cfs": cfs, "skipped": skipped, "calibration": cal.round(4).to_dict("records"),
                   "summary": summ.round(4).to_dict("records"),
                   "best_already_slot2": df[df["engine"] == "basic"].groupby("league")["best_actual_slot"]
                   .apply(lambda s: float((s == 2).mean())).to_dict()},
                  f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    chart(summ, cfs)
    print(f"\n저장 완료 ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
