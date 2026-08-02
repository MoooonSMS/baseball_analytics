"""'최고타자 2번 이동' 반사실 시뮬레이션의 공용 코어 (KBO/MLB 공유).

KBO(best_hitter_move_sim.py)와 MLB(best_hitter_move_sim_mlb.py)는 데이터 스키마와
포지션 코드만 다를 뿐, 파이프라인(주전 9인 선정 -> 최고타자 지정 -> 실제슬롯/2번슬롯
시뮬레이션 -> 대응표본 통계 -> 차트)이 동일하다. 리그별 차이는 LeagueConfig로 주입하고,
각 리그 파일은 자신의 Config를 만들어 main()을 호출하는 얇은 엔트리포인트가 된다.
"""
import json
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs" / "leadoff_analysis"

BLUE = "#2a78d6"
GREEN = "#0b7d2b"
WARN = "#c0392b"


@dataclass
class LeagueConfig:
    league: str  # "KBO" | "MLB"
    teams: list
    years: list
    seed_base: int
    valid_pos: set
    foreigner_fallback: bool  # KBO만: 이름잘림 4자 접두어 매칭 폴백
    metric_key: str  # "woba" | "ops" — 최고타자 랭킹 지표
    load_players: Callable[[], pd.DataFrame]
    load_box: Callable[[int], pd.DataFrame]
    filter_team_box: Callable[[pd.DataFrame, str], pd.DataFrame]
    pos_key: Callable[[pd.DataFrame], pd.Series]
    build_events: Callable[[object, pd.Series], tuple]
    real_team_rpg: Callable[[int, str], float]
    png_name: str
    json_name: str
    csv_name: str
    chart_title_prefix: str
    pilot_team: str
    pilot_year: int
    x_label: str = "팀 최고타자를 2번으로 옮겼을 때 득점 이득 (점/144경기)"


def setup_style():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sns.set_theme(style="whitegrid", font="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False


def resolve_player_row(cfg: LeagueConfig, name: str, pq: pd.DataFrame):
    """이름 exact match, KBO는 실패시 외국인 선수 이름잘림(4자) 접두어 매칭 폴백."""
    if name in pq.index:
        row = pq.loc[name]
        return row.iloc[0] if isinstance(row, pd.DataFrame) else row
    if cfg.foreigner_fallback and len(name) == 4:
        cand = pq[pq.index.str.startswith(name)]
        if len(cand) == 1:
            return cand.iloc[0]
    return None


def select_position_starters(cfg: LeagueConfig, b: pd.DataFrame) -> set:
    """포지션별로 그 자리 타석수가 가장 많은 선수를 그리디로 배정.

    한 선수가 2개 이상 포지션에서 동시에 1위면, 타석수가 가장 높은(=진짜
    주포지션) 자리에 배정되고 밀려난 포지션은 차순위 선수가 채운다.
    """
    b = b.copy()
    b["pos_key"] = cfg.pos_key(b)
    valid = b[b["pos_key"].isin(cfg.valid_pos)].copy()
    valid["pa_row"] = valid["ab"] + valid["bb"]

    by_pos = valid.groupby(["name", "pos_key"], as_index=False)["pa_row"].sum()
    total_pa = valid.groupby("name")["pa_row"].sum().rename("total_pa")
    by_pos = by_pos.merge(total_pa, on="name")
    by_pos = by_pos.sort_values(["pa_row", "total_pa", "name"], ascending=[False, False, True])

    assigned_pos: dict = {}
    assigned_name: set = set()
    for r in by_pos.itertuples(index=False):
        pos = r.pos_key
        if pos in assigned_pos or r.name in assigned_name:
            continue
        assigned_pos[pos] = r.name
        assigned_name.add(r.name)

    if len(assigned_pos) < 9:  # 포지션 코드가 비는 경우 폴백(예: 2021 NL 유니버설 DH 이전)
        pool = total_pa[~total_pa.index.isin(assigned_name)].sort_values(ascending=False)
        for _p in cfg.valid_pos - set(assigned_pos):
            if pool.empty:
                break
            assigned_name.add(pool.index[0])
            pool = pool.iloc[1:]

    return assigned_name


def load_team_lineup(cfg: LeagueConfig, year: int, team: str, players: pd.DataFrame):
    """해당 팀-시즌 포지션별 최다타석 9인('주전')의 이벤트벡터·평균타순·최빈타순·랭킹지표."""
    b = cfg.load_box(year)
    b = cfg.filter_team_box(b, team).copy()
    if b.empty:
        return []
    pq = players[(players["team"] == team) & (players["year"] == year)].set_index("name")

    starter_names = select_position_starters(cfg, b)
    if len(starter_names) < 9:
        return []

    modal = (b.groupby(["name", "batOrder"]).size().reset_index(name="n")
             .sort_values("n", ascending=False).drop_duplicates("name")
             .set_index("name")["batOrder"])
    agg = b.groupby("name").agg(
        ab=("ab", "sum"), hit=("hit", "sum"), hr=("hr", "sum"), bb=("bb", "sum"),
        pa_games=("gameId", "nunique"), mean_order=("batOrder", "mean")).reset_index()
    agg["modal_order"] = agg["name"].map(modal)
    agg["pa_est"] = (agg["ab"] + agg["bb"]) / 0.97
    regulars = agg[agg["name"].isin(starter_names)].copy()

    lineup = []
    for _, r in regulars.iterrows():
        name = r["name"]
        row = resolve_player_row(cfg, name, pq)
        ev, metric, src = cfg.build_events(row, r)
        lineup.append({"name": name, "events": ev, "mean_order": float(r["mean_order"]),
                       "modal_order": int(r["modal_order"]), "pa_est": int(r["pa_est"]),
                       "src": src, "metric": metric})
    return sorted(lineup, key=lambda d: d["mean_order"])


def pick_best_hitter(lineup: list):
    known = [d for d in lineup if not np.isnan(d["metric"])]
    if not known:
        return None
    return max(known, key=lambda d: d["metric"])


def simulate_two_slots(lineup: list, target_name: str, n_games: int, seed: int) -> dict:
    """나머지 8인 고정, target을 실제 슬롯과 2번 슬롯에 각각 삽입해 팀 R/G 비교."""
    import lineup_sim as ls

    others = [d for d in lineup if d["name"] != target_name]
    others_ev = [d["events"] for d in others]
    target = next(d for d in lineup if d["name"] == target_name)

    def rpg_at(idx0: int) -> float:
        order = others_ev[:idx0] + [target["events"]] + others_ev[idx0:]
        return ls.simulate_lineup(np.array(order), n_games=n_games, seed=seed)["runs_per_game"]

    actual_slot = min(max(target["modal_order"], 1), 9)
    rpg_actual = rpg_at(actual_slot - 1)
    rpg_slot2 = rpg_at(1)
    return {"actual_slot": actual_slot, "rpg_actual": rpg_actual, "rpg_slot2": rpg_slot2}


def _run_one_team_season(cfg: LeagueConfig, year: int, team: str, n_games: int, seed_base: int):
    """워커 프로세스에서 팀-시즌 1건을 처리. 다른 팀-시즌과 완전히 독립적이라 병렬화 대상."""
    players = cfg.load_players()

    lineup = load_team_lineup(cfg, year, team, players)
    if len(lineup) < 9:
        return None, f"  [skip] {year} {team}: 주전 9인 미달({len(lineup)}명)"
    best = pick_best_hitter(lineup)
    if best is None:
        return None, f"  [skip] {year} {team}: {cfg.metric_key} 알려진 선수 없음"

    # zlib.crc32는 프로세스/플랫폼에 무관하게 결정론적(내장 hash()는 PYTHONHASHSEED
    # 솔팅으로 실행마다 문자열 해시가 달라져 시드가 재현되지 않았다).
    seed = seed_base + zlib.crc32(f"{year}{team}".encode()) % 10000
    sim = simulate_two_slots(lineup, best["name"], n_games=n_games, seed=seed)
    gain144 = (sim["rpg_slot2"] - sim["rpg_actual"]) * 144
    real_val = cfg.real_team_rpg(year, team)
    row = {
        "year": year, "team": team, "name": best["name"], cfg.metric_key: best["metric"],
        "actual_slot": sim["actual_slot"], "rpg_actual": sim["rpg_actual"],
        "rpg_slot2": sim["rpg_slot2"], "gain_144g": gain144,
        "real_team_rpg": real_val,
    }
    msg = (f"  {year} {team:3s} {best['name']:8s} {cfg.metric_key.upper()}={best['metric']:.3f}  "
           f"실제{sim['actual_slot']}번 R/G={sim['rpg_actual']:.3f} -> "
           f"2번 R/G={sim['rpg_slot2']:.3f}  gain={gain144:+.1f}점/144경기")
    return row, msg


def run_all(cfg: LeagueConfig, n_games: int, teams=None, years=None, n_workers: int = 1) -> pd.DataFrame:
    teams = cfg.teams if teams is None else teams
    years = cfg.years if years is None else years
    tasks = [(year, team) for year in years for team in teams]

    if n_workers <= 1:
        results = [_run_one_team_season(cfg, year, team, n_games, cfg.seed_base) for year, team in tasks]
    else:
        results = [None] * len(tasks)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            fut_to_idx = {
                ex.submit(_run_one_team_season, cfg, year, team, n_games, cfg.seed_base): i
                for i, (year, team) in enumerate(tasks)
            }
            for fut in as_completed(fut_to_idx):
                results[fut_to_idx[fut]] = fut.result()

    rows = []
    for row, msg in results:
        print(msg)
        if row is not None:
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["year", "team"]).reset_index(drop=True)
    return df


def aggregate_stats(df: pd.DataFrame) -> dict:
    gains = df["gain_144g"].to_numpy()
    t, p_t = stats.ttest_1samp(gains, 0)
    w, p_w = stats.wilcoxon(gains)
    ci = stats.t.interval(0.95, len(gains) - 1, loc=gains.mean(), scale=stats.sem(gains))
    calib = df.dropna(subset=["real_team_rpg"])
    calib_diff = (calib["rpg_actual"] - calib["real_team_rpg"])
    return {
        "n": len(gains), "mean_gain_144g": float(gains.mean()),
        "ci95_low": float(ci[0]), "ci95_high": float(ci[1]),
        "t": float(t), "p_ttest": float(p_t),
        "wilcoxon_w": float(w), "p_wilcoxon": float(p_w),
        "pct_positive": float((gains > 0).mean() * 100),
        "calibration_mean_diff_rpg": float(calib_diff.mean()),
        "calibration_mean_abs_diff_rpg": float(calib_diff.abs().mean()),
    }


def chart(cfg: LeagueConfig, df: pd.DataFrame, agg: dict):
    d = df.sort_values("gain_144g").reset_index(drop=True)
    colors = [GREEN if g > 0 else WARN for g in d["gain_144g"]]
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(np.arange(len(d)), d["gain_144g"], color=colors)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels([f"{r.year} {r.team} {r.name}" for r in d.itertuples()], fontsize=8)
    ax.axvline(0, color="black", lw=1)
    ax.axvline(agg["mean_gain_144g"], color=BLUE, ls="--", lw=2,
               label=f"평균 {agg['mean_gain_144g']:+.1f}점/144경기 "
                     f"(95% CI [{agg['ci95_low']:+.1f}, {agg['ci95_high']:+.1f}])")
    ax.set_xlabel(cfg.x_label)
    ax.set_title(f"{cfg.chart_title_prefix}\n"
                 f"대응표본 t-test p={agg['p_ttest']:.4f} | Wilcoxon p={agg['p_wilcoxon']:.4f} | "
                 f"이득 본 팀-시즌 {agg['pct_positive']:.0f}%", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / cfg.png_name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(cfg: LeagueConfig):
    import argparse
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=120000)
    ap.add_argument("--workers", type=int, default=max(1, min(12, mp.cpu_count() - 1)))
    ap.add_argument("--pilot", action="store_true", help=f"{cfg.pilot_team} {cfg.pilot_year}만 빠르게 실행")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()

    teams = [cfg.pilot_team] if args.pilot else cfg.teams
    years = [cfg.pilot_year] if args.pilot else cfg.years
    n_games = 5000 if args.pilot else args.n_games
    n_workers = 1 if args.pilot else args.workers

    print(f"n_games={n_games}, teams={len(teams)}, years={len(years)}, workers={n_workers}")
    df = run_all(cfg, n_games=n_games, teams=teams, years=years, n_workers=n_workers)

    if args.pilot:
        print(df.to_string(index=False))
        return

    agg = aggregate_stats(df)
    print("=" * 70)
    print(f"[집계] n={agg['n']}  평균 gain={agg['mean_gain_144g']:+.1f}점/144경기  "
          f"95% CI [{agg['ci95_low']:+.1f}, {agg['ci95_high']:+.1f}]")
    print(f"  paired t-test p={agg['p_ttest']:.4f} | Wilcoxon p={agg['p_wilcoxon']:.4f} | "
          f"이득 팀-시즌 비율={agg['pct_positive']:.1f}%")
    print(f"  캘리브레이션(시뮬 실제슬롯 R/G - 진짜 팀 R/G): "
          f"평균차={agg['calibration_mean_diff_rpg']:+.3f}  평균절대차={agg['calibration_mean_abs_diff_rpg']:.3f}")

    chart(cfg, df, agg)
    df.to_csv(OUT / cfg.csv_name, index=False, encoding="utf-8-sig")
    with open(OUT / cfg.json_name, "w", encoding="utf-8") as f:
        json.dump({"records": df.to_dict("records"), "aggregate": agg}, f,
                  ensure_ascii=False, indent=2, default=float)
    print(f"\n저장 완료: {OUT / cfg.json_name}, {OUT / cfg.csv_name}, {OUT / cfg.png_name}")
