"""Phase 2 준비: 선수-팀-시즌별 슬롯 점유 벡터(1~9번 타석).

경기별 박스스코어의 batOrder(교체 선수는 들어간 자리의 번호)와 PA 근사(ab+bb)로
선수가 각 슬롯에서 소화한 타석을 합산한다. MLB 박스에는 pa/hbp/sf가 있지만 KBO와
같은 방법을 쓰기 위해 두 리그 모두 ab+bb로 근사한다(§3-4, 한계 명시).

입력: data/raw/kbo_boxscore_batters_{y}.csv, data/raw/mlb30_boxscore_batters_{y}.csv,
      data/processed/h1_players.csv(커버리지 대조)
출력: data/processed/h1_slot_player.csv  (league, year, team, pid, name, slot_pa, s1..s9)
      data/processed/h1_slot_team.csv    (league, year, team, games, slot_pa, s1..s9)
      outputs/h1_suitable/phase2_slot_coverage.csv

실행: python src/h1_suitable/03_slot_occupancy.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import h1_common as H  # noqa: E402

SLOTS = [f"s{i}" for i in range(1, 10)]


def load_box() -> pd.DataFrame:
    k = H.load_kbo_box(C.SEASONS).assign(league="KBO")
    m = pd.concat([pd.read_csv(C.RAW / f"mlb30_boxscore_batters_{y}.csv", encoding="utf-8-sig")
                   .drop_duplicates(["gameId", "playerCode", "batOrder"]).assign(year=y)
                   for y in C.SEASONS], ignore_index=True).assign(league="MLB")
    cols = ["league", "year", "gameId", "team", "batOrder", "playerCode", "name", "ab", "bb"]
    b = pd.concat([k[cols], m[cols]], ignore_index=True)
    b = b[b["batOrder"].between(1, 9)].copy()
    b["pid"] = b["playerCode"].astype("int64")
    b["slot_pa"] = b["ab"] + b["bb"]
    return b


def slot_table(b: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    t = b.pivot_table(index=keys, columns="batOrder", values="slot_pa", aggfunc="sum", fill_value=0)
    t.columns = [f"s{int(c)}" for c in t.columns]
    t = t.reindex(columns=SLOTS, fill_value=0)
    t.insert(0, "slot_pa", t.sum(axis=1))
    return t.reset_index()


def main():
    b = load_box()
    names = b.drop_duplicates(["league", "year", "team", "pid"]).set_index(["league", "year", "team", "pid"])["name"]
    pl = slot_table(b, ["league", "year", "team", "pid"])
    pl.insert(4, "name", pd.MultiIndex.from_frame(pl[["league", "year", "team", "pid"]]).map(names))
    tm = slot_table(b, ["league", "year", "team"])
    tm.insert(3, "games", b.groupby(["league", "year", "team"])["gameId"].nunique().values)

    # 커버리지: 박스 ab+bb vs 시즌 기록(h1_players) ab+bb, 후보풀(PA>=300) 선수 기준
    p = pd.read_csv(C.PROC / "h1_players.csv", encoding="utf-8-sig")
    p = p[p["year"].isin(C.SEASONS)].assign(pid=lambda d: d["pid"].astype("int64"))
    p["ab_bb"] = p["ab"] + p["bb"]
    cov = p[p["pa"] >= C.POOL_MIN_PA].merge(pl[["league", "year", "team", "pid", "slot_pa"]],
                                            on=["league", "year", "team", "pid"], how="left")
    cov["ratio"] = cov["slot_pa"] / cov["ab_bb"]
    summ = cov.groupby(["league", "year"]).agg(
        pool_players=("pid", "size"), missing=("slot_pa", lambda s: int(s.isna().sum())),
        ratio_median=("ratio", "median"), within5pct=("ratio", lambda s: float((s - 1).abs().le(0.05).mean())),
    ).reset_index()
    team_games = tm.groupby(["league", "year"])["games"].agg(["min", "max"]).reset_index()
    summ = summ.merge(team_games, on=["league", "year"])
    print(summ.round(3).to_string(index=False))

    pl.to_csv(C.PROC / "h1_slot_player.csv", index=False, encoding="utf-8-sig")
    tm.to_csv(C.PROC / "h1_slot_team.csv", index=False, encoding="utf-8-sig")
    C.OUT.mkdir(parents=True, exist_ok=True)
    summ.to_csv(C.OUT / "phase2_slot_coverage.csv", index=False, encoding="utf-8-sig")
    print(f"선수-팀-시즌 {len(pl)}, 팀-시즌 {len(tm)} 저장")


if __name__ == "__main__":
    main()
