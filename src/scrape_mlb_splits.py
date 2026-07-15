"""MLB StatsAPI에서 팀별 타순(batting order) 스플릿을 수집한다.

sitCodes b1~b9 가 타순 1~9번을 의미한다. 정규시즌(gameType=R) 팀 단위 타격 스탯.
  https://statsapi.mlb.com/api/v1/teams/stats?season={y}&group=hitting&stats=statSplits&sitCodes=b{n}&sportId=1

출력: data/raw/mlb_slot_splits.csv (season, team, slot, PA/AB/H/2B/3B/HR/BB/HBP/SF, avg/obp/slg/ops ...)
"""
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

STAT_KEYS = [
    "plateAppearances", "atBats", "runs", "hits", "doubles", "triples", "homeRuns",
    "baseOnBalls", "intentionalWalks", "hitByPitch", "strikeOuts", "sacBunts", "sacFlies",
    "stolenBases", "caughtStealing", "totalBases", "rbi",
    "avg", "obp", "slg", "ops", "babip",
]


def fetch_slot(season: int, slot: int, session: requests.Session) -> list[dict]:
    url = (
        "https://statsapi.mlb.com/api/v1/teams/stats"
        f"?season={season}&group=hitting&stats=statSplits&sitCodes=b{slot}&sportId=1"
    )
    for attempt in range(4):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            break
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    resp.raise_for_status()
    stats = resp.json().get("stats", [])
    rows = []
    for block in stats:
        for sp in block.get("splits", []):
            row = {
                "season": season,
                "slot": slot,
                "team": sp["team"]["name"],
                "teamId": sp["team"]["id"],
            }
            st = sp.get("stat", {})
            for k in STAT_KEYS:
                row[k] = st.get(k)
            rows.append(row)
    return rows


def main(seasons=range(2010, 2026)):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "mlb_slot_splits.csv"
    done_seasons = set()
    if out.exists():
        done_seasons = set(pd.read_csv(out)["season"].unique())
    session = requests.Session()
    for season in seasons:
        if season in done_seasons:
            print(f"{season}: skip (already saved)", flush=True)
            continue
        rows = []
        for slot in range(1, 10):
            rows.extend(fetch_slot(season, slot, session))
            time.sleep(0.2)
        df = pd.DataFrame(rows)
        df.to_csv(out, mode="a", header=not out.exists(), index=False, encoding="utf-8-sig")
        print(f"{season}: {len(df)} rows appended", flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
