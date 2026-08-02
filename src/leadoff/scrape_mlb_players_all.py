"""MLB StatsAPI에서 팀x시즌별 '전체' 타자 시즌기록(규정타석 무관)을 수집한다.

기존 scrape_mlb_player_hitting.py는 규정타석 이상만 주므로, 부분출전 선수까지
포함해 '포지션별 최다타석' 로스터 선정과 '최고타자' 랭킹(OPS 기준)에 쓴다.

  https://statsapi.mlb.com/api/v1/stats?stats=season&group=hitting&season={y}
    &sportId=1&gameType=R&playerPool=all&teamId={id}

출력: data/raw/mlb_players_hitting_all_{year}.csv
"""
import time
from pathlib import Path

import pandas as pd
import requests

from scrape_mlb_boxscores import TEAMS, YEARS, HEADERS, _get

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

STAT_KEYS = [
    "plateAppearances", "atBats", "hits", "doubles", "triples", "homeRuns",
    "baseOnBalls", "intentionalWalks", "hitByPitch", "sacFlies", "strikeOuts",
    "obp", "slg", "ops",
]


def fetch_team_players(team_id: int, season: int, session: requests.Session) -> list[dict]:
    url = (f"https://statsapi.mlb.com/api/v1/stats?stats=season&group=hitting"
           f"&season={season}&sportId=1&gameType=R&playerPool=all&teamId={team_id}")
    r = _get(session, url)
    r.raise_for_status()
    splits = r.json().get("stats", [{}])[0].get("splits", [])
    rows = []
    for s in splits:
        st = s.get("stat", {})
        row = {"year": season, "name": s["player"]["fullName"], "team": s["team"]["id"]}
        for k in STAT_KEYS:
            row[k] = st.get(k)
        rows.append(row)
    return rows


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    id_to_abbr = {v: k for k, v in TEAMS.items()}

    for year in YEARS:
        rows = []
        for abbr, tid in TEAMS.items():
            rows.extend(fetch_team_players(tid, year, session))
            time.sleep(0.2)
        df = pd.DataFrame(rows)
        df["team"] = df["team"].map(id_to_abbr)
        for c in ["obp", "slg", "ops"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        out_path = DATA_DIR / f"mlb_players_hitting_all_{year}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"{year}: {len(df)}명 저장 -> {out_path}")


if __name__ == "__main__":
    main()
