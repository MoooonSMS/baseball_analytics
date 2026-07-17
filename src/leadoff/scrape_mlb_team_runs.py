"""MLB StatsAPI에서 팀-시즌 총 득점/PA를 수집한다 (타순 스플릿에는 runs가 없어 별도 수집).

출력: data/raw/mlb_team_runs.csv (season, team, teamId, gamesPlayed, plateAppearances, runs)
"""
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def fetch_season(season: int, session: requests.Session) -> list[dict]:
    url = (
        "https://statsapi.mlb.com/api/v1/teams/stats"
        f"?season={season}&group=hitting&stats=season&sportId=1"
    )
    resp = session.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    splits = resp.json()["stats"][0]["splits"]
    rows = []
    for sp in splits:
        st = sp["stat"]
        rows.append({
            "season": season,
            "team": sp["team"]["name"],
            "teamId": sp["team"]["id"],
            "gamesPlayed": st.get("gamesPlayed"),
            "plateAppearances": st.get("plateAppearances"),
            "runs": st.get("runs"),
        })
    return rows


def main(seasons=range(2010, 2026)):
    out = DATA_DIR / "mlb_team_runs.csv"
    done = set()
    if out.exists():
        done = set(pd.read_csv(out)["season"].unique())
    session = requests.Session()
    for season in seasons:
        if season in done:
            continue
        rows = fetch_season(season, session)
        df = pd.DataFrame(rows)
        df.to_csv(out, mode="a", header=not out.exists(), index=False, encoding="utf-8-sig")
        print(f"{season}: {len(df)} teams saved", flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
