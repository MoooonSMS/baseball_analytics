"""MLB StatsAPI에서 규정타석(Qualified) 타자의 개인 시즌 기록을 수집한다.

  https://statsapi.mlb.com/api/v1/stats?stats=season&group=hitting&season={y}
    &sportId=1&gameType=R&playerPool=Qualified&limit=300

KBO 리드오프 분석용 hitters_{y}.csv(3.1*경기수 규정타석 기준)와 표본 정의를
맞추기 위해 KBO 데이터가 있는 2021-2025년만 수집한다(연도 확장은 다른 목적).

출력: data/raw/mlb_player_hitting_{year}.csv
  (playerId, name, team, teamId, pa, ab, h, 2b, 3b, hr, bb, ibb, hbp, so, sf,
   sac, tb, avg, obp, slg, ops)
"""
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
YEARS = [2021, 2022, 2023, 2024, 2025]

# MLB 규정타석 기준(3.1 PA * 팀 경기수)에 못 미치는 시즌 보정용 폴백.
# playerPool=Qualified가 비어있거나 이상하면 전체 타자를 받아 PA 기준으로 직접 거른다.
MIN_PA_FALLBACK = 502


def _fetch(url: str, session: requests.Session) -> dict:
    for attempt in range(4):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def fetch_qualified(season: int, session: requests.Session) -> list[dict]:
    url = (
        "https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=hitting&season={season}&sportId=1"
        "&gameType=R&playerPool=Qualified&limit=300"
    )
    data = _fetch(url, session)
    stats = data.get("stats", [])
    splits = stats[0].get("splits", []) if stats else []

    if not splits:
        # 폴백: 전체 타자를 페이지네이션으로 받아 PA로 직접 거른다.
        splits = []
        offset = 0
        while True:
            url2 = (
                "https://statsapi.mlb.com/api/v1/stats"
                f"?stats=season&group=hitting&season={season}&sportId=1"
                f"&gameType=R&limit=300&offset={offset}"
            )
            data2 = _fetch(url2, session)
            stats2 = data2.get("stats", [])
            page = stats2[0].get("splits", []) if stats2 else []
            if not page:
                break
            splits.extend(page)
            offset += len(page)
            if len(page) < 300:
                break
        splits = [s for s in splits if s.get("stat", {}).get("plateAppearances", 0) >= MIN_PA_FALLBACK]

    rows = []
    for sp in splits:
        st = sp.get("stat", {})
        rows.append({
            "season": season,
            "playerId": sp["player"]["id"],
            "name": sp["player"]["fullName"],
            "team": sp.get("team", {}).get("name"),
            "teamId": sp.get("team", {}).get("id"),
            "pa": st.get("plateAppearances"),
            "ab": st.get("atBats"),
            "h": st.get("hits"),
            "2b": st.get("doubles"),
            "3b": st.get("triples"),
            "hr": st.get("homeRuns"),
            "bb": st.get("baseOnBalls"),
            "ibb": st.get("intentionalWalks"),
            "hbp": st.get("hitByPitch"),
            "so": st.get("strikeOuts"),
            "sf": st.get("sacFlies"),
            "sac": st.get("sacBunts"),
            "tb": st.get("totalBases"),
            "avg": st.get("avg"),
            "obp": st.get("obp"),
            "slg": st.get("slg"),
            "ops": st.get("ops"),
        })
    return rows


def main(years=YEARS):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    for y in years:
        out = DATA_DIR / f"mlb_player_hitting_{y}.csv"
        if out.exists():
            print(f"{y}: skip (already saved)", flush=True)
            continue
        rows = fetch_qualified(y, session)
        df = pd.DataFrame(rows)
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"{y}: {len(df)} qualified hitters saved -> {out}", flush=True)
        time.sleep(0.3)


if __name__ == "__main__":
    main()
