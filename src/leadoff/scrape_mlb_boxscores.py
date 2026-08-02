"""MLB StatsAPI에서 팀x시즌별 경기별 타자 박스스코어(타순/포지션 포함)를 수집한다.

KBO의 kbo_boxscore_batters_{year}.csv와 동일한 스키마로 저장해
best_hitter_move_sim.py 파이프라인을 그대로 재사용할 수 있게 한다.

  1) schedule API로 팀-시즌의 gamePk 열거
     https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId={id}&season={y}&gameType=R
  2) 경기마다 live feed에서 battingOrder/position/타격기록 추출
     https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live

표본: KBO(10팀x5년=50팀-시즌)와 직접 비교 가능하도록 MLB도 10팀x5년으로 고정.
중단 후 재실행하면 이미 저장된 gameId는 건너뛴다(연도별 CSV에 append).

출력: data/raw/mlb_boxscore_batters_{year}.csv
"""
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# 지구/시장규모를 섞은 고정 10팀 (KBO와 동일 표본크기로 비교하기 위한 임의 고정 선택)
TEAMS = {
    "SEA": 136, "SF": 137, "STL": 138, "TOR": 141, "ATL": 144,
    "NYY": 147, "BOS": 111, "CHC": 112, "HOU": 117, "LAD": 119,
}
YEARS = [2021, 2022, 2023, 2024, 2025]

FIELDS = [
    "gameId", "gameDate", "team", "opponent", "homeAway", "teamScore", "oppScore",
    "batOrder", "playerCode", "name", "pos", "ab", "hit", "hr", "bb", "kk", "run", "rbi",
]


def _get(session: requests.Session, url: str):
    for attempt in range(4):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            return r
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def fetch_schedule(team_id: int, season: int, session: requests.Session) -> list[int]:
    url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId={team_id}"
           f"&season={season}&gameType=R")
    r = _get(session, url)
    r.raise_for_status()
    pks = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("detailedState") == "Final":
                pks.append(g["gamePk"])
    return pks


def fetch_game_rows(game_pk: int, session: requests.Session, wanted_abbrevs: set) -> list[dict]:
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    r = _get(session, url)
    if r is None or r.status_code != 200:
        return []
    data = r.json()
    gd = data.get("gameData", {})
    if gd.get("status", {}).get("detailedState") != "Final":
        return []
    game_date = gd["datetime"]["officialDate"]
    home_abbr = gd["teams"]["home"]["abbreviation"]
    away_abbr = gd["teams"]["away"]["abbreviation"]
    ls = data.get("liveData", {}).get("linescore", {}).get("teams", {})
    home_score = ls.get("home", {}).get("runs")
    away_score = ls.get("away", {}).get("runs")

    box = data.get("liveData", {}).get("boxscore", {}).get("teams", {})
    rows = []
    for side, team_abbr, opp_abbr, team_score, opp_score in [
        ("home", home_abbr, away_abbr, home_score, away_score),
        ("away", away_abbr, home_abbr, away_score, home_score),
    ]:
        if team_abbr not in wanted_abbrevs:
            continue
        for pl in box.get(side, {}).get("players", {}).values():
            bo = pl.get("battingOrder")
            if not bo:
                continue
            bat = pl.get("stats", {}).get("batting", {})
            rows.append({
                "gameId": game_pk, "gameDate": game_date, "team": team_abbr,
                "opponent": opp_abbr, "homeAway": side,
                "teamScore": team_score, "oppScore": opp_score,
                "batOrder": int(bo) // 100,
                "playerCode": pl["person"]["id"], "name": pl["person"]["fullName"],
                "pos": pl.get("position", {}).get("abbreviation"),
                "ab": bat.get("atBats", 0), "hit": bat.get("hits", 0),
                "hr": bat.get("homeRuns", 0), "bb": bat.get("baseOnBalls", 0),
                "kk": bat.get("strikeOuts", 0), "run": bat.get("runs", 0),
                "rbi": bat.get("rbi", 0),
            })
    return rows


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    wanted_abbrevs = set(TEAMS.keys())

    for year in YEARS:
        out_path = DATA_DIR / f"mlb_boxscore_batters_{year}.csv"
        existing = pd.read_csv(out_path, encoding="utf-8-sig") if out_path.exists() else None
        done_ids = set(existing["gameId"].unique()) if existing is not None else set()

        needed_pks = set()
        for abbr, tid in TEAMS.items():
            needed_pks.update(fetch_schedule(tid, year, session))
            time.sleep(0.2)

        todo = sorted(needed_pks - done_ids)
        print(f"{year}: 전체 {len(needed_pks)}경기 중 기수집 {len(done_ids)}, 남은 {len(todo)}")

        rows = []
        for i, pk in enumerate(todo):
            rows.extend(fetch_game_rows(pk, session, wanted_abbrevs))
            if (i + 1) % 200 == 0:
                print(f"  {year}: {i + 1}/{len(todo)} 진행, 누적 {len(rows)}행")
                pd.concat([existing, pd.DataFrame(rows)] if existing is not None else [pd.DataFrame(rows)],
                         ignore_index=True).to_csv(out_path, index=False, encoding="utf-8-sig")
            time.sleep(0.15)

        new_df = pd.DataFrame(rows, columns=FIELDS)
        out = pd.concat([existing, new_df], ignore_index=True) if existing is not None else new_df
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"{year} 저장 완료: {len(out)}행 -> {out_path}")


if __name__ == "__main__":
    main()
