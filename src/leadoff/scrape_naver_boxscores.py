"""네이버 스포츠 API에서 KBO 정규시즌 경기별 타자 박스스코어를 수집한다.

타순(batOrder)이 게임 단위로 기록되어 있어 '강한 2번' 분석의 원천 데이터가 된다.

  1) schedule API로 시즌 전체 gameId 열거
     https://api-gw.sports.naver.com/schedule/games?...&fromDate=...&toDate=...
  2) 경기마다 record API에서 battersBoxscore 추출
     https://api-gw.sports.naver.com/schedule/games/{gameId}/record

중단 후 재실행하면 이미 저장된 gameId는 건너뛴다(연도별 CSV에 append).
"""
import csv
import sys
import time
from datetime import date
from pathlib import Path

import requests

BASE = "https://api-gw.sports.naver.com/schedule"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

# 시즌별 정규시즌 대략 범위 (여유 있게 잡아도 statusCode=RESULT 만 수집하므로 무해)
SEASON_RANGE = {
    2021: ("2021-04-01", "2021-11-05"),
    2022: ("2022-04-01", "2022-10-15"),
    2023: ("2023-04-01", "2023-10-20"),
    2024: ("2024-03-22", "2024-10-05"),
    2025: ("2025-03-20", "2025-10-05"),
}

FIELDS = [
    "gameId", "gameDate", "team", "opponent", "homeAway", "teamScore", "oppScore",
    "batOrder", "playerCode", "name", "pos", "ab", "hit", "hr", "bb", "kk", "run", "rbi",
]


def list_game_ids(year: int, session: requests.Session) -> list[dict]:
    """정규시즌(RESULT 상태) 경기 목록을 날짜 범위로 조회한다."""
    from_date, to_date = SEASON_RANGE[year]
    url = (
        f"{BASE}/games?fields=basic&upperCategoryId=kbaseball&categoryId=kbo"
        f"&fromDate={from_date}&toDate={to_date}&size=1000"
    )
    resp = session.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    games = resp.json()["result"]["games"]
    # 취소/서스펜디드 제외, 완료 경기만
    return [g for g in games if g["statusCode"] == "RESULT" and not g["cancel"]]


def fetch_batters(game: dict, session: requests.Session) -> list[dict]:
    gid = game["gameId"]
    resp = session.get(f"{BASE}/games/{gid}/record", headers=HEADERS, timeout=15)
    resp.raise_for_status()
    rd = (resp.json().get("result") or {}).get("recordData") or {}
    bb = rd.get("battersBoxscore") or {}
    rows = []
    for side in ("away", "home"):
        batters = bb.get(side) or []
        if side == "home":
            team, opp = game["homeTeamName"], game["awayTeamName"]
            ts, os_ = game["homeTeamScore"], game["awayTeamScore"]
        else:
            team, opp = game["awayTeamName"], game["homeTeamName"]
            ts, os_ = game["awayTeamScore"], game["homeTeamScore"]
        for b in batters:
            rows.append({
                "gameId": gid,
                "gameDate": game["gameDate"],
                "team": team,
                "opponent": opp,
                "homeAway": side,
                "teamScore": ts,
                "oppScore": os_,
                "batOrder": b.get("batOrder"),
                "playerCode": b.get("playerCode"),
                "name": b.get("name"),
                "pos": b.get("pos"),
                "ab": b.get("ab"),
                "hit": b.get("hit"),
                "hr": b.get("hr"),
                "bb": b.get("bb"),
                "kk": b.get("kk"),
                "run": b.get("run"),
                "rbi": b.get("rbi"),
            })
    return rows


def done_game_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {row["gameId"] for row in csv.DictReader(f)}


def scrape_year(year: int):
    out_path = DATA_DIR / f"kbo_boxscore_batters_{year}.csv"
    session = requests.Session()
    games = list_game_ids(year, session)
    done = done_game_ids(out_path)
    todo = [g for g in games if g["gameId"] not in done]
    print(f"[{year}] total={len(games)} done={len(done)} todo={len(todo)}", flush=True)

    write_header = not out_path.exists()
    with out_path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for i, game in enumerate(todo, 1):
            try:
                rows = fetch_batters(game, session)
            except Exception as e:  # 일시 오류는 건너뛰고 재실행 시 재시도
                print(f"  ! {game['gameId']}: {e}", flush=True)
                time.sleep(2)
                continue
            writer.writerows(rows)
            if i % 50 == 0:
                f.flush()
                print(f"  [{year}] {i}/{len(todo)}", flush=True)
            time.sleep(0.3)
    print(f"[{year}] saved -> {out_path}", flush=True)


def main(years):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for year in years:
        scrape_year(year)


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or list(SEASON_RANGE)
    main(years)
