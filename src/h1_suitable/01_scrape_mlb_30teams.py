"""MLB StatsAPI: 30팀 경기별 타순 박스스코어 + 타석 단위 play-by-play + 팀별 전체 타자 시즌 기록.

  일정    https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={y}&gameType=R  (날짜·팀·점수)
  경기    https://statsapi.mlb.com/api/v1/game/{gamePk}/boxscore, /playByPlay  (fields= 필터로 필요한 것만;
          처음엔 feed/live를 썼으나 경기당 약 800KB라 느리고 서버 부담이 커서 교체 — 파싱 결과 동일성 확인)
  타자    https://statsapi.mlb.com/api/v1/stats?stats=season&group=hitting&season={y}
            &sportId=1&gameType=R&playerPool=all&teamId={id}

박스스코어 스키마는 기존 scrape_mlb_boxscores.py(10팀)와 같다(기존 파일은 건드리지 않음).
play-by-play는 KBO relay와 같은 RE24 가중치 도출용이다. 타석마다 종료 시점의 주자·아웃과
득점을 저장하고, 시작 상태는 같은 반이닝의 직전 타석 종료 상태로 계산한다(KBO와 같은 규칙).

직렬 요청, 요청 간 0.3초, 재시도 백오프. 중단 후 재실행하면 이미 받은 gamePk는 건너뛴다.

실행: python src/h1_suitable/01_scrape_mlb_30teams.py [시즌 ...]
출력: data/raw/mlb30_boxscore_batters_{y}.csv, data/raw/mlb30_pbp_{y}.csv (gitignore),
      data/raw/mlb30_players_hitting_all_{y}.csv
"""
import csv
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

HEADERS = {"User-Agent": "Mozilla/5.0 (personal non-commercial baseball research)"}
DELAY = 0.3

BOX_FIELDS = ["gameId", "gameDate", "team", "opponent", "homeAway", "teamScore", "oppScore",
              "batOrder", "playerCode", "name", "pos", "ab", "hit", "hr", "bb", "kk", "run", "rbi",
              "d2", "d3", "ibb", "hbp", "sf", "pa"]
PBP_FIELDS = ["gameId", "atBatIndex", "inning", "half", "batTeam", "batterId", "batterName",
              "eventType", "outs_after", "b1", "b2", "b3", "runs", "isComplete"]
STAT_KEYS = ["plateAppearances", "atBats", "hits", "doubles", "triples", "homeRuns", "baseOnBalls",
             "intentionalWalks", "hitByPitch", "sacFlies", "sacBunts", "strikeOuts", "obp", "slg"]


def get(session, url):
    for attempt in range(5):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


def season_games(session, year) -> dict:
    """{gamePk: 경기 메타(날짜, 홈/원정 팀 id, 점수)}. 서스펜디드 경기는 같은 gamePk가 두 날짜에
    나오므로 마지막(완료) 항목을 쓴다."""
    d = get(session, f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={year}&gameType=R")
    games = {}
    for day in d.get("dates", []):
        for g in day.get("games", []):
            if g.get("status", {}).get("detailedState") == "Final":
                games[g["gamePk"]] = {
                    "date": g.get("officialDate", day["date"]),
                    "home": g["teams"]["home"]["team"]["id"], "away": g["teams"]["away"]["team"]["id"],
                    "home_score": g["teams"]["home"].get("score"), "away_score": g["teams"]["away"].get("score")}
    return games


def team_abbr(session, year) -> dict:
    return {t["id"]: t["abbreviation"]
            for t in get(session, f"https://statsapi.mlb.com/api/v1/teams?sportId=1&season={year}")["teams"]}


BOX_URL = ("https://statsapi.mlb.com/api/v1/game/{pk}/boxscore?fields=teams,home,away,players,person,id,"
           "fullName,battingOrder,position,abbreviation,stats,batting,atBats,hits,homeRuns,baseOnBalls,"
           "strikeOuts,runs,rbi,doubles,triples,intentionalWalks,hitByPitch,sacFlies,plateAppearances")
PBP_URL = ("https://statsapi.mlb.com/api/v1/game/{pk}/playByPlay?fields=allPlays,result,type,eventType,"
           "about,atBatIndex,halfInning,inning,isComplete,count,outs,matchup,batter,id,fullName,"
           "postOnFirst,postOnSecond,postOnThird,runners,details,isScoringEvent")


def fetch_light(session, pk, meta, abbr) -> dict:
    """필드 필터를 건 boxscore + playByPlay로 feed/live와 같은 구조의 최소 dict를 만든다
    (feed/live 대비 전송량 약 1/10)."""
    box = get(session, BOX_URL.format(pk=pk))
    time.sleep(DELAY)
    pbp = get(session, PBP_URL.format(pk=pk))
    if box is None or pbp is None:
        return None
    return {"gamePk": pk,
            "gameData": {"status": {"detailedState": "Final"}, "datetime": {"officialDate": meta["date"]},
                         "teams": {s: {"abbreviation": abbr[meta[s]]} for s in ["home", "away"]}},
            "liveData": {"linescore": {"teams": {s: {"runs": meta[f"{s}_score"]} for s in ["home", "away"]}},
                         "boxscore": box, "plays": pbp}}


def parse_game(data: dict):
    gd = data["gameData"]
    if gd.get("status", {}).get("detailedState") != "Final":
        return [], []
    pk = data["gamePk"]
    date = gd["datetime"]["officialDate"]
    abbr = {s: gd["teams"][s]["abbreviation"] for s in ["home", "away"]}
    ls = data["liveData"]["linescore"]["teams"]
    score = {s: ls[s].get("runs") for s in ["home", "away"]}
    box_rows = []
    for side, opp in [("home", "away"), ("away", "home")]:
        for pl in data["liveData"]["boxscore"]["teams"][side]["players"].values():
            bo = pl.get("battingOrder")
            if not bo:
                continue
            b = pl.get("stats", {}).get("batting", {})
            box_rows.append({
                "gameId": pk, "gameDate": date, "team": abbr[side], "opponent": abbr[opp],
                "homeAway": side, "teamScore": score[side], "oppScore": score[opp],
                "batOrder": int(bo) // 100, "playerCode": pl["person"]["id"],
                "name": pl["person"]["fullName"], "pos": pl.get("position", {}).get("abbreviation"),
                "ab": b.get("atBats", 0), "hit": b.get("hits", 0), "hr": b.get("homeRuns", 0),
                "bb": b.get("baseOnBalls", 0), "kk": b.get("strikeOuts", 0), "run": b.get("runs", 0),
                "rbi": b.get("rbi", 0), "d2": b.get("doubles", 0), "d3": b.get("triples", 0),
                "ibb": b.get("intentionalWalks", 0), "hbp": b.get("hitByPitch", 0),
                "sf": b.get("sacFlies", 0), "pa": b.get("plateAppearances", 0)})
    pbp_rows = []
    for p in data["liveData"]["plays"]["allPlays"]:
        if p["result"].get("type") != "atBat":
            continue
        m, ab = p["matchup"], p["about"]
        half = ab["halfInning"]
        pbp_rows.append({
            "gameId": pk, "atBatIndex": ab["atBatIndex"], "inning": ab["inning"], "half": half,
            "batTeam": abbr["away"] if half == "top" else abbr["home"],
            "batterId": m["batter"]["id"], "batterName": m["batter"]["fullName"],
            "eventType": p["result"].get("eventType"), "outs_after": p["count"]["outs"],
            "b1": (m.get("postOnFirst") or {}).get("id", 0),
            "b2": (m.get("postOnSecond") or {}).get("id", 0),
            "b3": (m.get("postOnThird") or {}).get("id", 0),
            "runs": sum(1 for r in p.get("runners", []) if r["details"].get("isScoringEvent")),
            "isComplete": ab.get("isComplete")})
    return box_rows, pbp_rows


def done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {int(r["gameId"]) for r in csv.DictReader(f)}


def scrape_games(session, year):
    box_path = C.RAW / f"mlb30_boxscore_batters_{year}.csv"
    pbp_path = C.RAW / f"mlb30_pbp_{year}.csv"
    games = season_games(session, year)
    abbr = team_abbr(session, year)
    pks = sorted(games)
    done = done_ids(box_path) & done_ids(pbp_path)
    todo = [pk for pk in pks if pk not in done]
    print(f"[{year}] 경기 {len(pks)}, 기수집 {len(done)}, 남은 {len(todo)}", flush=True)
    # 한쪽 파일에만 반쯤 써진 경기가 있으면 중복될 수 있어, 둘 다 있는 경기만 done으로 보고
    # 나머지는 이후 로드 단계에서 (gameId, 키) 중복 제거한다.
    new_b, new_p = not box_path.exists(), not pbp_path.exists()
    with box_path.open("a", encoding="utf-8-sig", newline="") as fb, \
         pbp_path.open("a", encoding="utf-8-sig", newline="") as fp:
        wb, wp = csv.DictWriter(fb, BOX_FIELDS), csv.DictWriter(fp, PBP_FIELDS)
        if new_b:
            wb.writeheader()
        if new_p:
            wp.writeheader()
        for i, pk in enumerate(todo, 1):
            try:
                data = fetch_light(session, pk, games[pk], abbr)
                box, pbp = parse_game(data) if data else ([], [])
            except Exception as e:  # 재실행 시 재시도
                print(f"  ! {pk}: {e}", flush=True)
                time.sleep(5)
                continue
            wp.writerows(pbp)  # pbp 먼저, box 나중: box에 있으면 pbp도 있다
            wb.writerows(box)
            if i % 100 == 0:
                fb.flush(), fp.flush()
                print(f"  [{year}] {i}/{len(todo)}", flush=True)
            time.sleep(DELAY)


def scrape_players(session, year):
    out = C.RAW / f"mlb30_players_hitting_all_{year}.csv"
    if out.exists():
        print(f"[{year}] players 이미 있음", flush=True)
        return
    teams = get(session, f"https://statsapi.mlb.com/api/v1/teams?sportId=1&season={year}")["teams"]
    rows = []
    for t in teams:
        d = get(session, "https://statsapi.mlb.com/api/v1/stats?stats=season&group=hitting"
                         f"&season={year}&sportId=1&gameType=R&playerPool=all&teamId={t['id']}")
        for s in d.get("stats", [{}])[0].get("splits", []):
            st = s.get("stat", {})
            rows.append({"year": year, "playerId": s["player"]["id"], "name": s["player"]["fullName"],
                         "team": t["abbreviation"], **{k: st.get(k) for k in STAT_KEYS}})
        time.sleep(DELAY)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, ["year", "playerId", "name", "team"] + STAT_KEYS)
        w.writeheader()
        w.writerows(rows)
    print(f"[{year}] players {len(rows)}행 ({len(teams)}팀)", flush=True)


def main(years):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    s = requests.Session()
    for y in years:
        scrape_players(s, y)
    for y in years:
        scrape_games(s, y)
    print("완료", flush=True)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or C.SEASONS)
