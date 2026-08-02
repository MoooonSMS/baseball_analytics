"""KBO 경기의 Naver relay API에서 매 플레이의 베이스-아웃 상태 전이 원본 이벤트를 수집한다.

  https://api-gw.sports.naver.com/schedule/games/{gameId}/relay?inning={n}

페이지네이션은 `no=`/`startNo=` 같은 파라미터가 아니라 **`inning=`**로 이루어진다(실측
확인 — no/startNo/size/count 등은 전부 무시되고 항상 최근 6개 플레이만 반환됨).
inning=1..(경기 종료 이닝)을 순회하면 플레이 순번(`no`)이 0부터 끊김·중복 없이 이어진다
(9이닝 경기에서 실측: inning=1..9 합산 시 no 0-95가 정확히 커버됨). 경기 종료 이닝 수는
파라미터 없이 1회 호출한 응답의 top-level `inn` 필드로 얻는다(연장전 대응). 존재하지
않는 inning 값을 넣으면 에러 없이 빈 리스트가 온다.

응답 구조: `result.textRelayData.textRelays[]`는 타석(`no`) 단위 리스트이고, 각 타석은
`textOptions[]` 하위에 투구/도루/타격결과/득점 등 개별 이벤트가 `seqno` 오름차순으로
들어있다. `base1/base2/base3`는 불리언이 아니라 **그 루에 있는 주자의 타순번호**
(0=비어있음)다 — 연속 이벤트를 비교하면 이름 매칭 없이도 어느 타순 주자가 어디로
이동했는지 특정할 수 있다. `type` 코드는 실측으로 다음을 확인:
  0=이닝시작, 1=투구(볼/스트라이크/파울/타격), 2=투수교체, 7=기타(비디오판독 등),
  8=타자소개, 13=타자 아웃/볼넷으로 타석종료, 23=안타/몸에맞는공으로 타석종료,
  14=주자 개별 진루·도루·도루실패, 24=주자 득점, 99=구분선/경기결과 요약.

원본을 가공 없이 그대로 저장하고, base-out 상태 전이 집계는 Phase 2
(build_situational_probs.py)에서 한다.

gameId 목록은 scrape_naver_boxscores.py가 이미 모아둔 kbo_boxscore_batters_{year}.csv의
고유 gameId를 재사용한다(일정을 다시 긁지 않음). 중단 후 재실행하면 이미 저장된
gameId는 건너뛴다(연도별 CSV에 append).

출력: data/raw/kbo_relay_events_{year}.csv
"""
import csv
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api-gw.sports.naver.com/schedule/games"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
YEARS = [2021, 2022, 2023, 2024, 2025]

FIELDS = ["gameId", "no", "seqno", "inn", "homeOrAway", "type", "text",
          "base1", "base2", "base3", "out"]


def game_ids_for_year(year: int) -> list[str]:
    df = pd.read_csv(DATA_DIR / f"kbo_boxscore_batters_{year}.csv", encoding="utf-8-sig")
    return sorted(df["gameId"].astype(str).unique())


def _get(session: requests.Session, url: str) -> dict:
    for attempt in range(4):
        try:
            r = session.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            r.encoding = "utf-8"  # 자동감지가 실패해 한글이 깨지는 경우가 있어 명시
            return r.json()
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))


def fetch_game_events(game_id: str, session: requests.Session) -> list[dict]:
    base_url = f"{BASE}/{game_id}/relay"
    data = _get(session, base_url)
    trd = (data.get("result") or {}).get("textRelayData") or {}
    final_inn = trd.get("inn") or 0
    if not final_inn:
        return []

    rows = []
    seen_no = set()
    for inn in range(1, final_inn + 1):
        data = _get(session, base_url + f"?inning={inn}")
        trd = (data.get("result") or {}).get("textRelayData") or {}
        for play in trd.get("textRelays") or []:
            no = play.get("no")
            if no in seen_no:
                continue
            seen_no.add(no)
            for opt in play.get("textOptions") or []:
                gs = opt.get("currentGameState") or {}
                rows.append({
                    "gameId": game_id, "no": no, "seqno": opt.get("seqno"),
                    "inn": play.get("inn"), "homeOrAway": play.get("homeOrAway"),
                    "type": opt.get("type"), "text": opt.get("text"),
                    "base1": gs.get("base1"), "base2": gs.get("base2"),
                    "base3": gs.get("base3"), "out": gs.get("out"),
                })
    return rows


def done_game_ids(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {row["gameId"] for row in csv.DictReader(f)}


def scrape_year(year: int):
    out_path = DATA_DIR / f"kbo_relay_events_{year}.csv"
    session = requests.Session()
    game_ids = game_ids_for_year(year)
    done = done_game_ids(out_path)
    todo = [g for g in game_ids if g not in done]
    print(f"[{year}] total={len(game_ids)} done={len(done)} todo={len(todo)}", flush=True)

    write_header = not out_path.exists()
    with out_path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for i, gid in enumerate(todo, 1):
            try:
                rows = fetch_game_events(gid, session)
            except Exception as e:  # 일시 오류는 건너뛰고 재실행 시 재시도
                print(f"  ! {gid}: {e}", flush=True)
                time.sleep(2)
                continue
            writer.writerows(rows)
            if i % 50 == 0:
                f.flush()
                print(f"  [{year}] {i}/{len(todo)}", flush=True)
            time.sleep(0.2)
    print(f"[{year}] saved -> {out_path}", flush=True)


def main(years=YEARS):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for year in years:
        scrape_year(year)


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or YEARS
    main(years)
