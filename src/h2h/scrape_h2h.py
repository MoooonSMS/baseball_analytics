"""KBO 공식기록실 팀순위 페이지에서 '상대전적표'(팀 간 시즌 승-패-무 매트릭스)를 스크래핑한다.

선수 기본기록 페이지(HitterBasic 등)와 달리 이 페이지는 연도 선택 컨트롤이
ddlSeason이 아니라 ddlYear라는 이름을 쓴다. postback 방식 자체는 동일.
"""
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = "https://www.koreabaseball.com/Record/TeamRank/TeamRank.aspx"
YEAR_CTRL = "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlYear"

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


def _hidden_fields(soup: BeautifulSoup) -> dict:
    fields = {}
    for inp in soup.find_all("input", type="hidden"):
        name = inp.get("name")
        if name:
            fields[name] = inp.get("value", "")
    return fields


def scrape_h2h_year(year: int) -> pd.DataFrame:
    session = requests.Session()
    resp = session.get(URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    form = _hidden_fields(soup)
    form["__EVENTTARGET"] = YEAR_CTRL
    form["__EVENTARGUMENT"] = ""
    form[YEAR_CTRL] = str(year)
    resp = session.post(URL, data=form, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    table = soup.find_all("table")[1]  # [0]=순위표, [1]=상대전적표
    headers = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]
    opponent_cols = [h.split("(")[0] for h in headers[1:-1]]  # '합계' 제외, 팀명만

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        team = cells[0]
        for opponent, record in zip(opponent_cols, cells[1:-1]):
            if record == "■":  # 자기 자신과의 대결란
                continue
            win, loss, draw = (int(x) for x in record.split("-"))
            rows.append(
                {
                    "year": year,
                    "team": team,
                    "opponent": opponent,
                    "win": win,
                    "loss": loss,
                    "draw": draw,
                }
            )
    return pd.DataFrame(rows)


def main(years: list[int]):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for year in years:
        df = scrape_h2h_year(year)
        out_path = DATA_DIR / f"team_h2h_{year}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"saved {out_path} ({len(df)} rows)")
        time.sleep(0.5)


if __name__ == "__main__":
    main(years=[2024])
