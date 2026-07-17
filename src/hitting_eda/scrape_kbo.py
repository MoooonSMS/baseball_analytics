"""KBO 공식기록실(koreabaseball.com)에서 타자/투수 기본기록을 스크래핑한다.

이 사이트는 ASP.NET WebForms 기반이라 연도 변경이나 페이지 이동이
일반 GET 파라미터가 아니라 __doPostBack() 자바스크립트 postback으로 처리된다.
따라서 매 요청마다 __VIEWSTATE 등 hidden 필드를 그대로 들고 다니면서
POST로 폼을 다시 제출하는 방식으로 흉내낸다.
"""
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.koreabaseball.com"
# 타자는 두 탭을 합쳐야 OPS 계열 지표까지 얻을 수 있다.
#   Basic1: AVG, G, PA, AB, R, H, 2B, 3B, HR, TB, RBI, SAC, SF
#   Basic2: BB, IBB, HBP, SO, GDP, SLG, OBP, OPS, MH, RISP, PH-BA
HITTER_URLS = [
    f"{BASE_URL}/Record/Player/HitterBasic/Basic1.aspx",
    f"{BASE_URL}/Record/Player/HitterBasic/Basic2.aspx",
]
# 투수는 Basic1에 ERA/WHIP까지 이미 포함되어 있어 한 탭이면 충분.
PITCHER_URL = f"{BASE_URL}/Record/Player/PitcherBasic/Basic1.aspx"

SEASON_CTRL = "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlSeason$ddlSeason"
PAGE_BTN_PREFIX = "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ucPager$btnNo"

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


def _hidden_fields(soup: BeautifulSoup) -> dict:
    """페이지의 모든 hidden input 값을 dict로 뽑아낸다 (__VIEWSTATE 등)."""
    fields = {}
    for inp in soup.find_all("input", type="hidden"):
        name = inp.get("name")
        if name:
            fields[name] = inp.get("value", "")
    return fields


def _parse_table(soup: BeautifulSoup) -> pd.DataFrame:
    table = soup.find("table", class_="tData01")
    headers = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) == len(headers):
            rows.append(cells)
    return pd.DataFrame(rows, columns=headers)


def _page_links(soup: BeautifulSoup) -> dict[int, str]:
    """paging 영역에서 {페이지번호: postback 대상 이름} 을 찾는다.

    btnFirst/btnLast 같은 화살표 링크는 텍스트가 숫자가 아니라 자동으로 제외된다.
    """
    paging = soup.find("div", class_="paging")
    if not paging:
        return {}
    links = {}
    for a in paging.find_all("a"):
        text = a.get_text(strip=True)
        if not text.isdigit():
            continue
        onclick = a.get("href", "")
        if "btnNo" not in onclick:
            continue
        start = onclick.find("__doPostBack('") + len("__doPostBack('")
        end = onclick.find("'", start)
        links[int(text)] = onclick[start:end]
    return links


def scrape_year(session: requests.Session, url: str, year: int) -> pd.DataFrame:
    resp = session.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # 1) 연도 선택 postback
    form = _hidden_fields(soup)
    form["__EVENTTARGET"] = SEASON_CTRL
    form["__EVENTARGUMENT"] = ""
    form[SEASON_CTRL] = str(year)
    resp = session.post(url, data=form, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    all_pages = [_parse_table(soup)]

    # 2) 아직 방문하지 않은 다음 페이지 번호를 순서대로 postback
    visited_pages = {1}
    while True:
        links = _page_links(soup)
        remaining = sorted(p for p in links if p not in visited_pages)
        if not remaining:
            break
        next_page = remaining[0]
        visited_pages.add(next_page)
        form = _hidden_fields(soup)
        form["__EVENTTARGET"] = links[next_page]
        form["__EVENTARGUMENT"] = ""
        resp = session.post(url, data=form, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        all_pages.append(_parse_table(soup))
        time.sleep(0.5)

    df = pd.concat(all_pages, ignore_index=True)
    df["year"] = year
    return df


def scrape_hitters_year(year: int) -> pd.DataFrame:
    """Basic1 + Basic2 탭을 선수명+팀명 기준으로 합친다."""
    tabs = []
    for url in HITTER_URLS:
        session = requests.Session()
        tabs.append(scrape_year(session, url, year))
        time.sleep(0.5)

    df = tabs[0]
    for other in tabs[1:]:
        shared = [c for c in ("순위", "AVG") if c in other.columns]
        other = other.drop(columns=shared, errors="ignore")
        df = df.merge(other, on=["선수명", "팀명", "year"], how="left")
    return df


def main(years: list[int]):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for year in years:
        df = scrape_hitters_year(year)
        out_path = DATA_DIR / f"hitters_{year}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"saved {out_path} ({len(df)} rows)")

    for year in years:
        session = requests.Session()
        df = scrape_year(session, PITCHER_URL, year)
        out_path = DATA_DIR / f"pitchers_{year}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"saved {out_path} ({len(df)} rows)")
        time.sleep(0.5)


if __name__ == "__main__":
    main(years=[2021, 2022, 2023, 2024, 2025])
