"""wOBA / wRC+ 계산 유틸.

스탯티즈 연도별 상수(statiz_woba_constants.csv)의 선형가중치를 사용해
KBO 공식기록실 시즌 기록으로 wOBA와 wRC+(파크팩터 미반영)를 계산한다.

주의: 파크팩터를 반영하지 않으므로 스탯티즈 공식 wRC+와 약간 다르다.
잠실 홈 타자는 과소평가, 타자친화 구장 홈 타자는 과대평가될 수 있다.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

_CONST = None
_MLB_CONST = None


def constants() -> pd.DataFrame:
    global _CONST
    if _CONST is None:
        _CONST = pd.read_csv(DATA_DIR / "raw" / "statiz_woba_constants.csv").set_index("Year")
    return _CONST


def mlb_constants() -> pd.DataFrame:
    """FanGraphs Guts! 페이지의 연도별 wOBA 선형가중치(wBB/wHBP/w1B/w2B/w3B/wHR)."""
    global _MLB_CONST
    if _MLB_CONST is None:
        _MLB_CONST = pd.read_csv(DATA_DIR / "raw" / "mlb_woba_constants.csv").set_index("season")
    return _MLB_CONST


def add_woba(df: pd.DataFrame, year_col="year", bb="bb", ibb=None, hbp="hbp",
             h="hits", d2="doubles", d3="triples", hr="hr", ab="ab", sf=None) -> pd.Series:
    """카운팅 스탯 컬럼으로 wOBA를 계산해 Series로 반환.

    IBB(고의사구)와 SF 컬럼이 있으면 반영, 없으면 0으로 둔다.
    분모는 AB + (BB-IBB) + HBP + SF (statiz ePA 방식에서 SH 제외와 동일).
    """
    c = constants()
    w = df[year_col].map(c["eBB"])
    w1, w2, w3, w4 = (df[year_col].map(c[k]) for k in ["1B", "2B", "3B", "HR"])
    ibb_v = df[ibb] if ibb and ibb in df.columns else 0
    sf_v = df[sf] if sf and sf in df.columns else 0
    ubb = df[bb] - ibb_v
    singles = df[h] - df[d2] - df[d3] - df[hr]
    num = w * (ubb + df[hbp]) + w1 * singles + w2 * df[d2] + w3 * df[d3] + w4 * df[hr]
    den = df[ab] + ubb + df[hbp] + sf_v
    return num / den.replace(0, pd.NA)


def add_mlb_woba(df: pd.DataFrame, year_col="season", bb="baseOnBalls", ibb="intentionalWalks",
                  hbp="hitByPitch", h="hits", d2="doubles", d3="triples", hr="homeRuns",
                  ab="atBats", sf="sacFlies") -> pd.Series:
    """FanGraphs 표준 wOBA 공식 (wBB/wHBP 가중치를 분리해서 반영).

    분모는 AB + (BB-IBB) + HBP + SF (statiz add_woba와 같은 ePA 정의).
    """
    c = mlb_constants()
    wbb, whbp = df[year_col].map(c["wBB"]), df[year_col].map(c["wHBP"])
    w1, w2, w3, w4 = (df[year_col].map(c[k]) for k in ["w1B", "w2B", "w3B", "wHR"])
    ubb = df[bb] - df[ibb]
    singles = df[h] - df[d2] - df[d3] - df[hr]
    num = wbb * ubb + whbp * df[hbp] + w1 * singles + w2 * df[d2] + w3 * df[d3] + w4 * df[hr]
    den = df[ab] + ubb + df[hbp] + df[sf]
    return num / den.replace(0, pd.NA)


def add_wrc_plus(df: pd.DataFrame, woba_col="woba", year_col="year",
                 lg: pd.DataFrame | None = None) -> pd.Series:
    """wRC+ = ((wOBA - lgwOBA)/scale + lgR/PA) / (lgR/PA) * 100  (파크팩터 없음)

    lg: year별 lgRpPA(리그 득점/타석)를 담은 DataFrame(index=year, col='rppa').
        None이면 statiz 상수의 lg wOBA만 쓰고 R/PA는 df에서 추정 불가하므로 필수.
    """
    c = constants()
    lg_woba = df[year_col].map(c["wOBA"])
    scale = df[year_col].map(c["Scale"])
    rppa = df[year_col].map(lg["rppa"])
    return ((df[woba_col] - lg_woba) / scale + rppa) / rppa * 100
