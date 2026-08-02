"""타순(라인업) 몬테카를로 시뮬레이터.

야구 타순 최적화 질문("이 타자를 몇 번에 두는 게 팀 득점에 유리한가")을
base-out 상태 몬테카를로로 근사한다. The Book(Tango et al.)의 라인업 분석과
같은 계열의 단순화 모델이다.

이벤트 모델(타석당): {BB+HBP, 1B, 2B, 3B, HR, out} — 선수 시즌 비율에서 산출.

주루 진루 규칙에는 두 가지 모드가 있다:
  - **기본(situational_probs=None)**: 고정 확률 상수(P_1B_1ST_TO_3RD, P_2B_1ST_SCORES)
    기반 결정론적 basic 모델. 병살·도루 없음. 기존 동작 그대로이며, 이 인자를 넘기지
    않는 모든 호출(MLB 등 KBO 외 리그 포함)은 영향받지 않는다.
  - **situational_probs를 넘기면(KBO 전용)**: build_situational_probs.py가 KBO 5개
    시즌 실측 relay 데이터로 뽑은 조건부 확률로 1루주자 진루(단타/2루타 시, 아웃카운트별
    층화)를 결정하고, 병살(1루주자+2아웃 미만에서 out 발생 시)과 도루(1루주자+2루
    비어있을 때 매 타석 전 시도)를 추가로 시뮬레이션한다. load_situational_probs()로
    `data/processed/kbo_situational_probs.json`을 로드해서 넘긴다.

두 모드 공통 규칙:
  BB/HBP : 강제 진루만 (밀어내기)
  1B     : 타자→1루, 2·3루주자 득점, 1루주자는 모드에 따라 결정
  2B     : 타자→2루, 2·3루주자 득점, 1루주자는 모드에 따라 결정
  3B     : 타자→3루, 모든 주자 득점
  HR     : 타자 포함 전원 득점
  out    : 진루 없음 (situational_probs 모드에서는 병살 가능)

한계(situational_probs 모드에서도 여전히 남음): 희생타·실책·태그업·투수교체·
2루주자 도루(3루 시도)·선수별 GIDP/SB 성향(리그 평균만 사용)을 반영하지 않는다.
절대 득점 수준보다 '슬롯 간 상대 비교'에 쓰는 것이 목적이며, 캘리브레이션으로
실제 리그 R/G에 맞춘 뒤 해석한다.
"""
import json
from pathlib import Path

import numpy as np

EVENTS = ["bb", "1b", "2b", "3b", "hr", "out"]
# 리그 평균 보정용(풀슬래시 없을 때): 비홈런 안타 중 2B/3B 비율, 타석당 HBP율 (KBO 2024 근사)
LG_2B_SHARE = 0.202
LG_3B_SHARE = 0.017
LG_HBP_RATE = 0.011

# 추가 주루(리그 관측 근사) — situational_probs=None일 때만 쓰는 캘리브레이션용 폴백.
P_1B_1ST_TO_3RD = 0.28    # 단타 때 1루주자가 3루까지 (아니면 2루)
P_2B_1ST_SCORES = 0.45    # 2루타 때 1루주자가 홈까지 (아니면 3루)

_PROBS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "kbo_situational_probs.json"
_probs_cache = None


def load_situational_probs(path=None) -> dict:
    """build_situational_probs.py 산출물(KBO 실측 상황조건부 확률표)을 로드한다."""
    global _probs_cache
    if path is None and _probs_cache is not None:
        return _probs_cache
    with open(path or _PROBS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if path is None:
        _probs_cache = data
    return data


def events_from_full(pa, ab, h, d2, d3, hr, bb, hbp, sf=0) -> np.ndarray:
    """풀슬래시(2B/3B 포함)로 타석당 이벤트 확률 벡터 [bb,1b,2b,3b,hr,out]."""
    singles = h - d2 - d3 - hr
    reached = (bb + hbp) + singles + d2 + d3 + hr
    outs = pa - reached
    counts = np.array([bb + hbp, singles, d2, d3, hr, max(outs, 0)], dtype=float)
    return counts / counts.sum()


def events_from_boxscore(pa_est, ab, h, hr, bb) -> np.ndarray:
    """2B/3B가 없는 boxscore 집계용: 리그 평균 분할로 보정."""
    nonhr_hits = h - hr
    d2 = nonhr_hits * LG_2B_SHARE
    d3 = nonhr_hits * LG_3B_SHARE
    singles = nonhr_hits - d2 - d3
    hbp = pa_est * LG_HBP_RATE
    bb_hbp = bb + hbp
    reached = bb_hbp + singles + d2 + d3 + hr
    outs = pa_est - reached
    counts = np.array([bb_hbp, singles, d2, d3, hr, max(outs, 0)], dtype=float)
    return counts / counts.sum()


_FATE_KEYS = ("2nd", "3rd", "scored", "out")


def _fate_weights(table: dict, outs: int) -> np.ndarray:
    """runner_on_1st_after_1B/2B 테이블에서 out별 분포 조회, 표본부족시 overall로 백오프.

    '1st'(주자가 원래 자리에 그대로 남는 경우)는 배터가 그 루를 차지하므로 물리적으로
    불가능 — 실측 데이터의 잡음(약 0.1% 미만, 드문 기록 예외)이라 제외하고 나머지
    4개 결과로 재정규화한다.
    """
    by_out = table.get("by_out_before", {})
    d = by_out.get(str(outs)) or table["overall"]
    w = np.array([d.get(k, 0.0) for k in _FATE_KEYS], dtype=float)
    return w / w.sum()


def _draw_fate(rng: np.random.Generator, weights: np.ndarray) -> str:
    return _FATE_KEYS[np.searchsorted(np.cumsum(weights), rng.random())]


def simulate_lineup(events9: np.ndarray, n_games: int = 50000, innings: int = 9,
                    seed: int = 0, situational_probs: dict | None = None) -> dict:
    """9인 라인업(events9: shape (9,6))으로 n_games 시뮬. 팀 R/G와 타자별 타점/득점 반환.

    situational_probs: load_situational_probs()로 로드한 KBO 실측 확률표. None(기본값)
    이면 고정 확률 기반 단순 모델(병살·도루 없음)로 동작한다.

    타점(rbi): 각 타자의 타석에서 홈인한 주자 수(자기 홈런 포함)의 경기당 평균.
    득점(runs): 각 타자 본인이 홈인한 횟수의 경기당 평균.
    """
    rng = np.random.default_rng(seed)
    cum = np.cumsum(events9, axis=1)  # (9,6) 누적확률

    sp = situational_probs
    gidp_rate = sp["gidp"]["rate"] if sp else 0.0
    steal_attempt_rate = sp["steal"]["attempt_rate_per_pa_on_base"] if sp else 0.0
    steal_success_rate = sp["steal"]["success_rate"] if sp else 0.0
    adv_1b_on_1b = sp["runner_on_1st_after_1B"] if sp else None
    adv_1b_on_2b = sp["runner_on_1st_after_2B"] if sp else None

    team_runs = 0
    rbi = np.zeros(9)
    runs = np.zeros(9)

    for _ in range(n_games):
        batter = 0
        for _inning in range(innings):
            outs = 0
            # 베이스: 각 루의 주자 '타순 인덱스'(-1=비어있음)
            bases = [-1, -1, -1]  # [1루, 2루, 3루]
            while outs < 3:
                b = batter % 9

                if sp and bases[0] != -1 and bases[1] == -1 and rng.random() < steal_attempt_rate:
                    if rng.random() < steal_success_rate:
                        bases[1] = bases[0]
                        bases[0] = -1
                    else:
                        bases[0] = -1
                        outs += 1
                        continue  # 도루실패 3아웃이면 이 타자는 타석에 서지 못함

                r = rng.random()
                ev = np.searchsorted(cum[b], r)  # 0..5

                if ev == 5:  # out
                    if sp and bases[0] != -1 and outs < 2 and rng.random() < gidp_rate:
                        outs += 2  # 병살: 타자 + 1루주자
                        bases[0] = -1
                    else:
                        outs += 1
                elif ev == 0:  # bb/hbp (강제 진루)
                    if bases[0] != -1:
                        if bases[1] != -1:
                            if bases[2] != -1:  # 만루 밀어내기 득점
                                scorer = bases[2]
                                runs[scorer] += 1
                                rbi[b] += 1
                                team_runs += 1
                            bases[2] = bases[1]
                        bases[1] = bases[0]
                    bases[0] = b
                else:
                    # 안타류: 각 주자별 새 위치를 명시적으로. scored=이 타석 홈인 수.
                    r1, r2, r3 = bases  # 기존 주자
                    new_bases = [-1, -1, -1]
                    scored = 0

                    if ev == 1:      # 1B: 2·3루 득점, 1루주자는 확률적, 타자→1루
                        if r3 != -1:
                            runs[r3] += 1; scored += 1
                        if r2 != -1:
                            runs[r2] += 1; scored += 1
                        if r1 != -1:
                            if sp:
                                fate = _draw_fate(rng, _fate_weights(adv_1b_on_1b, outs))
                            else:
                                fate = "3rd" if rng.random() < P_1B_1ST_TO_3RD else "2nd"
                            if fate == "2nd":
                                new_bases[1] = r1
                            elif fate == "3rd":
                                new_bases[2] = r1
                            elif fate == "scored":
                                runs[r1] += 1; scored += 1
                            elif fate == "out":
                                outs += 1
                        new_bases[0] = b
                    elif ev == 2:    # 2B: 2·3루 득점, 1루주자는 확률적, 타자→2루
                        if r3 != -1:
                            runs[r3] += 1; scored += 1
                        if r2 != -1:
                            runs[r2] += 1; scored += 1
                        if r1 != -1:
                            if sp:
                                fate = _draw_fate(rng, _fate_weights(adv_1b_on_2b, outs))
                            else:
                                fate = "scored" if rng.random() < P_2B_1ST_SCORES else "3rd"
                            if fate == "3rd":
                                new_bases[2] = r1
                            elif fate == "scored":
                                runs[r1] += 1; scored += 1
                            elif fate == "out":
                                outs += 1
                        new_bases[1] = b
                    elif ev == 3:    # 3B: 전 주자 득점, 타자→3루
                        for rr in (r1, r2, r3):
                            if rr != -1:
                                runs[rr] += 1; scored += 1
                        new_bases[2] = b
                    else:            # 4 == HR: 전 주자 + 타자 득점
                        for rr in (r1, r2, r3):
                            if rr != -1:
                                runs[rr] += 1; scored += 1
                        runs[b] += 1; scored += 1

                    rbi[b] += scored
                    team_runs += scored
                    bases = new_bases

                batter += 1

    return {
        "runs_per_game": team_runs / n_games,
        "rbi_per_game": rbi / n_games,   # 길이 9, 슬롯별
        "runs_by_slot": runs / n_games,  # 길이 9, 슬롯별
    }
