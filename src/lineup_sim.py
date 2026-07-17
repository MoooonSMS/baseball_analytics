"""타순(라인업) 몬테카를로 시뮬레이터.

야구 타순 최적화 질문("이 타자를 몇 번에 두는 게 팀 득점에 유리한가")을
base-out 상태 몬테카를로로 근사한다. The Book(Tango et al.)의 라인업 분석과
같은 계열의 단순화 모델이다.

이벤트 모델(타석당): {BB+HBP, 1B, 2B, 3B, HR, out} — 선수 시즌 비율에서 산출.
주루 진루 규칙(결정론적 basic 모델):
  BB/HBP : 강제 진루만 (밀어내기)
  1B     : 타자→1루, 1루주자→2루, 2·3루주자 득점
  2B     : 타자→2루, 1루주자→3루, 2·3루주자 득점
  3B     : 타자→3루, 모든 주자 득점
  HR     : 타자 포함 전원 득점
  out    : 진루 없음

한계(명시): 도루/도루자, 병살, 희생타, 실책, 태그업, 투수교체, 상황별 타격을
반영하지 않는다. 절대 득점 수준보다 '슬롯 간 상대 비교'에 쓰는 것이 목적이며,
캘리브레이션으로 실제 리그 R/G에 맞춘 뒤 해석한다.
"""
import numpy as np

EVENTS = ["bb", "1b", "2b", "3b", "hr", "out"]
# 리그 평균 보정용(풀슬래시 없을 때): 비홈런 안타 중 2B/3B 비율, 타석당 HBP율 (KBO 2024 근사)
LG_2B_SHARE = 0.202
LG_3B_SHARE = 0.017
LG_HBP_RATE = 0.011

# 추가 주루(리그 관측 근사) — 캘리브레이션용. 결정론 basic 모델을 실제 R/G에 근접시킨다.
P_1B_1ST_TO_3RD = 0.28    # 단타 때 1루주자가 3루까지 (아니면 2루)
P_2B_1ST_SCORES = 0.45    # 2루타 때 1루주자가 홈까지 (아니면 3루)


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


def simulate_lineup(events9: np.ndarray, n_games: int = 50000, innings: int = 9,
                    seed: int = 0) -> dict:
    """9인 라인업(events9: shape (9,6))으로 n_games 시뮬. 팀 R/G와 타자별 타점/득점 반환.

    타점(rbi): 각 타자의 타석에서 홈인한 주자 수(자기 홈런 포함)의 경기당 평균.
    득점(runs): 각 타자 본인이 홈인한 횟수의 경기당 평균.
    """
    rng = np.random.default_rng(seed)
    cum = np.cumsum(events9, axis=1)  # (9,6) 누적확률

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
                r = rng.random()
                ev = np.searchsorted(cum[b], r)  # 0..5

                if ev == 5:  # out
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

                    if ev == 1:      # 1B: 2·3루 득점, 1루→2루(또는 3루), 타자→1루
                        if r3 != -1:
                            runs[r3] += 1; scored += 1
                        if r2 != -1:
                            runs[r2] += 1; scored += 1
                        if r1 != -1:
                            if rng.random() < P_1B_1ST_TO_3RD:
                                new_bases[2] = r1
                            else:
                                new_bases[1] = r1
                        new_bases[0] = b
                    elif ev == 2:    # 2B: 2·3루 득점, 1루→홈(또는 3루), 타자→2루
                        if r3 != -1:
                            runs[r3] += 1; scored += 1
                        if r2 != -1:
                            runs[r2] += 1; scored += 1
                        if r1 != -1:
                            if rng.random() < P_2B_1ST_SCORES:
                                runs[r1] += 1; scored += 1
                            else:
                                new_bases[2] = r1
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
