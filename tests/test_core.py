"""핵심 로직 회귀 테스트: 중계 문자 분류기(classify_outcome)와 라인업 시뮬레이터(simulate_lineup).

실행: python tests/test_core.py   (pytest로도 수집됨)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "leadoff"))
import lineup_sim as ls  # noqa: E402
from build_situational_probs import classify_outcome  # noqa: E402

# 2024 relay 원문에서 그대로 가져온 예시
CASES = [
    ("신민재 : 좌익수 오른쪽 1루타", "1B"),
    ("페라자 : 2루수 왼쪽 내야안타", "1B"),
    ("하주석 : 우익수 오른쪽 2루타", "2B"),
    ("박준영 : 좌중간 3루타", "3B"),
    ("한유섬 : 우익수 뒤 홈런 (홈런거리:115M)", "HR"),
    ("오지환 : 볼넷", "BB"),
    ("로하스 : 자동 고의4구", "IBB"),
    ("최재훈 : 몸에 맞는 볼", "HBP"),
    ("노시환 : 삼진 아웃", "K"),
    ("전준우 : 포수 스트라이크 낫 아웃 (포수 태그아웃)", "K"),
    ("황재균 : 스트라이크 낫아웃 폭투", "K"),
    ("김강민 : 유격수 병살타 아웃 (유격수->2루수->1루수 송구아웃)", "GIDP"),
    ("윤동희 : 3루수 병살타로 출루", "FC"),
    ("정수빈 : 1루수 희생번트 아웃 (1루수->2루수 1루 터치아웃)", "SAC_BUNT"),
    ("서건창 : 3루수 번트 아웃 (3루수->1루수 송구아웃)", "OUT"),
    ("오태곤 : 포수 파울플라이 아웃 (번트)", "OUT"),
    ("박해민 : 중견수 희생플라이 아웃", "SAC_FLY"),
    ("신민재 : 2루수 땅볼 실책으로 출루 (2루수 포구 실책)", "ERROR"),
    ("문현빈 : 3루수 앞 땅볼로 출루", "FC"),
    ("김현수 : 좌익수 플라이 아웃", "OUT"),
]

BB, S1, S2, S3, HR, OUT = np.eye(6)


def test_classify_outcome():
    bad = [(t, classify_outcome(t), want) for t, want in CASES if classify_outcome(t) != want]
    assert not bad, bad


def test_home_run_then_three_outs():
    lineup = np.array([HR] + [OUT] * 8)
    r = ls.simulate_lineup(lineup, n_games=5, innings=1)
    assert r["runs_per_game"] == 1
    assert r["rbi_per_game"][0] == 1 and r["runs_by_slot"][0] == 1


def test_bases_loaded_walk_forces_in_run():
    lineup = np.array([BB] * 4 + [OUT] * 5)
    r = ls.simulate_lineup(lineup, n_games=5, innings=1)
    assert r["runs_per_game"] == 1
    assert r["rbi_per_game"][3] == 1   # 밀어내기 타점은 4번 타자
    assert r["runs_by_slot"][0] == 1   # 득점은 3루에 있던 1번 타자


def test_run_accounting_balances():
    """모든 득점은 정확히 한 명의 득점이자 한 명의 타점(도루·병살·주루사로 득점이 새거나 생기면 안 됨)."""
    rng = np.random.default_rng(0)
    lineup = rng.dirichlet([2, 4, 1, 0.2, 0.6, 14], size=9)
    for sp in (None, ls.load_situational_probs()):
        r = ls.simulate_lineup(lineup, n_games=2000, seed=3, situational_probs=sp)
        assert np.isclose(r["runs_by_slot"].sum(), r["runs_per_game"])
        assert np.isclose(r["rbi_per_game"].sum(), r["runs_per_game"])


def test_all_outs_score_nothing():
    r = ls.simulate_lineup(np.array([OUT] * 9), n_games=10)
    assert r["runs_per_game"] == 0


if __name__ == "__main__":
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
