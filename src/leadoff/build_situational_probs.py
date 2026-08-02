"""KBO relay 이벤트(scrape_kbo_relay.py)에서 상황조건부 확률표를 추출한다.

`lineup_sim.py`의 고정 상수(P_1B_1ST_TO_3RD=0.28, P_2B_1ST_SCORES=0.45)를 실제 KBO
데이터 기반 조건부 확률로 교체하고, 병살·도루 확률을 추가하기 위한 입력을 만든다.

## 타석(PA) 단위 그룹핑
relay 이벤트는 (gameId, no)가 타석 1개다. `no` 그룹 안에는 배터 본인의 결과 행
(type=13 아웃/볼넷, type=23 안타/몸에맞는공)뿐 아니라, 그 타석 도중 발생한 도루·도루실패
(type=14, 배터가 타격하기 *전*)나 안타 이후의 추가 포스아웃(type=14, 배터 결과 행 *다음*)도
같이 들어있다(실측 확인 — 예: 땅볼로 출루한 배터의 결과 행은 아직 3루주자 포스아웃을
반영하지 못한 상태이고, 그 다음 행에서 out이 +1 된다). 그래서:
  - **before_state** = 이 타석 시작 시점 상태 = 직전 no 그룹의 **마지막 행** 상태
    (경기 첫 타석은 이닝시작 표시 행 자체가 이미 0아웃·주자없음이라 별도 처리 불필요)
  - **after_state** = 이 타석의 **마지막 행**(seqno 최댓값) 상태 — 도루/포스아웃 등
    타석 도중·직후에 벌어진 모든 후속 이벤트까지 반영된 최종 상태
  - **outcome_type** = 그 타석 안의 type∈{13,23} 행 텍스트로 분류 (배터 자신의 결과)

base1/base2/base3는 불리언이 아니라 그 루에 있는 주자의 **타순번호**(0=비어있음)다.
이 성질을 이용해, before_state에서 특정 루에 있던 주자(타순번호)가 after_state의
어느 루에 나타나는지 추적하면 "그 주자가 실제로 어디까지 갔는지"를 이름 매칭 없이
정확히 알 수 있다(사라졌는데 아웃카운트가 늘었으면 아웃, 안 늘었으면 득점).

## 파싱 함정 (스크래핑 후 검증 단계에서 실측으로 확인한 것들)
- "내야안타"/"번트안타"는 텍스트에 "1루타"가 없지만 실제 단타다 — 1B로 분류해야
  boxscore 안타 합계와 정확히 일치한다(2024 시즌 검증: 13,996건 일치).
- type=7(비디오 판독 등) 텍스트에도 "홈런" 같은 키워드가 섞여 들어가 안타 유형
  키워드 매칭을 type∈{13,23} 행으로 한정하지 않으면 중복 집계된다.

출력: data/processed/kbo_situational_probs.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
YEARS = [2021, 2022, 2023, 2024, 2025]

MIN_CELL_N = 30  # out별 층화 표본이 이 미만이면 out 무시하고 합산치로 백오프


def classify_outcome(text: str) -> str:
    """배터 결과 행(type 13/23) 텍스트를 타석 결과 카테고리로 분류."""
    if not isinstance(text, str):
        return "UNKNOWN"
    if "병살" in text and "출루" not in text:  # "병살타로 출루"(타자는 세이프, 드묾)는 FC로 취급
        return "GIDP"
    if "희생번트" in text or ("번트" in text and "아웃" in text and "안타" not in text):
        return "SAC_BUNT"
    if "희생플라이" in text:
        return "SAC_FLY"
    if "홈런" in text:
        return "HR"
    if "3루타" in text:
        return "3B"
    if "2루타" in text:
        return "2B"
    if "1루타" in text or "내야안타" in text or "번트안타" in text:
        return "1B"
    if "몸에 맞는 볼" in text:
        return "HBP"
    if "볼넷" in text:
        return "BB"
    if "삼진" in text:
        return "K"
    if "실책" in text and "출루" in text:
        return "ERROR"
    if "출루" in text:  # 필드아웃/포스아웃 등으로 타자만 살아나가는 경우(FC)
        return "FC"
    if "아웃" in text:
        return "OUT"
    return "UNKNOWN"


def load_events(years=YEARS) -> pd.DataFrame:
    dfs = [pd.read_csv(RAW / f"kbo_relay_events_{y}.csv", encoding="utf-8-sig") for y in years]
    df = pd.concat(dfs, ignore_index=True)
    return df.sort_values(["gameId", "seqno"]).reset_index(drop=True)


def build_pa_table(df: pd.DataFrame) -> pd.DataFrame:
    """이벤트 원본 -> 타석(PA) 단위 before/after 상태 + outcome_type 테이블."""
    records = []
    for gid, g in df.groupby("gameId", sort=False):
        g = g.sort_values("seqno")
        prev_state = (0, 0, 0, 0)  # base1, base2, base3, out — 경기 시작은 항상 0
        for no_val, grp in g.groupby("no", sort=False):
            grp = grp.sort_values("seqno")
            end_rows = grp[grp["type"].isin([13, 23])]
            last = grp.iloc[-1]
            after_state = (int(last["base1"]), int(last["base2"]), int(last["base3"]), int(last["out"]))
            if len(end_rows) == 0:
                # 이닝시작 표시행 등 실제 타석이 아닌 그룹 — before만 갱신하고 스킵
                prev_state = after_state
                continue
            outcome = classify_outcome(end_rows.iloc[0]["text"])
            records.append({
                "gameId": gid, "no": no_val, "outcome": outcome,
                "b1_before": prev_state[0], "b2_before": prev_state[1],
                "b3_before": prev_state[2], "out_before": prev_state[3],
                "b1_after": after_state[0], "b2_after": after_state[1],
                "b3_after": after_state[2], "out_after": after_state[3],
            })
            prev_state = after_state
    return pd.DataFrame(records)


def _runner_fate(row, before_col: str) -> str:
    """before_col(예: 'b1_before')에 있던 주자가 after_state 어디로 갔는지 판정."""
    runner = row[before_col]
    if runner == 0:
        return "empty"
    if row["b1_after"] == runner:
        return "1st"
    if row["b2_after"] == runner:
        return "2nd"
    if row["b3_after"] == runner:
        return "3rd"
    # 어느 루에도 없음 -> 득점했거나 아웃됨
    if row["out_after"] > row["out_before"]:
        return "out"
    return "scored"


def runner_advance_table(pa: pd.DataFrame, outcome: str, before_col: str) -> dict:
    sub = pa[(pa["outcome"] == outcome) & (pa[before_col] != 0)].copy()
    if sub.empty:
        return {"n": 0}
    sub["fate"] = sub.apply(lambda r: _runner_fate(r, before_col), axis=1)

    def dist(frame):
        n = len(frame)
        counts = frame["fate"].value_counts()
        return {"n": n, **{k: float(counts.get(k, 0) / n) for k in
                            ["1st", "2nd", "3rd", "scored", "out"]}}

    result = {"overall": dist(sub)}
    by_out = {}
    for out_before, grp in sub.groupby("out_before"):
        if len(grp) >= MIN_CELL_N:
            by_out[str(out_before)] = dist(grp)
    result["by_out_before"] = by_out
    return result


REACHED_OUTCOMES = {"1B", "2B", "3B", "HR", "BB", "HBP"}


def gidp_stats(pa: pd.DataFrame) -> dict:
    """GIDP율 = P(병살 | lineup_sim의 'out' 이벤트가 뽑힘, 1루주자 있음, 2아웃 미만).

    lineup_sim.py의 이벤트 벡터는 [bb,1b,2b,3b,hr,out] 6개뿐이라 'out' 한 버킷에
    삼진·병살·희생번트·필드아웃·(타자 세이프인) FC까지 다 뭉쳐 있다(안타/볼넷/사구가
    아닌 모든 결과가 'out'으로 샘플링됨 — FC가 타자 관점에서 'out'으로 잘못 집계되는
    것도 lineup_sim의 기존 단순화 한계이지 여기서 새로 생기는 문제가 아니다). 따라서
    Phase 3에서 "이미 뽑힌 out을 병살로 업그레이드할지" 결정할 때 쓸 조건부 확률은
    분모를 안타/볼넷/사구를 제외한 전체(=lineup_sim이 'out'으로 볼 모든 결과)로 잡아야
    lineup_sim의 이벤트 정의와 정확히 맞물린다. 이렇게 재는 게 통상 알려진 GIDP율
    (~10%대)과도 실측으로 가장 가깝다(분모를 배티드볼 아웃만으로 좁히면 18-22%로
    과대추정됨 — 삼진처럼 애초에 병살이 불가능한 이벤트가 lineup_sim에서는 구분 없이
    같은 'out'으로 뽑히는데 그걸 분모에서 빼버리면 실제 시뮬레이션 상황과 안 맞기 때문).
    """
    base = pa[(pa["b1_before"] != 0) & (pa["out_before"] < 2)]
    sim_out_bucket = base[~base["outcome"].isin(REACHED_OUTCOMES)]
    gidp = sim_out_bucket[sim_out_bucket["outcome"] == "GIDP"]
    n = len(sim_out_bucket)
    return {
        "gidp": len(gidp), "sim_out_opportunities": n,
        "rate": float(len(gidp) / n) if n else None,
    }


def steal_stats(df: pd.DataFrame, pa: pd.DataFrame) -> dict:
    steal_rows = df[df["type"] == 14]
    attempts = steal_rows[steal_rows["text"].str.contains("도루", na=False) &
                          ~steal_rows["text"].str.contains("견제", na=False)]
    success = attempts[~attempts["text"].str.contains("실패", na=False)]
    on_base_pa = pa[(pa["b1_before"] != 0) | (pa["b2_before"] != 0)]
    n_att = len(attempts)
    return {
        "attempts": n_att, "success": len(success),
        "success_rate": float(len(success) / n_att) if n_att else None,
        "pa_with_runner_on_base": len(on_base_pa),
        "attempt_rate_per_pa_on_base": float(n_att / len(on_base_pa)) if len(on_base_pa) else None,
    }


def outcome_summary(pa: pd.DataFrame) -> dict:
    return pa["outcome"].value_counts().to_dict()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    PROC.mkdir(parents=True, exist_ok=True)

    df = load_events(YEARS)
    print(f"이벤트 {len(df)}행, {df['gameId'].nunique()}경기 로드")

    pa = build_pa_table(df)
    print(f"타석 {len(pa)}개 추출")
    print("outcome 분포:", outcome_summary(pa))

    result = {
        "years": YEARS,
        "n_pa": len(pa),
        "outcome_counts": outcome_summary(pa),
        "runner_on_1st_after_1B": runner_advance_table(pa, "1B", "b1_before"),
        "runner_on_1st_after_2B": runner_advance_table(pa, "2B", "b1_before"),
        "runner_on_2nd_after_1B": runner_advance_table(pa, "1B", "b2_before"),
        "gidp": gidp_stats(pa),
        "steal": steal_stats(df, pa),
    }

    out_path = PROC / "kbo_situational_probs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=float)
    print(f"저장 완료 -> {out_path}")

    print("\n[1루주자 | 단타]", result["runner_on_1st_after_1B"]["overall"])
    print("[1루주자 | 2루타]", result["runner_on_1st_after_2B"]["overall"])
    print("[병살]", result["gidp"])
    print("[도루]", result["steal"])


if __name__ == "__main__":
    main()
