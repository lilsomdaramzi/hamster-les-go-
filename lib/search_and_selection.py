"""
high(고해상도)와 low_all(저해상도) 두 그룹 모두 잘 맞는 판정 파라미터를 찾기 위한 핵심 함수 모음.

한 (base_threshold, slope) 지점의 성공 여부를 정할 때, 아래 세 가지를 같이 고려한다:
    1. min(high 성공률, low_all 성공률) — 한쪽만 잘 되는 건 배제
    2. high 성공률 + low_all 성공률 — 전체적으로 잘 되는지
    3. |high 성공률 - low_all 성공률| — 두 그룹이 비슷하게 잘 되는지 (격차)
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# tconfig.py, transient.py는 정식 설치된 패키지가 아니라 lib/ 폴더 안의 그냥 파일이라서,
# 파이썬이 이 파일들을 찾을 수 있도록 lib/ 폴더를 import 검색 경로(sys.path)에 직접 추가해줘야 한다.
# PROJECT_ROOT는 이 파일(search_and_selection.py) 위치 기준으로 자동 계산되므로,
# 저장소를 어느 폴더/서버로 옮기든 경로를 따로 고칠 필요가 없다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIB = PROJECT_ROOT / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from tconfig import (  # noqa: E402
    LOW_THRESHOLD,
    fit_exp_params,
    load_low_relative_errors,
    load_low_window_times,
    load_relative_errors,
    load_window_times,
    predictability_from_fit_params,
    select_log_tol_from_cache,
)
from transient import (  # noqa: E402
    choose_optimal_transient,
    classify_boundary,
    compute_heatmap,
    find_boundary_tolerances,
    tolerance_values,
)


# OUT/FIG: 실행 스크립트(01_run_sweep.py)가 자기 output 경로로 다시 지정해서 씀.
# 여기서는 폴더를 만들지 않고 이름만 잡아둠 (덮어쓰기 전에 잠깐이라도 빈 폴더가 생기는 걸 방지).
OUT = PROJECT_ROOT / "output"
FIG = OUT / "figures"

# rel_err + window_times가 있는 곳 (hamster 폴더별로 나뉨)
LOW_RESULTS = PROJECT_ROOT / "data" / "low"
LOW_METADATA = PROJECT_ROOT / "data" / "metadata" / "low_hamster_metadata.csv"
HIGH_RESULTS = PROJECT_ROOT / "data" / "high"
HIGH_METADATA = PROJECT_ROOT / "data" / "metadata" / "high_hamster_metadata.csv"


FREQ_WINDOW_SIZE = 120           # HPR 계산할 때 롤링윈도우 크기 (시간 단위 포인트 개수)
FREQ_WINDOW_DAYS = FREQ_WINDOW_SIZE / 24  # 위를 일(day) 단위로 환산 (5일)
SUCCESS_DAYS = 14                # first-drop이 onset 며칠 전까지면 "성공"으로 볼지
SLOPES = np.round(np.arange(0.0, 0.0151, 0.00075), 5)      # tilted HPR 기울기 후보 (히트맵 세로축, 21개)
BASE_THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)  # HPR 기준선 후보 (히트맵 가로축, 19개) -> 19x21=399칸
TOLERANCES = tolerance_values(-1.10, -2.40, -0.05)          # predictability 계산용 log-tolerance 후보값들

# --- 여기부터가 "480개 조합"을 만드는 4개 그리드 ---
TMAX_GRID = [30, 45, 60, 75, 90]        # transient를 최대 며칠까지 탐색할지 (5가지)
WINDOW_SIZES = [5, 10, 15, 20, 25]      # transient 판정용 내부 윈도우 (규칙 계산에 쓰임, 고정값)
BOUNDARY_FRACTION = 0.02                # transient 판정 경계 민감도 (고정값)
DEFAULT_WINDOW_DAYS = 15                # window_days 기본값 (그리드 대신 단일값 쓸 때)
DEFAULT_ALPHA = 0.02                    # alpha 기본값 (그리드 대신 단일값 쓸 때)
WINDOW_GRID = [10, 15, 20, 25]          # log_tol 고를 때 볼 윈도우 일수 (4가지)
ALPHA_GRID = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]  # log_tol 고를 때 민감도 (6가지)
# 4(규칙) x 5(TMAX) x 4(window) x 6(alpha) = 480개 조합

# 판정규칙 4종: continuation_days = "며칠 연속 안정적이어야 확정할지" (클수록 엄격함)
RULE_VARIANTS = {
    "baseline_5day": dict(      # 원래 방식: 5일 연속 확인
        window_sizes=WINDOW_SIZES,
        boundary_fraction=BOUNDARY_FRACTION,
        continuation_days=5,
        require_strict=True,
    ),
    "continuation_2": dict(     # 2일만 확인 (덜 엄격)
        window_sizes=WINDOW_SIZES,
        boundary_fraction=BOUNDARY_FRACTION,
        continuation_days=2,
        require_strict=True,
    ),
    "continuation_3": dict(     # 3일 확인
        window_sizes=WINDOW_SIZES,
        boundary_fraction=BOUNDARY_FRACTION,
        continuation_days=3,
        require_strict=True,
    ),
    "continuation_4": dict(     # 4일 확인
        window_sizes=WINDOW_SIZES,
        boundary_fraction=BOUNDARY_FRACTION,
        continuation_days=4,
        require_strict=True,
    ),
}
RULE_ORDER = ["baseline_5day", "continuation_2", "continuation_3", "continuation_4"]

START = time.time()


def log(message: str) -> None:
    """진행 상황을 경과 시간과 함께 출력한다."""
    print(f"[{time.time() - START:7.1f}s] {message}", flush=True)


def load_entry_full(hamster_id: str, dataset: str, results_dir: Path, cold_start, onset_time) -> dict:
    """hamster 한 마리의 rel_err + window_times를 불러와 지수함수를 피팅한다 (predictability 계산 전 단계).
    결과는 이후 build_tol_cache가 재사용할 수 있도록 dict 형태로 반환한다."""
    if dataset == "lowinterp":
        raw = load_low_relative_errors(results_dir, hamster_id)
        times = load_low_window_times(results_dir, hamster_id)
    else:
        raw = load_relative_errors(results_dir, hamster_id, dataset)
        times = load_window_times(results_dir, hamster_id, dataset)

    window_times = pd.to_datetime(raw.index.map(times))
    valid = window_times.notna()
    raw = raw.loc[raw.index[valid]]
    window_times = pd.to_datetime(window_times[valid])

    fit_params = fit_exp_params(raw).loc[raw.index]
    rel_days = ((window_times - onset_time) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    return {
        "fit_params": fit_params,
        "rel_days": rel_days,
        "cold_start": cold_start,
        "onset_time": onset_time,
        "cold_to_onset_days": float((onset_time - cold_start) / pd.Timedelta(days=1)),
        "tol_cache": None,
    }


def build_tol_cache(entry: dict) -> dict[float, pd.DataFrame]:
    """이 hamster에 대해, 후보 log_tol 값마다 predictability 곡선을 미리 계산해서 캐싱한다.
    이미 계산해뒀으면 다시 계산하지 않고 캐시를 그대로 반환한다."""
    if entry["tol_cache"] is not None:
        return entry["tol_cache"]

    cache = {}
    for log_tol in TOLERANCES:
        pred = predictability_from_fit_params(entry["fit_params"], float(log_tol))
        cache[float(log_tol)] = pd.DataFrame(
            {
                "Relative_Day": entry["rel_days"],
                "Date": entry["onset_time"] + pd.to_timedelta(entry["rel_days"], unit="D"),
                "Predictability": pred,
            }
        )
    entry["tol_cache"] = cache
    return cache


def classify_transients(entry: dict, max_t: int) -> tuple[dict[int, bool], dict[int, bool]]:
    """1일째부터 max_t일째까지, 각 transient 후보일이 '안정적인지(monotonic/strict)'를 미리 다 판정해둔다.
    뒤에서 판정규칙(continuation_days)이 달라져도 이 결과를 그대로 재사용한다."""
    tol_cache = build_tol_cache(entry)
    monotonic_by_t = {}
    strict_by_t = {}
    for t in range(1, max_t + 1):
        heatmap = compute_heatmap(
            data_cache=tol_cache,
            tolerances=TOLERANCES,
            window_sizes=WINDOW_SIZES,
            cold_start=entry["cold_start"],
            transient=t,
            low_threshold=LOW_THRESHOLD,
        )
        boundaries = find_boundary_tolerances(
            heatmap=heatmap,
            tolerances=TOLERANCES,
            boundary_fraction=BOUNDARY_FRACTION,
        )
        monotonic, strict = classify_boundary(boundaries)
        monotonic_by_t[t] = monotonic
        strict_by_t[t] = strict
    return monotonic_by_t, strict_by_t


def selected_transient(classification: tuple[dict[int, bool], dict[int, bool]], rule: dict, tmax: int) -> int:
    """classify_transients 결과에 판정규칙(rule)을 적용해서, 이 조합에서 쓸 transient 하루를 확정한다."""
    monotonic_by_t, strict_by_t = classification
    return int(
        choose_optimal_transient(
            transients=np.arange(1, int(tmax) + 1),
            monotonic_by_t=monotonic_by_t,
            strict_by_t=strict_by_t,
            continuation_days=rule["continuation_days"],
            require_strict=rule["require_strict"],
        )
    )


def build_hpr_curve(entry: dict, selected_t: int, log_tol: float) -> dict:
    """확정된 transient일 + log_tol로 predictability 곡선을 가져와서 HPR(롤링 평균)로 바꾼다.
    transient 이후 구간만 남긴 (날짜, HPR) 쌍을 돌려준다."""
    df = build_tol_cache(entry)[float(log_tol)]
    rel_days = df["Relative_Day"].to_numpy(dtype=float)
    pred = df["Predictability"].to_numpy(dtype=float)
    low_state = (pred <= LOW_THRESHOLD).astype(float)
    hpr_full = 1.0 - pd.Series(low_state).rolling(window=FREQ_WINDOW_SIZE, center=False).mean().to_numpy()
    rel_hpr = rel_days[FREQ_WINDOW_SIZE - 1 :]
    hpr = hpr_full[FREQ_WINDOW_SIZE - 1 :]

    cold_rel = float((entry["cold_start"] - entry["onset_time"]) / pd.Timedelta(days=1))
    transient_rel = cold_rel + float(selected_t)
    start_rel = transient_rel + FREQ_WINDOW_DAYS
    mask = rel_hpr >= start_rel
    return {
        "cold_rel": cold_rel,
        "valid_days": rel_hpr[mask],
        "valid_hpr": hpr[mask],
    }


def first_drop_dynamic(curve: dict, base_thresh: float, slope: float) -> float:
    """base_threshold와 slope로 만든 기울어진 기준선 아래로 HPR이 처음 내려가는 날(days, onset 기준)을 찾는다.
    한 번도 안 내려가면 NaN (판정 실패, fallback)."""
    dyn = np.clip(base_thresh + slope * (curve["valid_days"] - curve["cold_rel"]), 0, 1)
    idx = np.where(curve["valid_hpr"] < dyn)[0]
    return np.nan if len(idx) == 0 else float(curve["valid_days"][idx[0]])


def is_success(day: float) -> bool:
    """first-drop이 onset SUCCESS_DAYS일 전 ~ onset 사이면 성공으로 본다."""
    return bool(np.isfinite(day) and -SUCCESS_DAYS <= day <= 0)


def first_drop_table(
    curves: dict[str, dict],
    base_threshold: float,
    slope: float,
    cohort: str,
    selected_t_by_h: dict[str, int],
    log_tol_by_h: dict[str, float],
) -> pd.DataFrame:
    """hamster마다 first-drop 날짜와 성공 여부를 표로 정리한다 (first-drop 막대그래프용)."""
    rows = []
    for hamster_id, curve in curves.items():
        day = first_drop_dynamic(curve, base_threshold, slope)
        rows.append({
            "cohort": cohort,
            "hamster_id": hamster_id,
            "selected_transient_days": selected_t_by_h[hamster_id],
            "selected_log_tol": log_tol_by_h[hamster_id],
            "base_threshold": base_threshold,
            "slope": slope,
            "first_drop_day": day,
            "success": is_success(day),
        })
    out = pd.DataFrame(rows)
    out["hamster_num"] = out["hamster_id"].str.replace("#", "", regex=False).astype(int)
    return out.sort_values(["cohort", "hamster_num"]).drop(columns=["hamster_num"])


def build_full_curve(entry: dict, selected_t: int, log_tol: float, base: float, slope: float) -> dict:
    """build_hpr_curve보다 더 많은 정보(원본 predictability, tilted HPR 전체 구간 등)를 담은 버전.
    HPR 그림(예측 곡선 전체를 보여주는 그림)을 그릴 때만 필요하다."""
    df = build_tol_cache(entry)[float(log_tol)]
    rel_days = df["Relative_Day"].to_numpy(dtype=float)
    pred = df["Predictability"].to_numpy(dtype=float)
    low_state = (pred <= LOW_THRESHOLD).astype(float)
    hpr_full = 1.0 - pd.Series(low_state).rolling(window=FREQ_WINDOW_SIZE, center=False).mean().to_numpy()
    rel_hpr = rel_days[FREQ_WINDOW_SIZE - 1:]
    hpr = hpr_full[FREQ_WINDOW_SIZE - 1:]

    cold_rel = float((entry["cold_start"] - entry["onset_time"]) / pd.Timedelta(days=1))
    transient_rel = cold_rel + float(selected_t)
    analysis_start_rel = transient_rel + FREQ_WINDOW_DAYS
    tilted_hpr = hpr - float(slope) * (rel_hpr - cold_rel)

    mask = rel_hpr >= analysis_start_rel
    valid_days = rel_hpr[mask]
    valid_tilted = tilted_hpr[mask]
    drop_idx = np.where(valid_tilted < float(base))[0]
    first_drop_day = np.nan if len(drop_idx) == 0 else float(valid_days[drop_idx[0]])
    success = bool(np.isfinite(first_drop_day) and -SUCCESS_DAYS <= first_drop_day <= 0)
    return {
        "transient_days": int(selected_t),
        "log_tol": float(log_tol),
        "cold_rel": cold_rel,
        "transient_rel": transient_rel,
        "analysis_start_rel": analysis_start_rel,
        "rel_days": rel_days,
        "predictability": pred,
        "rel_hpr": rel_hpr,
        "hpr": hpr,
        "tilted_hpr": tilted_hpr,
        "valid_days": valid_days,
        "valid_tilted_hpr": valid_tilted,
        "first_drop_day": first_drop_day,
        "success": success,
        "best_base": float(base),
        "best_slope": float(slope),
    }


_RAW_TEMP_CACHE: dict[str, pd.DataFrame] = {}


def load_raw_body_temp(hamster_id: str, cohort_key: str) -> pd.Series:
    """data/raw_temperature/{high|low}_body_temp.csv에서 이 hamster의 체온 시계열만 뽑아온다."""
    file_key = "high" if cohort_key == "high" else "low"
    if file_key not in _RAW_TEMP_CACHE:
        raw = pd.read_csv(PROJECT_ROOT / "data" / "raw_temperature" / f"{file_key}_body_temp.csv", encoding="utf-8-sig")
        raw["Time"] = pd.to_datetime(raw["Time"])
        _RAW_TEMP_CACHE[file_key] = raw
    raw = _RAW_TEMP_CACHE[file_key]
    sub = raw[raw["Hamster_ID"] == hamster_id]
    return sub.set_index("Time")["Value"].sort_index()


def build_selection(
    entries: dict[str, dict],
    classifications: dict[str, tuple[dict[int, bool], dict[int, bool]]],
    rule: dict,
    tmax: int,
    window_days: int,
    alpha: float,
) -> tuple[dict[str, int], dict[str, float]]:
    """hamster마다 transient일과 log_tol을 확정한다 (한 파라미터 조합 안에서 hamster별로 값이 다를 수 있음)."""
    selected_t_by_h = {}
    log_tol_by_h = {}
    for hamster_id, entry in entries.items():
        t = selected_transient(classifications[hamster_id], rule, tmax)
        log_tol = select_log_tol_from_cache(
            data_cache=build_tol_cache(entry),
            tolerances=TOLERANCES,
            cold_start=entry["cold_start"],
            transient_days=t,
            window_days=window_days,
            alpha=alpha,
        )
        if not np.isfinite(log_tol):
            log_tol = float(np.min(TOLERANCES))
        selected_t_by_h[hamster_id] = t
        log_tol_by_h[hamster_id] = float(log_tol)
    return selected_t_by_h, log_tol_by_h


def compute_curves(
    entries: dict[str, dict],
    selected_t_by_h: dict[str, int],
    log_tol_by_h: dict[str, float],
) -> dict[str, dict]:
    """모든 hamster에 대해 build_hpr_curve를 한 번씩 호출해서 HPR 곡선을 모은다."""
    return {
        hamster_id: build_hpr_curve(entries[hamster_id], selected_t_by_h[hamster_id], log_tol_by_h[hamster_id])
        for hamster_id in entries
    }


def compute_success_matrix(curves: dict[str, dict]) -> np.ndarray:
    """base_threshold x slope 399칸을 다 돌면서, 각 칸에서 전체 hamster 중 몇 %가 성공인지 계산한다.
    이게 히트맵 한 장(한 코호트분)을 만든다."""
    hamster_ids = list(curves)
    matrix = np.zeros((len(SLOPES), len(BASE_THRESHOLDS)), dtype=float)
    for i, slope in enumerate(SLOPES):
        for j, base in enumerate(BASE_THRESHOLDS):
            drops = [first_drop_dynamic(curves[h], float(base), float(slope)) for h in hamster_ids]
            matrix[i, j] = 100.0 * sum(is_success(day) for day in drops) / len(hamster_ids)
    return matrix


def best_cell(matrix: np.ndarray) -> dict:
    """히트맵 한 장에서 성공률이 가장 높은 칸 하나를 찾는다 (한 코호트만 볼 때 씀)."""
    i, j = np.unravel_index(np.nanargmax(matrix), matrix.shape)
    return {
        "base_threshold": float(BASE_THRESHOLDS[j]),
        "slope": float(SLOPES[i]),
        "success_pct": float(matrix[i, j]),
        "base_index": int(j),
        "slope_index": int(i),
    }


def choose_shared_cell(high_matrix: np.ndarray, low_all_matrix: np.ndarray) -> dict:
    """high/low 히트맵 두 장을 같이 보고, min(high,low)가 제일 높은 칸을 우선으로 최종 지점을 고른다.
    (choose_shared_cell_sum_plateau로 교체해서 쓰는 게 기본 설정이지만, 대안 기준으로 남겨둠.)"""
    combined = high_matrix + low_all_matrix
    minimum = np.minimum(high_matrix, low_all_matrix)
    gap = np.abs(high_matrix - low_all_matrix)

    rows = []
    for i, slope in enumerate(SLOPES):
        for j, base in enumerate(BASE_THRESHOLDS):
            rows.append(
                {
                    "slope_index": i,
                    "base_index": j,
                    "slope": float(slope),
                    "base_threshold": float(base),
                    "high_success_pct": float(high_matrix[i, j]),
                    "low_all_success_pct": float(low_all_matrix[i, j]),
                    "combined_success_pct": float(combined[i, j]),
                    "min_success_pct": float(minimum[i, j]),
                    "gap_pct": float(gap[i, j]),
                }
            )
    ranking = pd.DataFrame(rows).sort_values(
        ["min_success_pct", "combined_success_pct", "gap_pct", "slope", "base_threshold"],
        ascending=[False, False, True, True, True],
    )
    top = ranking.iloc[0].to_dict()
    max_sum = float(top["combined_success_pct"])
    max_min = float(np.nanmax(minimum))
    top.update(
        {
            "n_cells_total": int(combined.size),
            "n_cells_at_max_min": int(np.isclose(minimum, max_min).sum()),
            "pct_cells_at_max_min": float(np.isclose(minimum, max_min).mean() * 100.0),
            "n_cells_at_max_sum": int(np.isclose(combined, max_sum).sum()),
            "n_cells_sum_ge_130": int((combined >= 130.0).sum()),
            "pct_cells_sum_ge_130": float((combined >= 130.0).mean() * 100.0),
            "n_cells_sum_ge_120": int((combined >= 120.0).sum()),
            "pct_cells_sum_ge_120": float((combined >= 120.0).mean() * 100.0),
            "n_cells_both_ge_60": int(((high_matrix >= 60.0) & (low_all_matrix >= 60.0)).sum()),
            "pct_cells_both_ge_60": float(((high_matrix >= 60.0) & (low_all_matrix >= 60.0)).mean() * 100.0),
            "n_cells_min_ge_50": int((minimum >= 50.0).sum()),
            "pct_cells_min_ge_50": float((minimum >= 50.0).mean() * 100.0),
        }
    )
    return top, ranking


def matrix_df(matrix: np.ndarray) -> pd.DataFrame:
    """399칸 성공률 배열을 base_threshold/slope 라벨이 붙은 표로 바꾼다 (CSV 저장용)."""
    return pd.DataFrame(matrix, index=pd.Index(SLOPES, name="slope"), columns=[f"{b:.2f}" for b in BASE_THRESHOLDS])


def make_summary_row(
    label: str,
    rule_name: str,
    tmax: int,
    window_days: int,
    alpha: float,
    high_entries: dict[str, dict],
    low_entries: dict[str, dict],
    high_classifications: dict[str, tuple[dict[int, bool], dict[int, bool]]],
    low_classifications: dict[str, tuple[dict[int, bool], dict[int, bool]]],
    save_prefix: str | None = None,
) -> tuple[dict, pd.DataFrame, dict]:
    """파라미터 조합 하나를 끝까지 채점하는 핵심 함수.
    high/low 각각 transient·log_tol 확정 -> HPR 곡선 -> 399칸 히트맵을 만들고,
    choose_shared_cell로 최종 (base,slope) 지점을 골라 한 줄(row)로 요약해 반환한다."""
    rule = RULE_VARIANTS[rule_name]

    high_t, high_lt = build_selection(high_entries, high_classifications, rule, tmax, window_days, alpha)
    low_t, low_lt = build_selection(low_entries, low_classifications, rule, tmax, window_days, alpha)
    high_curves = compute_curves(high_entries, high_t, high_lt)
    low_curves = compute_curves(low_entries, low_t, low_lt)
    high_matrix = compute_success_matrix(high_curves)
    low_matrix = compute_success_matrix(low_curves)

    high_best = best_cell(high_matrix)
    low_best = best_cell(low_matrix)
    shared, ranking = choose_shared_cell(high_matrix, low_matrix)

    row = {
        "label": label,
        "rule": rule_name,
        "tmax": int(tmax),
        "window_days": int(window_days),
        "alpha": float(alpha),
        "shared_base_threshold": shared["base_threshold"],
        "shared_slope": shared["slope"],
        "shared_high_success_pct": shared["high_success_pct"],
        "shared_low_all_success_pct": shared["low_all_success_pct"],
        "shared_combined_success_pct": shared["combined_success_pct"],
        "shared_min_success_pct": shared["min_success_pct"],
        "shared_gap_pct": shared["gap_pct"],
        "n_cells_total": shared["n_cells_total"],
        "n_cells_at_max_min": shared["n_cells_at_max_min"],
        "pct_cells_at_max_min": shared["pct_cells_at_max_min"],
        "n_cells_at_max_sum": shared["n_cells_at_max_sum"],
        "n_cells_sum_ge_130": shared["n_cells_sum_ge_130"],
        "pct_cells_sum_ge_130": shared["pct_cells_sum_ge_130"],
        "n_cells_sum_ge_120": shared["n_cells_sum_ge_120"],
        "pct_cells_sum_ge_120": shared["pct_cells_sum_ge_120"],
        "n_cells_both_ge_60": shared["n_cells_both_ge_60"],
        "pct_cells_both_ge_60": shared["pct_cells_both_ge_60"],
        "n_cells_min_ge_50": shared["n_cells_min_ge_50"],
        "pct_cells_min_ge_50": shared["pct_cells_min_ge_50"],
        "high_individual_best_pct": high_best["success_pct"],
        "high_individual_best_base": high_best["base_threshold"],
        "high_individual_best_slope": high_best["slope"],
        "low_all_individual_best_pct": low_best["success_pct"],
        "low_all_individual_best_base": low_best["base_threshold"],
        "low_all_individual_best_slope": low_best["slope"],
        "high_fallback_rate_pct": 100.0 * sum(t == int(tmax) for t in high_t.values()) / len(high_t),
        "low_all_fallback_rate_pct": 100.0 * sum(t == int(tmax) for t in low_t.values()) / len(low_t),
        "high_median_transient": float(np.median(list(high_t.values()))),
        "low_all_median_transient": float(np.median(list(low_t.values()))),
    }

    artifacts = {
        "high_matrix": high_matrix,
        "low_all_matrix": low_matrix,
        "combined_matrix": high_matrix + low_matrix,
        "min_matrix": np.minimum(high_matrix, low_matrix),
        "high_t": high_t,
        "low_t": low_t,
        "high_lt": high_lt,
        "low_lt": low_lt,
        "high_curves": high_curves,
        "low_curves": low_curves,
    }

    if save_prefix is not None:
        matrix_df(high_matrix).to_csv(OUT / f"{save_prefix}_high_success_matrix.csv")
        matrix_df(low_matrix).to_csv(OUT / f"{save_prefix}_low_all_success_matrix.csv")
        matrix_df(artifacts["combined_matrix"]).to_csv(OUT / f"{save_prefix}_combined_success_matrix.csv")
        matrix_df(artifacts["min_matrix"]).to_csv(OUT / f"{save_prefix}_min_success_matrix.csv")
        ranking.to_csv(OUT / f"{save_prefix}_shared_base_slope_ranking.csv", index=False)

    return row, ranking, artifacts


def set_heatmap_ticks(ax) -> None:
    x_pos, x_lbl = [], []
    for val in np.arange(0.1, 1.0, 0.1):
        idx = int(np.abs(BASE_THRESHOLDS - val).argmin())
        x_pos.append(idx)
        x_lbl.append(f"{val:.1f}")
    y_pos, y_lbl = [], []
    for val in np.arange(0.0, 0.0151, 0.003):
        idx = int(np.abs(SLOPES - val).argmin())
        y_pos.append(idx)
        y_lbl.append(f"{val:.3f}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_lbl)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_lbl)
    ax.set_xlabel("Base threshold")


def choose_shared_cell_sum_plateau(high_matrix: np.ndarray, low_all_matrix: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """base x slope 히트맵에서 최종 셀을 고른다.
    성공률 합이 제일 높은 칸 하나만 보는 게 아니라, 그 주변에 비슷하게 좋은 칸이
    얼마나 넓게 퍼져 있는지(plateau)까지 같이 봐서, 우연히 한 점만 좋은 경우를 걸러낸다."""
    combined = high_matrix + low_all_matrix
    minimum = np.minimum(high_matrix, low_all_matrix)
    gap = np.abs(high_matrix - low_all_matrix)

    rows = []
    for i, slope in enumerate(SLOPES):
        for j, base in enumerate(BASE_THRESHOLDS):
            rows.append(
                {
                    "slope_index": i,
                    "base_index": j,
                    "slope": float(slope),
                    "base_threshold": float(base),
                    "high_success_pct": float(high_matrix[i, j]),
                    "low_all_success_pct": float(low_all_matrix[i, j]),
                    "combined_success_pct": float(combined[i, j]),
                    "min_success_pct": float(minimum[i, j]),
                    "gap_pct": float(gap[i, j]),
                }
            )
    ranking = pd.DataFrame(rows)
    max_sum = float(np.nanmax(combined))
    max_min = float(np.nanmax(minimum))
    n_at_max_sum = int(np.isclose(combined, max_sum).sum())

    ranking = ranking.sort_values(
        ["combined_success_pct", "gap_pct", "min_success_pct", "slope", "base_threshold"],
        ascending=[False, True, False, True, True],
    )
    top = ranking.iloc[0].to_dict()
    top.update(
        {
            "n_cells_total": int(combined.size),
            "n_cells_at_max_sum": n_at_max_sum,
            "pct_cells_at_max_sum": float(n_at_max_sum / combined.size * 100.0),
            "n_cells_at_max_min": int(np.isclose(minimum, max_min).sum()),
            "pct_cells_at_max_min": float(np.isclose(minimum, max_min).mean() * 100.0),
            "n_cells_sum_ge_120": int((combined >= 120.0).sum()),
            "pct_cells_sum_ge_120": float((combined >= 120.0).mean() * 100.0),
            "n_cells_both_ge_60": int(((high_matrix >= 60.0) & (low_all_matrix >= 60.0)).sum()),
            "pct_cells_both_ge_60": float(((high_matrix >= 60.0) & (low_all_matrix >= 60.0)).mean() * 100.0),
            "n_cells_min_ge_50": int((minimum >= 50.0).sum()),
            "pct_cells_min_ge_50": float((minimum >= 50.0).mean() * 100.0),
            "n_cells_sum_ge_130": int((combined >= 130.0).sum()),
            "pct_cells_sum_ge_130": float((combined >= 130.0).mean() * 100.0),
        }
    )
    return top, ranking


