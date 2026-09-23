"""transient 판정에 쓰는 함수들."""
from __future__ import annotations

import numpy as np
import pandas as pd


def tolerance_values(start: float, end: float, step: float) -> np.ndarray:
    """predictability 계산에 쓸 log-tolerance 후보값 목록을 만든다."""
    values = np.arange(start, end + step / 2, step)
    return np.round(values, 2)


def compute_heatmap(data_cache, tolerances, window_sizes, cold_start, transient, low_threshold):
    """특정 transient 후보 하나에 대해, tolerance x window 격자마다
    '저예측성(predictability <= low_threshold) 비율'을 계산한다."""
    heatmap = np.zeros((len(tolerances), len(window_sizes)), dtype=float)
    start_time = cold_start + pd.Timedelta(days=int(transient))

    for i, tol in enumerate(tolerances):
        df = data_cache.get(float(tol))
        if df is None:
            heatmap[i, :] = np.nan
            continue

        vals = df["Predictability"].to_numpy(dtype=float)
        dates = pd.to_datetime(df["Date"])

        for j, window_days in enumerate(window_sizes):
            end_time = start_time + pd.Timedelta(days=int(window_days))
            mask = (dates >= start_time) & (dates <= end_time)
            heatmap[i, j] = (
                float(np.mean(vals[mask] <= low_threshold))
                if int(np.sum(mask)) > 0 else np.nan
            )

    return heatmap


def find_boundary_tolerances(heatmap, tolerances, boundary_fraction):
    """히트맵의 각 window 열에서, 저예측성 비율이 boundary_fraction을 넘는 경계 tolerance 값을 찾는다."""
    boundaries = []

    for j in range(heatmap.shape[1]):
        col_vals = heatmap[:, j]
        valid = ~np.isnan(col_vals)
        if not np.any(valid):
            boundaries.append(np.nan)
            continue

        boundary_tol = None
        for i in range(len(tolerances) - 1):
            curr = col_vals[i]
            next_ = col_vals[i + 1]
            if np.isnan(curr) or np.isnan(next_):
                continue
            if (curr <= boundary_fraction) and (next_ > boundary_fraction):
                boundary_tol = float(tolerances[i])
                break

        if boundary_tol is None:
            valid_vals = col_vals[valid]
            if np.all(valid_vals <= boundary_fraction):
                boundary_tol = float(tolerances[-1])
            else:
                boundary_tol = float(tolerances[0])

        boundaries.append(boundary_tol)

    return np.array(boundaries, dtype=float)


def classify_boundary(boundaries):
    """경계 tolerance 값들이 window가 커질수록 계속 증가(monotonic)하는지 판정한다.
    이게 '안정적으로 예측이 잘 되는 상태'인지를 판단하는 기준이 된다."""
    if np.any(np.isnan(boundaries)):
        return False, False

    diff = np.diff(boundaries)
    monotonic = bool(np.all(diff >= 0))
    strict = bool(monotonic and np.any(diff > 0))
    return monotonic, strict


def choose_optimal_transient(
    transients,
    monotonic_by_t: dict[int, bool],
    strict_by_t: dict[int, bool],
    continuation_days: int = 5,
    require_strict: bool = True,
) -> int:
    """transient 후보(1일째, 2일째, ...)를 앞에서부터 보면서,
    그날이 안정적이고(strict) 그 뒤로도 continuation_days일 연속 안정적이면 그날을 transient로 확정한다.
    끝까지 만족하는 날이 없으면 마지막 후보일(fallback)을 쓴다."""
    fallback = int(max(transients))
    anchor_by_t = strict_by_t if require_strict else monotonic_by_t

    for transient in transients:
        transient = int(transient)
        if not anchor_by_t.get(transient, False):
            continue

        future_range = range(transient + 1, transient + 1 + continuation_days)
        if len(future_range) and max(future_range) > fallback:
            continue

        if all(monotonic_by_t.get(future, False) for future in future_range):
            return transient

    return fallback
