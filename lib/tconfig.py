"""rel_err를 불러와서 predictability로 바꾸는 핵심 계산 함수들."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


LOW_THRESHOLD = 0.1        # predictability가 이 값 이하면 "저예측성"으로 침
OPT_TOL_WINDOW_DAYS = 15   # log_tol 고를 때 볼 기본 윈도우 일수
OPT_TOL_ALPHA = 0.02       # log_tol 고를 때 기본 민감도


def safe_id(hamster_id: str) -> str:
    """'#14' -> 'h14' 처럼, 파일/폴더명에 쓸 수 있는 형태로 바꾼다."""
    return str(hamster_id).replace("#", "h")


def exp_func(x, a, b, c):
    """예측오차 곡선에 맞출 지수함수 모양: a*exp(-b*x)+c."""
    return a * np.exp(-b * x) + c


def train_sizes_from_columns(columns) -> np.ndarray:
    """rel_err 표의 열 이름(예: 'train_12')에서 숫자만 뽑아낸다."""
    sizes = []
    for col in columns:
        text = str(col)
        sizes.append(int(text.split("_")[-1]) if "_" in text else int(text))
    return np.asarray(sizes, dtype=float)


def load_low_relative_errors(low_results: Path, hamster_id: str) -> pd.DataFrame:
    return load_relative_errors(low_results, hamster_id, "lowinterp")


def load_relative_errors(results_dir: Path, hamster_id: str, dataset: str) -> pd.DataFrame:
    """data/{high|low}/{hamster}/{dataset}_{hamster}_relative_errors.csv를 읽는다."""
    sid = safe_id(hamster_id)
    path = Path(results_dir) / sid / f"{dataset}_{sid}_relative_errors.csv"
    raw = pd.read_csv(path, index_col=0)
    raw = raw.apply(pd.to_numeric, errors="coerce").replace(0, np.nan)
    raw.index = pd.to_numeric(raw.index, errors="coerce").astype("Int64")
    return raw


def load_low_window_times(low_results: Path, hamster_id: str) -> pd.Series:
    return load_window_times(low_results, hamster_id, "lowinterp")


def load_window_times(results_dir: Path, hamster_id: str, dataset: str) -> pd.Series:
    """window_start(번호) -> 실제 시각(window_time) 매핑을 읽는다.
    이 파일이 없으면 훨씬 큰 relative_errors_long.csv에서 대신 읽는다 (느림, 되도록 피할 것)."""
    sid = safe_id(hamster_id)
    hamster_dir = Path(results_dir) / sid
    path = hamster_dir / f"{dataset}_{sid}_window_times.csv"
    if path.exists():
        wt = pd.read_csv(path)
    else:
        wt = pd.read_csv(
            hamster_dir / f"{dataset}_{sid}_relative_errors_long.csv",
            usecols=["window_start", "window_time"],
        )
    wt = wt.drop_duplicates("window_start").copy()
    wt["window_time"] = pd.to_datetime(wt["window_time"])
    return wt.set_index("window_start")["window_time"]


def fit_exp_params(raw_df: pd.DataFrame) -> pd.DataFrame:
    """rel_err 표의 각 시간 구간(행)마다, 예측오차 곡선에 지수함수를 피팅해서
    계수(a, b, c)를 구한다. 이 계수가 이후 predictability 계산의 재료가 된다."""
    train_sizes = train_sizes_from_columns(raw_df.columns)
    max_training = float(np.nanmax(train_sizes))
    rows = []
    for window_start, row in raw_df.iterrows():
        y_raw = row.to_numpy(dtype=float)
        mask = np.isfinite(y_raw) & (y_raw > 0)
        rec = {
            "window_start": int(window_start),
            "a": np.nan, "b": np.nan, "c": np.nan,
            "fit_ok": False,
            "max_training": max_training,
        }
        if mask.sum() >= 3:
            x = train_sizes[mask]
            y = np.log10(y_raw[mask])
            try:
                popt, _ = curve_fit(exp_func, x, y, p0=(1, 0.01, -2), maxfev=10000)
                rec.update({"a": float(popt[0]), "b": float(popt[1]), "c": float(popt[2]), "fit_ok": True})
            except Exception:
                pass
        rows.append(rec)
    return pd.DataFrame(rows).set_index("window_start")


def predictability_from_fit_params(fit_params: pd.DataFrame, log_tol: float) -> np.ndarray:
    """fit_exp_params가 만든 계수와 tolerance 하나를 받아서,
    '이 정도 오차 안에서 예측이 유지되는 비율(predictability, 0~1)'을 시간 구간마다 계산한다."""
    tol = float(log_tol)
    max_training = float(fit_params["max_training"].iloc[0])
    pred = []
    for row in fit_params.itertuples():
        x_cross = max_training
        if bool(row.fit_ok) and np.isfinite(row.a) and np.isfinite(row.b) and np.isfinite(row.c):
            if row.a > 0 and row.b > 0 and tol > row.c:
                value = (tol - row.c) / row.a
                if value > 0:
                    x_cross = -np.log(value) / row.b
                    if (not np.isfinite(x_cross)) or x_cross > max_training:
                        x_cross = max_training
                    if x_cross < 0:
                        x_cross = 0.0
        pred.append(1.0 - float(x_cross) / max_training)
    return np.asarray(pred, dtype=float)


def select_log_tol_from_cache(
    data_cache,          # {log_tol: predictability DataFrame} 형태의 캐시
    tolerances,          # 후보 log_tol 값들
    cold_start,          # 이 hamster의 추위 시작일
    transient_days,      # transient로 확정된 날짜 (며칠째)
    window_days=OPT_TOL_WINDOW_DAYS,
    alpha=OPT_TOL_ALPHA,
    low_threshold=LOW_THRESHOLD,
):
    """transient 이후 window_days 구간에서, 저예측성 비율이 alpha 이상 되는
    가장 관대한(느슨한) log_tol 값을 고른다."""
    start_time = pd.to_datetime(cold_start) + pd.Timedelta(days=int(transient_days))
    end_time = start_time + pd.Timedelta(days=int(window_days))

    for log_tol in sorted(tolerances, reverse=True):
        df = data_cache[float(log_tol)]
        sub = df[(df["Date"] >= start_time) & (df["Date"] <= end_time)]
        low_fraction = np.nan if len(sub) == 0 else float(np.mean(sub["Predictability"] <= low_threshold))
        if np.isfinite(low_fraction) and low_fraction >= alpha:
            return float(log_tol)
    return np.nan
