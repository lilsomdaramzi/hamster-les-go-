"""
파라미터 탐색 (Step 1)

고해상도(high) / 저해상도(low_all) 두 그룹 다 잘 맞는 판정 파라미터를 찾는다.
rule(판정규칙) x T_MAX(transient 탐색범위) x window_days x alpha = 480개 조합을 다 돌려서
성공률 성적표를 만들고, 그중 24가지 기준 순서로 봤을 때 안정적으로 뽑히는 후보를 추린다.
마지막에 그 후보들만 predictability를 저장해서, 이후 dip 검증 단계에서 다시 계산 안 해도 되게 한다.

실행: main() 함수가 시작점이다 (파일 맨 아래).
    main() 안에서 순서대로 이렇게 진행된다.
    1. load_entries()        데이터 로딩 (LOW_EXCLUDE_IDS/HIGH_EXCLUDE_IDS로 제외 개체 반영)
    2~3. 캐싱/분류            predictability, transient 안정성 미리 계산
    4. evaluate_combo()       조합 하나를 채점 (predictability -> HPR -> 성공/실패까지 전부 이 안에서 계산됨)
                              -> 이걸 480번 반복해서 성적표(all_combinations)를 만든다
    5. save_criterion_order_sensitivity()   기준 4개의 우선순위를 24가지로 바꿔가며 그래도 같은 후보가
                                             뽑히는지 확인 -> 최종 후보 1개 확정
    6. save_top_case()        그 최종 후보 하나에 대해서만 상세 히트맵/그림을 저장
    7. save_predictability_for_candidates() 확정된 후보의 predictability를 저장 (dip 단계에서 재사용)
"""
from __future__ import annotations

from pathlib import Path
import itertools
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# lib/ 폴더 안의 .py 파일들을 import할 수 있도록 검색 경로에 추가한다.
# 이 스크립트(01_run_sweep.py) 위치 기준으로 경로를 계산하므로 어느 서버로 옮겨도 그대로 동작한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIB = PROJECT_ROOT / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

import search_and_selection as lib  # 파라미터 탐색용 핵심 함수 모음 (lib/search_and_selection.py)
choose_shared_cell_sum_plateau = lib.choose_shared_cell_sum_plateau  # plateau(안정 구간) 기준 선택 함수

# SWEEP_OUT: 결과를 저장할 폴더 이름 (안 정하면 output/param_sweep)
OUTPUT_NAME = os.environ.get("SWEEP_OUT", "param_sweep")
# LOW_EXCLUDE_IDS / HIGH_EXCLUDE_IDS: 각 코호트에서 뺄 개체. 예: '#19,#22'
# 공식 최종 결과는 둘 다 '#19,#22'를 넣어야 나옴. 안 넣으면 다른 파라미터가 뽑힘.
# 데이터 자체는 안 지우고, 이 명령행 환경변수로만 제외 여부를 정한다.
LOW_EXCLUDE_IDS = tuple(
    item.strip()
    for item in os.environ.get("LOW_EXCLUDE_IDS", "").split(",")
    if item.strip()
)
HIGH_EXCLUDE_IDS = tuple(
    item.strip()
    for item in os.environ.get("HIGH_EXCLUDE_IDS", "").split(",")
    if item.strip()
)

OUT = PROJECT_ROOT / "output" / OUTPUT_NAME
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
lib.OUT = OUT  # search_and_selection.py 쪽 함수들도 이 폴더에 쓰도록 덮어씀
lib.FIG = FIG
lib.choose_shared_cell = choose_shared_cell_sum_plateau  # 기본 선택 기준 대신 plateau 기준을 쓰도록 교체


# "좋은 영역" 판단 기준값들 — 이 이상이면 그 (base,slope) 칸을 "괜찮다"고 침
GOOD_SUM_THRESHOLDS = [100, 105, 110, 115, 120]  # high+low 성공률 합이 이 이상인 칸 개수를 셈
WITHIN_MAX_DELTAS = [5, 10]  # 이 조합의 최댓값에서 몇 %p 이내인 칸까지 "좋다"고 볼지


def load_entries():
    """메타데이터(cold_start, onset_time)를 읽고, 제외 개체를 거른 뒤,
    hamster별로 relative_error를 불러와 predictability 계산 준비(fit_params)까지 해둔다."""
    low_meta = pd.read_csv(lib.LOW_METADATA, parse_dates=["cold_start", "onset_time"])
    high_meta = pd.read_csv(lib.HIGH_METADATA, parse_dates=["cold_start", "onset_time"])
    if LOW_EXCLUDE_IDS:
        low_meta = low_meta[~low_meta["hamster_id"].isin(LOW_EXCLUDE_IDS)].copy()
    if HIGH_EXCLUDE_IDS:
        high_meta = high_meta[~high_meta["hamster_id"].isin(HIGH_EXCLUDE_IDS)].copy()
    low_entries = {
        rec.hamster_id: lib.load_entry_full(rec.hamster_id, "lowinterp", lib.LOW_RESULTS, rec.cold_start, rec.onset_time)
        for rec in low_meta.itertuples(index=False)
    }
    high_entries = {
        rec.hamster_id: lib.load_entry_full(rec.hamster_id, "high", lib.HIGH_RESULTS, rec.cold_start, rec.onset_time)
        for rec in high_meta.itertuples(index=False)
    }
    return high_entries, low_entries


def evaluate_combo(
    rule_name: str,     # 판정규칙 이름, 예: "continuation_3"
    tmax: int,           # transient를 며칠까지 탐색할지
    window_days: int,    # log_tol 고를 때 볼 윈도우 일수
    alpha: float,        # log_tol 고를 때 민감도
    high_entries,
    low_entries,
    high_classifications,
    low_classifications,
):
    """파라미터 조합 하나를 실제로 채점한다.
    내부에서 predictability -> HPR -> tilted HPR -> 성공/실패 판정까지 다 계산해서
    base x slope 399칸 히트맵(high/low 각각)을 만들고, 거기서 최고 셀을 찾아 한 줄(row)로 요약한다."""
    row, ranking, artifacts = lib.make_summary_row(
        label="global_good_area",
        rule_name=rule_name,
        tmax=tmax,
        window_days=window_days,
        alpha=alpha,
        high_entries=high_entries,
        low_entries=low_entries,
        high_classifications=high_classifications,
        low_classifications=low_classifications,
    )
    combined = artifacts["combined_matrix"]  # high% + low% (칸마다)
    minimum = artifacts["min_matrix"]        # min(high%, low%) (칸마다)
    high = artifacts["high_matrix"]
    low = artifacts["low_all_matrix"]
    max_sum = float(np.nanmax(combined))

    # 이 조합의 히트맵에서 "좋은 칸"이 몇 개나 되는지 센다.
    # 최고점이 딱 한 칸뿐이면 우연히 거기서만 잘 맞았을 수 있어서 못 믿지만,
    # 좋은 칸이 넓게 퍼져 있으면 이 조합이 실제로 안정적으로 잘 맞는다고 볼 수 있다.
    for threshold in GOOD_SUM_THRESHOLDS:
        row[f"n_cells_sum_ge_{threshold}"] = int((combined >= float(threshold)).sum())
        row[f"pct_cells_sum_ge_{threshold}"] = float((combined >= float(threshold)).mean() * 100.0)
    for delta in WITHIN_MAX_DELTAS:
        row[f"n_cells_within_{delta}_of_max_sum"] = int((combined >= max_sum - float(delta)).sum())
        row[f"pct_cells_within_{delta}_of_max_sum"] = float((combined >= max_sum - float(delta)).mean() * 100.0)
    row["n_cells_both_ge_50"] = int(((high >= 50.0) & (low >= 50.0)).sum())
    row["n_cells_both_ge_55"] = int(((high >= 55.0) & (low >= 55.0)).sum())
    row["n_cells_sum_ge_110_gap_le_10"] = int(((combined >= 110.0) & (np.abs(high - low) <= 10.0)).sum())  # 합도 좋고 high/low 격차도 작은 칸
    row["n_cells_sum_ge_110_min_ge_50"] = int(((combined >= 110.0) & (minimum >= 50.0)).sum())
    return row, ranking, artifacts


# 24가지 기준 순서 테스트에 쓰는 4개 핵심 지표
CRITERIA_ORDER_KEYS = ("fallback_low", "sum_high", "gap_low", "good_area_high")


def sort_by_criteria_order(df: pd.DataFrame, order: tuple[str, ...]) -> pd.DataFrame:
    """4개 지표(fallback/합/격차/좋은영역)를 주어진 우선순위(order)대로 정렬.
    어떤 지표를 1순위로 볼지 바꿔가며 그래도 같은 조합이 뽑히는지 보려는 용도."""
    ranked = df.copy()
    ranked["total_fallback_rate_pct"] = ranked["high_fallback_rate_pct"] + ranked["low_all_fallback_rate_pct"]
    sort_cols = []
    ascending = []
    for criterion in order:
        if criterion == "fallback_low":
            sort_cols.append("total_fallback_rate_pct")
            ascending.append(True)
        elif criterion == "sum_high":
            sort_cols.append("shared_combined_success_pct")
            ascending.append(False)
        elif criterion == "gap_low":
            sort_cols.append("shared_gap_pct")
            ascending.append(True)
        elif criterion == "good_area_high":
            sort_cols.append("n_cells_sum_ge_110_gap_le_10")
            ascending.append(False)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
    sort_cols += ["tmax", "rule", "window_days", "alpha", "shared_base_threshold", "shared_slope"]  # 완전 동점일 때 최후 기준
    ascending += [True, True, True, True, True, True]
    return ranked.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def save_criterion_order_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """4개 지표의 순서를 다 바꿔가며(4! = 24가지) 1등을 뽑아보고,
    그중 몇 번이나 같은 조합이 뽑히는지 세어서 저장. n_orders_selected가 클수록 안정적인 후보.
    최종 후보 목록(1등이 맨 위)을 반환해서, 호출한 쪽에서 최종 채택 조합을 바로 쓸 수 있게 한다."""
    rows = []
    for order in itertools.permutations(CRITERIA_ORDER_KEYS):
        top = sort_by_criteria_order(df, order).iloc[0]
        rows.append(
            {
                "order": " > ".join(order),
                "first_criterion": order[0],
                "rule": top["rule"],
                "tmax": int(top["tmax"]),
                "window_days": int(top["window_days"]),
                "alpha": float(top["alpha"]),
                "base": float(top["shared_base_threshold"]),
                "slope": float(top["shared_slope"]),
                "high": float(top["shared_high_success_pct"]),
                "low_all": float(top["shared_low_all_success_pct"]),
                "sum": float(top["shared_combined_success_pct"]),
                "gap": float(top["shared_gap_pct"]),
                "good_area": int(top["n_cells_sum_ge_110_gap_le_10"]),
                "sum_ge_110_cells": int(top["n_cells_sum_ge_110"]),
                "fallback_high": float(top["high_fallback_rate_pct"]),
                "fallback_low_all": float(top["low_all_fallback_rate_pct"]),
                "fallback_total": float(top["high_fallback_rate_pct"] + top["low_all_fallback_rate_pct"]),
            }
        )
    all_orders = pd.DataFrame(rows)
    all_orders.to_csv(OUT / "criterion_order_sensitivity_all_24_permutations.csv", index=False)  # 24줄 전체 기록

    group_cols = [
        "rule", "tmax", "window_days", "alpha", "base", "slope",
        "high", "low_all", "sum", "gap", "good_area", "fallback_total",
    ]
    selected_summary = (
        all_orders.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n_orders_selected")  # 같은 조합이 24번 중 몇 번 1등했는지
        .sort_values(
            ["n_orders_selected", "fallback_total", "sum", "gap", "good_area", "tmax"],
            ascending=[False, True, False, True, False, True],
        )
        .reset_index(drop=True)
    )
    selected_summary.to_csv(OUT / "criterion_order_sensitivity_selected_summary.csv", index=False)  # 최종 후보 목록
    return selected_summary


def save_top_case(top: pd.Series, high_entries, low_entries, high_classifications, low_classifications, prefix: str):
    """최종 후보 조합 하나에 대해 399칸(base 19 x slope 21) 히트맵을 계산하고,
    같은 결과를 4단계로 압축해서 저장한다.
      1. {prefix}_{high,low_all,combined,min}_success_matrix.csv  399칸 원자료 4개 (값만 다름)
      2. {prefix}_top80_base_slope.csv                           399칸을 성적순 정렬, 상위 80개
      3. {prefix}_summary.csv                                    1등 지점 하나의 요약 1줄
      4. figures/{prefix}_high_lowall_two_panel_heatmaps.png      1번의 high/low 표를 그림으로"""
    final_row, final_ranking, final_artifacts = evaluate_combo(
        rule_name=str(top["rule"]),
        tmax=int(top["tmax"]),
        window_days=int(top["window_days"]),
        alpha=float(top["alpha"]),
        high_entries=high_entries,
        low_entries=low_entries,
        high_classifications=high_classifications,
        low_classifications=low_classifications,
    )
    final_row = {**final_row, **final_ranking.iloc[0].to_dict()}
    lib.matrix_df(final_artifacts["high_matrix"]).to_csv(OUT / f"{prefix}_high_success_matrix.csv")
    lib.matrix_df(final_artifacts["low_all_matrix"]).to_csv(OUT / f"{prefix}_low_all_success_matrix.csv")
    lib.matrix_df(final_artifacts["combined_matrix"]).to_csv(OUT / f"{prefix}_combined_success_matrix.csv")
    lib.matrix_df(final_artifacts["min_matrix"]).to_csv(OUT / f"{prefix}_min_success_matrix.csv")
    pd.DataFrame([final_row]).to_csv(OUT / f"{prefix}_summary.csv", index=False)
    final_ranking.head(80).to_csv(OUT / f"{prefix}_top80_base_slope.csv", index=False)
    plot_two_cohort_heatmaps(
        final_artifacts["high_matrix"],
        final_artifacts["low_all_matrix"],
        final_row,
        f"{prefix}_high_lowall_two_panel_heatmaps.png",
    )
    return final_row, final_artifacts


def plot_two_cohort_heatmaps(high_matrix: np.ndarray, low_matrix: np.ndarray, final_row: dict, filename: str):
    """high/low 성공률 히트맵을 나란히 그리고, 채택된 (base,slope) 지점에 별표를 찍음."""
    panels = [
        ("High success (%)", high_matrix),
        ("Low-all success (%)", low_matrix),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True)
    star_x = int(final_row["base_index"])
    star_y = int(final_row["slope_index"])
    for ax, (title, matrix) in zip(axes, panels):
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="auto", origin="lower")
        ax.scatter(star_x, star_y, s=145, c="gold", edgecolors="black", marker="*", zorder=5)  # 채택 지점 표시
        lib.set_heatmap_ticks(ax)
        ax.set_title(title, fontsize=11, fontweight="bold")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Slope")
    fig.suptitle(
        (
            f"Shared threshold: base={final_row['shared_base_threshold']:.2f}, "
            f"slope={final_row['shared_slope']:.5f}, "
            f"high={final_row['shared_high_success_pct']:.1f}%, "
            f"low-all={final_row['shared_low_all_success_pct']:.1f}%"
        ),
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(FIG / filename, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_transient_bars(selection_by_cohort: dict, final_row: dict, filename: str) -> None:
    """hamster별로 '추위시작 -> transient -> onset'까지 며칠 걸렸는지 막대그래프로 보여준다."""
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.0), sharex=False)
    colors = {"pre": "#6A51A3", "post": "#9ECAE1", "fallback": "#FDE0E0"}
    for ax, (cohort_key, label) in zip(axes, [("high", "High"), ("low_all", "Low-all")]):
        sel = selection_by_cohort[cohort_key]
        entries = sel["entries"]
        st = sel["selected_t"]
        hids = sorted(entries, key=lambda h: int(h.replace("#", "")))
        total = np.array([entries[h]["cold_to_onset_days"] for h in hids], dtype=float)
        transient = np.array([st[h] for h in hids], dtype=float)
        after = np.maximum(total - transient, 0.0)
        fallback = transient >= int(final_row["tmax"])  # transient가 T_MAX까지 못 찾고 끝까지 간 경우
        y = np.arange(len(hids))
        ax.barh(y, transient, color=colors["pre"], alpha=0.86, edgecolor="white", height=0.72, label="Cold start -> transient")
        ax.barh(y, after, left=transient, color=colors["post"], alpha=0.94, edgecolor="white", height=0.72, label="Transient -> onset")
        if fallback.any():
            ax.barh(y[fallback], after[fallback], left=transient[fallback], color=colors["fallback"], alpha=0.72, height=0.72, label="Fallback")
        ax.set_yticks(y)
        ax.set_yticklabels(hids, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Days from cold start")
        ax.set_title(f"{label}: transient selection", fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(
        f"Selected rule: {final_row['rule']}, TMAX={int(final_row['tmax'])}, "
        f"window={int(final_row['window_days'])}d, alpha={float(final_row['alpha']):g}",
        fontsize=13, fontweight="bold", y=1.11,
    )
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_first_drop_panels(fd_by_cohort: dict, final_row: dict, filename: str) -> None:
    """hamster별 first-drop 날짜를 막대그래프로 보여준다. 초록=성공, 빨강=실패."""
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharey=True)
    for ax, (cohort_key, label) in zip(axes, [("high", "High"), ("low_all", "Low-all")]):
        fd = fd_by_cohort[cohort_key].copy()
        fd["hamster_num"] = fd["hamster_id"].str.replace("#", "", regex=False).astype(int)
        fd = fd.sort_values("hamster_num")
        x = np.arange(len(fd))
        vals = fd["first_drop_day"].to_numpy(dtype=float)
        finite = np.isfinite(vals)
        colors = np.array(["#2E7D32" if ok else "#C62828" for ok in fd["success"]])
        ax.axhspan(-lib.SUCCESS_DAYS, 0, color="#DFF1E3", alpha=0.72, zorder=0)  # 성공 판정 구간
        ax.bar(x[finite], vals[finite], color=colors[finite], edgecolor="black", linewidth=0.5, zorder=3)
        if (~finite).any():
            ax.scatter(x[~finite], np.zeros((~finite).sum()), marker="x", s=58, color="#C62828", linewidth=2, zorder=4)
        ax.axhline(-lib.SUCCESS_DAYS, color="black", linestyle="--", alpha=0.65, linewidth=1.1)
        ax.axhline(0, color="black", linewidth=0.9, alpha=0.65)
        ax.set_xticks(x)
        ax.set_xticklabels(fd["hamster_id"], rotation=45, ha="left", fontsize=8)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", labeltop=True, labelbottom=False, top=True, bottom=False, pad=3)
        ax.grid(axis="y", alpha=0.22)
        ax.set_title(f"{label}: {int(fd['success'].sum())}/{len(fd)} success", fontsize=11, fontweight="bold", pad=28)
    axes[0].set_ylabel("First drop day relative to onset")
    fig.suptitle(
        f"First drop, base={final_row['shared_base_threshold']:.2f}, slope={final_row['shared_slope']:.5f}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG / filename, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _first_drop_label(curve: dict) -> str:
    fd = curve["first_drop_day"]
    if not np.isfinite(fd):
        return "NEVER DROPS"
    if abs(fd) < 0.1:
        minutes = int(round(fd * 24.0 * 60.0))
        label = f"{minutes:+d} min" if minutes else "0 min"
    else:
        label = f"{fd:.1f}d"
    if fd < -lib.SUCCESS_DAYS:
        return f"TOO EARLY ({label})"
    if fd <= 0:
        return f"SUCCESS ({label})"
    return f"TOO LATE ({label})"


def plot_hpr_grid(curves_by_h: dict, cohort_label: str, final_row: dict, filename: str, ncols: int = 5) -> None:
    """hamster마다 predictability/HPR/tilted HPR 곡선을 작은 그래프로 그려서 격자로 모아 보여준다."""
    hids = sorted(curves_by_h, key=lambda h: int(h.replace("#", "")))
    n = len(hids)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.25 * ncols, 3.05 * nrows + 1.3), sharex=False, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    n_success = 0
    for idx, hid in enumerate(hids):
        ax = axes[idx]
        curve = curves_by_h[hid]
        cold_rel = curve["cold_rel"]
        x_min, x_max = max(-140.0, cold_rel - 3.0), 2.0
        ax.axvspan(-lib.SUCCESS_DAYS, 0, color="#DFF1E3", alpha=0.75, zorder=0)
        ax.axvline(cold_rel, color="#1565C0", ls="--", lw=1.2, alpha=0.9)
        ax.axvline(curve["transient_rel"], color="#F57C00", lw=1.7, alpha=0.95, zorder=5)
        ax.axvline(0, color="#111111", ls=":", lw=1.4, alpha=0.95)
        pred_mask = (curve["rel_days"] >= x_min) & (curve["rel_days"] <= x_max)
        hpr_mask = (curve["rel_hpr"] >= x_min) & (curve["rel_hpr"] <= x_max)
        ax.plot(curve["rel_days"][pred_mask], curve["predictability"][pred_mask], color="#6C79FF", alpha=0.38, lw=0.65, zorder=1)
        ax.plot(curve["rel_hpr"][hpr_mask], curve["hpr"][hpr_mask], color="#F2A000", lw=1.5, zorder=3)
        ax.plot(curve["rel_hpr"][hpr_mask], curve["tilted_hpr"][hpr_mask], color="#111111", lw=1.45, zorder=4)
        ax.axhline(curve["best_base"], color="#D62728", lw=1.25, zorder=2)
        fd = curve["first_drop_day"]
        success = curve["success"]
        n_success += int(success)
        title_color = "#1B7A2E" if success else "#C62828"
        if np.isfinite(fd) and len(curve["valid_days"]) > 0:
            y = np.interp(fd, curve["valid_days"], curve["valid_tilted_hpr"])
            ax.scatter([fd], [y], s=38, color=title_color, edgecolor="black", linewidth=0.55, zorder=6)
        ax.set_title(f"{hid} [{_first_drop_label(curve)}]", fontsize=8.3, color=title_color, fontweight="bold")
        ax.text(
            0.02, 0.03, f"T={curve['transient_days']}  log_tol={curve['log_tol']:.2f}",
            transform=ax.transAxes, fontsize=7.4, color="#333333", va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#AAAAAA", alpha=0.84, lw=0.55), zorder=7,
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.04, 1.04)
        ax.grid(alpha=0.14)
    for idx in range(n, len(axes)):
        axes[idx].axis("off")
    handles = [
        plt.Line2D([0], [0], color="#6C79FF", lw=1.0, alpha=0.5, label="Predictability"),
        plt.Line2D([0], [0], color="#F2A000", lw=1.8, label="HPR"),
        plt.Line2D([0], [0], color="#111111", lw=1.6, label="Tilted HPR"),
        plt.Line2D([0], [0], color="#D62728", lw=1.4, label="Base threshold"),
        plt.Line2D([0], [0], color="#1565C0", lw=1.2, ls="--", label="Cold start"),
        plt.Line2D([0], [0], color="#F57C00", lw=1.7, label="Selected transient"),
        plt.Line2D([0], [0], color="#111111", lw=1.4, ls=":", label="Onset"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False, fontsize=8.3, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(
        f"{cohort_label}: HPR and first drop ({n_success}/{n} success; "
        f"base={final_row['shared_base_threshold']:.2f}, slope={final_row['shared_slope']:.5f})",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.subplots_adjust(top=0.86, hspace=0.52, wspace=0.12)
    fig.savefig(FIG / filename, dpi=185, bbox_inches="tight")
    plt.close(fig)


def plot_body_temp_grid(cohort_key: str, entries: dict, selected_t: dict, final_row: dict, filename: str, ncols: int = 5) -> None:
    """hamster마다 원본 체온 곡선을 그려서 격자로 모아 보여준다 (transient 표시 포함)."""
    hids = sorted(entries, key=lambda h: int(h.replace("#", "")))
    n = len(hids)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.25 * ncols, 3.0 * nrows + 1.1), sharex=False, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for idx, hid in enumerate(hids):
        ax = axes[idx]
        entry = entries[hid]
        temp = lib.load_raw_body_temp(hid, cohort_key)
        cold_start, onset = entry["cold_start"], entry["onset_time"]
        rel = ((temp.index - cold_start) / pd.Timedelta(days=1)).to_numpy(dtype=float)
        onset_rel = entry["cold_to_onset_days"]
        values = temp.to_numpy(dtype=float)
        mask = (rel >= -10.0) & (rel <= onset_rel + 2.0)
        t = int(selected_t[hid])
        is_fallback = t >= int(final_row["tmax"])
        if is_fallback:
            ax.axvspan(t, onset_rel, color="#FDE0E0", alpha=0.68, zorder=0)
        ax.plot(rel[mask], values[mask], color="#37474F", lw=0.58, zorder=2)
        ax.axvline(0, color="#1565C0", ls="--", lw=1.25)
        ax.axvline(t, color="#F57C00", lw=1.65)
        ax.axvline(onset_rel, color="#111111", ls=":", lw=1.35)
        title_color = "#C62828" if is_fallback else "#111111"
        suffix = " fallback" if is_fallback else ""
        ax.set_title(f"{hid}  T={t}{suffix}", fontsize=8.6, color=title_color, fontweight="bold" if is_fallback else "normal")
        ax.set_ylim(30, 40)
        ax.set_xlim(max(-10.0, np.nanmin(rel[mask]) if mask.any() else -10.0), onset_rel + 2.0)
        ax.grid(alpha=0.14)
    for idx in range(n, len(axes)):
        axes[idx].axis("off")
    handles = [
        plt.Line2D([0], [0], color="#37474F", lw=1.2, label="Body temp"),
        plt.Line2D([0], [0], color="#1565C0", lw=1.25, ls="--", label="Cold start"),
        plt.Line2D([0], [0], color="#F57C00", lw=1.65, label="Selected transient"),
        plt.Line2D([0], [0], color="#111111", lw=1.35, ls=":", label="Onset"),
    ]
    cohort_label = "High" if cohort_key == "high" else "Low-all"
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8.8, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f"{cohort_label}: body temperature until onset with selected transient", fontsize=13, fontweight="bold", y=1.06)
    axes[0].set_ylabel("Body temp (C)")
    fig.text(0.5, 0.006, "Days from cold start", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(FIG / filename, dpi=185, bbox_inches="tight")
    plt.close(fig)


def save_predictability_for_candidates(high_entries, low_entries, high_classifications, low_classifications):
    """480개 전부가 아니라, criterion_order로 뽑힌 후보들만 predictability를 계산해서 저장.
    이 파일 하나만 있으면 이후 dip 단계에서 predictability를 다시 계산할 필요가 없음."""
    candidates_path = OUT / "criterion_order_sensitivity_selected_summary.csv"
    candidates = pd.read_csv(candidates_path)
    rows = []
    for cand in candidates.itertuples(index=False):
        rule = lib.RULE_VARIANTS[cand.rule]
        for cohort, entries, classifications in (
            ("high", high_entries, high_classifications),
            ("low_all", low_entries, low_classifications),
        ):
            selected_t_by_h, log_tol_by_h = lib.build_selection(
                entries, classifications, rule, int(cand.tmax), int(cand.window_days), float(cand.alpha)
            )
            for hamster_id, entry in entries.items():
                curve = lib.build_tol_cache(entry)[float(log_tol_by_h[hamster_id])]
                for _, curve_row in curve.iterrows():
                    rows.append({
                        "candidate": f"{cand.rule}_T{int(cand.tmax)}_W{int(cand.window_days)}_a{cand.alpha:g}",
                        "hamster": hamster_id,
                        "cohort": cohort,
                        "window_time": curve_row["Date"],
                        "predictability": curve_row["Predictability"],
                    })
    pred_df = pd.DataFrame(rows)
    pred_out = PROJECT_ROOT / "output" / "predictability_cache"
    pred_out.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_out / "predictability_selected_candidates.csv", index=False)
    lib.log(f"predictability 저장 완료: {len(pred_df)}행, 후보 {candidates['rule'].nunique()}종")


def main():
    start = time.time()

    # 1) 데이터 로딩 (원본 rel_err + window_times + 메타데이터)
    high_entries, low_entries = load_entries()
    lib.log(f"loaded entries: high={len(high_entries)}, low_all={len(low_entries)}")

    # 2) 모든 tolerance 후보에 대해 predictability를 미리 계산해서 메모리에 캐싱 (뒤에서 반복 재사용)
    lib.log("building predictability caches")
    for entry in list(high_entries.values()) + list(low_entries.values()):
        lib.build_tol_cache(entry)

    # 3) T_MAX=90까지 모든 후보일에 대해 "안정적인지(monotonic/strict)" 미리 분류 (여기가 제일 오래 걸림, ~90초)
    lib.log("classifying transients up to TMAX=90")
    high_classifications = {hid: lib.classify_transients(entry, max(lib.TMAX_GRID)) for hid, entry in high_entries.items()}
    low_classifications = {hid: lib.classify_transients(entry, max(lib.TMAX_GRID)) for hid, entry in low_entries.items()}
    lib.log("classification complete")

    # 4) 480개 조합(규칙4 x T_MAX5 x window4 x alpha6)을 전부 채점 -> 성적표 한 줄씩 쌓음
    rows = []
    for rule_name in lib.RULE_ORDER:
        for tmax in lib.TMAX_GRID:
            for window_days in lib.WINDOW_GRID:
                for alpha in lib.ALPHA_GRID:
                    row, _, _ = evaluate_combo(
                        rule_name, tmax, window_days, alpha,
                        high_entries, low_entries, high_classifications, low_classifications,
                    )
                    rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "global_good_area_all_combinations.csv", index=False)  # 480줄 전체 성적표

    # 5) 4개 지표(합/격차/좋은영역/fallback)의 우선순위를 24가지로 바꿔가며 1등을 뽑아봄
    #    -> 그래도 안정적으로 뽑히는 후보 목록을 얻는다 (n_orders_selected가 클수록 안정적)
    selected_summary = save_criterion_order_sensitivity(df)

    # 6) 그 최종 후보(1등, n_orders_selected가 가장 큰 조합) 하나에 대해서만
    #    상세 히트맵/행렬/그림을 저장한다.
    final_candidate = selected_summary.iloc[0]
    final_row, final_artifacts = save_top_case(
        final_candidate, high_entries, low_entries, high_classifications, low_classifications, "final_candidate"
    )

    # 6-1) 나머지 시각화: transient 막대그래프 / first-drop 막대그래프 / HPR 격자 / 원본 체온 격자
    base = float(final_row["shared_base_threshold"])
    slope = float(final_row["shared_slope"])

    high_fd = lib.first_drop_table(
        final_artifacts["high_curves"], base, slope, "high", final_artifacts["high_t"], final_artifacts["high_lt"]
    )
    low_fd = lib.first_drop_table(
        final_artifacts["low_curves"], base, slope, "low_all", final_artifacts["low_t"], final_artifacts["low_lt"]
    )
    pd.concat([high_fd, low_fd], ignore_index=True).to_csv(OUT / "final_candidate_first_drop_days.csv", index=False)

    selection_by_cohort = {
        "high": {"entries": high_entries, "selected_t": final_artifacts["high_t"]},
        "low_all": {"entries": low_entries, "selected_t": final_artifacts["low_t"]},
    }
    plot_transient_bars(selection_by_cohort, final_row, "final_candidate_transient_bars.png")
    plot_first_drop_panels({"high": high_fd, "low_all": low_fd}, final_row, "final_candidate_first_drop.png")

    lib.log("building full HPR curves for plotting")
    high_full = {
        hid: lib.build_full_curve(high_entries[hid], final_artifacts["high_t"][hid], final_artifacts["high_lt"][hid], base, slope)
        for hid in high_entries
    }
    low_full = {
        hid: lib.build_full_curve(low_entries[hid], final_artifacts["low_t"][hid], final_artifacts["low_lt"][hid], base, slope)
        for hid in low_entries
    }
    plot_hpr_grid(high_full, "High", final_row, "final_candidate_hpr_high.png")
    plot_hpr_grid(low_full, "Low-all", final_row, "final_candidate_hpr_low_all.png")

    lib.log("plotting raw body temperature grids")
    plot_body_temp_grid("high", high_entries, final_artifacts["high_t"], final_row, "final_candidate_bodytemp_high.png")
    plot_body_temp_grid("low_all", low_entries, final_artifacts["low_t"], final_row, "final_candidate_bodytemp_low_all.png")

    # 7) 뽑힌 후보들의 predictability를 저장 (dip 단계에서 재사용, 재계산 불필요)
    save_predictability_for_candidates(high_entries, low_entries, high_classifications, low_classifications)

    # 실행 조건 기록 (나중에 "이 결과가 어떤 설정으로 나왔는지" 확인용)
    with open(OUT / "run_config.txt", "w") as f:
        f.write("global_sweep=rule x TMAX x window_days x alpha\n")
        f.write("main_question=which final hyperparameter combination has broad good-performing base/slope area?\n")
        f.write("good_area_metrics=sum>=100/105/110/115/120, within 5/10 of max sum, both>=50/55, sum>=110 & gap<=10\n")
        f.write(f"n_combinations={len(df)}\n")
        f.write(f"low_exclude_ids={','.join(LOW_EXCLUDE_IDS) if LOW_EXCLUDE_IDS else 'none'}\n")
        f.write(f"high_exclude_ids={','.join(HIGH_EXCLUDE_IDS) if HIGH_EXCLUDE_IDS else 'none'}\n")
        f.write(f"elapsed_seconds={time.time() - start:.1f}\n")

    lib.log("DONE")
    print("최종 채택 후보 (n_orders_selected가 가장 큰 조합):")
    print(final_candidate.to_string())
    print()
    print("24가지 기준 순서에서 뽑힌 후보 전체:")
    print(selected_summary.to_string(index=False))


if __name__ == "__main__":
    main()
