"""
Script 77
=========
Cross-Country Calibration and Threshold-Shift Analysis for CartoGraph-X.

Purpose
-------
Diagnose whether zero-shot cross-country degradation is primarily caused by:

1. representation/ranking failure, or
2. probability/decision-threshold shift between national cartographic domains.

IMPORTANT
---------
The target-domain oracle threshold is DIAGNOSTIC ONLY.

It uses target test labels and therefore MUST NOT be reported as genuine
zero-shot performance. The locked source-validation threshold remains the
official zero-shot result.

Inputs
------
data/processed/cross_country/cartographx_models/
    full_all_runs.csv

    lt_to_pl_zero_shot/
        seed_42/
            test_predictions.csv
            run_report.json
        ...
        seed_51/

    pl_to_lt_zero_shot/
        seed_42/
            test_predictions.csv
            run_report.json
        ...
        seed_51/

Outputs
-------
data/processed/cross_country/calibration_analysis/

    zero_shot_calibration_runs.csv
    zero_shot_calibration_summary.csv
    threshold_shift_summary.csv
    calibration_bins.csv
    class_probability_summary.csv
    city_calibration_metrics.csv
    threshold_sweep.csv
    calibration_analysis_report.json

    figures/
        threshold_shift_by_seed.png
        mcc_locked_vs_oracle.png
        ba_locked_vs_oracle.png
        calibration_lt_to_pl.png
        calibration_pl_to_lt.png
        probability_distribution_lt_to_pl.png
        probability_distribution_pl_to_lt.png
        threshold_vs_mcc_pl_to_lt.png
        threshold_vs_mcc_lt_to_pl.png
        predicted_vs_actual_prevalence.png

No models are trained by this script.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(r"D:\Research\CartoGen")

MODEL_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "cross_country"
    / "cartographx_models"
)

OUTPUT_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "cross_country"
    / "calibration_analysis"
)

FIGURE_ROOT = OUTPUT_ROOT / "figures"

FULL_RUN_FILE = MODEL_ROOT / "full_all_runs.csv"

PROTOCOLS = [
    "lt_to_pl_zero_shot",
    "pl_to_lt_zero_shot",
]

SEEDS = list(range(42, 52))

N_CALIBRATION_BINS = 10

# Fine grid is acceptable here because this is analysis only.
THRESHOLD_GRID = np.round(
    np.arange(0.01, 1.00, 0.01),
    2,
)

EPS = 1e-12


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def banner(title: str) -> None:
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def section(title: str) -> None:
    print()
    print("-" * 120)
    print(title)
    print("-" * 120)


def ensure_directories() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)


def json_safe(value):
    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value

    if pd.isna(value):
        return None

    return value


def save_json(path: Path, obj: dict) -> None:
    path.write_text(
        json.dumps(
            json_safe(obj),
            indent=2,
        ),
        encoding="utf-8",
    )


def safe_mean(values) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return float("nan")

    return float(np.mean(x))


def safe_std(values) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) <= 1:
        return float("nan")

    return float(np.std(x, ddof=1))


def mean_ci95(values) -> Tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    mean = float(np.mean(x))

    if len(x) <= 1:
        return mean, float("nan"), float("nan")

    sd = float(np.std(x, ddof=1))
    half = 1.96 * sd / math.sqrt(len(x))

    return (
        mean,
        mean - half,
        mean + half,
    )


# =============================================================================
# VALIDATION
# =============================================================================

def verify_inputs() -> None:
    banner("1. VERIFY INPUTS")

    if not FULL_RUN_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n{FULL_RUN_FILE}"
        )

    full = pd.read_csv(FULL_RUN_FILE)

    required_full = {
        "protocol",
        "seed",
        "threshold",
        "test_mcc",
        "test_balanced_accuracy",
        "test_roc_auc",
        "test_pr_auc",
    }

    missing = required_full - set(full.columns)

    if missing:
        raise KeyError(
            "Missing columns in full_all_runs.csv:\n"
            + "\n".join(sorted(missing))
        )

    print("[OK]", FULL_RUN_FILE)

    failures = []

    for protocol in PROTOCOLS:
        for seed in SEEDS:

            run_dir = (
                MODEL_ROOT
                / protocol
                / f"seed_{seed}"
            )

            pred = run_dir / "test_predictions.csv"
            report = run_dir / "run_report.json"

            if not pred.exists():
                failures.append(str(pred))

            if not report.exists():
                failures.append(str(report))

    if failures:
        raise FileNotFoundError(
            "Missing required run files:\n"
            + "\n".join(failures)
        )

    print(
        f"[OK] {len(PROTOCOLS) * len(SEEDS)} "
        "zero-shot prediction/report pairs"
    )


# =============================================================================
# METRICS
# =============================================================================

def binary_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> Dict[str, float]:

    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)

    y_pred = (
        probability >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else float("nan")
    )

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "specificity": float(specificity),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
            )
        ),
        "predicted_prevalence": float(
            np.mean(y_pred)
        ),
        "actual_prevalence": float(
            np.mean(y_true)
        ),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probability,
            )
        )

        metrics["pr_auc"] = float(
            average_precision_score(
                y_true,
                probability,
            )
        )

    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    metrics["brier"] = float(
        brier_score_loss(
            y_true,
            probability,
        )
    )

    return metrics


# =============================================================================
# EXPECTED CALIBRATION ERROR
# =============================================================================

def calibration_table(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int = 10,
) -> Tuple[pd.DataFrame, float]:

    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(
        probability,
        dtype=float,
    )

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    # right=True would put exactly 0 in -1,
    # so clip explicitly.
    bin_index = np.digitize(
        probability,
        edges[1:-1],
        right=False,
    )

    rows = []

    n = len(y_true)
    ece = 0.0

    for b in range(n_bins):

        mask = bin_index == b
        count = int(mask.sum())

        lower = float(edges[b])
        upper = float(edges[b + 1])

        if count == 0:
            rows.append(
                {
                    "bin": b,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "count": 0,
                    "mean_probability": np.nan,
                    "observed_frequency": np.nan,
                    "absolute_gap": np.nan,
                }
            )
            continue

        mean_probability = float(
            np.mean(probability[mask])
        )

        observed_frequency = float(
            np.mean(y_true[mask])
        )

        gap = abs(
            mean_probability
            - observed_frequency
        )

        ece += (
            count / n
        ) * gap

        rows.append(
            {
                "bin": b,
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "mean_probability": mean_probability,
                "observed_frequency": observed_frequency,
                "absolute_gap": gap,
            }
        )

    return pd.DataFrame(rows), float(ece)


# =============================================================================
# CALIBRATION INTERCEPT / SLOPE
# =============================================================================

def calibration_slope_intercept(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> Tuple[float, float]:

    """
    Diagnostic logistic calibration model:

        logit(P(Y=1)) = intercept + slope * logit(predicted_probability)

    Uses sklearn LogisticRegression on the prediction logit.

    This is diagnostic only.
    """

    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return float("nan"), float("nan")

    y_true = np.asarray(y_true, dtype=int)

    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")

    p = np.clip(
        np.asarray(probability, dtype=float),
        1e-6,
        1.0 - 1e-6,
    )

    logits = np.log(
        p / (1.0 - p)
    ).reshape(-1, 1)

    try:
        model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=5000,
        )

        model.fit(
            logits,
            y_true,
        )

        intercept = float(
            model.intercept_[0]
        )

        slope = float(
            model.coef_[0, 0]
        )

        return intercept, slope

    except Exception:

        # Compatibility fallback for older sklearn.
        try:
            model = LogisticRegression(
                C=1e12,
                solver="lbfgs",
                max_iter=5000,
            )

            model.fit(
                logits,
                y_true,
            )

            return (
                float(model.intercept_[0]),
                float(model.coef_[0, 0]),
            )

        except Exception:
            return (
                float("nan"),
                float("nan"),
            )


# =============================================================================
# ORACLE THRESHOLD
# =============================================================================

def find_oracle_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:

    """
    DIAGNOSTIC ONLY.

    Select target-test threshold maximizing MCC.

    Tie breaking:
        1. highest MCC
        2. highest balanced accuracy
        3. threshold closest to 0.5
        4. lower threshold
    """

    rows = []

    for threshold in THRESHOLD_GRID:

        metrics = binary_metrics(
            y_true,
            probability,
            float(threshold),
        )

        rows.append(
            {
                "threshold": float(threshold),
                "mcc": metrics["mcc"],
                "balanced_accuracy":
                    metrics["balanced_accuracy"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "predicted_prevalence":
                    metrics["predicted_prevalence"],
            }
        )

    sweep = pd.DataFrame(rows)

    ranked = sweep.copy()

    ranked["distance_from_half"] = (
        ranked["threshold"] - 0.5
    ).abs()

    ranked = ranked.sort_values(
        [
            "mcc",
            "balanced_accuracy",
            "distance_from_half",
            "threshold",
        ],
        ascending=[
            False,
            False,
            True,
            True,
        ],
    )

    best_threshold = float(
        ranked.iloc[0]["threshold"]
    )

    best_metrics = binary_metrics(
        y_true,
        probability,
        best_threshold,
    )

    return (
        best_threshold,
        best_metrics,
        sweep,
    )


# =============================================================================
# CLASS PROBABILITY SUMMARY
# =============================================================================

def class_probability_rows(
    protocol: str,
    seed: int,
    y_true: np.ndarray,
    probability: np.ndarray,
) -> List[dict]:

    rows = []

    for label in [0, 1]:

        values = probability[
            y_true == label
        ]

        if len(values) == 0:
            continue

        rows.append(
            {
                "protocol": protocol,
                "seed": seed,
                "class_label": label,
                "n": len(values),
                "mean_probability":
                    float(np.mean(values)),
                "std_probability":
                    float(np.std(values, ddof=1))
                    if len(values) > 1
                    else np.nan,
                "median_probability":
                    float(np.median(values)),
                "q05":
                    float(np.quantile(values, 0.05)),
                "q25":
                    float(np.quantile(values, 0.25)),
                "q75":
                    float(np.quantile(values, 0.75)),
                "q95":
                    float(np.quantile(values, 0.95)),
            }
        )

    return rows


# =============================================================================
# CITY METRICS
# =============================================================================

def city_metrics(
    data: pd.DataFrame,
    protocol: str,
    seed: int,
    locked_threshold: float,
) -> List[dict]:

    rows = []

    if "tile_id" not in data.columns:
        return rows

    # sample_uid format:
    # country::city::index
    if "sample_uid" in data.columns:
        city = (
            data["sample_uid"]
            .astype(str)
            .str.split("::")
            .str[1]
        )
    else:
        city = (
            data["tile_id"]
            .astype(str)
            .str.split("_")
            .str[0]
        )

    tmp = data.copy()
    tmp["_city"] = city

    for city_name, g in tmp.groupby("_city"):

        y = g["y_true"].to_numpy(dtype=int)
        p = g["prob_retained"].to_numpy(dtype=float)

        if len(g) == 0:
            continue

        m = binary_metrics(
            y,
            p,
            locked_threshold,
        )

        _, ece = calibration_table(
            y,
            p,
            N_CALIBRATION_BINS,
        )

        rows.append(
            {
                "protocol": protocol,
                "seed": seed,
                "city": city_name,
                "n": len(g),
                "actual_prevalence":
                    m["actual_prevalence"],
                "predicted_prevalence":
                    m["predicted_prevalence"],
                "balanced_accuracy":
                    m["balanced_accuracy"],
                "mcc":
                    m["mcc"],
                "roc_auc":
                    m["roc_auc"],
                "pr_auc":
                    m["pr_auc"],
                "brier":
                    m["brier"],
                "ece":
                    ece,
            }
        )

    return rows


# =============================================================================
# LOAD RUN
# =============================================================================

def load_run(
    protocol: str,
    seed: int,
) -> Tuple[pd.DataFrame, dict]:

    run_dir = (
        MODEL_ROOT
        / protocol
        / f"seed_{seed}"
    )

    pred_path = (
        run_dir
        / "test_predictions.csv"
    )

    report_path = (
        run_dir
        / "run_report.json"
    )

    data = pd.read_csv(pred_path)

    required = {
        "y_true",
        "prob_retained",
        "y_pred",
    }

    missing = required - set(data.columns)

    if missing:
        raise KeyError(
            f"{pred_path}\n"
            f"Missing columns: {sorted(missing)}"
        )

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    return data, report


# =============================================================================
# ANALYZE ONE RUN
# =============================================================================

def analyze_run(
    protocol: str,
    seed: int,
) -> Tuple[
    dict,
    pd.DataFrame,
    List[dict],
    List[dict],
    pd.DataFrame,
]:

    data, report = load_run(
        protocol,
        seed,
    )

    y = data["y_true"].to_numpy(
        dtype=int
    )

    p = data["prob_retained"].to_numpy(
        dtype=float
    )

    locked_threshold = float(
        report["threshold"]
    )

    # ---------------------------------------------------------
    # Locked zero-shot result
    # ---------------------------------------------------------

    locked = binary_metrics(
        y,
        p,
        locked_threshold,
    )

    # Verify consistency with saved predictions.
    recomputed_y_pred = (
        p >= locked_threshold
    ).astype(int)

    saved_y_pred = data[
        "y_pred"
    ].to_numpy(dtype=int)

    prediction_match = bool(
        np.array_equal(
            recomputed_y_pred,
            saved_y_pred,
        )
    )

    # ---------------------------------------------------------
    # Calibration
    # ---------------------------------------------------------

    cal_table, ece = calibration_table(
        y,
        p,
        N_CALIBRATION_BINS,
    )

    intercept, slope = (
        calibration_slope_intercept(
            y,
            p,
        )
    )

    # ---------------------------------------------------------
    # Target oracle
    # DIAGNOSTIC ONLY
    # ---------------------------------------------------------

    (
        oracle_threshold,
        oracle,
        threshold_sweep,
    ) = find_oracle_threshold(
        y,
        p,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    row = {
        "protocol": protocol,
        "seed": seed,
        "target_country":
            str(data["country"].iloc[0])
            if "country" in data.columns
            else "",
        "n_test": len(data),

        "actual_prevalence":
            locked["actual_prevalence"],

        "locked_threshold":
            locked_threshold,

        "locked_predicted_prevalence":
            locked["predicted_prevalence"],

        "locked_accuracy":
            locked["accuracy"],

        "locked_balanced_accuracy":
            locked["balanced_accuracy"],

        "locked_precision":
            locked["precision"],

        "locked_recall":
            locked["recall"],

        "locked_specificity":
            locked["specificity"],

        "locked_f1":
            locked["f1"],

        "locked_mcc":
            locked["mcc"],

        "roc_auc":
            locked["roc_auc"],

        "pr_auc":
            locked["pr_auc"],

        "brier":
            locked["brier"],

        "ece":
            ece,

        "calibration_intercept":
            intercept,

        "calibration_slope":
            slope,

        "oracle_threshold":
            oracle_threshold,

        "oracle_predicted_prevalence":
            oracle["predicted_prevalence"],

        "oracle_balanced_accuracy":
            oracle["balanced_accuracy"],

        "oracle_mcc":
            oracle["mcc"],

        "oracle_f1":
            oracle["f1"],

        "threshold_shift":
            oracle_threshold
            - locked_threshold,

        "absolute_threshold_shift":
            abs(
                oracle_threshold
                - locked_threshold
            ),

        "recoverable_mcc":
            oracle["mcc"]
            - locked["mcc"],

        "recoverable_balanced_accuracy":
            oracle["balanced_accuracy"]
            - locked["balanced_accuracy"],

        "prevalence_error_locked":
            locked["predicted_prevalence"]
            - locked["actual_prevalence"],

        "prevalence_error_oracle":
            oracle["predicted_prevalence"]
            - oracle["actual_prevalence"],

        "saved_prediction_matches_threshold":
            prediction_match,

        "best_epoch":
            report.get(
                "best_epoch",
                np.nan,
            ),

        "validation_mcc":
            report.get(
                "validation_mcc",
                np.nan,
            ),

        "validation_balanced_accuracy":
            report.get(
                "validation_balanced_accuracy",
                np.nan,
            ),
    }

    # ---------------------------------------------------------
    # Add protocol/seed to calibration table
    # ---------------------------------------------------------

    cal_table.insert(
        0,
        "seed",
        seed,
    )

    cal_table.insert(
        0,
        "protocol",
        protocol,
    )

    # ---------------------------------------------------------
    # Add protocol/seed to threshold sweep
    # ---------------------------------------------------------

    threshold_sweep.insert(
        0,
        "seed",
        seed,
    )

    threshold_sweep.insert(
        0,
        "protocol",
        protocol,
    )

    class_rows = class_probability_rows(
        protocol,
        seed,
        y,
        p,
    )

    city_rows = city_metrics(
        data,
        protocol,
        seed,
        locked_threshold,
    )

    return (
        row,
        cal_table,
        class_rows,
        city_rows,
        threshold_sweep,
    )


# =============================================================================
# AGGREGATE SUMMARY
# =============================================================================

def build_summary(
    runs: pd.DataFrame,
) -> pd.DataFrame:

    metrics = [
        "locked_threshold",
        "oracle_threshold",
        "threshold_shift",
        "absolute_threshold_shift",

        "actual_prevalence",
        "locked_predicted_prevalence",
        "oracle_predicted_prevalence",

        "locked_balanced_accuracy",
        "oracle_balanced_accuracy",
        "recoverable_balanced_accuracy",

        "locked_mcc",
        "oracle_mcc",
        "recoverable_mcc",

        "roc_auc",
        "pr_auc",
        "brier",
        "ece",

        "calibration_intercept",
        "calibration_slope",
    ]

    rows = []

    for protocol, g in runs.groupby(
        "protocol"
    ):

        for metric in metrics:

            values = g[
                metric
            ].to_numpy(dtype=float)

            mean, lo, hi = mean_ci95(
                values
            )

            rows.append(
                {
                    "protocol": protocol,
                    "metric": metric,
                    "runs": int(
                        np.isfinite(values).sum()
                    ),
                    "mean": mean,
                    "std": safe_std(values),
                    "median": float(
                        np.nanmedian(values)
                    ),
                    "min": float(
                        np.nanmin(values)
                    ),
                    "max": float(
                        np.nanmax(values)
                    ),
                    "ci95_lower": lo,
                    "ci95_upper": hi,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# THRESHOLD-SHIFT CORRELATIONS
# =============================================================================

def correlation_diagnostics(
    runs: pd.DataFrame,
) -> List[dict]:

    rows = []

    for protocol, g in runs.groupby(
        "protocol"
    ):

        comparisons = [
            (
                "absolute_threshold_shift",
                "recoverable_mcc",
            ),
            (
                "absolute_threshold_shift",
                "locked_mcc",
            ),
            (
                "prevalence_error_locked",
                "locked_mcc",
            ),
            (
                "locked_threshold",
                "locked_mcc",
            ),
            (
                "roc_auc",
                "locked_mcc",
            ),
        ]

        for xcol, ycol in comparisons:

            x = g[xcol].to_numpy(
                dtype=float
            )

            y = g[ycol].to_numpy(
                dtype=float
            )

            mask = (
                np.isfinite(x)
                & np.isfinite(y)
            )

            if mask.sum() < 3:
                rho = np.nan
                p_value = np.nan

            elif spearmanr is None:
                rho = np.nan
                p_value = np.nan

            else:
                result = spearmanr(
                    x[mask],
                    y[mask],
                )

                rho = float(
                    result.statistic
                )

                p_value = float(
                    result.pvalue
                )

            rows.append(
                {
                    "protocol": protocol,
                    "x": xcol,
                    "y": ycol,
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "n": int(mask.sum()),
                }
            )

    return rows


# =============================================================================
# FIGURES
# =============================================================================

def save_figure(
    fig,
    filename: str,
) -> None:

    png = (
        FIGURE_ROOT
        / f"{filename}.png"
    )

    pdf = (
        FIGURE_ROOT
        / f"{filename}.pdf"
    )

    fig.tight_layout()

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)


def figure_threshold_shift(
    runs: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for protocol, g in runs.groupby(
        "protocol"
    ):

        g = g.sort_values("seed")

        ax.plot(
            g["seed"],
            g["locked_threshold"],
            marker="o",
            label=f"{protocol} locked",
        )

        ax.plot(
            g["seed"],
            g["oracle_threshold"],
            marker="s",
            linestyle="--",
            label=f"{protocol} oracle diagnostic",
        )

    ax.set_xlabel("Seed")
    ax.set_ylabel("Decision threshold")
    ax.set_title(
        "Source-selected vs target-oracle thresholds"
    )
    ax.legend()

    save_figure(
        fig,
        "threshold_shift_by_seed",
    )


def figure_locked_vs_oracle(
    runs: pd.DataFrame,
    metric: str,
    filename: str,
    ylabel: str,
) -> None:

    protocols = PROTOCOLS

    x = np.arange(
        len(SEEDS)
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    width = 0.35

    for i, protocol in enumerate(
        protocols
    ):

        g = (
            runs[
                runs["protocol"]
                == protocol
            ]
            .sort_values("seed")
        )

        offset = (
            -width / 2
            if i == 0
            else width / 2
        )

        ax.plot(
            x,
            g[f"locked_{metric}"],
            marker="o",
            label=f"{protocol} locked",
        )

        ax.plot(
            x,
            g[f"oracle_{metric}"],
            marker="s",
            linestyle="--",
            label=f"{protocol} oracle diagnostic",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        SEEDS
    )

    ax.set_xlabel("Seed")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"Locked zero-shot vs diagnostic oracle {ylabel}"
    )
    ax.legend()

    save_figure(
        fig,
        filename,
    )


def figure_calibration(
    calibration_bins: pd.DataFrame,
    protocol: str,
    filename: str,
) -> None:

    g = calibration_bins[
        calibration_bins["protocol"]
        == protocol
    ].copy()

    # Average bin statistics across seeds,
    # weighted indirectly by equal test size.
    avg = (
        g.groupby("bin")
        .agg(
            mean_probability=(
                "mean_probability",
                "mean",
            ),
            observed_frequency=(
                "observed_frequency",
                "mean",
            ),
        )
        .reset_index()
    )

    avg = avg.dropna()

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    ax.plot(
        avg["mean_probability"],
        avg["observed_frequency"],
        marker="o",
        label=protocol,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel(
        "Mean predicted retention probability"
    )

    ax.set_ylabel(
        "Observed retention frequency"
    )

    ax.set_title(
        f"Zero-shot calibration: {protocol}"
    )

    ax.legend()

    save_figure(
        fig,
        filename,
    )


def figure_probability_distribution(
    protocol: str,
    filename: str,
) -> None:

    # Pool probabilities across seeds for visualization.
    # This is a descriptive figure only.
    class0 = []
    class1 = []

    for seed in SEEDS:

        data, _ = load_run(
            protocol,
            seed,
        )

        y = data["y_true"].to_numpy(
            dtype=int
        )

        p = data[
            "prob_retained"
        ].to_numpy(dtype=float)

        class0.extend(
            p[y == 0].tolist()
        )

        class1.extend(
            p[y == 1].tolist()
        )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    bins = np.linspace(
        0,
        1,
        41,
    )

    ax.hist(
        class0,
        bins=bins,
        density=True,
        alpha=0.45,
        label="Eliminated (y=0)",
    )

    ax.hist(
        class1,
        bins=bins,
        density=True,
        alpha=0.45,
        label="Retained (y=1)",
    )

    ax.set_xlabel(
        "Predicted retention probability"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        f"Target-domain score distributions: {protocol}"
    )

    ax.legend()

    save_figure(
        fig,
        filename,
    )


def figure_threshold_vs_mcc(
    threshold_sweep: pd.DataFrame,
    runs: pd.DataFrame,
    protocol: str,
    filename: str,
) -> None:

    g = threshold_sweep[
        threshold_sweep["protocol"]
        == protocol
    ]

    avg = (
        g.groupby("threshold")["mcc"]
        .agg(
            ["mean", "std"]
        )
        .reset_index()
    )

    protocol_runs = runs[
        runs["protocol"]
        == protocol
    ]

    locked_mean = float(
        protocol_runs[
            "locked_threshold"
        ].mean()
    )

    oracle_mean = float(
        protocol_runs[
            "oracle_threshold"
        ].mean()
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        avg["threshold"],
        avg["mean"],
        label="Mean target MCC",
    )

    ax.fill_between(
        avg["threshold"],
        avg["mean"] - avg["std"],
        avg["mean"] + avg["std"],
        alpha=0.2,
        label="±1 SD",
    )

    ax.axvline(
        locked_mean,
        linestyle="--",
        label=(
            f"Mean locked threshold "
            f"({locked_mean:.2f})"
        ),
    )

    ax.axvline(
        oracle_mean,
        linestyle=":",
        label=(
            f"Mean oracle threshold "
            f"({oracle_mean:.2f})"
        ),
    )

    ax.set_xlabel(
        "Decision threshold"
    )

    ax.set_ylabel(
        "Target-domain MCC"
    )

    ax.set_title(
        f"Threshold sensitivity: {protocol}"
    )

    ax.legend()

    save_figure(
        fig,
        filename,
    )


def figure_prevalence(
    runs: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    for protocol, g in runs.groupby(
        "protocol"
    ):

        ax.scatter(
            g["actual_prevalence"],
            g["locked_predicted_prevalence"],
            label=protocol,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect prevalence match",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel(
        "Actual target retention prevalence"
    )

    ax.set_ylabel(
        "Predicted retained proportion"
    )

    ax.set_title(
        "Zero-shot prevalence shift"
    )

    ax.legend()

    save_figure(
        fig,
        "predicted_vs_actual_prevalence",
    )


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================

def print_protocol_summary(
    runs: pd.DataFrame,
) -> None:

    banner(
        "4. CROSS-COUNTRY CALIBRATION DIAGNOSTICS"
    )

    for protocol, g in runs.groupby(
        "protocol"
    ):

        print()
        print(protocol.upper())
        print("-" * 80)

        print(
            "Actual prevalence:             "
            f"{g['actual_prevalence'].mean():.4f}"
        )

        print(
            "Locked predicted prevalence:   "
            f"{g['locked_predicted_prevalence'].mean():.4f}"
        )

        print(
            "Mean locked threshold:         "
            f"{g['locked_threshold'].mean():.4f}"
        )

        print(
            "Mean oracle threshold:         "
            f"{g['oracle_threshold'].mean():.4f}"
        )

        print(
            "Mean threshold shift:          "
            f"{g['threshold_shift'].mean():+.4f}"
        )

        print()
        print(
            "Locked MCC:                    "
            f"{g['locked_mcc'].mean():.4f} "
            f"± {g['locked_mcc'].std(ddof=1):.4f}"
        )

        print(
            "Oracle MCC [DIAGNOSTIC]:       "
            f"{g['oracle_mcc'].mean():.4f} "
            f"± {g['oracle_mcc'].std(ddof=1):.4f}"
        )

        print(
            "Recoverable MCC:               "
            f"{g['recoverable_mcc'].mean():+.4f}"
        )

        print()
        print(
            "Locked BA:                     "
            f"{g['locked_balanced_accuracy'].mean():.4f} "
            f"± {g['locked_balanced_accuracy'].std(ddof=1):.4f}"
        )

        print(
            "Oracle BA [DIAGNOSTIC]:        "
            f"{g['oracle_balanced_accuracy'].mean():.4f} "
            f"± {g['oracle_balanced_accuracy'].std(ddof=1):.4f}"
        )

        print(
            "Recoverable BA:                "
            f"{g['recoverable_balanced_accuracy'].mean():+.4f}"
        )

        print()
        print(
            "ROC-AUC:                       "
            f"{g['roc_auc'].mean():.4f}"
        )

        print(
            "PR-AUC:                        "
            f"{g['pr_auc'].mean():.4f}"
        )

        print(
            "Brier:                         "
            f"{g['brier'].mean():.4f}"
        )

        print(
            "ECE:                           "
            f"{g['ece'].mean():.4f}"
        )

        print(
            "Calibration intercept:         "
            f"{g['calibration_intercept'].mean():.4f}"
        )

        print(
            "Calibration slope:             "
            f"{g['calibration_slope'].mean():.4f}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    banner(
        "CARTOGEN — SCRIPT 77: "
        "CROSS-COUNTRY CALIBRATION AND THRESHOLD-SHIFT ANALYSIS"
    )

    print("Model root:", MODEL_ROOT)
    print("Output:    ", OUTPUT_ROOT)
    print("Protocols: ", PROTOCOLS)
    print("Seeds:     ", SEEDS)

    print()
    print(
        "NOTE: Target-domain oracle thresholds are "
        "DIAGNOSTIC ONLY and are not zero-shot results."
    )

    ensure_directories()

    verify_inputs()

    banner("2. ANALYZE 20 ZERO-SHOT RUNS")

    run_rows = []
    calibration_frames = []
    class_rows = []
    city_rows = []
    sweep_frames = []

    total = (
        len(PROTOCOLS)
        * len(SEEDS)
    )

    counter = 0

    for protocol in PROTOCOLS:

        for seed in SEEDS:

            counter += 1

            print(
                f"[{counter:02d}/{total:02d}] "
                f"{protocol} | seed {seed}"
            )

            (
                row,
                cal,
                classes,
                cities,
                sweep,
            ) = analyze_run(
                protocol,
                seed,
            )

            run_rows.append(row)
            calibration_frames.append(cal)
            class_rows.extend(classes)
            city_rows.extend(cities)
            sweep_frames.append(sweep)

    runs = pd.DataFrame(
        run_rows
    )

    calibration_bins = pd.concat(
        calibration_frames,
        ignore_index=True,
    )

    class_probabilities = pd.DataFrame(
        class_rows
    )

    city_results = pd.DataFrame(
        city_rows
    )

    threshold_sweep = pd.concat(
        sweep_frames,
        ignore_index=True,
    )

    banner("3. VERIFY LOCKED PREDICTIONS")

    mismatch = runs[
        ~runs[
            "saved_prediction_matches_threshold"
        ]
    ]

    if len(mismatch) == 0:
        print(
            "[PASS] All saved y_pred values match "
            "prob_retained >= saved threshold."
        )
    else:
        print(
            "[WARNING] Prediction/threshold mismatch "
            "detected:"
        )

        print(
            mismatch[
                [
                    "protocol",
                    "seed",
                    "locked_threshold",
                ]
            ].to_string(
                index=False
            )
        )

    summary = build_summary(
        runs
    )

    correlation_rows = (
        correlation_diagnostics(
            runs
        )
    )

    correlations = pd.DataFrame(
        correlation_rows
    )

    print_protocol_summary(
        runs
    )

    # -------------------------------------------------------------------------
    # Save tables
    # -------------------------------------------------------------------------

    banner("5. SAVE ANALYSIS TABLES")

    paths = {
        "runs":
            OUTPUT_ROOT
            / "zero_shot_calibration_runs.csv",

        "summary":
            OUTPUT_ROOT
            / "zero_shot_calibration_summary.csv",

        "threshold_shift":
            OUTPUT_ROOT
            / "threshold_shift_summary.csv",

        "calibration_bins":
            OUTPUT_ROOT
            / "calibration_bins.csv",

        "class_probabilities":
            OUTPUT_ROOT
            / "class_probability_summary.csv",

        "city":
            OUTPUT_ROOT
            / "city_calibration_metrics.csv",

        "threshold_sweep":
            OUTPUT_ROOT
            / "threshold_sweep.csv",

        "correlations":
            OUTPUT_ROOT
            / "threshold_shift_correlations.csv",
    }

    runs.to_csv(
        paths["runs"],
        index=False,
    )

    summary.to_csv(
        paths["summary"],
        index=False,
    )

    calibration_bins.to_csv(
        paths["calibration_bins"],
        index=False,
    )

    class_probabilities.to_csv(
        paths["class_probabilities"],
        index=False,
    )

    city_results.to_csv(
        paths["city"],
        index=False,
    )

    threshold_sweep.to_csv(
        paths["threshold_sweep"],
        index=False,
    )

    correlations.to_csv(
        paths["correlations"],
        index=False,
    )

    threshold_columns = [
        "protocol",
        "seed",
        "actual_prevalence",
        "locked_threshold",
        "oracle_threshold",
        "threshold_shift",
        "absolute_threshold_shift",
        "locked_predicted_prevalence",
        "oracle_predicted_prevalence",
        "locked_mcc",
        "oracle_mcc",
        "recoverable_mcc",
        "locked_balanced_accuracy",
        "oracle_balanced_accuracy",
        "recoverable_balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "brier",
        "ece",
    ]

    runs[
        threshold_columns
    ].to_csv(
        paths["threshold_shift"],
        index=False,
    )

    for name, path in paths.items():
        print(
            f"[SAVED] {name:20s} -> {path}"
        )

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------

    banner("6. GENERATE FIGURES")

    figure_threshold_shift(
        runs
    )

    print(
        "[SAVED] threshold_shift_by_seed"
    )

    figure_locked_vs_oracle(
        runs,
        metric="mcc",
        filename="mcc_locked_vs_oracle",
        ylabel="MCC",
    )

    print(
        "[SAVED] mcc_locked_vs_oracle"
    )

    figure_locked_vs_oracle(
        runs,
        metric="balanced_accuracy",
        filename="ba_locked_vs_oracle",
        ylabel="Balanced accuracy",
    )

    print(
        "[SAVED] ba_locked_vs_oracle"
    )

    figure_calibration(
        calibration_bins,
        "lt_to_pl_zero_shot",
        "calibration_lt_to_pl",
    )

    print(
        "[SAVED] calibration_lt_to_pl"
    )

    figure_calibration(
        calibration_bins,
        "pl_to_lt_zero_shot",
        "calibration_pl_to_lt",
    )

    print(
        "[SAVED] calibration_pl_to_lt"
    )

    figure_probability_distribution(
        "lt_to_pl_zero_shot",
        "probability_distribution_lt_to_pl",
    )

    print(
        "[SAVED] probability_distribution_lt_to_pl"
    )

    figure_probability_distribution(
        "pl_to_lt_zero_shot",
        "probability_distribution_pl_to_lt",
    )

    print(
        "[SAVED] probability_distribution_pl_to_lt"
    )

    figure_threshold_vs_mcc(
        threshold_sweep,
        runs,
        "lt_to_pl_zero_shot",
        "threshold_vs_mcc_lt_to_pl",
    )

    print(
        "[SAVED] threshold_vs_mcc_lt_to_pl"
    )

    figure_threshold_vs_mcc(
        threshold_sweep,
        runs,
        "pl_to_lt_zero_shot",
        "threshold_vs_mcc_pl_to_lt",
    )

    print(
        "[SAVED] threshold_vs_mcc_pl_to_lt"
    )

    figure_prevalence(
        runs
    )

    print(
        "[SAVED] predicted_vs_actual_prevalence"
    )

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------

    banner("7. SAVE FINAL REPORT")

    report = {
        "script":
            "77_cross_country_calibration_analysis.py",

        "analysis_type":
            "post-hoc diagnostic analysis",

        "models_retrained":
            False,

        "official_zero_shot_performance":
            "locked source-validation threshold",

        "oracle_threshold_status":
            (
                "diagnostic only; target labels are used "
                "and these values must not be reported as "
                "genuine zero-shot performance"
            ),

        "protocols":
            PROTOCOLS,

        "seeds":
            SEEDS,

        "n_runs":
            len(runs),

        "protocol_summary": {},

        "outputs": {
            k: str(v)
            for k, v in paths.items()
        },

        "figure_directory":
            str(FIGURE_ROOT),
    }

    for protocol, g in runs.groupby(
        "protocol"
    ):

        report[
            "protocol_summary"
        ][protocol] = {

            "actual_prevalence_mean":
                g[
                    "actual_prevalence"
                ].mean(),

            "locked_threshold_mean":
                g[
                    "locked_threshold"
                ].mean(),

            "oracle_threshold_mean":
                g[
                    "oracle_threshold"
                ].mean(),

            "threshold_shift_mean":
                g[
                    "threshold_shift"
                ].mean(),

            "locked_mcc_mean":
                g[
                    "locked_mcc"
                ].mean(),

            "locked_mcc_std":
                g[
                    "locked_mcc"
                ].std(ddof=1),

            "oracle_mcc_mean":
                g[
                    "oracle_mcc"
                ].mean(),

            "recoverable_mcc_mean":
                g[
                    "recoverable_mcc"
                ].mean(),

            "locked_balanced_accuracy_mean":
                g[
                    "locked_balanced_accuracy"
                ].mean(),

            "oracle_balanced_accuracy_mean":
                g[
                    "oracle_balanced_accuracy"
                ].mean(),

            "recoverable_balanced_accuracy_mean":
                g[
                    "recoverable_balanced_accuracy"
                ].mean(),

            "roc_auc_mean":
                g[
                    "roc_auc"
                ].mean(),

            "pr_auc_mean":
                g[
                    "pr_auc"
                ].mean(),

            "brier_mean":
                g[
                    "brier"
                ].mean(),

            "ece_mean":
                g[
                    "ece"
                ].mean(),

            "calibration_intercept_mean":
                g[
                    "calibration_intercept"
                ].mean(),

            "calibration_slope_mean":
                g[
                    "calibration_slope"
                ].mean(),
        }

    report_path = (
        OUTPUT_ROOT
        / "calibration_analysis_report.json"
    )

    save_json(
        report_path,
        report,
    )

    print(
        "Report:",
        report_path,
    )

    # -------------------------------------------------------------------------
    # Final interpretation aid
    # -------------------------------------------------------------------------

    banner("8. DIAGNOSTIC INTERPRETATION")

    for protocol, g in runs.groupby(
        "protocol"
    ):

        mcc_gain = float(
            g[
                "recoverable_mcc"
            ].mean()
        )

        ba_gain = float(
            g[
                "recoverable_balanced_accuracy"
            ].mean()
        )

        shift = float(
            g[
                "absolute_threshold_shift"
            ].mean()
        )

        print()
        print(protocol)

        print(
            f"  Mean |threshold shift|: {shift:.4f}"
        )

        print(
            f"  Mean recoverable MCC:   {mcc_gain:+.4f}"
        )

        print(
            f"  Mean recoverable BA:    {ba_gain:+.4f}"
        )

        if (
            mcc_gain >= 0.10
            and shift >= 0.10
        ):
            print(
                "  Diagnostic signal: STRONG threshold/calibration contribution."
            )

        elif mcc_gain >= 0.05:
            print(
                "  Diagnostic signal: MODERATE threshold/calibration contribution."
            )

        else:
            print(
                "  Diagnostic signal: LIMITED threshold contribution; "
                "representation/domain shift likely dominates."
            )

    banner("SCRIPT 77 COMPLETE")

    print(
        "[PASS] CROSS-COUNTRY CALIBRATION AND "
        "THRESHOLD-SHIFT ANALYSIS COMPLETE."
    )

    print()
    print(
        "IMPORTANT: oracle-threshold metrics are "
        "diagnostic upper bounds only."
    )


if __name__ == "__main__":
    main()