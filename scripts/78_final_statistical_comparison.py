"""
Script 78
=========
Final Statistical Comparison and Publication-Table Generation for CartoGen.

This script DOES NOT train any model.

It combines completed multi-run results from:
    - classical/tabular baselines
    - graph-aware baselines
    - CartoGraph-X
    - Script 77 cross-country calibration diagnostics

Main outputs:
    1. unified run-level results
    2. mean ± SD + 95% CI
    3. best / second-best methods
    4. per-protocol rankings
    5. overall model rankings
    6. paired CartoGraph-X vs baseline statistical tests
    7. effect sizes
    8. Holm multiple-comparison correction
    9. transfer-gap analysis
   10. joint-vs-single-country analysis
   11. publication-ready CSV tables
   12. statistical audit JSON

IMPORTANT
---------
Paired tests are performed ONLY when:
    - the same protocol exists,
    - the same metric exists,
    - seed identifiers can be aligned,
    - at least 5 paired observations are available.

For deterministic baselines whose 10 repeated runs are identical, the script
retains the runs but explicitly flags zero-variance comparisons.

The Script 77 target-oracle results are diagnostic only and are NOT included
as competing model results.

Author: CartoGen experimental pipeline
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.stats import (
    shapiro,
    ttest_rel,
    wilcoxon,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(r"D:\Research\CartoGen")

CROSS_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "cross_country"
)

BASELINE_ROOT = CROSS_ROOT / "baseline_models"
GRAPH_ROOT = CROSS_ROOT / "graph_aware_models"
CARTOGRAPHX_ROOT = CROSS_ROOT / "cartographx_models"
CALIBRATION_ROOT = CROSS_ROOT / "calibration_analysis"

OUTPUT_ROOT = CROSS_ROOT / "final_statistics"

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

PROTOCOL_ORDER = [
    "lt_to_lt",
    "pl_to_pl",
    "lt_to_pl_zero_shot",
    "pl_to_lt_zero_shot",
    "joint_to_lt",
    "joint_to_pl",
]

PRIMARY_METRICS = [
    "balanced_accuracy",
    "f1",
    "mcc",
    "roc_auc",
    "pr_auc",
]

HEADLINE_METRICS = [
    "balanced_accuracy",
    "mcc",
    "roc_auc",
    "pr_auc",
]

SEEDS = list(range(42, 52))

ALPHA = 0.05

MIN_PAIRED_RUNS = 5

CARTOGRAPHX_NAME = "CartoGraph-X"


# =============================================================================
# PRINTING
# =============================================================================

def banner(title: str) -> None:
    print()
    print("=" * 125)
    print(title)
    print("=" * 125)


def section(title: str) -> None:
    print()
    print("-" * 125)
    print(title)
    print("-" * 125)


# =============================================================================
# HELPERS
# =============================================================================

def normalize_name(x: str) -> str:
    return (
        str(x)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


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

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
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


def save_json(path: Path, data: dict) -> None:

    path.write_text(
        json.dumps(
            json_safe(data),
            indent=2,
        ),
        encoding="utf-8",
    )


def find_first_existing(
    candidates: List[Path],
) -> Optional[Path]:

    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p

    return None


def mean_ci95(values) -> Tuple[float, float, float, float]:

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    mean = float(np.mean(x))

    if len(x) == 1:
        return (
            mean,
            np.nan,
            np.nan,
            np.nan,
        )

    sd = float(np.std(x, ddof=1))

    half = (
        1.96
        * sd
        / math.sqrt(len(x))
    )

    return (
        mean,
        sd,
        mean - half,
        mean + half,
    )


# =============================================================================
# COLUMN DETECTION
# =============================================================================

def detect_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:

    lookup = {
        normalize_name(c): c
        for c in df.columns
    }

    for candidate in candidates:

        key = normalize_name(candidate)

        if key in lookup:
            return lookup[key]

    return None


def detect_protocol_column(
    df: pd.DataFrame,
) -> Optional[str]:

    return detect_column(
        df,
        [
            "protocol",
            "experiment",
            "scenario",
            "transfer_protocol",
        ],
    )


def detect_seed_column(
    df: pd.DataFrame,
) -> Optional[str]:

    return detect_column(
        df,
        [
            "seed",
            "random_seed",
            "run_seed",
        ],
    )


def detect_model_column(
    df: pd.DataFrame,
) -> Optional[str]:

    return detect_column(
        df,
        [
            "model",
            "model_name",
            "classifier",
            "method",
            "estimator",
            "algorithm",
        ],
    )


def detect_feature_column(
    df: pd.DataFrame,
) -> Optional[str]:

    return detect_column(
        df,
        [
            "feature_set",
            "features",
            "representation",
            "feature_group",
        ],
    )


# =============================================================================
# METRIC DETECTION
# =============================================================================

METRIC_CANDIDATES = {

    "balanced_accuracy": [
        "test_balanced_accuracy",
        "balanced_accuracy",
        "balanced_acc",
        "test_ba",
        "ba",
    ],

    "f1": [
        "test_f1",
        "f1",
        "f1_score",
        "test_f1_score",
    ],

    "mcc": [
        "test_mcc",
        "mcc",
        "matthews_corrcoef",
        "matthews_correlation_coefficient",
    ],

    "roc_auc": [
        "test_roc_auc",
        "roc_auc",
        "auc",
        "test_auc",
    ],

    "pr_auc": [
        "test_pr_auc",
        "pr_auc",
        "average_precision",
        "test_average_precision",
    ],
}


def detect_metric_columns(
    df: pd.DataFrame,
) -> Dict[str, str]:

    result = {}

    for metric, candidates in (
        METRIC_CANDIDATES.items()
    ):

        c = detect_column(
            df,
            candidates,
        )

        if c is not None:
            result[metric] = c

    return result


# =============================================================================
# STANDARDIZE GENERIC MULTI-RUN FILE
# =============================================================================

def standardize_multirun_file(
    path: Path,
    family: str,
) -> pd.DataFrame:

    raw = pd.read_csv(path)

    protocol_col = detect_protocol_column(raw)
    seed_col = detect_seed_column(raw)
    model_col = detect_model_column(raw)
    feature_col = detect_feature_column(raw)

    metric_cols = detect_metric_columns(raw)

    if protocol_col is None:
        raise KeyError(
            f"No protocol column detected in:\n{path}\n"
            f"Columns: {raw.columns.tolist()}"
        )

    if seed_col is None:
        raise KeyError(
            f"No seed column detected in:\n{path}\n"
            f"Columns: {raw.columns.tolist()}"
        )

    if len(metric_cols) == 0:
        raise KeyError(
            f"No metric columns detected in:\n{path}"
        )

    rows = []

    for _, r in raw.iterrows():

        protocol = str(
            r[protocol_col]
        )

        try:
            seed = int(
                r[seed_col]
            )
        except Exception:
            continue

        if model_col is not None:
            model = str(
                r[model_col]
            )
        else:
            model = family

        feature_set = ""

        if feature_col is not None:
            feature_set = str(
                r[feature_col]
            )

        if (
            feature_set
            and feature_set.lower() != "nan"
        ):
            display_model = (
                f"{model} [{feature_set}]"
            )
        else:
            display_model = model

        for metric, col in (
            metric_cols.items()
        ):

            try:
                value = float(r[col])
            except Exception:
                continue

            rows.append(
                {
                    "family": family,
                    "model": display_model,
                    "raw_model": model,
                    "feature_set": feature_set,
                    "protocol": protocol,
                    "seed": seed,
                    "metric": metric,
                    "value": value,
                    "source_file": str(path),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# LOAD CLASSICAL BASELINES
# =============================================================================

def load_classical_results() -> pd.DataFrame:

    section("LOAD CLASSICAL BASELINES")

    candidates = [
        BASELINE_ROOT / "all_runs_metrics.csv",
        BASELINE_ROOT / "all_runs_test_results.csv",
        BASELINE_ROOT / "multi_run_results.csv",
        BASELINE_ROOT / "all_runs.csv",
    ]

    path = find_first_existing(
        candidates
    )

    if path is None:

        print(
            "[WARNING] No classical run-level file found."
        )

        return pd.DataFrame()

    print("[FOUND]", path)

    df = standardize_multirun_file(
        path,
        "Classical",
    )

    print(
        f"[OK] {len(df):,} long-format metric rows"
    )

    return df


# =============================================================================
# LOAD GRAPH-AWARE RESULTS
# =============================================================================

def load_graph_results() -> pd.DataFrame:

    section("LOAD GRAPH-AWARE BASELINES")

    candidates = [
        GRAPH_ROOT / "all_runs_test_results.csv",
        GRAPH_ROOT / "all_runs_metrics.csv",
        GRAPH_ROOT / "multi_run_results.csv",
        GRAPH_ROOT / "all_runs.csv",
    ]

    path = find_first_existing(
        candidates
    )

    if path is None:

        print(
            "[WARNING] No graph-aware run-level file found."
        )

        return pd.DataFrame()

    print("[FOUND]", path)

    df = standardize_multirun_file(
        path,
        "Graph-aware",
    )

    print(
        f"[OK] {len(df):,} long-format metric rows"
    )

    return df


# =============================================================================
# LOAD CARTOGRAPH-X
# =============================================================================

def load_cartographx_results() -> pd.DataFrame:

    section("LOAD CARTOGRAPH-X")

    path = (
        CARTOGRAPHX_ROOT
        / "full_all_runs.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    raw = pd.read_csv(path)

    required = {
        "protocol",
        "seed",
    }

    missing = required - set(raw.columns)

    if missing:
        raise KeyError(
            f"Missing CartoGraph-X columns: "
            f"{sorted(missing)}"
        )

    metric_cols = detect_metric_columns(
        raw
    )

    rows = []

    for _, r in raw.iterrows():

        protocol = str(
            r["protocol"]
        )

        seed = int(
            r["seed"]
        )

        for metric, col in (
            metric_cols.items()
        ):

            value = float(
                r[col]
            )

            rows.append(
                {
                    "family": "Proposed",
                    "model": CARTOGRAPHX_NAME,
                    "raw_model": CARTOGRAPHX_NAME,
                    "feature_set": "",
                    "protocol": protocol,
                    "seed": seed,
                    "metric": metric,
                    "value": value,
                    "source_file": str(path),
                }
            )

    df = pd.DataFrame(rows)

    print(
        f"[OK] {len(df):,} long-format metric rows"
    )

    return df


# =============================================================================
# CLEAN PROTOCOLS
# =============================================================================

def normalize_protocol(
    protocol: str,
) -> str:

    p = normalize_name(protocol)

    aliases = {

        "lt_lt": "lt_to_lt",
        "lithuania_to_lithuania": "lt_to_lt",

        "pl_pl": "pl_to_pl",
        "poland_to_poland": "pl_to_pl",

        "lt_pl": "lt_to_pl_zero_shot",
        "lt_to_pl": "lt_to_pl_zero_shot",
        "lithuania_to_poland":
            "lt_to_pl_zero_shot",

        "pl_lt": "pl_to_lt_zero_shot",
        "pl_to_lt": "pl_to_lt_zero_shot",
        "poland_to_lithuania":
            "pl_to_lt_zero_shot",

        "joint_lt": "joint_to_lt",
        "joint_lithuania": "joint_to_lt",

        "joint_pl": "joint_to_pl",
        "joint_poland": "joint_to_pl",
    }

    return aliases.get(
        p,
        p,
    )


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def build_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    grouped = data.groupby(
        [
            "family",
            "model",
            "protocol",
            "metric",
        ],
        dropna=False,
    )

    for (
        family,
        model,
        protocol,
        metric,
    ), g in grouped:

        x = g["value"].to_numpy(
            dtype=float
        )

        x = x[
            np.isfinite(x)
        ]

        if len(x) == 0:
            continue

        mean, sd, lo, hi = mean_ci95(
            x
        )

        rows.append(
            {
                "family": family,
                "model": model,
                "protocol": protocol,
                "metric": metric,
                "n_runs": len(x),
                "mean": mean,
                "std": sd,
                "median":
                    float(np.median(x)),
                "min":
                    float(np.min(x)),
                "max":
                    float(np.max(x)),
                "ci95_lower": lo,
                "ci95_upper": hi,
                "zero_variance":
                    bool(
                        len(x) > 1
                        and np.std(
                            x,
                            ddof=1,
                        ) < 1e-12
                    ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# RANKING
# =============================================================================

def add_rankings(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    d = summary.copy()

    d["rank"] = np.nan

    for (
        protocol,
        metric,
    ), idx in d.groupby(
        [
            "protocol",
            "metric",
        ]
    ).groups.items():

        values = d.loc[
            idx,
            "mean",
        ]

        d.loc[
            idx,
            "rank",
        ] = values.rank(
            ascending=False,
            method="min",
        )

    return d


def overall_ranking(
    ranked: pd.DataFrame,
) -> pd.DataFrame:

    d = ranked[
        ranked["metric"].isin(
            HEADLINE_METRICS
        )
    ].copy()

    rows = []

    for (
        family,
        model,
    ), g in d.groupby(
        [
            "family",
            "model",
        ]
    ):

        rows.append(
            {
                "family": family,
                "model": model,
                "comparisons":
                    len(g),
                "mean_rank":
                    float(
                        g["rank"].mean()
                    ),
                "median_rank":
                    float(
                        g["rank"].median()
                    ),
                "rank_1_count":
                    int(
                        (g["rank"] == 1)
                        .sum()
                    ),
                "rank_top2_count":
                    int(
                        (g["rank"] <= 2)
                        .sum()
                    ),
            }
        )

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(
            [
                "mean_rank",
                "rank_1_count",
            ],
            ascending=[
                True,
                False,
            ],
        )

        out["overall_rank"] = (
            np.arange(
                1,
                len(out) + 1,
            )
        )

    return out


# =============================================================================
# BEST AND SECOND-BEST
# =============================================================================

def best_method_table(
    ranked: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        protocol,
        metric,
    ), g in ranked.groupby(
        [
            "protocol",
            "metric",
        ]
    ):

        g = g.sort_values(
            [
                "mean",
                "std",
            ],
            ascending=[
                False,
                True,
            ],
        )

        if len(g) == 0:
            continue

        best = g.iloc[0]

        second = (
            g.iloc[1]
            if len(g) > 1
            else None
        )

        row = {
            "protocol": protocol,
            "metric": metric,

            "best_model":
                best["model"],

            "best_family":
                best["family"],

            "best_mean":
                best["mean"],

            "best_std":
                best["std"],

            "best_ci95_lower":
                best["ci95_lower"],

            "best_ci95_upper":
                best["ci95_upper"],
        }

        if second is not None:

            row.update(
                {
                    "second_model":
                        second["model"],

                    "second_family":
                        second["family"],

                    "second_mean":
                        second["mean"],

                    "second_std":
                        second["std"],

                    "absolute_gap":
                        best["mean"]
                        - second["mean"],
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# EFFECT SIZE
# =============================================================================

def paired_cohens_d(
    x: np.ndarray,
    y: np.ndarray,
) -> float:

    diff = x - y

    if len(diff) < 2:
        return np.nan

    sd = np.std(
        diff,
        ddof=1,
    )

    if sd < 1e-12:

        if abs(
            np.mean(diff)
        ) < 1e-12:
            return 0.0

        return (
            np.inf
            if np.mean(diff) > 0
            else -np.inf
        )

    return float(
        np.mean(diff)
        / sd
    )


def rank_biserial_from_diff(
    diff: np.ndarray,
) -> float:

    d = np.asarray(
        diff,
        dtype=float,
    )

    d = d[
        np.isfinite(d)
    ]

    d = d[
        np.abs(d) > 1e-12
    ]

    if len(d) == 0:
        return 0.0

    ranks = pd.Series(
        np.abs(d)
    ).rank(
        method="average"
    ).to_numpy()

    positive = ranks[
        d > 0
    ].sum()

    negative = ranks[
        d < 0
    ].sum()

    denominator = (
        positive
        + negative
    )

    if denominator == 0:
        return 0.0

    return float(
        (positive - negative)
        / denominator
    )


# =============================================================================
# HOLM CORRECTION
# =============================================================================

def holm_adjust(
    p_values: List[float],
) -> List[float]:

    p = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full(
        len(p),
        np.nan,
        dtype=float,
    )

    valid = np.where(
        np.isfinite(p)
    )[0]

    if len(valid) == 0:
        return adjusted.tolist()

    order = valid[
        np.argsort(
            p[valid]
        )
    ]

    m = len(order)

    previous = 0.0

    for rank, idx in enumerate(
        order
    ):

        multiplier = (
            m - rank
        )

        value = min(
            1.0,
            p[idx]
            * multiplier,
        )

        value = max(
            value,
            previous,
        )

        adjusted[idx] = value
        previous = value

    return adjusted.tolist()


# =============================================================================
# PAIRED TEST
# =============================================================================

def paired_test(
    proposed: pd.DataFrame,
    baseline: pd.DataFrame,
) -> Optional[dict]:

    a = proposed[
        [
            "seed",
            "value",
        ]
    ].rename(
        columns={
            "value":
                "proposed_value"
        }
    )

    b = baseline[
        [
            "seed",
            "value",
        ]
    ].rename(
        columns={
            "value":
                "baseline_value"
        }
    )

    paired = a.merge(
        b,
        on="seed",
        how="inner",
    )

    paired = paired[
        np.isfinite(
            paired["proposed_value"]
        )
        & np.isfinite(
            paired["baseline_value"]
        )
    ]

    if len(paired) < MIN_PAIRED_RUNS:
        return None

    x = paired[
        "proposed_value"
    ].to_numpy(dtype=float)

    y = paired[
        "baseline_value"
    ].to_numpy(dtype=float)

    diff = x - y

    mean_diff = float(
        np.mean(diff)
    )

    if len(diff) > 1:

        sd_diff = float(
            np.std(
                diff,
                ddof=1,
            )
        )

        half = (
            1.96
            * sd_diff
            / math.sqrt(len(diff))
        )

        ci_lower = (
            mean_diff - half
        )

        ci_upper = (
            mean_diff + half
        )

    else:

        sd_diff = np.nan
        ci_lower = np.nan
        ci_upper = np.nan

    zero_variance_diff = bool(
        len(diff) > 1
        and np.std(
            diff,
            ddof=1,
        ) < 1e-12
    )

    # ---------------------------------------------------------
    # Normality of paired differences
    # ---------------------------------------------------------

    if (
        len(diff) >= 3
        and not zero_variance_diff
    ):

        try:
            shapiro_result = shapiro(
                diff
            )

            shapiro_p = float(
                shapiro_result.pvalue
            )

        except Exception:
            shapiro_p = np.nan

    else:
        shapiro_p = np.nan

    # ---------------------------------------------------------
    # Statistical test
    # ---------------------------------------------------------

    if zero_variance_diff:

        if abs(mean_diff) < 1e-12:

            test_name = (
                "identical_paired_values"
            )

            statistic = 0.0
            p_value = 1.0

        else:

            test_name = (
                "constant_nonzero_difference"
            )

            statistic = np.nan
            p_value = 0.0

    elif (
        np.isfinite(shapiro_p)
        and shapiro_p >= ALPHA
    ):

        result = ttest_rel(
            x,
            y,
        )

        test_name = (
            "paired_t_test"
        )

        statistic = float(
            result.statistic
        )

        p_value = float(
            result.pvalue
        )

    else:

        try:

            result = wilcoxon(
                diff,
                zero_method="wilcox",
                alternative="two-sided",
            )

            test_name = (
                "wilcoxon_signed_rank"
            )

            statistic = float(
                result.statistic
            )

            p_value = float(
                result.pvalue
            )

        except Exception:

            test_name = (
                "wilcoxon_failed"
            )

            statistic = np.nan
            p_value = np.nan

    return {
        "n_pairs": len(paired),

        "proposed_mean":
            float(np.mean(x)),

        "baseline_mean":
            float(np.mean(y)),

        "mean_difference":
            mean_diff,

        "difference_std":
            sd_diff,

        "difference_ci95_lower":
            ci_lower,

        "difference_ci95_upper":
            ci_upper,

        "shapiro_p":
            shapiro_p,

        "test":
            test_name,

        "statistic":
            statistic,

        "p_value":
            p_value,

        "paired_cohens_d":
            paired_cohens_d(
                x,
                y,
            ),

        "rank_biserial":
            rank_biserial_from_diff(
                diff
            ),

        "proposed_wins":
            int(
                np.sum(diff > 0)
            ),

        "ties":
            int(
                np.sum(
                    np.abs(diff)
                    <= 1e-12
                )
            ),

        "baseline_wins":
            int(
                np.sum(diff < 0)
            ),

        "zero_variance_difference":
            zero_variance_diff,
    }


# =============================================================================
# COMPARE CARTOGRAPH-X TO ALL BASELINES
# =============================================================================

def statistical_comparisons(
    data: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    proposed = data[
        data["model"]
        == CARTOGRAPHX_NAME
    ]

    baselines = data[
        data["model"]
        != CARTOGRAPHX_NAME
    ]

    for protocol in PROTOCOL_ORDER:

        for metric in HEADLINE_METRICS:

            p = proposed[
                (
                    proposed["protocol"]
                    == protocol
                )
                & (
                    proposed["metric"]
                    == metric
                )
            ]

            if len(p) == 0:
                continue

            candidates = baselines[
                (
                    baselines["protocol"]
                    == protocol
                )
                & (
                    baselines["metric"]
                    == metric
                )
            ]

            for (
                family,
                model,
            ), b in candidates.groupby(
                [
                    "family",
                    "model",
                ]
            ):

                result = paired_test(
                    p,
                    b,
                )

                if result is None:
                    continue

                row = {
                    "protocol":
                        protocol,

                    "metric":
                        metric,

                    "proposed_model":
                        CARTOGRAPHX_NAME,

                    "baseline_family":
                        family,

                    "baseline_model":
                        model,
                }

                row.update(result)

                rows.append(row)

    out = pd.DataFrame(rows)

    if len(out) == 0:
        return out

    # Holm correction within each protocol/metric family.
    out["p_holm"] = np.nan

    for (
        protocol,
        metric,
    ), idx in out.groupby(
        [
            "protocol",
            "metric",
        ]
    ).groups.items():

        adjusted = holm_adjust(
            out.loc[
                idx,
                "p_value",
            ].tolist()
        )

        out.loc[
            idx,
            "p_holm",
        ] = adjusted

    out["significant_0_05"] = (
        out["p_holm"]
        < ALPHA
    )

    return out


# =============================================================================
# STRONGEST BASELINE COMPARISON
# =============================================================================

def strongest_baseline_tests(
    comparisons: pd.DataFrame,
) -> pd.DataFrame:

    if len(comparisons) == 0:
        return comparisons.copy()

    rows = []

    for (
        protocol,
        metric,
    ), g in comparisons.groupby(
        [
            "protocol",
            "metric",
        ]
    ):

        # strongest baseline by mean performance
        best = g.sort_values(
            "baseline_mean",
            ascending=False,
        ).iloc[0]

        rows.append(
            best.to_dict()
        )

    return pd.DataFrame(rows)


# =============================================================================
# TRANSFER GAP
# =============================================================================

def extract_model_runs(
    data: pd.DataFrame,
    model: str,
    protocol: str,
    metric: str,
) -> pd.DataFrame:

    return data[
        (
            data["model"] == model
        )
        & (
            data["protocol"] == protocol
        )
        & (
            data["metric"] == metric
        )
    ][
        [
            "seed",
            "value",
        ]
    ].copy()


def transfer_gap_analysis(
    data: pd.DataFrame,
) -> pd.DataFrame:

    mappings = [
        (
            "LT_to_PL",
            "lt_to_lt",
            "lt_to_pl_zero_shot",
        ),
        (
            "PL_to_LT",
            "pl_to_pl",
            "pl_to_lt_zero_shot",
        ),
    ]

    rows = []

    models = sorted(
        data["model"]
        .dropna()
        .unique()
    )

    for model in models:

        for metric in HEADLINE_METRICS:

            for (
                direction,
                in_domain,
                zero_shot,
            ) in mappings:

                a = extract_model_runs(
                    data,
                    model,
                    in_domain,
                    metric,
                ).rename(
                    columns={
                        "value":
                            "in_domain"
                    }
                )

                b = extract_model_runs(
                    data,
                    model,
                    zero_shot,
                    metric,
                ).rename(
                    columns={
                        "value":
                            "zero_shot"
                    }
                )

                paired = a.merge(
                    b,
                    on="seed",
                    how="inner",
                )

                if len(paired) == 0:
                    continue

                paired["gap"] = (
                    paired["zero_shot"]
                    - paired["in_domain"]
                )

                x = paired[
                    "gap"
                ].to_numpy(dtype=float)

                mean, sd, lo, hi = mean_ci95(
                    x
                )

                rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "direction": direction,
                        "n_pairs": len(paired),
                        "in_domain_mean":
                            float(
                                paired[
                                    "in_domain"
                                ].mean()
                            ),
                        "zero_shot_mean":
                            float(
                                paired[
                                    "zero_shot"
                                ].mean()
                            ),
                        "transfer_gap_mean":
                            mean,
                        "transfer_gap_std":
                            sd,
                        "transfer_gap_ci95_lower":
                            lo,
                        "transfer_gap_ci95_upper":
                            hi,
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# JOINT VS SINGLE COUNTRY
# =============================================================================

def joint_training_analysis(
    data: pd.DataFrame,
) -> pd.DataFrame:

    mappings = [
        (
            "Lithuania",
            "lt_to_lt",
            "joint_to_lt",
        ),
        (
            "Poland",
            "pl_to_pl",
            "joint_to_pl",
        ),
    ]

    rows = []

    models = sorted(
        data["model"]
        .dropna()
        .unique()
    )

    for model in models:

        for metric in HEADLINE_METRICS:

            for (
                country,
                single_protocol,
                joint_protocol,
            ) in mappings:

                single = extract_model_runs(
                    data,
                    model,
                    single_protocol,
                    metric,
                ).rename(
                    columns={
                        "value":
                            "single_country"
                    }
                )

                joint = extract_model_runs(
                    data,
                    model,
                    joint_protocol,
                    metric,
                ).rename(
                    columns={
                        "value":
                            "joint_country"
                    }
                )

                paired = single.merge(
                    joint,
                    on="seed",
                    how="inner",
                )

                if len(paired) == 0:
                    continue

                diff = (
                    paired[
                        "joint_country"
                    ]
                    - paired[
                        "single_country"
                    ]
                ).to_numpy(dtype=float)

                mean, sd, lo, hi = mean_ci95(
                    diff
                )

                rows.append(
                    {
                        "country":
                            country,

                        "model":
                            model,

                        "metric":
                            metric,

                        "n_pairs":
                            len(paired),

                        "single_country_mean":
                            float(
                                paired[
                                    "single_country"
                                ].mean()
                            ),

                        "joint_country_mean":
                            float(
                                paired[
                                    "joint_country"
                                ].mean()
                            ),

                        "joint_minus_single":
                            mean,

                        "difference_std":
                            sd,

                        "ci95_lower":
                            lo,

                        "ci95_upper":
                            hi,
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# PUBLICATION TABLE
# =============================================================================

def formatted_performance_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    d = summary[
        summary["metric"].isin(
            HEADLINE_METRICS
        )
    ].copy()

    d["mean_sd"] = d.apply(
        lambda r:
        (
            f"{r['mean']:.4f} "
            f"± {r['std']:.4f}"
            if np.isfinite(
                r["std"]
            )
            else
            f"{r['mean']:.4f}"
        ),
        axis=1,
    )

    table = d.pivot_table(
        index=[
            "family",
            "model",
        ],
        columns=[
            "protocol",
            "metric",
        ],
        values="mean_sd",
        aggfunc="first",
    )

    return table.reset_index()


def protocol_publication_tables(
    summary: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:

    result = {}

    for protocol in PROTOCOL_ORDER:

        g = summary[
            (
                summary["protocol"]
                == protocol
            )
            & (
                summary["metric"]
                .isin(
                    HEADLINE_METRICS
                )
            )
        ].copy()

        if len(g) == 0:
            continue

        g["mean_sd"] = g.apply(
            lambda r:
            (
                f"{r['mean']:.4f} "
                f"± {r['std']:.4f}"
                if np.isfinite(
                    r["std"]
                )
                else
                f"{r['mean']:.4f}"
            ),
            axis=1,
        )

        table = g.pivot_table(
            index=[
                "family",
                "model",
            ],
            columns="metric",
            values="mean_sd",
            aggfunc="first",
        ).reset_index()

        desired = [
            "family",
            "model",
            "balanced_accuracy",
            "mcc",
            "roc_auc",
            "pr_auc",
        ]

        existing = [
            c
            for c in desired
            if c in table.columns
        ]

        table = table[
            existing
        ]

        result[protocol] = table

    return result


# =============================================================================
# SCRIPT 77 DIAGNOSTICS
# =============================================================================

def load_calibration_diagnostics() -> pd.DataFrame:

    path = (
        CALIBRATION_ROOT
        / "zero_shot_calibration_runs.csv"
    )

    if not path.exists():
        print(
            "[INFO] Script 77 diagnostic file not found."
        )
        return pd.DataFrame()

    d = pd.read_csv(path)

    print(
        f"[OK] Script 77 diagnostics: {len(d)} runs"
    )

    return d


# =============================================================================
# AUDIT
# =============================================================================

def build_audit(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> dict:

    audit = {
        "protocols": {},
        "warnings": [],
    }

    for protocol in PROTOCOL_ORDER:

        p = data[
            data["protocol"]
            == protocol
        ]

        if len(p) == 0:
            audit["warnings"].append(
                f"No data for protocol {protocol}"
            )
            continue

        models = sorted(
            p["model"]
            .unique()
            .tolist()
        )

        audit[
            "protocols"
        ][protocol] = {
            "models": models,
            "n_models": len(models),
        }

    carto = summary[
        summary["model"]
        == CARTOGRAPHX_NAME
    ]

    for _, r in carto.iterrows():

        if r["n_runs"] != 10:

            audit["warnings"].append(
                f"CartoGraph-X {r['protocol']} "
                f"{r['metric']} has "
                f"{r['n_runs']} runs instead of 10."
            )

    zero_var = summary[
        summary["zero_variance"]
        == True
    ]

    audit[
        "zero_variance_results"
    ] = int(
        len(zero_var)
    )

    audit[
        "paired_comparisons"
    ] = int(
        len(comparisons)
    )

    if len(comparisons) > 0:

        audit[
            "significant_after_holm"
        ] = int(
            comparisons[
                "significant_0_05"
            ].sum()
        )

    else:

        audit[
            "significant_after_holm"
        ] = 0

    return audit


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    banner(
        "CARTOGEN — SCRIPT 78: "
        "FINAL STATISTICAL COMPARISON"
    )

    print("Input root: ", CROSS_ROOT)
    print("Output:     ", OUTPUT_ROOT)
    print("Seeds:      ", SEEDS)
    print("Alpha:      ", ALPHA)

    banner("1. LOAD COMPLETED EXPERIMENTS")

    frames = []

    classical = load_classical_results()

    if len(classical) > 0:
        frames.append(classical)

    graph = load_graph_results()

    if len(graph) > 0:
        frames.append(graph)

    cartographx = load_cartographx_results()

    frames.append(
        cartographx
    )

    if len(frames) == 0:
        raise RuntimeError(
            "No experimental results were loaded."
        )

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    data["protocol"] = (
        data["protocol"]
        .map(normalize_protocol)
    )

    data = data[
        data["protocol"]
        .isin(PROTOCOL_ORDER)
    ].copy()

    data = data[
        data["metric"]
        .isin(PRIMARY_METRICS)
    ].copy()

    data["seed"] = pd.to_numeric(
        data["seed"],
        errors="coerce",
    )

    data["value"] = pd.to_numeric(
        data["value"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "seed",
            "value",
        ]
    )

    data["seed"] = (
        data["seed"]
        .astype(int)
    )

    data = data[
        data["seed"]
        .isin(SEEDS)
    ]

    print()
    print(
        "Unified metric rows:",
        f"{len(data):,}",
    )

    print(
        "Models:",
        data["model"]
        .nunique(),
    )

    print(
        "Protocols:",
        sorted(
            data["protocol"]
            .unique()
            .tolist()
        ),
    )

    print(
        "Metrics:",
        sorted(
            data["metric"]
            .unique()
            .tolist()
        ),
    )

    # -------------------------------------------------------------------------
    # Save unified run data
    # -------------------------------------------------------------------------

    unified_path = (
        OUTPUT_ROOT
        / "unified_run_level_results.csv"
    )

    data.to_csv(
        unified_path,
        index=False,
    )

    print(
        "[SAVED]",
        unified_path,
    )

    # -------------------------------------------------------------------------
    # Aggregate
    # -------------------------------------------------------------------------

    banner("2. MEAN ± SD AND 95% CONFIDENCE INTERVALS")

    summary = build_summary(
        data
    )

    ranked = add_rankings(
        summary
    )

    summary_path = (
        OUTPUT_ROOT
        / "model_performance_summary.csv"
    )

    ranked.to_csv(
        summary_path,
        index=False,
    )

    print(
        "[SAVED]",
        summary_path,
    )

    # CartoGraph-X headline
    section("CARTOGRAPH-X HEADLINE RESULTS")

    cx = ranked[
        (
            ranked["model"]
            == CARTOGRAPHX_NAME
        )
        & (
            ranked["metric"]
            .isin(
                HEADLINE_METRICS
            )
        )
    ].copy()

    cx["protocol_order"] = (
        cx["protocol"]
        .map(
            {
                p: i
                for i, p in enumerate(
                    PROTOCOL_ORDER
                )
            }
        )
    )

    cx = cx.sort_values(
        [
            "protocol_order",
            "metric",
        ]
    )

    print(
        cx[
            [
                "protocol",
                "metric",
                "mean",
                "std",
                "ci95_lower",
                "ci95_upper",
                "rank",
            ]
        ].to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Ranking
    # -------------------------------------------------------------------------

    banner("3. MODEL RANKING")

    overall = overall_ranking(
        ranked
    )

    overall_path = (
        OUTPUT_ROOT
        / "overall_model_ranking.csv"
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    print(
        overall.to_string(
            index=False
        )
    )

    print(
        "\n[SAVED]",
        overall_path,
    )

    # -------------------------------------------------------------------------
    # Best and second best
    # -------------------------------------------------------------------------

    banner("4. BEST / SECOND-BEST METHODS")

    best = best_method_table(
        ranked
    )

    best_path = (
        OUTPUT_ROOT
        / "best_second_best_methods.csv"
    )

    best.to_csv(
        best_path,
        index=False,
    )

    print(
        best[
            best["metric"]
            .isin(
                HEADLINE_METRICS
            )
        ].to_string(
            index=False
        )
    )

    print(
        "\n[SAVED]",
        best_path,
    )

    # -------------------------------------------------------------------------
    # Statistical testing
    # -------------------------------------------------------------------------

    banner(
        "5. PAIRED STATISTICAL COMPARISONS"
    )

    comparisons = statistical_comparisons(
        data
    )

    comparison_path = (
        OUTPUT_ROOT
        / "cartographx_vs_all_baselines_statistics.csv"
    )

    comparisons.to_csv(
        comparison_path,
        index=False,
    )

    if len(comparisons) == 0:

        print(
            "[WARNING] No paired comparisons could be constructed."
        )

    else:

        print(
            "Paired comparisons:",
            len(comparisons),
        )

        print(
            "Significant after Holm:",
            int(
                comparisons[
                    "significant_0_05"
                ].sum()
            ),
        )

    print(
        "[SAVED]",
        comparison_path,
    )

    # -------------------------------------------------------------------------
    # Strongest baseline only
    # -------------------------------------------------------------------------

    banner(
        "6. CARTOGRAPH-X VS STRONGEST BASELINE"
    )

    strongest = strongest_baseline_tests(
        comparisons
    )

    strongest_path = (
        OUTPUT_ROOT
        / "cartographx_vs_strongest_baseline.csv"
    )

    strongest.to_csv(
        strongest_path,
        index=False,
    )

    if len(strongest) > 0:

        display_cols = [
            "protocol",
            "metric",
            "baseline_model",
            "proposed_mean",
            "baseline_mean",
            "mean_difference",
            "test",
            "p_value",
            "p_holm",
            "paired_cohens_d",
            "rank_biserial",
            "proposed_wins",
            "baseline_wins",
        ]

        display_cols = [
            c
            for c in display_cols
            if c in strongest.columns
        ]

        print(
            strongest[
                display_cols
            ].to_string(
                index=False
            )
        )

    print(
        "\n[SAVED]",
        strongest_path,
    )

    # -------------------------------------------------------------------------
    # Transfer gaps
    # -------------------------------------------------------------------------

    banner(
        "7. CROSS-COUNTRY TRANSFER GAP"
    )

    transfer = transfer_gap_analysis(
        data
    )

    transfer_path = (
        OUTPUT_ROOT
        / "cross_country_transfer_gap.csv"
    )

    transfer.to_csv(
        transfer_path,
        index=False,
    )

    carto_transfer = transfer[
        transfer["model"]
        == CARTOGRAPHX_NAME
    ]

    if len(carto_transfer) > 0:

        print(
            carto_transfer.to_string(
                index=False
            )
        )

    print(
        "\n[SAVED]",
        transfer_path,
    )

    # -------------------------------------------------------------------------
    # Joint vs country-specific
    # -------------------------------------------------------------------------

    banner(
        "8. JOINT VS SINGLE-COUNTRY TRAINING"
    )

    joint = joint_training_analysis(
        data
    )

    joint_path = (
        OUTPUT_ROOT
        / "joint_vs_single_country.csv"
    )

    joint.to_csv(
        joint_path,
        index=False,
    )

    carto_joint = joint[
        joint["model"]
        == CARTOGRAPHX_NAME
    ]

    if len(carto_joint) > 0:

        print(
            carto_joint.to_string(
                index=False
            )
        )

    print(
        "\n[SAVED]",
        joint_path,
    )

    # -------------------------------------------------------------------------
    # Script 77 calibration diagnostics
    # -------------------------------------------------------------------------

    banner(
        "9. SCRIPT 77 CALIBRATION DIAGNOSTICS"
    )

    calibration = (
        load_calibration_diagnostics()
    )

    calibration_copy_path = (
        OUTPUT_ROOT
        / "zero_shot_calibration_diagnostics.csv"
    )

    if len(calibration) > 0:

        calibration.to_csv(
            calibration_copy_path,
            index=False,
        )

        diagnostic_cols = [
            "protocol",
            "seed",
            "actual_prevalence",
            "locked_threshold",
            "oracle_threshold",
            "threshold_shift",
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

        diagnostic_cols = [
            c
            for c in diagnostic_cols
            if c in calibration.columns
        ]

        diagnostic_summary = (
            calibration[
                diagnostic_cols
            ]
            .groupby(
                "protocol"
            )
            .mean(
                numeric_only=True
            )
            .reset_index()
        )

        diagnostic_summary_path = (
            OUTPUT_ROOT
            / "zero_shot_calibration_diagnostic_summary.csv"
        )

        diagnostic_summary.to_csv(
            diagnostic_summary_path,
            index=False,
        )

        print(
            diagnostic_summary.to_string(
                index=False
            )
        )

        print(
            "\n[SAVED]",
            diagnostic_summary_path,
        )

    # -------------------------------------------------------------------------
    # Publication tables
    # -------------------------------------------------------------------------

    banner(
        "10. PUBLICATION TABLES"
    )

    publication = formatted_performance_table(
        summary
    )

    publication_path = (
        OUTPUT_ROOT
        / "publication_all_protocols_mean_sd.csv"
    )

    publication.to_csv(
        publication_path,
        index=False,
    )

    print(
        "[SAVED]",
        publication_path,
    )

    protocol_tables = (
        protocol_publication_tables(
            summary
        )
    )

    for protocol, table in (
        protocol_tables.items()
    ):

        path = (
            OUTPUT_ROOT
            / f"publication_{protocol}.csv"
        )

        table.to_csv(
            path,
            index=False,
        )

        print(
            "[SAVED]",
            path,
        )

    # -------------------------------------------------------------------------
    # Compact CartoGraph-X table
    # -------------------------------------------------------------------------

    carto_summary = summary[
        (
            summary["model"]
            == CARTOGRAPHX_NAME
        )
        & (
            summary["metric"]
            .isin(
                HEADLINE_METRICS
            )
        )
    ].copy()

    carto_summary["mean_sd"] = (
        carto_summary.apply(
            lambda r:
            f"{r['mean']:.4f} ± {r['std']:.4f}",
            axis=1,
        )
    )

    carto_table = (
        carto_summary.pivot_table(
            index="protocol",
            columns="metric",
            values="mean_sd",
            aggfunc="first",
        )
        .reset_index()
    )

    carto_table["protocol_order"] = (
        carto_table["protocol"]
        .map(
            {
                p: i
                for i, p in enumerate(
                    PROTOCOL_ORDER
                )
            }
        )
    )

    carto_table = (
        carto_table
        .sort_values(
            "protocol_order"
        )
        .drop(
            columns=[
                "protocol_order"
            ]
        )
    )

    carto_table_path = (
        OUTPUT_ROOT
        / "publication_cartographx_main_results.csv"
    )

    carto_table.to_csv(
        carto_table_path,
        index=False,
    )

    print(
        "[SAVED]",
        carto_table_path,
    )

    # -------------------------------------------------------------------------
    # Audit
    # -------------------------------------------------------------------------

    banner(
        "11. STATISTICAL AUDIT"
    )

    audit = build_audit(
        data,
        summary,
        comparisons,
    )

    audit.update(
        {
            "script":
                "78_final_statistical_comparison.py",

            "models_retrained":
                False,

            "seeds":
                SEEDS,

            "primary_metrics":
                PRIMARY_METRICS,

            "headline_metrics":
                HEADLINE_METRICS,

            "alpha":
                ALPHA,

            "multiple_comparison_correction":
                "Holm",

            "normality_test":
                "Shapiro-Wilk on paired differences",

            "parametric_test":
                "paired t-test",

            "nonparametric_test":
                "Wilcoxon signed-rank",

            "effect_sizes":
                [
                    "paired Cohen's d",
                    "rank-biserial correlation",
                ],

            "oracle_threshold_policy":
                (
                    "Script 77 target-oracle results "
                    "are diagnostic only and are excluded "
                    "from competitive model ranking."
                ),
        }
    )

    audit_path = (
        OUTPUT_ROOT
        / "final_statistical_audit.json"
    )

    save_json(
        audit_path,
        audit,
    )

    print(
        "Protocols analyzed:",
        len(
            audit["protocols"]
        ),
    )

    print(
        "Paired comparisons:",
        audit[
            "paired_comparisons"
        ],
    )

    print(
        "Significant after Holm:",
        audit[
            "significant_after_holm"
        ],
    )

    print(
        "Zero-variance aggregate results:",
        audit[
            "zero_variance_results"
        ],
    )

    if audit["warnings"]:

        print()
        print("WARNINGS:")

        for w in audit["warnings"]:
            print(" -", w)

    else:

        print(
            "[PASS] No major statistical audit warnings."
        )

    print(
        "\n[SAVED]",
        audit_path,
    )

    # -------------------------------------------------------------------------
    # Final status
    # -------------------------------------------------------------------------

    banner(
        "SCRIPT 78 COMPLETE"
    )

    print(
        "[PASS] FINAL MULTI-RUN STATISTICAL "
        "COMPARISON COMPLETE."
    )

    print()
    print(
        "No model was retrained or modified."
    )

    print()
    print(
        "Primary output directory:"
    )

    print(
        OUTPUT_ROOT
    )


if __name__ == "__main__":
    main()