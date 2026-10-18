"""
Script 73
=========
Cross-country transfer and domain-generalization analysis for CartoGen /
CartoGraph-X.

Purpose
-------
This script DOES NOT train models.

It analyses completed experimental results from:
    Script 66 : classical tabular baselines
    Script 71 : CartoGraph-X

Primary scientific questions
----------------------------
1. How well does each country-specific model perform in-domain?
2. How much performance is lost under zero-shot geographic transfer?
3. Does joint-country training improve target-country performance?
4. Is transfer asymmetric between Lithuania and Poland?
5. Does CartoGraph-X reduce the cross-country generalization gap relative
   to classical baselines?
6. Are the conclusions stable across seeds when final multi-run results
   become available?

Supported protocols
-------------------
lt_to_lt
pl_to_pl
lt_to_pl_zero_shot
pl_to_lt_zero_shot
joint_to_lt
joint_to_pl

Modes
-----
Smoke:
    python scripts/73_analyze_cross_country_generalization.py --mode smoke

Final:
    python scripts/73_analyze_cross_country_generalization.py --mode final

Auto:
    python scripts/73_analyze_cross_country_generalization.py --mode auto

The script automatically searches result CSV/JSON files produced by
Scripts 66 and 71. It is deliberately tolerant of small differences in
column naming.

Outputs
-------
data/processed/cross_country/domain_generalization/
    standardized_results.csv
    protocol_summary.csv
    transfer_analysis.csv
    target_country_analysis.csv
    model_generalization_summary.csv
    per_seed_contrasts.csv
    domain_generalization_report.json
    figures/
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

# Important on Windows:
# prevent Tkinter/main-thread errors during batch figure generation.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed" / "cross_country"

BASELINE_DIR = PROCESSED / "baseline_models"
CARTOGRAPHX_DIR = PROCESSED / "cartographx_models"

OUT_DIR = PROCESSED / "domain_generalization"
FIG_DIR = OUT_DIR / "figures"

PROTOCOLS = [
    "lt_to_lt",
    "pl_to_pl",
    "lt_to_pl_zero_shot",
    "pl_to_lt_zero_shot",
    "joint_to_lt",
    "joint_to_pl",
]

PROTOCOL_DISPLAY = {
    "lt_to_lt": "LT → LT",
    "pl_to_pl": "PL → PL",
    "lt_to_pl_zero_shot": "LT → PL",
    "pl_to_lt_zero_shot": "PL → LT",
    "joint_to_lt": "Joint → LT",
    "joint_to_pl": "Joint → PL",
}

PROTOCOL_ORDER = {p: i for i, p in enumerate(PROTOCOLS)}

TARGET_COUNTRY = {
    "lt_to_lt": "Lithuania",
    "pl_to_pl": "Poland",
    "lt_to_pl_zero_shot": "Poland",
    "pl_to_lt_zero_shot": "Lithuania",
    "joint_to_lt": "Lithuania",
    "joint_to_pl": "Poland",
}

SOURCE_DOMAIN = {
    "lt_to_lt": "Lithuania",
    "pl_to_pl": "Poland",
    "lt_to_pl_zero_shot": "Lithuania",
    "pl_to_lt_zero_shot": "Poland",
    "joint_to_lt": "Joint",
    "joint_to_pl": "Joint",
}

PROTOCOL_TYPE = {
    "lt_to_lt": "within_country",
    "pl_to_pl": "within_country",
    "lt_to_pl_zero_shot": "zero_shot",
    "pl_to_lt_zero_shot": "zero_shot",
    "joint_to_lt": "joint",
    "joint_to_pl": "joint",
}

METRICS = [
    "mcc",
    "balanced_accuracy",
    "f1",
    "roc_auc",
    "pr_auc",
]

PRIMARY_METRIC = "mcc"

MODEL_ORDER = [
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "cartographx",
]


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def section(title: str) -> None:
    print()
    print("-" * 120)
    print(title)
    print("-" * 120)


def banner(title: str) -> None:
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def clean_name(x) -> str:
    return str(x).strip().lower().replace("-", "_").replace(" ", "_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [clean_name(c) for c in d.columns]
    return d


def first_existing(
    df: pd.DataFrame,
    names: Sequence[str],
    required: bool = False,
) -> Optional[str]:
    cols = set(df.columns)

    for name in names:
        n = clean_name(name)
        if n in cols:
            return n

    if required:
        raise KeyError(
            f"Could not find any of {list(names)}.\n"
            f"Available columns:\n{df.columns.tolist()}"
        )

    return None


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if pd.isna(obj) if not isinstance(obj, (dict, list, tuple)) else False:
        return None

    return obj


def save_figure(fig: plt.Figure, stem: str) -> None:
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"

    fig.savefig(
        png,
        dpi=350,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"PNG: {png}")
    print(f"PDF: {pdf}")


def normalize_protocol(value: str) -> Optional[str]:
    s = clean_name(value)

    replacements = {
        "lt_lt": "lt_to_lt",
        "lithuania_to_lithuania": "lt_to_lt",
        "pl_pl": "pl_to_pl",
        "poland_to_poland": "pl_to_pl",
        "lt_pl": "lt_to_pl_zero_shot",
        "lt_to_pl": "lt_to_pl_zero_shot",
        "lithuania_to_poland": "lt_to_pl_zero_shot",
        "pl_lt": "pl_to_lt_zero_shot",
        "pl_to_lt": "pl_to_lt_zero_shot",
        "poland_to_lithuania": "pl_to_lt_zero_shot",
        "joint_lt": "joint_to_lt",
        "joint_lithuania": "joint_to_lt",
        "joint_pl": "joint_to_pl",
        "joint_poland": "joint_to_pl",
    }

    s = replacements.get(s, s)

    if s in PROTOCOLS:
        return s

    for protocol in PROTOCOLS:
        if protocol in s:
            return protocol

    return None


def normalize_model(value: str) -> str:
    s = clean_name(value)

    aliases = {
        "lr": "logistic_regression",
        "logistic": "logistic_regression",
        "logisticregression": "logistic_regression",
        "randomforest": "random_forest",
        "rf": "random_forest",
        "extratrees": "extra_trees",
        "et": "extra_trees",
        "histgradientboosting": "hist_gradient_boosting",
        "hist_gradient_boost": "hist_gradient_boosting",
        "hgb": "hist_gradient_boosting",
        "cartograph_x": "cartographx",
        "cartograph_x_full": "cartographx",
        "cartographx_full": "cartographx",
        "full_cartographx": "cartographx",
        "full": "cartographx",
    }

    return aliases.get(s, s)


def model_display(model: str) -> str:
    mapping = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
        "hist_gradient_boosting": "HistGradientBoosting",
        "cartographx": "CartoGraph-X",
    }

    return mapping.get(model, model.replace("_", " ").title())


# =============================================================================
# RESULT DISCOVERY
# =============================================================================

def candidate_csv_files(root: Path) -> List[Path]:
    if not root.exists():
        return []

    ignore_tokens = [
        "prediction",
        "embedding",
        "history",
        "epoch",
        "feature",
        "fusion",
        "confusion",
        "node",
        "edge",
        "operation_distribution",
    ]

    files = []

    for p in root.rglob("*.csv"):
        name = clean_name(p.name)

        if any(token in name for token in ignore_tokens):
            continue

        files.append(p)

    return files


def infer_protocol_from_path(path: Path) -> Optional[str]:
    text = clean_name(str(path))

    # Longer names first.
    for protocol in [
        "lt_to_pl_zero_shot",
        "pl_to_lt_zero_shot",
        "joint_to_lt",
        "joint_to_pl",
        "lt_to_lt",
        "pl_to_pl",
    ]:
        if protocol in text:
            return protocol

    return None


def infer_seed_from_path(path: Path) -> Optional[int]:
    text = clean_name(str(path))

    patterns = [
        r"seed[_\-]?(\d+)",
        r"run[_\-]?(\d+)",
    ]

    for pattern in patterns:
        m = re.search(pattern, text)

        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass

    return None


def find_metric_column(df: pd.DataFrame, metric: str) -> Optional[str]:
    aliases = {
        "mcc": [
            "mcc",
            "test_mcc",
            "matthews_corrcoef",
            "matthews_correlation_coefficient",
        ],
        "balanced_accuracy": [
            "balanced_accuracy",
            "balanced_acc",
            "ba",
            "test_balanced_accuracy",
            "test_ba",
        ],
        "f1": [
            "f1",
            "f1_score",
            "test_f1",
        ],
        "roc_auc": [
            "roc_auc",
            "auc",
            "test_roc_auc",
            "test_auc",
        ],
        "pr_auc": [
            "pr_auc",
            "average_precision",
            "ap",
            "test_pr_auc",
        ],
    }

    return first_existing(df, aliases[metric], required=False)


def dataframe_contains_metrics(df: pd.DataFrame) -> bool:
    d = normalize_columns(df)
    return find_metric_column(d, "mcc") is not None


def extract_rows_from_dataframe(
    df: pd.DataFrame,
    source_file: Path,
    default_model: Optional[str],
) -> pd.DataFrame:

    d = normalize_columns(df)

    if not dataframe_contains_metrics(d):
        return pd.DataFrame()

    protocol_col = first_existing(
        d,
        [
            "protocol",
            "experiment",
            "protocol_name",
            "evaluation_protocol",
        ],
        required=False,
    )

    model_col = first_existing(
        d,
        [
            "model",
            "model_name",
            "method",
            "architecture",
        ],
        required=False,
    )

    seed_col = first_existing(
        d,
        [
            "seed",
            "random_seed",
            "run_seed",
        ],
        required=False,
    )

    split_col = first_existing(
        d,
        [
            "split",
            "stage",
            "evaluation_split",
            "dataset_split",
        ],
        required=False,
    )

    # If this table includes train/validation/test rows, only use test.
    if split_col is not None:
        split_values = d[split_col].astype(str).map(clean_name)

        if (split_values == "test").any():
            d = d.loc[split_values == "test"].copy()

    rows = []

    path_protocol = infer_protocol_from_path(source_file)
    path_seed = infer_seed_from_path(source_file)

    for _, r in d.iterrows():

        if protocol_col is not None:
            protocol = normalize_protocol(r[protocol_col])
        else:
            protocol = path_protocol

        if protocol is None:
            continue

        if model_col is not None:
            model = normalize_model(r[model_col])
        elif default_model is not None:
            model = default_model
        else:
            model = "unknown"

        if seed_col is not None:
            seed = pd.to_numeric(
                pd.Series([r[seed_col]]),
                errors="coerce",
            ).iloc[0]

            seed = int(seed) if pd.notna(seed) else path_seed
        else:
            seed = path_seed

        if seed is None:
            seed = 42

        record = {
            "model": model,
            "protocol": protocol,
            "seed": int(seed),
            "source_file": str(source_file),
        }

        for metric in METRICS:
            col = find_metric_column(d, metric)

            if col is not None:
                value = pd.to_numeric(
                    pd.Series([r[col]]),
                    errors="coerce",
                ).iloc[0]

                record[metric] = (
                    float(value) if pd.notna(value) else np.nan
                )
            else:
                record[metric] = np.nan

        rows.append(record)

    return pd.DataFrame(rows)


def discover_csv_results(
    root: Path,
    default_model: Optional[str],
) -> pd.DataFrame:

    extracted = []

    for path in candidate_csv_files(root):

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        try:
            x = extract_rows_from_dataframe(
                df,
                source_file=path,
                default_model=default_model,
            )
        except Exception:
            continue

        if not x.empty:
            extracted.append(x)

    if not extracted:
        return pd.DataFrame()

    return pd.concat(extracted, ignore_index=True)


# =============================================================================
# JSON FALLBACK
# =============================================================================

def flatten_json(
    obj,
    prefix: str = "",
    output: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:

    if output is None:
        output = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten_json(value, new_prefix, output)

    elif isinstance(obj, list):
        # Lists of scalar values are kept as-is.
        if all(
            not isinstance(x, (dict, list))
            for x in obj
        ):
            output[prefix] = obj

    else:
        output[prefix] = obj

    return output


def metric_from_flat_json(
    flat: Dict[str, object],
    metric: str,
) -> Optional[float]:

    aliases = {
        "mcc": [
            "test.mcc",
            "test_mcc",
            "final_test.mcc",
            "test_metrics.mcc",
            "metrics.mcc",
            "mcc",
        ],
        "balanced_accuracy": [
            "test.balanced_accuracy",
            "test.ba",
            "test_balanced_accuracy",
            "final_test.balanced_accuracy",
            "test_metrics.balanced_accuracy",
            "balanced_accuracy",
            "ba",
        ],
        "f1": [
            "test.f1",
            "test_f1",
            "final_test.f1",
            "test_metrics.f1",
            "f1",
        ],
        "roc_auc": [
            "test.roc_auc",
            "test_roc_auc",
            "final_test.roc_auc",
            "test_metrics.roc_auc",
            "roc_auc",
        ],
        "pr_auc": [
            "test.pr_auc",
            "test_pr_auc",
            "final_test.pr_auc",
            "test_metrics.pr_auc",
            "pr_auc",
        ],
    }

    normalized = {
        clean_name(k.replace(".", "_")): v
        for k, v in flat.items()
    }

    for alias in aliases[metric]:
        a = clean_name(alias.replace(".", "_"))

        if a in normalized:
            try:
                return float(normalized[a])
            except Exception:
                pass

    # Last resort: suffix matching.
    for key, value in normalized.items():
        for alias in aliases[metric]:
            a = clean_name(alias.replace(".", "_"))

            if key.endswith(a):
                try:
                    return float(value)
                except Exception:
                    pass

    return None


def discover_json_results(
    root: Path,
    default_model: str,
) -> pd.DataFrame:

    if not root.exists():
        return pd.DataFrame()

    rows = []

    for path in root.rglob("*.json"):

        protocol = infer_protocol_from_path(path)

        if protocol is None:
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        flat = flatten_json(obj)

        mcc = metric_from_flat_json(flat, "mcc")

        if mcc is None:
            continue

        seed = infer_seed_from_path(path)

        if seed is None:
            seed_candidates = [
                v
                for k, v in flat.items()
                if clean_name(k).endswith("seed")
            ]

            seed = None

            for candidate in seed_candidates:
                try:
                    seed = int(candidate)
                    break
                except Exception:
                    continue

        if seed is None:
            seed = 42

        row = {
            "model": default_model,
            "protocol": protocol,
            "seed": seed,
            "source_file": str(path),
        }

        for metric in METRICS:
            value = metric_from_flat_json(flat, metric)
            row[metric] = np.nan if value is None else value

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# RESULT CLEANING
# =============================================================================

def deduplicate_results(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return df

    d = df.copy()

    d["model"] = d["model"].map(normalize_model)
    d["protocol"] = d["protocol"].map(normalize_protocol)

    d = d[d["protocol"].isin(PROTOCOLS)].copy()

    for metric in METRICS:
        d[metric] = safe_numeric(d[metric])

    d["seed"] = safe_numeric(d["seed"]).fillna(42).astype(int)

    # Prefer rows with the largest number of available metrics.
    d["_metric_count"] = d[METRICS].notna().sum(axis=1)

    d = d.sort_values(
        ["model", "protocol", "seed", "_metric_count"],
        ascending=[True, True, True, False],
    )

    d = d.drop_duplicates(
        subset=["model", "protocol", "seed"],
        keep="first",
    )

    d = d.drop(columns="_metric_count")

    d["protocol_display"] = d["protocol"].map(PROTOCOL_DISPLAY)
    d["source_domain"] = d["protocol"].map(SOURCE_DOMAIN)
    d["target_country"] = d["protocol"].map(TARGET_COUNTRY)
    d["protocol_type"] = d["protocol"].map(PROTOCOL_TYPE)

    d["_protocol_order"] = d["protocol"].map(PROTOCOL_ORDER)

    d = d.sort_values(
        ["model", "_protocol_order", "seed"]
    ).drop(columns="_protocol_order")

    return d.reset_index(drop=True)


def load_all_results() -> pd.DataFrame:

    section("1. DISCOVER EXPERIMENTAL RESULTS")

    baseline = discover_csv_results(
        BASELINE_DIR,
        default_model=None,
    )

    cartographx = discover_csv_results(
        CARTOGRAPHX_DIR,
        default_model="cartographx",
    )

    # JSON fallback if CartoGraph-X summary CSVs are absent.
    if cartographx.empty:
        print(
            "[INFO] No CartoGraph-X metric CSV detected. "
            "Trying JSON reports..."
        )

        cartographx = discover_json_results(
            CARTOGRAPHX_DIR,
            default_model="cartographx",
        )

    frames = []

    if not baseline.empty:
        frames.append(baseline)

    if not cartographx.empty:
        frames.append(cartographx)

    if not frames:
        raise FileNotFoundError(
            "No usable experimental result files were found.\n\n"
            f"Checked:\n"
            f"  {BASELINE_DIR}\n"
            f"  {CARTOGRAPHX_DIR}\n\n"
            "Complete Script 66 and Script 71 first."
        )

    d = pd.concat(frames, ignore_index=True)

    d = deduplicate_results(d)

    print(f"Standardized result rows: {len(d):,}")
    print(f"Models: {sorted(d['model'].unique().tolist())}")
    print(f"Protocols: {sorted(d['protocol'].unique().tolist())}")
    print(f"Seeds: {sorted(d['seed'].unique().tolist())}")

    print()
    print("AVAILABLE RUNS")

    availability = (
        d.groupby(["model", "protocol"])
        .agg(
            runs=("seed", "nunique"),
            mcc_available=("mcc", "count"),
        )
        .reset_index()
    )

    print(availability.to_string(index=False))

    return d


# =============================================================================
# SUMMARIES
# =============================================================================

def summarize_protocols(df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for (model, protocol), g in df.groupby(
        ["model", "protocol"],
        sort=False,
    ):

        row = {
            "model": model,
            "model_display": model_display(model),
            "protocol": protocol,
            "protocol_display": PROTOCOL_DISPLAY[protocol],
            "protocol_type": PROTOCOL_TYPE[protocol],
            "source_domain": SOURCE_DOMAIN[protocol],
            "target_country": TARGET_COUNTRY[protocol],
            "runs": int(g["seed"].nunique()),
        }

        for metric in METRICS:
            x = g[metric].dropna().astype(float)

            row[f"{metric}_mean"] = (
                float(x.mean()) if len(x) else np.nan
            )

            row[f"{metric}_std"] = (
                float(x.std(ddof=1)) if len(x) > 1 else np.nan
            )

            row[f"{metric}_sem"] = (
                float(x.sem(ddof=1)) if len(x) > 1 else np.nan
            )

            if len(x) > 1:
                row[f"{metric}_ci95_low"] = (
                    float(x.mean() - 1.96 * x.sem(ddof=1))
                )

                row[f"{metric}_ci95_high"] = (
                    float(x.mean() + 1.96 * x.sem(ddof=1))
                )
            else:
                row[f"{metric}_ci95_low"] = np.nan
                row[f"{metric}_ci95_high"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)

    out["_order"] = out["protocol"].map(PROTOCOL_ORDER)

    out = out.sort_values(
        ["model", "_order"]
    ).drop(columns="_order")

    return out.reset_index(drop=True)


# =============================================================================
# CROSS-COUNTRY CONTRASTS
# =============================================================================

def get_mean_metric(
    summary: pd.DataFrame,
    model: str,
    protocol: str,
    metric: str,
) -> float:

    x = summary[
        (summary["model"] == model)
        & (summary["protocol"] == protocol)
    ]

    if x.empty:
        return np.nan

    return float(x.iloc[0][f"{metric}_mean"])


def build_transfer_analysis(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    models = summary["model"].unique()

    definitions = [
        {
            "target_country": "Lithuania",
            "within_protocol": "lt_to_lt",
            "zero_shot_protocol": "pl_to_lt_zero_shot",
            "joint_protocol": "joint_to_lt",
        },
        {
            "target_country": "Poland",
            "within_protocol": "pl_to_pl",
            "zero_shot_protocol": "lt_to_pl_zero_shot",
            "joint_protocol": "joint_to_pl",
        },
    ]

    for model in models:
        for spec in definitions:

            row = {
                "model": model,
                "model_display": model_display(model),
                "target_country": spec["target_country"],
                "within_protocol": spec["within_protocol"],
                "zero_shot_protocol": spec["zero_shot_protocol"],
                "joint_protocol": spec["joint_protocol"],
            }

            for metric in METRICS:

                within = get_mean_metric(
                    summary,
                    model,
                    spec["within_protocol"],
                    metric,
                )

                zero = get_mean_metric(
                    summary,
                    model,
                    spec["zero_shot_protocol"],
                    metric,
                )

                joint = get_mean_metric(
                    summary,
                    model,
                    spec["joint_protocol"],
                    metric,
                )

                row[f"{metric}_within"] = within
                row[f"{metric}_zero_shot"] = zero
                row[f"{metric}_joint"] = joint

                # Positive = zero-shot is worse than in-domain.
                row[f"{metric}_zero_shot_drop"] = (
                    within - zero
                    if pd.notna(within) and pd.notna(zero)
                    else np.nan
                )

                # Positive = adding second country improves target.
                row[f"{metric}_joint_gain"] = (
                    joint - within
                    if pd.notna(joint) and pd.notna(within)
                    else np.nan
                )

                # Positive = joint model improves over zero-shot.
                row[f"{metric}_joint_vs_zero_shot"] = (
                    joint - zero
                    if pd.notna(joint) and pd.notna(zero)
                    else np.nan
                )

            rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# PER-SEED CONTRASTS
# =============================================================================

def build_per_seed_contrasts(df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    definitions = [
        (
            "Lithuania",
            "lt_to_lt",
            "pl_to_lt_zero_shot",
            "joint_to_lt",
        ),
        (
            "Poland",
            "pl_to_pl",
            "lt_to_pl_zero_shot",
            "joint_to_pl",
        ),
    ]

    for model in df["model"].unique():

        dm = df[df["model"] == model]

        for target, within_p, zero_p, joint_p in definitions:

            within = dm[dm["protocol"] == within_p].set_index("seed")
            zero = dm[dm["protocol"] == zero_p].set_index("seed")
            joint = dm[dm["protocol"] == joint_p].set_index("seed")

            common_all = (
                set(within.index)
                & set(zero.index)
                & set(joint.index)
            )

            for seed in sorted(common_all):

                row = {
                    "model": model,
                    "model_display": model_display(model),
                    "target_country": target,
                    "seed": int(seed),
                }

                for metric in METRICS:

                    w = within.loc[seed, metric]
                    z = zero.loc[seed, metric]
                    j = joint.loc[seed, metric]

                    row[f"{metric}_within"] = w
                    row[f"{metric}_zero_shot"] = z
                    row[f"{metric}_joint"] = j

                    row[f"{metric}_zero_shot_drop"] = w - z
                    row[f"{metric}_joint_gain"] = j - w
                    row[f"{metric}_joint_vs_zero_shot"] = j - z

                rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# MODEL-LEVEL GENERALIZATION SUMMARY
# =============================================================================

def build_model_generalization_summary(
    transfer: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for model, g in transfer.groupby("model"):

        row = {
            "model": model,
            "model_display": model_display(model),
        }

        for metric in METRICS:

            for contrast in [
                "zero_shot_drop",
                "joint_gain",
                "joint_vs_zero_shot",
            ]:

                col = f"{metric}_{contrast}"

                x = g[col].dropna()

                row[f"mean_{col}"] = (
                    float(x.mean()) if len(x) else np.nan
                )

        # Transfer asymmetry:
        # positive means Lithuania -> Poland suffers a larger MCC drop
        # than Poland -> Lithuania.
        lt = g[g["target_country"] == "Lithuania"]
        pl = g[g["target_country"] == "Poland"]

        if not lt.empty and not pl.empty:

            lt_drop = lt.iloc[0]["mcc_zero_shot_drop"]
            pl_drop = pl.iloc[0]["mcc_zero_shot_drop"]

            if pd.notna(lt_drop) and pd.notna(pl_drop):
                row["mcc_transfer_asymmetry_pl_minus_lt"] = (
                    pl_drop - lt_drop
                )
            else:
                row["mcc_transfer_asymmetry_pl_minus_lt"] = np.nan
        else:
            row["mcc_transfer_asymmetry_pl_minus_lt"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# FIGURES
# =============================================================================

def plot_protocol_metric(
    summary: pd.DataFrame,
    metric: str,
    stem: str,
    ylabel: str,
) -> None:

    available = summary[
        summary[f"{metric}_mean"].notna()
    ].copy()

    if available.empty:
        return

    models = [
        m
        for m in MODEL_ORDER
        if m in available["model"].unique()
    ]

    for m in available["model"].unique():
        if m not in models:
            models.append(m)

    protocols = [
        p
        for p in PROTOCOLS
        if p in available["protocol"].unique()
    ]

    fig, ax = plt.subplots(figsize=(13, 6.5))

    x = np.arange(len(protocols))
    n_models = max(len(models), 1)
    width = min(0.16, 0.8 / n_models)

    for i, model in enumerate(models):

        vals = []
        errs = []

        for protocol in protocols:

            r = available[
                (available["model"] == model)
                & (available["protocol"] == protocol)
            ]

            if r.empty:
                vals.append(np.nan)
                errs.append(0.0)
            else:
                vals.append(
                    r.iloc[0][f"{metric}_mean"]
                )

                sd = r.iloc[0][f"{metric}_std"]

                errs.append(
                    0.0 if pd.isna(sd) else sd
                )

        offset = (
            i - (n_models - 1) / 2
        ) * width

        ax.bar(
            x + offset,
            vals,
            width=width,
            yerr=errs if any(v > 0 for v in errs) else None,
            capsize=3,
            label=model_display(model),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [PROTOCOL_DISPLAY[p] for p in protocols]
    )

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Evaluation protocol")
    ax.set_title(
        f"Cross-country road-selection performance: {ylabel}"
    )

    ax.axhline(
        0.0 if metric == "mcc" else 0.5,
        linewidth=0.8,
        linestyle="--",
    )

    ax.legend(
        frameon=False,
        ncol=2,
    )

    fig.tight_layout()

    save_figure(fig, stem)


def plot_cartographx_transfer_matrix(
    summary: pd.DataFrame,
) -> None:

    d = summary[
        summary["model"] == "cartographx"
    ].copy()

    if d.empty:
        return

    matrix = np.full((3, 2), np.nan)

    mapping = {
        (0, 0): "lt_to_lt",
        (0, 1): "lt_to_pl_zero_shot",
        (1, 0): "pl_to_lt_zero_shot",
        (1, 1): "pl_to_pl",
        (2, 0): "joint_to_lt",
        (2, 1): "joint_to_pl",
    }

    for (r, c), protocol in mapping.items():

        x = d[d["protocol"] == protocol]

        if not x.empty:
            matrix[r, c] = x.iloc[0]["mcc_mean"]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    im = ax.imshow(
        matrix,
        aspect="auto",
        vmin=np.nanmin(matrix) if np.isfinite(matrix).any() else 0,
        vmax=np.nanmax(matrix) if np.isfinite(matrix).any() else 1,
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Test: Lithuania", "Test: Poland"])

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(
        [
            "Train: Lithuania",
            "Train: Poland",
            "Train: Joint",
        ]
    )

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):

            if np.isfinite(matrix[r, c]):
                ax.text(
                    c,
                    r,
                    f"{matrix[r, c]:.3f}",
                    ha="center",
                    va="center",
                )

    ax.set_title(
        "CartoGraph-X cross-country transfer matrix (MCC)"
    )

    fig.colorbar(
        im,
        ax=ax,
        label="Matthews correlation coefficient",
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure_03_cartographx_transfer_matrix",
    )


def plot_generalization_gaps(
    transfer: pd.DataFrame,
) -> None:

    if transfer.empty:
        return

    d = transfer[
        transfer["mcc_zero_shot_drop"].notna()
    ].copy()

    if d.empty:
        return

    models = [
        m
        for m in MODEL_ORDER
        if m in d["model"].unique()
    ]

    for m in d["model"].unique():
        if m not in models:
            models.append(m)

    countries = ["Lithuania", "Poland"]

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(models))
    width = 0.34

    for i, country in enumerate(countries):

        values = []

        for model in models:

            r = d[
                (d["model"] == model)
                & (d["target_country"] == country)
            ]

            values.append(
                np.nan
                if r.empty
                else r.iloc[0]["mcc_zero_shot_drop"]
            )

        ax.bar(
            x + (i - 0.5) * width,
            values,
            width=width,
            label=f"Target: {country}",
        )

    ax.axhline(0, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [model_display(m) for m in models],
        rotation=20,
        ha="right",
    )

    ax.set_ylabel(
        "MCC zero-shot drop\n(within-country − zero-shot)"
    )

    ax.set_title(
        "Cross-country generalization gap"
    )

    ax.legend(frameon=False)

    fig.tight_layout()

    save_figure(
        fig,
        "figure_04_zero_shot_generalization_gap",
    )


def plot_joint_gain(
    transfer: pd.DataFrame,
) -> None:

    if transfer.empty:
        return

    d = transfer[
        transfer["mcc_joint_gain"].notna()
    ].copy()

    if d.empty:
        return

    models = [
        m
        for m in MODEL_ORDER
        if m in d["model"].unique()
    ]

    for m in d["model"].unique():
        if m not in models:
            models.append(m)

    countries = ["Lithuania", "Poland"]

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(models))
    width = 0.34

    for i, country in enumerate(countries):

        values = []

        for model in models:

            r = d[
                (d["model"] == model)
                & (d["target_country"] == country)
            ]

            values.append(
                np.nan
                if r.empty
                else r.iloc[0]["mcc_joint_gain"]
            )

        ax.bar(
            x + (i - 0.5) * width,
            values,
            width=width,
            label=f"Target: {country}",
        )

    ax.axhline(0, linewidth=0.8)

    ax.set_xticks(x)

    ax.set_xticklabels(
        [model_display(m) for m in models],
        rotation=20,
        ha="right",
    )

    ax.set_ylabel(
        "MCC joint-country gain\n(joint − target-only)"
    )

    ax.set_title(
        "Benefit of multi-country training"
    )

    ax.legend(frameon=False)

    fig.tight_layout()

    save_figure(
        fig,
        "figure_05_joint_country_gain",
    )


def plot_cartographx_three_regimes(
    transfer: pd.DataFrame,
) -> None:

    d = transfer[
        transfer["model"] == "cartographx"
    ].copy()

    if d.empty:
        return

    countries = ["Lithuania", "Poland"]

    fig, ax = plt.subplots(figsize=(8.5, 6))

    x = np.arange(2)
    width = 0.24

    regimes = [
        ("mcc_within", "Within-country"),
        ("mcc_zero_shot", "Zero-shot"),
        ("mcc_joint", "Joint-country"),
    ]

    for i, (column, label) in enumerate(regimes):

        values = []

        for country in countries:

            r = d[d["target_country"] == country]

            values.append(
                np.nan
                if r.empty
                else r.iloc[0][column]
            )

        ax.bar(
            x + (i - 1) * width,
            values,
            width=width,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(countries)

    ax.set_ylabel("MCC")

    ax.set_title(
        "CartoGraph-X geographic generalization regimes"
    )

    ax.axhline(0, linewidth=0.8)

    ax.legend(frameon=False)

    fig.tight_layout()

    save_figure(
        fig,
        "figure_06_cartographx_generalization_regimes",
    )


# =============================================================================
# CONSOLE REPORT
# =============================================================================

def print_cartographx_analysis(
    summary: pd.DataFrame,
    transfer: pd.DataFrame,
) -> None:

    section("5. CARTOGRAPH-X CROSS-COUNTRY ANALYSIS")

    d = summary[
        summary["model"] == "cartographx"
    ].copy()

    if d.empty:
        print(
            "[WARNING] CartoGraph-X results were not found."
        )
        return

    for protocol in PROTOCOLS:

        r = d[d["protocol"] == protocol]

        if r.empty:
            print(
                f"{PROTOCOL_DISPLAY[protocol]:15s} MISSING"
            )
            continue

        r = r.iloc[0]

        print(
            f"{PROTOCOL_DISPLAY[protocol]:15s} "
            f"MCC={r['mcc_mean']:.4f} | "
            f"BA={r['balanced_accuracy_mean']:.4f} | "
            f"F1={r['f1_mean']:.4f} | "
            f"ROC-AUC={r['roc_auc_mean']:.4f} | "
            f"PR-AUC={r['pr_auc_mean']:.4f} | "
            f"runs={int(r['runs'])}"
        )

    print()
    print("TRANSFER EFFECTS")

    ct = transfer[
        transfer["model"] == "cartographx"
    ]

    for _, r in ct.iterrows():

        print()
        print(r["target_country"])

        print(
            f"  Within-country MCC:      "
            f"{r['mcc_within']:.4f}"
        )

        print(
            f"  Zero-shot MCC:           "
            f"{r['mcc_zero_shot']:.4f}"
        )

        print(
            f"  Joint-country MCC:       "
            f"{r['mcc_joint']:.4f}"
        )

        print(
            f"  Zero-shot MCC drop:      "
            f"{r['mcc_zero_shot_drop']:+.4f}"
        )

        print(
            f"  Joint-training MCC gain: "
            f"{r['mcc_joint_gain']:+.4f}"
        )

        print(
            f"  Joint vs zero-shot:      "
            f"{r['mcc_joint_vs_zero_shot']:+.4f}"
        )


# =============================================================================
# READINESS AUDIT
# =============================================================================

def readiness_audit(
    df: pd.DataFrame,
    mode: str,
) -> Dict[str, bool]:

    section("7. ANALYSIS READINESS AUDIT")

    checks = {}

    checks["cartographx_present"] = (
        "cartographx" in df["model"].unique()
    )

    cx = df[df["model"] == "cartographx"]

    for protocol in PROTOCOLS:
        checks[
            f"cartographx_{protocol}_present"
        ] = (
            protocol in cx["protocol"].unique()
        )

    checks["both_zero_shot_directions_present"] = all(
        p in cx["protocol"].unique()
        for p in [
            "lt_to_pl_zero_shot",
            "pl_to_lt_zero_shot",
        ]
    )

    checks["both_joint_targets_present"] = all(
        p in cx["protocol"].unique()
        for p in [
            "joint_to_lt",
            "joint_to_pl",
        ]
    )

    if mode == "final":

        run_counts = (
            cx.groupby("protocol")["seed"]
            .nunique()
            .reindex(PROTOCOLS)
            .fillna(0)
        )

        checks["ten_runs_each_cartographx_protocol"] = bool(
            (run_counts >= 10).all()
        )

    for name, passed in checks.items():
        print(
            f"{name:55s} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    return checks


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["smoke", "final", "auto"],
        default="auto",
        help=(
            "smoke = accepts one run; "
            "final = expects >=10 CartoGraph-X runs/protocol; "
            "auto = detects run count."
        ),
    )

    args = parser.parse_args()

    ensure_dirs()

    banner(
        "CARTOGEN — CROSS-COUNTRY GENERALIZATION AND DOMAIN-SHIFT ANALYSIS"
    )

    print(f"Baseline results:    {BASELINE_DIR}")
    print(f"CartoGraph-X results:{CARTOGRAPHX_DIR}")
    print(f"Output:              {OUT_DIR}")
    print(f"Requested mode:      {args.mode}")

    results = load_all_results()

    # -------------------------------------------------------------------------
    # Detect mode
    # -------------------------------------------------------------------------

    if args.mode == "auto":

        cx = results[
            results["model"] == "cartographx"
        ]

        if not cx.empty:

            counts = (
                cx.groupby("protocol")["seed"]
                .nunique()
                .reindex(PROTOCOLS)
                .fillna(0)
            )

            mode = (
                "final"
                if (counts >= 10).all()
                else "smoke"
            )
        else:
            mode = "smoke"

    else:
        mode = args.mode

    print(f"Effective mode:      {mode}")

    # -------------------------------------------------------------------------
    # Standardized results
    # -------------------------------------------------------------------------

    section("2. STANDARDIZED TEST RESULTS")

    display_cols = [
        "model",
        "protocol",
        "seed",
        "mcc",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    print(
        results[display_cols]
        .to_string(index=False)
    )

    results.to_csv(
        OUT_DIR / "standardized_results.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Protocol summaries
    # -------------------------------------------------------------------------

    section("3. PROTOCOL-LEVEL SUMMARY")

    summary = summarize_protocols(results)

    print_cols = [
        "model_display",
        "protocol_display",
        "runs",
        "mcc_mean",
        "balanced_accuracy_mean",
        "f1_mean",
        "roc_auc_mean",
        "pr_auc_mean",
    ]

    print(
        summary[print_cols]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    summary.to_csv(
        OUT_DIR / "protocol_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Transfer analysis
    # -------------------------------------------------------------------------

    section("4. GENERALIZATION CONTRASTS")

    transfer = build_transfer_analysis(summary)

    transfer_cols = [
        "model_display",
        "target_country",
        "mcc_within",
        "mcc_zero_shot",
        "mcc_joint",
        "mcc_zero_shot_drop",
        "mcc_joint_gain",
        "mcc_joint_vs_zero_shot",
    ]

    print(
        transfer[transfer_cols]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    transfer.to_csv(
        OUT_DIR / "transfer_analysis.csv",
        index=False,
    )

    transfer.to_csv(
        OUT_DIR / "target_country_analysis.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # CartoGraph-X interpretation
    # -------------------------------------------------------------------------

    print_cartographx_analysis(
        summary,
        transfer,
    )

    # -------------------------------------------------------------------------
    # Model generalization summary
    # -------------------------------------------------------------------------

    section("6. MODEL-LEVEL GENERALIZATION SUMMARY")

    model_summary = build_model_generalization_summary(
        transfer
    )

    cols = [
        "model_display",
        "mean_mcc_zero_shot_drop",
        "mean_mcc_joint_gain",
        "mean_mcc_joint_vs_zero_shot",
        "mcc_transfer_asymmetry_pl_minus_lt",
    ]

    available_cols = [
        c for c in cols
        if c in model_summary.columns
    ]

    print(
        model_summary[available_cols]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    model_summary.to_csv(
        OUT_DIR / "model_generalization_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Paired per-seed contrasts
    # -------------------------------------------------------------------------

    per_seed = build_per_seed_contrasts(results)

    per_seed.to_csv(
        OUT_DIR / "per_seed_contrasts.csv",
        index=False,
    )

    if mode == "smoke":
        print()
        print(
            "[INFO] Smoke mode: effect estimates are descriptive only. "
            "Paired significance testing is deferred until the final "
            "10-seed experiment."
        )

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------

    section("7. GENERATE FIGURES")

    plot_protocol_metric(
        summary,
        metric="mcc",
        stem="figure_01_mcc_by_protocol",
        ylabel="Matthews correlation coefficient",
    )

    plot_protocol_metric(
        summary,
        metric="balanced_accuracy",
        stem="figure_02_balanced_accuracy_by_protocol",
        ylabel="Balanced accuracy",
    )

    plot_cartographx_transfer_matrix(summary)

    plot_generalization_gaps(transfer)

    plot_joint_gain(transfer)

    plot_cartographx_three_regimes(transfer)

    # -------------------------------------------------------------------------
    # Readiness
    # -------------------------------------------------------------------------

    checks = readiness_audit(
        results,
        mode=mode,
    )

    # -------------------------------------------------------------------------
    # Scientific conclusions
    # -------------------------------------------------------------------------

    section("8. AUTOMATIC SCIENTIFIC INTERPRETATION")

    cx = transfer[
        transfer["model"] == "cartographx"
    ]

    conclusions = []

    if len(cx) == 2:

        for _, r in cx.iterrows():

            target = r["target_country"]

            if pd.notna(r["mcc_zero_shot_drop"]):

                if r["mcc_zero_shot_drop"] > 0:
                    msg = (
                        f"Zero-shot transfer to {target} exhibits an "
                        f"MCC generalization loss of "
                        f"{r['mcc_zero_shot_drop']:.4f} relative to "
                        f"target-country training."
                    )
                else:
                    msg = (
                        f"Zero-shot transfer to {target} does not reduce "
                        f"MCC relative to target-country training "
                        f"(difference={r['mcc_zero_shot_drop']:.4f})."
                    )

                conclusions.append(msg)
                print(msg)

            if pd.notna(r["mcc_joint_gain"]):

                if r["mcc_joint_gain"] > 0:
                    msg = (
                        f"Joint-country training improves MCC on {target} "
                        f"by {r['mcc_joint_gain']:.4f} over "
                        f"target-country-only training."
                    )
                else:
                    msg = (
                        f"Joint-country training does not improve MCC on "
                        f"{target}; change relative to target-only "
                        f"training is {r['mcc_joint_gain']:.4f}."
                    )

                conclusions.append(msg)
                print(msg)

    if not conclusions:
        print(
            "[INFO] Complete all six CartoGraph-X protocols to obtain "
            "automatic cross-country conclusions."
        )

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------

    report = {
        "script": 73,
        "analysis": (
            "cross-country generalization and domain-shift analysis"
        ),
        "mode": mode,
        "primary_metric": PRIMARY_METRIC,
        "protocols": PROTOCOLS,
        "models": sorted(
            results["model"].unique().tolist()
        ),
        "seeds": sorted(
            results["seed"].unique().tolist()
        ),
        "n_standardized_result_rows": int(len(results)),
        "readiness_checks": checks,
        "scientific_conclusions": conclusions,
        "output_files": {
            "standardized_results": str(
                OUT_DIR / "standardized_results.csv"
            ),
            "protocol_summary": str(
                OUT_DIR / "protocol_summary.csv"
            ),
            "transfer_analysis": str(
                OUT_DIR / "transfer_analysis.csv"
            ),
            "model_generalization_summary": str(
                OUT_DIR / "model_generalization_summary.csv"
            ),
            "per_seed_contrasts": str(
                OUT_DIR / "per_seed_contrasts.csv"
            ),
        },
    }

    report_path = (
        OUT_DIR / "domain_generalization_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_safe(report),
            f,
            indent=2,
            ensure_ascii=False,
        )

    # -------------------------------------------------------------------------
    # Final status
    # -------------------------------------------------------------------------

    section("9. OUTPUTS")

    print(
        f"Standardized results: "
        f"{OUT_DIR / 'standardized_results.csv'}"
    )

    print(
        f"Protocol summary:      "
        f"{OUT_DIR / 'protocol_summary.csv'}"
    )

    print(
        f"Transfer analysis:     "
        f"{OUT_DIR / 'transfer_analysis.csv'}"
    )

    print(
        f"Model summary:         "
        f"{OUT_DIR / 'model_generalization_summary.csv'}"
    )

    print(
        f"Per-seed contrasts:    "
        f"{OUT_DIR / 'per_seed_contrasts.csv'}"
    )

    print(
        f"Report:                "
        f"{report_path}"
    )

    print(
        f"Figures (PNG + PDF):   "
        f"{FIG_DIR}"
    )

    banner("SCRIPT 73 COMPLETE")

    required_smoke = [
        "cartographx_present",
        "cartographx_lt_to_lt_present",
        "cartographx_pl_to_pl_present",
        "cartographx_lt_to_pl_zero_shot_present",
        "cartographx_pl_to_lt_zero_shot_present",
        "cartographx_joint_to_lt_present",
        "cartographx_joint_to_pl_present",
    ]

    smoke_pass = all(
        checks.get(x, False)
        for x in required_smoke
    )

    if smoke_pass:

        if mode == "final":

            final_pass = checks.get(
                "ten_runs_each_cartographx_protocol",
                False,
            )

            if final_pass:
                print(
                    "[PASS] FINAL CROSS-COUNTRY GENERALIZATION "
                    "ANALYSIS COMPLETE."
                )
            else:
                print(
                    "[CHECK] Cross-country protocols are complete, "
                    "but final mode requires at least 10 seeds for "
                    "each CartoGraph-X protocol."
                )

        else:
            print(
                "[PASS] CROSS-COUNTRY GENERALIZATION SMOKE "
                "ANALYSIS COMPLETE."
            )

            print(
                "NEXT: Script 74 — reconstruct CartoGraph-X "
                "predictions spatially and generate actual "
                "generalized road-map outputs."
            )

    else:
        print(
            "[CHECK] Some CartoGraph-X protocols are missing. "
            "Complete Script 71 for all six protocols before "
            "interpreting cross-country generalization."
        )


if __name__ == "__main__":
    main()