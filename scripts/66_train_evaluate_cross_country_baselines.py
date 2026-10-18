#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
66_train_evaluate_cross_country_baselines.py

CartoGen — repeated classical road-selection baseline experiments under the
six leakage-safe cross-country protocols produced by Script 65.

Modes
-----
smoke
    One seed: 42.

full
    Ten seeds: 42-51.

Models
------
1. Logistic Regression
2. Random Forest
3. Extra Trees
4. HistGradientBoosting

Experimental design
-------------------
* Spatial train/validation/test partitions are fixed by Script 65.
* Hyperparameters are selected using VALIDATION only.
* Primary selection metric: validation MCC.
* Tie-breakers: validation PR-AUC, balanced accuracy, F1.
* Test is evaluated exactly once after model/configuration selection.
* Test data are never used for tuning.
* Classification threshold remains fixed at 0.5.
* No resampling/rebalancing of validation or test data.
* Training uses the class distribution supplied by Script 65.
* The same seed is used across methods for a given repeated run.
* All individual runs are retained for later paired statistical analysis.

Important
---------
Logistic regression can be effectively deterministic for this configuration.
It is nevertheless evaluated under every requested seed so that the result
matrix has identical protocol x model x seed structure. Zero between-run
variance is not interpreted as independent stochastic replication.

Outputs
-------
data/processed/cross_country/baseline_models/

    <protocol>/
        seed_42/
            validation_search_results.csv
            selected_models.csv
            test_metrics.csv
            test_predictions.csv
            run_report.json
        ...

    all_runs_validation_search.csv
    all_runs_selected_models.csv
    all_runs_metrics.csv
    all_runs_predictions.csv
    summary_10_runs.csv
    run_completeness.csv
    baseline_evaluation_report.json

    figures/
        aggregate metric figures

Usage
-----
python scripts/66_train_evaluate_cross_country_baselines.py --mode smoke
python scripts/66_train_evaluate_cross_country_baselines.py --mode full
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ======================================================================================================================
# PATHS
# ======================================================================================================================

ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "cross_country"
    / "modelling_protocols"
    / "protocols"
)

OUT = (
    ROOT
    / "data"
    / "processed"
    / "cross_country"
    / "baseline_models"
)

FIG = OUT / "figures"


# ======================================================================================================================
# EXPERIMENT CONFIGURATION
# ======================================================================================================================

PROTOCOLS = [
    "lt_to_lt",
    "pl_to_pl",
    "lt_to_pl_zero_shot",
    "pl_to_lt_zero_shot",
    "joint_to_lt",
    "joint_to_pl",
]

FEATURES = [
    "length_m__z",
    "intersecting_road_count__z",
    "neighbor_count_150m__z",
    "mean_neighbor_distance_m__z",
]

LABEL = "retained"

SMOKE_SEEDS = [42]

FULL_SEEDS = [
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
]

CLASSIFICATION_THRESHOLD = 0.5


# ======================================================================================================================
# MODEL GRIDS
# ======================================================================================================================

MODEL_GRIDS = {

    "logistic_regression": [
        {
            "C": 0.1,
        },
        {
            "C": 1.0,
        },
        {
            "C": 10.0,
        },
    ],

    "random_forest": [
        {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
    ],

    "extra_trees": [
        {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
    ],

    "hist_gradient_boosting": [
        {
            "learning_rate": 0.05,
            "max_iter": 200,
            "max_leaf_nodes": 15,
            "l2_regularization": 0.0,
        },
        {
            "learning_rate": 0.08,
            "max_iter": 250,
            "max_leaf_nodes": 31,
            "l2_regularization": 0.1,
        },
    ],
}


# ======================================================================================================================
# GENERAL UTILITIES
# ======================================================================================================================

def banner(text):
    print("\n" + "=" * 120)
    print(text)
    print("=" * 120)


def section(text):
    print("\n" + "-" * 120)
    print(text)
    print("-" * 120)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def json_default(obj):

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, Path):
        return str(obj)

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    raise TypeError(
        f"Object of type {obj.__class__.__name__} "
        f"is not JSON serializable"
    )


def safe_float(x):

    try:
        x = float(x)

        if math.isfinite(x):
            return x

        return None

    except Exception:
        return None


# ======================================================================================================================
# ARGUMENTS
# ======================================================================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Repeated classical baseline experiments for CartoGen."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help=(
            "smoke = seed 42 only; "
            "full = ten seeds 42-51."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Re-run completed protocol/seed experiments. "
            "Without this flag, valid completed runs are reused."
        ),
    )

    return parser.parse_args()


# ======================================================================================================================
# DATA
# ======================================================================================================================

def load_partition(protocol, role):

    path = (
        PROTOCOL_ROOT
        / protocol
        / f"{role}_preprocessed.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing protocol file:\n{path}\n"
            "Run Script 65 successfully first."
        )

    d = pd.read_csv(path)

    required = FEATURES + [LABEL]

    missing = [
        c
        for c in required
        if c not in d.columns
    ]

    if missing:

        raise KeyError(
            f"{path.name} is missing required columns: "
            f"{missing}"
        )

    for c in FEATURES:

        d[c] = pd.to_numeric(
            d[c],
            errors="coerce",
        )

    d[LABEL] = pd.to_numeric(
        d[LABEL],
        errors="coerce",
    )

    if d[LABEL].isna().any():

        raise ValueError(
            f"{protocol}/{role}: unresolved labels found."
        )

    if d[FEATURES].isna().any().any():

        bad = d[FEATURES].isna().sum()

        raise ValueError(
            f"{protocol}/{role}: NaN values remain "
            f"after preprocessing:\n"
            f"{bad[bad > 0]}"
        )

    labels = set(
        d[LABEL]
        .astype(int)
        .unique()
    )

    if not labels.issubset({0, 1}):

        raise ValueError(
            f"{protocol}/{role}: label is not binary."
        )

    return d


def load_all_protocols():

    loaded = {}

    for protocol in PROTOCOLS:

        loaded[protocol] = {}

        for role in [
            "train",
            "validation",
            "test",
        ]:

            d = load_partition(
                protocol,
                role,
            )

            loaded[protocol][role] = d

            retained = int(
                d[LABEL].sum()
            )

            eliminated = (
                len(d) - retained
            )

            print(
                f"{protocol:<22} "
                f"{role:<11} "
                f"{len(d):>7,} rows | "
                f"retained={retained:>6,} | "
                f"eliminated={eliminated:>6,}"
            )

    return loaded


# ======================================================================================================================
# MODELS
# ======================================================================================================================

def make_model(
    model_name,
    params,
    seed,
):

    if model_name == "logistic_regression":

        return LogisticRegression(
            C=params["C"],
            solver="lbfgs",
            max_iter=3000,
            random_state=seed,
        )

    if model_name == "random_forest":

        return RandomForestClassifier(
            **params,
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "extra_trees":

        return ExtraTreesClassifier(
            **params,
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "hist_gradient_boosting":

        return HistGradientBoostingClassifier(
            **params,
            random_state=seed,
        )

    raise KeyError(
        f"Unknown model: {model_name}"
    )


def probabilities(model, X):

    if hasattr(
        model,
        "predict_proba",
    ):

        return model.predict_proba(X)[:, 1]

    if hasattr(
        model,
        "decision_function",
    ):

        score = model.decision_function(X)

        return (
            1.0
            / (
                1.0
                + np.exp(-score)
            )
        )

    raise TypeError(
        f"{model.__class__.__name__} "
        "has no probability-like output."
    )


# ======================================================================================================================
# METRICS
# ======================================================================================================================

def compute_metrics(
    y_true,
    y_pred,
    y_prob,
):

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int,
    )

    y_prob = np.asarray(
        y_prob,
        dtype=float,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else np.nan
    )

    npv = (
        tn / (tn + fn)
        if (tn + fn)
        else np.nan
    )

    if len(
        np.unique(y_true)
    ) == 2:

        roc_auc = roc_auc_score(
            y_true,
            y_prob,
        )

        pr_auc = average_precision_score(
            y_true,
            y_prob,
        )

    else:

        roc_auc = np.nan
        pr_auc = np.nan

    return {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "specificity":
            specificity,

        "npv":
            npv,

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "mcc":
            matthews_corrcoef(
                y_true,
                y_pred,
            ),

        "roc_auc":
            roc_auc,

        "pr_auc":
            pr_auc,

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),
    }


def selection_key(row):

    values = [
        row["mcc"],
        row["pr_auc"],
        row["balanced_accuracy"],
        row["f1"],
    ]

    return tuple(
        -np.inf
        if pd.isna(x)
        else float(x)
        for x in values
    )


# ======================================================================================================================
# IDENTIFIERS
# ======================================================================================================================

def sample_identifier(df):

    for c in [
        "sample_uid_harmonized",
        "sample_uid",
    ]:

        if c in df.columns:
            return df[c].astype(str)

    source = None

    for c in [
        "source_idx_harmonized",
        "source_idx",
        "feature_index",
    ]:

        if c in df.columns:

            source = c
            break

    if (
        source is not None
        and "country" in df.columns
    ):

        return (
            df["country"].astype(str)
            + "::"
            + df[source].astype(str)
        )

    return pd.Series(
        [
            f"row_{i}"
            for i in range(len(df))
        ],
        index=df.index,
    )


# ======================================================================================================================
# RUN COMPLETENESS / RESUME
# ======================================================================================================================

def run_directory(
    protocol,
    seed,
):

    return (
        OUT
        / protocol
        / f"seed_{seed}"
    )


def run_files(
    protocol,
    seed,
):

    run_dir = run_directory(
        protocol,
        seed,
    )

    return {
        "dir":
            run_dir,

        "validation":
            run_dir
            / "validation_search_results.csv",

        "selected":
            run_dir
            / "selected_models.csv",

        "metrics":
            run_dir
            / "test_metrics.csv",

        "predictions":
            run_dir
            / "test_predictions.csv",

        "report":
            run_dir
            / "run_report.json",
    }


def run_is_complete(
    protocol,
    seed,
):

    files = run_files(
        protocol,
        seed,
    )

    required = [
        files["validation"],
        files["selected"],
        files["metrics"],
        files["predictions"],
        files["report"],
    ]

    if not all(
        p.exists()
        for p in required
    ):
        return False

    try:

        metrics = pd.read_csv(
            files["metrics"]
        )

        selected = pd.read_csv(
            files["selected"]
        )

        predictions = pd.read_csv(
            files["predictions"]
        )

        if len(metrics) != len(MODEL_GRIDS):
            return False

        if len(selected) != len(MODEL_GRIDS):
            return False

        if set(
            metrics["model"]
        ) != set(
            MODEL_GRIDS.keys()
        ):
            return False

        if predictions.empty:
            return False

        with open(
            files["report"],
            "r",
            encoding="utf-8",
        ) as f:

            report = json.load(f)

        if report.get("status") != "PASS":
            return False

        return True

    except Exception:
        return False


def load_completed_run(
    protocol,
    seed,
):

    files = run_files(
        protocol,
        seed,
    )

    return (
        pd.read_csv(
            files["validation"]
        ),
        pd.read_csv(
            files["selected"]
        ),
        pd.read_csv(
            files["metrics"]
        ),
        pd.read_csv(
            files["predictions"]
        ),
    )


# ======================================================================================================================
# SINGLE PROTOCOL × SEED EXPERIMENT
# ======================================================================================================================

def run_protocol_seed(
    protocol,
    seed,
    data,
):

    set_seed(seed)

    run_start = time.perf_counter()

    files = run_files(
        protocol,
        seed,
    )

    files["dir"].mkdir(
        parents=True,
        exist_ok=True,
    )

    train = data["train"]
    val = data["validation"]
    test = data["test"]

    X_train = train[
        FEATURES
    ].to_numpy(
        dtype=float
    )

    y_train = train[
        LABEL
    ].to_numpy(
        dtype=int
    )

    X_val = val[
        FEATURES
    ].to_numpy(
        dtype=float
    )

    y_val = val[
        LABEL
    ].to_numpy(
        dtype=int
    )

    X_test = test[
        FEATURES
    ].to_numpy(
        dtype=float
    )

    y_test = test[
        LABEL
    ].to_numpy(
        dtype=int
    )

    validation_rows = []
    selected_rows = []
    test_rows = []
    prediction_frames = []

    selected_for_protocol = {}

    # ------------------------------------------------------------------------------------------------------------------
    # VALIDATION SEARCH
    # ------------------------------------------------------------------------------------------------------------------

    for model_name, grid in MODEL_GRIDS.items():

        print(
            f"\n{model_name}"
        )

        candidates = []

        for (
            config_id,
            params,
        ) in enumerate(
            grid,
            start=1,
        ):

            fit_start = (
                time.perf_counter()
            )

            model = make_model(
                model_name,
                params,
                seed,
            )

            model.fit(
                X_train,
                y_train,
            )

            validation_fit_seconds = (
                time.perf_counter()
                - fit_start
            )

            val_prob = probabilities(
                model,
                X_val,
            )

            val_pred = (
                val_prob
                >= CLASSIFICATION_THRESHOLD
            ).astype(int)

            metrics = compute_metrics(
                y_val,
                val_pred,
                val_prob,
            )

            row = {
                "protocol":
                    protocol,

                "seed":
                    seed,

                "model":
                    model_name,

                "config_id":
                    config_id,

                "params":
                    json.dumps(
                        params,
                        sort_keys=True,
                    ),

                "classification_threshold":
                    CLASSIFICATION_THRESHOLD,

                "validation_fit_seconds":
                    validation_fit_seconds,

                **metrics,
            }

            validation_rows.append(
                row
            )

            candidates.append(
                (
                    row,
                    params,
                )
            )

            print(
                f"  config {config_id}: "
                f"MCC={metrics['mcc']:.4f} | "
                f"BA={metrics['balanced_accuracy']:.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"ROC-AUC={metrics['roc_auc']:.4f} | "
                f"PR-AUC={metrics['pr_auc']:.4f}"
            )

        (
            best_row,
            best_params,
        ) = max(
            candidates,
            key=lambda item:
                selection_key(
                    item[0]
                ),
        )

        selected_for_protocol[
            model_name
        ] = {
            "params":
                best_params,

            "config_id":
                best_row[
                    "config_id"
                ],

            "validation_metrics":
                best_row,
        }

        selected_rows.append(
            {
                "protocol":
                    protocol,

                "seed":
                    seed,

                "model":
                    model_name,

                "selected_config_id":
                    best_row[
                        "config_id"
                    ],

                "params":
                    best_row[
                        "params"
                    ],

                "classification_threshold":
                    CLASSIFICATION_THRESHOLD,

                "validation_mcc":
                    best_row[
                        "mcc"
                    ],

                "validation_balanced_accuracy":
                    best_row[
                        "balanced_accuracy"
                    ],

                "validation_f1":
                    best_row[
                        "f1"
                    ],

                "validation_roc_auc":
                    best_row[
                        "roc_auc"
                    ],

                "validation_pr_auc":
                    best_row[
                        "pr_auc"
                    ],
            }
        )

        print(
            f"  SELECTED config "
            f"{best_row['config_id']} "
            f"using validation MCC."
        )

    # ------------------------------------------------------------------------------------------------------------------
    # TEST EVALUATION
    # ------------------------------------------------------------------------------------------------------------------

    print(
        "\nTEST EVALUATION"
    )

    for (
        model_name,
        selected_info,
    ) in selected_for_protocol.items():

        params = (
            selected_info[
                "params"
            ]
        )

        selected_config_id = (
            selected_info[
                "config_id"
            ]
        )

        validation_metrics = (
            selected_info[
                "validation_metrics"
            ]
        )

        fit_start = (
            time.perf_counter()
        )

        # Important:
        # Refit on TRAIN only.
        # Validation is not merged into training.
        model = make_model(
            model_name,
            params,
            seed,
        )

        model.fit(
            X_train,
            y_train,
        )

        final_fit_seconds = (
            time.perf_counter()
            - fit_start
        )

        inference_start = (
            time.perf_counter()
        )

        test_prob = probabilities(
            model,
            X_test,
        )

        inference_seconds = (
            time.perf_counter()
            - inference_start
        )

        test_pred = (
            test_prob
            >= CLASSIFICATION_THRESHOLD
        ).astype(int)

        metrics = compute_metrics(
            y_test,
            test_pred,
            test_prob,
        )

        row = {
            "protocol":
                protocol,

            "seed":
                seed,

            "model":
                model_name,

            "selected_config_id":
                selected_config_id,

            "params":
                json.dumps(
                    params,
                    sort_keys=True,
                ),

            "classification_threshold":
                CLASSIFICATION_THRESHOLD,

            "train_samples":
                len(train),

            "validation_samples":
                len(val),

            "test_samples":
                len(test),

            "test_prevalence":
                float(
                    y_test.mean()
                ),

            "validation_mcc":
                validation_metrics[
                    "mcc"
                ],

            "validation_balanced_accuracy":
                validation_metrics[
                    "balanced_accuracy"
                ],

            "validation_f1":
                validation_metrics[
                    "f1"
                ],

            "validation_roc_auc":
                validation_metrics[
                    "roc_auc"
                ],

            "validation_pr_auc":
                validation_metrics[
                    "pr_auc"
                ],

            "fit_seconds":
                final_fit_seconds,

            "inference_seconds":
                inference_seconds,

            **metrics,
        }

        test_rows.append(
            row
        )

        print(
            f"{model_name:<24} "
            f"BA={metrics['balanced_accuracy']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"MCC={metrics['mcc']:.4f} | "
            f"ROC-AUC={metrics['roc_auc']:.4f} | "
            f"PR-AUC={metrics['pr_auc']:.4f}"
        )

        pred = pd.DataFrame(
            {
                "protocol":
                    protocol,

                "seed":
                    seed,

                "model":
                    model_name,

                "sample_uid":
                    sample_identifier(
                        test
                    ).values,

                "y_true":
                    y_test,

                "y_pred":
                    test_pred,

                "prob_retained":
                    test_prob,
            }
        )

        for c in [
            "country",
            "city",
            "city_harmonized",
            "split_harmonized",
            "tile_id",
            "tile_id_harmonized",
            "source_idx",
            "source_idx_harmonized",
        ]:

            if c in test.columns:

                pred[c] = (
                    test[c].values
                )

        prediction_frames.append(
            pred
        )

    # ------------------------------------------------------------------------------------------------------------------
    # SAVE RUN
    # ------------------------------------------------------------------------------------------------------------------

    validation_df = pd.DataFrame(
        validation_rows
    )

    selected_df = pd.DataFrame(
        selected_rows
    )

    test_df = pd.DataFrame(
        test_rows
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    validation_df.to_csv(
        files["validation"],
        index=False,
    )

    selected_df.to_csv(
        files["selected"],
        index=False,
    )

    test_df.to_csv(
        files["metrics"],
        index=False,
    )

    predictions_df.to_csv(
        files["predictions"],
        index=False,
    )

    runtime_seconds = (
        time.perf_counter()
        - run_start
    )

    checks = {
        "four_models_evaluated":
            len(test_df)
            == len(MODEL_GRIDS),

        "four_models_selected":
            len(selected_df)
            == len(MODEL_GRIDS),

        "all_predictions_present":
            not predictions_df.empty,

        "all_test_labels_binary":
            set(
                predictions_df[
                    "y_true"
                ].unique()
            ).issubset(
                {0, 1}
            ),

        "all_probabilities_finite":
            np.isfinite(
                predictions_df[
                    "prob_retained"
                ]
            ).all(),

        "probabilities_in_unit_interval":
            predictions_df[
                "prob_retained"
            ].between(
                0,
                1,
            ).all(),

        "test_not_used_for_tuning":
            True,

        "threshold_fixed_at_0_5":
            (
                CLASSIFICATION_THRESHOLD
                == 0.5
            ),
    }

    all_pass = all(
        checks.values()
    )

    report = {
        "script":
            "66_train_evaluate_cross_country_baselines.py",

        "protocol":
            protocol,

        "seed":
            seed,

        "features":
            FEATURES,

        "label":
            LABEL,

        "models":
            list(
                MODEL_GRIDS.keys()
            ),

        "classification_threshold":
            CLASSIFICATION_THRESHOLD,

        "selection_partition":
            "validation_only",

        "primary_selection_metric":
            "MCC",

        "selection_tie_breakers": [
            "PR-AUC",
            "balanced_accuracy",
            "F1",
        ],

        "test_used_for_selection":
            False,

        "runtime_seconds":
            runtime_seconds,

        "checks":
            checks,

        "status":
            (
                "PASS"
                if all_pass
                else "CHECK"
            ),
    }

    with open(
        files["report"],
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            default=json_default,
        )

    if not all_pass:

        raise RuntimeError(
            f"Audit failed for "
            f"{protocol}, seed {seed}."
        )

    return (
        validation_df,
        selected_df,
        test_df,
        predictions_df,
    )


# ======================================================================================================================
# AGGREGATION
# ======================================================================================================================

def aggregate_summary(
    all_metrics,
):

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "npv",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
        "fit_seconds",
        "inference_seconds",
    ]

    rows = []

    grouped = all_metrics.groupby(
        [
            "protocol",
            "model",
        ],
        sort=False,
    )

    for (
        protocol,
        model,
    ), g in grouped:

        row = {
            "protocol":
                protocol,

            "model":
                model,

            "n_runs":
                int(
                    g["seed"].nunique()
                ),
        }

        for metric in metric_columns:

            values = pd.to_numeric(
                g[metric],
                errors="coerce",
            )

            row[
                f"{metric}_mean"
            ] = values.mean()

            row[
                f"{metric}_std"
            ] = values.std(
                ddof=1
            )

            row[
                f"{metric}_min"
            ] = values.min()

            row[
                f"{metric}_max"
            ] = values.max()

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ======================================================================================================================
# COMPLETENESS AUDIT
# ======================================================================================================================

def build_completeness_table(
    all_metrics,
    seeds,
):

    rows = []

    for protocol in PROTOCOLS:

        for model in MODEL_GRIDS:

            subset = all_metrics[
                (
                    all_metrics[
                        "protocol"
                    ]
                    == protocol
                )
                &
                (
                    all_metrics[
                        "model"
                    ]
                    == model
                )
            ]

            present = sorted(
                subset[
                    "seed"
                ]
                .astype(int)
                .unique()
                .tolist()
            )

            missing = [
                seed
                for seed in seeds
                if seed not in present
            ]

            rows.append(
                {
                    "protocol":
                        protocol,

                    "model":
                        model,

                    "expected_runs":
                        len(seeds),

                    "observed_runs":
                        len(present),

                    "present_seeds":
                        ",".join(
                            map(
                                str,
                                present,
                            )
                        ),

                    "missing_seeds":
                        ",".join(
                            map(
                                str,
                                missing,
                            )
                        ),

                    "complete":
                        len(missing) == 0,
                }
            )

    return pd.DataFrame(
        rows
    )


# ======================================================================================================================
# AGGREGATE FIGURES
# ======================================================================================================================

def save_aggregate_figures(
    summary_df,
):

    FIG.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = [
        "balanced_accuracy",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
    ]

    for metric in metrics:

        mean_col = (
            f"{metric}_mean"
        )

        std_col = (
            f"{metric}_std"
        )

        pivot_mean = (
            summary_df
            .pivot(
                index="protocol",
                columns="model",
                values=mean_col,
            )
            .reindex(
                PROTOCOLS
            )
        )

        pivot_std = (
            summary_df
            .pivot(
                index="protocol",
                columns="model",
                values=std_col,
            )
            .reindex(
                PROTOCOLS
            )
        )

        fig, ax = plt.subplots(
            figsize=(11.5, 6.2)
        )

        x = np.arange(
            len(
                pivot_mean.index
            )
        )

        n_models = len(
            pivot_mean.columns
        )

        width = (
            0.8
            / max(
                n_models,
                1,
            )
        )

        for i, model in enumerate(
            pivot_mean.columns
        ):

            offset = (
                i
                - (
                    n_models - 1
                )
                / 2
            ) * width

            means = (
                pivot_mean[
                    model
                ].values
            )

            errors = (
                pivot_std[
                    model
                ]
                .fillna(0)
                .values
            )

            ax.bar(
                x + offset,
                means,
                width=width,
                yerr=errors,
                capsize=2,
                label=model,
            )

        ax.set_xticks(
            x,
            labels=pivot_mean.index,
            rotation=25,
            ha="right",
        )

        ax.set_ylabel(
            metric
            .replace(
                "_",
                " ",
            )
            .title()
        )

        ax.set_title(
            "Classical baselines: "
            f"{metric.replace('_', ' ')} "
            "across repeated runs"
        )

        ax.legend(
            fontsize=8
        )

        fig.tight_layout()

        stem = (
            FIG
            / f"aggregate__{metric}"
        )

        fig.savefig(
            stem.with_suffix(
                ".png"
            ),
            dpi=300,
        )

        fig.savefig(
            stem.with_suffix(
                ".pdf"
            )
        )

        plt.close(
            fig
        )


# ======================================================================================================================
# MAIN
# ======================================================================================================================

def main():

    args = parse_args()

    warnings.filterwarnings(
        "ignore",
        category=ConvergenceWarning,
    )

    if args.mode == "smoke":

        seeds = (
            SMOKE_SEEDS
        )

    else:

        seeds = (
            FULL_SEEDS
        )

    banner(
        "CARTOGEN — REPEATED CROSS-COUNTRY "
        "CLASSICAL ROAD-SELECTION BASELINES"
    )

    print(
        f"Protocols: {PROTOCOL_ROOT}"
    )

    print(
        f"Output:    {OUT}"
    )

    print(
        f"Mode:      {args.mode}"
    )

    print(
        f"Seeds:     {seeds}"
    )

    print(
        "Models:    "
        + ", ".join(
            MODEL_GRIDS.keys()
        )
    )

    print(
        f"Threshold: {CLASSIFICATION_THRESHOLD}"
    )

    print(
        "Resume:    "
        + (
            "OFF (--overwrite)"
            if args.overwrite
            else "ON"
        )
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIG.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------------------------------------------------------

    section(
        "1. VERIFY PROTOCOL DATASETS"
    )

    loaded = load_all_protocols()

    # ------------------------------------------------------------------------------------------------------------------
    # EXPERIMENT MATRIX
    # ------------------------------------------------------------------------------------------------------------------

    total_runs = (
        len(PROTOCOLS)
        * len(seeds)
    )

    section(
        "2. RUN EXPERIMENT MATRIX"
    )

    print(
        f"Protocol × seed runs: "
        f"{total_runs}"
    )

    print(
        f"Model evaluations:    "
        f"{total_runs * len(MODEL_GRIDS)}"
    )

    all_validation = []
    all_selected = []
    all_metrics = []
    all_predictions = []

    experiment_start = (
        time.perf_counter()
    )

    run_number = 0

    for seed in seeds:

        for protocol in PROTOCOLS:

            run_number += 1

            print(
                "\n"
                + "=" * 120
            )

            print(
                f"RUN {run_number}/{total_runs} | "
                f"{protocol.upper()} | "
                f"SEED {seed}"
            )

            print(
                "=" * 120
            )

            if (
                not args.overwrite
                and run_is_complete(
                    protocol,
                    seed,
                )
            ):

                print(
                    "[RESUME] Completed run found. "
                    "Loading existing outputs."
                )

                (
                    validation_df,
                    selected_df,
                    test_df,
                    predictions_df,
                ) = load_completed_run(
                    protocol,
                    seed,
                )

            else:

                (
                    validation_df,
                    selected_df,
                    test_df,
                    predictions_df,
                ) = run_protocol_seed(
                    protocol=protocol,
                    seed=seed,
                    data=loaded[
                        protocol
                    ],
                )

            all_validation.append(
                validation_df
            )

            all_selected.append(
                selected_df
            )

            all_metrics.append(
                test_df
            )

            all_predictions.append(
                predictions_df
            )

    # ------------------------------------------------------------------------------------------------------------------
    # COMBINE
    # ------------------------------------------------------------------------------------------------------------------

    section(
        "3. COMBINE ALL RUNS"
    )

    all_validation_df = (
        pd.concat(
            all_validation,
            ignore_index=True,
        )
    )

    all_selected_df = (
        pd.concat(
            all_selected,
            ignore_index=True,
        )
    )

    all_metrics_df = (
        pd.concat(
            all_metrics,
            ignore_index=True,
        )
    )

    all_predictions_df = (
        pd.concat(
            all_predictions,
            ignore_index=True,
        )
    )

    all_validation_df.to_csv(
        OUT
        / "all_runs_validation_search.csv",
        index=False,
    )

    all_selected_df.to_csv(
        OUT
        / "all_runs_selected_models.csv",
        index=False,
    )

    all_metrics_df.to_csv(
        OUT
        / "all_runs_metrics.csv",
        index=False,
    )

    all_predictions_df.to_csv(
        OUT
        / "all_runs_predictions.csv",
        index=False,
    )

    print(
        f"Validation rows: "
        f"{len(all_validation_df):,}"
    )

    print(
        f"Selected rows:   "
        f"{len(all_selected_df):,}"
    )

    print(
        f"Metric rows:     "
        f"{len(all_metrics_df):,}"
    )

    print(
        f"Prediction rows: "
        f"{len(all_predictions_df):,}"
    )

    # ------------------------------------------------------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------------------------------------------------------

    section(
        "4. REPEATED-RUN SUMMARY"
    )

    summary_df = (
        aggregate_summary(
            all_metrics_df
        )
    )

    summary_df.to_csv(
        OUT
        / "summary_10_runs.csv",
        index=False,
    )

    display_columns = [
        "protocol",
        "model",
        "n_runs",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "f1_mean",
        "f1_std",
        "mcc_mean",
        "mcc_std",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "pr_auc_std",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------------------------------------------------------
    # COMPLETENESS
    # ------------------------------------------------------------------------------------------------------------------

    section(
        "5. RUN COMPLETENESS AUDIT"
    )

    completeness_df = (
        build_completeness_table(
            all_metrics_df,
            seeds,
        )
    )

    completeness_df.to_csv(
        OUT
        / "run_completeness.csv",
        index=False,
    )

    print(
        completeness_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------------------------------------------------------
    # DETERMINISM / VARIANCE DIAGNOSTIC
    # ------------------------------------------------------------------------------------------------------------------

    section(
        "6. BETWEEN-SEED VARIANCE DIAGNOSTIC"
    )

    variance_rows = []

    for (
        protocol,
        model,
    ), g in all_metrics_df.groupby(
        [
            "protocol",
            "model",
        ],
        sort=False,
    ):

        mcc_std = (
            pd.to_numeric(
                g["mcc"],
                errors="coerce",
            )
            .std(
                ddof=1
            )
        )

        ba_std = (
            pd.to_numeric(
                g[
                    "balanced_accuracy"
                ],
                errors="coerce",
            )
            .std(
                ddof=1
            )
        )

        prediction_variation = (
            np.nan
        )

        if len(seeds) > 1:

            subset = (
                all_predictions_df[
                    (
                        all_predictions_df[
                            "protocol"
                        ]
                        == protocol
                    )
                    &
                    (
                        all_predictions_df[
                            "model"
                        ]
                        == model
                    )
                ]
            )

            if not subset.empty:

                pred_pivot = (
                    subset
                    .pivot_table(
                        index="sample_uid",
                        columns="seed",
                        values="prob_retained",
                        aggfunc="first",
                    )
                )

                if (
                    pred_pivot.shape[1]
                    > 1
                ):

                    prediction_variation = (
                        pred_pivot
                        .std(
                            axis=1,
                            ddof=1,
                        )
                        .mean()
                    )

        variance_rows.append(
            {
                "protocol":
                    protocol,

                "model":
                    model,

                "mcc_std":
                    mcc_std,

                "balanced_accuracy_std":
                    ba_std,

                "mean_prediction_std_across_seeds":
                    prediction_variation,

                "effectively_deterministic":
                    (
                        (
                            pd.isna(mcc_std)
                            or abs(mcc_std)
                            < 1e-12
                        )
                        and
                        (
                            pd.isna(ba_std)
                            or abs(ba_std)
                            < 1e-12
                        )
                    ),
            }
        )

    variance_df = pd.DataFrame(
        variance_rows
    )

    variance_df.to_csv(
        OUT
        / "between_seed_variance.csv",
        index=False,
    )

    print(
        variance_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------------------------------------------------------
    # FIGURES
    # ------------------------------------------------------------------------------------------------------------------

    section(
        "7. GENERATE AGGREGATE FIGURES"
    )

    save_aggregate_figures(
        summary_df
    )

    print(
        f"Figures: {FIG}"
    )

    # ------------------------------------------------------------------------------------------------------------------
    # FINAL AUDIT
    # ------------------------------------------------------------------------------------------------------------------

    section(
        "8. FINAL BASELINE AUDIT"
    )

    expected_metric_rows = (
        len(PROTOCOLS)
        * len(seeds)
        * len(MODEL_GRIDS)
    )

    expected_selected_rows = (
        expected_metric_rows
    )

    expected_validation_rows = (
        len(PROTOCOLS)
        * len(seeds)
        * sum(
            len(grid)
            for grid
            in MODEL_GRIDS.values()
        )
    )

    expected_predictions = 0

    for protocol in PROTOCOLS:

        expected_predictions += (
            len(
                loaded[
                    protocol
                ][
                    "test"
                ]
            )
            * len(seeds)
            * len(MODEL_GRIDS)
        )

    checks = {

        "six_protocols_evaluated":
            all_metrics_df[
                "protocol"
            ].nunique()
            == len(PROTOCOLS),

        "all_requested_seeds_present":
            set(
                all_metrics_df[
                    "seed"
                ]
                .astype(int)
                .unique()
            )
            == set(seeds),

        "four_models_per_protocol_seed":
            len(
                all_metrics_df
            )
            == expected_metric_rows,

        "all_validation_search_rows_present":
            len(
                all_validation_df
            )
            == expected_validation_rows,

        "all_selected_model_rows_present":
            len(
                all_selected_df
            )
            == expected_selected_rows,

        "all_predictions_present":
            len(
                all_predictions_df
            )
            == expected_predictions,

        "unique_metric_keys":
            not all_metrics_df.duplicated(
                subset=[
                    "protocol",
                    "seed",
                    "model",
                ]
            ).any(),

        "unique_selected_keys":
            not all_selected_df.duplicated(
                subset=[
                    "protocol",
                    "seed",
                    "model",
                ]
            ).any(),

        "all_test_labels_binary":
            set(
                all_predictions_df[
                    "y_true"
                ].unique()
            ).issubset(
                {0, 1}
            ),

        "all_probabilities_finite":
            np.isfinite(
                all_predictions_df[
                    "prob_retained"
                ]
            ).all(),

        "probabilities_in_unit_interval":
            all_predictions_df[
                "prob_retained"
            ].between(
                0,
                1,
            ).all(),

        "run_matrix_complete":
            bool(
                completeness_df[
                    "complete"
                ].all()
            ),

        "no_test_used_for_tuning":
            True,

        "threshold_fixed_at_0_5":
            (
                CLASSIFICATION_THRESHOLD
                == 0.5
            ),
    }

    for key, value in checks.items():

        print(
            f"{key:<50} "
            f"{'PASS' if value else 'FAIL'}"
        )

    all_pass = all(
        checks.values()
    )

    total_runtime = (
        time.perf_counter()
        - experiment_start
    )

    # ------------------------------------------------------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------------------------------------------------------

    report = {

        "script":
            "66_train_evaluate_cross_country_baselines.py",

        "mode":
            args.mode,

        "seeds":
            seeds,

        "n_seeds":
            len(seeds),

        "protocols":
            PROTOCOLS,

        "features":
            FEATURES,

        "label":
            LABEL,

        "models":
            MODEL_GRIDS,

        "selection": {
            "partition":
                "validation_only",

            "primary_metric":
                "MCC",

            "tie_breakers": [
                "PR-AUC",
                "balanced_accuracy",
                "F1",
            ],

            "classification_threshold":
                CLASSIFICATION_THRESHOLD,

            "test_used_for_selection":
                False,

            "validation_merged_into_training":
                False,
        },

        "repeated_run_design": {
            "spatial_partitions":
                "fixed_across_seeds",

            "same_seed_across_models":
                True,

            "smoke_seeds":
                SMOKE_SEEDS,

            "full_seeds":
                FULL_SEEDS,

            "note":
                (
                    "Deterministic methods may produce "
                    "zero between-seed variance; repeated "
                    "rows are retained to preserve the "
                    "paired experimental matrix."
                ),
        },

        "expected_counts": {
            "metric_rows":
                expected_metric_rows,

            "selected_rows":
                expected_selected_rows,

            "validation_search_rows":
                expected_validation_rows,

            "prediction_rows":
                expected_predictions,
        },

        "observed_counts": {
            "metric_rows":
                len(
                    all_metrics_df
                ),

            "selected_rows":
                len(
                    all_selected_df
                ),

            "validation_search_rows":
                len(
                    all_validation_df
                ),

            "prediction_rows":
                len(
                    all_predictions_df
                ),
        },

        "runtime_seconds":
            total_runtime,

        "checks":
            checks,

        "status":
            (
                "PASS"
                if all_pass
                else "CHECK"
            ),
    }

    with open(
        OUT
        / "baseline_evaluation_report.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            default=json_default,
        )

    if not all_pass:

        raise RuntimeError(
            "Repeated baseline audit failed. "
            "Inspect output files."
        )

    # ------------------------------------------------------------------------------------------------------------------
    # COMPLETE
    # ------------------------------------------------------------------------------------------------------------------

    banner(
        "SCRIPT 66 COMPLETE"
    )

    print(
        f"Mode:                  {args.mode}"
    )

    print(
        f"Seeds:                 {seeds}"
    )

    print(
        f"All validation search: "
        f"{OUT / 'all_runs_validation_search.csv'}"
    )

    print(
        f"All selected models:   "
        f"{OUT / 'all_runs_selected_models.csv'}"
    )

    print(
        f"All run metrics:       "
        f"{OUT / 'all_runs_metrics.csv'}"
    )

    print(
        f"All predictions:       "
        f"{OUT / 'all_runs_predictions.csv'}"
    )

    print(
        f"Repeated-run summary:  "
        f"{OUT / 'summary_10_runs.csv'}"
    )

    print(
        f"Completeness:          "
        f"{OUT / 'run_completeness.csv'}"
    )

    print(
        f"Variance diagnostic:   "
        f"{OUT / 'between_seed_variance.csv'}"
    )

    print(
        f"Report:                "
        f"{OUT / 'baseline_evaluation_report.json'}"
    )

    print(
        f"Figures:               {FIG}"
    )

    print(
        f"Runtime:               "
        f"{total_runtime / 60:.2f} min"
    )

    print(
        "\n[PASS] REPEATED CLASSICAL "
        "CROSS-COUNTRY BASELINES COMPLETE."
    )

    if args.mode == "smoke":

        print(
            "\nSMOKE TEST COMPLETE."
        )

        print(
            "If the outputs are correct, run:"
        )

        print(
            "\npython scripts\\"
            "66_train_evaluate_cross_country_baselines.py "
            "--mode full"
        )

    else:

        print(
            "\n10-RUN BASELINE EXPERIMENT COMPLETE."
        )

        print(
            "Keep these outputs fixed for the final "
            "paired statistical analysis."
        )


if __name__ == "__main__":
    main()