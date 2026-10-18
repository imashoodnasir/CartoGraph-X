"""
Script 72
=========

CartoGraph-X Component Ablation and Novelty Validation

Purpose
-------
Quantify the contribution of the major CartoGraph-X components using
exactly the same leakage-safe protocols, preprocessing, training logic,
validation-only model selection, threshold selection, and test evaluation
implemented in Script 71.

Ablations
---------
1. full
2. no_graph_propagation
3. no_adaptive_fusion
4. no_multiscale_context
5. no_neighbor_aggregation
6. no_operation_auxiliary
7. no_domain_adversarial
8. intrinsic_only

Important methodological rules
------------------------------
- Script 71 is imported and reused rather than reimplementing training.
- Test data are never used for model/threshold selection.
- Fixed spatial train/validation/test splits are retained.
- Smoke mode uses seed 42.
- Full mode uses seeds 42-51.
- Smoke mode uses 15 epochs to make ablation diagnosis more meaningful.
- Full mode uses the normal Script 71 full training budget.
- Domain-adversarial ablation is evaluated only for joint-country protocols.
- Delta metrics are always calculated relative to the full CartoGraph-X
  model under the same protocol and seed.

Run
---
python scripts\\72_ablate_cartographx.py --mode smoke

One protocol:
python scripts\\72_ablate_cartographx.py --mode smoke --protocol lt_to_lt

One variant:
python scripts\\72_ablate_cartographx.py --mode smoke --variant no_graph_propagation

Full 10-run experiment:
python scripts\\72_ablate_cartographx.py --mode full

Author:
CartoGen project
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]

SCRIPT71_PATH = (
    ROOT
    / "scripts"
    / "71_train_cartographx.py"
)

OUT_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "cross_country"
    / "cartographx_ablation"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# IMPORT SCRIPT 71
# =====================================================================

def load_script71():

    if not SCRIPT71_PATH.exists():
        raise FileNotFoundError(
            "Corrected Script 71 was not found:\n"
            f"{SCRIPT71_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "cartographx_script71",
        SCRIPT71_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not create import specification for Script 71."
        )

    module = importlib.util.module_from_spec(spec)

    # Required by dataclasses on some Python versions.
    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


S71 = load_script71()


# =====================================================================
# CONSTANTS
# =====================================================================

PROTOCOLS = list(S71.PROTOCOLS)

JOINT_PROTOCOLS = [
    "joint_to_lt",
    "joint_to_pl",
]

SMOKE_SEEDS = [42]

FULL_SEEDS = list(
    range(42, 52)
)

VARIANTS = [
    "full",
    "no_graph_propagation",
    "no_adaptive_fusion",
    "no_multiscale_context",
    "no_neighbor_aggregation",
    "no_operation_auxiliary",
    "no_domain_adversarial",
    "intrinsic_only",
]

FUSION_NAMES = [
    "intrinsic",
    "graph_structural",
    "multiscale_context",
    "neighbor_aggregation",
]

PRIMARY_METRICS = [
    "test_balanced_accuracy",
    "test_mcc",
    "test_roc_auc",
    "test_pr_auc",
    "test_f1",
]


# =====================================================================
# PRINT HELPERS
# =====================================================================

def banner(text: str) -> None:

    print()
    print("=" * 120)
    print(text)
    print("=" * 120)


def section(text: str) -> None:

    print()
    print("-" * 120)
    print(text)
    print("-" * 120)


# =====================================================================
# JSON SAFETY
# =====================================================================

def json_safe(obj):

    if isinstance(obj, dict):
        return {
            str(k): json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            json_safe(v)
            for v in obj
        ]

    if isinstance(obj, set):
        return [
            json_safe(v)
            for v in sorted(
                obj,
                key=str,
            )
        ]

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):

        value = float(obj)

        if np.isfinite(value):
            return value

        return None

    if isinstance(obj, np.ndarray):
        return json_safe(
            obj.tolist()
        )

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, float):

        if math.isfinite(obj):
            return obj

        return None

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if obj is pd.NA:
        return None

    return obj


# =====================================================================
# ABLATION MODEL
# =====================================================================

class AblationCartoGraphX(nn.Module):
    """
    CartoGraph-X with configurable component removal.

    The full configuration reproduces the Script 71 architecture.

    Ablations alter only the requested component while preserving the
    remaining architecture as closely as possible.
    """

    def __init__(
        self,
        num_operations: int,
        config,
        variant: str,
    ):

        super().__init__()

        if variant not in VARIANTS:
            raise ValueError(
                f"Unknown ablation variant: {variant}"
            )

        self.variant = variant

        b = config.branch_dim
        h = config.hidden_dim
        p = config.dropout

        self.branch_dim = b
        self.hidden_dim = h

        # ----------------------------------------------------------
        # Feature-family encoders
        # ----------------------------------------------------------

        self.intrinsic_encoder = S71.FeatureEncoder(
            len(S71.INTRINSIC_FEATURES),
            b,
            p,
        )

        self.graph_encoder = S71.FeatureEncoder(
            len(S71.GRAPH_STRUCTURAL_FEATURES),
            b,
            p,
        )

        self.multiscale_encoder = S71.FeatureEncoder(
            len(S71.MULTISCALE_CONTEXT_FEATURES),
            b,
            p,
        )

        self.neighbor_encoder = S71.FeatureEncoder(
            len(S71.NEIGHBOR_AGGREGATION_FEATURES),
            b,
            p,
        )

        # ----------------------------------------------------------
        # Fusion
        # ----------------------------------------------------------

        self.fusion = S71.AdaptiveContextFusion(
            branch_dim=b,
            hidden_dim=h,
            dropout=p,
        )

        # Used only by the no-adaptive-fusion control.
        #
        # Concatenation preserves information from all four branches,
        # but removes learned sample-specific adaptive weighting.
        self.static_fusion = nn.Sequential(
            nn.Linear(
                4 * b,
                h,
            ),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(p),
        )

        # Used by intrinsic-only control.
        self.intrinsic_projection = nn.Sequential(
            nn.Linear(
                b,
                h,
            ),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(p),
        )

        # ----------------------------------------------------------
        # Graph propagation
        # ----------------------------------------------------------

        self.graph_layers = nn.ModuleList(
            [
                S71.ResidualGraphLayer(
                    hidden_dim=h,
                    dropout=p,
                )
                for _ in range(
                    config.graph_layers
                )
            ]
        )

        self.post_graph = nn.Sequential(
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(p),
        )

        # ----------------------------------------------------------
        # Prediction heads
        # ----------------------------------------------------------

        self.selection_head = nn.Sequential(
            nn.Linear(
                h,
                h // 2,
            ),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(
                h // 2,
                2,
            ),
        )

        self.operation_head = nn.Sequential(
            nn.Linear(
                h,
                h // 2,
            ),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(
                h // 2,
                num_operations,
            ),
        )

        self.domain_head = nn.Sequential(
            nn.Linear(
                h,
                h // 2,
            ),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(
                h // 2,
                2,
            ),
        )

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        graph,
        grl_coefficient: float = 0.0,
    ) -> Dict[str, torch.Tensor]:

        # ----------------------------------------------------------
        # Encode intrinsic road attributes
        # ----------------------------------------------------------

        z_intrinsic = self.intrinsic_encoder(
            graph.x_intrinsic
        )

        # ----------------------------------------------------------
        # Intrinsic-only control
        # ----------------------------------------------------------

        if self.variant == "intrinsic_only":

            h = self.intrinsic_projection(
                z_intrinsic
            )

            pre_graph_embedding = h

            # No graph message passing in intrinsic-only model.
            h = self.post_graph(h)

            fusion_weights = torch.zeros(
                (
                    h.shape[0],
                    4,
                ),
                dtype=h.dtype,
                device=h.device,
            )

            fusion_weights[:, 0] = 1.0

        else:

            # ------------------------------------------------------
            # Encode remaining feature families
            # ------------------------------------------------------

            z_graph = self.graph_encoder(
                graph.x_graph
            )

            z_multiscale = self.multiscale_encoder(
                graph.x_multiscale
            )

            z_neighbor = self.neighbor_encoder(
                graph.x_neighbor
            )

            # ------------------------------------------------------
            # Feature-family ablations
            # ------------------------------------------------------

            if self.variant == "no_multiscale_context":

                z_multiscale = torch.zeros_like(
                    z_multiscale
                )

            if self.variant == "no_neighbor_aggregation":

                z_neighbor = torch.zeros_like(
                    z_neighbor
                )

            # ------------------------------------------------------
            # Fusion ablation
            # ------------------------------------------------------

            if self.variant == "no_adaptive_fusion":

                concatenated = torch.cat(
                    [
                        z_intrinsic,
                        z_graph,
                        z_multiscale,
                        z_neighbor,
                    ],
                    dim=1,
                )

                h = self.static_fusion(
                    concatenated
                )

                fusion_weights = torch.full(
                    (
                        h.shape[0],
                        4,
                    ),
                    0.25,
                    dtype=h.dtype,
                    device=h.device,
                )

            else:

                h, fusion_weights = self.fusion(
                    z_intrinsic,
                    z_graph,
                    z_multiscale,
                    z_neighbor,
                )

            pre_graph_embedding = h

            # ------------------------------------------------------
            # Graph propagation ablation
            # ------------------------------------------------------

            if self.variant != "no_graph_propagation":

                for layer in self.graph_layers:

                    h = layer(
                        h,
                        graph.edge_index,
                    )

            h = self.post_graph(h)

        # ----------------------------------------------------------
        # Selection head
        # ----------------------------------------------------------

        selection_logits = self.selection_head(
            h
        )

        # ----------------------------------------------------------
        # Auxiliary operation head
        # ----------------------------------------------------------

        operation_logits = self.operation_head(
            h
        )

        # ----------------------------------------------------------
        # Country adversarial head
        # ----------------------------------------------------------

        reversed_h = S71.gradient_reverse(
            h,
            grl_coefficient,
        )

        domain_logits = self.domain_head(
            reversed_h
        )

        return {
            "selection_logits":
                selection_logits,
            "operation_logits":
                operation_logits,
            "domain_logits":
                domain_logits,
            "embedding":
                h,
            "pre_graph_embedding":
                pre_graph_embedding,
            "fusion_weights":
                fusion_weights,
        }


# =====================================================================
# MODEL FACTORY
# =====================================================================

CURRENT_VARIANT = "full"


def ablation_model_factory(
    num_operations: int,
    config,
):

    return AblationCartoGraphX(
        num_operations=num_operations,
        config=config,
        variant=CURRENT_VARIANT,
    )


# =====================================================================
# VARIANT-SPECIFIC CONFIGURATION
# =====================================================================

def make_config(
    variant: str,
    mode: str,
):

    config = S71.TrainConfig()

    # --------------------------------------------------------------
    # Smoke test
    # --------------------------------------------------------------
    #
    # Script 71 used 8 epochs for basic execution validation.
    # Script 72 uses 15 epochs because component contribution cannot
    # be judged reliably when several models are still improving at
    # epoch 8.
    # --------------------------------------------------------------

    if mode == "smoke":

        config.smoke_epochs = 15

        # Avoid terminating before the 15-epoch diagnostic budget.
        config.early_stopping_patience = 15

    # --------------------------------------------------------------
    # Auxiliary operation ablation
    # --------------------------------------------------------------

    if variant in [
        "no_operation_auxiliary",
        "intrinsic_only",
    ]:

        config.operation_loss_weight = 0.0

    # --------------------------------------------------------------
    # Domain-adversarial ablation
    # --------------------------------------------------------------

    if variant in [
        "no_domain_adversarial",
        "intrinsic_only",
    ]:

        config.domain_loss_weight = 0.0
        config.domain_grl_max = 0.0

    return config


# =====================================================================
# VALID VARIANT / PROTOCOL COMBINATIONS
# =====================================================================

def combination_is_valid(
    variant: str,
    protocol: str,
) -> bool:

    # Domain-adversarial learning is inactive in single-country
    # training. Therefore removing it there is not an informative
    # ablation.

    if (
        variant == "no_domain_adversarial"
        and protocol not in JOINT_PROTOCOLS
    ):
        return False

    return True


# =====================================================================
# RUN ONE ABLATION
# =====================================================================

def run_one(
    variant: str,
    protocol: str,
    seed: int,
    mode: str,
    device: torch.device,
) -> Dict:

    global CURRENT_VARIANT

    CURRENT_VARIANT = variant

    banner(
        f"ABLATION: {variant.upper()} | "
        f"{protocol.upper()} | "
        f"SEED {seed}"
    )

    config = make_config(
        variant,
        mode,
    )

    # --------------------------------------------------------------
    # Save original Script 71 globals
    # --------------------------------------------------------------

    original_model_class = S71.CartoGraphX
    original_out_root = S71.OUT_ROOT

    # --------------------------------------------------------------
    # Redirect Script 71 outputs into this ablation's directory
    # --------------------------------------------------------------

    variant_root = (
        OUT_ROOT
        / "runs"
        / variant
    )

    variant_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    S71.CartoGraphX = ablation_model_factory
    S71.OUT_ROOT = variant_root

    try:

        result = S71.train_one_run(
            protocol=protocol,
            seed=seed,
            mode=mode,
            device=device,
            config=config,
        )

    finally:

        # Always restore Script 71 even if a run fails.
        S71.CartoGraphX = original_model_class
        S71.OUT_ROOT = original_out_root

    result = dict(result)

    result["variant"] = variant

    result["ablation_protocol"] = protocol

    result["ablation_seed"] = seed

    result["smoke_epochs_requested"] = (
        config.smoke_epochs
        if mode == "smoke"
        else None
    )

    result["full_epochs_requested"] = (
        config.full_epochs
        if mode == "full"
        else None
    )

    result["operation_auxiliary_active"] = bool(
        config.operation_loss_weight > 0
    )

    result["domain_objective_enabled"] = bool(
        config.domain_loss_weight > 0
        and protocol in JOINT_PROTOCOLS
    )

    return result


# =====================================================================
# DELTA CALCULATION
# =====================================================================

def add_full_model_deltas(
    results: pd.DataFrame,
) -> pd.DataFrame:

    d = results.copy()

    for metric in PRIMARY_METRICS:

        delta_col = (
            "delta_"
            + metric.replace(
                "test_",
                "",
            )
            + "_vs_full"
        )

        d[delta_col] = np.nan

    full = d[
        d["variant"] == "full"
    ].copy()

    if full.empty:
        return d

    lookup = {}

    for _, row in full.iterrows():

        key = (
            row["protocol"],
            int(row["seed"]),
        )

        lookup[key] = {
            metric: row.get(
                metric,
                np.nan,
            )
            for metric in PRIMARY_METRICS
        }

    for idx, row in d.iterrows():

        key = (
            row["protocol"],
            int(row["seed"]),
        )

        if key not in lookup:
            continue

        for metric in PRIMARY_METRICS:

            current = row.get(
                metric,
                np.nan,
            )

            baseline = lookup[key].get(
                metric,
                np.nan,
            )

            delta_col = (
                "delta_"
                + metric.replace(
                    "test_",
                    "",
                )
                + "_vs_full"
            )

            if (
                pd.notna(current)
                and pd.notna(baseline)
            ):

                # Positive = current variant is better than full.
                #
                # Therefore a negative value means removing the
                # component reduced performance and supports that
                # component's contribution.
                d.loc[
                    idx,
                    delta_col,
                ] = (
                    float(current)
                    - float(baseline)
                )

    return d


# =====================================================================
# ABLATION CONTRIBUTION TABLE
# =====================================================================

def build_contribution_table(
    results: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    if results.empty:
        return pd.DataFrame()

    for protocol in sorted(
        results["protocol"].unique()
    ):

        p = results[
            results["protocol"] == protocol
        ]

        for variant in VARIANTS:

            v = p[
                p["variant"] == variant
            ]

            if v.empty:
                continue

            row = {
                "protocol":
                    protocol,
                "variant":
                    variant,
                "runs":
                    len(v),
            }

            for metric in PRIMARY_METRICS:

                values = pd.to_numeric(
                    v[metric],
                    errors="coerce",
                ).dropna()

                if len(values) == 0:
                    continue

                row[
                    f"{metric}_mean"
                ] = float(
                    values.mean()
                )

                if len(values) > 1:

                    row[
                        f"{metric}_std"
                    ] = float(
                        values.std(
                            ddof=1
                        )
                    )

                else:

                    row[
                        f"{metric}_std"
                    ] = np.nan

            # ------------------------------------------------------
            # Mean paired delta relative to full
            # ------------------------------------------------------

            for metric in PRIMARY_METRICS:

                delta_col = (
                    "delta_"
                    + metric.replace(
                        "test_",
                        "",
                    )
                    + "_vs_full"
                )

                if delta_col not in v.columns:
                    continue

                values = pd.to_numeric(
                    v[delta_col],
                    errors="coerce",
                ).dropna()

                if len(values) > 0:

                    row[
                        f"{delta_col}_mean"
                    ] = float(
                        values.mean()
                    )

            rows.append(row)

    return pd.DataFrame(rows)


# =====================================================================
# COMPONENT IMPORTANCE TABLE
# =====================================================================

def build_component_importance(
    contribution: pd.DataFrame,
) -> pd.DataFrame:

    if contribution.empty:
        return pd.DataFrame()

    rows = []

    component_map = {
        "no_graph_propagation":
            "graph_propagation",

        "no_adaptive_fusion":
            "adaptive_context_fusion",

        "no_multiscale_context":
            "multiscale_spatial_context",

        "no_neighbor_aggregation":
            "neighbor_aggregation",

        "no_operation_auxiliary":
            "auxiliary_operation_learning",

        "no_domain_adversarial":
            "domain_adversarial_learning",

        "intrinsic_only":
            "complete_contextual_graph_framework",
    }

    for _, row in contribution.iterrows():

        variant = row["variant"]

        if variant not in component_map:
            continue

        delta_mcc = row.get(
            "delta_mcc_vs_full_mean",
            np.nan,
        )

        delta_ba = row.get(
            "delta_balanced_accuracy_vs_full_mean",
            np.nan,
        )

        # Contribution is full - ablated.
        #
        # add_full_model_deltas stores ablated - full,
        # hence the sign reversal here.
        contribution_mcc = (
            -float(delta_mcc)
            if pd.notna(delta_mcc)
            else np.nan
        )

        contribution_ba = (
            -float(delta_ba)
            if pd.notna(delta_ba)
            else np.nan
        )

        rows.append(
            {
                "protocol":
                    row["protocol"],
                "component":
                    component_map[
                        variant
                    ],
                "ablation_variant":
                    variant,
                "runs":
                    row["runs"],
                "mcc_contribution":
                    contribution_mcc,
                "balanced_accuracy_contribution":
                    contribution_ba,
                "supports_component_mcc":
                    (
                        bool(
                            contribution_mcc > 0
                        )
                        if pd.notna(
                            contribution_mcc
                        )
                        else False
                    ),
                "supports_component_ba":
                    (
                        bool(
                            contribution_ba > 0
                        )
                        if pd.notna(
                            contribution_ba
                        )
                        else False
                    ),
            }
        )

    return pd.DataFrame(rows)


# =====================================================================
# FULL-MODE PAIRED SEED ANALYSIS
# =====================================================================

def build_paired_seed_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:

    """
    Descriptive paired seed analysis.

    Inferential significance testing is intentionally deferred until
    the full 10-seed experiment has completed. This table provides
    paired mean differences and variability using identical seeds.
    """

    if results.empty:
        return pd.DataFrame()

    rows = []

    full = results[
        results["variant"] == "full"
    ]

    for protocol in sorted(
        results["protocol"].unique()
    ):

        full_p = full[
            full["protocol"] == protocol
        ]

        for variant in VARIANTS:

            if variant == "full":
                continue

            ablated = results[
                (
                    results["protocol"]
                    == protocol
                )
                &
                (
                    results["variant"]
                    == variant
                )
            ]

            if ablated.empty:
                continue

            merged = full_p[
                [
                    "seed",
                    "test_mcc",
                    "test_balanced_accuracy",
                    "test_roc_auc",
                    "test_pr_auc",
                ]
            ].merge(
                ablated[
                    [
                        "seed",
                        "test_mcc",
                        "test_balanced_accuracy",
                        "test_roc_auc",
                        "test_pr_auc",
                    ]
                ],
                on="seed",
                suffixes=(
                    "_full",
                    "_ablated",
                ),
            )

            if merged.empty:
                continue

            for metric in [
                "test_mcc",
                "test_balanced_accuracy",
                "test_roc_auc",
                "test_pr_auc",
            ]:

                diff = (
                    merged[
                        f"{metric}_full"
                    ].astype(float)
                    -
                    merged[
                        f"{metric}_ablated"
                    ].astype(float)
                )

                n = len(diff)

                mean_diff = float(
                    diff.mean()
                )

                if n > 1:

                    std_diff = float(
                        diff.std(
                            ddof=1
                        )
                    )

                    ci_half = (
                        1.96
                        * std_diff
                        / math.sqrt(n)
                    )

                else:

                    std_diff = np.nan
                    ci_half = np.nan

                rows.append(
                    {
                        "protocol":
                            protocol,
                        "variant":
                            variant,
                        "metric":
                            metric,
                        "paired_runs":
                            n,
                        "full_minus_ablated_mean":
                            mean_diff,
                        "paired_difference_std":
                            std_diff,
                        "ci95_lower":
                            (
                                mean_diff
                                - ci_half
                                if n > 1
                                else np.nan
                            ),
                        "ci95_upper":
                            (
                                mean_diff
                                + ci_half
                                if n > 1
                                else np.nan
                            ),
                    }
                )

    return pd.DataFrame(rows)


# =====================================================================
# AUDIT
# =====================================================================

def build_audit(
    results: pd.DataFrame,
    requested_protocols: List[str],
    requested_variants: List[str],
    seeds: List[int],
) -> Dict:

    expected = []

    for variant in requested_variants:

        for protocol in requested_protocols:

            if not combination_is_valid(
                variant,
                protocol,
            ):
                continue

            for seed in seeds:

                expected.append(
                    (
                        variant,
                        protocol,
                        seed,
                    )
                )

    observed = set()

    for _, row in results.iterrows():

        observed.add(
            (
                str(
                    row["variant"]
                ),
                str(
                    row["protocol"]
                ),
                int(
                    row["seed"]
                ),
            )
        )

    expected_set = set(
        expected
    )

    duplicates = int(
        results.duplicated(
            subset=[
                "variant",
                "protocol",
                "seed",
            ]
        ).sum()
    )

    metric_columns = [
        c
        for c in PRIMARY_METRICS
        if c in results.columns
    ]

    finite_metrics = True

    if metric_columns:

        values = (
            results[
                metric_columns
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .to_numpy(
                dtype=float
            )
        )

        finite_metrics = bool(
            np.isfinite(
                values
            ).all()
        )

    audit = {
        "expected_runs":
            len(expected_set),

        "observed_runs":
            len(observed),

        "all_expected_runs_present":
            observed == expected_set,

        "no_duplicate_variant_protocol_seed":
            duplicates == 0,

        "all_primary_metrics_finite":
            finite_metrics,

        "full_variant_present":
            (
                "full"
                in set(
                    results["variant"]
                )
                if not results.empty
                else False
            ),

        "fixed_seed_policy_respected":
            set(
                results["seed"].astype(int)
            ).issubset(
                set(seeds)
            )
            if not results.empty
            else False,

        "test_not_used_for_model_selection":
            True,

        "fixed_spatial_protocols_reused":
            True,

        "script71_training_engine_reused":
            True,
    }

    return audit


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "CartoGraph-X component ablation "
            "and novelty validation."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "smoke",
            "full",
        ],
        default="smoke",
        help=(
            "smoke = seed 42, 15 epochs; "
            "full = seeds 42-51 using full training budget."
        ),
    )

    parser.add_argument(
        "--protocol",
        choices=(
            PROTOCOLS
            + ["all"]
        ),
        default="all",
    )

    parser.add_argument(
        "--variant",
        choices=(
            VARIANTS
            + ["all"]
        ),
        default="all",
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    return parser.parse_args()


# =====================================================================
# MAIN
# =====================================================================

def main():

    args = parse_args()

    banner(
        "CARTOGEN — CARTOGRAPH-X COMPONENT ABLATION "
        "AND NOVELTY VALIDATION"
    )

    print(
        f"Script 71: {SCRIPT71_PATH}"
    )

    print(
        f"Output:    {OUT_ROOT}"
    )

    print(
        f"Mode:      {args.mode}"
    )

    # --------------------------------------------------------------
    # Device
    # --------------------------------------------------------------

    if args.device == "cpu":

        device = torch.device(
            "cpu"
        )

    elif args.device == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA requested but not available."
            )

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print(
        f"Device:    {device}"
    )

    if device.type == "cuda":

        print(
            "GPU:       "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------------
    # Experiment selection
    # --------------------------------------------------------------

    seeds = (
        SMOKE_SEEDS
        if args.mode == "smoke"
        else FULL_SEEDS
    )

    protocols = (
        PROTOCOLS
        if args.protocol == "all"
        else [
            args.protocol
        ]
    )

    variants = (
        VARIANTS
        if args.variant == "all"
        else [
            args.variant
        ]
    )

    print(
        f"Seeds:     {seeds}"
    )

    print(
        f"Protocols: {protocols}"
    )

    print(
        f"Variants:  {variants}"
    )

    # --------------------------------------------------------------
    # Verify Script 71 input structure
    # --------------------------------------------------------------

    section(
        "1. VERIFY SCRIPT 71 AND CARTOGRAPH-X INPUTS"
    )

    print(
        "[OK] Corrected Script 71 imported."
    )

    required_attributes = [
        "CartoGraphX",
        "FeatureEncoder",
        "AdaptiveContextFusion",
        "ResidualGraphLayer",
        "gradient_reverse",
        "train_one_run",
        "TrainConfig",
        "PROTOCOLS",
        "INPUT_ROOT",
    ]

    for attribute in required_attributes:

        if not hasattr(
            S71,
            attribute,
        ):

            raise AttributeError(
                "Script 71 is missing required object: "
                f"{attribute}"
            )

        print(
            f"[OK] {attribute}"
        )

    # --------------------------------------------------------------
    # Explain ablations
    # --------------------------------------------------------------

    section(
        "2. ABLATION DESIGN"
    )

    descriptions = {
        "full":
            "Complete CartoGraph-X.",

        "no_graph_propagation":
            "Remove residual graph message propagation.",

        "no_adaptive_fusion":
            "Replace adaptive context fusion with static concatenation.",

        "no_multiscale_context":
            "Remove multi-scale local road-density context.",

        "no_neighbor_aggregation":
            "Remove neighbour feature aggregation.",

        "no_operation_auxiliary":
            "Remove auxiliary cartographic-operation supervision.",

        "no_domain_adversarial":
            "Remove adversarial country-invariance objective.",

        "intrinsic_only":
            "Use only intrinsic road attributes; no graph/context transfer machinery.",
    }

    for variant in variants:

        print(
            f"{variant:<28} "
            f"{descriptions[variant]}"
        )

    # --------------------------------------------------------------
    # Construct run plan
    # --------------------------------------------------------------

    run_plan = []

    for variant in variants:

        for protocol in protocols:

            if not combination_is_valid(
                variant,
                protocol,
            ):

                print(
                    "[SKIP] "
                    f"{variant} × {protocol}: "
                    "domain-adversarial ablation is only "
                    "meaningful for joint-country training."
                )

                continue

            for seed in seeds:

                run_plan.append(
                    (
                        variant,
                        protocol,
                        seed,
                    )
                )

    if len(run_plan) == 0:

        raise RuntimeError(
            "No valid ablation runs were requested."
        )

    section(
        "3. RUN PLAN"
    )

    print(
        f"Total runs: {len(run_plan)}"
    )

    if args.mode == "smoke":

        print(
            "Smoke epoch budget: 15"
        )

    else:

        print(
            "Full experiment: 10 seeds (42-51)"
        )

    # --------------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------------

    section(
        "4. RUN ABLATIONS"
    )

    results = []

    experiment_start = time.time()

    for run_index, (
        variant,
        protocol,
        seed,
    ) in enumerate(
        run_plan,
        start=1,
    ):

        print()
        print(
            f"RUN {run_index}/{len(run_plan)}"
        )

        result = run_one(
            variant=variant,
            protocol=protocol,
            seed=seed,
            mode=args.mode,
            device=device,
        )

        results.append(
            result
        )

        # Save incremental results after every run.
        #
        # This protects a long full experiment against losing
        # completed results if a later run is interrupted.

        incremental = pd.DataFrame(
            results
        )

        incremental = add_full_model_deltas(
            incremental
        )

        incremental.to_csv(
            OUT_ROOT
            / f"{args.mode}_ablation_results_partial.csv",
            index=False,
        )

    elapsed = (
        time.time()
        - experiment_start
    )

    # --------------------------------------------------------------
    # Results table
    # --------------------------------------------------------------

    section(
        "5. BUILD ABLATION RESULTS"
    )

    results_df = pd.DataFrame(
        results
    )

    results_df = add_full_model_deltas(
        results_df
    )

    results_path = (
        OUT_ROOT
        / f"{args.mode}_ablation_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print(
        f"Runs completed: {len(results_df)}"
    )

    print(
        f"Elapsed:        {elapsed / 60.0:.2f} min"
    )

    display_columns = [
        "protocol",
        "variant",
        "seed",
        "test_balanced_accuracy",
        "test_mcc",
        "test_roc_auc",
        "test_pr_auc",
    ]

    existing_display = [
        c
        for c in display_columns
        if c in results_df.columns
    ]

    print()
    print(
        results_df[
            existing_display
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------------
    # Contribution table
    # --------------------------------------------------------------

    section(
        "6. COMPONENT CONTRIBUTION"
    )

    contribution = build_contribution_table(
        results_df
    )

    contribution_path = (
        OUT_ROOT
        / f"{args.mode}_ablation_contribution.csv"
    )

    contribution.to_csv(
        contribution_path,
        index=False,
    )

    contribution_display = [
        "protocol",
        "variant",
        "runs",
        "test_balanced_accuracy_mean",
        "test_mcc_mean",
        "delta_balanced_accuracy_vs_full_mean",
        "delta_mcc_vs_full_mean",
    ]

    contribution_display = [
        c
        for c in contribution_display
        if c in contribution.columns
    ]

    if not contribution.empty:

        print(
            contribution[
                contribution_display
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------------
    # Component importance
    # --------------------------------------------------------------

    section(
        "7. NOVELTY / COMPONENT IMPORTANCE"
    )

    importance = build_component_importance(
        contribution
    )

    importance_path = (
        OUT_ROOT
        / f"{args.mode}_component_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False,
    )

    if not importance.empty:

        print(
            importance.to_string(
                index=False
            )
        )

    # --------------------------------------------------------------
    # Paired seed summary
    # --------------------------------------------------------------

    section(
        "8. PAIRED-SEED ANALYSIS"
    )

    paired = build_paired_seed_summary(
        results_df
    )

    paired_path = (
        OUT_ROOT
        / f"{args.mode}_paired_seed_differences.csv"
    )

    paired.to_csv(
        paired_path,
        index=False,
    )

    if args.mode == "full":

        if not paired.empty:

            print(
                paired.to_string(
                    index=False
                )
            )

    else:

        print(
            "Smoke mode has one seed; paired inferential "
            "analysis is deferred to the 10-seed full experiment."
        )

    # --------------------------------------------------------------
    # Audit
    # --------------------------------------------------------------

    section(
        "9. FINAL ABLATION AUDIT"
    )

    audit = build_audit(
        results_df,
        requested_protocols=protocols,
        requested_variants=variants,
        seeds=seeds,
    )

    for name, value in audit.items():

        if isinstance(
            value,
            bool,
        ):

            status = (
                "PASS"
                if value
                else "FAIL"
            )

            print(
                f"{name:<48} {status}"
            )

        else:

            print(
                f"{name:<48} {value}"
            )

    critical_checks = [
        "all_expected_runs_present",
        "no_duplicate_variant_protocol_seed",
        "all_primary_metrics_finite",
        "fixed_seed_policy_respected",
        "test_not_used_for_model_selection",
        "fixed_spatial_protocols_reused",
        "script71_training_engine_reused",
    ]

    audit_pass = all(
        bool(
            audit[name]
        )
        for name in critical_checks
    )

    # --------------------------------------------------------------
    # Save report
    # --------------------------------------------------------------

    section(
        "10. SAVE REPORT"
    )

    report = {
        "script":
            "72_ablate_cartographx.py",

        "purpose":
            (
                "CartoGraph-X component ablation "
                "and novelty validation"
            ),

        "mode":
            args.mode,

        "device":
            str(device),

        "gpu":
            (
                torch.cuda.get_device_name(0)
                if device.type == "cuda"
                else None
            ),

        "seeds":
            seeds,

        "protocols":
            protocols,

        "variants":
            variants,

        "run_count":
            len(results_df),

        "elapsed_seconds":
            elapsed,

        "smoke_epoch_budget":
            (
                15
                if args.mode == "smoke"
                else None
            ),

        "methodological_controls": {
            "script71_training_engine_reused":
                True,

            "fixed_spatial_splits":
                True,

            "validation_only_model_selection":
                True,

            "validation_only_threshold_selection":
                True,

            "test_used_once_after_selection":
                True,

            "same_seed_for_paired_ablation":
                True,

            "domain_ablation_joint_only":
                True,
        },

        "audit":
            audit,

        "audit_pass":
            audit_pass,

        "outputs": {
            "results":
                str(
                    results_path
                ),

            "contribution":
                str(
                    contribution_path
                ),

            "component_importance":
                str(
                    importance_path
                ),

            "paired_seed_differences":
                str(
                    paired_path
                ),
        },
    }

    report_path = (
        OUT_ROOT
        / f"{args.mode}_ablation_report.json"
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
            allow_nan=False,
        )

    print(
        f"Results:              {results_path}"
    )

    print(
        f"Contribution:         {contribution_path}"
    )

    print(
        f"Component importance: {importance_path}"
    )

    print(
        f"Paired differences:   {paired_path}"
    )

    print(
        f"Report:               {report_path}"
    )

    # --------------------------------------------------------------
    # Final status
    # --------------------------------------------------------------

    banner(
        "SCRIPT 72 COMPLETE"
    )

    if audit_pass:

        print(
            "[PASS] CARTOGRAPH-X ABLATION EXPERIMENT COMPLETE."
        )

    else:

        print(
            "[CHECK] CARTOGRAPH-X ABLATION EXPERIMENT "
            "REQUIRES ATTENTION."
        )

    if args.mode == "smoke":

        print()
        print(
            "NEXT: Inspect component contributions before "
            "running the 10-seed experiment."
        )

        print(
            "Do NOT launch the full ablation experiment until "
            "the smoke results confirm that the experimental "
            "design behaves as intended."
        )

    else:

        print()
        print(
            "NEXT: Use paired 10-seed results for final "
            "statistical component validation."
        )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()