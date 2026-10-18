"""
Script 70
=========
Prepare leakage-safe CartoGraph-X inputs for cross-country cartographic
road-generalization modelling.

This script DOES NOT train CartoGraph-X.

It:
1. Loads the harmonized Lithuania + Poland road benchmark.
2. Loads the leakage-safe graph nodes and graph edges from Script 68.
3. Validates node/edge alignment.
4. Defines intrinsic, graph, spatial-context, and domain metadata.
5. Encodes binary road-selection supervision.
6. Encodes road-operation supervision.
7. Creates country/domain labels.
8. Preserves the audited train/validation/test splits.
9. Builds the six modelling protocols:
       LT -> LT
       PL -> PL
       LT -> PL zero-shot
       PL -> LT zero-shot
       Joint -> LT
       Joint -> PL
10. Fits preprocessing statistics ONLY on each protocol's training data.
11. Produces standardized node-feature tables.
12. Converts graph edges to protocol-local integer node indices.
13. Saves metadata required by the future CartoGraph-X model.
14. Performs strict leakage and integrity audits.

IMPORTANT
---------
No target labels are used to construct graph edges or preprocessing statistics.

No validation/test statistics are used for feature normalization.

No graph edge is allowed to cross:
    country
    split
    tile

This script prepares data only. Neural model training begins in Script 71.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

HARMONIZED_DIR = (
    ROOT
    / "data"
    / "processed"
    / "cross_country"
    / "harmonized_roads"
)

GRAPH_DIR = (
    ROOT
    / "data"
    / "processed"
    / "cross_country"
    / "graph_representations"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "cross_country"
    / "cartographx_inputs"
)

PROTOCOL_DIR = OUTPUT_DIR / "protocols"

BENCHMARK_PATH = (
    HARMONIZED_DIR
    / "lithuania_poland_road_benchmark.csv"
)

GRAPH_NODES_PATH = (
    GRAPH_DIR
    / "road_graph_nodes.csv"
)

GRAPH_EDGES_PATH = (
    GRAPH_DIR
    / "road_graph_edges.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONSTANTS
# =============================================================================

VALID_COUNTRIES = {"Lithuania", "Poland"}

VALID_SPLITS = {
    "train",
    "validation",
    "test",
}

COUNTRY_TO_ID = {
    "Lithuania": 0,
    "Poland": 1,
}

ID_TO_COUNTRY = {
    0: "Lithuania",
    1: "Poland",
}


# -----------------------------------------------------------------------------
# Model feature groups
# -----------------------------------------------------------------------------
#
# These are intentionally separated because CartoGraph-X will not simply
# concatenate everything into one classical feature vector.
#
# Script 71 can give each group a separate encoder.
#

INTRINSIC_FEATURES = [
    "length_m",
    "intersecting_road_count",
    "neighbor_count_150m",
    "mean_neighbor_distance_m",
]

GRAPH_STRUCTURAL_FEATURES = [
    "graph_degree",
    "graph_mean_edge_distance_m",
    "graph_min_edge_distance_m",
    "graph_max_edge_distance_m",
    "graph_local_clustering",
    "graph_component_size",
    "graph_is_isolated",
]

MULTISCALE_CONTEXT_FEATURES = [
    "local_road_count_50m",
    "local_road_count_100m",
    "local_road_count_150m",
    "local_road_count_250m",
]

NEIGHBOR_AGGREGATION_FEATURES = [
    "nbr_mean_length_m",
    "nbr_std_length_m",
    "nbr_mean_intersecting_road_count",
    "nbr_std_intersecting_road_count",
    "nbr_mean_neighbor_count_150m",
    "nbr_std_neighbor_count_150m",
    "nbr_mean_mean_neighbor_distance_m",
    "nbr_std_mean_neighbor_distance_m",
]

SPATIAL_METADATA = [
    "centroid_x",
    "centroid_y",
]

ALL_MODEL_FEATURES = (
    INTRINSIC_FEATURES
    + GRAPH_STRUCTURAL_FEATURES
    + MULTISCALE_CONTEXT_FEATURES
    + NEIGHBOR_AGGREGATION_FEATURES
)


# -----------------------------------------------------------------------------
# Six experimental protocols
# -----------------------------------------------------------------------------

PROTOCOLS = {
    "lt_to_lt": {
        "train_countries": ["Lithuania"],
        "validation_countries": ["Lithuania"],
        "test_countries": ["Lithuania"],
    },

    "pl_to_pl": {
        "train_countries": ["Poland"],
        "validation_countries": ["Poland"],
        "test_countries": ["Poland"],
    },

    "lt_to_pl_zero_shot": {
        "train_countries": ["Lithuania"],
        "validation_countries": ["Lithuania"],
        "test_countries": ["Poland"],
    },

    "pl_to_lt_zero_shot": {
        "train_countries": ["Poland"],
        "validation_countries": ["Poland"],
        "test_countries": ["Lithuania"],
    },

    "joint_to_lt": {
        "train_countries": ["Lithuania", "Poland"],
        "validation_countries": ["Lithuania"],
        "test_countries": ["Lithuania"],
    },

    "joint_to_pl": {
        "train_countries": ["Lithuania", "Poland"],
        "validation_countries": ["Poland"],
        "test_countries": ["Poland"],
    },
}


# =============================================================================
# PRINT HELPERS
# =============================================================================

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


def fail(message: str) -> None:
    raise RuntimeError(message)


# =============================================================================
# COLUMN HELPERS
# =============================================================================

def require_columns(
    df: pd.DataFrame,
    columns: List[str],
    name: str,
) -> None:

    missing = [
        c
        for c in columns
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            f"{name} is missing required columns:\n"
            f"{missing}\n\n"
            f"Available columns:\n"
            f"{df.columns.tolist()}"
        )


def first_existing(
    df: pd.DataFrame,
    candidates: List[str],
    required: bool = True,
) -> str | None:

    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise KeyError(
            f"None of the candidate columns were found:\n"
            f"{candidates}\n\n"
            f"Available columns:\n"
            f"{df.columns.tolist()}"
        )

    return None


# =============================================================================
# LOAD DATA
# =============================================================================

def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    for path in [
        BENCHMARK_PATH,
        GRAPH_NODES_PATH,
        GRAPH_EDGES_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    benchmark = pd.read_csv(BENCHMARK_PATH)
    nodes = pd.read_csv(GRAPH_NODES_PATH)
    edges = pd.read_csv(GRAPH_EDGES_PATH)

    return benchmark, nodes, edges


# =============================================================================
# NORMALIZE GRAPH NODE SCHEMA
# =============================================================================

def normalize_nodes(
    nodes: pd.DataFrame,
) -> pd.DataFrame:

    d = nodes.copy()

    uid_col = first_existing(
        d,
        [
            "sample_uid",
            "node_uid",
            "uid",
        ],
    )

    country_col = first_existing(
        d,
        [
            "country",
            "country_harmonized",
        ],
    )

    city_col = first_existing(
        d,
        [
            "city_harmonized",
            "city",
        ],
    )

    split_col = first_existing(
        d,
        [
            "split_harmonized",
            "split",
        ],
    )

    tile_col = first_existing(
        d,
        [
            "tile_id_harmonized",
            "tile_id",
        ],
    )

    retained_col = first_existing(
        d,
        [
            "retained",
            "selection_binary",
        ],
    )

    operation_col = first_existing(
        d,
        [
            "operation_harmonized",
            "road_operation",
            "operation",
        ],
        required=False,
    )

    rename = {
        uid_col: "sample_uid",
        country_col: "country",
        city_col: "city",
        split_col: "split",
        tile_col: "tile_id",
        retained_col: "retained",
    }

    if operation_col is not None:
        rename[operation_col] = "operation"

    d = d.rename(columns=rename)

    # Remove duplicate columns that may appear after normalization.
    d = d.loc[:, ~d.columns.duplicated()].copy()

    d["sample_uid"] = d["sample_uid"].astype(str)
    d["country"] = d["country"].astype(str)
    d["city"] = d["city"].astype(str)
    d["split"] = d["split"].astype(str)
    d["tile_id"] = d["tile_id"].astype(str)

    d["retained"] = pd.to_numeric(
        d["retained"],
        errors="coerce",
    )

    return d


# =============================================================================
# NORMALIZE EDGE SCHEMA
# =============================================================================

def normalize_edges(
    edges: pd.DataFrame,
) -> pd.DataFrame:

    d = edges.copy()

    source_col = first_existing(
        d,
        [
            "source_uid",
            "source_sample_uid",
            "source",
            "src_uid",
            "src",
            "u",
        ],
    )

    target_col = first_existing(
        d,
        [
            "target_uid",
            "target_sample_uid",
            "target",
            "dst_uid",
            "dst",
            "v",
        ],
    )

    distance_col = first_existing(
        d,
        [
            "distance_m",
            "edge_distance_m",
            "spatial_distance_m",
            "distance",
        ],
        required=False,
    )

    rename = {
        source_col: "source_uid",
        target_col: "target_uid",
    }

    if distance_col is not None:
        rename[distance_col] = "edge_distance_m"

    d = d.rename(columns=rename)

    d["source_uid"] = d["source_uid"].astype(str)
    d["target_uid"] = d["target_uid"].astype(str)

    if "edge_distance_m" not in d.columns:
        d["edge_distance_m"] = np.nan

    return d


# =============================================================================
# VERIFY BENCHMARK/NODE ALIGNMENT
# =============================================================================

def verify_node_alignment(
    benchmark: pd.DataFrame,
    nodes: pd.DataFrame,
) -> None:

    require_columns(
        benchmark,
        ["sample_uid"],
        "Harmonized benchmark",
    )

    benchmark_uids = set(
        benchmark["sample_uid"].astype(str)
    )

    node_uids = set(
        nodes["sample_uid"].astype(str)
    )

    missing_from_nodes = (
        benchmark_uids - node_uids
    )

    extra_nodes = (
        node_uids - benchmark_uids
    )

    print(
        f"Benchmark samples:          "
        f"{len(benchmark_uids):,}"
    )

    print(
        f"Graph-node samples:         "
        f"{len(node_uids):,}"
    )

    print(
        f"Benchmark missing from graph:"
        f" {len(missing_from_nodes):,}"
    )

    print(
        f"Graph nodes not benchmark:  "
        f"{len(extra_nodes):,}"
    )

    if missing_from_nodes:
        fail(
            "Some harmonized benchmark samples are missing "
            "from the graph representation."
        )

    if extra_nodes:
        fail(
            "Graph representation contains samples not present "
            "in the harmonized benchmark."
        )


# =============================================================================
# VERIFY GRAPH LEAKAGE
# =============================================================================

def audit_edges(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> Dict[str, int]:

    metadata = (
        nodes[
            [
                "sample_uid",
                "country",
                "split",
                "tile_id",
            ]
        ]
        .drop_duplicates("sample_uid")
        .set_index("sample_uid")
    )

    known = set(metadata.index)

    unknown_source = int(
        (~edges["source_uid"].isin(known)).sum()
    )

    unknown_target = int(
        (~edges["target_uid"].isin(known)).sum()
    )

    self_edges = int(
        (
            edges["source_uid"]
            == edges["target_uid"]
        ).sum()
    )

    valid = edges[
        edges["source_uid"].isin(known)
        & edges["target_uid"].isin(known)
    ].copy()

    src = metadata.loc[
        valid["source_uid"]
    ].reset_index(drop=True)

    dst = metadata.loc[
        valid["target_uid"]
    ].reset_index(drop=True)

    cross_country = int(
        (
            src["country"].to_numpy()
            != dst["country"].to_numpy()
        ).sum()
    )

    cross_split = int(
        (
            src["split"].to_numpy()
            != dst["split"].to_numpy()
        ).sum()
    )

    cross_tile = int(
        (
            src["tile_id"].to_numpy()
            != dst["tile_id"].to_numpy()
        ).sum()
    )

    return {
        "unknown_source_endpoints": unknown_source,
        "unknown_target_endpoints": unknown_target,
        "self_edges": self_edges,
        "cross_country_edges": cross_country,
        "cross_split_edges": cross_split,
        "cross_tile_edges": cross_tile,
    }


# =============================================================================
# PREPARE MODEL NODE TABLE
# =============================================================================

def prepare_node_table(
    nodes: pd.DataFrame,
) -> pd.DataFrame:

    d = nodes.copy()

    required = [
        "sample_uid",
        "country",
        "city",
        "split",
        "tile_id",
        "retained",
        "centroid_x",
        "centroid_y",
    ]

    require_columns(
        d,
        required,
        "Graph nodes",
    )

    missing_features = [
        c
        for c in ALL_MODEL_FEATURES
        if c not in d.columns
    ]

    if missing_features:
        raise KeyError(
            "The graph-node table is missing CartoGraph-X "
            "features:\n"
            f"{missing_features}\n\n"
            "Run Script 68 successfully before Script 70."
        )

    if d["sample_uid"].duplicated().any():
        dup = int(
            d["sample_uid"].duplicated().sum()
        )
        fail(
            f"Graph nodes contain {dup:,} duplicate sample_uid values."
        )

    d["country_id"] = (
        d["country"]
        .map(COUNTRY_TO_ID)
    )

    if d["country_id"].isna().any():
        bad = sorted(
            d.loc[
                d["country_id"].isna(),
                "country",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        fail(
            f"Unexpected countries: {bad}"
        )

    d["country_id"] = (
        d["country_id"]
        .astype(int)
    )

    d["retained"] = (
        pd.to_numeric(
            d["retained"],
            errors="coerce",
        )
    )

    if d["retained"].isna().any():
        fail(
            "Some road-selection labels are unresolved."
        )

    if not set(
        d["retained"].astype(int).unique()
    ).issubset({0, 1}):

        fail(
            "retained must contain only binary values 0/1."
        )

    d["retained"] = (
        d["retained"]
        .astype(int)
    )

    if not set(
        d["country"].unique()
    ).issubset(VALID_COUNTRIES):

        fail(
            "Unexpected country values detected."
        )

    if not set(
        d["split"].unique()
    ).issubset(VALID_SPLITS):

        fail(
            "Unexpected split values detected."
        )

    # -------------------------------------------------------------------------
    # Operation supervision
    # -------------------------------------------------------------------------

    if "operation" not in d.columns:
        d["operation"] = np.where(
            d["retained"].eq(0),
            "elimination",
            "retained_unspecified",
        )

    d["operation"] = (
        d["operation"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Force eliminated roads into the elimination operation.
    d.loc[
        d["retained"].eq(0),
        "operation",
    ] = "elimination"

    # Retained roads with no operation still remain usable for selection.
    retained_missing = (
        d["retained"].eq(1)
        & d["operation"].isin(
            [
                "",
                "nan",
                "none",
                "null",
            ]
        )
    )

    d.loc[
        retained_missing,
        "operation",
    ] = "retained_unspecified"

    # Stable operation vocabulary.
    operation_values = sorted(
        d["operation"]
        .dropna()
        .unique()
        .tolist()
    )

    operation_to_id = {
        op: i
        for i, op in enumerate(operation_values)
    }

    d["operation_id"] = (
        d["operation"]
        .map(operation_to_id)
        .astype(int)
    )

    return d


# =============================================================================
# FEATURE MISSINGNESS
# =============================================================================

def feature_audit(
    nodes: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for feature in ALL_MODEL_FEATURES:

        x = pd.to_numeric(
            nodes[feature],
            errors="coerce",
        )

        finite = np.isfinite(
            x.to_numpy(dtype=float)
        )

        valid_values = (
            x.to_numpy(dtype=float)[finite]
        )

        rows.append(
            {
                "feature": feature,
                "group": feature_group(feature),
                "missing": int((~finite).sum()),
                "missing_pct": float(
                    100.0 * (~finite).mean()
                ),
                "n_unique": int(
                    pd.Series(valid_values)
                    .nunique()
                ),
                "mean": (
                    float(np.mean(valid_values))
                    if len(valid_values)
                    else np.nan
                ),
                "std": (
                    float(np.std(valid_values))
                    if len(valid_values)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def feature_group(feature: str) -> str:

    if feature in INTRINSIC_FEATURES:
        return "intrinsic"

    if feature in GRAPH_STRUCTURAL_FEATURES:
        return "graph_structural"

    if feature in MULTISCALE_CONTEXT_FEATURES:
        return "multiscale_context"

    if feature in NEIGHBOR_AGGREGATION_FEATURES:
        return "neighbor_aggregation"

    return "unknown"


# =============================================================================
# FIT TRAIN-ONLY PREPROCESSOR
# =============================================================================

def fit_preprocessor(
    train: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:

    stats = {}

    for feature in ALL_MODEL_FEATURES:

        x = pd.to_numeric(
            train[feature],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        )

        median = x.median()

        if pd.isna(median):
            median = 0.0

        filled = x.fillna(median)

        mean = float(
            filled.mean()
        )

        std = float(
            filled.std(ddof=0)
        )

        if (
            not np.isfinite(std)
            or std < 1e-12
        ):
            std = 1.0

        stats[feature] = {
            "median": float(median),
            "mean": mean,
            "std": std,
        }

    return stats


# =============================================================================
# APPLY PREPROCESSOR
# =============================================================================

def apply_preprocessor(
    df: pd.DataFrame,
    stats: Dict[str, Dict[str, float]],
) -> pd.DataFrame:

    out = df.copy()

    for feature in ALL_MODEL_FEATURES:

        x = pd.to_numeric(
            out[feature],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        )

        median = stats[feature]["median"]
        mean = stats[feature]["mean"]
        std = stats[feature]["std"]

        x = x.fillna(median)

        out[
            f"x_{feature}"
        ] = (
            (x - mean)
            / std
        ).astype(np.float32)

    return out


# =============================================================================
# PROTOCOL SUBSETS
# =============================================================================

def get_protocol_subset(
    nodes: pd.DataFrame,
    protocol: Dict,
    stage: str,
) -> pd.DataFrame:

    if stage == "train":

        countries = (
            protocol["train_countries"]
        )

        split = "train"

    elif stage == "validation":

        countries = (
            protocol["validation_countries"]
        )

        split = "validation"

    elif stage == "test":

        countries = (
            protocol["test_countries"]
        )

        split = "test"

    else:
        raise ValueError(stage)

    out = nodes[
        nodes["country"].isin(countries)
        & nodes["split"].eq(split)
    ].copy()

    return out


# =============================================================================
# BUILD LOCAL NODE INDEX
# =============================================================================

def add_local_node_index(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = (
        df
        .sort_values(
            [
                "country",
                "city",
                "tile_id",
                "sample_uid",
            ]
        )
        .reset_index(drop=True)
        .copy()
    )

    out.insert(
        0,
        "node_index",
        np.arange(
            len(out),
            dtype=np.int64,
        ),
    )

    return out


# =============================================================================
# SUBSET EDGES TO A PROTOCOL STAGE
# =============================================================================

def subset_edges(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
) -> pd.DataFrame:

    uid_to_index = dict(
        zip(
            nodes["sample_uid"],
            nodes["node_index"],
        )
    )

    allowed = set(
        uid_to_index.keys()
    )

    e = edges[
        edges["source_uid"].isin(allowed)
        & edges["target_uid"].isin(allowed)
    ].copy()

    e["source_index"] = (
        e["source_uid"]
        .map(uid_to_index)
        .astype(np.int64)
    )

    e["target_index"] = (
        e["target_uid"]
        .map(uid_to_index)
        .astype(np.int64)
    )

    # Keep deterministic ordering.
    e = (
        e.sort_values(
            [
                "source_index",
                "target_index",
            ]
        )
        .reset_index(drop=True)
    )

    return e


# =============================================================================
# VERIFY STAGE EDGE INTEGRITY
# =============================================================================

def verify_stage_edges(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    stage_name: str,
) -> Dict[str, int]:

    n = len(nodes)

    if len(edges) == 0:
        print(
            f"[WARNING] {stage_name}: no graph edges."
        )

        return {
            "nodes": n,
            "edges": 0,
            "invalid_edge_indices": 0,
        }

    invalid = int(
        (
            (edges["source_index"] < 0)
            | (edges["source_index"] >= n)
            | (edges["target_index"] < 0)
            | (edges["target_index"] >= n)
        ).sum()
    )

    if invalid:
        fail(
            f"{stage_name} contains {invalid:,} invalid "
            "local edge indices."
        )

    return {
        "nodes": n,
        "edges": len(edges),
        "invalid_edge_indices": invalid,
    }


# =============================================================================
# SAVE FEATURE MATRICES
# =============================================================================

def save_feature_matrix(
    nodes: pd.DataFrame,
    path: Path,
) -> None:

    x_columns = [
        f"x_{c}"
        for c in ALL_MODEL_FEATURES
    ]

    metadata_columns = [
        "node_index",
        "sample_uid",
        "country",
        "country_id",
        "city",
        "tile_id",
        "split",
        "retained",
        "operation",
        "operation_id",
        "centroid_x",
        "centroid_y",
    ]

    keep = (
        metadata_columns
        + x_columns
    )

    require_columns(
        nodes,
        keep,
        "Prepared protocol nodes",
    )

    nodes[
        keep
    ].to_csv(
        path,
        index=False,
    )


# =============================================================================
# CLASS WEIGHTS
# =============================================================================

def calculate_binary_class_weights(
    train: pd.DataFrame,
) -> Dict[str, float]:

    counts = (
        train["retained"]
        .value_counts()
        .to_dict()
    )

    n0 = int(
        counts.get(0, 0)
    )

    n1 = int(
        counts.get(1, 0)
    )

    total = n0 + n1

    if n0 == 0 or n1 == 0:
        fail(
            "Training set contains only one selection class."
        )

    weight_0 = (
        total
        / (2.0 * n0)
    )

    weight_1 = (
        total
        / (2.0 * n1)
    )

    return {
        "elimination_0": float(weight_0),
        "retained_1": float(weight_1),
        "n_elimination": n0,
        "n_retained": n1,
    }


# =============================================================================
# OPERATION WEIGHTS
# =============================================================================

def calculate_operation_weights(
    train: pd.DataFrame,
) -> Dict[str, float]:

    counts = (
        train["operation"]
        .value_counts()
        .sort_index()
    )

    total = int(
        counts.sum()
    )

    k = len(counts)

    result = {}

    for operation, count in counts.items():

        if count == 0:
            continue

        result[str(operation)] = float(
            total
            / (k * int(count))
        )

    return result


# =============================================================================
# PROTOCOL LEAKAGE AUDIT
# =============================================================================

def protocol_leakage_audit(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> Dict[str, int]:

    train_uids = set(
        train["sample_uid"]
    )

    val_uids = set(
        validation["sample_uid"]
    )

    test_uids = set(
        test["sample_uid"]
    )

    train_tiles = set(
        zip(
            train["country"],
            train["tile_id"],
        )
    )

    val_tiles = set(
        zip(
            validation["country"],
            validation["tile_id"],
        )
    )

    test_tiles = set(
        zip(
            test["country"],
            test["tile_id"],
        )
    )

    return {
        "train_validation_uid_overlap": len(
            train_uids & val_uids
        ),

        "train_test_uid_overlap": len(
            train_uids & test_uids
        ),

        "validation_test_uid_overlap": len(
            val_uids & test_uids
        ),

        "train_validation_tile_overlap": len(
            train_tiles & val_tiles
        ),

        "train_test_tile_overlap": len(
            train_tiles & test_tiles
        ),

        "validation_test_tile_overlap": len(
            val_tiles & test_tiles
        ),
    }


# =============================================================================
# PROTOCOL SUMMARY
# =============================================================================

def summarize_stage(
    protocol_name: str,
    stage: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> Dict:

    retained = int(
        nodes["retained"].sum()
    )

    eliminated = int(
        len(nodes) - retained
    )

    return {
        "protocol": protocol_name,
        "stage": stage,
        "samples": int(len(nodes)),
        "edges": int(len(edges)),
        "countries": "|".join(
            sorted(
                nodes["country"]
                .unique()
                .tolist()
            )
        ),
        "cities": int(
            nodes["city"].nunique()
        ),
        "tiles": int(
            nodes[
                [
                    "country",
                    "tile_id",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        ),
        "retained": retained,
        "eliminated": eliminated,
        "retention_pct": (
            100.0
            * retained
            / len(nodes)
            if len(nodes)
            else np.nan
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    banner(
        "CARTOGEN — PREPARE CARTOGRAPH-X "
        "LEAKAGE-SAFE MODEL INPUTS"
    )

    print(
        f"Benchmark: {BENCHMARK_PATH}"
    )

    print(
        f"Graph nodes: {GRAPH_NODES_PATH}"
    )

    print(
        f"Graph edges: {GRAPH_EDGES_PATH}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    # =========================================================================
    # 1. LOAD
    # =========================================================================

    section(
        "1. LOAD HARMONIZED BENCHMARK AND GRAPH REPRESENTATION"
    )

    benchmark, nodes_raw, edges_raw = load_inputs()

    print(
        f"Benchmark rows: {len(benchmark):,}"
    )

    print(
        f"Graph nodes:    {len(nodes_raw):,}"
    )

    print(
        f"Graph edges:    {len(edges_raw):,}"
    )

    # =========================================================================
    # 2. NORMALIZE
    # =========================================================================

    section(
        "2. NORMALIZE GRAPH SCHEMA"
    )

    nodes = normalize_nodes(
        nodes_raw
    )

    edges = normalize_edges(
        edges_raw
    )

    print(
        f"Countries: {sorted(nodes['country'].unique())}"
    )

    print(
        f"Splits:    {sorted(nodes['split'].unique())}"
    )

    print(
        f"Cities:    {sorted(nodes['city'].unique())}"
    )

    print(
        f"Tiles:     {nodes[['country','tile_id']].drop_duplicates().shape[0]:,}"
    )

    # =========================================================================
    # 3. ALIGNMENT
    # =========================================================================

    section(
        "3. VERIFY BENCHMARK / GRAPH NODE ALIGNMENT"
    )

    verify_node_alignment(
        benchmark,
        nodes,
    )

    # =========================================================================
    # 4. PREPARE NODE TABLE
    # =========================================================================

    section(
        "4. PREPARE CARTOGRAPH-X NODE SUPERVISION"
    )

    nodes = prepare_node_table(
        nodes
    )

    print(
        "\nCountry distribution:"
    )

    print(
        nodes["country"]
        .value_counts()
        .to_string()
    )

    print(
        "\nSplit distribution:"
    )

    print(
        nodes["split"]
        .value_counts()
        .to_string()
    )

    print(
        "\nSelection distribution:"
    )

    print(
        nodes["retained"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nOperation distribution:"
    )

    print(
        nodes["operation"]
        .value_counts()
        .to_string()
    )

    # =========================================================================
    # 5. EDGE LEAKAGE AUDIT
    # =========================================================================

    section(
        "5. GLOBAL GRAPH LEAKAGE AUDIT"
    )

    edge_audit = audit_edges(
        nodes,
        edges,
    )

    for key, value in edge_audit.items():

        print(
            f"{key:35s} {value:,}"
        )

    edge_pass = all(
        value == 0
        for value in edge_audit.values()
    )

    if not edge_pass:
        fail(
            "Global graph leakage audit failed."
        )

    print(
        "\n[PASS] Graph contains no cross-country, "
        "cross-split, or cross-tile leakage."
    )

    # =========================================================================
    # 6. FEATURE AUDIT
    # =========================================================================

    section(
        "6. CARTOGRAPH-X FEATURE AUDIT"
    )

    feature_summary = feature_audit(
        nodes
    )

    print(
        feature_summary.to_string(
            index=False
        )
    )

    feature_summary.to_csv(
        OUTPUT_DIR
        / "cartographx_feature_audit.csv",
        index=False,
    )

    # =========================================================================
    # 7. FEATURE GROUPS
    # =========================================================================

    section(
        "7. MODEL FEATURE GROUPS"
    )

    groups = {
        "intrinsic_features":
            INTRINSIC_FEATURES,

        "graph_structural_features":
            GRAPH_STRUCTURAL_FEATURES,

        "multiscale_context_features":
            MULTISCALE_CONTEXT_FEATURES,

        "neighbor_aggregation_features":
            NEIGHBOR_AGGREGATION_FEATURES,

        "spatial_metadata_not_direct_model_input":
            SPATIAL_METADATA,
    }

    for name, features in groups.items():

        print(
            f"\n{name}"
        )

        for f in features:
            print(
                f"  {f}"
            )

    # =========================================================================
    # 8. OPERATION VOCABULARY
    # =========================================================================

    section(
        "8. OPERATION VOCABULARY"
    )

    operation_vocab = (
        nodes[
            [
                "operation",
                "operation_id",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "operation_id"
        )
    )

    print(
        operation_vocab.to_string(
            index=False
        )
    )

    operation_vocab.to_csv(
        OUTPUT_DIR
        / "operation_vocabulary.csv",
        index=False,
    )

    # =========================================================================
    # 9. BUILD PROTOCOLS
    # =========================================================================

    section(
        "9. BUILD SIX LEAKAGE-SAFE CARTOGRAPH-X PROTOCOLS"
    )

    protocol_summary_rows = []
    protocol_reports = {}

    for protocol_name, protocol in PROTOCOLS.items():

        print()
        print(
            "=" * 120
        )
        print(
            protocol_name.upper()
        )
        print(
            "=" * 120
        )

        protocol_path = (
            PROTOCOL_DIR
            / protocol_name
        )

        protocol_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ---------------------------------------------------------------------
        # Raw protocol subsets
        # ---------------------------------------------------------------------

        train = get_protocol_subset(
            nodes,
            protocol,
            "train",
        )

        validation = get_protocol_subset(
            nodes,
            protocol,
            "validation",
        )

        test = get_protocol_subset(
            nodes,
            protocol,
            "test",
        )

        if len(train) == 0:
            fail(
                f"{protocol_name}: empty training set."
            )

        if len(validation) == 0:
            fail(
                f"{protocol_name}: empty validation set."
            )

        if len(test) == 0:
            fail(
                f"{protocol_name}: empty test set."
            )

        # ---------------------------------------------------------------------
        # Leakage audit before preprocessing
        # ---------------------------------------------------------------------

        leakage = protocol_leakage_audit(
            train,
            validation,
            test,
        )

        print(
            "\nLeakage audit:"
        )

        for key, value in leakage.items():

            print(
                f"  {key:35s} {value:,}"
            )

        if any(
            value != 0
            for value in leakage.values()
        ):
            fail(
                f"{protocol_name}: protocol leakage detected."
            )

        # ---------------------------------------------------------------------
        # TRAIN-ONLY preprocessing
        # ---------------------------------------------------------------------

        preprocessing = fit_preprocessor(
            train
        )

        train = apply_preprocessor(
            train,
            preprocessing,
        )

        validation = apply_preprocessor(
            validation,
            preprocessing,
        )

        test = apply_preprocessor(
            test,
            preprocessing,
        )

        # ---------------------------------------------------------------------
        # Local deterministic node indices
        # ---------------------------------------------------------------------

        train = add_local_node_index(
            train
        )

        validation = add_local_node_index(
            validation
        )

        test = add_local_node_index(
            test
        )

        # ---------------------------------------------------------------------
        # Protocol-stage graph edges
        # ---------------------------------------------------------------------

        train_edges = subset_edges(
            edges,
            train,
        )

        validation_edges = subset_edges(
            edges,
            validation,
        )

        test_edges = subset_edges(
            edges,
            test,
        )

        train_edge_audit = verify_stage_edges(
            train,
            train_edges,
            f"{protocol_name}/train",
        )

        val_edge_audit = verify_stage_edges(
            validation,
            validation_edges,
            f"{protocol_name}/validation",
        )

        test_edge_audit = verify_stage_edges(
            test,
            test_edges,
            f"{protocol_name}/test",
        )

        # ---------------------------------------------------------------------
        # Class weights
        # ---------------------------------------------------------------------

        selection_weights = (
            calculate_binary_class_weights(
                train
            )
        )

        operation_weights = (
            calculate_operation_weights(
                train
            )
        )

        # ---------------------------------------------------------------------
        # Summary
        # ---------------------------------------------------------------------

        stage_data = [
            (
                "train",
                train,
                train_edges,
            ),
            (
                "validation",
                validation,
                validation_edges,
            ),
            (
                "test",
                test,
                test_edges,
            ),
        ]

        for (
            stage,
            stage_nodes,
            stage_edges,
        ) in stage_data:

            summary = summarize_stage(
                protocol_name,
                stage,
                stage_nodes,
                stage_edges,
            )

            protocol_summary_rows.append(
                summary
            )

            print(
                f"\n{stage:12s} "
                f"samples={len(stage_nodes):7,d} | "
                f"edges={len(stage_edges):7,d} | "
                f"retained={int(stage_nodes['retained'].sum()):7,d} | "
                f"eliminated={int((stage_nodes['retained'] == 0).sum()):7,d}"
            )

        # ---------------------------------------------------------------------
        # Save prepared node tables
        # ---------------------------------------------------------------------

        save_feature_matrix(
            train,
            protocol_path
            / "train_nodes.csv",
        )

        save_feature_matrix(
            validation,
            protocol_path
            / "validation_nodes.csv",
        )

        save_feature_matrix(
            test,
            protocol_path
            / "test_nodes.csv",
        )

        # ---------------------------------------------------------------------
        # Save graph edges
        # ---------------------------------------------------------------------

        train_edges.to_csv(
            protocol_path
            / "train_edges.csv",
            index=False,
        )

        validation_edges.to_csv(
            protocol_path
            / "validation_edges.csv",
            index=False,
        )

        test_edges.to_csv(
            protocol_path
            / "test_edges.csv",
            index=False,
        )

        # ---------------------------------------------------------------------
        # Save train-only preprocessing
        # ---------------------------------------------------------------------

        with open(
            protocol_path
            / "preprocessing.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                preprocessing,
                f,
                indent=2,
            )

        # ---------------------------------------------------------------------
        # Save class weights
        # ---------------------------------------------------------------------

        weights = {
            "selection":
                selection_weights,

            "operation":
                operation_weights,
        }

        with open(
            protocol_path
            / "class_weights.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                weights,
                f,
                indent=2,
            )

        # ---------------------------------------------------------------------
        # Save protocol definition
        # ---------------------------------------------------------------------

        protocol_report = {
            "protocol_name":
                protocol_name,

            "definition":
                protocol,

            "feature_groups":
                groups,

            "all_model_features":
                ALL_MODEL_FEATURES,

            "n_model_features":
                len(ALL_MODEL_FEATURES),

            "leakage_audit":
                leakage,

            "train_edge_audit":
                train_edge_audit,

            "validation_edge_audit":
                val_edge_audit,

            "test_edge_audit":
                test_edge_audit,

            "selection_class_weights":
                selection_weights,

            "operation_class_weights":
                operation_weights,

            "preprocessing_fit_on":
                "training partition only",
        }

        protocol_reports[
            protocol_name
        ] = protocol_report

        with open(
            protocol_path
            / "protocol_metadata.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                protocol_report,
                f,
                indent=2,
            )

    # =========================================================================
    # 10. SAVE GLOBAL NODE TABLE
    # =========================================================================

    section(
        "10. SAVE GLOBAL CARTOGRAPH-X NODE METADATA"
    )

    global_columns = [
        "sample_uid",
        "country",
        "country_id",
        "city",
        "tile_id",
        "split",
        "retained",
        "operation",
        "operation_id",
        "centroid_x",
        "centroid_y",
    ] + ALL_MODEL_FEATURES

    nodes[
        global_columns
    ].to_csv(
        OUTPUT_DIR
        / "cartographx_global_nodes.csv",
        index=False,
    )

    edges.to_csv(
        OUTPUT_DIR
        / "cartographx_global_edges.csv",
        index=False,
    )

    # =========================================================================
    # 11. PROTOCOL SUMMARY
    # =========================================================================

    section(
        "11. PROTOCOL SUMMARY"
    )

    protocol_summary = pd.DataFrame(
        protocol_summary_rows
    )

    print(
        protocol_summary.to_string(
            index=False
        )
    )

    protocol_summary.to_csv(
        OUTPUT_DIR
        / "cartographx_protocol_summary.csv",
        index=False,
    )

    # =========================================================================
    # 12. FINAL AUDIT
    # =========================================================================

    section(
        "12. FINAL CARTOGRAPH-X INPUT AUDIT"
    )

    checks = {}

    checks[
        "both_countries_present"
    ] = (
        set(nodes["country"].unique())
        == VALID_COUNTRIES
    )

    checks[
        "all_three_splits_present"
    ] = (
        set(nodes["split"].unique())
        == VALID_SPLITS
    )

    checks[
        "unique_sample_uids"
    ] = (
        not nodes[
            "sample_uid"
        ].duplicated().any()
    )

    checks[
        "binary_selection_resolved"
    ] = (
        set(
            nodes["retained"]
            .unique()
        ).issubset(
            {0, 1}
        )
    )

    checks[
        "country_labels_resolved"
    ] = (
        not nodes[
            "country_id"
        ].isna().any()
    )

    checks[
        "operation_labels_resolved"
    ] = (
        not nodes[
            "operation_id"
        ].isna().any()
    )

    checks[
        "all_model_features_present"
    ] = all(
        feature in nodes.columns
        for feature in ALL_MODEL_FEATURES
    )

    checks[
        "no_cross_country_edges"
    ] = (
        edge_audit[
            "cross_country_edges"
        ]
        == 0
    )

    checks[
        "no_cross_split_edges"
    ] = (
        edge_audit[
            "cross_split_edges"
        ]
        == 0
    )

    checks[
        "no_cross_tile_edges"
    ] = (
        edge_audit[
            "cross_tile_edges"
        ]
        == 0
    )

    checks[
        "no_unknown_edge_endpoints"
    ] = (
        edge_audit[
            "unknown_source_endpoints"
        ]
        == 0
        and
        edge_audit[
            "unknown_target_endpoints"
        ]
        == 0
    )

    checks[
        "all_six_protocols_created"
    ] = (
        len(protocol_reports)
        == 6
    )

    checks[
        "train_only_preprocessing"
    ] = True

    all_pass = True

    for name, value in checks.items():

        status = (
            "PASS"
            if value
            else "FAIL"
        )

        print(
            f"{name:45s} {status}"
        )

        if not value:
            all_pass = False

    # =========================================================================
    # 13. GLOBAL REPORT
    # =========================================================================

    report = {
        "script":
            "70_prepare_cartographx_inputs.py",

        "benchmark_rows":
            int(len(benchmark)),

        "graph_nodes":
            int(len(nodes)),

        "graph_edges":
            int(len(edges)),

        "countries":
            sorted(
                nodes["country"]
                .unique()
                .tolist()
            ),

        "splits":
            sorted(
                nodes["split"]
                .unique()
                .tolist()
            ),

        "cities":
            sorted(
                nodes["city"]
                .unique()
                .tolist()
            ),

        "tiles":
            int(
                nodes[
                    [
                        "country",
                        "tile_id",
                    ]
                ]
                .drop_duplicates()
                .shape[0]
            ),

        "feature_groups":
            groups,

        "all_model_features":
            ALL_MODEL_FEATURES,

        "n_model_features":
            len(ALL_MODEL_FEATURES),

        "edge_leakage_audit":
            edge_audit,

        "checks":
            checks,

        "protocols":
            protocol_reports,

        "status":
            (
                "PASS"
                if all_pass
                else "FAIL"
            ),
    }

    with open(
        OUTPUT_DIR
        / "cartographx_input_report.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # =========================================================================
    # 14. OUTPUTS
    # =========================================================================

    section(
        "13. OUTPUTS"
    )

    print(
        "Global nodes:      "
        f"{OUTPUT_DIR / 'cartographx_global_nodes.csv'}"
    )

    print(
        "Global edges:      "
        f"{OUTPUT_DIR / 'cartographx_global_edges.csv'}"
    )

    print(
        "Feature audit:     "
        f"{OUTPUT_DIR / 'cartographx_feature_audit.csv'}"
    )

    print(
        "Operation vocab:   "
        f"{OUTPUT_DIR / 'operation_vocabulary.csv'}"
    )

    print(
        "Protocol summary:  "
        f"{OUTPUT_DIR / 'cartographx_protocol_summary.csv'}"
    )

    print(
        "Input report:      "
        f"{OUTPUT_DIR / 'cartographx_input_report.json'}"
    )

    print(
        "Protocol folders:  "
        f"{PROTOCOL_DIR}"
    )

    banner(
        "SCRIPT 70 COMPLETE"
    )

    if all_pass:

        print(
            "[PASS] CARTOGRAPH-X INPUT PIPELINE "
            "IS LEAKAGE-SAFE AND MODEL-READY."
        )

        print(
            "NEXT: Script 71 — implement the CartoGraph-X "
            "neural architecture."
        )

    else:

        print(
            "[FAIL] CARTOGRAPH-X INPUT PIPELINE "
            "REQUIRES ATTENTION."
        )

        sys.exit(1)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()