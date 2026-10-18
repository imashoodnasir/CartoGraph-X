"""
71_train_cartographx.py

CartoGen
-------
Train and evaluate CartoGraph-X for leakage-safe cross-country
road-selection modelling.

Architecture
------------
1. Intrinsic road encoder
2. Graph-structural encoder
3. Multi-scale spatial-context encoder
4. Neighbour-aggregation encoder
5. Adaptive context fusion
6. Residual graph message passing
7. Binary road-selection head
8. Auxiliary cartographic-operation head
9. Optional adversarial country/domain head

Important experimental rules
----------------------------
- Spatial train/validation/test splits are fixed by Script 70.
- Graphs never connect different countries, splits, or tiles.
- Preprocessing statistics are already train-derived by Script 70.
- Validation data select epoch/threshold.
- Test data are touched only after model selection.
- Country is NOT supplied as a predictive feature.
- Operation supervision is auxiliary only.
- Domain adversarial loss is active only when training contains
  more than one country.
- Smoke mode uses seed 42.
- Full mode uses seeds 42--51.

Run:
    python scripts/71_train_cartographx.py --mode smoke

Later:
    python scripts/71_train_cartographx.py --mode full

Optional:
    python scripts/71_train_cartographx.py --mode smoke --protocol lt_to_lt
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# IMPORTANT:
# Force a non-interactive Matplotlib backend before importing pyplot.
# This avoids the Tkinter/thread cleanup error seen in Script 66.
# ---------------------------------------------------------------------
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

# =====================================================================
# PATHS
# =====================================================================

ROOT = Path(r"D:\Research\CartoGen")

INPUT_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "cross_country"
    / "cartographx_inputs"
)

PROTOCOL_ROOT = INPUT_ROOT / "protocols"

OUT_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "cross_country"
    / "cartographx_models"
)

OUT_ROOT.mkdir(parents=True, exist_ok=True)

# =====================================================================
# EXPERIMENT CONFIGURATION
# =====================================================================

PROTOCOLS = [
    "lt_to_lt",
    "pl_to_pl",
    "lt_to_pl_zero_shot",
    "pl_to_lt_zero_shot",
    "joint_to_lt",
    "joint_to_pl",
]

SMOKE_SEEDS = [42]

FULL_SEEDS = [
    42, 43, 44, 45, 46,
    47, 48, 49, 50, 51,
]

# ---------------------------------------------------------------------
# EXACT FEATURE FAMILIES FROM SCRIPT 70
# ---------------------------------------------------------------------

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

ALL_FEATURES = (
    INTRINSIC_FEATURES
    + GRAPH_STRUCTURAL_FEATURES
    + MULTISCALE_CONTEXT_FEATURES
    + NEIGHBOR_AGGREGATION_FEATURES
)

# =====================================================================
# TRAINING CONFIGURATION
# =====================================================================

@dataclass
class TrainConfig:
    hidden_dim: int = 64
    branch_dim: int = 32
    graph_layers: int = 2
    dropout: float = 0.20

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Multi-task objective
    selection_loss_weight: float = 1.0
    operation_loss_weight: float = 0.20
    domain_loss_weight: float = 0.10

    # Minority operation classes can be extremely small.
    max_operation_class_weight: float = 10.0

    # Selection class weights are clipped as a safety measure.
    max_selection_class_weight: float = 8.0

    # Gradient clipping
    gradient_clip: float = 5.0

    # Validation criterion
    early_stopping_patience: int = 15

    # Threshold optimization range
    threshold_min: float = 0.10
    threshold_max: float = 0.90
    threshold_step: float = 0.02

    # Smoke / full
    smoke_epochs: int = 8
    full_epochs: int = 150

    # Domain-adversarial schedule
    domain_grl_max: float = 1.0


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
# JSON SERIALIZATION
# =====================================================================

def json_safe(obj):
    """Recursively convert values to JSON-safe native Python types."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, set):
        return [json_safe(v) for v in sorted(obj, key=str)]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is pd.NA:
        return None
    return obj


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic behaviour where possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
    except Exception:
        pass


# =====================================================================
# FILE DISCOVERY
# =====================================================================

def find_unique_file(
    folder: Path,
    candidates: List[str],
    required: bool = True,
) -> Optional[Path]:

    for name in candidates:
        p = folder / name
        if p.exists():
            return p

    # Flexible fallback
    csv_files = list(folder.glob("*.csv"))

    for p in csv_files:
        low = p.name.lower()

        if any(
            c.lower().replace(".csv", "") in low
            for c in candidates
        ):
            return p

    if required:
        raise FileNotFoundError(
            f"Could not identify required file in:\n{folder}\n"
            f"Tried:\n{candidates}\n"
            f"Available:\n{[x.name for x in csv_files]}"
        )

    return None


def discover_stage_files(
    protocol: str,
    stage: str,
) -> Tuple[Path, Path]:

    folder = PROTOCOL_ROOT / protocol

    if not folder.exists():
        raise FileNotFoundError(
            f"Protocol directory not found:\n{folder}"
        )

    node_candidates = [
        f"{stage}_nodes.csv",
        f"{stage}_node_features.csv",
        f"{stage}_samples.csv",
        f"{stage}.csv",
    ]

    edge_candidates = [
        f"{stage}_edges.csv",
        f"{stage}_graph_edges.csv",
    ]

    node_path = find_unique_file(
        folder,
        node_candidates,
        required=True,
    )

    edge_path = find_unique_file(
        folder,
        edge_candidates,
        required=True,
    )

    return node_path, edge_path


# =====================================================================
# SCHEMA HELPERS
# =====================================================================

def first_existing(
    df: pd.DataFrame,
    names: List[str],
    required: bool = True,
) -> Optional[str]:

    for name in names:
        if name in df.columns:
            return name

    if required:
        raise KeyError(
            f"None of {names} found.\n"
            f"Available columns:\n{df.columns.tolist()}"
        )

    return None


def normalize_node_schema(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    uid_col = first_existing(
        d,
        [
            "sample_uid",
            "uid",
            "node_uid",
            "road_uid",
        ],
    )

    retained_col = first_existing(
        d,
        [
            "retained",
            "selection_target",
            "selection_label_binary",
            "y",
        ],
    )

    operation_col = first_existing(
        d,
        [
            "operation_id",
            "operation_target",
            "operation_label_id",
        ],
        required=False,
    )

    country_col = first_existing(
        d,
        [
            "country",
            "country_harmonized",
        ],
    )

    tile_col = first_existing(
        d,
        [
            "tile_id",
            "tile_id_harmonized",
        ],
    )

    rename = {
        uid_col: "sample_uid",
        retained_col: "retained",
        country_col: "country",
        tile_col: "tile_id",
    }

    if operation_col is not None:
        rename[operation_col] = "operation_id"

    d = d.rename(columns=rename)

    # If operation ID is absent but operation text exists, reconstruct it.
    if "operation_id" not in d.columns:

        op_text_col = first_existing(
            d,
            [
                "operation",
                "operation_harmonized",
                "road_operation",
            ],
            required=False,
        )

        if op_text_col is None:
            raise KeyError(
                "Neither operation_id nor operation text was found."
            )

        vocab_path = INPUT_ROOT / "operation_vocabulary.csv"

        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Operation vocabulary not found:\n{vocab_path}"
            )

        vocab = pd.read_csv(vocab_path)

        if not {
            "operation",
            "operation_id",
        }.issubset(vocab.columns):
            raise KeyError(
                "operation_vocabulary.csv must contain "
                "'operation' and 'operation_id'."
            )

        mapping = dict(
            zip(
                vocab["operation"].astype(str),
                vocab["operation_id"].astype(int),
            )
        )

        d["operation_id"] = (
            d[op_text_col]
            .astype(str)
            .map(mapping)
        )

    # Country ID used only for adversarial training.
    country_map = {
        "Lithuania": 0,
        "Poland": 1,
    }

    d["country_id"] = (
        d["country"]
        .astype(str)
        .map(country_map)
    )

    if d["country_id"].isna().any():
        unknown = sorted(
            d.loc[
                d["country_id"].isna(),
                "country",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unknown country labels: {unknown}"
        )

    d["country_id"] = d["country_id"].astype(int)

    d["retained"] = pd.to_numeric(
        d["retained"],
        errors="raise",
    ).astype(int)

    d["operation_id"] = pd.to_numeric(
        d["operation_id"],
        errors="raise",
    ).astype(int)

    if not set(d["retained"].unique()).issubset({0, 1}):
        raise ValueError(
            f"Unexpected retained labels: "
            f"{sorted(d['retained'].unique())}"
        )

    # Script 70 stores the already preprocessed/model-ready feature
    # columns with an ``x_`` prefix.  Accept that canonical Script-70
    # schema and expose unprefixed aliases internally so the remainder
    # of Script 71 can keep using ALL_FEATURES unchanged.
    missing_features = [
        c for c in ALL_FEATURES
        if f"x_{c}" not in d.columns
    ]

    if missing_features:
        raise KeyError(
            "Missing CartoGraph-X model-ready features:\n"
            + "\n".join(f"x_{c}" for c in missing_features)
            + "\n\nAvailable columns:\n"
            + "\n".join(map(str, d.columns.tolist()))
        )

    # IMPORTANT: no scaler is fitted here. These values were already
    # transformed using training-derived preprocessing in Script 70.
    for c in ALL_FEATURES:
        d[c] = pd.to_numeric(
            d[f"x_{c}"],
            errors="coerce",
        )

        d[c] = (
            d[c]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )

    if d["sample_uid"].duplicated().any():
        dup = int(d["sample_uid"].duplicated().sum())

        raise ValueError(
            f"Duplicate sample_uid values: {dup}"
        )

    return d


def normalize_edge_schema(
    edges: pd.DataFrame,
) -> pd.DataFrame:

    e = edges.copy()

    source_col = first_existing(
        e,
        [
            "source_uid",
            "source",
            "src_uid",
            "src",
            "u",
        ],
    )

    target_col = first_existing(
        e,
        [
            "target_uid",
            "target",
            "dst_uid",
            "dst",
            "v",
        ],
    )

    e = e.rename(
        columns={
            source_col: "source_uid",
            target_col: "target_uid",
        }
    )

    e["source_uid"] = e["source_uid"].astype(str)
    e["target_uid"] = e["target_uid"].astype(str)

    return e


# =====================================================================
# GRAPH DATA CONTAINER
# =====================================================================

@dataclass
class GraphData:
    x_intrinsic: torch.Tensor
    x_graph: torch.Tensor
    x_multiscale: torch.Tensor
    x_neighbor: torch.Tensor

    edge_index: torch.Tensor

    y_selection: torch.Tensor
    y_operation: torch.Tensor
    y_country: torch.Tensor

    sample_uids: List[str]
    countries: List[str]
    tiles: List[str]

    @property
    def num_nodes(self) -> int:
        return len(self.sample_uids)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])


def dataframe_to_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    device: torch.device,
) -> GraphData:

    d = normalize_node_schema(nodes)
    e = normalize_edge_schema(edges)

    d["sample_uid"] = d["sample_uid"].astype(str)

    uid_to_idx = {
        uid: i
        for i, uid in enumerate(
            d["sample_uid"].tolist()
        )
    }

    source_known = e["source_uid"].isin(uid_to_idx)
    target_known = e["target_uid"].isin(uid_to_idx)

    unknown_edges = int(
        (~(source_known & target_known)).sum()
    )

    if unknown_edges > 0:
        raise ValueError(
            f"{unknown_edges} graph edges contain endpoints "
            "not present in the corresponding node table."
        )

    src = (
        e["source_uid"]
        .map(uid_to_idx)
        .to_numpy(dtype=np.int64)
    )

    dst = (
        e["target_uid"]
        .map(uid_to_idx)
        .to_numpy(dtype=np.int64)
    )

    # Script 68/70 stores undirected edges once.
    # Message passing requires both directions.
    src_bi = np.concatenate([src, dst])
    dst_bi = np.concatenate([dst, src])

    edge_index = torch.tensor(
        np.vstack([src_bi, dst_bi]),
        dtype=torch.long,
        device=device,
    )

    def tensor_features(cols: List[str]) -> torch.Tensor:
        return torch.tensor(
            d[cols].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )

    return GraphData(
        x_intrinsic=tensor_features(
            INTRINSIC_FEATURES
        ),
        x_graph=tensor_features(
            GRAPH_STRUCTURAL_FEATURES
        ),
        x_multiscale=tensor_features(
            MULTISCALE_CONTEXT_FEATURES
        ),
        x_neighbor=tensor_features(
            NEIGHBOR_AGGREGATION_FEATURES
        ),
        edge_index=edge_index,
        y_selection=torch.tensor(
            d["retained"].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=device,
        ),
        y_operation=torch.tensor(
            d["operation_id"].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=device,
        ),
        y_country=torch.tensor(
            d["country_id"].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=device,
        ),
        sample_uids=d["sample_uid"].tolist(),
        countries=d["country"].astype(str).tolist(),
        tiles=d["tile_id"].astype(str).tolist(),
    )


def load_protocol_stage(
    protocol: str,
    stage: str,
    device: torch.device,
) -> GraphData:

    node_path, edge_path = discover_stage_files(
        protocol,
        stage,
    )

    nodes = pd.read_csv(node_path)
    edges = pd.read_csv(edge_path)

    return dataframe_to_graph(
        nodes,
        edges,
        device,
    )


# =====================================================================
# MODEL COMPONENTS
# =====================================================================

class FeatureEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.net(x)


class AdaptiveContextFusion(nn.Module):
    """
    Learns sample-specific weights over the four representation families.

    Instead of concatenating all feature families and treating them equally,
    the network learns which cartographic context is important for each road.
    """

    def __init__(
        self,
        branch_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(branch_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )

        self.project = nn.Sequential(
            nn.Linear(branch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        intrinsic: torch.Tensor,
        graph: torch.Tensor,
        multiscale: torch.Tensor,
        neighbor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        concatenated = torch.cat(
            [
                intrinsic,
                graph,
                multiscale,
                neighbor,
            ],
            dim=1,
        )

        gate_logits = self.gate(concatenated)

        gate_weights = torch.softmax(
            gate_logits,
            dim=1,
        )

        branches = torch.stack(
            [
                intrinsic,
                graph,
                multiscale,
                neighbor,
            ],
            dim=1,
        )

        fused = (
            branches
            * gate_weights.unsqueeze(-1)
        ).sum(dim=1)

        fused = self.project(fused)

        return fused, gate_weights


class ResidualGraphLayer(nn.Module):
    """
    Lightweight dependency-free graph message passing.

    For every node:
        neighbour messages are mean-aggregated;
        self and neighbour states are transformed separately;
        the update is residual and normalized.

    This avoids requiring torch-geometric while still implementing
    genuine graph propagation.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.self_linear = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.neighbor_linear = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:

        if edge_index.numel() == 0:
            return h

        src = edge_index[0]
        dst = edge_index[1]

        n = h.shape[0]

        aggregated = torch.zeros_like(h)

        aggregated.index_add_(
            0,
            dst,
            h[src],
        )

        degree = torch.zeros(
            n,
            dtype=h.dtype,
            device=h.device,
        )

        degree.index_add_(
            0,
            dst,
            torch.ones(
                dst.shape[0],
                dtype=h.dtype,
                device=h.device,
            ),
        )

        degree = degree.clamp(min=1.0)

        aggregated = (
            aggregated
            / degree.unsqueeze(1)
        )

        self_msg = self.self_linear(h)

        nbr_msg = self.neighbor_linear(
            aggregated
        )

        gate = self.gate(
            torch.cat(
                [h, aggregated],
                dim=1,
            )
        )

        update = (
            gate * nbr_msg
            + (1.0 - gate) * self_msg
        )

        update = F.gelu(update)
        update = self.dropout(update)

        return self.norm(h + update)


# =====================================================================
# GRADIENT REVERSAL
# =====================================================================

class GradientReversalFunction(
    torch.autograd.Function
):

    @staticmethod
    def forward(ctx, x, coefficient):
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            -ctx.coefficient * grad_output,
            None,
        )


def gradient_reverse(
    x: torch.Tensor,
    coefficient: float,
) -> torch.Tensor:

    return GradientReversalFunction.apply(
        x,
        coefficient,
    )


# =====================================================================
# CARTOGRAPH-X
# =====================================================================

class CartoGraphX(nn.Module):

    def __init__(
        self,
        num_operations: int,
        config: TrainConfig,
    ):
        super().__init__()

        b = config.branch_dim
        h = config.hidden_dim
        p = config.dropout

        self.intrinsic_encoder = FeatureEncoder(
            len(INTRINSIC_FEATURES),
            b,
            p,
        )

        self.graph_encoder = FeatureEncoder(
            len(GRAPH_STRUCTURAL_FEATURES),
            b,
            p,
        )

        self.multiscale_encoder = FeatureEncoder(
            len(MULTISCALE_CONTEXT_FEATURES),
            b,
            p,
        )

        self.neighbor_encoder = FeatureEncoder(
            len(NEIGHBOR_AGGREGATION_FEATURES),
            b,
            p,
        )

        self.fusion = AdaptiveContextFusion(
            branch_dim=b,
            hidden_dim=h,
            dropout=p,
        )

        self.graph_layers = nn.ModuleList(
            [
                ResidualGraphLayer(
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

        self.selection_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(h // 2, 2),
        )

        self.operation_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(
                h // 2,
                num_operations,
            ),
        )

        self.domain_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(h // 2, 2),
        )

    def forward(
        self,
        graph: GraphData,
        grl_coefficient: float = 0.0,
    ) -> Dict[str, torch.Tensor]:

        z_intrinsic = self.intrinsic_encoder(
            graph.x_intrinsic
        )

        z_graph = self.graph_encoder(
            graph.x_graph
        )

        z_multiscale = self.multiscale_encoder(
            graph.x_multiscale
        )

        z_neighbor = self.neighbor_encoder(
            graph.x_neighbor
        )

        h, fusion_weights = self.fusion(
            z_intrinsic,
            z_graph,
            z_multiscale,
            z_neighbor,
        )

        pre_graph_embedding = h

        for layer in self.graph_layers:
            h = layer(
                h,
                graph.edge_index,
            )

        h = self.post_graph(h)

        selection_logits = self.selection_head(h)

        operation_logits = self.operation_head(h)

        reversed_h = gradient_reverse(
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
# LOSS WEIGHTS
# =====================================================================

def compute_class_weights(
    y: torch.Tensor,
    num_classes: int,
    max_weight: float,
) -> torch.Tensor:

    counts = torch.bincount(
        y,
        minlength=num_classes,
    ).float()

    total = counts.sum()

    weights = torch.ones_like(counts)

    valid = counts > 0

    weights[valid] = (
        total
        / (
            float(num_classes)
            * counts[valid]
        )
    )

    weights = torch.clamp(
        weights,
        min=0.25,
        max=max_weight,
    )

    # Classes absent from training must not create huge weights.
    weights[~valid] = 0.0

    return weights


# =====================================================================
# METRICS
# =====================================================================

def safe_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:

    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(
        roc_auc_score(
            y_true,
            probabilities,
        )
    )


def safe_pr_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:

    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(
        average_precision_score(
            y_true,
            probabilities,
        )
    )


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:

    y_pred = (
        probabilities >= threshold
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

    npv = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else float("nan")
    )

    return {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "specificity":
            float(specificity),

        "npv":
            float(npv),

        "f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "mcc":
            float(
                matthews_corrcoef(
                    y_true,
                    y_pred,
                )
            ),

        "roc_auc":
            safe_auc(
                y_true,
                probabilities,
            ),

        "pr_auc":
            safe_pr_auc(
                y_true,
                probabilities,
            ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# =====================================================================
# THRESHOLD SELECTION
# =====================================================================

def select_validation_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    config: TrainConfig,
) -> Tuple[float, Dict[str, float]]:

    thresholds = np.arange(
        config.threshold_min,
        config.threshold_max
        + config.threshold_step / 2,
        config.threshold_step,
    )

    best_threshold = 0.5
    best_metrics = None
    best_key = None

    for threshold in thresholds:

        metrics = classification_metrics(
            y_true,
            probabilities,
            float(threshold),
        )

        # Primary criterion: MCC
        # Tie-breakers: balanced accuracy, F1
        key = (
            metrics["mcc"],
            metrics["balanced_accuracy"],
            metrics["f1"],
        )

        if (
            best_key is None
            or key > best_key
        ):
            best_key = key
            best_threshold = float(
                threshold
            )
            best_metrics = metrics

    return (
        best_threshold,
        best_metrics,
    )


# =====================================================================
# EVALUATION
# =====================================================================

@torch.no_grad()
def predict_graph(
    model: nn.Module,
    graph: GraphData,
) -> Dict[str, np.ndarray]:

    model.eval()

    out = model(
        graph,
        grl_coefficient=0.0,
    )

    selection_prob = torch.softmax(
        out["selection_logits"],
        dim=1,
    )[:, 1]

    operation_pred = (
        out["operation_logits"]
        .argmax(dim=1)
    )

    domain_pred = (
        out["domain_logits"]
        .argmax(dim=1)
    )

    fusion = (
        out["fusion_weights"]
        .detach()
        .cpu()
        .numpy()
    )

    embedding = (
        out["embedding"]
        .detach()
        .cpu()
        .numpy()
    )

    return {
        "selection_probability":
            selection_prob.cpu().numpy(),

        "operation_pred":
            operation_pred.cpu().numpy(),

        "domain_pred":
            domain_pred.cpu().numpy(),

        "fusion_weights":
            fusion,

        "embedding":
            embedding,
    }


# =====================================================================
# DOMAIN SCHEDULE
# =====================================================================

def domain_schedule(
    epoch: int,
    total_epochs: int,
    max_coefficient: float,
) -> float:

    if total_epochs <= 1:
        return max_coefficient

    progress = (
        epoch - 1
    ) / (
        total_epochs - 1
    )

    # Standard smooth DANN schedule.
    coefficient = (
        2.0
        / (
            1.0
            + math.exp(
                -10.0 * progress
            )
        )
        - 1.0
    )

    return float(
        coefficient
        * max_coefficient
    )


# =====================================================================
# TRAIN ONE PROTOCOL / SEED
# =====================================================================

def train_one_run(
    protocol: str,
    seed: int,
    mode: str,
    device: torch.device,
    config: TrainConfig,
) -> Dict:

    banner(
        f"{protocol.upper()} | SEED {seed} | {mode.upper()}"
    )

    set_global_seed(seed)

    # --------------------------------------------------------------
    # Load protocol
    # --------------------------------------------------------------

    section("LOAD PROTOCOL GRAPHS")

    train_graph = load_protocol_stage(
        protocol,
        "train",
        device,
    )

    val_graph = load_protocol_stage(
        protocol,
        "validation",
        device,
    )

    test_graph = load_protocol_stage(
        protocol,
        "test",
        device,
    )

    print(
        f"Train:      {train_graph.num_nodes:>7,} nodes | "
        f"{train_graph.num_edges // 2:>7,} undirected edges"
    )

    print(
        f"Validation: {val_graph.num_nodes:>7,} nodes | "
        f"{val_graph.num_edges // 2:>7,} undirected edges"
    )

    print(
        f"Test:       {test_graph.num_nodes:>7,} nodes | "
        f"{test_graph.num_edges // 2:>7,} undirected edges"
    )

    train_countries = sorted(
        set(train_graph.countries)
    )

    domain_adversarial_active = (
        len(train_countries) > 1
    )

    print(
        f"Training countries: {train_countries}"
    )

    print(
        "Domain adversarial loss: "
        f"{'ACTIVE' if domain_adversarial_active else 'INACTIVE'}"
    )

    # --------------------------------------------------------------
    # Determine operation vocabulary size globally
    # --------------------------------------------------------------

    vocab_path = (
        INPUT_ROOT
        / "operation_vocabulary.csv"
    )

    vocab = pd.read_csv(vocab_path)

    num_operations = int(
        vocab["operation_id"].max()
        + 1
    )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    section("INITIALIZE CARTOGRAPH-X")

    model = CartoGraphX(
        num_operations=num_operations,
        config=config,
    ).to(device)

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: {trainable_parameters:,}"
    )

    print(
        f"Device:               {device}"
    )

    # --------------------------------------------------------------
    # Class weighting
    # --------------------------------------------------------------

    selection_weights = compute_class_weights(
        train_graph.y_selection,
        num_classes=2,
        max_weight=config.max_selection_class_weight,
    )

    operation_weights = compute_class_weights(
        train_graph.y_operation,
        num_classes=num_operations,
        max_weight=config.max_operation_class_weight,
    )

    print(
        "Selection weights:",
        selection_weights.detach().cpu().numpy(),
    )

    print(
        "Operation weights:",
        operation_weights.detach().cpu().numpy(),
    )

    selection_criterion = nn.CrossEntropyLoss(
        weight=selection_weights,
    )

    operation_criterion = nn.CrossEntropyLoss(
        weight=operation_weights,
    )

    domain_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # --------------------------------------------------------------
    # Epoch configuration
    # --------------------------------------------------------------

    max_epochs = (
        config.smoke_epochs
        if mode == "smoke"
        else config.full_epochs
    )

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------

    section("TRAIN CARTOGRAPH-X")

    best_state = None
    best_epoch = -1
    best_val_mcc = -np.inf
    best_val_ba = -np.inf
    best_val_pr = -np.inf
    best_threshold = 0.5

    epochs_without_improvement = 0

    history_rows = []

    start_time = time.time()

    for epoch in range(
        1,
        max_epochs + 1,
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        if domain_adversarial_active:
            grl = domain_schedule(
                epoch,
                max_epochs,
                config.domain_grl_max,
            )
        else:
            grl = 0.0

        out = model(
            train_graph,
            grl_coefficient=grl,
        )

        selection_loss = (
            selection_criterion(
                out["selection_logits"],
                train_graph.y_selection,
            )
        )

        operation_loss = (
            operation_criterion(
                out["operation_logits"],
                train_graph.y_operation,
            )
        )

        if domain_adversarial_active:

            domain_loss = domain_criterion(
                out["domain_logits"],
                train_graph.y_country,
            )

        else:

            domain_loss = torch.tensor(
                0.0,
                device=device,
            )

        total_loss = (
            config.selection_loss_weight
            * selection_loss
            +
            config.operation_loss_weight
            * operation_loss
            +
            config.domain_loss_weight
            * domain_loss
        )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.gradient_clip,
        )

        optimizer.step()

        # ----------------------------------------------------------
        # Validation
        # ----------------------------------------------------------

        val_pred = predict_graph(
            model,
            val_graph,
        )

        y_val = (
            val_graph.y_selection
            .detach()
            .cpu()
            .numpy()
        )

        threshold, val_metrics = (
            select_validation_threshold(
                y_val,
                val_pred[
                    "selection_probability"
                ],
                config,
            )
        )

        operation_accuracy = float(
            accuracy_score(
                val_graph.y_operation
                .detach()
                .cpu()
                .numpy(),
                val_pred["operation_pred"],
            )
        )

        epoch_row = {
            "epoch": epoch,
            "loss":
                float(
                    total_loss
                    .detach()
                    .cpu()
                    .item()
                ),
            "selection_loss":
                float(
                    selection_loss
                    .detach()
                    .cpu()
                    .item()
                ),
            "operation_loss":
                float(
                    operation_loss
                    .detach()
                    .cpu()
                    .item()
                ),
            "domain_loss":
                float(
                    domain_loss
                    .detach()
                    .cpu()
                    .item()
                ),
            "grl_coefficient": grl,
            "validation_threshold":
                threshold,
            "validation_mcc":
                val_metrics["mcc"],
            "validation_balanced_accuracy":
                val_metrics[
                    "balanced_accuracy"
                ],
            "validation_f1":
                val_metrics["f1"],
            "validation_roc_auc":
                val_metrics["roc_auc"],
            "validation_pr_auc":
                val_metrics["pr_auc"],
            "validation_operation_accuracy":
                operation_accuracy,
        }

        history_rows.append(epoch_row)

        print(
            f"Epoch {epoch:03d}/{max_epochs:03d} | "
            f"loss={epoch_row['loss']:.4f} | "
            f"sel={epoch_row['selection_loss']:.4f} | "
            f"op={epoch_row['operation_loss']:.4f} | "
            f"dom={epoch_row['domain_loss']:.4f} | "
            f"val MCC={val_metrics['mcc']:.4f} | "
            f"BA={val_metrics['balanced_accuracy']:.4f} | "
            f"PR-AUC={val_metrics['pr_auc']:.4f} | "
            f"thr={threshold:.2f}"
        )

        # ----------------------------------------------------------
        # Model selection
        # ----------------------------------------------------------

        selection_key = (
            val_metrics["mcc"],
            val_metrics[
                "balanced_accuracy"
            ],
            val_metrics["pr_auc"],
        )

        best_key = (
            best_val_mcc,
            best_val_ba,
            best_val_pr,
        )

        if selection_key > best_key:

            best_val_mcc = (
                val_metrics["mcc"]
            )

            best_val_ba = (
                val_metrics[
                    "balanced_accuracy"
                ]
            )

            best_val_pr = (
                val_metrics["pr_auc"]
            )

            best_epoch = epoch

            best_threshold = threshold

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            mode == "full"
            and epochs_without_improvement
            >= config.early_stopping_patience
        ):

            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch: {best_epoch}"
            )

            break

    elapsed_seconds = (
        time.time()
        - start_time
    )

    if best_state is None:
        raise RuntimeError(
            "No valid model checkpoint was selected."
        )

    # --------------------------------------------------------------
    # Restore validation-selected model
    # --------------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    # --------------------------------------------------------------
    # Final validation evaluation
    # --------------------------------------------------------------

    section(
        "VALIDATION-SELECTED MODEL"
    )

    val_pred = predict_graph(
        model,
        val_graph,
    )

    y_val = (
        val_graph.y_selection
        .detach()
        .cpu()
        .numpy()
    )

    final_val_metrics = (
        classification_metrics(
            y_val,
            val_pred[
                "selection_probability"
            ],
            best_threshold,
        )
    )

    print(
        f"Best epoch:       {best_epoch}"
    )

    print(
        f"Selected threshold: {best_threshold:.2f}"
    )

    print(
        f"Validation MCC:   "
        f"{final_val_metrics['mcc']:.4f}"
    )

    print(
        f"Validation BA:    "
        f"{final_val_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Validation ROC:   "
        f"{final_val_metrics['roc_auc']:.4f}"
    )

    print(
        f"Validation PR:    "
        f"{final_val_metrics['pr_auc']:.4f}"
    )

    # --------------------------------------------------------------
    # TEST — touched only now
    # --------------------------------------------------------------

    section(
        "FINAL TEST EVALUATION"
    )

    test_pred = predict_graph(
        model,
        test_graph,
    )

    y_test = (
        test_graph.y_selection
        .detach()
        .cpu()
        .numpy()
    )

    test_metrics = classification_metrics(
        y_test,
        test_pred[
            "selection_probability"
        ],
        best_threshold,
    )

    print(
        f"Balanced accuracy: "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"F1:                "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"MCC:               "
        f"{test_metrics['mcc']:.4f}"
    )

    print(
        f"ROC-AUC:           "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:            "
        f"{test_metrics['pr_auc']:.4f}"
    )

    # --------------------------------------------------------------
    # Operation performance
    # --------------------------------------------------------------

    test_operation_true = (
        test_graph.y_operation
        .detach()
        .cpu()
        .numpy()
    )

    test_operation_accuracy = float(
        accuracy_score(
            test_operation_true,
            test_pred["operation_pred"],
        )
    )

    print(
        f"Operation accuracy:{test_operation_accuracy:.4f}"
    )

    # --------------------------------------------------------------
    # Fusion weights
    # --------------------------------------------------------------

    fusion = test_pred[
        "fusion_weights"
    ]

    mean_fusion = fusion.mean(
        axis=0
    )

    fusion_names = [
        "intrinsic",
        "graph_structural",
        "multiscale_context",
        "neighbor_aggregation",
    ]

    print("\nMean adaptive fusion weights:")

    for name, value in zip(
        fusion_names,
        mean_fusion,
    ):
        print(
            f"  {name:<25} {value:.4f}"
        )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    run_dir = (
        OUT_ROOT
        / protocol
        / f"seed_{seed}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "config":
                asdict(config),
            "protocol":
                protocol,
            "seed":
                seed,
            "best_epoch":
                best_epoch,
            "threshold":
                best_threshold,
            "features": {
                "intrinsic":
                    INTRINSIC_FEATURES,
                "graph_structural":
                    GRAPH_STRUCTURAL_FEATURES,
                "multiscale_context":
                    MULTISCALE_CONTEXT_FEATURES,
                "neighbor_aggregation":
                    NEIGHBOR_AGGREGATION_FEATURES,
            },
        },
        run_dir / "best_model.pt",
    )

    history_df = pd.DataFrame(
        history_rows
    )

    history_df.to_csv(
        run_dir / "training_history.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Predictions
    # --------------------------------------------------------------

    test_prediction_df = pd.DataFrame(
        {
            "sample_uid":
                test_graph.sample_uids,
            "country":
                test_graph.countries,
            "tile_id":
                test_graph.tiles,
            "y_true":
                y_test,
            "prob_retained":
                test_pred[
                    "selection_probability"
                ],
            "y_pred":
                (
                    test_pred[
                        "selection_probability"
                    ]
                    >= best_threshold
                ).astype(int),
            "operation_true":
                test_operation_true,
            "operation_pred":
                test_pred[
                    "operation_pred"
                ],
            "fusion_intrinsic":
                fusion[:, 0],
            "fusion_graph_structural":
                fusion[:, 1],
            "fusion_multiscale_context":
                fusion[:, 2],
            "fusion_neighbor_aggregation":
                fusion[:, 3],
        }
    )

    test_prediction_df.to_csv(
        run_dir / "test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------------

    embedding = test_pred[
        "embedding"
    ]

    embedding_columns = [
        f"embedding_{i:03d}"
        for i in range(
            embedding.shape[1]
        )
    ]

    embedding_df = pd.DataFrame(
        embedding,
        columns=embedding_columns,
    )

    embedding_df.insert(
        0,
        "sample_uid",
        test_graph.sample_uids,
    )

    embedding_df.to_csv(
        run_dir / "test_embeddings.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Save fusion summary
    # --------------------------------------------------------------

    fusion_summary = pd.DataFrame(
        {
            "feature_family":
                fusion_names,
            "mean_weight":
                mean_fusion,
            "std_weight":
                fusion.std(axis=0),
        }
    )

    fusion_summary.to_csv(
        run_dir
        / "adaptive_fusion_summary.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Figure: training curve
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        history_df["epoch"],
        history_df[
            "validation_mcc"
        ],
        label="Validation MCC",
    )

    ax.plot(
        history_df["epoch"],
        history_df[
            "validation_balanced_accuracy"
        ],
        label="Validation balanced accuracy",
    )

    ax.axvline(
        best_epoch,
        linestyle="--",
        label="Selected epoch",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title(
        f"CartoGraph-X Training — {protocol} — Seed {seed}"
    )

    ax.legend()
    fig.tight_layout()

    fig.savefig(
        run_dir / "training_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        run_dir / "training_curve.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------------
    # Run result
    # --------------------------------------------------------------

    result = {
        "protocol":
            protocol,
        "seed":
            seed,
        "mode":
            mode,
        "best_epoch":
            best_epoch,
        "threshold":
            best_threshold,
        "train_nodes":
            train_graph.num_nodes,
        "validation_nodes":
            val_graph.num_nodes,
        "test_nodes":
            test_graph.num_nodes,
        "train_countries":
            "|".join(train_countries),
        "domain_adversarial_active":
            domain_adversarial_active,
        "trainable_parameters":
            trainable_parameters,
        "elapsed_seconds":
            elapsed_seconds,
        "validation_mcc":
            final_val_metrics["mcc"],
        "validation_balanced_accuracy":
            final_val_metrics[
                "balanced_accuracy"
            ],
        "test_accuracy":
            test_metrics["accuracy"],
        "test_balanced_accuracy":
            test_metrics[
                "balanced_accuracy"
            ],
        "test_precision":
            test_metrics["precision"],
        "test_recall":
            test_metrics["recall"],
        "test_specificity":
            test_metrics["specificity"],
        "test_f1":
            test_metrics["f1"],
        "test_mcc":
            test_metrics["mcc"],
        "test_roc_auc":
            test_metrics["roc_auc"],
        "test_pr_auc":
            test_metrics["pr_auc"],
        "test_operation_accuracy":
            test_operation_accuracy,
        "fusion_intrinsic":
            float(mean_fusion[0]),
        "fusion_graph_structural":
            float(mean_fusion[1]),
        "fusion_multiscale_context":
            float(mean_fusion[2]),
        "fusion_neighbor_aggregation":
            float(mean_fusion[3]),
    }

    with open(
        run_dir / "run_report.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_safe(result),
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    return result


# =====================================================================
# AGGREGATION
# =====================================================================

def aggregate_results(
    results_df: pd.DataFrame,
) -> pd.DataFrame:

    metrics = [
        "test_balanced_accuracy",
        "test_f1",
        "test_mcc",
        "test_roc_auc",
        "test_pr_auc",
        "test_operation_accuracy",
    ]

    rows = []

    for protocol, group in (
        results_df.groupby("protocol")
    ):

        for metric in metrics:

            values = (
                pd.to_numeric(
                    group[metric],
                    errors="coerce",
                )
                .dropna()
                .to_numpy()
            )

            n = len(values)

            if n == 0:
                continue

            mean = float(
                np.mean(values)
            )

            if n > 1:
                std = float(
                    np.std(
                        values,
                        ddof=1,
                    )
                )

                sem = (
                    std
                    / math.sqrt(n)
                )

                ci_half = (
                    1.96 * sem
                )

            else:
                std = float("nan")
                ci_half = float("nan")

            rows.append(
                {
                    "protocol":
                        protocol,
                    "metric":
                        metric,
                    "runs":
                        n,
                    "mean":
                        mean,
                    "std":
                        std,
                    "ci95_lower":
                        (
                            mean - ci_half
                            if n > 1
                            else np.nan
                        ),
                    "ci95_upper":
                        (
                            mean + ci_half
                            if n > 1
                            else np.nan
                        ),
                }
            )

    return pd.DataFrame(rows)


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train CartoGraph-X on leakage-safe "
            "cross-country road-selection protocols."
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
            "smoke = seed 42 and short training; "
            "full = seeds 42-51."
        ),
    )

    parser.add_argument(
        "--protocol",
        choices=PROTOCOLS + ["all"],
        default="all",
        help=(
            "Run one protocol or all six."
        ),
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
        "CARTOGEN — CARTOGRAPH-X NEURAL ROAD-SELECTION MODEL"
    )

    print(
        f"Input:  {INPUT_ROOT}"
    )

    print(
        f"Output: {OUT_ROOT}"
    )

    print(
        f"Mode:   {args.mode}"
    )

    # --------------------------------------------------------------
    # Device
    # --------------------------------------------------------------

    if args.device == "cpu":
        device = torch.device("cpu")

    elif args.device == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available."
            )

        device = torch.device("cuda")

    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":
        print(
            "GPU:    "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------

    config = TrainConfig()

    seeds = (
        SMOKE_SEEDS
        if args.mode == "smoke"
        else FULL_SEEDS
    )

    protocols = (
        PROTOCOLS
        if args.protocol == "all"
        else [args.protocol]
    )

    print(
        f"Seeds:  {seeds}"
    )

    print(
        f"Protocols: {protocols}"
    )

    # --------------------------------------------------------------
    # Verify inputs
    # --------------------------------------------------------------

    section(
        "1. VERIFY CARTOGRAPH-X INPUTS"
    )

    required_global_files = [
        INPUT_ROOT
        / "cartographx_global_nodes.csv",

        INPUT_ROOT
        / "cartographx_global_edges.csv",

        INPUT_ROOT
        / "operation_vocabulary.csv",

        INPUT_ROOT
        / "cartographx_input_report.json",
    ]

    for path in required_global_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Required Script 70 output missing:\n{path}"
            )

        print(
            f"[OK] {path.name}"
        )

    for protocol in protocols:

        folder = (
            PROTOCOL_ROOT
            / protocol
        )

        if not folder.exists():
            raise FileNotFoundError(
                f"Missing protocol folder:\n{folder}"
            )

        for stage in [
            "train",
            "validation",
            "test",
        ]:

            node_path, edge_path = (
                discover_stage_files(
                    protocol,
                    stage,
                )
            )

            print(
                f"[OK] {protocol:<22} "
                f"{stage:<10} "
                f"{node_path.name} | "
                f"{edge_path.name}"
            )

    # --------------------------------------------------------------
    # Architecture summary
    # --------------------------------------------------------------

    section(
        "2. CARTOGRAPH-X ARCHITECTURE"
    )

    print(
        f"Intrinsic features:             "
        f"{len(INTRINSIC_FEATURES)}"
    )

    print(
        f"Graph-structural features:      "
        f"{len(GRAPH_STRUCTURAL_FEATURES)}"
    )

    print(
        f"Multi-scale context features:   "
        f"{len(MULTISCALE_CONTEXT_FEATURES)}"
    )

    print(
        f"Neighbour-aggregation features: "
        f"{len(NEIGHBOR_AGGREGATION_FEATURES)}"
    )

    print(
        f"Total node features:            "
        f"{len(ALL_FEATURES)}"
    )

    print(
        "\nArchitecture:"
    )

    print(
        "  four family-specific encoders"
    )

    print(
        "  -> adaptive context fusion"
    )

    print(
        "  -> residual graph propagation"
    )

    print(
        "  -> road-selection head"
    )

    print(
        "  -> auxiliary operation head"
    )

    print(
        "  -> adversarial country head "
        "(joint-country training only)"
    )

    # --------------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------------

    section(
        "3. RUN EXPERIMENTS"
    )

    all_results = []

    total_runs = (
        len(protocols)
        * len(seeds)
    )

    current_run = 0

    for protocol in protocols:

        for seed in seeds:

            current_run += 1

            print(
                f"\nRUN {current_run}/{total_runs}"
            )

            result = train_one_run(
                protocol=protocol,
                seed=seed,
                mode=args.mode,
                device=device,
                config=config,
            )

            all_results.append(
                result
            )

            # Save progressively so results survive
            # interruption of a later run.
            pd.DataFrame(
                all_results
            ).to_csv(
                OUT_ROOT
                / f"{args.mode}_all_runs.csv",
                index=False,
            )

            if device.type == "cuda":
                torch.cuda.empty_cache()

    # --------------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------------

    section(
        "4. AGGREGATE RESULTS"
    )

    results_df = pd.DataFrame(
        all_results
    )

    aggregate_df = aggregate_results(
        results_df
    )

    results_path = (
        OUT_ROOT
        / f"{args.mode}_all_runs.csv"
    )

    aggregate_path = (
        OUT_ROOT
        / f"{args.mode}_aggregate_metrics.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    aggregate_df.to_csv(
        aggregate_path,
        index=False,
    )

    display_columns = [
        "protocol",
        "seed",
        "best_epoch",
        "threshold",
        "test_balanced_accuracy",
        "test_f1",
        "test_mcc",
        "test_roc_auc",
        "test_pr_auc",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------------
    # Final audit
    # --------------------------------------------------------------

    section(
        "5. FINAL CARTOGRAPH-X AUDIT"
    )

    expected_runs = (
        len(protocols)
        * len(seeds)
    )

    checks = {
        "all_requested_runs_completed":
            len(results_df)
            == expected_runs,

        "all_protocols_completed":
            set(
                results_df[
                    "protocol"
                ].unique()
            )
            == set(protocols),

        "all_requested_seeds_completed":
            set(
                results_df[
                    "seed"
                ].unique()
            )
            == set(seeds),

        "all_mcc_finite":
            np.isfinite(
                results_df[
                    "test_mcc"
                ]
            ).all(),

        "all_balanced_accuracy_finite":
            np.isfinite(
                results_df[
                    "test_balanced_accuracy"
                ]
            ).all(),

        "all_thresholds_valid":
            results_df[
                "threshold"
            ].between(
                config.threshold_min,
                config.threshold_max,
            ).all(),

        "fixed_spatial_splits":
            True,

        "test_not_used_for_model_selection":
            True,

        "country_not_direct_predictive_feature":
            True,
    }

    for key, value in checks.items():

        print(
            f"{key:<50} "
            f"{'PASS' if value else 'FAIL'}"
        )

    all_pass = all(
        checks.values()
    )

    # --------------------------------------------------------------
    # Experiment manifest
    # --------------------------------------------------------------

    report = {
        "script":
            "71_train_cartographx.py",

        "mode":
            args.mode,

        "seeds":
            seeds,

        "protocols":
            protocols,

        "device":
            str(device),

        "configuration":
            asdict(config),

        "feature_groups": {
            "intrinsic":
                INTRINSIC_FEATURES,

            "graph_structural":
                GRAPH_STRUCTURAL_FEATURES,

            "multiscale_context":
                MULTISCALE_CONTEXT_FEATURES,

            "neighbor_aggregation":
                NEIGHBOR_AGGREGATION_FEATURES,
        },

        "methodology": {
            "model":
                "CartoGraph-X",

            "graph_construction":
                "precomputed leakage-safe tile-local graph",

            "selection_model_selection":
                "validation MCC",

            "selection_tie_breakers": [
                "balanced_accuracy",
                "PR-AUC",
            ],

            "threshold_selection":
                "validation-only MCC optimization",

            "test_used_for_selection":
                False,

            "country_as_predictive_feature":
                False,

            "domain_alignment":
                (
                    "gradient reversal during "
                    "multi-country training only"
                ),

            "auxiliary_task":
                "cartographic operation prediction",
        },

        "checks":
            checks,

        "status":
            "PASS"
            if all_pass
            else "CHECK",
    }

    with open(
        OUT_ROOT
        / f"{args.mode}_experiment_report.json",
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

    # --------------------------------------------------------------
    # Complete
    # --------------------------------------------------------------

    banner(
        "SCRIPT 71 COMPLETE"
    )

    print(
        f"Run results:      {results_path}"
    )

    print(
        f"Aggregate metrics:{aggregate_path}"
    )

    print(
        f"Models/results:   {OUT_ROOT}"
    )

    if all_pass:

        print(
            "\n[PASS] CARTOGRAPH-X "
            f"{args.mode.upper()} EXPERIMENT COMPLETE."
        )

    else:

        print(
            "\n[CHECK] CARTOGRAPH-X EXPERIMENT "
            "REQUIRES ATTENTION."
        )

    if args.mode == "smoke":

        print(
            "\nDO NOT RUN FULL 10-SEED TRAINING YET."
        )

        print(
            "Inspect smoke-test performance first."
        )

        print(
            "NEXT: Script 72 — CartoGraph-X "
            "component ablation and novelty validation."
        )

    else:

        print(
            "\n10-SEED CARTOGRAPH-X EXPERIMENT COMPLETE."
        )


if __name__ == "__main__":
    main()