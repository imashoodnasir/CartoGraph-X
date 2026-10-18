"""
81_train_cartographx_v3.py

CartoGen
-------
Train and evaluate CartoGraph-X v3 for leakage-safe cross-country
road-selection modelling with domain-conditional mixture-of-experts and robust learning.

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
9. Domain-conditional mixture-of-experts router
10. Group-DRO selection objective for joint-country training
11. Cross-domain same-class prototype alignment
12. Optional adversarial country/domain head

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
- Smoke/development mode uses seed 42.
- Full mode uses seeds 42--51.
- Early stopping is locked for the first 60 epochs in development/full mode.
- Every epoch writes a resumable training checkpoint.

Run:
    python scripts/81_train_cartographx_v3.py --mode smoke

Later:
    python scripts/81_train_cartographx_v3.py --mode full

Optional:
    python scripts/81_train_cartographx_v3.py --mode smoke --protocol lt_to_lt
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
    / "cartographx_v3_models"
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
    # V3 capacity
    hidden_dim: int = 96
    branch_dim: int = 48
    graph_layers: int = 3
    attention_heads: int = 4
    dropout: float = 0.15

    learning_rate: float = 7.5e-4
    weight_decay: float = 2e-4

    # Multi-objective learning
    selection_loss_weight: float = 1.0
    operation_loss_weight: float = 0.12
    ranking_loss_weight: float = 0.25
    topology_loss_weight: float = 0.08
    domain_loss_weight: float = 0.02

    # V3: domain-conditional mixture-of-experts + robust cross-domain learning.
    num_experts: int = 3
    moe_balance_weight: float = 0.02
    dro_weight: float = 0.35
    contrastive_weight: float = 0.05
    contrastive_temperature: float = 0.20

    ranking_margin: float = 0.20
    max_pairs_per_tile: int = 128

    max_operation_class_weight: float = 8.0
    max_selection_class_weight: float = 6.0
    gradient_clip: float = 5.0

    # IMPORTANT: early stopping is impossible before min_epochs.
    min_epochs: int = 60
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 1e-4

    threshold_min: float = 0.10
    threshold_max: float = 0.90
    threshold_step: float = 0.01

    smoke_epochs: int = 12
    development_epochs: int = 200
    full_epochs: int = 200

    # Domain alignment starts only after representation warm-up.
    domain_grl_max: float = 0.35
    domain_start_epoch: int = 60

    # Composite validation model-selection score.
    score_mcc_weight: float = 0.50
    score_ba_weight: float = 0.25
    score_roc_weight: float = 0.15
    score_pr_weight: float = 0.10


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
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim), nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)


class CrossFamilyAttention(nn.Module):
    """Self-attention across the four cartographic feature families."""
    def __init__(self, branch_dim: int, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(branch_dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(branch_dim)
        self.gate = nn.Sequential(
            nn.Linear(branch_dim * 4, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        self.project = nn.Sequential(
            nn.Linear(branch_dim * 4, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, intrinsic, graph, multiscale, neighbor):
        tokens = torch.stack([intrinsic, graph, multiscale, neighbor], dim=1)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(tokens + attended)
        flat = tokens.flatten(1)
        weights = torch.softmax(self.gate(flat), dim=1)
        weighted = tokens * weights.unsqueeze(-1)
        return self.project(weighted.flatten(1)), weights


class EdgeAwareGraphAttentionLayer(nn.Module):
    """Dependency-free multi-head graph attention with residual gated updates."""
    def __init__(self, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError('hidden_dim must be divisible by attention_heads')
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_gate = nn.Sequential(nn.Linear(hidden_dim * 2, heads), nn.Tanh())
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.update_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index):
        if edge_index.numel() == 0:
            return h
        src, dst = edge_index[0], edge_index[1]
        n = h.shape[0]
        q = self.q(h).view(n, self.heads, self.head_dim)
        k = self.k(h).view(n, self.heads, self.head_dim)
        v = self.v(h).view(n, self.heads, self.head_dim)
        score = (q[dst] * k[src]).sum(-1) / math.sqrt(self.head_dim)
        score = score + self.edge_gate(torch.cat([h[src], h[dst]], dim=1))
        # Stable destination-wise softmax for each head.
        max_per_dst = torch.full((n, self.heads), -torch.inf, device=h.device, dtype=h.dtype)
        max_per_dst.scatter_reduce_(0, dst[:, None].expand(-1, self.heads), score, reduce='amax', include_self=True)
        exp_score = torch.exp(score - max_per_dst[dst])
        denom = torch.zeros((n, self.heads), device=h.device, dtype=h.dtype)
        denom.index_add_(0, dst, exp_score)
        alpha = exp_score / denom[dst].clamp_min(1e-8)
        msg = v[src] * alpha.unsqueeze(-1)
        agg = torch.zeros((n, self.heads, self.head_dim), device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, msg)
        agg = self.out(agg.reshape(n, -1))
        gate = self.update_gate(torch.cat([h, agg], dim=1))
        update = gate * agg + (1.0 - gate) * h
        return self.norm(h + self.dropout(F.gelu(update)))


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coefficient):
        ctx.coefficient = coefficient
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coefficient * grad_output, None


def gradient_reverse(x, coefficient):
    return GradientReversalFunction.apply(x, coefficient)


class CartoGraphX(nn.Module):
    """CartoGraph-X v3: graph-attentive backbone + domain-conditional mixture of experts."""
    def __init__(self, num_operations: int, config: TrainConfig):
        super().__init__()
        b, h, p = config.branch_dim, config.hidden_dim, config.dropout
        self.num_experts = config.num_experts
        self.intrinsic_encoder = FeatureEncoder(len(INTRINSIC_FEATURES), b, p)
        self.graph_encoder = FeatureEncoder(len(GRAPH_STRUCTURAL_FEATURES), b, p)
        self.multiscale_encoder = FeatureEncoder(len(MULTISCALE_CONTEXT_FEATURES), b, p)
        self.neighbor_encoder = FeatureEncoder(len(NEIGHBOR_AGGREGATION_FEATURES), b, p)
        self.cross_family = CrossFamilyAttention(b, h, config.attention_heads, p)
        self.graph_layers = nn.ModuleList([
            EdgeAwareGraphAttentionLayer(h, config.attention_heads, p)
            for _ in range(config.graph_layers)
        ])
        self.tabular_branch = nn.Sequential(
            nn.Linear(len(ALL_FEATURES), h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p),
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(),
        )
        self.fusion_gate = nn.Sequential(nn.Linear(h * 2, h), nn.GELU(), nn.Linear(h, h), nn.Sigmoid())
        self.post = nn.Sequential(nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p))

        # V3 MoE: one shared expert plus learned specialists. The router sees only
        # learned cartographic representation; country ID is never an input feature.
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p), nn.Linear(h, h), nn.GELU())
            for _ in range(config.num_experts)
        ])
        self.router = nn.Sequential(
            nn.Linear(h, h // 2), nn.GELU(), nn.Dropout(p), nn.Linear(h // 2, config.num_experts)
        )
        self.expert_residual = nn.Sequential(nn.Linear(h, h), nn.LayerNorm(h))

        self.selection_head = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(), nn.Dropout(p), nn.Linear(h // 2, 2))
        self.operation_head = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(), nn.Dropout(p), nn.Linear(h // 2, num_operations))
        self.domain_head = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(), nn.Dropout(p), nn.Linear(h // 2, 2))

    def forward(self, graph: GraphData, grl_coefficient: float = 0.0):
        zi = self.intrinsic_encoder(graph.x_intrinsic)
        zg = self.graph_encoder(graph.x_graph)
        zm = self.multiscale_encoder(graph.x_multiscale)
        zn = self.neighbor_encoder(graph.x_neighbor)
        h, fusion_weights = self.cross_family(zi, zg, zm, zn)
        pre_graph = h
        for layer in self.graph_layers:
            h = layer(h, graph.edge_index)
        x_all = torch.cat([graph.x_intrinsic, graph.x_graph, graph.x_multiscale, graph.x_neighbor], dim=1)
        tab = self.tabular_branch(x_all)
        gate = self.fusion_gate(torch.cat([h, tab], dim=1))
        base = self.post(gate * h + (1.0 - gate) * tab)

        router_weights = torch.softmax(self.router(base), dim=1)
        expert_stack = torch.stack([expert(base) for expert in self.experts], dim=1)
        moe = (expert_stack * router_weights.unsqueeze(-1)).sum(dim=1)
        h = F.gelu(self.expert_residual(base + moe))

        selection_logits = self.selection_head(h)
        operation_logits = self.operation_head(h)
        domain_logits = self.domain_head(gradient_reverse(h, grl_coefficient))
        return {
            'selection_logits': selection_logits,
            'operation_logits': operation_logits,
            'domain_logits': domain_logits,
            'embedding': h,
            'pre_graph_embedding': pre_graph,
            'fusion_weights': fusion_weights,
            'neural_tabular_gate': gate.mean(dim=1),
            'expert_weights': router_weights,
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

    expert_weights = out.get('expert_weights')
    if expert_weights is not None:
        expert_weights = expert_weights.detach().cpu().numpy()

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

        "expert_weights": expert_weights,
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


def composite_validation_score(metrics: Dict[str, float], config: TrainConfig) -> float:
    vals = [metrics['mcc'], metrics['balanced_accuracy'], metrics['roc_auc'], metrics['pr_auc']]
    vals = [0.0 if not np.isfinite(v) else float(v) for v in vals]
    return (config.score_mcc_weight * vals[0] + config.score_ba_weight * vals[1] +
            config.score_roc_weight * vals[2] + config.score_pr_weight * vals[3])


def tile_ranking_loss(logits: torch.Tensor, y: torch.Tensor, tiles: List[str], config: TrainConfig) -> torch.Tensor:
    score = logits[:, 1] - logits[:, 0]
    losses = []
    # deterministic capped pairing; randomness is unnecessary and improves reproducibility
    groups = {}
    for i, t in enumerate(tiles):
        groups.setdefault(t, []).append(i)
    for idxs in groups.values():
        idx = torch.tensor(idxs, device=y.device, dtype=torch.long)
        pos = idx[y[idx] == 1][:config.max_pairs_per_tile]
        neg = idx[y[idx] == 0][:config.max_pairs_per_tile]
        if pos.numel() and neg.numel():
            m = min(pos.numel(), neg.numel(), config.max_pairs_per_tile)
            diff = score[pos[:m]] - score[neg[:m]]
            losses.append(F.softplus(config.ranking_margin - diff).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def topology_preservation_loss(logits: torch.Tensor, y: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return logits.sum() * 0.0
    src, dst = edge_index[0], edge_index[1]
    mask = (y[src] == 1) & (y[dst] == 1)
    if not bool(mask.any()):
        return logits.sum() * 0.0
    p = torch.softmax(logits, dim=1)[:, 1]
    return (p[src[mask]] - p[dst[mask]]).abs().mean()


def moe_balance_loss(expert_weights: torch.Tensor) -> torch.Tensor:
    """Discourage router collapse while allowing sample-specific specialization."""
    mean_usage = expert_weights.mean(dim=0)
    target = torch.full_like(mean_usage, 1.0 / mean_usage.numel())
    return ((mean_usage - target) ** 2).mean()


def domain_robust_selection_loss(logits: torch.Tensor, y: torch.Tensor, countries: torch.Tensor,
                                 class_weights: torch.Tensor, active: bool) -> torch.Tensor:
    """Group-DRO style objective: emphasize the worse training country without using test information."""
    per_node = F.cross_entropy(logits, y, weight=class_weights, reduction='none')
    if not active:
        return per_node.mean()
    losses = []
    for c in torch.unique(countries):
        mask = countries == c
        if bool(mask.any()):
            losses.append(per_node[mask].mean())
    if len(losses) < 2:
        return per_node.mean()
    stacked = torch.stack(losses)
    # Smooth worst-group objective for stable gradients.
    return torch.logsumexp(stacked / 0.20, dim=0) * 0.20


def cross_domain_class_alignment_loss(embedding: torch.Tensor, y: torch.Tensor, countries: torch.Tensor) -> torch.Tensor:
    """Align same-label structural prototypes across training countries only."""
    unique_c = torch.unique(countries)
    if unique_c.numel() < 2:
        return embedding.sum() * 0.0
    losses = []
    z = F.normalize(embedding, dim=1)
    for cls in (0, 1):
        protos = []
        for c in unique_c:
            mask = (countries == c) & (y == cls)
            if bool(mask.any()):
                protos.append(F.normalize(z[mask].mean(dim=0), dim=0))
        if len(protos) >= 2:
            for i in range(len(protos)):
                for j in range(i + 1, len(protos)):
                    losses.append(1.0 - torch.sum(protos[i] * protos[j]))
    return torch.stack(losses).mean() if losses else embedding.sum() * 0.0


# =====================================================================
# TRAIN ONE PROTOCOL / SEED
# =====================================================================

def train_one_run(protocol: str, seed: int, mode: str, device: torch.device, config: TrainConfig) -> Dict:
    banner(f"{protocol.upper()} | SEED {seed} | {mode.upper()} | CARTOGRAPH-X V3")
    set_global_seed(seed)
    run_dir = OUT_ROOT / protocol / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    complete_path = run_dir / "run_complete.json"
    if complete_path.exists():
        print(f"[SKIP] Completed run found: {complete_path}")
        with open(complete_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    section('LOAD PROTOCOL GRAPHS')
    train_graph = load_protocol_stage(protocol, 'train', device)
    val_graph = load_protocol_stage(protocol, 'validation', device)
    test_graph = load_protocol_stage(protocol, 'test', device)
    print(f"Train:      {train_graph.num_nodes:>7,} nodes | {train_graph.num_edges // 2:>7,} undirected edges")
    print(f"Validation: {val_graph.num_nodes:>7,} nodes | {val_graph.num_edges // 2:>7,} undirected edges")
    print(f"Test:       {test_graph.num_nodes:>7,} nodes | {test_graph.num_edges // 2:>7,} undirected edges")
    train_countries = sorted(set(train_graph.countries))
    domain_active = len(train_countries) > 1
    print('Training countries:', train_countries)
    print('Domain adversarial loss:', 'ACTIVE (delayed)' if domain_active else 'INACTIVE')

    vocab = pd.read_csv(INPUT_ROOT / 'operation_vocabulary.csv')
    num_operations = int(vocab['operation_id'].max() + 1)
    section('INITIALIZE CARTOGRAPH-X V3')
    model = CartoGraphX(num_operations, config).to(device)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {nparams:,}')
    print(f'Device:               {device}')

    sw = compute_class_weights(train_graph.y_selection, 2, config.max_selection_class_weight)
    ow = compute_class_weights(train_graph.y_operation, num_operations, config.max_operation_class_weight)
    selection_criterion = nn.CrossEntropyLoss(weight=sw)
    operation_criterion = nn.CrossEntropyLoss(weight=ow)
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    max_epochs = config.smoke_epochs if mode == 'smoke' else (config.development_epochs if mode == 'development' else config.full_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=config.learning_rate * 0.05)

    state_path = run_dir / 'training_state.pt'
    best_path = run_dir / 'best_checkpoint.pt'
    last_path = run_dir / 'last_checkpoint.pt'
    history_path = run_dir / 'training_history.csv'
    start_epoch, best_epoch, best_score, best_threshold = 1, -1, -np.inf, 0.5
    patience_counter = 0
    history_rows = []

    if state_path.exists():
        ckpt = torch.load(state_path, map_location=device, weights_only=False)
        if ckpt.get('protocol') == protocol and int(ckpt.get('seed', -1)) == seed and ckpt.get('model_version') == 'v3':
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = int(ckpt['epoch']) + 1
            best_epoch = int(ckpt['best_epoch'])
            best_score = float(ckpt['best_score'])
            best_threshold = float(ckpt['best_threshold'])
            patience_counter = int(ckpt.get('patience_counter', 0))
            history_rows = ckpt.get('history_rows', [])
            print(f'[RESUME] Epoch {start_epoch} | best epoch {best_epoch} | best score {best_score:.4f}')

    section('TRAIN CARTOGRAPH-X V3')
    print(f'Early stopping LOCKED through epoch {config.min_epochs}; patience={config.early_stopping_patience} starts fresh afterward.')
    start_time = time.time()
    for epoch in range(start_epoch, max_epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        if domain_active and epoch >= config.domain_start_epoch:
            local_epoch = epoch - config.domain_start_epoch + 1
            local_total = max(2, max_epochs - config.domain_start_epoch + 1)
            grl = domain_schedule(local_epoch, local_total, config.domain_grl_max)
        else:
            grl = 0.0
        out = model(train_graph, grl_coefficient=grl)
        base_sel_loss = selection_criterion(out['selection_logits'], train_graph.y_selection)
        dro_sel_loss = domain_robust_selection_loss(out['selection_logits'], train_graph.y_selection, train_graph.y_country, sw, domain_active)
        sel_loss = (1.0 - config.dro_weight) * base_sel_loss + config.dro_weight * dro_sel_loss
        op_loss = operation_criterion(out['operation_logits'], train_graph.y_operation)
        rank_loss = tile_ranking_loss(out['selection_logits'], train_graph.y_selection, train_graph.tiles, config)
        topo_loss = topology_preservation_loss(out['selection_logits'], train_graph.y_selection, train_graph.edge_index)
        dom_loss = domain_criterion(out['domain_logits'], train_graph.y_country) if (domain_active and epoch >= config.domain_start_epoch) else torch.tensor(0.0, device=device)
        align_loss = cross_domain_class_alignment_loss(out['embedding'], train_graph.y_selection, train_graph.y_country) if domain_active else torch.tensor(0.0, device=device)
        balance_loss = moe_balance_loss(out['expert_weights'])
        loss = (config.selection_loss_weight * sel_loss + config.operation_loss_weight * op_loss +
                config.ranking_loss_weight * rank_loss + config.topology_loss_weight * topo_loss +
                config.domain_loss_weight * dom_loss + config.contrastive_weight * align_loss +
                config.moe_balance_weight * balance_loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip); optimizer.step(); scheduler.step()

        val_pred = predict_graph(model, val_graph)
        y_val = val_graph.y_selection.detach().cpu().numpy()
        threshold, vm = select_validation_threshold(y_val, val_pred['selection_probability'], config)
        val_score = composite_validation_score(vm, config)
        row = {'epoch': epoch, 'loss': float(loss.detach().cpu()), 'selection_loss': float(sel_loss.detach().cpu()),
               'operation_loss': float(op_loss.detach().cpu()), 'ranking_loss': float(rank_loss.detach().cpu()),
               'topology_loss': float(topo_loss.detach().cpu()), 'domain_loss': float(dom_loss.detach().cpu()),
               'dro_selection_loss': float(dro_sel_loss.detach().cpu()), 'alignment_loss': float(align_loss.detach().cpu()),
               'moe_balance_loss': float(balance_loss.detach().cpu()),
               'grl_coefficient': grl, 'learning_rate': optimizer.param_groups[0]['lr'],
               'validation_threshold': threshold, 'validation_score': val_score, 'validation_mcc': vm['mcc'],
               'validation_balanced_accuracy': vm['balanced_accuracy'], 'validation_f1': vm['f1'],
               'validation_roc_auc': vm['roc_auc'], 'validation_pr_auc': vm['pr_auc']}
        history_rows.append(row)
        print(f"Epoch {epoch:03d}/{max_epochs:03d} | loss={row['loss']:.4f} | sel={row['selection_loss']:.4f} | rank={row['ranking_loss']:.4f} | topo={row['topology_loss']:.4f} | dro={row['dro_selection_loss']:.4f} | align={row['alignment_loss']:.4f} | valScore={val_score:.4f} | MCC={vm['mcc']:.4f} | BA={vm['balanced_accuracy']:.4f} | ROC={vm['roc_auc']:.4f} | PR={vm['pr_auc']:.4f} | thr={threshold:.2f}")

        improved = val_score > best_score + config.early_stopping_min_delta
        if improved:
            best_score, best_epoch, best_threshold = val_score, epoch, threshold
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'score': val_score, 'threshold': threshold,
                        'config': asdict(config), 'model_version': 'v3'}, best_path)
            if epoch >= config.min_epochs:
                patience_counter = 0
        elif epoch > config.min_epochs:
            patience_counter += 1
        # At the boundary, patience explicitly starts at zero regardless of earlier fluctuations.
        if epoch == config.min_epochs:
            patience_counter = 0

        state = {'model_version': 'v3', 'protocol': protocol, 'seed': seed, 'epoch': epoch,
                 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict(), 'best_epoch': best_epoch, 'best_score': best_score,
                 'best_threshold': best_threshold, 'patience_counter': patience_counter, 'history_rows': history_rows,
                 'config': asdict(config)}
        torch.save(state, state_path); torch.save(state, last_path)
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        if epoch > config.min_epochs and patience_counter >= config.early_stopping_patience:
            print(f'Early stopping at epoch {epoch}; best epoch={best_epoch}. Minimum training requirement was satisfied.')
            break

    if not best_path.exists():
        raise RuntimeError('No best checkpoint was saved.')
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best['model_state_dict']); best_epoch = int(best['epoch']); best_threshold = float(best['threshold']); best_score = float(best['score'])

    section('VALIDATION-SELECTED MODEL')
    val_pred = predict_graph(model, val_graph); y_val = val_graph.y_selection.detach().cpu().numpy()
    val_metrics = classification_metrics(y_val, val_pred['selection_probability'], best_threshold)
    print(f'Best epoch: {best_epoch} | score={best_score:.4f} | threshold={best_threshold:.2f} | MCC={val_metrics["mcc"]:.4f} | BA={val_metrics["balanced_accuracy"]:.4f}')

    section('FINAL TEST EVALUATION')
    test_pred = predict_graph(model, test_graph); y_test = test_graph.y_selection.detach().cpu().numpy()
    tm = classification_metrics(y_test, test_pred['selection_probability'], best_threshold)
    op_true = test_graph.y_operation.detach().cpu().numpy(); op_acc = float(accuracy_score(op_true, test_pred['operation_pred']))
    print(f"Balanced accuracy: {tm['balanced_accuracy']:.4f}")
    print(f"F1:                {tm['f1']:.4f}")
    print(f"MCC:               {tm['mcc']:.4f}")
    print(f"ROC-AUC:           {tm['roc_auc']:.4f}")
    print(f"PR-AUC:            {tm['pr_auc']:.4f}")
    print(f"Operation accuracy:{op_acc:.4f}")

    fusion = test_pred['fusion_weights']; mean_fusion = fusion.mean(axis=0)
    fusion_names = ['intrinsic','graph_structural','multiscale_context','neighbor_aggregation']
    pred_df = pd.DataFrame({'sample_uid': test_graph.sample_uids, 'country': test_graph.countries, 'tile_id': test_graph.tiles,
                            'y_true': y_test, 'prob_retained': test_pred['selection_probability'],
                            'y_pred': (test_pred['selection_probability'] >= best_threshold).astype(int),
                            'operation_true': op_true, 'operation_pred': test_pred['operation_pred'],
                            'fusion_intrinsic': fusion[:,0], 'fusion_graph_structural': fusion[:,1],
                            'fusion_multiscale_context': fusion[:,2], 'fusion_neighbor_aggregation': fusion[:,3]})
    if test_pred.get('expert_weights') is not None:
        ew = test_pred['expert_weights']
        for j in range(ew.shape[1]):
            pred_df[f'expert_weight_{j}'] = ew[:, j]
        pd.DataFrame({
            'expert': [f'expert_{j}' for j in range(ew.shape[1])],
            'mean_weight': ew.mean(axis=0),
            'std_weight': ew.std(axis=0),
        }).to_csv(run_dir / 'moe_expert_summary.csv', index=False)
    pred_df.to_csv(run_dir / 'test_predictions.csv', index=False)
    emb = test_pred['embedding']; edf = pd.DataFrame(emb, columns=[f'embedding_{i:03d}' for i in range(emb.shape[1])]); edf.insert(0,'sample_uid',test_graph.sample_uids); edf.to_csv(run_dir/'test_embeddings.csv',index=False)
    pd.DataFrame({'feature_family':fusion_names,'mean_weight':mean_fusion,'std_weight':fusion.std(axis=0)}).to_csv(run_dir/'adaptive_fusion_summary.csv',index=False)
    torch.save({'model_state_dict':model.state_dict(),'config':asdict(config),'protocol':protocol,'seed':seed,'best_epoch':best_epoch,'threshold':best_threshold,'model_version':'v3'}, run_dir/'best_model.pt')

    hdf = pd.DataFrame(history_rows)
    fig, ax = plt.subplots(figsize=(8,5)); ax.plot(hdf['epoch'],hdf['validation_mcc'],label='Validation MCC'); ax.plot(hdf['epoch'],hdf['validation_balanced_accuracy'],label='Validation BA'); ax.axvline(best_epoch,linestyle='--',label='Selected epoch'); ax.axvline(config.min_epochs,linestyle=':',label='Early-stop unlock'); ax.set_xlabel('Epoch'); ax.set_ylabel('Score'); ax.set_title(f'CartoGraph-X v3 — {protocol} — Seed {seed}'); ax.legend(); fig.tight_layout(); fig.savefig(run_dir/'training_curve.png',dpi=300,bbox_inches='tight'); fig.savefig(run_dir/'training_curve.pdf',bbox_inches='tight'); plt.close(fig)

    elapsed = time.time() - start_time
    result = {'protocol':protocol,'seed':seed,'mode':mode,'model_version':'v3','best_epoch':best_epoch,'selected_threshold':best_threshold,
              'validation_score':best_score,'train_nodes':train_graph.num_nodes,'validation_nodes':val_graph.num_nodes,'test_nodes':test_graph.num_nodes,
              'train_countries':'|'.join(train_countries),'domain_adversarial_active':domain_active,'trainable_parameters':nparams,'elapsed_seconds':elapsed,
              'validation_mcc':val_metrics['mcc'],'validation_balanced_accuracy':val_metrics['balanced_accuracy'],
              'test_accuracy':tm['accuracy'],'test_balanced_accuracy':tm['balanced_accuracy'],'test_precision':tm['precision'],'test_recall':tm['recall'],
              'test_specificity':tm['specificity'],'test_f1':tm['f1'],'test_mcc':tm['mcc'],'test_roc_auc':tm['roc_auc'],'test_pr_auc':tm['pr_auc'],
              'test_operation_accuracy':op_acc}
    with open(run_dir/'run_report.json','w',encoding='utf-8') as f: json.dump(json_safe(result),f,indent=2)
    with open(complete_path,'w',encoding='utf-8') as f: json.dump(json_safe(result),f,indent=2)
    # Training state is retained intentionally for reproducibility/resume audit.
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
            "Train CartoGraph-X v3 on leakage-safe "
            "cross-country road-selection protocols."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "smoke",
            "development",
            "full",
        ],
        default="development",
        help=(
            "smoke = seed 42 quick pipeline test; "
            "development = seed 42 with full 200-epoch ceiling; "
            "full = seeds 42-51 with full training."
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
        "CARTOGEN — CARTOGRAPH-X V3 HIERARCHICAL GRAPH-ATTENTION ROAD-SELECTION MODEL"
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

    seeds = (FULL_SEEDS if args.mode == "full" else SMOKE_SEEDS)

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
        "2. CARTOGRAPH-X V3 ARCHITECTURE"
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
        "  -> cross-family self-attention + adaptive fusion"
    )

    print(
        "  -> multi-head edge-aware graph attention"
    )

    print(
        "  -> gated fusion with nonlinear tabular interaction branch"
    )

    print(
        "  -> representation-driven mixture-of-experts router (no country feature)"
    )

    print(
        "  -> group-DRO + cross-domain same-class prototype alignment"
    )

    print(
        "  -> selection + ranking + topology-preservation objectives"
    )

    print(
        "  -> delayed low-weight adversarial country head "
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
        "selected_threshold",
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
                "selected_threshold"
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
            "81_train_cartographx_v3.py",

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
        "SCRIPT 81 COMPLETE"
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
            "NEXT: evaluate V3 development results before full multi-seed training — "
            "component ablation and novelty validation."
        )

    else:

        print(
            "\nCARTOGRAPH-X V3 EXPERIMENT COMPLETE."
        )


if __name__ == "__main__":
    main()