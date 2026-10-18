"""
Script 75
=========
Quantitative Spatial and Cartographic Quality Evaluation for CartoGraph-X.

Purpose
-------
Evaluate whether CartoGraph-X preserves the spatial, structural, and
cartographic properties of the generalized road network, beyond ordinary
road-wise classification accuracy.

The script evaluates:

A. Selection fidelity
   - Retention rate
   - Retention-rate error
   - Precision
   - Recall
   - F1
   - Balanced accuracy
   - MCC

B. Network structural preservation
   - Number of connected components
   - Largest connected component ratio
   - Isolated-road ratio
   - Mean graph degree
   - Degree-distribution Jensen-Shannon divergence
   - Connectivity preservation score

C. Spatial/cartographic preservation
   - Total retained road length
   - Total-length ratio
   - Total-length relative error
   - Mean road length
   - Mean-length relative error
   - Orientation-distribution divergence
   - Spatial occupancy / coverage similarity
   - Local road-density distribution divergence

D. Cross-country generalization
   - Within-country
   - Zero-shot
   - Joint-country

E. Multi-run evaluation
   --mode smoke:
       one seed, default seed 42

   --mode final:
       all available seeds are evaluated separately
       and summarized as mean ± standard deviation

Important methodological choice
-------------------------------
The primary reference network is the subset of source roads for which
y_true == 1.

This is deliberate.

CartoGraph-X predicts road retention/elimination. It does not directly
predict geometric simplification, displacement, or newly drawn target
geometry. Therefore, structural comparison should compare:

    reference retained source roads
                versus
    predicted retained source roads

This keeps the evaluation aligned with the actual learning task.

The independently generalized target geometry generated/loaded in Script 74
is useful for qualitative cartographic visualization, but is not used as
the primary structural reference here.

Author:
    CartoGen / CartoGraph-X pipeline
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

import networkx as nx


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"

CROSS = PROCESSED / "cross_country"

MODEL_DIR = CROSS / "cartographx_models"

MAP_DIR = CROSS / "cartographx_maps"

GPKG_DIR = MAP_DIR / "geopackages"

OUT_DIR = CROSS / "cartographic_quality"

TABLE_DIR = OUT_DIR / "tables"

FIG_DIR = OUT_DIR / "figures"

RUN_DIR = OUT_DIR / "per_run"


# =============================================================================
# EXPERIMENT DEFINITIONS
# =============================================================================

COUNTRIES = [
    "Lithuania",
    "Poland",
]

CITIES = {
    "Lithuania": [
        "Kaunas",
        "Klaipeda",
        "Vilnius",
    ],
    "Poland": [
        "Gdansk",
        "Krakow",
        "Warsaw",
    ],
}

CITY_KEYS = {
    "Kaunas": "kaunas",
    "Klaipeda": "klaipeda",
    "Vilnius": "vilnius",
    "Gdansk": "gdansk",
    "Krakow": "krakow",
    "Warsaw": "warsaw",
}

PROTOCOL_COUNTRY = {
    "lt_to_lt": "Lithuania",
    "pl_to_pl": "Poland",
    "lt_to_pl_zero_shot": "Poland",
    "pl_to_lt_zero_shot": "Lithuania",
    "joint_to_lt": "Lithuania",
    "joint_to_pl": "Poland",
}

PROTOCOL_REGIME = {
    "lt_to_lt": "within_country",
    "pl_to_pl": "within_country",
    "lt_to_pl_zero_shot": "zero_shot",
    "pl_to_lt_zero_shot": "zero_shot",
    "joint_to_lt": "joint_country",
    "joint_to_pl": "joint_country",
}

PROTOCOL_DISPLAY = {
    "lt_to_lt": "LT → LT",
    "pl_to_pl": "PL → PL",
    "lt_to_pl_zero_shot": "LT → PL",
    "pl_to_lt_zero_shot": "PL → LT",
    "joint_to_lt": "Joint → LT",
    "joint_to_pl": "Joint → PL",
}

PROTOCOLS = list(PROTOCOL_COUNTRY.keys())


# =============================================================================
# CONFIGURATION
# =============================================================================

# Graph construction for evaluation.
#
# These values intentionally match the graph construction philosophy
# used earlier in the pipeline:
#
#   k nearest neighbours
#   maximum connection distance
#
# Graphs are reconstructed independently for reference and prediction.
#
GRAPH_K = 6
GRAPH_MAX_DISTANCE_M = 250.0


# Spatial occupancy grid.
#
# This is not a rasterization of the actual road line geometry.
# Instead, each road contributes to cells touched by its geometry.
#
# 100 m gives useful local-scale spatial coverage statistics while
# remaining computationally light.
#
GRID_SIZE_M = 100.0


# Local density calculation.
#
# Count retained road centroids within this radius.
#
DENSITY_RADIUS_M = 250.0


# Orientation histogram.
#
# Road orientation is axial:
# 0 degrees and 180 degrees represent the same direction.
#
ORIENTATION_BINS = np.arange(
    0.0,
    180.0 + 15.0,
    15.0,
)


# Degree histogram.
#
DEGREE_BINS = np.arange(
    -0.5,
    20.5,
    1.0,
)


DPI = 350


# =============================================================================
# UTILITIES
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


def ensure_dirs() -> None:

    for p in [
        OUT_DIR,
        TABLE_DIR,
        FIG_DIR,
        RUN_DIR,
    ]:
        p.mkdir(
            parents=True,
            exist_ok=True,
        )


def clean_name(x) -> str:

    return (
        str(x)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def normalize_city(x: str) -> str:

    s = clean_name(x)

    mapping = {
        "kaunas": "Kaunas",
        "klaipeda": "Klaipeda",
        "vilnius": "Vilnius",
        "gdansk": "Gdansk",
        "gdańsk": "Gdansk",
        "krakow": "Krakow",
        "kraków": "Krakow",
        "warsaw": "Warsaw",
        "warszawa": "Warsaw",
    }

    return mapping.get(
        s,
        str(x),
    )


def json_safe(obj):

    if isinstance(obj, dict):
        return {
            str(k): json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [
            json_safe(v)
            for v in obj
        ]

    if isinstance(obj, tuple):
        return [
            json_safe(v)
            for v in obj
        ]

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):

        if np.isnan(obj):
            return None

        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


def safe_div(
    numerator: float,
    denominator: float,
) -> float:

    if denominator == 0:
        return np.nan

    return float(
        numerator / denominator
    )


def relative_error(
    predicted: float,
    reference: float,
) -> float:

    if reference == 0:
        return np.nan

    return float(
        abs(predicted - reference)
        / abs(reference)
    )


def save_figure(
    fig: plt.Figure,
    stem: str,
) -> None:

    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"

    fig.savefig(
        png,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"PNG: {png}")
    print(f"PDF: {pdf}")


# =============================================================================
# SAMPLE UID
# =============================================================================

def parse_sample_uid(
    uid: str,
) -> Tuple[str, str, int]:

    parts = str(uid).split("::")

    if len(parts) < 3:
        raise ValueError(
            f"Unexpected sample_uid: {uid}"
        )

    country = parts[0]
    city = normalize_city(
        parts[1]
    )

    try:
        source_id = int(
            parts[-1]
        )

    except Exception as e:

        raise ValueError(
            f"Could not recover source ID "
            f"from {uid}"
        ) from e

    return (
        country,
        city,
        source_id,
    )


def add_uid_fields(
    d: pd.DataFrame,
) -> pd.DataFrame:

    out = d.copy()

    parsed = out[
        "sample_uid"
    ].map(
        parse_sample_uid
    )

    out["uid_country"] = [
        x[0]
        for x in parsed
    ]

    out["uid_city"] = [
        x[1]
        for x in parsed
    ]

    out["source_id"] = [
        x[2]
        for x in parsed
    ]

    return out


# =============================================================================
# PREDICTION FILES
# =============================================================================

def prediction_files(
    protocol: str,
) -> List[Path]:

    p = MODEL_DIR / protocol

    if not p.exists():
        return []

    return sorted(
        p.glob(
            "seed_*/test_predictions.csv"
        )
    )


def seed_from_path(
    path: Path,
) -> int:

    match = re.search(
        r"seed_(\d+)",
        str(path),
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            f"Cannot recover seed from "
            f"{path}"
        )

    return int(
        match.group(1)
    )


def load_prediction_file(
    path: Path,
) -> pd.DataFrame:

    d = pd.read_csv(path)

    required = [
        "sample_uid",
        "country",
        "tile_id",
        "y_true",
        "prob_retained",
        "y_pred",
    ]

    missing = [
        c
        for c in required
        if c not in d.columns
    ]

    if missing:
        raise KeyError(
            f"{path}\n"
            f"Missing columns: {missing}"
        )

    d = add_uid_fields(d)

    d["seed"] = seed_from_path(
        path
    )

    for c in [
        "y_true",
        "y_pred",
        "prob_retained",
    ]:

        d[c] = pd.to_numeric(
            d[c],
            errors="coerce",
        )

    if d[
        [
            "y_true",
            "y_pred",
        ]
    ].isna().any().any():

        raise RuntimeError(
            f"Invalid labels in {path}"
        )

    d["y_true"] = (
        d["y_true"].astype(int)
    )

    d["y_pred"] = (
        d["y_pred"].astype(int)
    )

    return d


# =============================================================================
# GEOMETRY LOADING
# =============================================================================

def lithuania_source_path(
    city: str,
) -> Path:

    key = CITY_KEYS[city]

    return (
        PROCESSED
        / "lithuania"
        / "road_correspondence"
        / f"{key}_central_road_correspondence.gpkg"
    )


def poland_source_path(
    city: str,
) -> Path:

    key = CITY_KEYS[city]

    return (
        PROCESSED
        / "poland"
        / "learning_dataset"
        / f"{key}_road_learning_dataset.gpkg"
    )


def load_source_geometry(
    country: str,
    city: str,
) -> gpd.GeoDataFrame:

    if country == "Lithuania":

        p = lithuania_source_path(
            city
        )

        if not p.exists():
            raise FileNotFoundError(p)

        g = gpd.read_file(
            p,
            layer="source_roads",
        )

        if (
            "cartogen_source_id"
            not in g.columns
        ):
            raise KeyError(
                f"cartogen_source_id "
                f"missing from {p}"
            )

        g["source_id"] = (
            pd.to_numeric(
                g[
                    "cartogen_source_id"
                ],
                errors="coerce",
            )
        )

    elif country == "Poland":

        p = poland_source_path(
            city
        )

        if not p.exists():
            raise FileNotFoundError(p)

        g = gpd.read_file(
            p,
            layer="road_learning_samples",
        )

        if "source_idx" not in g.columns:
            raise KeyError(
                f"source_idx missing "
                f"from {p}"
            )

        g["source_id"] = (
            pd.to_numeric(
                g["source_idx"],
                errors="coerce",
            )
        )

    else:

        raise ValueError(
            country
        )

    g = g[
        g["source_id"].notna()
    ].copy()

    g["source_id"] = (
        g["source_id"].astype(int)
    )

    if g["source_id"].duplicated().any():

        duplicates = int(
            g["source_id"]
            .duplicated()
            .sum()
        )

        raise RuntimeError(
            f"{country}/{city}: "
            f"{duplicates} duplicated "
            f"source IDs."
        )

    g["country_eval"] = country
    g["city_eval"] = city

    return g


# =============================================================================
# JOIN PREDICTION TO GEOMETRY
# =============================================================================

def map_prediction_geometry(
    source: gpd.GeoDataFrame,
    prediction: pd.DataFrame,
    city: str,
) -> gpd.GeoDataFrame:

    p = prediction[
        prediction["uid_city"]
        == city
    ].copy()

    if p.empty:

        return gpd.GeoDataFrame(
            columns=[
                "sample_uid",
                "source_id",
                "tile_id",
                "y_true",
                "y_pred",
                "prob_retained",
                "geometry",
            ],
            geometry="geometry",
            crs=source.crs,
        )

    if p[
        "source_id"
    ].duplicated().any():

        raise RuntimeError(
            f"{city}: prediction "
            f"source_id duplicated."
        )

    keep = [
        "sample_uid",
        "source_id",
        "tile_id",
        "y_true",
        "y_pred",
        "prob_retained",
        "seed",
    ]

    merged = source.merge(
        p[keep],
        on="source_id",
        how="inner",
        validate="one_to_one",
    )

    return gpd.GeoDataFrame(
        merged,
        geometry="geometry",
        crs=source.crs,
    )


# =============================================================================
# GEOMETRY CHARACTERISTICS
# =============================================================================

def geometry_length(
    g: gpd.GeoDataFrame,
) -> np.ndarray:

    if g.empty:
        return np.array(
            [],
            dtype=float,
        )

    return (
        g.geometry.length
        .to_numpy(dtype=float)
    )


def line_orientation(
    geom,
) -> float:

    """
    Axial orientation in [0, 180).

    For MultiLineString, the longest component
    determines orientation.
    """

    if geom is None:
        return np.nan

    if geom.is_empty:
        return np.nan

    try:

        if geom.geom_type == "MultiLineString":

            lines = list(
                geom.geoms
            )

            if not lines:
                return np.nan

            geom = max(
                lines,
                key=lambda x: x.length,
            )

        coords = list(
            geom.coords
        )

        if len(coords) < 2:
            return np.nan

        x1, y1 = coords[0]
        x2, y2 = coords[-1]

        dx = x2 - x1
        dy = y2 - y1

        if (
            abs(dx) < 1e-12
            and abs(dy) < 1e-12
        ):
            return np.nan

        angle = math.degrees(
            math.atan2(
                dy,
                dx,
            )
        )

        angle = (
            angle % 180.0
        )

        return float(angle)

    except Exception:

        return np.nan


def orientation_distribution(
    g: gpd.GeoDataFrame,
) -> np.ndarray:

    if g.empty:

        return np.zeros(
            len(
                ORIENTATION_BINS
            ) - 1,
            dtype=float,
        )

    values = np.array(
        [
            line_orientation(x)
            for x in g.geometry
        ],
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return np.zeros(
            len(
                ORIENTATION_BINS
            ) - 1,
            dtype=float,
        )

    hist, _ = np.histogram(
        values,
        bins=ORIENTATION_BINS,
    )

    hist = hist.astype(float)

    if hist.sum() > 0:
        hist /= hist.sum()

    return hist


# =============================================================================
# DISTRIBUTION DIVERGENCE
# =============================================================================

def js_divergence(
    p: np.ndarray,
    q: np.ndarray,
) -> float:

    p = np.asarray(
        p,
        dtype=float,
    )

    q = np.asarray(
        q,
        dtype=float,
    )

    if (
        p.sum() <= 0
        and q.sum() <= 0
    ):
        return 0.0

    if p.sum() <= 0:
        return 1.0

    if q.sum() <= 0:
        return 1.0

    p = p / p.sum()
    q = q / q.sum()

    # scipy returns Jensen-Shannon distance.
    # Squaring gives Jensen-Shannon divergence.
    distance = jensenshannon(
        p,
        q,
        base=2,
    )

    if not np.isfinite(distance):
        return np.nan

    return float(
        distance ** 2
    )


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def graph_from_roads(
    g: gpd.GeoDataFrame,
    k: int = GRAPH_K,
    max_distance: float = GRAPH_MAX_DISTANCE_M,
) -> nx.Graph:

    """
    Construct a proximity graph from retained road centroids.

    Nodes = retained source road objects.

    Edge = one road is among the k nearest retained roads and
           centroid distance <= max_distance.

    This graph is only used for evaluation.

    The same deterministic rule is applied to reference and
    predicted networks.
    """

    graph = nx.Graph()

    n = len(g)

    if n == 0:
        return graph

    centroids = g.geometry.centroid

    coords = np.column_stack(
        [
            centroids.x.to_numpy(),
            centroids.y.to_numpy(),
        ]
    )

    for i in range(n):
        graph.add_node(i)

    if n == 1:
        return graph

    # Chunked distance computation avoids storing a full
    # N x N matrix.
    chunk_size = 1000

    for start in range(
        0,
        n,
        chunk_size,
    ):

        stop = min(
            start + chunk_size,
            n,
        )

        block = coords[
            start:stop
        ]

        dx = (
            block[:, None, 0]
            - coords[None, :, 0]
        )

        dy = (
            block[:, None, 1]
            - coords[None, :, 1]
        )

        dist2 = (
            dx * dx
            + dy * dy
        )

        for local_i in range(
            stop - start
        ):

            i = start + local_i

            dist2[
                local_i,
                i,
            ] = np.inf

            candidate_count = min(
                k,
                n - 1,
            )

            if candidate_count <= 0:
                continue

            idx = np.argpartition(
                dist2[local_i],
                candidate_count - 1,
            )[
                :candidate_count
            ]

            for j in idx:

                distance = math.sqrt(
                    float(
                        dist2[
                            local_i,
                            j,
                        ]
                    )
                )

                if distance <= max_distance:

                    graph.add_edge(
                        i,
                        int(j),
                        distance_m=distance,
                    )

    return graph


# =============================================================================
# GRAPH STATISTICS
# =============================================================================

def graph_statistics(
    graph: nx.Graph,
) -> Dict[str, float]:

    n = graph.number_of_nodes()
    e = graph.number_of_edges()

    if n == 0:

        return {
            "nodes": 0,
            "edges": 0,
            "components": 0,
            "largest_component_ratio": np.nan,
            "isolated_ratio": np.nan,
            "mean_degree": np.nan,
        }

    components = list(
        nx.connected_components(
            graph
        )
    )

    largest = (
        max(
            len(c)
            for c in components
        )
        if components
        else 0
    )

    isolated = sum(
        1
        for _, degree
        in graph.degree()
        if degree == 0
    )

    degrees = np.array(
        [
            degree
            for _, degree
            in graph.degree()
        ],
        dtype=float,
    )

    return {
        "nodes": int(n),
        "edges": int(e),
        "components": int(
            len(components)
        ),
        "largest_component_ratio": (
            float(largest / n)
        ),
        "isolated_ratio": (
            float(isolated / n)
        ),
        "mean_degree": float(
            degrees.mean()
        ),
    }


def degree_distribution(
    graph: nx.Graph,
) -> np.ndarray:

    if (
        graph.number_of_nodes()
        == 0
    ):

        return np.zeros(
            len(DEGREE_BINS) - 1,
            dtype=float,
        )

    degrees = np.array(
        [
            d
            for _, d
            in graph.degree()
        ],
        dtype=float,
    )

    degrees = np.clip(
        degrees,
        0,
        DEGREE_BINS[-1] - 0.5,
    )

    hist, _ = np.histogram(
        degrees,
        bins=DEGREE_BINS,
    )

    hist = hist.astype(float)

    if hist.sum() > 0:
        hist /= hist.sum()

    return hist


# =============================================================================
# SPATIAL OCCUPANCY
# =============================================================================

def occupied_cells(
    g: gpd.GeoDataFrame,
    grid_size: float = GRID_SIZE_M,
) -> set:

    """
    Return grid cells touched by road geometry.

    Geometry is sampled at intervals no larger than half
    the grid size so long roads contribute to multiple cells.
    """

    cells = set()

    if g.empty:
        return cells

    step = grid_size / 2.0

    for geom in g.geometry:

        if geom is None:
            continue

        if geom.is_empty:
            continue

        parts = []

        if geom.geom_type == "LineString":

            parts = [geom]

        elif geom.geom_type == "MultiLineString":

            parts = list(
                geom.geoms
            )

        else:

            continue

        for line in parts:

            length = float(
                line.length
            )

            if length <= 0:

                point = line.centroid

                cell = (
                    int(
                        math.floor(
                            point.x
                            / grid_size
                        )
                    ),
                    int(
                        math.floor(
                            point.y
                            / grid_size
                        )
                    ),
                )

                cells.add(cell)
                continue

            n_samples = max(
                2,
                int(
                    math.ceil(
                        length / step
                    )
                ) + 1,
            )

            distances = np.linspace(
                0.0,
                length,
                n_samples,
            )

            for distance in distances:

                point = line.interpolate(
                    float(distance)
                )

                cell = (
                    int(
                        math.floor(
                            point.x
                            / grid_size
                        )
                    ),
                    int(
                        math.floor(
                            point.y
                            / grid_size
                        )
                    ),
                )

                cells.add(cell)

    return cells


def occupancy_metrics(
    reference: gpd.GeoDataFrame,
    predicted: gpd.GeoDataFrame,
) -> Dict[str, float]:

    ref = occupied_cells(
        reference
    )

    pred = occupied_cells(
        predicted
    )

    if (
        len(ref) == 0
        and len(pred) == 0
    ):

        return {
            "reference_cells": 0,
            "predicted_cells": 0,
            "coverage_jaccard": 1.0,
            "coverage_recall": 1.0,
            "coverage_precision": 1.0,
        }

    intersection = len(
        ref & pred
    )

    union = len(
        ref | pred
    )

    return {
        "reference_cells": len(ref),
        "predicted_cells": len(pred),
        "coverage_jaccard": safe_div(
            intersection,
            union,
        ),
        "coverage_recall": safe_div(
            intersection,
            len(ref),
        ),
        "coverage_precision": safe_div(
            intersection,
            len(pred),
        ),
    }


# =============================================================================
# LOCAL DENSITY
# =============================================================================

def centroid_neighbor_counts(
    g: gpd.GeoDataFrame,
    radius: float = DENSITY_RADIUS_M,
) -> np.ndarray:

    n = len(g)

    if n == 0:

        return np.array(
            [],
            dtype=float,
        )

    centroids = g.geometry.centroid

    coords = np.column_stack(
        [
            centroids.x.to_numpy(),
            centroids.y.to_numpy(),
        ]
    )

    counts = np.zeros(
        n,
        dtype=float,
    )

    r2 = radius * radius

    chunk_size = 1000

    for start in range(
        0,
        n,
        chunk_size,
    ):

        stop = min(
            start + chunk_size,
            n,
        )

        block = coords[
            start:stop
        ]

        dx = (
            block[:, None, 0]
            - coords[None, :, 0]
        )

        dy = (
            block[:, None, 1]
            - coords[None, :, 1]
        )

        dist2 = (
            dx * dx
            + dy * dy
        )

        inside = (
            dist2 <= r2
        )

        block_counts = (
            inside.sum(axis=1)
            - 1
        )

        counts[
            start:stop
        ] = block_counts

    return counts


def density_distribution(
    g: gpd.GeoDataFrame,
) -> np.ndarray:

    counts = (
        centroid_neighbor_counts(
            g
        )
    )

    if len(counts) == 0:

        return np.zeros(
            21,
            dtype=float,
        )

    # Adaptive bins based on local count.
    #
    # 0-19 individually, 20+ pooled.
    hist = np.zeros(
        21,
        dtype=float,
    )

    for x in counts:

        idx = min(
            int(x),
            20,
        )

        hist[idx] += 1

    if hist.sum() > 0:
        hist /= hist.sum()

    return hist


# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:

    return {
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
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
            )
        ),
    }


# =============================================================================
# NETWORK EVALUATION
# =============================================================================

def evaluate_network(
    mapped: gpd.GeoDataFrame,
    country: str,
    city: str,
    protocol: str,
    seed: int,
) -> Dict:

    if mapped.empty:

        raise RuntimeError(
            f"No mapped predictions for "
            f"{country}/{city}/{protocol}/"
            f"seed {seed}"
        )

    y_true = (
        mapped["y_true"]
        .to_numpy(dtype=int)
    )

    y_pred = (
        mapped["y_pred"]
        .to_numpy(dtype=int)
    )

    reference = mapped[
        mapped["y_true"] == 1
    ].copy()

    predicted = mapped[
        mapped["y_pred"] == 1
    ].copy()

    # -------------------------------------------------------------------------
    # Selection fidelity
    # -------------------------------------------------------------------------

    cls = classification_metrics(
        y_true,
        y_pred,
    )

    reference_retention_rate = float(
        y_true.mean()
    )

    predicted_retention_rate = float(
        y_pred.mean()
    )

    retention_rate_error = abs(
        predicted_retention_rate
        - reference_retention_rate
    )

    # -------------------------------------------------------------------------
    # Length statistics
    # -------------------------------------------------------------------------

    ref_lengths = geometry_length(
        reference
    )

    pred_lengths = geometry_length(
        predicted
    )

    ref_total_length = float(
        ref_lengths.sum()
    )

    pred_total_length = float(
        pred_lengths.sum()
    )

    ref_mean_length = (
        float(ref_lengths.mean())
        if len(ref_lengths)
        else np.nan
    )

    pred_mean_length = (
        float(pred_lengths.mean())
        if len(pred_lengths)
        else np.nan
    )

    length_ratio = safe_div(
        pred_total_length,
        ref_total_length,
    )

    total_length_relative_error = (
        relative_error(
            pred_total_length,
            ref_total_length,
        )
    )

    mean_length_relative_error = (
        relative_error(
            pred_mean_length,
            ref_mean_length,
        )
    )

    # -------------------------------------------------------------------------
    # Orientation
    # -------------------------------------------------------------------------

    ref_orientation = (
        orientation_distribution(
            reference
        )
    )

    pred_orientation = (
        orientation_distribution(
            predicted
        )
    )

    orientation_jsd = (
        js_divergence(
            ref_orientation,
            pred_orientation,
        )
    )

    # -------------------------------------------------------------------------
    # Graph
    # -------------------------------------------------------------------------

    ref_graph = graph_from_roads(
        reference
    )

    pred_graph = graph_from_roads(
        predicted
    )

    ref_graph_stats = (
        graph_statistics(
            ref_graph
        )
    )

    pred_graph_stats = (
        graph_statistics(
            pred_graph
        )
    )

    ref_degree = (
        degree_distribution(
            ref_graph
        )
    )

    pred_degree = (
        degree_distribution(
            pred_graph
        )
    )

    degree_jsd = (
        js_divergence(
            ref_degree,
            pred_degree,
        )
    )

    component_relative_error = (
        relative_error(
            pred_graph_stats[
                "components"
            ],
            ref_graph_stats[
                "components"
            ],
        )
    )

    largest_component_error = abs(
        pred_graph_stats[
            "largest_component_ratio"
        ]
        - ref_graph_stats[
            "largest_component_ratio"
        ]
    )

    isolated_ratio_error = abs(
        pred_graph_stats[
            "isolated_ratio"
        ]
        - ref_graph_stats[
            "isolated_ratio"
        ]
    )

    mean_degree_relative_error = (
        relative_error(
            pred_graph_stats[
                "mean_degree"
            ],
            ref_graph_stats[
                "mean_degree"
            ],
        )
    )

    # -------------------------------------------------------------------------
    # Spatial coverage
    # -------------------------------------------------------------------------

    coverage = occupancy_metrics(
        reference,
        predicted,
    )

    # -------------------------------------------------------------------------
    # Local density
    # -------------------------------------------------------------------------

    ref_density = (
        density_distribution(
            reference
        )
    )

    pred_density = (
        density_distribution(
            predicted
        )
    )

    density_jsd = (
        js_divergence(
            ref_density,
            pred_density,
        )
    )

    # -------------------------------------------------------------------------
    # Connectivity preservation score
    # -------------------------------------------------------------------------
    #
    # Composite bounded score:
    #
    # 1 = perfect structural agreement.
    #
    # We deliberately keep the raw component metrics as well,
    # so the paper does not depend only on this composite.
    #

    structural_errors = np.array(
        [
            min(
                1.0,
                component_relative_error
                if np.isfinite(
                    component_relative_error
                )
                else 1.0,
            ),
            min(
                1.0,
                largest_component_error
                if np.isfinite(
                    largest_component_error
                )
                else 1.0,
            ),
            min(
                1.0,
                isolated_ratio_error
                if np.isfinite(
                    isolated_ratio_error
                )
                else 1.0,
            ),
            min(
                1.0,
                degree_jsd
                if np.isfinite(
                    degree_jsd
                )
                else 1.0,
            ),
        ],
        dtype=float,
    )

    connectivity_preservation = float(
        1.0
        - structural_errors.mean()
    )

    # -------------------------------------------------------------------------
    # Cartographic preservation score
    # -------------------------------------------------------------------------
    #
    # Again: this is supplementary.
    # Raw interpretable metrics remain primary.
    #

    cartographic_errors = np.array(
        [
            min(
                1.0,
                retention_rate_error,
            ),
            min(
                1.0,
                total_length_relative_error
                if np.isfinite(
                    total_length_relative_error
                )
                else 1.0,
            ),
            min(
                1.0,
                orientation_jsd
                if np.isfinite(
                    orientation_jsd
                )
                else 1.0,
            ),
            min(
                1.0,
                density_jsd
                if np.isfinite(
                    density_jsd
                )
                else 1.0,
            ),
            1.0
            - (
                coverage[
                    "coverage_jaccard"
                ]
                if np.isfinite(
                    coverage[
                        "coverage_jaccard"
                    ]
                )
                else 0.0
            ),
        ],
        dtype=float,
    )

    cartographic_preservation = float(
        1.0
        - cartographic_errors.mean()
    )

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    result = {
        "country": country,
        "city": city,
        "protocol": protocol,
        "regime": PROTOCOL_REGIME[
            protocol
        ],
        "seed": int(seed),

        "samples": int(
            len(mapped)
        ),

        "reference_retained": int(
            len(reference)
        ),

        "predicted_retained": int(
            len(predicted)
        ),

        "reference_retention_rate": (
            reference_retention_rate
        ),

        "predicted_retention_rate": (
            predicted_retention_rate
        ),

        "retention_rate_error": (
            retention_rate_error
        ),

        **cls,

        "reference_total_length_m": (
            ref_total_length
        ),

        "predicted_total_length_m": (
            pred_total_length
        ),

        "total_length_ratio": (
            length_ratio
        ),

        "total_length_relative_error": (
            total_length_relative_error
        ),

        "reference_mean_length_m": (
            ref_mean_length
        ),

        "predicted_mean_length_m": (
            pred_mean_length
        ),

        "mean_length_relative_error": (
            mean_length_relative_error
        ),

        "orientation_jsd": (
            orientation_jsd
        ),

        "reference_graph_nodes": (
            ref_graph_stats["nodes"]
        ),

        "predicted_graph_nodes": (
            pred_graph_stats["nodes"]
        ),

        "reference_graph_edges": (
            ref_graph_stats["edges"]
        ),

        "predicted_graph_edges": (
            pred_graph_stats["edges"]
        ),

        "reference_components": (
            ref_graph_stats[
                "components"
            ]
        ),

        "predicted_components": (
            pred_graph_stats[
                "components"
            ]
        ),

        "component_relative_error": (
            component_relative_error
        ),

        "reference_largest_component_ratio": (
            ref_graph_stats[
                "largest_component_ratio"
            ]
        ),

        "predicted_largest_component_ratio": (
            pred_graph_stats[
                "largest_component_ratio"
            ]
        ),

        "largest_component_ratio_error": (
            largest_component_error
        ),

        "reference_isolated_ratio": (
            ref_graph_stats[
                "isolated_ratio"
            ]
        ),

        "predicted_isolated_ratio": (
            pred_graph_stats[
                "isolated_ratio"
            ]
        ),

        "isolated_ratio_error": (
            isolated_ratio_error
        ),

        "reference_mean_degree": (
            ref_graph_stats[
                "mean_degree"
            ]
        ),

        "predicted_mean_degree": (
            pred_graph_stats[
                "mean_degree"
            ]
        ),

        "mean_degree_relative_error": (
            mean_degree_relative_error
        ),

        "degree_jsd": (
            degree_jsd
        ),

        "connectivity_preservation_score": (
            connectivity_preservation
        ),

        "reference_occupied_cells": (
            coverage[
                "reference_cells"
            ]
        ),

        "predicted_occupied_cells": (
            coverage[
                "predicted_cells"
            ]
        ),

        "coverage_jaccard": (
            coverage[
                "coverage_jaccard"
            ]
        ),

        "coverage_recall": (
            coverage[
                "coverage_recall"
            ]
        ),

        "coverage_precision": (
            coverage[
                "coverage_precision"
            ]
        ),

        "density_jsd": (
            density_jsd
        ),

        "cartographic_preservation_score": (
            cartographic_preservation
        ),
    }

    return result


# =============================================================================
# SUMMARY
# =============================================================================

def summarize_runs(
    d: pd.DataFrame,
) -> pd.DataFrame:

    group_cols = [
        "country",
        "city",
        "protocol",
        "regime",
    ]

    excluded = set(
        group_cols
        + ["seed"]
    )

    numeric_cols = [
        c
        for c in d.columns
        if c not in excluded
        and pd.api.types.is_numeric_dtype(
            d[c]
        )
    ]

    rows = []

    for keys, g in d.groupby(
        group_cols,
        dropna=False,
    ):

        row = {
            c: v
            for c, v
            in zip(
                group_cols,
                keys,
            )
        }

        row["runs"] = int(
            g["seed"].nunique()
        )

        for col in numeric_cols:

            values = pd.to_numeric(
                g[col],
                errors="coerce",
            )

            row[
                f"{col}_mean"
            ] = float(
                values.mean()
            )

            row[
                f"{col}_std"
            ] = float(
                values.std(
                    ddof=1
                )
            ) if len(values) > 1 else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_country_protocol(
    d: pd.DataFrame,
) -> pd.DataFrame:

    """
    Macro-average cities within each seed first.

    This prevents large cities from dominating the
    country-level comparison.
    """

    metric_cols = [
        "mcc",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "retention_rate_error",
        "total_length_relative_error",
        "mean_length_relative_error",
        "orientation_jsd",
        "component_relative_error",
        "largest_component_ratio_error",
        "isolated_ratio_error",
        "mean_degree_relative_error",
        "degree_jsd",
        "connectivity_preservation_score",
        "coverage_jaccard",
        "coverage_recall",
        "coverage_precision",
        "density_jsd",
        "cartographic_preservation_score",
    ]

    seed_rows = []

    for (
        country,
        protocol,
        regime,
        seed,
    ), g in d.groupby(
        [
            "country",
            "protocol",
            "regime",
            "seed",
        ]
    ):

        row = {
            "country": country,
            "protocol": protocol,
            "regime": regime,
            "seed": int(seed),
            "cities": int(
                g["city"].nunique()
            ),
        }

        for metric in metric_cols:

            row[metric] = float(
                pd.to_numeric(
                    g[metric],
                    errors="coerce",
                ).mean()
            )

        seed_rows.append(row)

    seed_df = pd.DataFrame(
        seed_rows
    )

    summary_rows = []

    for (
        country,
        protocol,
        regime,
    ), g in seed_df.groupby(
        [
            "country",
            "protocol",
            "regime",
        ]
    ):

        row = {
            "country": country,
            "protocol": protocol,
            "regime": regime,
            "runs": int(
                g["seed"].nunique()
            ),
        }

        for metric in metric_cols:

            values = pd.to_numeric(
                g[metric],
                errors="coerce",
            )

            row[
                f"{metric}_mean"
            ] = float(
                values.mean()
            )

            row[
                f"{metric}_std"
            ] = float(
                values.std(
                    ddof=1
                )
            ) if len(values) > 1 else 0.0

        summary_rows.append(row)

    summary = pd.DataFrame(
        summary_rows
    )

    return seed_df, summary


# =============================================================================
# PAPER TABLE
# =============================================================================

def mean_sd_string(
    mean: float,
    sd: float,
    digits: int = 3,
) -> str:

    if (
        not np.isfinite(mean)
    ):
        return "NA"

    if (
        not np.isfinite(sd)
    ):
        sd = 0.0

    return (
        f"{mean:.{digits}f} ± "
        f"{sd:.{digits}f}"
    )


def build_paper_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for _, r in summary.iterrows():

        rows.append(
            {
                "Country": r[
                    "country"
                ],

                "Training regime": r[
                    "regime"
                ],

                "Protocol": (
                    PROTOCOL_DISPLAY[
                        r["protocol"]
                    ]
                ),

                "MCC": mean_sd_string(
                    r["mcc_mean"],
                    r["mcc_std"],
                ),

                "Balanced accuracy": (
                    mean_sd_string(
                        r[
                            "balanced_accuracy_mean"
                        ],
                        r[
                            "balanced_accuracy_std"
                        ],
                    )
                ),

                "Retention error ↓": (
                    mean_sd_string(
                        r[
                            "retention_rate_error_mean"
                        ],
                        r[
                            "retention_rate_error_std"
                        ],
                    )
                ),

                "Length error ↓": (
                    mean_sd_string(
                        r[
                            "total_length_relative_error_mean"
                        ],
                        r[
                            "total_length_relative_error_std"
                        ],
                    )
                ),

                "Orientation JSD ↓": (
                    mean_sd_string(
                        r[
                            "orientation_jsd_mean"
                        ],
                        r[
                            "orientation_jsd_std"
                        ],
                    )
                ),

                "Degree JSD ↓": (
                    mean_sd_string(
                        r[
                            "degree_jsd_mean"
                        ],
                        r[
                            "degree_jsd_std"
                        ],
                    )
                ),

                "Coverage Jaccard ↑": (
                    mean_sd_string(
                        r[
                            "coverage_jaccard_mean"
                        ],
                        r[
                            "coverage_jaccard_std"
                        ],
                    )
                ),

                "Connectivity ↑": (
                    mean_sd_string(
                        r[
                            "connectivity_preservation_score_mean"
                        ],
                        r[
                            "connectivity_preservation_score_std"
                        ],
                    )
                ),

                "Cartographic preservation ↑": (
                    mean_sd_string(
                        r[
                            "cartographic_preservation_score_mean"
                        ],
                        r[
                            "cartographic_preservation_score_std"
                        ],
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# FIGURES
# =============================================================================

def figure_preservation_scores(
    summary: pd.DataFrame,
) -> None:

    if summary.empty:
        return

    countries = [
        x
        for x in COUNTRIES
        if x
        in summary[
            "country"
        ].unique()
    ]

    for country in countries:

        g = summary[
            summary["country"]
            == country
        ].copy()

        regime_order = [
            "within_country",
            "zero_shot",
            "joint_country",
        ]

        g["order"] = g[
            "regime"
        ].map(
            {
                x: i
                for i, x
                in enumerate(
                    regime_order
                )
            }
        )

        g = g.sort_values(
            "order"
        )

        labels = [
            x.replace(
                "_",
                " ",
            ).title()
            for x in g["regime"]
        ]

        x = np.arange(
            len(g)
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        width = 0.36

        ax.bar(
            x - width / 2,
            g[
                "connectivity_preservation_score_mean"
            ],
            width,
            yerr=g[
                "connectivity_preservation_score_std"
            ],
            capsize=4,
            label="Connectivity preservation",
        )

        ax.bar(
            x + width / 2,
            g[
                "cartographic_preservation_score_mean"
            ],
            width,
            yerr=g[
                "cartographic_preservation_score_std"
            ],
            capsize=4,
            label="Cartographic preservation",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        ax.set_ylim(
            0,
            1.05,
        )

        ax.set_ylabel(
            "Preservation score"
        )

        ax.set_title(
            f"{country}: structural and cartographic preservation"
        )

        ax.legend(
            frameon=False
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        fig.tight_layout()

        save_figure(
            fig,
            f"{clean_name(country)}_preservation_scores",
        )


def figure_error_metrics(
    summary: pd.DataFrame,
) -> None:

    if summary.empty:
        return

    metrics = [
        (
            "retention_rate_error_mean",
            "Retention-rate error",
        ),
        (
            "total_length_relative_error_mean",
            "Road-length error",
        ),
        (
            "orientation_jsd_mean",
            "Orientation JSD",
        ),
        (
            "degree_jsd_mean",
            "Degree JSD",
        ),
        (
            "density_jsd_mean",
            "Density JSD",
        ),
    ]

    for country in COUNTRIES:

        g = summary[
            summary["country"]
            == country
        ].copy()

        if g.empty:
            continue

        regime_order = [
            "within_country",
            "zero_shot",
            "joint_country",
        ]

        g["order"] = g[
            "regime"
        ].map(
            {
                x: i
                for i, x
                in enumerate(
                    regime_order
                )
            }
        )

        g = g.sort_values(
            "order"
        )

        labels = [
            x.replace(
                "_",
                " ",
            ).title()
            for x in g["regime"]
        ]

        x = np.arange(
            len(g)
        )

        fig, ax = plt.subplots(
            figsize=(9, 5.5)
        )

        width = (
            0.8
            / len(metrics)
        )

        for i, (
            metric,
            label,
        ) in enumerate(metrics):

            offset = (
                i
                - (
                    len(metrics) - 1
                )
                / 2
            ) * width

            ax.bar(
                x + offset,
                g[metric],
                width,
                label=label,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        ax.set_ylabel(
            "Error / divergence"
        )

        ax.set_title(
            f"{country}: cartographic distortion by training regime"
        )

        ax.legend(
            frameon=False,
            fontsize=8,
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        fig.tight_layout()

        save_figure(
            fig,
            f"{clean_name(country)}_cartographic_distortion",
        )


def figure_city_heatmap(
    per_run: pd.DataFrame,
    metric: str,
    title: str,
    stem: str,
) -> None:

    if per_run.empty:
        return

    city_mean = (
        per_run
        .groupby(
            [
                "city",
                "regime",
            ]
        )[metric]
        .mean()
        .unstack()
    )

    desired_cols = [
        x
        for x in [
            "within_country",
            "zero_shot",
            "joint_country",
        ]
        if x in city_mean.columns
    ]

    city_mean = city_mean[
        desired_cols
    ]

    if city_mean.empty:
        return

    matrix = (
        city_mean
        .to_numpy(
            dtype=float
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="viridis",
    )

    ax.set_xticks(
        np.arange(
            len(city_mean.columns)
        )
    )

    ax.set_xticklabels(
        [
            x.replace(
                "_",
                " ",
            ).title()
            for x in city_mean.columns
        ],
        rotation=20,
        ha="right",
    )

    ax.set_yticks(
        np.arange(
            len(city_mean.index)
        )
    )

    ax.set_yticklabels(
        city_mean.index
    )

    for i in range(
        matrix.shape[0]
    ):

        for j in range(
            matrix.shape[1]
        ):

            value = matrix[
                i,
                j,
            ]

            if np.isfinite(value):

                ax.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                )

    cbar = fig.colorbar(
        im,
        ax=ax,
    )

    cbar.set_label(
        metric.replace(
            "_",
            " ",
        )
    )

    ax.set_title(title)

    fig.tight_layout()

    save_figure(
        fig,
        stem,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "smoke",
            "final",
            "auto",
        ],
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    ensure_dirs()

    banner(
        "CARTOGEN — QUANTITATIVE SPATIAL AND CARTOGRAPHIC QUALITY EVALUATION"
    )

    print(
        f"Models: {MODEL_DIR}"
    )

    print(
        f"Output: {OUT_DIR}"
    )

    print(
        f"Requested mode: {args.mode}"
    )

    print(
        f"Smoke seed: {args.seed}"
    )

    print(
        f"Graph k: {GRAPH_K}"
    )

    print(
        f"Graph max distance: "
        f"{GRAPH_MAX_DISTANCE_M:.1f} m"
    )

    print(
        f"Occupancy grid: "
        f"{GRID_SIZE_M:.1f} m"
    )

    print(
        f"Density radius: "
        f"{DENSITY_RADIUS_M:.1f} m"
    )

    # =========================================================================
    # 1. VERIFY RUNS
    # =========================================================================

    section(
        "1. VERIFY CARTOGRAPH-X RUNS"
    )

    available_files = {}

    run_counts = {}

    for protocol in PROTOCOLS:

        files = prediction_files(
            protocol
        )

        available_files[
            protocol
        ] = files

        run_counts[
            protocol
        ] = len(files)

        seeds = [
            seed_from_path(x)
            for x in files
        ]

        print(
            f"{protocol:24s} "
            f"{len(files):2d} run(s) | "
            f"seeds={seeds}"
        )

    missing_protocols = [
        p
        for p in PROTOCOLS
        if run_counts[p] == 0
    ]

    if missing_protocols:

        raise FileNotFoundError(
            "Missing CartoGraph-X results for: "
            f"{missing_protocols}"
        )

    # =========================================================================
    # 2. DETERMINE MODE
    # =========================================================================

    section(
        "2. DETERMINE EVALUATION MODE"
    )

    if args.mode == "auto":

        if all(
            run_counts[p] >= 10
            for p in PROTOCOLS
        ):
            mode = "final"
        else:
            mode = "smoke"

    else:

        mode = args.mode

    print(
        f"Effective mode: {mode}"
    )

    if mode == "smoke":

        selected_files = {}

        for protocol in PROTOCOLS:

            lookup = {
                seed_from_path(p): p
                for p in available_files[
                    protocol
                ]
            }

            if args.seed not in lookup:

                raise FileNotFoundError(
                    f"{protocol}: seed "
                    f"{args.seed} unavailable. "
                    f"Available seeds: "
                    f"{sorted(lookup)}"
                )

            selected_files[
                protocol
            ] = [
                lookup[args.seed]
            ]

    else:

        selected_files = {
            protocol: files
            for protocol, files
            in available_files.items()
        }

        min_runs = min(
            len(v)
            for v
            in selected_files.values()
        )

        print(
            f"Minimum available runs across "
            f"protocols: {min_runs}"
        )

        if min_runs < 2:

            raise RuntimeError(
                "Final mode requires at least "
                "two runs per protocol."
            )

    # =========================================================================
    # 3. LOAD SOURCE GEOMETRY
    # =========================================================================

    section(
        "3. LOAD SOURCE ROAD GEOMETRY"
    )

    source_geometry = {}

    for country in COUNTRIES:

        source_geometry[
            country
        ] = {}

        for city in CITIES[
            country
        ]:

            g = load_source_geometry(
                country,
                city,
            )

            source_geometry[
                country
            ][city] = g

            print(
                f"{country:10s} | "
                f"{city:10s} | "
                f"{len(g):6,d} roads | "
                f"CRS={g.crs}"
            )

    # =========================================================================
    # 4. EVALUATE EACH RUN
    # =========================================================================

    section(
        "4. EVALUATE EACH PROTOCOL × RUN × CITY"
    )

    results = []

    join_audit = []

    total_jobs = sum(
        len(selected_files[p])
        * len(
            CITIES[
                PROTOCOL_COUNTRY[p]
            ]
        )
        for p in PROTOCOLS
    )

    job = 0

    for protocol in PROTOCOLS:

        country = (
            PROTOCOL_COUNTRY[
                protocol
            ]
        )

        for prediction_path in (
            selected_files[
                protocol
            ]
        ):

            prediction = (
                load_prediction_file(
                    prediction_path
                )
            )

            seed = int(
                prediction[
                    "seed"
                ].iloc[0]
            )

            for city in CITIES[
                country
            ]:

                job += 1

                print()
                print(
                    f"[{job:03d}/{total_jobs:03d}] "
                    f"{protocol} | "
                    f"seed={seed} | "
                    f"{city}"
                )

                source = (
                    source_geometry[
                        country
                    ][city]
                )

                city_prediction = (
                    prediction[
                        prediction[
                            "uid_city"
                        ]
                        == city
                    ]
                )

                mapped = (
                    map_prediction_geometry(
                        source,
                        prediction,
                        city,
                    )
                )

                expected = len(
                    city_prediction
                )

                actual = len(
                    mapped
                )

                join_pct = (
                    100.0
                    * actual
                    / expected
                    if expected
                    else 0.0
                )

                join_audit.append(
                    {
                        "country": country,
                        "city": city,
                        "protocol": protocol,
                        "seed": seed,
                        "prediction_rows": (
                            expected
                        ),
                        "mapped_rows": (
                            actual
                        ),
                        "join_pct": (
                            join_pct
                        ),
                    }
                )

                print(
                    f"  predictions: {expected:,}"
                )

                print(
                    f"  mapped:      {actual:,}"
                )

                print(
                    f"  join:        "
                    f"{join_pct:.2f}%"
                )

                if expected == 0:

                    warnings.warn(
                        f"No predictions for "
                        f"{protocol}/{seed}/{city}"
                    )

                    continue

                if actual != expected:

                    raise RuntimeError(
                        f"Incomplete geometry join "
                        f"for {protocol}/"
                        f"seed {seed}/{city}: "
                        f"{actual}/{expected}"
                    )

                result = evaluate_network(
                    mapped=mapped,
                    country=country,
                    city=city,
                    protocol=protocol,
                    seed=seed,
                )

                results.append(
                    result
                )

                print(
                    "  "
                    f"MCC={result['mcc']:.4f} | "
                    f"BA={result['balanced_accuracy']:.4f} | "
                    f"coverage={result['coverage_jaccard']:.4f} | "
                    f"connectivity={result['connectivity_preservation_score']:.4f} | "
                    f"cartographic={result['cartographic_preservation_score']:.4f}"
                )

    per_run = pd.DataFrame(
        results
    )

    join_audit_df = pd.DataFrame(
        join_audit
    )

    if per_run.empty:

        raise RuntimeError(
            "No spatial evaluation results "
            "were generated."
        )

    # =========================================================================
    # 5. SAVE RAW PER-RUN RESULTS
    # =========================================================================

    section(
        "5. SAVE PER-RUN SPATIAL METRICS"
    )

    per_run_path = (
        RUN_DIR
        / "cartographic_quality_per_run.csv"
    )

    per_run.to_csv(
        per_run_path,
        index=False,
    )

    join_path = (
        TABLE_DIR
        / "geometry_join_audit.csv"
    )

    join_audit_df.to_csv(
        join_path,
        index=False,
    )

    print(
        f"Per-run metrics: {per_run_path}"
    )

    print(
        f"Join audit:      {join_path}"
    )

    # =========================================================================
    # 6. CITY × PROTOCOL SUMMARY
    # =========================================================================

    section(
        "6. CITY-LEVEL MULTI-RUN SUMMARY"
    )

    city_summary = summarize_runs(
        per_run
    )

    city_summary_path = (
        TABLE_DIR
        / "city_protocol_summary.csv"
    )

    city_summary.to_csv(
        city_summary_path,
        index=False,
    )

    print(
        city_summary[
            [
                "country",
                "city",
                "protocol",
                "regime",
                "runs",
                "mcc_mean",
                "mcc_std",
                "coverage_jaccard_mean",
                "coverage_jaccard_std",
                "connectivity_preservation_score_mean",
                "connectivity_preservation_score_std",
                "cartographic_preservation_score_mean",
                "cartographic_preservation_score_std",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    # =========================================================================
    # 7. COUNTRY × PROTOCOL SUMMARY
    # =========================================================================

    section(
        "7. COUNTRY-LEVEL GENERALIZATION SUMMARY"
    )

    (
        country_seed,
        country_summary,
    ) = summarize_country_protocol(
        per_run
    )

    country_seed_path = (
        TABLE_DIR
        / "country_protocol_per_seed.csv"
    )

    country_summary_path = (
        TABLE_DIR
        / "country_protocol_summary.csv"
    )

    country_seed.to_csv(
        country_seed_path,
        index=False,
    )

    country_summary.to_csv(
        country_summary_path,
        index=False,
    )

    display_cols = [
        "country",
        "protocol",
        "regime",
        "runs",
        "mcc_mean",
        "mcc_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "coverage_jaccard_mean",
        "coverage_jaccard_std",
        "connectivity_preservation_score_mean",
        "connectivity_preservation_score_std",
        "cartographic_preservation_score_mean",
        "cartographic_preservation_score_std",
    ]

    print(
        country_summary[
            display_cols
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    # =========================================================================
    # 8. PAPER TABLE
    # =========================================================================

    section(
        "8. GENERATE PAPER-READY TABLE"
    )

    paper_table = (
        build_paper_table(
            country_summary
        )
    )

    paper_table_path = (
        TABLE_DIR
        / "paper_table_cartographic_quality.csv"
    )

    paper_table.to_csv(
        paper_table_path,
        index=False,
    )

    print(
        paper_table.to_string(
            index=False
        )
    )

    # =========================================================================
    # 9. FIGURES
    # =========================================================================

    section(
        "9. GENERATE SPATIAL QUALITY FIGURES"
    )

    figure_preservation_scores(
        country_summary
    )

    figure_error_metrics(
        country_summary
    )

    figure_city_heatmap(
        per_run,
        metric=(
            "cartographic_preservation_score"
        ),
        title=(
            "Cartographic preservation "
            "across cities and training regimes"
        ),
        stem=(
            "figure_cartographic_preservation_city_heatmap"
        ),
    )

    figure_city_heatmap(
        per_run,
        metric=(
            "connectivity_preservation_score"
        ),
        title=(
            "Connectivity preservation "
            "across cities and training regimes"
        ),
        stem=(
            "figure_connectivity_preservation_city_heatmap"
        ),
    )

    figure_city_heatmap(
        per_run,
        metric=(
            "coverage_jaccard"
        ),
        title=(
            "Spatial coverage preservation "
            "across cities and training regimes"
        ),
        stem=(
            "figure_coverage_jaccard_city_heatmap"
        ),
    )

    # =========================================================================
    # 10. GENERALIZATION GAINS
    # =========================================================================

    section(
        "10. JOINT-COUNTRY GENERALIZATION GAINS"
    )

    gain_rows = []

    metrics_for_gain = [
        "mcc",
        "balanced_accuracy",
        "coverage_jaccard",
        "connectivity_preservation_score",
        "cartographic_preservation_score",
    ]

    for country in COUNTRIES:

        g = country_seed[
            country_seed[
                "country"
            ]
            == country
        ]

        within = g[
            g["regime"]
            == "within_country"
        ].copy()

        zero = g[
            g["regime"]
            == "zero_shot"
        ].copy()

        joint = g[
            g["regime"]
            == "joint_country"
        ].copy()

        # Seed-matched comparison where possible.
        joint_within = joint.merge(
            within,
            on="seed",
            suffixes=(
                "_joint",
                "_within",
            ),
        )

        joint_zero = joint.merge(
            zero,
            on="seed",
            suffixes=(
                "_joint",
                "_zero",
            ),
        )

        for metric in (
            metrics_for_gain
        ):

            if not joint_within.empty:

                diff = (
                    joint_within[
                        f"{metric}_joint"
                    ]
                    - joint_within[
                        f"{metric}_within"
                    ]
                )

                gain_rows.append(
                    {
                        "country": country,
                        "comparison": (
                            "joint_minus_within"
                        ),
                        "metric": metric,
                        "runs": int(
                            len(diff)
                        ),
                        "mean_gain": float(
                            diff.mean()
                        ),
                        "std_gain": float(
                            diff.std(
                                ddof=1
                            )
                        )
                        if len(diff) > 1
                        else 0.0,
                    }
                )

            if not joint_zero.empty:

                diff = (
                    joint_zero[
                        f"{metric}_joint"
                    ]
                    - joint_zero[
                        f"{metric}_zero"
                    ]
                )

                gain_rows.append(
                    {
                        "country": country,
                        "comparison": (
                            "joint_minus_zero_shot"
                        ),
                        "metric": metric,
                        "runs": int(
                            len(diff)
                        ),
                        "mean_gain": float(
                            diff.mean()
                        ),
                        "std_gain": float(
                            diff.std(
                                ddof=1
                            )
                        )
                        if len(diff) > 1
                        else 0.0,
                    }
                )

    gain_df = pd.DataFrame(
        gain_rows
    )

    gain_path = (
        TABLE_DIR
        / "joint_country_generalization_gains.csv"
    )

    gain_df.to_csv(
        gain_path,
        index=False,
    )

    if not gain_df.empty:

        print(
            gain_df.to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.4f}"
                ),
            )
        )

    # =========================================================================
    # 11. FINAL AUDIT
    # =========================================================================

    section(
        "11. FINAL CARTOGRAPHIC QUALITY AUDIT"
    )

    checks = {}

    checks[
        "both_countries_present"
    ] = (
        set(
            per_run[
                "country"
            ].unique()
        )
        == set(COUNTRIES)
    )

    checks[
        "all_six_cities_present"
    ] = (
        per_run[
            "city"
        ].nunique()
        == 6
    )

    checks[
        "all_six_protocols_present"
    ] = (
        set(
            per_run[
                "protocol"
            ].unique()
        )
        == set(PROTOCOLS)
    )

    checks[
        "all_geometry_joins_complete"
    ] = bool(
        (
            join_audit_df[
                "prediction_rows"
            ]
            == join_audit_df[
                "mapped_rows"
            ]
        ).all()
    )

    checks[
        "no_centroid_identity_matching"
    ] = True

    checks[
        "reference_is_ground_truth_retained_source_network"
    ] = True

    checks[
        "classification_metrics_computed"
    ] = bool(
        per_run[
            [
                "mcc",
                "balanced_accuracy",
                "f1",
            ]
        ]
        .notna()
        .all()
        .all()
    )

    checks[
        "network_metrics_computed"
    ] = bool(
        per_run[
            [
                "degree_jsd",
                "connectivity_preservation_score",
            ]
        ]
        .notna()
        .all()
        .all()
    )

    checks[
        "spatial_coverage_metrics_computed"
    ] = bool(
        per_run[
            [
                "coverage_jaccard",
                "coverage_recall",
            ]
        ]
        .notna()
        .all()
        .all()
    )

    checks[
        "cartographic_metrics_computed"
    ] = bool(
        per_run[
            [
                "orientation_jsd",
                "density_jsd",
                "cartographic_preservation_score",
            ]
        ]
        .notna()
        .all()
        .all()
    )

    if mode == "final":

        checks[
            "multiple_runs_evaluated"
        ] = (
            per_run[
                "seed"
            ].nunique()
            >= 2
        )

    for key, value in (
        checks.items()
    ):

        print(
            f"{key:65s} "
            f"{'PASS' if value else 'FAIL'}"
        )

    # =========================================================================
    # 12. REPORT
    # =========================================================================

    report = {
        "script": 75,
        "mode": mode,
        "requested_seed": (
            args.seed
            if mode == "smoke"
            else None
        ),
        "graph_configuration": {
            "k": GRAPH_K,
            "max_distance_m": (
                GRAPH_MAX_DISTANCE_M
            ),
        },
        "spatial_configuration": {
            "grid_size_m": (
                GRID_SIZE_M
            ),
            "density_radius_m": (
                DENSITY_RADIUS_M
            ),
            "orientation_bin_width_deg": 15,
        },
        "reference_definition": (
            "Source road geometries with "
            "ground-truth y_true == 1"
        ),
        "run_counts": run_counts,
        "evaluated_rows": int(
            len(per_run)
        ),
        "audit": checks,
        "outputs": {
            "per_run_metrics": str(
                per_run_path
            ),
            "join_audit": str(
                join_path
            ),
            "city_summary": str(
                city_summary_path
            ),
            "country_seed_summary": str(
                country_seed_path
            ),
            "country_summary": str(
                country_summary_path
            ),
            "paper_table": str(
                paper_table_path
            ),
            "generalization_gains": str(
                gain_path
            ),
            "figures": str(
                FIG_DIR
            ),
        },
    }

    report_path = (
        OUT_DIR
        / "cartographic_quality_report.json"
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

    # =========================================================================
    # 13. OUTPUT
    # =========================================================================

    section(
        "12. OUTPUTS"
    )

    print(
        f"Per-run metrics:       "
        f"{per_run_path}"
    )

    print(
        f"City summary:          "
        f"{city_summary_path}"
    )

    print(
        f"Country/run summary:   "
        f"{country_seed_path}"
    )

    print(
        f"Country summary:       "
        f"{country_summary_path}"
    )

    print(
        f"Paper table:           "
        f"{paper_table_path}"
    )

    print(
        f"Generalization gains:  "
        f"{gain_path}"
    )

    print(
        f"Figures:               "
        f"{FIG_DIR}"
    )

    print(
        f"Report:                "
        f"{report_path}"
    )

    banner(
        "SCRIPT 75 COMPLETE"
    )

    if all(
        checks.values()
    ):

        print(
            "[PASS] SPATIAL AND CARTOGRAPHIC "
            "QUALITY EVALUATION COMPLETE."
        )

    else:

        print(
            "[CHECK] EVALUATION COMPLETED "
            "WITH ONE OR MORE AUDIT WARNINGS."
        )

    print()

    if mode == "smoke":

        print(
            "Smoke evaluation complete. "
            "Do not use seed-42-only values as "
            "the final paper results."
        )

        print(
            "NEXT: validate these outputs, then "
            "Script 76 will orchestrate the final "
            "10-run experiment for baselines and "
            "CartoGraph-X."
        )

    else:

        print(
            "Multi-run evaluation complete."
        )

        print(
            "NEXT: Script 77 — paired statistical "
            "testing, confidence intervals, final "
            "tables, and publication figures."
        )


if __name__ == "__main__":
    main()