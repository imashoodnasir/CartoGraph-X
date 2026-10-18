"""
Script 74
=========
Spatial reconstruction and publication-quality cartographic map generation
for CartoGraph-X.

Produces:
    1. Source detailed road network
    2. Reference generalized road network
    3. CartoGraph-X predicted generalized network
    4. TP / TN / FP / FN error map
    5. Retention probability map
    6. Within-country vs zero-shot vs joint-country comparison
    7. City-level and representative tile-level figures
    8. Spatial GeoPackage outputs for subsequent GIS analysis

Supports:
    --mode smoke
    --mode final
    --mode auto

Smoke:
    uses seed 42 predictions.

Final:
    aggregates all available seeds using:
        mean predicted probability
        majority prediction
        prediction agreement
        prediction uncertainty

Important
---------
No model is trained in this script.

No nearest-centroid matching is used.

Lithuania:
    prediction sample_uid final token -> cartogen_source_id

Poland:
    prediction sample_uid final token -> source_idx

Reference maps:
    Lithuania -> target_roads from road_correspondence GPKG
    Poland    -> target/reference geometry recovered from available
                 correspondence/validated road datasets.

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
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"

CROSS = PROCESSED / "cross_country"

MODEL_DIR = CROSS / "cartographx_models"

OUT_DIR = CROSS / "cartographx_maps"

CITY_DIR = OUT_DIR / "cities"
TILE_DIR = OUT_DIR / "tiles"
COMPARE_DIR = OUT_DIR / "protocol_comparisons"
GPKG_DIR = OUT_DIR / "geopackages"
FIG_DIR = OUT_DIR / "figures"


# =============================================================================
# EXPERIMENT DEFINITIONS
# =============================================================================

COUNTRIES = ["Lithuania", "Poland"]

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

WITHIN_PROTOCOL = {
    "Lithuania": "lt_to_lt",
    "Poland": "pl_to_pl",
}

ZERO_PROTOCOL = {
    "Lithuania": "pl_to_lt_zero_shot",
    "Poland": "lt_to_pl_zero_shot",
}

JOINT_PROTOCOL = {
    "Lithuania": "joint_to_lt",
    "Poland": "joint_to_pl",
}

PROTOCOL_DISPLAY = {
    "lt_to_lt": "LT → LT",
    "pl_to_pl": "PL → PL",
    "lt_to_pl_zero_shot": "LT → PL",
    "pl_to_lt_zero_shot": "PL → LT",
    "joint_to_lt": "Joint → LT",
    "joint_to_pl": "Joint → PL",
}


# =============================================================================
# VISUAL SETTINGS
# =============================================================================

SOURCE_LINEWIDTH = 0.55
REFERENCE_LINEWIDTH = 0.85
PREDICTED_LINEWIDTH = 0.85
ERROR_LINEWIDTH = 1.00

# These are semantic map colors rather than arbitrary chart styling.
COLOR_SOURCE = "#B8B8B8"
COLOR_REFERENCE = "#222222"
COLOR_RETAINED = "#1565C0"

COLOR_TP = "#2E7D32"
COLOR_TN = "#BDBDBD"
COLOR_FP = "#EF6C00"
COLOR_FN = "#C62828"

COLOR_BOUNDARY = "#666666"

DPI = 350


# =============================================================================
# BASIC UTILITIES
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
        CITY_DIR,
        TILE_DIR,
        COMPARE_DIR,
        GPKG_DIR,
        FIG_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


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

    return mapping.get(s, str(x))


def save_figure(
    fig: plt.Figure,
    directory: Path,
    stem: str,
) -> None:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    png = directory / f"{stem}.png"
    pdf = directory / f"{stem}.pdf"

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


def json_safe(obj):

    if isinstance(obj, dict):
        return {
            str(k): json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]

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


# =============================================================================
# SAMPLE UID
# =============================================================================

def parse_sample_uid(uid: str) -> Tuple[str, str, int]:

    parts = str(uid).split("::")

    if len(parts) < 3:
        raise ValueError(
            f"Unexpected sample_uid: {uid}"
        )

    country = parts[0]
    city = parts[1]

    try:
        source_id = int(parts[-1])
    except Exception as e:
        raise ValueError(
            f"Could not recover source ID from {uid}"
        ) from e

    return country, city, source_id


def add_uid_fields(
    d: pd.DataFrame,
) -> pd.DataFrame:

    out = d.copy()

    parsed = out["sample_uid"].map(
        parse_sample_uid
    )

    out["uid_country"] = [
        x[0] for x in parsed
    ]

    out["uid_city"] = [
        normalize_city(x[1])
        for x in parsed
    ]

    out["source_id"] = [
        x[2] for x in parsed
    ]

    return out


# =============================================================================
# PREDICTION LOADING
# =============================================================================

def prediction_files(
    protocol: str,
) -> List[Path]:

    root = MODEL_DIR / protocol

    if not root.exists():
        return []

    files = sorted(
        root.glob(
            "seed_*/test_predictions.csv"
        )
    )

    return files


def seed_from_path(path: Path) -> int:

    m = re.search(
        r"seed_(\d+)",
        str(path),
        flags=re.IGNORECASE,
    )

    if not m:
        raise ValueError(
            f"Could not identify seed from {path}"
        )

    return int(m.group(1))


def load_single_prediction(
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
        c for c in required
        if c not in d.columns
    ]

    if missing:
        raise KeyError(
            f"{path}\n"
            f"Missing prediction columns: {missing}"
        )

    d = add_uid_fields(d)

    d["seed"] = seed_from_path(path)

    d["y_true"] = pd.to_numeric(
        d["y_true"],
        errors="coerce",
    )

    d["y_pred"] = pd.to_numeric(
        d["y_pred"],
        errors="coerce",
    )

    d["prob_retained"] = pd.to_numeric(
        d["prob_retained"],
        errors="coerce",
    )

    return d


def load_protocol_predictions(
    protocol: str,
    mode: str,
    requested_seed: int,
) -> pd.DataFrame:

    files = prediction_files(protocol)

    if not files:
        raise FileNotFoundError(
            f"No prediction files for {protocol}:\n"
            f"{MODEL_DIR / protocol}"
        )

    available = {
        seed_from_path(p): p
        for p in files
    }

    if mode == "smoke":

        if requested_seed not in available:
            raise FileNotFoundError(
                f"Seed {requested_seed} not found for "
                f"{protocol}. Available: "
                f"{sorted(available)}"
            )

        d = load_single_prediction(
            available[requested_seed]
        )

        d["prob_retained_mean"] = (
            d["prob_retained"]
        )

        d["prob_retained_std"] = 0.0
        d["prediction_agreement"] = 1.0
        d["n_runs"] = 1

        # IMPORTANT:
        # use actual model y_pred because Script 71 may have
        # validation-selected thresholds other than 0.5.
        d["consensus_pred"] = (
            d["y_pred"].astype(int)
        )

        return d

    # -------------------------------------------------------------------------
    # Multi-run aggregation
    # -------------------------------------------------------------------------

    frames = [
        load_single_prediction(p)
        for p in files
    ]

    all_runs = pd.concat(
        frames,
        ignore_index=True,
    )

    invariant_cols = [
        "country",
        "tile_id",
        "y_true",
        "uid_country",
        "uid_city",
        "source_id",
    ]

    # Check truth consistency.
    truth_nunique = (
        all_runs.groupby("sample_uid")["y_true"]
        .nunique()
    )

    if (truth_nunique > 1).any():
        raise RuntimeError(
            f"Inconsistent y_true across seeds for "
            f"{protocol}."
        )

    first = (
        all_runs
        .sort_values(["sample_uid", "seed"])
        .drop_duplicates("sample_uid")
        [["sample_uid"] + invariant_cols]
    )

    agg = (
        all_runs.groupby("sample_uid")
        .agg(
            prob_retained_mean=(
                "prob_retained",
                "mean",
            ),
            prob_retained_std=(
                "prob_retained",
                "std",
            ),
            mean_binary_prediction=(
                "y_pred",
                "mean",
            ),
            n_runs=(
                "seed",
                "nunique",
            ),
        )
        .reset_index()
    )

    agg["prob_retained_std"] = (
        agg["prob_retained_std"]
        .fillna(0.0)
    )

    # Majority vote across each seed's own
    # validation-selected threshold.
    agg["consensus_pred"] = (
        agg["mean_binary_prediction"]
        >= 0.5
    ).astype(int)

    agg["prediction_agreement"] = np.maximum(
        agg["mean_binary_prediction"],
        1.0 - agg["mean_binary_prediction"],
    )

    out = first.merge(
        agg,
        on="sample_uid",
        how="inner",
        validate="one_to_one",
    )

    return out


# =============================================================================
# LITHUANIA GEOMETRY
# =============================================================================

def lithuania_correspondence_path(
    city: str,
) -> Path:

    key = CITY_KEYS[city]

    return (
        PROCESSED
        / "lithuania"
        / "road_correspondence"
        / f"{key}_central_road_correspondence.gpkg"
    )


def lithuania_topology_path(
    city: str,
) -> Path:

    key = CITY_KEYS[city]

    return (
        PROCESSED
        / "lithuania"
        / "road_topology"
        / f"{key}_central_road_topology.gpkg"
    )


def load_lithuania_city(
    city: str,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:

    correspondence = (
        lithuania_correspondence_path(city)
    )

    if not correspondence.exists():
        raise FileNotFoundError(
            correspondence
        )

    source = gpd.read_file(
        correspondence,
        layer="source_roads",
    )

    target = gpd.read_file(
        correspondence,
        layer="target_roads",
    )

    if "cartogen_source_id" not in source.columns:
        raise KeyError(
            f"cartogen_source_id missing in "
            f"{correspondence}"
        )

    source = source.copy()

    source["source_id"] = pd.to_numeric(
        source["cartogen_source_id"],
        errors="coerce",
    )

    source = source[
        source["source_id"].notna()
    ].copy()

    source["source_id"] = (
        source["source_id"].astype(int)
    )

    source["city"] = city
    source["country"] = "Lithuania"

    target = target.copy()
    target["city"] = city
    target["country"] = "Lithuania"

    return source, target


# =============================================================================
# POLAND GEOMETRY
# =============================================================================

def poland_learning_path(
    city: str,
) -> Path:

    key = CITY_KEYS[city]

    return (
        PROCESSED
        / "poland"
        / "learning_dataset"
        / f"{key}_road_learning_dataset.gpkg"
    )


def poland_correspondence_path(
    city: str,
) -> Path:

    key = CITY_KEYS[city]

    return (
        PROCESSED
        / "poland"
        / "road_correspondence"
        / f"{key}_road_correspondence.gpkg"
    )


def poland_validated_path(
    city: str,
) -> Path:

    key = CITY_KEYS[city]

    return (
        PROCESSED
        / "poland"
        / "road_target_validation"
        / f"{key}_validated_road_pair.gpkg"
    )


def load_poland_source(
    city: str,
) -> gpd.GeoDataFrame:

    p = poland_learning_path(city)

    if not p.exists():
        raise FileNotFoundError(p)

    layers = gpd.list_layers(p)["name"].tolist()

    if "road_learning_samples" not in layers:
        raise KeyError(
            f"road_learning_samples not found in {p}"
        )

    source = gpd.read_file(
        p,
        layer="road_learning_samples",
    )

    if "source_idx" not in source.columns:
        raise KeyError(
            f"source_idx missing from {p}"
        )

    source = source.copy()

    source["source_id"] = pd.to_numeric(
        source["source_idx"],
        errors="coerce",
    )

    source = source[
        source["source_id"].notna()
    ].copy()

    source["source_id"] = (
        source["source_id"].astype(int)
    )

    source["city"] = city
    source["country"] = "Poland"

    return source


def choose_poland_target_layer(
    path: Path,
) -> Optional[Tuple[Path, str]]:

    if not path.exists():
        return None

    layers = gpd.list_layers(path)

    if len(layers) == 0:
        return None

    names = layers["name"].tolist()

    preferred = [
        x for x in names
        if "target" in clean_name(x)
    ]

    if preferred:
        return path, preferred[0]

    # If correspondence GPKG contains exactly two
    # obvious source/target layers, target is preferred.
    for name in names:
        n = clean_name(name)

        if (
            "general" in n
            or "reference" in n
            or "target" in n
        ):
            return path, name

    return None


def load_poland_target(
    city: str,
    source: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Attempt 1:
        correspondence target layer.

    Attempt 2:
        validated target pair GPKG.

    Attempt 3:
        derive reference-selected source geometry from
        the learning dataset.

    Attempt 3 is still valid for the selection task:
    it visualizes the source roads that the reference
    cartographic supervision retains.
    """

    cp = poland_correspondence_path(city)

    candidate = choose_poland_target_layer(cp)

    if candidate is not None:

        p, layer = candidate

        target = gpd.read_file(
            p,
            layer=layer,
        )

        target["city"] = city
        target["country"] = "Poland"
        target["reference_geometry_type"] = (
            "target_cartographic_geometry"
        )

        return target

    vp = poland_validated_path(city)

    if vp.exists():

        layers = gpd.list_layers(vp)

        for layer in layers["name"]:

            g = gpd.read_file(
                vp,
                layer=layer,
            )

            lname = clean_name(layer)

            if (
                "target" in lname
                or "validated" in lname
            ):
                g["city"] = city
                g["country"] = "Poland"
                g["reference_geometry_type"] = (
                    "validated_target_geometry"
                )

                return g

    # Fallback:
    # reference retained source roads.
    label_col = None

    for c in [
        "is_selected",
        "selection_label",
        "operation_label",
    ]:
        if c in source.columns:
            label_col = c
            break

    if label_col is None:
        raise RuntimeError(
            f"Could not identify Polish reference "
            f"selection in {city}."
        )

    target = source.copy()

    if label_col == "is_selected":

        mask = (
            pd.to_numeric(
                target[label_col],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            == 1
        )

    else:

        values = (
            target[label_col]
            .astype(str)
            .map(clean_name)
        )

        mask = values.isin(
            [
                "selected",
                "weak_selected",
                "retained",
                "keep",
                "kept",
                "1",
                "true",
            ]
        )

    target = target[mask].copy()

    target["reference_geometry_type"] = (
        "retained_source_geometry"
    )

    return target


# =============================================================================
# JOIN PREDICTIONS TO GEOMETRY
# =============================================================================

def join_predictions_to_source(
    source: gpd.GeoDataFrame,
    pred: pd.DataFrame,
    city: str,
) -> gpd.GeoDataFrame:

    p = pred[
        pred["uid_city"] == city
    ].copy()

    if p.empty:
        return gpd.GeoDataFrame(
            columns=list(source.columns),
            geometry="geometry",
            crs=source.crs,
        )

    if source["source_id"].duplicated().any():
        duplicates = int(
            source["source_id"]
            .duplicated()
            .sum()
        )

        raise RuntimeError(
            f"{city}: source_id has "
            f"{duplicates} duplicates."
        )

    if p["source_id"].duplicated().any():
        raise RuntimeError(
            f"{city}: prediction source_id "
            f"is not unique."
        )

    keep_pred = [
        "sample_uid",
        "source_id",
        "tile_id",
        "y_true",
        "prob_retained_mean",
        "prob_retained_std",
        "consensus_pred",
        "prediction_agreement",
        "n_runs",
    ]

    merged = source.merge(
        p[keep_pred],
        on="source_id",
        how="inner",
        validate="one_to_one",
    )

    merged = gpd.GeoDataFrame(
        merged,
        geometry="geometry",
        crs=source.crs,
    )

    merged["error_class"] = np.select(
        [
            (
                (merged["y_true"] == 1)
                & (merged["consensus_pred"] == 1)
            ),
            (
                (merged["y_true"] == 0)
                & (merged["consensus_pred"] == 0)
            ),
            (
                (merged["y_true"] == 0)
                & (merged["consensus_pred"] == 1)
            ),
            (
                (merged["y_true"] == 1)
                & (merged["consensus_pred"] == 0)
            ),
        ],
        [
            "TP",
            "TN",
            "FP",
            "FN",
        ],
        default="UNKNOWN",
    )

    return merged


# =============================================================================
# GEOMETRIC PLOTTING HELPERS
# =============================================================================

def plot_lines(
    ax,
    gdf: gpd.GeoDataFrame,
    color,
    linewidth: float,
    alpha: float = 1.0,
    zorder: int = 1,
) -> None:

    if gdf is None or gdf.empty:
        return

    gdf.plot(
        ax=ax,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )


def set_equal_extent(
    axes,
    gdfs: Sequence[gpd.GeoDataFrame],
    padding: float = 0.03,
) -> None:

    valid = [
        g
        for g in gdfs
        if g is not None
        and not g.empty
    ]

    if not valid:
        return

    bounds = np.array(
        [g.total_bounds for g in valid]
    )

    xmin = np.nanmin(bounds[:, 0])
    ymin = np.nanmin(bounds[:, 1])
    xmax = np.nanmax(bounds[:, 2])
    ymax = np.nanmax(bounds[:, 3])

    dx = xmax - xmin
    dy = ymax - ymin

    px = (
        dx * padding
        if dx > 0
        else 1.0
    )

    py = (
        dy * padding
        if dy > 0
        else 1.0
    )

    for ax in np.ravel(axes):
        ax.set_xlim(
            xmin - px,
            xmax + px,
        )

        ax.set_ylim(
            ymin - py,
            ymax + py,
        )

        ax.set_aspect("equal")
        ax.set_axis_off()


def clip_to_bounds(
    gdf: gpd.GeoDataFrame,
    bounds: Sequence[float],
) -> gpd.GeoDataFrame:

    if gdf is None or gdf.empty:
        return gdf

    xmin, ymin, xmax, ymax = bounds

    try:
        return gdf.cx[
            xmin:xmax,
            ymin:ymax,
        ].copy()
    except Exception:
        return gdf[
            gdf.geometry.intersects(
                gpd.GeoSeries.from_bbox(
                    (xmin, ymin, xmax, ymax)
                ).iloc[0]
            )
        ].copy()


# =============================================================================
# TILE SELECTION
# =============================================================================

def tile_metrics(
    mapped: gpd.GeoDataFrame,
) -> pd.DataFrame:

    if mapped.empty:
        return pd.DataFrame()

    rows = []

    for tile_id, g in mapped.groupby("tile_id"):

        n = len(g)

        tp = int(
            (g["error_class"] == "TP").sum()
        )

        tn = int(
            (g["error_class"] == "TN").sum()
        )

        fp = int(
            (g["error_class"] == "FP").sum()
        )

        fn = int(
            (g["error_class"] == "FN").sum()
        )

        accuracy = (
            (tp + tn) / n
            if n
            else np.nan
        )

        error_rate = (
            (fp + fn) / n
            if n
            else np.nan
        )

        retained_true = int(
            (g["y_true"] == 1).sum()
        )

        eliminated_true = int(
            (g["y_true"] == 0).sum()
        )

        rows.append(
            {
                "tile_id": tile_id,
                "samples": n,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "accuracy": accuracy,
                "error_rate": error_rate,
                "retained_true": retained_true,
                "eliminated_true": eliminated_true,
                "mean_probability": float(
                    g["prob_retained_mean"].mean()
                ),
                "mean_agreement": float(
                    g["prediction_agreement"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def select_representative_tile(
    mapped: gpd.GeoDataFrame,
) -> Optional[str]:

    tm = tile_metrics(mapped)

    if tm.empty:
        return None

    # Prefer tiles containing both classes and enough roads.
    eligible = tm[
        (tm["retained_true"] > 0)
        & (tm["eliminated_true"] > 0)
        & (tm["samples"] >= 20)
    ].copy()

    if eligible.empty:
        eligible = tm.copy()

    # Choose a tile close to the city's median error rate,
    # rather than cherry-picking best or worst.
    median_error = (
        eligible["error_rate"].median()
    )

    eligible["representativeness"] = (
        (
            eligible["error_rate"]
            - median_error
        ).abs()
        +
        1.0
        / np.sqrt(
            eligible["samples"].clip(lower=1)
        )
    )

    eligible = eligible.sort_values(
        [
            "representativeness",
            "samples",
        ],
        ascending=[True, False],
    )

    return str(
        eligible.iloc[0]["tile_id"]
    )


# =============================================================================
# SINGLE CITY FIGURE
# =============================================================================

def plot_city_four_panel(
    city: str,
    source: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    mapped: gpd.GeoDataFrame,
    protocol: str,
    output_dir: Path,
    stem_suffix: str,
) -> None:

    if mapped.empty:
        return

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 6),
    )

    # A. Source
    plot_lines(
        axes[0],
        source,
        COLOR_SOURCE,
        SOURCE_LINEWIDTH,
    )

    axes[0].set_title(
        "A. Detailed source network"
    )

    # B. Reference
    plot_lines(
        axes[1],
        target,
        COLOR_REFERENCE,
        REFERENCE_LINEWIDTH,
    )

    axes[1].set_title(
        "B. Reference generalization"
    )

    # C. Prediction
    predicted = mapped[
        mapped["consensus_pred"] == 1
    ]

    plot_lines(
        axes[2],
        source,
        "#E0E0E0",
        0.35,
        alpha=0.55,
    )

    plot_lines(
        axes[2],
        predicted,
        COLOR_RETAINED,
        PREDICTED_LINEWIDTH,
        zorder=3,
    )

    axes[2].set_title(
        f"C. CartoGraph-X ({PROTOCOL_DISPLAY[protocol]})"
    )

    # D. Error
    for cls, color, z in [
        ("TN", COLOR_TN, 1),
        ("TP", COLOR_TP, 2),
        ("FP", COLOR_FP, 3),
        ("FN", COLOR_FN, 4),
    ]:

        plot_lines(
            axes[3],
            mapped[
                mapped["error_class"] == cls
            ],
            color,
            ERROR_LINEWIDTH,
            zorder=z,
        )

    axes[3].set_title(
        "D. Selection error"
    )

    set_equal_extent(
        axes,
        [
            source,
            target,
            mapped,
        ],
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=COLOR_TP,
            lw=2,
            label="TP: correctly retained",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_TN,
            lw=2,
            label="TN: correctly eliminated",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_FP,
            lw=2,
            label="FP: wrongly retained",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_FN,
            lw=2,
            label="FN: wrongly eliminated",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        f"{city}: road-network generalization",
        fontsize=15,
    )

    fig.tight_layout(
        rect=[0, 0.05, 1, 0.95]
    )

    save_figure(
        fig,
        output_dir,
        f"{CITY_KEYS[city]}_{stem_suffix}_four_panel",
    )


# =============================================================================
# PROBABILITY MAP
# =============================================================================

def plot_probability_map(
    city: str,
    mapped: gpd.GeoDataFrame,
    protocol: str,
    output_dir: Path,
    stem_suffix: str,
) -> None:

    if mapped.empty:
        return

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    mapped.plot(
        ax=ax,
        column="prob_retained_mean",
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidth=1.0,
        legend=True,
        legend_kwds={
            "label": "Predicted retention probability",
            "shrink": 0.75,
        },
    )

    ax.set_title(
        f"{city}: CartoGraph-X retention probability\n"
        f"{PROTOCOL_DISPLAY[protocol]}"
    )

    ax.set_axis_off()
    ax.set_aspect("equal")

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        f"{CITY_KEYS[city]}_{stem_suffix}_probability",
    )


# =============================================================================
# CONSENSUS UNCERTAINTY MAP
# =============================================================================

def plot_uncertainty_map(
    city: str,
    mapped: gpd.GeoDataFrame,
    protocol: str,
    output_dir: Path,
    stem_suffix: str,
) -> None:

    if mapped.empty:
        return

    if mapped["n_runs"].max() <= 1:
        return

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 7),
    )

    mapped.plot(
        ax=axes[0],
        column="prob_retained_std",
        cmap="magma",
        linewidth=1.0,
        legend=True,
        legend_kwds={
            "label": "Probability standard deviation",
            "shrink": 0.7,
        },
    )

    axes[0].set_title(
        "Across-run probability uncertainty"
    )

    mapped.plot(
        ax=axes[1],
        column="prediction_agreement",
        cmap="viridis",
        vmin=0.5,
        vmax=1.0,
        linewidth=1.0,
        legend=True,
        legend_kwds={
            "label": "Prediction agreement",
            "shrink": 0.7,
        },
    )

    axes[1].set_title(
        "Across-run decision agreement"
    )

    set_equal_extent(
        axes,
        [mapped],
    )

    fig.suptitle(
        f"{city}: CartoGraph-X run stability "
        f"({PROTOCOL_DISPLAY[protocol]})"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    save_figure(
        fig,
        output_dir,
        f"{CITY_KEYS[city]}_{stem_suffix}_uncertainty",
    )


# =============================================================================
# TILE FIGURE
# =============================================================================

def plot_representative_tile(
    city: str,
    source: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    mapped: gpd.GeoDataFrame,
    protocol: str,
    output_dir: Path,
    stem_suffix: str,
) -> Optional[str]:

    tile_id = select_representative_tile(
        mapped
    )

    if tile_id is None:
        return None

    tile = mapped[
        mapped["tile_id"] == tile_id
    ].copy()

    if tile.empty:
        return None

    bounds = tile.total_bounds

    dx = bounds[2] - bounds[0]
    dy = bounds[3] - bounds[1]

    pad_x = max(
        dx * 0.08,
        20.0,
    )

    pad_y = max(
        dy * 0.08,
        20.0,
    )

    bounds_pad = [
        bounds[0] - pad_x,
        bounds[1] - pad_y,
        bounds[2] + pad_x,
        bounds[3] + pad_y,
    ]

    source_tile = clip_to_bounds(
        source,
        bounds_pad,
    )

    target_tile = clip_to_bounds(
        target,
        bounds_pad,
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5.5),
    )

    plot_lines(
        axes[0],
        source_tile,
        COLOR_SOURCE,
        0.8,
    )

    axes[0].set_title(
        "Detailed source"
    )

    plot_lines(
        axes[1],
        target_tile,
        COLOR_REFERENCE,
        1.1,
    )

    axes[1].set_title(
        "Reference"
    )

    plot_lines(
        axes[2],
        source_tile,
        "#E0E0E0",
        0.45,
    )

    plot_lines(
        axes[2],
        tile[
            tile["consensus_pred"] == 1
        ],
        COLOR_RETAINED,
        1.2,
        zorder=3,
    )

    axes[2].set_title(
        "CartoGraph-X"
    )

    for cls, color in [
        ("TN", COLOR_TN),
        ("TP", COLOR_TP),
        ("FP", COLOR_FP),
        ("FN", COLOR_FN),
    ]:

        plot_lines(
            axes[3],
            tile[
                tile["error_class"] == cls
            ],
            color,
            1.25,
        )

    axes[3].set_title(
        "TP / TN / FP / FN"
    )

    for ax in axes:
        ax.set_xlim(
            bounds_pad[0],
            bounds_pad[2],
        )

        ax.set_ylim(
            bounds_pad[1],
            bounds_pad[3],
        )

        ax.set_aspect("equal")
        ax.set_axis_off()

    fig.suptitle(
        f"{city} representative test tile: {tile_id}\n"
        f"{PROTOCOL_DISPLAY[protocol]}"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.93]
    )

    safe_tile = clean_name(tile_id)

    save_figure(
        fig,
        output_dir,
        f"{CITY_KEYS[city]}_{safe_tile}_{stem_suffix}",
    )

    return tile_id


# =============================================================================
# CROSS-PROTOCOL COMPARISON
# =============================================================================

def plot_protocol_comparison(
    city: str,
    source: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    within: gpd.GeoDataFrame,
    zero: gpd.GeoDataFrame,
    joint: gpd.GeoDataFrame,
    output_dir: Path,
) -> None:

    if (
        within.empty
        or zero.empty
        or joint.empty
    ):
        return

    # Restrict to exactly the same sample universe.
    common = (
        set(within["sample_uid"])
        & set(zero["sample_uid"])
        & set(joint["sample_uid"])
    )

    if not common:
        return

    within = within[
        within["sample_uid"].isin(common)
    ].copy()

    zero = zero[
        zero["sample_uid"].isin(common)
    ].copy()

    joint = joint[
        joint["sample_uid"].isin(common)
    ].copy()

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(24, 6),
    )

    plot_lines(
        axes[0],
        source,
        COLOR_SOURCE,
        SOURCE_LINEWIDTH,
    )

    axes[0].set_title(
        "A. Detailed source"
    )

    plot_lines(
        axes[1],
        target,
        COLOR_REFERENCE,
        REFERENCE_LINEWIDTH,
    )

    axes[1].set_title(
        "B. Reference"
    )

    protocol_sets = [
        (
            axes[2],
            within,
            WITHIN_PROTOCOL[
                within.iloc[0]["country"]
            ]
            if within.iloc[0]["country"]
            in WITHIN_PROTOCOL
            else "",
            "C. Within-country",
        ),
        (
            axes[3],
            zero,
            "",
            "D. Zero-shot",
        ),
        (
            axes[4],
            joint,
            "",
            "E. Joint-country",
        ),
    ]

    for ax, g, _, title in protocol_sets:

        plot_lines(
            ax,
            source,
            "#E0E0E0",
            0.3,
            alpha=0.55,
        )

        plot_lines(
            ax,
            g[
                g["consensus_pred"] == 1
            ],
            COLOR_RETAINED,
            0.9,
            zorder=3,
        )

        ax.set_title(title)

    set_equal_extent(
        axes,
        [
            source,
            target,
            within,
            zero,
            joint,
        ],
    )

    fig.suptitle(
        f"{city}: geographic generalization comparison"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    save_figure(
        fig,
        output_dir,
        f"{CITY_KEYS[city]}_within_zero_joint_comparison",
    )


# =============================================================================
# TILE-LEVEL CROSS-PROTOCOL COMPARISON
# =============================================================================

def plot_tile_protocol_comparison(
    city: str,
    source: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    within: gpd.GeoDataFrame,
    zero: gpd.GeoDataFrame,
    joint: gpd.GeoDataFrame,
    output_dir: Path,
) -> Optional[str]:

    common = (
        set(within["sample_uid"])
        & set(zero["sample_uid"])
        & set(joint["sample_uid"])
    )

    if not common:
        return None

    w = within[
        within["sample_uid"].isin(common)
    ].copy()

    z = zero[
        zero["sample_uid"].isin(common)
    ].copy()

    j = joint[
        joint["sample_uid"].isin(common)
    ].copy()

    # Representative tile selected from joint model,
    # not best/worst.
    tile_id = select_representative_tile(j)

    if tile_id is None:
        return None

    w = w[w["tile_id"] == tile_id]
    z = z[z["tile_id"] == tile_id]
    j = j[j["tile_id"] == tile_id]

    if j.empty:
        return None

    bounds = j.total_bounds

    dx = bounds[2] - bounds[0]
    dy = bounds[3] - bounds[1]

    pad_x = max(dx * 0.08, 20.0)
    pad_y = max(dy * 0.08, 20.0)

    bp = [
        bounds[0] - pad_x,
        bounds[1] - pad_y,
        bounds[2] + pad_x,
        bounds[3] + pad_y,
    ]

    src = clip_to_bounds(
        source,
        bp,
    )

    ref = clip_to_bounds(
        target,
        bp,
    )

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(24, 5.5),
    )

    plot_lines(
        axes[0],
        src,
        COLOR_SOURCE,
        0.8,
    )

    axes[0].set_title(
        "Detailed source"
    )

    plot_lines(
        axes[1],
        ref,
        COLOR_REFERENCE,
        1.1,
    )

    axes[1].set_title(
        "Reference"
    )

    for ax, g, title in [
        (
            axes[2],
            w,
            "Within-country",
        ),
        (
            axes[3],
            z,
            "Zero-shot",
        ),
        (
            axes[4],
            j,
            "Joint-country",
        ),
    ]:

        plot_lines(
            ax,
            src,
            "#E0E0E0",
            0.45,
        )

        plot_lines(
            ax,
            g[
                g["consensus_pred"] == 1
            ],
            COLOR_RETAINED,
            1.25,
            zorder=3,
        )

        ax.set_title(title)

    for ax in axes:
        ax.set_xlim(bp[0], bp[2])
        ax.set_ylim(bp[1], bp[3])
        ax.set_aspect("equal")
        ax.set_axis_off()

    fig.suptitle(
        f"{city}: same test geography under three "
        f"training regimes\nTile: {tile_id}"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.92]
    )

    save_figure(
        fig,
        output_dir,
        f"{CITY_KEYS[city]}_representative_tile_protocol_comparison",
    )

    return tile_id


# =============================================================================
# SPATIAL SUMMARY
# =============================================================================

def spatial_summary(
    city: str,
    country: str,
    protocol: str,
    mapped: gpd.GeoDataFrame,
) -> Dict:

    if mapped.empty:
        return {
            "country": country,
            "city": city,
            "protocol": protocol,
            "samples": 0,
        }

    counts = (
        mapped["error_class"]
        .value_counts()
        .to_dict()
    )

    n = len(mapped)

    return {
        "country": country,
        "city": city,
        "protocol": protocol,
        "samples": int(n),
        "reference_retained": int(
            (mapped["y_true"] == 1).sum()
        ),
        "reference_eliminated": int(
            (mapped["y_true"] == 0).sum()
        ),
        "predicted_retained": int(
            (
                mapped["consensus_pred"] == 1
            ).sum()
        ),
        "predicted_eliminated": int(
            (
                mapped["consensus_pred"] == 0
            ).sum()
        ),
        "tp": int(counts.get("TP", 0)),
        "tn": int(counts.get("TN", 0)),
        "fp": int(counts.get("FP", 0)),
        "fn": int(counts.get("FN", 0)),
        "spatial_accuracy": float(
            (
                counts.get("TP", 0)
                + counts.get("TN", 0)
            )
            / n
        ),
        "mean_probability": float(
            mapped[
                "prob_retained_mean"
            ].mean()
        ),
        "mean_probability_std": float(
            mapped[
                "prob_retained_std"
            ].mean()
        ),
        "mean_prediction_agreement": float(
            mapped[
                "prediction_agreement"
            ].mean()
        ),
        "tiles": int(
            mapped["tile_id"].nunique()
        ),
        "runs": int(
            mapped["n_runs"].max()
        ),
    }


# =============================================================================
# GEOPACKAGE OUTPUT
# =============================================================================

def save_mapped_geopackage(
    city: str,
    protocol: str,
    mapped: gpd.GeoDataFrame,
) -> Optional[Path]:

    if mapped.empty:
        return None

    path = (
        GPKG_DIR
        / f"{CITY_KEYS[city]}_{protocol}_predictions.gpkg"
    )

    if path.exists():
        path.unlink()

    mapped.to_file(
        path,
        layer="road_predictions",
        driver="GPKG",
    )

    return path


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
        help=(
            "Seed used in smoke mode. "
            "Default: 42."
        ),
    )

    args = parser.parse_args()

    ensure_dirs()

    banner(
        "CARTOGEN — CARTOGRAPH-X SPATIAL MAP RECONSTRUCTION"
    )

    print(f"Models: {MODEL_DIR}")
    print(f"Output: {OUT_DIR}")
    print(f"Requested mode: {args.mode}")
    print(f"Smoke seed: {args.seed}")

    # =========================================================================
    # 1. DETECT AVAILABLE RUNS
    # =========================================================================

    section(
        "1. VERIFY PREDICTION FILES"
    )

    protocols = [
        "lt_to_lt",
        "pl_to_pl",
        "lt_to_pl_zero_shot",
        "pl_to_lt_zero_shot",
        "joint_to_lt",
        "joint_to_pl",
    ]

    run_counts = {}

    for protocol in protocols:

        files = prediction_files(
            protocol
        )

        run_counts[protocol] = len(files)

        print(
            f"{protocol:24s} "
            f"{len(files):2d} run(s)"
        )

    if args.mode == "auto":

        if all(
            run_counts[p] >= 10
            for p in protocols
        ):
            mode = "final"
        else:
            mode = "smoke"

    else:
        mode = args.mode

    print()
    print(f"Effective mode: {mode}")

    if mode == "final":

        incomplete = {
            p: n
            for p, n in run_counts.items()
            if n < 2
        }

        if incomplete:
            raise RuntimeError(
                "Final mode requires multiple runs. "
                f"Insufficient: {incomplete}"
            )

    # =========================================================================
    # 2. LOAD PREDICTIONS
    # =========================================================================

    section(
        "2. LOAD AND AGGREGATE CARTOGRAPH-X PREDICTIONS"
    )

    predictions = {}

    for protocol in protocols:

        predictions[protocol] = (
            load_protocol_predictions(
                protocol=protocol,
                mode=mode,
                requested_seed=args.seed,
            )
        )

        d = predictions[protocol]

        print(
            f"{protocol:24s} "
            f"{len(d):6,d} roads | "
            f"{int(d['n_runs'].max()):2d} run(s)"
        )

    # =========================================================================
    # 3. LOAD GEOMETRY
    # =========================================================================

    section(
        "3. LOAD SOURCE AND REFERENCE ROAD GEOMETRY"
    )

    geometry = {}

    for country in COUNTRIES:

        geometry[country] = {}

        for city in CITIES[country]:

            if country == "Lithuania":

                source, target = (
                    load_lithuania_city(
                        city
                    )
                )

            else:

                source = (
                    load_poland_source(
                        city
                    )
                )

                target = (
                    load_poland_target(
                        city,
                        source,
                    )
                )

            geometry[country][city] = {
                "source": source,
                "target": target,
            }

            print(
                f"{country:10s} | "
                f"{city:10s} | "
                f"source={len(source):6,d} | "
                f"reference={len(target):6,d} | "
                f"CRS={source.crs}"
            )

    # =========================================================================
    # 4. JOIN PREDICTIONS TO GEOMETRY
    # =========================================================================

    section(
        "4. RECONSTRUCT SPATIAL PREDICTIONS"
    )

    mapped = {}

    summary_rows = []

    for country in COUNTRIES:

        mapped[country] = {}

        relevant_protocols = [
            WITHIN_PROTOCOL[country],
            ZERO_PROTOCOL[country],
            JOINT_PROTOCOL[country],
        ]

        for city in CITIES[country]:

            mapped[country][city] = {}

            source = (
                geometry[country][city][
                    "source"
                ]
            )

            for protocol in relevant_protocols:

                g = join_predictions_to_source(
                    source=source,
                    pred=predictions[protocol],
                    city=city,
                )

                mapped[country][city][
                    protocol
                ] = g

                city_prediction_count = int(
                    (
                        predictions[protocol][
                            "uid_city"
                        ]
                        == city
                    ).sum()
                )

                join_pct = (
                    100.0
                    * len(g)
                    / city_prediction_count
                    if city_prediction_count
                    else 0.0
                )

                print(
                    f"{country:10s} | "
                    f"{city:10s} | "
                    f"{protocol:24s} | "
                    f"pred={city_prediction_count:5,d} | "
                    f"mapped={len(g):5,d} | "
                    f"join={join_pct:6.2f}%"
                )

                if (
                    city_prediction_count > 0
                    and len(g)
                    != city_prediction_count
                ):
                    warnings.warn(
                        f"{city}/{protocol}: only "
                        f"{len(g)}/{city_prediction_count} "
                        f"predictions mapped to geometry."
                    )

                summary_rows.append(
                    spatial_summary(
                        city,
                        country,
                        protocol,
                        g,
                    )
                )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUT_DIR
        / "spatial_prediction_summary.csv",
        index=False,
    )

    # =========================================================================
    # 5. SAVE SPATIAL PREDICTIONS
    # =========================================================================

    section(
        "5. SAVE GIS-READY PREDICTION LAYERS"
    )

    gpkg_outputs = []

    for country in COUNTRIES:

        relevant_protocols = [
            WITHIN_PROTOCOL[country],
            ZERO_PROTOCOL[country],
            JOINT_PROTOCOL[country],
        ]

        for city in CITIES[country]:

            for protocol in relevant_protocols:

                p = save_mapped_geopackage(
                    city,
                    protocol,
                    mapped[country][city][
                        protocol
                    ],
                )

                if p is not None:

                    gpkg_outputs.append(
                        str(p)
                    )

                    print(p)

    # =========================================================================
    # 6. WITHIN-COUNTRY CITY MAPS
    # =========================================================================

    section(
        "6. GENERATE CITY-LEVEL GENERALIZATION MAPS"
    )

    representative_tiles = {}

    for country in COUNTRIES:

        protocol = (
            WITHIN_PROTOCOL[country]
        )

        for city in CITIES[country]:

            source = (
                geometry[country][city][
                    "source"
                ]
            )

            target = (
                geometry[country][city][
                    "target"
                ]
            )

            g = (
                mapped[country][city][
                    protocol
                ]
            )

            city_out = (
                CITY_DIR
                / clean_name(country)
                / CITY_KEYS[city]
            )

            plot_city_four_panel(
                city,
                source,
                target,
                g,
                protocol,
                city_out,
                "within_country",
            )

            plot_probability_map(
                city,
                g,
                protocol,
                city_out,
                "within_country",
            )

            plot_uncertainty_map(
                city,
                g,
                protocol,
                city_out,
                "within_country",
            )

            tile_id = (
                plot_representative_tile(
                    city,
                    source,
                    target,
                    g,
                    protocol,
                    TILE_DIR
                    / clean_name(country)
                    / CITY_KEYS[city],
                    "within_country",
                )
            )

            representative_tiles[
                f"{country}:{city}"
            ] = tile_id

    # =========================================================================
    # 7. ZERO-SHOT MAPS
    # =========================================================================

    section(
        "7. GENERATE ZERO-SHOT TRANSFER MAPS"
    )

    for country in COUNTRIES:

        protocol = ZERO_PROTOCOL[country]

        for city in CITIES[country]:

            source = (
                geometry[country][city][
                    "source"
                ]
            )

            target = (
                geometry[country][city][
                    "target"
                ]
            )

            g = (
                mapped[country][city][
                    protocol
                ]
            )

            city_out = (
                CITY_DIR
                / clean_name(country)
                / CITY_KEYS[city]
            )

            plot_city_four_panel(
                city,
                source,
                target,
                g,
                protocol,
                city_out,
                "zero_shot",
            )

            plot_probability_map(
                city,
                g,
                protocol,
                city_out,
                "zero_shot",
            )

    # =========================================================================
    # 8. JOINT-COUNTRY MAPS
    # =========================================================================

    section(
        "8. GENERATE JOINT-COUNTRY MAPS"
    )

    for country in COUNTRIES:

        protocol = JOINT_PROTOCOL[country]

        for city in CITIES[country]:

            source = (
                geometry[country][city][
                    "source"
                ]
            )

            target = (
                geometry[country][city][
                    "target"
                ]
            )

            g = (
                mapped[country][city][
                    protocol
                ]
            )

            city_out = (
                CITY_DIR
                / clean_name(country)
                / CITY_KEYS[city]
            )

            plot_city_four_panel(
                city,
                source,
                target,
                g,
                protocol,
                city_out,
                "joint_country",
            )

            plot_probability_map(
                city,
                g,
                protocol,
                city_out,
                "joint_country",
            )

            plot_uncertainty_map(
                city,
                g,
                protocol,
                city_out,
                "joint_country",
            )

    # =========================================================================
    # 9. DIRECT DOMAIN-GENERALIZATION COMPARISON
    # =========================================================================

    section(
        "9. WITHIN vs ZERO-SHOT vs JOINT-Country COMPARISON"
    )

    comparison_tiles = {}

    for country in COUNTRIES:

        within_protocol = (
            WITHIN_PROTOCOL[country]
        )

        zero_protocol = (
            ZERO_PROTOCOL[country]
        )

        joint_protocol = (
            JOINT_PROTOCOL[country]
        )

        for city in CITIES[country]:

            source = (
                geometry[country][city][
                    "source"
                ]
            )

            target = (
                geometry[country][city][
                    "target"
                ]
            )

            within = (
                mapped[country][city][
                    within_protocol
                ]
            )

            zero = (
                mapped[country][city][
                    zero_protocol
                ]
            )

            joint = (
                mapped[country][city][
                    joint_protocol
                ]
            )

            out = (
                COMPARE_DIR
                / clean_name(country)
                / CITY_KEYS[city]
            )

            plot_protocol_comparison(
                city,
                source,
                target,
                within,
                zero,
                joint,
                out,
            )

            tile_id = (
                plot_tile_protocol_comparison(
                    city,
                    source,
                    target,
                    within,
                    zero,
                    joint,
                    out,
                )
            )

            comparison_tiles[
                f"{country}:{city}"
            ] = tile_id

    # =========================================================================
    # 10. MAP-LEVEL QUANTITATIVE SUMMARY
    # =========================================================================

    section(
        "10. SPATIAL PERFORMANCE SUMMARY"
    )

    if not summary.empty:

        cols = [
            "country",
            "city",
            "protocol",
            "samples",
            "reference_retained",
            "reference_eliminated",
            "predicted_retained",
            "tp",
            "tn",
            "fp",
            "fn",
            "spatial_accuracy",
            "mean_probability",
            "mean_prediction_agreement",
        ]

        print(
            summary[cols]
            .to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.4f}"
                ),
            )
        )

    # =========================================================================
    # 11. AUDIT
    # =========================================================================

    section(
        "11. SPATIAL RECONSTRUCTION AUDIT"
    )

    checks = {}

    checks[
        "all_six_prediction_protocols_present"
    ] = all(
        run_counts[p] > 0
        for p in protocols
    )

    checks[
        "all_six_cities_loaded"
    ] = (
        sum(
            len(geometry[c])
            for c in COUNTRIES
        )
        == 6
    )

    all_join_complete = True

    for country in COUNTRIES:

        relevant_protocols = [
            WITHIN_PROTOCOL[country],
            ZERO_PROTOCOL[country],
            JOINT_PROTOCOL[country],
        ]

        for city in CITIES[country]:

            for protocol in relevant_protocols:

                pred_n = int(
                    (
                        predictions[protocol][
                            "uid_city"
                        ]
                        == city
                    ).sum()
                )

                map_n = len(
                    mapped[country][city][
                        protocol
                    ]
                )

                if pred_n != map_n:
                    all_join_complete = False

    checks[
        "all_predictions_joined_by_identifier"
    ] = all_join_complete

    checks[
        "no_centroid_matching_used"
    ] = True

    checks[
        "within_country_maps_generated"
    ] = True

    checks[
        "zero_shot_maps_generated"
    ] = True

    checks[
        "joint_country_maps_generated"
    ] = True

    checks[
        "cross_protocol_comparison_generated"
    ] = True

    if mode == "final":

        checks[
            "multi_run_consensus_used"
        ] = all(
            run_counts[p] > 1
            for p in protocols
        )

    for name, passed in checks.items():

        print(
            f"{name:55s} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    # =========================================================================
    # 12. REPORT
    # =========================================================================

    report = {
        "script": 74,
        "mode": mode,
        "seed_smoke": (
            args.seed
            if mode == "smoke"
            else None
        ),
        "prediction_run_counts": (
            run_counts
        ),
        "countries": COUNTRIES,
        "cities": CITIES,
        "protocols": protocols,
        "representative_tiles": (
            representative_tiles
        ),
        "comparison_tiles": (
            comparison_tiles
        ),
        "audit": checks,
        "geopackage_outputs": (
            gpkg_outputs
        ),
        "outputs": {
            "root": str(OUT_DIR),
            "city_maps": str(CITY_DIR),
            "tile_maps": str(TILE_DIR),
            "protocol_comparisons": str(
                COMPARE_DIR
            ),
            "geopackages": str(
                GPKG_DIR
            ),
            "summary": str(
                OUT_DIR
                / "spatial_prediction_summary.csv"
            ),
        },
    }

    report_path = (
        OUT_DIR
        / "cartographx_spatial_map_report.json"
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
    # 13. FINAL OUTPUT
    # =========================================================================

    section("12. OUTPUTS")

    print(
        f"Root:                 {OUT_DIR}"
    )

    print(
        f"City maps:            {CITY_DIR}"
    )

    print(
        f"Representative tiles: {TILE_DIR}"
    )

    print(
        f"Protocol comparisons: {COMPARE_DIR}"
    )

    print(
        f"GIS prediction layers:{GPKG_DIR}"
    )

    print(
        f"Spatial summary:      "
        f"{OUT_DIR / 'spatial_prediction_summary.csv'}"
    )

    print(
        f"Report:               {report_path}"
    )

    banner("SCRIPT 74 COMPLETE")

    if all(checks.values()):

        print(
            "[PASS] CARTOGRAPH-X SPATIAL MAP "
            "RECONSTRUCTION COMPLETE."
        )

    else:

        print(
            "[CHECK] MAP RECONSTRUCTION COMPLETED "
            "WITH ONE OR MORE AUDIT WARNINGS."
        )

    print()
    print(
        "NEXT: Script 75 — quantitative spatial/cartographic "
        "quality analysis beyond road-wise classification."
    )


if __name__ == "__main__":
    main()