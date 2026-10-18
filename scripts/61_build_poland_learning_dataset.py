from pathlib import Path
import json
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.strtree import STRtree

warnings.filterwarnings("ignore")

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

ROOT = Path(__file__).resolve().parents[1]
POLAND = ROOT / "data" / "processed" / "poland"

VALIDATED_DIR = POLAND / "road_target_validation"
CORRESPONDENCE_DIR = POLAND / "road_correspondence"
TOPOLOGY_DIR = POLAND / "road_topology"

OUTPUT_DIR = POLAND / "learning_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CITIES = {
    "warsaw": "Warsaw",
    "krakow": "Krakow",
    "gdansk": "Gdansk",
}

EXPECTED_EPSG = 2180
NEIGHBOR_RADIUS_M = 150.0


# ==================================================================================================
# PRINTING
# ==================================================================================================

def banner(text):
    print()
    print("=" * 120)
    print(text)
    print("=" * 120)


def section(text):
    print()
    print("-" * 120)
    print(text)
    print("-" * 120)


# ==================================================================================================
# HELPERS
# ==================================================================================================

def safe_div(a, b, default=0.0):
    try:
        b = float(b)
        if b == 0:
            return default
        return float(a) / b
    except Exception:
        return default


def normalize_geometry(gdf):
    gdf = gdf.copy()

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    invalid = ~gdf.geometry.is_valid

    if invalid.any():
        gdf.loc[invalid, "geometry"] = (
            gdf.loc[invalid, "geometry"].buffer(0)
        )

    gdf = gdf[
        gdf.geometry.notna()
        & (~gdf.geometry.is_empty)
        & gdf.geometry.is_valid
    ].copy()

    return gdf.reset_index(drop=True)


def inspect_layers(path):
    return list(gpd.list_layers(path)["name"])


def read_source_roads(path):
    layers = inspect_layers(path)

    preferred = [
        "source_roads",
        "bdot10k_roads",
        "source",
    ]

    for layer in preferred:
        if layer in layers:
            return gpd.read_file(path, layer=layer), layer

    # Try identifying by name
    candidates = [
        x for x in layers
        if "source" in x.lower()
        and "road" in x.lower()
    ]

    if len(candidates) == 1:
        return gpd.read_file(path, layer=candidates[0]), candidates[0]

    raise RuntimeError(
        f"Could not identify source road layer in:\n{path}\n"
        f"Available layers: {layers}"
    )


# ==================================================================================================
# GEOMETRIC FEATURES
# ==================================================================================================

def extract_geometric_features(gdf):
    section(f"Extracting geometric features for {len(gdf):,} roads...")

    rows = []

    for i, geom in enumerate(gdf.geometry):
        length = float(geom.length)

        centroid = geom.centroid

        minx, miny, maxx, maxy = geom.bounds

        bbox_width = maxx - minx
        bbox_height = maxy - miny
        bbox_diagonal = np.hypot(
            bbox_width,
            bbox_height,
        )

        # ------------------------------------------------------------------------------------------
        # Vertices
        # ------------------------------------------------------------------------------------------

        try:
            coords = list(geom.coords)
        except Exception:
            coords = []

        vertex_count = len(coords)

        # ------------------------------------------------------------------------------------------
        # Endpoint distance / sinuosity
        # ------------------------------------------------------------------------------------------

        if len(coords) >= 2:
            x1, y1 = coords[0][:2]
            x2, y2 = coords[-1][:2]

            endpoint_distance = float(
                np.hypot(
                    x2 - x1,
                    y2 - y1,
                )
            )
        else:
            endpoint_distance = 0.0

        if endpoint_distance > 0:
            sinuosity = length / endpoint_distance
        else:
            sinuosity = 1.0

        # ------------------------------------------------------------------------------------------
        # Orientation
        # ------------------------------------------------------------------------------------------

        if len(coords) >= 2:
            dx = coords[-1][0] - coords[0][0]
            dy = coords[-1][1] - coords[0][1]

            orientation = (
                np.degrees(np.arctan2(dy, dx)) + 180.0
            ) % 180.0
        else:
            orientation = 0.0

        # ------------------------------------------------------------------------------------------
        # Segment statistics
        # ------------------------------------------------------------------------------------------

        segment_lengths = []

        if len(coords) >= 2:
            for a, b in zip(coords[:-1], coords[1:]):
                segment_lengths.append(
                    np.hypot(
                        b[0] - a[0],
                        b[1] - a[1],
                    )
                )

        if segment_lengths:
            mean_segment_length = float(
                np.mean(segment_lengths)
            )
            min_segment_length = float(
                np.min(segment_lengths)
            )
            max_segment_length = float(
                np.max(segment_lengths)
            )
        else:
            mean_segment_length = 0.0
            min_segment_length = 0.0
            max_segment_length = 0.0

        # ------------------------------------------------------------------------------------------
        # Curvature proxy
        # ------------------------------------------------------------------------------------------

        angle_changes = []

        if len(coords) >= 3:
            for j in range(1, len(coords) - 1):
                p0 = np.array(coords[j - 1][:2], dtype=float)
                p1 = np.array(coords[j][:2], dtype=float)
                p2 = np.array(coords[j + 1][:2], dtype=float)

                v1 = p1 - p0
                v2 = p2 - p1

                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)

                if n1 > 0 and n2 > 0:
                    cosine = np.dot(v1, v2) / (n1 * n2)
                    cosine = np.clip(cosine, -1.0, 1.0)

                    angle = np.degrees(
                        np.arccos(cosine)
                    )

                    angle_changes.append(
                        abs(180.0 - angle)
                    )

        if angle_changes:
            mean_angle_change = float(
                np.mean(angle_changes)
            )
            max_angle_change = float(
                np.max(angle_changes)
            )
        else:
            mean_angle_change = 0.0
            max_angle_change = 0.0

        rows.append(
            {
                "source_idx": i,
                "length_m": length,
                "vertex_count": vertex_count,
                "endpoint_distance_m": endpoint_distance,
                "sinuosity": sinuosity,
                "orientation_deg": orientation,
                "bbox_width_m": bbox_width,
                "bbox_height_m": bbox_height,
                "bbox_diagonal_m": bbox_diagonal,
                "mean_segment_length_m": mean_segment_length,
                "min_segment_length_m": min_segment_length,
                "max_segment_length_m": max_segment_length,
                "mean_angle_change_deg": mean_angle_change,
                "max_angle_change_deg": max_angle_change,
                "centroid_x": centroid.x,
                "centroid_y": centroid.y,
            }
        )

    return pd.DataFrame(rows)


# ==================================================================================================
# NEIGHBORHOOD FEATURES
# ==================================================================================================

def extract_neighborhood_features(gdf):
    section("Computing road neighborhood context...")

    geoms = list(gdf.geometry)
    tree = STRtree(geoms)

    rows = []

    for i, geom in enumerate(geoms):
        search_geom = geom.buffer(
            NEIGHBOR_RADIUS_M
        )

        candidate_indices = tree.query(
            search_geom
        )

        neighbor_count = 0
        intersecting_count = 0

        neighbor_lengths = []
        distances = []

        for candidate in candidate_indices:
            j = int(candidate)

            if j == i:
                continue

            other = geoms[j]

            distance = float(
                geom.distance(other)
            )

            if distance <= NEIGHBOR_RADIUS_M:
                neighbor_count += 1

                neighbor_lengths.append(
                    other.length
                )

                distances.append(
                    distance
                )

                if geom.intersects(other):
                    intersecting_count += 1

        if neighbor_lengths:
            mean_neighbor_length = float(
                np.mean(neighbor_lengths)
            )
        else:
            mean_neighbor_length = 0.0

        if distances:
            nearest_distance = float(
                np.min(distances)
            )

            mean_distance = float(
                np.mean(distances)
            )
        else:
            nearest_distance = NEIGHBOR_RADIUS_M
            mean_distance = NEIGHBOR_RADIUS_M

        rows.append(
            {
                "neighbor_count_150m": neighbor_count,
                "intersecting_road_count": intersecting_count,
                "mean_neighbor_length_m": mean_neighbor_length,
                "nearest_neighbor_distance_m": nearest_distance,
                "mean_neighbor_distance_m": mean_distance,
            }
        )

        if (
            (i + 1) % 1000 == 0
            or i + 1 == len(geoms)
        ):
            print(
                f"  {i + 1:,}/{len(geoms):,}"
            )

    return pd.DataFrame(rows)


# ==================================================================================================
# CITY PROCESSING
# ==================================================================================================

def process_city(city_key, city_name):
    banner(f"{city_key.upper()} ({city_name})")

    validated_path = (
        VALIDATED_DIR
        / f"{city_key}_validated_road_pair.gpkg"
    )

    labels_path = (
        CORRESPONDENCE_DIR
        / f"{city_key}_source_road_labels.csv"
    )

    correspondence_path = (
        CORRESPONDENCE_DIR
        / f"{city_key}_road_correspondence_pairs.csv"
    )

    operations_path = (
        TOPOLOGY_DIR
        / f"{city_key}_road_operations.csv"
    )

    for path in [
        validated_path,
        labels_path,
        correspondence_path,
        operations_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    # ==============================================================================================
    # READ SOURCE GEOMETRY
    # ==============================================================================================

    section("READING VALIDATED SOURCE ROADS")

    source, source_layer = read_source_roads(
        validated_path
    )

    source = normalize_geometry(source)

    if source.crs is None:
        raise RuntimeError(
            f"No CRS in {validated_path}"
        )

    if source.crs.to_epsg() != EXPECTED_EPSG:
        source = source.to_crs(
            epsg=EXPECTED_EPSG
        )

    print(f"Layer:        {source_layer}")
    print(f"Source roads: {len(source):,}")
    print(f"CRS:          {source.crs}")

    # ==============================================================================================
    # READ SUPERVISION
    # ==============================================================================================

    section("READING ROAD SUPERVISION")

    labels = pd.read_csv(labels_path)
    correspondence = pd.read_csv(
        correspondence_path
    )
    operations = pd.read_csv(
        operations_path
    )

    print(f"Source labels:        {len(labels):,}")
    print(f"Accepted pairs:       {len(correspondence):,}")
    print(f"Target operations:    {len(operations):,}")

    required_label_cols = {
        "source_idx",
        "target_idx",
        "selection_label",
        "source_coverage",
        "target_coverage",
        "distance_m",
        "match_score",
    }

    missing = required_label_cols - set(
        labels.columns
    )

    if missing:
        raise RuntimeError(
            f"Missing source-label columns: {missing}"
        )

    required_operation_cols = {
        "target_id",
        "source_count",
        "operation",
        "source_total_length_m",
        "target_length_m",
        "length_ratio",
        "geometry_distance_m",
        "source_vertices",
        "target_vertices",
        "vertex_ratio",
    }

    missing = required_operation_cols - set(
        operations.columns
    )

    if missing:
        raise RuntimeError(
            f"Missing operation columns: {missing}"
        )

    # ==============================================================================================
    # INDEX VALIDATION
    # ==============================================================================================

    section("VALIDATING SOURCE INDEX ALIGNMENT")

    labels = labels.sort_values(
        "source_idx"
    ).reset_index(drop=True)

    expected_indices = np.arange(
        len(source)
    )

    actual_indices = labels[
        "source_idx"
    ].to_numpy(dtype=int)

    print(
        f"Geometry rows: {len(source):,}"
    )
    print(
        f"Label rows:    {len(labels):,}"
    )

    if len(source) != len(labels):
        raise RuntimeError(
            "Source geometry count does not match "
            "source label count."
        )

    if not np.array_equal(
        expected_indices,
        actual_indices,
    ):
        raise RuntimeError(
            "source_idx is not a complete sequential "
            "0..N-1 mapping. Do not continue."
        )

    print("[PASS] source_idx aligns with validated geometry.")

    # ==============================================================================================
    # INPUT FEATURES
    # ==============================================================================================

    geometric = extract_geometric_features(
        source
    )

    neighborhood = extract_neighborhood_features(
        source
    )

    features = geometric.merge(
        neighborhood,
        left_index=True,
        right_index=True,
        how="left",
    )

    # ==============================================================================================
    # ATTACH SOURCE-LEVEL LABELS
    # ==============================================================================================

    section("ATTACHING SELECTION / ELIMINATION LABELS")

    learning = features.merge(
        labels,
        on="source_idx",
        how="left",
        validate="one_to_one",
    )

    if learning["selection_label"].isna().any():
        raise RuntimeError(
            "Missing source selection labels after merge."
        )

    # ==============================================================================================
    # ATTACH TARGET-LEVEL OPERATIONS
    # ==============================================================================================

    section("ATTACHING TARGET GENERALIZATION OPERATIONS")

    operation_columns = [
        "target_id",
        "source_count",
        "operation",
        "source_total_length_m",
        "target_length_m",
        "length_ratio",
        "geometry_distance_m",
        "source_vertices",
        "target_vertices",
        "vertex_ratio",
    ]

    operation_lookup = operations[
        operation_columns
    ].copy()

    operation_lookup = operation_lookup.rename(
        columns={
            "target_id": "target_idx",
            "source_count": "target_source_count",
            "operation": "target_operation",
            "source_total_length_m":
                "operation_source_total_length_m",
            "target_length_m":
                "operation_target_length_m",
            "length_ratio":
                "operation_length_ratio",
            "geometry_distance_m":
                "operation_geometry_distance_m",
            "source_vertices":
                "operation_source_vertices",
            "target_vertices":
                "operation_target_vertices",
            "vertex_ratio":
                "operation_vertex_ratio",
        }
    )

    if operation_lookup[
        "target_idx"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate target_idx values in operation table."
        )

    learning = learning.merge(
        operation_lookup,
        on="target_idx",
        how="left",
        validate="many_to_one",
    )

    # Elimination roads have no target operation.
    learning["target_operation"] = (
        learning["target_operation"]
        .fillna("none_eliminated")
    )

    # ==============================================================================================
    # HIERARCHICAL SUPERVISION
    # ==============================================================================================

    section("BUILDING HIERARCHICAL ROAD LABELS")

    learning["is_selected"] = (
        learning["selection_label"]
        .isin(["selected", "weak_selected"])
    ).astype(int)

    learning["is_eliminated"] = (
        learning["selection_label"]
        == "elimination"
    ).astype(int)

    # Final operation label:
    #
    # elimination
    # merge_or_chain
    # displacement_or_reshape
    # retained_or_simplified
    # etc.
    #
    # Selection is the first decision.
    # For selected roads, target_operation is the
    # second-level generalization decision.

    learning["operation_label"] = np.where(
        learning["selection_label"].eq("elimination"),
        "elimination",
        learning["target_operation"],
    )

    # ==============================================================================================
    # CONFIDENCE CATEGORIES
    # ==============================================================================================

    learning["match_confidence"] = "not_applicable"

    selected_mask = learning["selection_label"].isin(
        ["selected", "weak_selected"]
    )

    learning.loc[
        selected_mask
        & (learning["match_score"] >= 0.80),
        "match_confidence",
    ] = "strong"

    learning.loc[
        selected_mask
        & (learning["match_score"] >= 0.65)
        & (learning["match_score"] < 0.80),
        "match_confidence",
    ] = "moderate"

    learning.loc[
        selected_mask
        & (learning["match_score"] < 0.65),
        "match_confidence",
    ] = "weak"

    # ==============================================================================================
    # ADD METADATA
    # ==============================================================================================

    learning.insert(
        0,
        "country",
        "Poland",
    )

    learning.insert(
        1,
        "city",
        city_name,
    )

    learning.insert(
        2,
        "city_key",
        city_key,
    )

    learning.insert(
        3,
        "feature_family",
        "road",
    )

    learning.insert(
        4,
        "source_dataset",
        "BDOT10k",
    )

    learning.insert(
        5,
        "target_dataset",
        "BDOO",
    )

    # ==============================================================================================
    # DIAGNOSTICS
    # ==============================================================================================

    section("LABEL DISTRIBUTION")

    print("Selection:")
    print(
        learning[
            "selection_label"
        ].value_counts().to_string()
    )

    print()
    print("Final operation:")
    print(
        learning[
            "operation_label"
        ].value_counts().to_string()
    )

    print()
    print("Match confidence:")
    print(
        learning[
            "match_confidence"
        ].value_counts().to_string()
    )

    selected = learning[
        learning["selection_label"].isin(
            ["selected", "weak_selected"]
        )
    ]

    selected_without_operation = selected[
        selected["target_operation"]
        == "none_eliminated"
    ]

    print()
    print(
        "Retained roads without target operation: "
        f"{len(selected_without_operation):,}"
    )

    # This can be non-zero if Script 60 deliberately
    # excluded some target matches, but we report it.
    # Do not silently convert these to elimination.

    # ==============================================================================================
    # SAVE CSV
    # ==============================================================================================

    city_csv = (
        OUTPUT_DIR
        / f"{city_key}_road_learning_dataset.csv"
    )

    learning.to_csv(
        city_csv,
        index=False,
    )

    # ==============================================================================================
    # SAVE GEOSPATIAL VERSION
    # ==============================================================================================

    geo = source.copy()

    geo["source_idx"] = np.arange(
        len(geo)
    )

    # Avoid duplicate original attribute names
    # by adding only learning-specific columns.
    learning_columns = [
        c for c in learning.columns
        if c not in geo.columns
        or c == "source_idx"
    ]

    geo = geo.merge(
        learning[learning_columns],
        on="source_idx",
        how="left",
        validate="one_to_one",
    )

    gpkg_path = (
        OUTPUT_DIR
        / f"{city_key}_road_learning_dataset.gpkg"
    )

    if gpkg_path.exists():
        gpkg_path.unlink()

    geo.to_file(
        gpkg_path,
        layer="road_learning_samples",
        driver="GPKG",
    )

    # ==============================================================================================
    # CITY SUMMARY
    # ==============================================================================================

    summary = {
        "city": city_name,
        "city_key": city_key,
        "samples": int(len(learning)),
        "selected": int(
            learning["selection_label"].isin(
                ["selected", "weak_selected"]
            ).sum()
        ),
        "strong_selected": int(
            (learning["selection_label"] == "selected").sum()
        ),
        "weak_selected": int(
            (learning["selection_label"] == "weak_selected").sum()
        ),
        "eliminated": int(
            (
                learning["selection_label"]
                == "elimination"
            ).sum()
        ),
        "selection_rate_pct": float(
            learning["selection_label"].isin(
                ["selected", "weak_selected"]
            ).mean()
            * 100.0
        ),
        "elimination_rate_pct": float(
            (
                learning["selection_label"]
                == "elimination"
            ).mean()
            * 100.0
        ),
        "retained_without_target_operation": int(
            len(selected_without_operation)
        ),
        "operation_distribution": {
            str(k): int(v)
            for k, v in learning[
                "operation_label"
            ].value_counts().to_dict().items()
        },
        "csv": str(city_csv),
        "gpkg": str(gpkg_path),
    }

    print()
    print("Saved CSV:")
    print(city_csv)

    print()
    print("Saved GeoPackage:")
    print(gpkg_path)

    return learning, summary


# ==================================================================================================
# MAIN
# ==================================================================================================

def main():
    banner(
        "CARTOGEN — BUILD POLISH ROAD LEARNING DATASET"
    )

    datasets = []
    summaries = []

    for city_key, city_name in CITIES.items():
        learning, summary = process_city(
            city_key,
            city_name,
        )

        datasets.append(
            learning
        )

        summaries.append(
            summary
        )

    # ==============================================================================================
    # GLOBAL DATASET
    # ==============================================================================================

    banner("GLOBAL POLISH ROAD LEARNING DATASET")

    global_df = pd.concat(
        datasets,
        ignore_index=True,
    )

    global_csv = (
        OUTPUT_DIR
        / "poland_road_learning_dataset.csv"
    )

    global_df.to_csv(
        global_csv,
        index=False,
    )

    # ==============================================================================================
    # CITY SUMMARY
    # ==============================================================================================

    summary_df = pd.DataFrame(
        [
            {
                "city": x["city"],
                "samples": x["samples"],
                "selected": x["selected"],
                "strong_selected": x["strong_selected"],
                "weak_selected": x["weak_selected"],
                "eliminated": x["eliminated"],
                "selection_rate_pct":
                    x["selection_rate_pct"],
                "elimination_rate_pct":
                    x["elimination_rate_pct"],
                "retained_without_target_operation":
                    x[
                        "retained_without_target_operation"
                    ],
            }
            for x in summaries
        ]
    )

    summary_csv = (
        OUTPUT_DIR
        / "poland_road_learning_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    print()
    print(
        summary_df.to_string(
            index=False
        )
    )

    # ==============================================================================================
    # GLOBAL LABEL DISTRIBUTION
    # ==============================================================================================

    section("GLOBAL SELECTION DISTRIBUTION")

    selection_summary = (
        global_df[
            "selection_label"
        ]
        .value_counts()
        .rename_axis("selection_label")
        .reset_index(name="count")
    )

    selection_summary["percentage"] = (
        selection_summary["count"]
        / len(global_df)
        * 100.0
    )

    print(
        selection_summary.to_string(
            index=False
        )
    )

    section("GLOBAL OPERATION DISTRIBUTION")

    operation_summary = (
        global_df[
            "operation_label"
        ]
        .value_counts()
        .rename_axis("operation_label")
        .reset_index(name="count")
    )

    operation_summary["percentage"] = (
        operation_summary["count"]
        / len(global_df)
        * 100.0
    )

    print(
        operation_summary.to_string(
            index=False
        )
    )

    selection_csv = (
        OUTPUT_DIR
        / "poland_selection_distribution.csv"
    )

    operation_csv = (
        OUTPUT_DIR
        / "poland_operation_distribution.csv"
    )

    selection_summary.to_csv(
        selection_csv,
        index=False,
    )

    operation_summary.to_csv(
        operation_csv,
        index=False,
    )

    # ==============================================================================================
    # VALIDATION
    # ==============================================================================================

    section("FINAL VALIDATION")

    expected_total = sum(
        x["samples"]
        for x in summaries
    )

    checks = {
        "global_count_correct":
            len(global_df) == expected_total,

        "three_cities_present":
            global_df["city"].nunique() == 3,

        "source_indices_present":
            global_df["source_idx"].notna().all(),

        "selection_labels_complete":
            global_df[
                "selection_label"
            ].notna().all(),

        "selection_labels_valid":
            set(
                global_df[
                    "selection_label"
                ].unique()
            ).issubset(
                {
                    "selected",
                    "weak_selected",
                    "elimination",
                }
            ),

        "binary_selection_consistent":
            (
                global_df["is_selected"]
                == global_df["selection_label"].isin(
                    ["selected", "weak_selected"]
                ).astype(int)
            ).all(),

        "binary_elimination_consistent":
            (
                global_df["is_eliminated"]
                == global_df["selection_label"].eq(
                    "elimination"
                ).astype(int)
            ).all(),

        "selection_partition_complete":
            (
                global_df["is_selected"]
                + global_df["is_eliminated"]
                == 1
            ).all(),

        "retained_have_target_operation":
            (
                global_df.loc[
                    global_df["is_selected"].eq(1),
                    "target_operation",
                ] != "none_eliminated"
            ).all(),

        "eliminated_operation_consistent":
            (
                global_df.loc[
                    global_df["is_eliminated"].eq(1),
                    "operation_label",
                ] == "elimination"
            ).all(),

        "length_positive":
            (
                global_df[
                    "length_m"
                ] > 0
            ).all(),

        "sinuosity_finite":
            np.isfinite(
                global_df[
                    "sinuosity"
                ].to_numpy(
                    dtype=float
                )
            ).all(),

        "no_duplicate_city_source_idx":
            not global_df[
                [
                    "city_key",
                    "source_idx",
                ]
            ].duplicated().any(),
    }

    for name, result in checks.items():
        print(
            f"{name:<45} "
            f"{'PASS' if bool(result) else 'FAIL'}"
        )

    if not all(
        bool(x)
        for x in checks.values()
    ):
        raise RuntimeError(
            "Polish road learning dataset "
            "failed validation."
        )

    # ==============================================================================================
    # MANIFEST
    # ==============================================================================================

    manifest = {
        "benchmark": "CartoGen Poland",
        "country": "Poland",
        "feature_family": "road",
        "source_dataset": "BDOT10k",
        "target_dataset": "BDOO",
        "crs": "EPSG:2180",
        "building_branch_available": False,
        "building_branch_reason": (
            "No semantically compatible individual "
            "generalized building target was identified "
            "in the acquired BDOO classes."
        ),
        "cities": summaries,
        "total_samples": int(
            len(global_df)
        ),
        "selection_distribution": {
            str(k): int(v)
            for k, v in global_df[
                "selection_label"
            ].value_counts().to_dict().items()
        },
        "operation_distribution": {
            str(k): int(v)
            for k, v in global_df[
                "operation_label"
            ].value_counts().to_dict().items()
        },
        "neighborhood_radius_m":
            NEIGHBOR_RADIUS_M,
        "validation_checks": {
            k: bool(v)
            for k, v in checks.items()
        },
    }

    manifest_path = (
        OUTPUT_DIR
        / "poland_road_learning_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ==============================================================================================
    # COMPLETE
    # ==============================================================================================

    banner("SCRIPT 61 COMPLETE")

    print()
    print("Global road learning dataset:")
    print(global_csv)

    print()
    print("Summary:")
    print(summary_csv)

    print()
    print("Manifest:")
    print(manifest_path)

    print()
    print(
        "[PASS] Polish road learning dataset "
        "created successfully."
    )

    print()
    print(
        "NEXT: Script 62 — construct leakage-safe "
        "Polish spatial benchmark tiles and "
        "train/validation/test splits."
    )


if __name__ == "__main__":
    main()