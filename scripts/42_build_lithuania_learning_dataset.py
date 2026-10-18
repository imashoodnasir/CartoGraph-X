from pathlib import Path
import json
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "validated_pairs"
)

BUILDING_OPERATIONS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "building_operations"
)

ROAD_TOPOLOGY_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "road_topology"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "learning_dataset"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CITIES = {
    "vilnius_central": "Vilnius",
    "kaunas_central": "Kaunas",
    "klaipeda_central": "Klaipeda",
}


# Neighborhood radii.
BUILDING_CONTEXT_RADIUS_M = 100.0
ROAD_CONTEXT_RADIUS_M = 150.0

# Number of nearest neighbors retained as compact context.
MAX_BUILDING_NEIGHBORS = 12
MAX_ROAD_NEIGHBORS = 12


# =============================================================================
# HELPERS
# =============================================================================

def clean_gdf(gdf):

    gdf = gdf.copy()

    gdf = gdf[
        gdf.geometry.notna()
    ].copy()

    gdf = gdf[
        ~gdf.geometry.is_empty
    ].copy()

    return gdf.reset_index(
        drop=True
    )


def safe_divide(a, b):

    if b is None:
        return 0.0

    if not np.isfinite(b):
        return 0.0

    if abs(b) < 1e-12:
        return 0.0

    return float(a / b)


def geometry_vertex_count(geometry):

    if geometry is None:
        return 0

    if geometry.is_empty:
        return 0

    geom_type = geometry.geom_type

    if geom_type == "Polygon":

        count = len(
            geometry.exterior.coords
        )

        for ring in geometry.interiors:
            count += len(
                ring.coords
            )

        return count

    if geom_type == "MultiPolygon":

        return sum(
            geometry_vertex_count(part)
            for part in geometry.geoms
        )

    if geom_type == "LineString":

        return len(
            geometry.coords
        )

    if geom_type == "MultiLineString":

        return sum(
            len(part.coords)
            for part in geometry.geoms
        )

    return 0


def compactness_polygon(geometry):

    area = float(
        geometry.area
    )

    perimeter = float(
        geometry.length
    )

    if perimeter <= 0:
        return 0.0

    return float(
        4.0
        * np.pi
        * area
        / (perimeter ** 2)
    )


def elongation(geometry):

    try:

        rectangle = (
            geometry
            .minimum_rotated_rectangle
        )

        coords = list(
            rectangle.exterior.coords
        )

        lengths = []

        for i in range(
            len(coords) - 1
        ):

            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]

            lengths.append(
                np.hypot(
                    x2 - x1,
                    y2 - y1
                )
            )

        lengths = [
            value
            for value in lengths
            if value > 1e-9
        ]

        if len(lengths) == 0:
            return 1.0

        longest = max(lengths)
        shortest = min(lengths)

        if shortest <= 0:
            return 1.0

        return float(
            longest / shortest
        )

    except Exception:

        return 1.0


def line_orientation(geometry):

    try:

        if geometry.geom_type == "MultiLineString":

            geometry = max(
                list(geometry.geoms),
                key=lambda x: x.length
            )

        coords = list(
            geometry.coords
        )

        if len(coords) < 2:
            return 0.0

        x1, y1 = coords[0]
        x2, y2 = coords[-1]

        angle = np.degrees(
            np.arctan2(
                y2 - y1,
                x2 - x1
            )
        )

        return float(
            angle % 180.0
        )

    except Exception:

        return 0.0


def straightness(geometry):

    try:

        if geometry.geom_type == "MultiLineString":

            parts = list(
                geometry.geoms
            )

            if not parts:
                return 0.0

            geometry = max(
                parts,
                key=lambda x: x.length
            )

        coords = list(
            geometry.coords
        )

        if len(coords) < 2:
            return 0.0

        start = np.array(
            coords[0],
            dtype=float
        )

        end = np.array(
            coords[-1],
            dtype=float
        )

        euclidean = np.linalg.norm(
            end - start
        )

        length = float(
            geometry.length
        )

        if length <= 0:
            return 0.0

        return float(
            euclidean / length
        )

    except Exception:

        return 0.0


# =============================================================================
# BUILDING FEATURE EXTRACTION
# =============================================================================

def extract_building_features(gdf):

    records = []

    print(
        f"  Extracting geometric features "
        f"for {len(gdf):,} buildings..."
    )

    for idx, geometry in enumerate(
        gdf.geometry
    ):

        centroid = (
            geometry.centroid
        )

        minx, miny, maxx, maxy = (
            geometry.bounds
        )

        records.append(
            {
                "feature_index":
                    idx,

                "centroid_x":
                    float(centroid.x),

                "centroid_y":
                    float(centroid.y),

                "area_m2":
                    float(
                        geometry.area
                    ),

                "perimeter_m":
                    float(
                        geometry.length
                    ),

                "vertex_count":
                    int(
                        geometry_vertex_count(
                            geometry
                        )
                    ),

                "compactness":
                    compactness_polygon(
                        geometry
                    ),

                "elongation":
                    elongation(
                        geometry
                    ),

                "bbox_width_m":
                    float(
                        maxx - minx
                    ),

                "bbox_height_m":
                    float(
                        maxy - miny
                    ),
            }
        )

    return pd.DataFrame(
        records
    )


# =============================================================================
# BUILDING CONTEXT
# =============================================================================

def add_building_context(
    buildings,
    features
):

    print(
        "  Computing building neighborhood context..."
    )

    spatial_index = (
        buildings.sindex
    )

    context_records = []

    for idx, geometry in enumerate(
        buildings.geometry
    ):

        centroid = (
            geometry.centroid
        )

        search_area = (
            centroid.buffer(
                BUILDING_CONTEXT_RADIUS_M
            )
        )

        candidate_indices = list(
            spatial_index.intersection(
                search_area.bounds
            )
        )

        neighbors = []

        for candidate_idx in candidate_indices:

            if candidate_idx == idx:
                continue

            candidate_geometry = (
                buildings.geometry.iloc[
                    candidate_idx
                ]
            )

            distance = float(
                geometry.distance(
                    candidate_geometry
                )
            )

            if (
                distance
                <= BUILDING_CONTEXT_RADIUS_M
            ):

                neighbors.append(
                    (
                        candidate_idx,
                        distance
                    )
                )

        neighbors.sort(
            key=lambda x: x[1]
        )

        nearest = (
            neighbors[
                :MAX_BUILDING_NEIGHBORS
            ]
        )

        distances = [
            value[1]
            for value in nearest
        ]

        neighbor_areas = [
            float(
                buildings.geometry.iloc[
                    value[0]
                ].area
            )
            for value in nearest
        ]

        context_records.append(
            {
                "neighbor_count_100m":
                    int(
                        len(neighbors)
                    ),

                "nearest_neighbor_m":
                    float(
                        min(distances)
                    )
                    if distances
                    else BUILDING_CONTEXT_RADIUS_M,

                "mean_neighbor_distance_m":
                    float(
                        np.mean(
                            distances
                        )
                    )
                    if distances
                    else BUILDING_CONTEXT_RADIUS_M,

                "mean_neighbor_area_m2":
                    float(
                        np.mean(
                            neighbor_areas
                        )
                    )
                    if neighbor_areas
                    else 0.0,
            }
        )

        if (
            (idx + 1) % 5000 == 0
            or idx + 1 == len(buildings)
        ):

            print(
                f"    {idx + 1:,}/"
                f"{len(buildings):,}"
            )

    context_df = pd.DataFrame(
        context_records
    )

    return pd.concat(
        [
            features.reset_index(
                drop=True
            ),
            context_df.reset_index(
                drop=True
            ),
        ],
        axis=1
    )


# =============================================================================
# ROAD FEATURE EXTRACTION
# =============================================================================

def extract_road_features(gdf):

    records = []

    print(
        f"  Extracting geometric features "
        f"for {len(gdf):,} roads..."
    )

    for idx, geometry in enumerate(
        gdf.geometry
    ):

        centroid = (
            geometry.centroid
        )

        records.append(
            {
                "feature_index":
                    idx,

                "centroid_x":
                    float(
                        centroid.x
                    ),

                "centroid_y":
                    float(
                        centroid.y
                    ),

                "length_m":
                    float(
                        geometry.length
                    ),

                "vertex_count":
                    int(
                        geometry_vertex_count(
                            geometry
                        )
                    ),

                "orientation_deg":
                    line_orientation(
                        geometry
                    ),

                "straightness":
                    straightness(
                        geometry
                    ),
            }
        )

    return pd.DataFrame(
        records
    )


# =============================================================================
# ROAD CONTEXT
# =============================================================================

def add_road_context(
    roads,
    features
):

    print(
        "  Computing road neighborhood context..."
    )

    spatial_index = (
        roads.sindex
    )

    context_records = []

    for idx, geometry in enumerate(
        roads.geometry
    ):

        centroid = (
            geometry.centroid
        )

        search_area = (
            centroid.buffer(
                ROAD_CONTEXT_RADIUS_M
            )
        )

        candidate_indices = list(
            spatial_index.intersection(
                search_area.bounds
            )
        )

        neighbors = []

        intersecting = 0

        for candidate_idx in candidate_indices:

            if candidate_idx == idx:
                continue

            candidate_geometry = (
                roads.geometry.iloc[
                    candidate_idx
                ]
            )

            distance = float(
                geometry.distance(
                    candidate_geometry
                )
            )

            if (
                distance
                <= ROAD_CONTEXT_RADIUS_M
            ):

                neighbors.append(
                    (
                        candidate_idx,
                        distance
                    )
                )

            if geometry.intersects(
                candidate_geometry
            ):
                intersecting += 1

        neighbors.sort(
            key=lambda x: x[1]
        )

        nearest = (
            neighbors[
                :MAX_ROAD_NEIGHBORS
            ]
        )

        distances = [
            item[1]
            for item in nearest
        ]

        context_records.append(
            {
                "neighbor_count_150m":
                    int(
                        len(neighbors)
                    ),

                "intersecting_road_count":
                    int(
                        intersecting
                    ),

                "nearest_road_m":
                    float(
                        min(distances)
                    )
                    if distances
                    else ROAD_CONTEXT_RADIUS_M,

                "mean_neighbor_distance_m":
                    float(
                        np.mean(
                            distances
                        )
                    )
                    if distances
                    else ROAD_CONTEXT_RADIUS_M,
            }
        )

        if (
            (idx + 1) % 1000 == 0
            or idx + 1 == len(roads)
        ):

            print(
                f"    {idx + 1:,}/"
                f"{len(roads):,}"
            )

    context_df = pd.DataFrame(
        context_records
    )

    return pd.concat(
        [
            features.reset_index(
                drop=True
            ),
            context_df.reset_index(
                drop=True
            ),
        ],
        axis=1
    )


# =============================================================================
# BUILDING OPERATION LABELS
# =============================================================================

def load_building_labels(
    window,
    source_count
):

    operations_path = (
        BUILDING_OPERATIONS_DIR
        / f"{window}_building_operations.csv"
    )

    eliminations_path = (
        BUILDING_OPERATIONS_DIR
        / f"{window}_building_eliminations.csv"
    )

    labels = pd.DataFrame(
        {
            "feature_index":
                np.arange(
                    source_count
                ),

            "building_operation":
                "unresolved",

            "target_count":
                0,
        }
    )

    if operations_path.exists():

        operations = pd.read_csv(
            operations_path
        )

        # Script 38 is target/generalized-object based.
        # Source indices may be stored explicitly,
        # or as serialized source lists.

        source_index_columns = [
            col
            for col in operations.columns
            if col.lower()
            in {
                "source_index",
                "source_idx",
                "local_index",
                "local_idx",
            }
        ]

        if source_index_columns:

            source_col = (
                source_index_columns[0]
            )

            operation_col = next(
                (
                    col
                    for col
                    in operations.columns
                    if col.lower()
                    == "operation"
                ),
                None
            )

            if operation_col:

                for row in operations.itertuples(
                    index=False
                ):

                    try:

                        source_idx = int(
                            getattr(
                                row,
                                source_col
                            )
                        )

                    except Exception:
                        continue

                    if (
                        0 <= source_idx
                        < source_count
                    ):

                        labels.loc[
                            source_idx,
                            "building_operation"
                        ] = getattr(
                            row,
                            operation_col
                        )

                        labels.loc[
                            source_idx,
                            "target_count"
                        ] += 1

    if eliminations_path.exists():

        eliminations = pd.read_csv(
            eliminations_path
        )

        candidate_columns = [
            col
            for col
            in eliminations.columns
            if col.lower()
            in {
                "source_index",
                "source_idx",
                "local_index",
                "local_idx",
            }
        ]

        if candidate_columns:

            col = candidate_columns[0]

            for value in eliminations[
                col
            ].dropna():

                try:
                    idx = int(value)

                except Exception:
                    continue

                if (
                    0 <= idx
                    < source_count
                ):

                    labels.loc[
                        idx,
                        "building_operation"
                    ] = "elimination"

    return labels


# =============================================================================
# ROAD OPERATION LABELS
# =============================================================================

def load_road_labels(
    window,
    source_count
):

    operation_path = (
        ROAD_TOPOLOGY_DIR
        / f"{window}_road_operations.csv"
    )

    elimination_path = (
        ROAD_TOPOLOGY_DIR
        / f"{window}_road_eliminations.csv"
    )

    labels = pd.DataFrame(
        {
            "feature_index":
                np.arange(
                    source_count
                ),

            "road_operation":
                "unresolved",
        }
    )

    # Script 41 operation table is target based,
    # so source elimination is the directly
    # transferable source-level supervision here.

    if elimination_path.exists():

        eliminated = pd.read_csv(
            elimination_path
        )

        source_col = next(
            (
                col
                for col
                in eliminated.columns
                if col.lower()
                in {
                    "source_index",
                    "source_idx",
                    "source_id",
                }
            ),
            None
        )

        if source_col:

            for value in eliminated[
                source_col
            ].dropna():

                try:

                    idx = int(
                        value
                    )

                except Exception:
                    continue

                if (
                    0 <= idx
                    < source_count
                ):

                    labels.loc[
                        idx,
                        "road_operation"
                    ] = "elimination"

    # Everything not eliminated is a selected
    # source-road candidate. Detailed graph labels
    # are preserved separately.

    labels.loc[
        labels[
            "road_operation"
        ]
        == "unresolved",
        "road_operation"
    ] = "selected"

    return labels


# =============================================================================
# PROCESS CITY
# =============================================================================

def process_city(
    window,
    city
):

    print()
    print("=" * 120)
    print(
        f"{window} ({city})"
    )
    print("=" * 120)

    gpkg = (
        VALIDATED_DIR
        / f"{window}_validated.gpkg"
    )

    if not gpkg.exists():

        raise FileNotFoundError(
            gpkg
        )

    print(
        "Reading validated source/target layers..."
    )

    source_buildings = clean_gdf(
        gpd.read_file(
            gpkg,
            layer="source_buildings"
        )
    )

    target_buildings = clean_gdf(
        gpd.read_file(
            gpkg,
            layer="target_buildings"
        )
    )

    source_roads = clean_gdf(
        gpd.read_file(
            gpkg,
            layer="source_roads"
        )
    )

    target_roads = clean_gdf(
        gpd.read_file(
            gpkg,
            layer="target_roads"
        )
    )

    print()
    print(
        f"Source buildings: "
        f"{len(source_buildings):,}"
    )

    print(
        f"Target buildings: "
        f"{len(target_buildings):,}"
    )

    print(
        f"Source roads:     "
        f"{len(source_roads):,}"
    )

    print(
        f"Target roads:     "
        f"{len(target_roads):,}"
    )

    # =================================================================
    # BUILDINGS
    # =================================================================

    print()
    print("-" * 120)
    print(
        "BUILDING BRANCH"
    )
    print("-" * 120)

    building_features = (
        extract_building_features(
            source_buildings
        )
    )

    building_features = (
        add_building_context(
            source_buildings,
            building_features
        )
    )

    building_labels = (
        load_building_labels(
            window,
            len(source_buildings)
        )
    )

    building_table = (
        building_features.merge(
            building_labels,
            on="feature_index",
            how="left"
        )
    )

    building_table[
        "city"
    ] = city

    building_table[
        "window"
    ] = window

    building_table[
        "country"
    ] = "Lithuania"

    building_table[
        "family"
    ] = "building"

    # Add geometry for spatial splitting later.
    building_gdf = (
        source_buildings[
            ["geometry"]
        ]
        .copy()
    )

    building_gdf[
        "feature_index"
    ] = np.arange(
        len(building_gdf)
    )

    building_gdf = (
        building_gdf.merge(
            building_table,
            on="feature_index",
            how="left"
        )
    )

    # =================================================================
    # ROADS
    # =================================================================

    print()
    print("-" * 120)
    print(
        "ROAD BRANCH"
    )
    print("-" * 120)

    road_features = (
        extract_road_features(
            source_roads
        )
    )

    road_features = (
        add_road_context(
            source_roads,
            road_features
        )
    )

    road_labels = (
        load_road_labels(
            window,
            len(source_roads)
        )
    )

    road_table = (
        road_features.merge(
            road_labels,
            on="feature_index",
            how="left"
        )
    )

    road_table[
        "city"
    ] = city

    road_table[
        "window"
    ] = window

    road_table[
        "country"
    ] = "Lithuania"

    road_table[
        "family"
    ] = "road"

    road_gdf = (
        source_roads[
            ["geometry"]
        ]
        .copy()
    )

    road_gdf[
        "feature_index"
    ] = np.arange(
        len(road_gdf)
    )

    road_gdf = (
        road_gdf.merge(
            road_table,
            on="feature_index",
            how="left"
        )
    )

    # =================================================================
    # TARGET REFERENCE TABLES
    # =================================================================

    target_building_features = (
        extract_building_features(
            target_buildings
        )
    )

    target_building_features[
        "city"
    ] = city

    target_building_features[
        "window"
    ] = window

    target_road_features = (
        extract_road_features(
            target_roads
        )
    )

    target_road_features[
        "city"
    ] = city

    target_road_features[
        "window"
    ] = window

    # =================================================================
    # SAVE CITY OUTPUT
    # =================================================================

    output_gpkg = (
        OUTPUT_DIR
        / f"{window}_learning_dataset.gpkg"
    )

    if output_gpkg.exists():
        output_gpkg.unlink()

    building_gdf.to_file(
        output_gpkg,
        layer="building_source_samples",
        driver="GPKG"
    )

    road_gdf.to_file(
        output_gpkg,
        layer="road_source_samples",
        driver="GPKG"
    )

    target_buildings.to_file(
        output_gpkg,
        layer="building_targets",
        driver="GPKG"
    )

    target_roads.to_file(
        output_gpkg,
        layer="road_targets",
        driver="GPKG"
    )

    building_csv = (
        OUTPUT_DIR
        / f"{window}_building_learning.csv"
    )

    road_csv = (
        OUTPUT_DIR
        / f"{window}_road_learning.csv"
    )

    target_building_csv = (
        OUTPUT_DIR
        / f"{window}_target_building_features.csv"
    )

    target_road_csv = (
        OUTPUT_DIR
        / f"{window}_target_road_features.csv"
    )

    building_table.to_csv(
        building_csv,
        index=False
    )

    road_table.to_csv(
        road_csv,
        index=False
    )

    target_building_features.to_csv(
        target_building_csv,
        index=False
    )

    target_road_features.to_csv(
        target_road_csv,
        index=False
    )

    building_operation_counts = (
        building_table[
            "building_operation"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    road_operation_counts = (
        road_table[
            "road_operation"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    summary = {
        "window":
            window,

        "city":
            city,

        "source_buildings":
            int(
                len(source_buildings)
            ),

        "target_buildings":
            int(
                len(target_buildings)
            ),

        "source_roads":
            int(
                len(source_roads)
            ),

        "target_roads":
            int(
                len(target_roads)
            ),

        "building_feature_columns":
            int(
                len(
                    building_table.columns
                )
            ),

        "road_feature_columns":
            int(
                len(
                    road_table.columns
                )
            ),

        "building_operation_counts":
            {
                str(k):
                    int(v)
                for k, v
                in building_operation_counts.items()
            },

        "road_operation_counts":
            {
                str(k):
                    int(v)
                for k, v
                in road_operation_counts.items()
            },
    }

    print()
    print(
        "Learning dataset created."
    )

    print(
        f"Building samples: "
        f"{len(building_table):,}"
    )

    print(
        f"Road samples:     "
        f"{len(road_table):,}"
    )

    print()
    print(
        "Building labels:"
    )

    for key, value in (
        building_operation_counts.items()
    ):

        print(
            f"  {str(key):<35}"
            f"{value:>8,}"
        )

    print()
    print(
        "Road labels:"
    )

    for key, value in (
        road_operation_counts.items()
    ):

        print(
            f"  {str(key):<35}"
            f"{value:>8,}"
        )

    print()
    print(
        f"GeoPackage:\n"
        f"{output_gpkg}"
    )

    return (
        summary,
        building_table,
        road_table
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 120)
    print(
        "CARTOGEN — BUILD UNIFIED "
        "LITHUANIAN LEARNING DATASET"
    )
    print("=" * 120)

    summaries = []

    all_buildings = []
    all_roads = []

    for window, city in (
        CITIES.items()
    ):

        (
            summary,
            buildings,
            roads
        ) = process_city(
            window,
            city
        )

        summaries.append(
            summary
        )

        all_buildings.append(
            buildings
        )

        all_roads.append(
            roads
        )

    # =================================================================
    # NATIONAL TABLES
    # =================================================================

    national_buildings = pd.concat(
        all_buildings,
        ignore_index=True
    )

    national_roads = pd.concat(
        all_roads,
        ignore_index=True
    )

    national_building_csv = (
        OUTPUT_DIR
        / "lithuania_building_learning_dataset.csv"
    )

    national_road_csv = (
        OUTPUT_DIR
        / "lithuania_road_learning_dataset.csv"
    )

    national_buildings.to_csv(
        national_building_csv,
        index=False
    )

    national_roads.to_csv(
        national_road_csv,
        index=False
    )

    # =================================================================
    # DATASET STATISTICS
    # =================================================================

    summary_df = pd.DataFrame(
        [
            {
                "city":
                    item["city"],

                "source_buildings":
                    item[
                        "source_buildings"
                    ],

                "target_buildings":
                    item[
                        "target_buildings"
                    ],

                "source_roads":
                    item[
                        "source_roads"
                    ],

                "target_roads":
                    item[
                        "target_roads"
                    ],

                "building_feature_columns":
                    item[
                        "building_feature_columns"
                    ],

                "road_feature_columns":
                    item[
                        "road_feature_columns"
                    ],
            }
            for item in summaries
        ]
    )

    summary_csv = (
        OUTPUT_DIR
        / "learning_dataset_summary.csv"
    )

    summary_json = (
        OUTPUT_DIR
        / "learning_dataset_manifest.json"
    )

    summary_df.to_csv(
        summary_csv,
        index=False
    )

    manifest = {
        "country":
            "Lithuania",

        "source_dataset":
            "GRPK",

        "target_dataset":
            "GDR50LT",

        "source_nominal_scale":
            "1:10000",

        "target_nominal_scale":
            "1:50000",

        "crs":
            "EPSG:3346",

        "cities":
            summaries,

        "national_building_samples":
            int(
                len(
                    national_buildings
                )
            ),

        "national_road_samples":
            int(
                len(
                    national_roads
                )
            ),

        "building_context_radius_m":
            BUILDING_CONTEXT_RADIUS_M,

        "road_context_radius_m":
            ROAD_CONTEXT_RADIUS_M,
    }

    with open(
        summary_json,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 120)
    print(
        "GLOBAL LITHUANIAN LEARNING DATASET"
    )
    print("=" * 120)

    print()
    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Total building samples: "
        f"{len(national_buildings):,}"
    )

    print(
        f"Total road samples:     "
        f"{len(national_roads):,}"
    )

    print()
    print("=" * 120)
    print(
        "SCRIPT 42 COMPLETE"
    )
    print("=" * 120)

    print()
    print(
        f"Buildings:\n"
        f"{national_building_csv}"
    )

    print()
    print(
        f"Roads:\n"
        f"{national_road_csv}"
    )

    print()
    print(
        f"Summary:\n"
        f"{summary_csv}"
    )

    print()
    print(
        f"Manifest:\n"
        f"{summary_json}"
    )


if __name__ == "__main__":
    main()