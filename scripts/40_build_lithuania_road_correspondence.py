from pathlib import Path
import json
import math

import geopandas as gpd
import numpy as np
import pandas as pd


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "road_correspondence"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


WINDOWS = {
    "vilnius_central": "Vilnius",
    "kaunas_central": "Kaunas",
    "klaipeda_central": "Klaipeda",
}


# Remove edge effects caused by clipping city windows.
INTERIOR_MARGIN_M = 50.0

# Candidate search distance.
SEARCH_DISTANCE_M = 25.0

# Buffers used for directional coverage.
SOURCE_BUFFER_M = 8.0
TARGET_BUFFER_M = 8.0

# Maximum useful distance for distance score.
DISTANCE_NORMALIZER_M = 20.0

# Candidate acceptance.
MIN_SOURCE_COVERAGE = 0.08
MIN_TARGET_COVERAGE = 0.08
MAX_DISTANCE_M = 20.0

# Final correspondence threshold.
MIN_MATCH_SCORE = 0.25


SEMANTIC_FIELDS = [
    "VARDAS",
    "GKODAS",
    "DANGA",
    "LYGMUO",
    "PASKIRTIS",
    "PLOTIS",
]


# =============================================================================
# HELPERS
# =============================================================================

def normalize_value(value):

    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    if value in {
        "",
        "NONE",
        "NULL",
        "NAN"
    }:
        return None

    return value


def find_column(gdf, requested):

    requested = requested.upper()

    for column in gdf.columns:

        if str(column).upper() == requested:
            return column

    return None


def get_bbox_polygon(gdf):

    xmin, ymin, xmax, ymax = (
        gdf.total_bounds
    )

    from shapely.geometry import box

    return box(
        xmin,
        ymin,
        xmax,
        ymax
    )


def interior_filter(
    gdf,
    reference_bounds,
    margin
):

    from shapely.geometry import box

    xmin, ymin, xmax, ymax = (
        reference_bounds
    )

    interior = box(
        xmin + margin,
        ymin + margin,
        xmax - margin,
        ymax - margin
    )

    centroids = gdf.geometry.centroid

    mask = centroids.within(
        interior
    )

    return (
        gdf.loc[mask]
        .copy()
        .reset_index(drop=True)
    )


def line_orientation(geometry):
    """
    Orientation of a road segment in degrees.

    Direction is ignored, therefore orientation
    lies in [0, 180).
    """

    if geometry is None:
        return None

    if geometry.is_empty:
        return None

    try:

        if geometry.geom_type == "MultiLineString":

            parts = list(
                geometry.geoms
            )

            if not parts:
                return None

            geometry = max(
                parts,
                key=lambda g: g.length
            )

        coords = list(
            geometry.coords
        )

        if len(coords) < 2:
            return None

        x1, y1 = coords[0]
        x2, y2 = coords[-1]

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return None

        angle = math.degrees(
            math.atan2(
                dy,
                dx
            )
        )

        angle = angle % 180.0

        return angle

    except Exception:

        return None


def orientation_difference(
    angle_a,
    angle_b
):

    if angle_a is None:
        return 90.0

    if angle_b is None:
        return 90.0

    difference = abs(
        angle_a - angle_b
    )

    return min(
        difference,
        180.0 - difference
    )


def safe_intersection_length(
    line,
    polygon
):

    try:

        intersection = (
            line.intersection(
                polygon
            )
        )

        return float(
            intersection.length
        )

    except Exception:

        return 0.0


def semantic_compatibility(
    source_row,
    target_row,
    source_columns,
    target_columns
):
    """
    Semantic compatibility is supporting evidence,
    not a road identifier.

    VARDAS receives the largest semantic weight.
    """

    field_weights = {
        "VARDAS": 3.0,
        "GKODAS": 1.5,
        "DANGA": 1.0,
        "LYGMUO": 1.0,
        "PASKIRTIS": 1.0,
        "PLOTIS": 0.5,
    }

    matched_weight = 0.0
    available_weight = 0.0

    matches = {}

    for field in SEMANTIC_FIELDS:

        source_column = (
            source_columns.get(
                field
            )
        )

        target_column = (
            target_columns.get(
                field
            )
        )

        if (
            source_column is None
            or target_column is None
        ):
            continue

        source_value = normalize_value(
            source_row[source_column]
        )

        target_value = normalize_value(
            target_row[target_column]
        )

        if (
            source_value is None
            or target_value is None
        ):
            matches[field] = None
            continue

        weight = field_weights[
            field
        ]

        available_weight += weight

        is_match = (
            source_value
            == target_value
        )

        matches[field] = (
            bool(is_match)
        )

        if is_match:
            matched_weight += weight

    if available_weight == 0:

        return 0.5, matches

    score = (
        matched_weight
        / available_weight
    )

    return float(score), matches


def calculate_candidate_metrics(
    source_row,
    target_row,
    source_columns,
    target_columns
):

    source_geometry = (
        source_row.geometry
    )

    target_geometry = (
        target_row.geometry
    )

    source_length = float(
        source_geometry.length
    )

    target_length = float(
        target_geometry.length
    )

    if (
        source_length <= 0
        or target_length <= 0
    ):
        return None

    distance = float(
        source_geometry.distance(
            target_geometry
        )
    )

    if distance > MAX_DISTANCE_M:
        return None

    target_buffer = (
        target_geometry.buffer(
            TARGET_BUFFER_M
        )
    )

    source_buffer = (
        source_geometry.buffer(
            SOURCE_BUFFER_M
        )
    )

    source_overlap_length = (
        safe_intersection_length(
            source_geometry,
            target_buffer
        )
    )

    target_overlap_length = (
        safe_intersection_length(
            target_geometry,
            source_buffer
        )
    )

    source_coverage = min(
        1.0,
        source_overlap_length
        / source_length
    )

    target_coverage = min(
        1.0,
        target_overlap_length
        / target_length
    )

    if (
        source_coverage
        < MIN_SOURCE_COVERAGE
        and
        target_coverage
        < MIN_TARGET_COVERAGE
    ):
        return None

    source_angle = (
        line_orientation(
            source_geometry
        )
    )

    target_angle = (
        line_orientation(
            target_geometry
        )
    )

    angle_difference = (
        orientation_difference(
            source_angle,
            target_angle
        )
    )

    orientation_score = max(
        0.0,
        1.0
        - angle_difference
        / 90.0
    )

    distance_score = max(
        0.0,
        1.0
        - distance
        / DISTANCE_NORMALIZER_M
    )

    semantic_score, semantic_matches = (
        semantic_compatibility(
            source_row,
            target_row,
            source_columns,
            target_columns
        )
    )

    # Geometry remains dominant.
    #
    # 75% geometry/topology
    # 25% semantic evidence

    score = (
        0.30 * source_coverage
        + 0.25 * target_coverage
        + 0.12 * distance_score
        + 0.08 * orientation_score
        + 0.25 * semantic_score
    )

    try:

        hausdorff = float(
            source_geometry.hausdorff_distance(
                target_geometry
            )
        )

    except Exception:

        hausdorff = np.nan

    return {
        "source_length_m":
            source_length,

        "target_length_m":
            target_length,

        "distance_m":
            distance,

        "hausdorff_m":
            hausdorff,

        "source_coverage":
            source_coverage,

        "target_coverage":
            target_coverage,

        "source_angle_deg":
            source_angle,

        "target_angle_deg":
            target_angle,

        "angle_difference_deg":
            angle_difference,

        "orientation_score":
            orientation_score,

        "distance_score":
            distance_score,

        "semantic_score":
            semantic_score,

        "match_score":
            score,

        "semantic_matches":
            semantic_matches,
    }


# =============================================================================
# CITY PROCESSING
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

    gpkg_path = (
        VALIDATED_DIR
        / f"{window}_validated.gpkg"
    )

    if not gpkg_path.exists():

        raise FileNotFoundError(
            gpkg_path
        )

    print(
        "Reading source roads..."
    )

    source = gpd.read_file(
        gpkg_path,
        layer="source_roads"
    )

    print(
        "Reading target roads..."
    )

    target = gpd.read_file(
        gpkg_path,
        layer="target_roads"
    )

    print(
        f"Raw source roads: "
        f"{len(source):,}"
    )

    print(
        f"Raw target roads: "
        f"{len(target):,}"
    )

    if source.crs != target.crs:

        target = target.to_crs(
            source.crs
        )

    bounds = source.total_bounds

    source = interior_filter(
        source,
        bounds,
        INTERIOR_MARGIN_M
    )

    target = interior_filter(
        target,
        bounds,
        INTERIOR_MARGIN_M
    )

    source = source[
        source.geometry.notna()
        & ~source.geometry.is_empty
    ].copy()

    target = target[
        target.geometry.notna()
        & ~target.geometry.is_empty
    ].copy()

    source.reset_index(
        drop=True,
        inplace=True
    )

    target.reset_index(
        drop=True,
        inplace=True
    )

    print(
        f"Interior source roads: "
        f"{len(source):,}"
    )

    print(
        f"Interior target roads: "
        f"{len(target):,}"
    )

    source_columns = {
        field:
            find_column(
                source,
                field
            )
        for field in SEMANTIC_FIELDS
    }

    target_columns = {
        field:
            find_column(
                target,
                field
            )
        for field in SEMANTIC_FIELDS
    }

    print()
    print(
        "Semantic fields available:"
    )

    for field in SEMANTIC_FIELDS:

        print(
            f"  {field:<12} "
            f"source={source_columns[field]} "
            f"target={target_columns[field]}"
        )

    # IDs independent of original dataset identifiers.
    source[
        "cartogen_source_id"
    ] = np.arange(
        len(source),
        dtype=int
    )

    target[
        "cartogen_target_id"
    ] = np.arange(
        len(target),
        dtype=int
    )

    print()
    print(
        "Building target spatial index..."
    )

    target_sindex = (
        target.sindex
    )

    records = []

    print(
        f"Evaluating "
        f"{len(source):,} source roads..."
    )

    for source_position, source_row in (
        source.iterrows()
    ):

        source_geometry = (
            source_row.geometry
        )

        search_geometry = (
            source_geometry.buffer(
                SEARCH_DISTANCE_M
            )
        )

        candidate_positions = list(
            target_sindex.intersection(
                search_geometry.bounds
            )
        )

        for target_position in candidate_positions:

            target_row = (
                target.iloc[
                    target_position
                ]
            )

            if not search_geometry.intersects(
                target_row.geometry
            ):
                continue

            metrics = (
                calculate_candidate_metrics(
                    source_row,
                    target_row,
                    source_columns,
                    target_columns
                )
            )

            if metrics is None:
                continue

            if (
                metrics[
                    "match_score"
                ]
                < MIN_MATCH_SCORE
            ):
                continue

            record = {
                "source_id":
                    int(
                        source_row[
                            "cartogen_source_id"
                        ]
                    ),

                "target_id":
                    int(
                        target_row[
                            "cartogen_target_id"
                        ]
                    ),

                "source_index":
                    int(source_position),

                "target_index":
                    int(target_position),
            }

            for key, value in (
                metrics.items()
            ):

                if key == "semantic_matches":
                    continue

                record[key] = value

            for field, match_value in (
                metrics[
                    "semantic_matches"
                ].items()
            ):

                record[
                    f"{field.lower()}_match"
                ] = match_value

            records.append(
                record
            )

        if (
            (source_position + 1)
            % 1000
            == 0
        ):

            print(
                f"  "
                f"{source_position + 1:,}"
                f"/"
                f"{len(source):,}"
            )

    pairs = pd.DataFrame(
        records
    )

    print()
    print(
        f"Accepted candidate pairs: "
        f"{len(pairs):,}"
    )

    if len(pairs) == 0:

        raise RuntimeError(
            f"No road correspondences "
            f"found for {city}."
        )

    # =================================================================
    # RELATIONSHIP STRUCTURE
    # =================================================================

    source_degree = (
        pairs.groupby(
            "source_id"
        )[
            "target_id"
        ]
        .nunique()
    )

    target_degree = (
        pairs.groupby(
            "target_id"
        )[
            "source_id"
        ]
        .nunique()
    )

    matched_source_ids = set(
        pairs["source_id"]
    )

    matched_target_ids = set(
        pairs["target_id"]
    )

    matched_source = len(
        matched_source_ids
    )

    matched_target = len(
        matched_target_ids
    )

    source_one_target = int(
        (source_degree == 1).sum()
    )

    source_multiple_targets = int(
        (source_degree > 1).sum()
    )

    target_one_source = int(
        (target_degree == 1).sum()
    )

    target_multiple_sources = int(
        (target_degree > 1).sum()
    )

    # Strongest candidate is retained as a diagnostic,
    # but NOT treated as the only ground truth.

    best_source_pairs = (
        pairs
        .sort_values(
            [
                "source_id",
                "match_score"
            ],
            ascending=[
                True,
                False
            ]
        )
        .drop_duplicates(
            "source_id"
        )
        .copy()
    )

    best_target_pairs = (
        pairs
        .sort_values(
            [
                "target_id",
                "match_score"
            ],
            ascending=[
                True,
                False
            ]
        )
        .drop_duplicates(
            "target_id"
        )
        .copy()
    )

    summary = {
        "window":
            window,

        "city":
            city,

        "source_count":
            int(len(source)),

        "target_count":
            int(len(target)),

        "candidate_pairs":
            int(len(pairs)),

        "matched_source":
            int(matched_source),

        "unmatched_source":
            int(
                len(source)
                - matched_source
            ),

        "matched_source_pct":
            float(
                100.0
                * matched_source
                / len(source)
            ),

        "matched_target":
            int(matched_target),

        "unmatched_target":
            int(
                len(target)
                - matched_target
            ),

        "matched_target_pct":
            float(
                100.0
                * matched_target
                / len(target)
            ),

        "source_one_target":
            source_one_target,

        "source_multiple_targets":
            source_multiple_targets,

        "target_one_source":
            target_one_source,

        "target_multiple_sources":
            target_multiple_sources,

        "mean_targets_per_matched_source":
            float(
                source_degree.mean()
            ),

        "mean_sources_per_matched_target":
            float(
                target_degree.mean()
            ),

        "mean_source_coverage":
            float(
                pairs[
                    "source_coverage"
                ].mean()
            ),

        "median_source_coverage":
            float(
                pairs[
                    "source_coverage"
                ].median()
            ),

        "mean_target_coverage":
            float(
                pairs[
                    "target_coverage"
                ].mean()
            ),

        "median_target_coverage":
            float(
                pairs[
                    "target_coverage"
                ].median()
            ),

        "mean_distance_m":
            float(
                pairs[
                    "distance_m"
                ].mean()
            ),

        "median_distance_m":
            float(
                pairs[
                    "distance_m"
                ].median()
            ),

        "mean_angle_difference_deg":
            float(
                pairs[
                    "angle_difference_deg"
                ].mean()
            ),

        "median_angle_difference_deg":
            float(
                pairs[
                    "angle_difference_deg"
                ].median()
            ),

        "mean_semantic_score":
            float(
                pairs[
                    "semantic_score"
                ].mean()
            ),

        "mean_match_score":
            float(
                pairs[
                    "match_score"
                ].mean()
            ),

        "median_match_score":
            float(
                pairs[
                    "match_score"
                ].median()
            ),
    }

    print()
    print("-" * 120)
    print(
        "ROAD CORRESPONDENCE SUMMARY"
    )
    print("-" * 120)

    for key, value in summary.items():

        if key in {
            "window",
            "city"
        }:
            continue

        if isinstance(
            value,
            float
        ):

            print(
                f"{key:<42}"
                f"{value:>12.3f}"
            )

        else:

            print(
                f"{key:<42}"
                f"{value:>12,}"
            )

    # =================================================================
    # SAVE TABLES
    # =================================================================

    pairs_path = (
        OUTPUT_DIR
        / f"{window}_road_pairs.csv"
    )

    best_source_path = (
        OUTPUT_DIR
        / f"{window}_best_source_pairs.csv"
    )

    best_target_path = (
        OUTPUT_DIR
        / f"{window}_best_target_pairs.csv"
    )

    pairs.to_csv(
        pairs_path,
        index=False
    )

    best_source_pairs.to_csv(
        best_source_path,
        index=False
    )

    best_target_pairs.to_csv(
        best_target_path,
        index=False
    )

    # =================================================================
    # DIAGNOSTIC GEOPACKAGE
    # =================================================================

    source_output = (
        source.copy()
    )

    target_output = (
        target.copy()
    )

    source_output[
        "match_count"
    ] = (
        source_output[
            "cartogen_source_id"
        ]
        .map(source_degree)
        .fillna(0)
        .astype(int)
    )

    target_output[
        "match_count"
    ] = (
        target_output[
            "cartogen_target_id"
        ]
        .map(target_degree)
        .fillna(0)
        .astype(int)
    )

    source_best_score = (
        pairs.groupby(
            "source_id"
        )[
            "match_score"
        ]
        .max()
    )

    target_best_score = (
        pairs.groupby(
            "target_id"
        )[
            "match_score"
        ]
        .max()
    )

    source_output[
        "best_score"
    ] = (
        source_output[
            "cartogen_source_id"
        ]
        .map(
            source_best_score
        )
    )

    target_output[
        "best_score"
    ] = (
        target_output[
            "cartogen_target_id"
        ]
        .map(
            target_best_score
        )
    )

    diagnostic_path = (
        OUTPUT_DIR
        / f"{window}_road_correspondence.gpkg"
    )

    if diagnostic_path.exists():
        diagnostic_path.unlink()

    source_output.to_file(
        diagnostic_path,
        layer="source_roads",
        driver="GPKG"
    )

    target_output.to_file(
        diagnostic_path,
        layer="target_roads",
        driver="GPKG"
    )

    print()
    print(
        f"Pairs CSV:\n"
        f"{pairs_path}"
    )

    print()
    print(
        f"Diagnostic GPKG:\n"
        f"{diagnostic_path}"
    )

    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 120)
    print(
        "CARTOGEN — BUILD LITHUANIAN "
        "ROAD CORRESPONDENCE"
    )
    print("=" * 120)

    summaries = []

    for window, city in WINDOWS.items():

        summary = process_city(
            window,
            city
        )

        summaries.append(
            summary
        )

    summary_df = pd.DataFrame(
        summaries
    )

    print()
    print("=" * 120)
    print(
        "GLOBAL ROAD CORRESPONDENCE SUMMARY"
    )
    print("=" * 120)

    display_columns = [
        "city",
        "source_count",
        "target_count",
        "candidate_pairs",
        "matched_source_pct",
        "matched_target_pct",
        "target_multiple_sources",
        "mean_source_coverage",
        "mean_target_coverage",
        "mean_semantic_score",
        "mean_match_score",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False,
            formatters={
                "matched_source_pct":
                    "{:.2f}".format,

                "matched_target_pct":
                    "{:.2f}".format,

                "mean_source_coverage":
                    "{:.3f}".format,

                "mean_target_coverage":
                    "{:.3f}".format,

                "mean_semantic_score":
                    "{:.3f}".format,

                "mean_match_score":
                    "{:.3f}".format,
            }
        )
    )

    summary_csv = (
        OUTPUT_DIR
        / "road_correspondence_summary.csv"
    )

    summary_json = (
        OUTPUT_DIR
        / "road_correspondence_summary.json"
    )

    summary_df.to_csv(
        summary_csv,
        index=False
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summaries,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 120)
    print(
        "SCRIPT 40 COMPLETE"
    )
    print("=" * 120)

    print()
    print(
        f"Summary CSV:\n"
        f"{summary_csv}"
    )

    print()
    print(
        f"Summary JSON:\n"
        f"{summary_json}"
    )


if __name__ == "__main__":
    main()