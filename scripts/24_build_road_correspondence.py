from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import geopandas as gpd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "os"
    / "development_windows"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cartogen"
    / "road_correspondence"
)

SUMMARY_FILE = (
    OUTPUT_ROOT
    / "road_correspondence_summary.json"
)

WINDOWS = [
    "london_central",
    "manchester_central",
    "birmingham_central",
]

SOURCE_LAYER = "roads_local"
TARGET_LAYER = "roads_regional"

# Search radius around detailed roads.
SEARCH_BUFFER_M = 30.0

# Buffer used to estimate how much of a local road
# follows a generalized regional road.
OVERLAP_BUFFER_M = 12.0

# Maximum acceptable geometry distance.
MAX_DISTANCE_M = 30.0

# Direction agreement threshold.
MAX_ANGLE_DIFFERENCE_DEG = 45.0

# Minimum buffered coverage of local road.
MIN_LOCAL_COVERAGE = 0.10

BOUNDARY_MARGIN_M = 30.0


# ============================================================
# GEOMETRY UTILITIES
# ============================================================

def ensure_id_column(
    gdf,
    name
):

    gdf = gdf.reset_index(
        drop=True
    ).copy()

    gdf[name] = np.arange(
        len(gdf),
        dtype=np.int64
    )

    return gdf


def remove_boundary_features(
    gdf,
    bounds,
    margin
):

    min_x, min_y, max_x, max_y = bounds

    centroids = gdf.geometry.centroid

    mask = (
        (centroids.x >= min_x + margin)
        & (centroids.x <= max_x - margin)
        & (centroids.y >= min_y + margin)
        & (centroids.y <= max_y - margin)
    )

    return gdf[
        mask
    ].copy()


def line_orientation(
    geometry
):

    """
    Approximate orientation using the vector from
    first to last coordinate.

    Orientation is axial, therefore 0 and 180 degrees
    represent the same direction.
    """

    if (
        geometry is None
        or geometry.is_empty
    ):
        return None

    if geometry.geom_type == "MultiLineString":

        if len(geometry.geoms) == 0:
            return None

        geometry = max(
            geometry.geoms,
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

    if (
        abs(dx) < 1e-12
        and abs(dy) < 1e-12
    ):
        return None

    angle = math.degrees(
        math.atan2(
            dy,
            dx
        )
    )

    angle = (
        angle + 180.0
    ) % 180.0

    return float(angle)


def angle_difference(
    angle_a,
    angle_b
):

    if (
        angle_a is None
        or angle_b is None
    ):
        return 90.0

    difference = abs(
        angle_a - angle_b
    )

    return float(
        min(
            difference,
            180.0 - difference
        )
    )


def calculate_metrics(
    local_geom,
    regional_geom
):

    local_length = float(
        local_geom.length
    )

    regional_length = float(
        regional_geom.length
    )

    geometry_distance = float(
        local_geom.distance(
            regional_geom
        )
    )

    centroid_distance = float(
        local_geom.centroid.distance(
            regional_geom.centroid
        )
    )

    # --------------------------------------------------------
    # Buffered line coverage
    #
    # Measure how much of the detailed/local line lies inside
    # a narrow buffer around the regional line.
    # --------------------------------------------------------

    regional_buffer = (
        regional_geom.buffer(
            OVERLAP_BUFFER_M
        )
    )

    covered_local = (
        local_geom.intersection(
            regional_buffer
        )
    )

    covered_length = float(
        covered_local.length
    )

    local_coverage = (
        covered_length
        / local_length
        if local_length > 0
        else 0.0
    )

    # Reverse coverage:
    # how much of the regional line is near this local road.

    local_buffer = (
        local_geom.buffer(
            OVERLAP_BUFFER_M
        )
    )

    covered_regional = (
        regional_geom.intersection(
            local_buffer
        )
    )

    regional_covered_length = float(
        covered_regional.length
    )

    regional_coverage = (
        regional_covered_length
        / regional_length
        if regional_length > 0
        else 0.0
    )

    local_angle = (
        line_orientation(
            local_geom
        )
    )

    regional_angle = (
        line_orientation(
            regional_geom
        )
    )

    angle_diff = (
        angle_difference(
            local_angle,
            regional_angle
        )
    )

    return {
        "local_length_m":
            local_length,

        "regional_length_m":
            regional_length,

        "geometry_distance_m":
            geometry_distance,

        "centroid_distance_m":
            centroid_distance,

        "covered_local_length_m":
            covered_length,

        "local_coverage":
            float(local_coverage),

        "regional_coverage":
            float(regional_coverage),

        "local_orientation_deg":
            local_angle,

        "regional_orientation_deg":
            regional_angle,

        "angle_difference_deg":
            angle_diff,
    }


# ============================================================
# MATCHING
# ============================================================

def build_correspondence(
    local,
    regional
):

    regional_sindex = (
        regional.sindex
    )

    pair_records = []

    total = len(local)

    print(
        f"  Matching "
        f"{total:,} local road segments..."
    )

    for counter, row in enumerate(
        local.itertuples(),
        start=1
    ):

        local_idx = int(
            row.local_road_idx
        )

        local_geom = (
            row.geometry
        )

        if (
            local_geom is None
            or local_geom.is_empty
            or local_geom.length <= 0
        ):
            continue

        search_geometry = (
            local_geom.buffer(
                SEARCH_BUFFER_M
            )
        )

        candidate_positions = list(
            regional_sindex.intersection(
                search_geometry.bounds
            )
        )

        accepted = []

        for position in (
            candidate_positions
        ):

            regional_row = (
                regional.iloc[
                    position
                ]
            )

            regional_idx = int(
                regional_row[
                    "regional_road_idx"
                ]
            )

            regional_geom = (
                regional_row.geometry
            )

            if (
                regional_geom is None
                or regional_geom.is_empty
                or regional_geom.length <= 0
            ):
                continue

            metrics = (
                calculate_metrics(
                    local_geom,
                    regional_geom
                )
            )

            distance_ok = (
                metrics[
                    "geometry_distance_m"
                ]
                <= MAX_DISTANCE_M
            )

            coverage_ok = (
                metrics[
                    "local_coverage"
                ]
                >= MIN_LOCAL_COVERAGE
            )

            angle_ok = (
                metrics[
                    "angle_difference_deg"
                ]
                <= MAX_ANGLE_DIFFERENCE_DEG
            )

            if not (
                distance_ok
                and (
                    coverage_ok
                    or angle_ok
                )
            ):
                continue

            distance_score = max(
                0.0,
                1.0
                - (
                    metrics[
                        "geometry_distance_m"
                    ]
                    / MAX_DISTANCE_M
                )
            )

            angle_score = max(
                0.0,
                1.0
                - (
                    metrics[
                        "angle_difference_deg"
                    ]
                    / 90.0
                )
            )

            score = (
                0.55
                * metrics[
                    "local_coverage"
                ]
                +
                0.20
                * metrics[
                    "regional_coverage"
                ]
                +
                0.15
                * distance_score
                +
                0.10
                * angle_score
            )

            accepted.append({
                "local_road_idx":
                    local_idx,

                "regional_road_idx":
                    regional_idx,

                **metrics,

                "match_score":
                    float(score),
            })

        if accepted:

            accepted.sort(
                key=lambda item:
                    item[
                        "match_score"
                    ],
                reverse=True
            )

            for rank, item in enumerate(
                accepted,
                start=1
            ):

                item[
                    "candidate_rank"
                ] = rank

                item[
                    "is_best_local_match"
                ] = (
                    rank == 1
                )

                pair_records.append(
                    item
                )

        if (
            counter % 2000 == 0
            or counter == total
        ):

            print(
                f"    "
                f"{counter:,}/"
                f"{total:,} processed"
            )

    return pd.DataFrame(
        pair_records
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize(
    local,
    regional,
    pairs
):

    if pairs.empty:

        return {
            "local_count":
                len(local),

            "regional_count":
                len(regional),

            "candidate_pairs":
                0,
        }

    best = pairs[
        pairs[
            "is_best_local_match"
        ]
    ].copy()

    matched_local = (
        best[
            "local_road_idx"
        ]
        .nunique()
    )

    matched_regional = (
        best[
            "regional_road_idx"
        ]
        .nunique()
    )

    contributors = (
        best.groupby(
            "regional_road_idx"
        )[
            "local_road_idx"
        ]
        .nunique()
    )

    return {

        "local_count":
            int(len(local)),

        "regional_count":
            int(len(regional)),

        "candidate_pairs":
            int(len(pairs)),

        "best_match_pairs":
            int(len(best)),

        "matched_local":
            int(matched_local),

        "unmatched_local":
            int(
                len(local)
                - matched_local
            ),

        "matched_local_pct":
            float(
                100.0
                * matched_local
                / len(local)
                if len(local)
                else 0.0
            ),

        "matched_regional":
            int(
                matched_regional
            ),

        "unmatched_regional":
            int(
                len(regional)
                - matched_regional
            ),

        "matched_regional_pct":
            float(
                100.0
                * matched_regional
                / len(regional)
                if len(regional)
                else 0.0
            ),

        "regional_with_one_source":
            int(
                (
                    contributors == 1
                ).sum()
            ),

        "regional_with_multiple_sources":
            int(
                (
                    contributors > 1
                ).sum()
            ),

        "mean_sources_per_regional":
            float(
                contributors.mean()
            ),

        "median_sources_per_regional":
            float(
                contributors.median()
            ),

        "max_sources_per_regional":
            int(
                contributors.max()
            ),

        "mean_local_coverage":
            float(
                best[
                    "local_coverage"
                ].mean()
            ),

        "median_local_coverage":
            float(
                best[
                    "local_coverage"
                ].median()
            ),

        "mean_geometry_distance_m":
            float(
                best[
                    "geometry_distance_m"
                ].mean()
            ),

        "median_geometry_distance_m":
            float(
                best[
                    "geometry_distance_m"
                ].median()
            ),

        "mean_angle_difference_deg":
            float(
                best[
                    "angle_difference_deg"
                ].mean()
            ),

        "median_angle_difference_deg":
            float(
                best[
                    "angle_difference_deg"
                ].median()
            ),

        "mean_match_score":
            float(
                best[
                    "match_score"
                ].mean()
            ),
    }


# ============================================================
# DIAGNOSTIC OUTPUT
# ============================================================

def save_diagnostics(
    output_file,
    local,
    regional,
    pairs
):

    if output_file.exists():
        output_file.unlink()

    best = pairs[
        pairs[
            "is_best_local_match"
        ]
    ].copy()

    matched_local_ids = set(
        best[
            "local_road_idx"
        ]
        .astype(int)
        .tolist()
    )

    matched_regional_ids = set(
        best[
            "regional_road_idx"
        ]
        .astype(int)
        .tolist()
    )

    local_output = (
        local.copy()
    )

    local_output[
        "matched"
    ] = (
        local_output[
            "local_road_idx"
        ]
        .isin(
            matched_local_ids
        )
    )

    regional_output = (
        regional.copy()
    )

    regional_output[
        "matched"
    ] = (
        regional_output[
            "regional_road_idx"
        ]
        .isin(
            matched_regional_ids
        )
    )

    contributor_counts = (
        best.groupby(
            "regional_road_idx"
        )[
            "local_road_idx"
        ]
        .nunique()
        .to_dict()
    )

    regional_output[
        "source_count"
    ] = (
        regional_output[
            "regional_road_idx"
        ]
        .map(
            contributor_counts
        )
        .fillna(0)
        .astype(int)
    )

    local_output.to_file(
        output_file,
        layer="local_roads",
        driver="GPKG",
        engine="pyogrio"
    )

    regional_output.to_file(
        output_file,
        layer="regional_roads",
        driver="GPKG",
        engine="pyogrio"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 120)
    print(
        "CARTOGEN — BUILD ROAD "
        "SPATIAL CORRESPONDENCE"
    )
    print("=" * 120)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    summaries = {}

    for window in WINDOWS:

        print()
        print("=" * 120)
        print(window)
        print("=" * 120)

        source = (
            DATA_ROOT
            / f"{window}.gpkg"
        )

        local = gpd.read_file(
            source,
            layer=SOURCE_LAYER,
            engine="pyogrio"
        )

        regional = gpd.read_file(
            source,
            layer=TARGET_LAYER,
            engine="pyogrio"
        )

        print(
            f"  Raw local roads: "
            f"{len(local):,}"
        )

        print(
            f"  Raw regional roads: "
            f"{len(regional):,}"
        )

        if (
            local.crs is None
            or regional.crs is None
        ):
            raise RuntimeError(
                "Missing CRS."
            )

        if (
            local.crs.to_epsg() != 27700
            or
            regional.crs.to_epsg() != 27700
        ):
            raise RuntimeError(
                "Expected EPSG:27700."
            )

        bounds = (
            local.total_bounds
        )

        local = (
            remove_boundary_features(
                local,
                bounds,
                BOUNDARY_MARGIN_M
            )
        )

        regional = (
            remove_boundary_features(
                regional,
                bounds,
                BOUNDARY_MARGIN_M
            )
        )

        local = ensure_id_column(
            local,
            "local_road_idx"
        )

        regional = ensure_id_column(
            regional,
            "regional_road_idx"
        )

        print(
            f"  Interior local roads: "
            f"{len(local):,}"
        )

        print(
            f"  Interior regional roads: "
            f"{len(regional):,}"
        )

        print(
            "  Building regional "
            "spatial index..."
        )

        pairs = (
            build_correspondence(
                local,
                regional
            )
        )

        print(
            f"\n  Candidate pairs: "
            f"{len(pairs):,}"
        )

        csv_file = (
            OUTPUT_ROOT
            / (
                f"{window}_"
                f"road_pairs.csv"
            )
        )

        pairs.to_csv(
            csv_file,
            index=False
        )

        summary = (
            summarize(
                local,
                regional,
                pairs
            )
        )

        summaries[
            window
        ] = summary

        print()
        print("-" * 120)
        print(
            "ROAD CORRESPONDENCE SUMMARY"
        )
        print("-" * 120)

        for key, value in (
            summary.items()
        ):

            if isinstance(
                value,
                float
            ):

                print(
                    f"  "
                    f"{key:<38}"
                    f"{value:,.3f}"
                )

            else:

                print(
                    f"  "
                    f"{key:<38}"
                    f"{value:,}"
                )

        diagnostic_file = (
            OUTPUT_ROOT
            / (
                f"{window}_"
                f"road_correspondence.gpkg"
            )
        )

        if not pairs.empty:

            save_diagnostics(
                diagnostic_file,
                local,
                regional,
                pairs
            )

        print(
            f"\n  CSV:\n"
            f"  {csv_file}"
        )

        print(
            f"\n  Diagnostic GPKG:\n"
            f"  {diagnostic_file}"
        )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summaries,
            file,
            indent=2
        )

    print()
    print("=" * 120)
    print("SCRIPT 24 COMPLETE")
    print("=" * 120)

    print(
        f"Summary:\n"
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()