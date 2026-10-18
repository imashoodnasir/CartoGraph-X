from pathlib import Path
import json
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
    / "road_correspondence_refined"
)

SUMMARY_FILE = (
    OUTPUT_ROOT
    / "refined_road_correspondence_summary.json"
)

WINDOWS = [
    "london_central",
    "manchester_central",
    "birmingham_central",
]

SOURCE_LAYER = "roads_local"
TARGET_LAYER = "roads_regional"


# ------------------------------------------------------------
# Matching parameters
# ------------------------------------------------------------

SEARCH_DISTANCE_M = 40.0

CORRIDOR_BUFFER_M = 15.0

MIN_LOCAL_COVERAGE = 0.15

MIN_REGIONAL_COVERAGE = 0.10

MAX_GEOMETRY_DISTANCE_M = 40.0

BOUNDARY_MARGIN_M = 40.0


# ============================================================
# HELPERS
# ============================================================

def ensure_ids(
    gdf,
    column
):

    gdf = (
        gdf
        .reset_index(drop=True)
        .copy()
    )

    gdf[column] = np.arange(
        len(gdf),
        dtype=np.int64
    )

    return gdf


def remove_boundary_features(
    gdf,
    bounds
):

    min_x, min_y, max_x, max_y = bounds

    centroids = (
        gdf.geometry.centroid
    )

    mask = (
        (
            centroids.x
            >= min_x + BOUNDARY_MARGIN_M
        )
        &
        (
            centroids.x
            <= max_x - BOUNDARY_MARGIN_M
        )
        &
        (
            centroids.y
            >= min_y + BOUNDARY_MARGIN_M
        )
        &
        (
            centroids.y
            <= max_y - BOUNDARY_MARGIN_M
        )
    )

    return gdf[
        mask
    ].copy()


def coverage(
    source,
    target,
    buffer_distance
):

    if source.length <= 0:
        return 0.0

    target_buffer = (
        target.buffer(
            buffer_distance
        )
    )

    intersection = (
        source.intersection(
            target_buffer
        )
    )

    return float(
        min(
            1.0,
            intersection.length
            / source.length
        )
    )


def pair_metrics(
    local_geom,
    regional_geom
):

    local_length = float(
        local_geom.length
    )

    regional_length = float(
        regional_geom.length
    )

    local_coverage = coverage(
        local_geom,
        regional_geom,
        CORRIDOR_BUFFER_M
    )

    regional_coverage = coverage(
        regional_geom,
        local_geom,
        CORRIDOR_BUFFER_M
    )

    distance = float(
        local_geom.distance(
            regional_geom
        )
    )

    hausdorff = float(
        local_geom.hausdorff_distance(
            regional_geom
        )
    )

    if (
        local_length > 0
        and regional_length > 0
    ):

        length_ratio = (
            min(
                local_length,
                regional_length
            )
            /
            max(
                local_length,
                regional_length
            )
        )

    else:

        length_ratio = 0.0

    return {
        "local_length_m":
            local_length,

        "regional_length_m":
            regional_length,

        "local_coverage":
            local_coverage,

        "regional_coverage":
            regional_coverage,

        "geometry_distance_m":
            distance,

        "hausdorff_distance_m":
            hausdorff,

        "length_compatibility":
            float(length_ratio),
    }


def score_pair(
    metrics
):

    distance_score = max(
        0.0,
        1.0
        -
        (
            metrics[
                "geometry_distance_m"
            ]
            /
            MAX_GEOMETRY_DISTANCE_M
        )
    )

    # Hausdorff is deliberately given
    # only moderate influence because
    # differently segmented roads can
    # have distant endpoints.

    hausdorff_score = max(
        0.0,
        1.0
        -
        min(
            metrics[
                "hausdorff_distance_m"
            ],
            150.0
        )
        / 150.0
    )

    return float(

        0.40
        * metrics[
            "local_coverage"
        ]

        +

        0.25
        * metrics[
            "regional_coverage"
        ]

        +

        0.15
        * distance_score

        +

        0.10
        * hausdorff_score

        +

        0.10
        * metrics[
            "length_compatibility"
        ]
    )


# ============================================================
# BUILD ALL PLAUSIBLE PAIRS
# ============================================================

def build_candidate_pairs(
    local,
    regional
):

    regional_index = (
        regional.sindex
    )

    records = []

    total = len(local)

    print(
        f"  Evaluating "
        f"{total:,} local roads..."
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

        search = (
            local_geom.buffer(
                SEARCH_DISTANCE_M
            )
        )

        candidates = list(
            regional_index.intersection(
                search.bounds
            )
        )

        for position in candidates:

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

            metrics = pair_metrics(
                local_geom,
                regional_geom
            )

            if (
                metrics[
                    "geometry_distance_m"
                ]
                > MAX_GEOMETRY_DISTANCE_M
            ):
                continue

            # At least one representation
            # must exhibit corridor support.
            if (
                metrics[
                    "local_coverage"
                ]
                < MIN_LOCAL_COVERAGE
                and
                metrics[
                    "regional_coverage"
                ]
                < MIN_REGIONAL_COVERAGE
            ):
                continue

            score = score_pair(
                metrics
            )

            records.append({

                "local_road_idx":
                    local_idx,

                "regional_road_idx":
                    regional_idx,

                **metrics,

                "match_score":
                    score,
            })

        if (
            counter % 2000 == 0
            or counter == total
        ):

            print(
                f"    "
                f"{counter:,}/"
                f"{total:,}"
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# BIDIRECTIONAL ASSIGNMENT
# ============================================================

def assign_matches(
    pairs
):

    if pairs.empty:
        return pairs

    pairs = (
        pairs
        .sort_values(
            "match_score",
            ascending=False
        )
        .copy()
    )

    # Best target for each detailed road.

    local_rank = (
        pairs
        .groupby(
            "local_road_idx"
        )[
            "match_score"
        ]
        .rank(
            method="first",
            ascending=False
        )
    )

    pairs[
        "local_rank"
    ] = local_rank.astype(int)

    # Best detailed contributor for each
    # regional road.

    regional_rank = (
        pairs
        .groupby(
            "regional_road_idx"
        )[
            "match_score"
        ]
        .rank(
            method="first",
            ascending=False
        )
    )

    pairs[
        "regional_rank"
    ] = regional_rank.astype(int)

    # --------------------------------------------------------
    # Retain a correspondence when:
    #
    # 1. it is the best regional target for the local road,
    # OR
    # 2. it is among strong contributors to a regional road.
    #
    # This avoids losing target roads simply because their
    # segmentation differs from the detailed representation.
    # --------------------------------------------------------

    pairs[
        "selected"
    ] = (
        (
            pairs[
                "local_rank"
            ]
            == 1
        )
        |
        (
            (
                pairs[
                    "regional_rank"
                ]
                <= 3
            )
            &
            (
                pairs[
                    "match_score"
                ]
                >= 0.25
            )
        )
    )

    return pairs


# ============================================================
# SUMMARY
# ============================================================

def summarize(
    local,
    regional,
    pairs
):

    selected = pairs[
        pairs["selected"]
    ].copy()

    matched_local = (
        selected[
            "local_road_idx"
        ]
        .nunique()
    )

    matched_regional = (
        selected[
            "regional_road_idx"
        ]
        .nunique()
    )

    contributors = (
        selected
        .groupby(
            "regional_road_idx"
        )[
            "local_road_idx"
        ]
        .nunique()
    )

    if len(contributors):

        one_source = int(
            (
                contributors == 1
            ).sum()
        )

        multiple_source = int(
            (
                contributors > 1
            ).sum()
        )

        mean_sources = float(
            contributors.mean()
        )

        median_sources = float(
            contributors.median()
        )

        max_sources = int(
            contributors.max()
        )

    else:

        one_source = 0
        multiple_source = 0
        mean_sources = 0.0
        median_sources = 0.0
        max_sources = 0

    return {

        "local_count":
            int(len(local)),

        "regional_count":
            int(len(regional)),

        "candidate_pairs":
            int(len(pairs)),

        "selected_pairs":
            int(len(selected)),

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
            int(matched_regional),

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
            one_source,

        "regional_with_multiple_sources":
            multiple_source,

        "mean_sources_per_regional":
            mean_sources,

        "median_sources_per_regional":
            median_sources,

        "max_sources_per_regional":
            max_sources,

        "mean_local_coverage":
            float(
                selected[
                    "local_coverage"
                ].mean()
                if len(selected)
                else 0.0
            ),

        "median_local_coverage":
            float(
                selected[
                    "local_coverage"
                ].median()
                if len(selected)
                else 0.0
            ),

        "mean_regional_coverage":
            float(
                selected[
                    "regional_coverage"
                ].mean()
                if len(selected)
                else 0.0
            ),

        "median_regional_coverage":
            float(
                selected[
                    "regional_coverage"
                ].median()
                if len(selected)
                else 0.0
            ),

        "mean_geometry_distance_m":
            float(
                selected[
                    "geometry_distance_m"
                ].mean()
                if len(selected)
                else 0.0
            ),

        "median_geometry_distance_m":
            float(
                selected[
                    "geometry_distance_m"
                ].median()
                if len(selected)
                else 0.0
            ),

        "mean_match_score":
            float(
                selected[
                    "match_score"
                ].mean()
                if len(selected)
                else 0.0
            ),
    }


# ============================================================
# DIAGNOSTICS
# ============================================================

def save_diagnostics(
    output,
    local,
    regional,
    pairs
):

    if output.exists():
        output.unlink()

    selected = pairs[
        pairs["selected"]
    ].copy()

    matched_local = set(
        selected[
            "local_road_idx"
        ]
        .astype(int)
    )

    matched_regional = set(
        selected[
            "regional_road_idx"
        ]
        .astype(int)
    )

    local_out = (
        local.copy()
    )

    local_out[
        "matched"
    ] = (
        local_out[
            "local_road_idx"
        ]
        .isin(
            matched_local
        )
    )

    regional_out = (
        regional.copy()
    )

    regional_out[
        "matched"
    ] = (
        regional_out[
            "regional_road_idx"
        ]
        .isin(
            matched_regional
        )
    )

    counts = (
        selected
        .groupby(
            "regional_road_idx"
        )[
            "local_road_idx"
        ]
        .nunique()
        .to_dict()
    )

    regional_out[
        "source_count"
    ] = (
        regional_out[
            "regional_road_idx"
        ]
        .map(counts)
        .fillna(0)
        .astype(int)
    )

    local_out.to_file(
        output,
        layer="local_roads",
        driver="GPKG",
        engine="pyogrio"
    )

    regional_out.to_file(
        output,
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
        "CARTOGEN — REFINE ROAD "
        "CORRESPONDENCE"
    )
    print("=" * 120)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    all_summaries = {}

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
            f"Raw local roads: "
            f"{len(local):,}"
        )

        print(
            f"Raw regional roads: "
            f"{len(regional):,}"
        )

        bounds = (
            local.total_bounds
        )

        local = (
            remove_boundary_features(
                local,
                bounds
            )
        )

        regional = (
            remove_boundary_features(
                regional,
                bounds
            )
        )

        local = ensure_ids(
            local,
            "local_road_idx"
        )

        regional = ensure_ids(
            regional,
            "regional_road_idx"
        )

        print(
            f"Interior local roads: "
            f"{len(local):,}"
        )

        print(
            f"Interior regional roads: "
            f"{len(regional):,}"
        )

        print(
            "Building candidate "
            "correspondences..."
        )

        pairs = (
            build_candidate_pairs(
                local,
                regional
            )
        )

        print(
            f"\nCandidate pairs: "
            f"{len(pairs):,}"
        )

        pairs = (
            assign_matches(
                pairs
            )
        )

        selected_count = int(
            pairs[
                "selected"
            ].sum()
            if len(pairs)
            else 0
        )

        print(
            f"Selected pairs: "
            f"{selected_count:,}"
        )

        summary = summarize(
            local,
            regional,
            pairs
        )

        all_summaries[
            window
        ] = summary

        print()
        print("-" * 120)
        print(
            "REFINED ROAD CORRESPONDENCE SUMMARY"
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
                    f"{key:<40}"
                    f"{value:,.3f}"
                )

            else:

                print(
                    f"{key:<40}"
                    f"{value:,}"
                )

        csv_file = (
            OUTPUT_ROOT
            / (
                f"{window}_"
                f"refined_road_pairs.csv"
            )
        )

        pairs.to_csv(
            csv_file,
            index=False
        )

        diagnostic = (
            OUTPUT_ROOT
            / (
                f"{window}_"
                f"refined_road_correspondence.gpkg"
            )
        )

        if not pairs.empty:

            save_diagnostics(
                diagnostic,
                local,
                regional,
                pairs
            )

        print(
            f"\nCSV:\n{csv_file}"
        )

        print(
            f"\nDiagnostic GPKG:\n"
            f"{diagnostic}"
        )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_summaries,
            file,
            indent=2
        )

    print()
    print("=" * 120)
    print("SCRIPT 25 COMPLETE")
    print("=" * 120)

    print(
        f"Summary:\n"
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()