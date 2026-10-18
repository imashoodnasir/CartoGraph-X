from pathlib import Path
import json
import math
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.strtree import STRtree


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "validated_pairs"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "building_correspondence"
)

WINDOW_JSON = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "lithuania"
    / "windows"
    / "lithuania_development_windows.json"
)

SOURCE_LAYER = "source_buildings"
TARGET_LAYER = "target_buildings"

# Ignore features touching the outer edge because their
# geometries may be clipped by the extraction window.
EDGE_BUFFER_M = 20.0

# Search distance for non-overlapping but displaced targets.
PROXIMITY_SEARCH_M = 25.0

# Minimum overlap conditions.
MIN_INTERSECTION_AREA_M2 = 1.0
MIN_SOURCE_COVERAGE = 0.05

# Fallback proximity threshold.
MAX_PROXIMITY_M = 15.0

# Progress output.
PROGRESS_INTERVAL = 2000


# ============================================================
# HELPERS
# ============================================================

def safe_divide(a, b):

    if b is None or b == 0:
        return 0.0

    return float(a) / float(b)


def get_window_bbox(window):

    return (
        float(window["xmin"]),
        float(window["ymin"]),
        float(window["xmax"]),
        float(window["ymax"])
    )


def create_interior_polygon(bbox):

    xmin, ymin, xmax, ymax = bbox

    return box(
        xmin + EDGE_BUFFER_M,
        ymin + EDGE_BUFFER_M,
        xmax - EDGE_BUFFER_M,
        ymax - EDGE_BUFFER_M
    )


def prepare_geodataframe(gdf):

    gdf = gdf.copy()

    # Remove empty/null geometries.
    gdf = gdf[
        gdf.geometry.notna()
        & (~gdf.geometry.is_empty)
    ].copy()

    # Repair invalid geometries if any remain.
    invalid = ~gdf.geometry.is_valid

    if invalid.any():

        print(
            f"  Repairing "
            f"{invalid.sum():,} invalid geometries..."
        )

        gdf.loc[
            invalid,
            "geometry"
        ] = (
            gdf.loc[
                invalid,
                "geometry"
            ]
            .buffer(0)
        )

    gdf = gdf[
        gdf.geometry.notna()
        & (~gdf.geometry.is_empty)
        & gdf.geometry.is_valid
    ].copy()

    gdf.reset_index(
        drop=True,
        inplace=True
    )

    return gdf


def filter_interior(
    gdf,
    interior
):

    # Use centroid to determine whether the feature belongs
    # to the interior evaluation region.
    centroids = gdf.geometry.centroid

    mask = centroids.within(
        interior
    )

    return (
        gdf.loc[mask]
        .copy()
        .reset_index(drop=True)
    )


def geometry_metrics(
    source_geom,
    target_geom
):

    source_area = source_geom.area
    target_area = target_geom.area

    intersection = (
        source_geom
        .intersection(target_geom)
    )

    intersection_area = (
        intersection.area
        if not intersection.is_empty
        else 0.0
    )

    union_area = (
        source_area
        + target_area
        - intersection_area
    )

    iou = safe_divide(
        intersection_area,
        union_area
    )

    source_coverage = safe_divide(
        intersection_area,
        source_area
    )

    target_coverage = safe_divide(
        intersection_area,
        target_area
    )

    distance = source_geom.distance(
        target_geom
    )

    centroid_distance = (
        source_geom.centroid
        .distance(
            target_geom.centroid
        )
    )

    area_ratio = safe_divide(
        target_area,
        source_area
    )

    return {
        "intersection_area_m2":
            intersection_area,

        "iou":
            iou,

        "source_coverage":
            source_coverage,

        "target_coverage":
            target_coverage,

        "geometry_distance_m":
            distance,

        "centroid_distance_m":
            centroid_distance,

        "source_area_m2":
            source_area,

        "target_area_m2":
            target_area,

        "area_ratio":
            area_ratio,
    }


def candidate_score(metrics):

    # Higher is better.
    #
    # Source coverage receives the strongest weight because
    # a detailed building may become part of a larger
    # generalized target polygon.

    overlap_score = (
        0.50
        * metrics["source_coverage"]
    )

    iou_score = (
        0.25
        * metrics["iou"]
    )

    target_score = (
        0.15
        * metrics["target_coverage"]
    )

    distance_score = (
        0.10
        * math.exp(
            -metrics["geometry_distance_m"]
            / 10.0
        )
    )

    return (
        overlap_score
        + iou_score
        + target_score
        + distance_score
    )


# ============================================================
# CORRESPONDENCE
# ============================================================

def build_correspondence(
    source,
    target
):

    target_geometries = (
        target.geometry.tolist()
    )

    tree = STRtree(
        target_geometries
    )

    candidate_records = []
    best_records = []

    print(
        "  Building target spatial index..."
    )

    print(
        f"  Matching "
        f"{len(source):,} source buildings..."
    )

    start_time = time.time()

    for source_idx, source_row in source.iterrows():

        source_geom = (
            source_row.geometry
        )

        # ----------------------------------------------------
        # FIRST SEARCH:
        # direct / nearby spatial candidates
        # ----------------------------------------------------

        search_geometry = (
            source_geom.buffer(
                PROXIMITY_SEARCH_M
            )
        )

        candidate_indices = tree.query(
            search_geometry
        )

        valid_candidates = []

        for target_idx in candidate_indices:

            target_idx = int(
                target_idx
            )

            target_geom = (
                target_geometries[
                    target_idx
                ]
            )

            metrics = geometry_metrics(
                source_geom,
                target_geom
            )

            overlap_valid = (
                metrics[
                    "intersection_area_m2"
                ]
                >= MIN_INTERSECTION_AREA_M2
                and
                metrics[
                    "source_coverage"
                ]
                >= MIN_SOURCE_COVERAGE
            )

            proximity_valid = (
                metrics[
                    "geometry_distance_m"
                ]
                <= MAX_PROXIMITY_M
            )

            if not (
                overlap_valid
                or proximity_valid
            ):
                continue

            if (
                metrics[
                    "intersection_area_m2"
                ]
                > 0
            ):
                match_type = "overlap"
            else:
                match_type = "proximity"

            score = candidate_score(
                metrics
            )

            record = {
                "source_idx":
                    int(source_idx),

                "target_idx":
                    target_idx,

                "match_type":
                    match_type,

                "match_score":
                    score,

                **metrics,
            }

            valid_candidates.append(
                record
            )

            candidate_records.append(
                record
            )

        # ----------------------------------------------------
        # SELECT BEST TARGET FOR THIS SOURCE
        # ----------------------------------------------------

        if valid_candidates:

            valid_candidates.sort(
                key=lambda x: (
                    x["match_score"],
                    x["source_coverage"],
                    x["iou"]
                ),
                reverse=True
            )

            best = (
                valid_candidates[0]
                .copy()
            )

            best[
                "candidate_count"
            ] = len(
                valid_candidates
            )

            best_records.append(
                best
            )

        if (
            (source_idx + 1)
            % PROGRESS_INTERVAL
            == 0
            or
            source_idx + 1
            == len(source)
        ):

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"    "
                f"{source_idx + 1:,}/"
                f"{len(source):,} "
                f"processed "
                f"({elapsed:.1f}s)"
            )

    candidate_df = pd.DataFrame(
        candidate_records
    )

    best_df = pd.DataFrame(
        best_records
    )

    return (
        candidate_df,
        best_df
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize_correspondence(
    source,
    target,
    candidate_df,
    best_df
):

    source_count = len(source)
    target_count = len(target)

    if best_df.empty:

        return {
            "source_count":
                source_count,

            "target_count":
                target_count,

            "candidate_pairs":
                len(candidate_df),

            "best_match_pairs":
                0,

            "matched_source":
                0,

            "unmatched_source":
                source_count,

            "matched_source_pct":
                0.0,

            "matched_target":
                0,

            "unmatched_target":
                target_count,

            "matched_target_pct":
                0.0,
        }

    matched_source = (
        best_df[
            "source_idx"
        ]
        .nunique()
    )

    matched_target = (
        best_df[
            "target_idx"
        ]
        .nunique()
    )

    target_source_counts = (
        best_df
        .groupby(
            "target_idx"
        )
        .size()
    )

    one_source = int(
        (
            target_source_counts
            == 1
        ).sum()
    )

    multiple_source = int(
        (
            target_source_counts
            > 1
        ).sum()
    )

    overlap_matches = int(
        (
            best_df[
                "match_type"
            ]
            == "overlap"
        ).sum()
    )

    proximity_matches = int(
        (
            best_df[
                "match_type"
            ]
            == "proximity"
        ).sum()
    )

    summary = {

        "source_count":
            source_count,

        "target_count":
            target_count,

        "candidate_pairs":
            len(candidate_df),

        "best_match_pairs":
            len(best_df),

        "matched_source":
            int(matched_source),

        "unmatched_source":
            int(
                source_count
                - matched_source
            ),

        "matched_source_pct":
            100.0
            * safe_divide(
                matched_source,
                source_count
            ),

        "matched_target":
            int(matched_target),

        "unmatched_target":
            int(
                target_count
                - matched_target
            ),

        "matched_target_pct":
            100.0
            * safe_divide(
                matched_target,
                target_count
            ),

        "target_with_one_source":
            one_source,

        "target_with_multiple_sources":
            multiple_source,

        "multi_source_pct":
            100.0
            * safe_divide(
                multiple_source,
                matched_target
            ),

        "mean_sources_per_target":
            float(
                target_source_counts.mean()
            ),

        "median_sources_per_target":
            float(
                target_source_counts.median()
            ),

        "max_sources_per_target":
            int(
                target_source_counts.max()
            ),

        "mean_best_match_iou":
            float(
                best_df[
                    "iou"
                ].mean()
            ),

        "median_best_match_iou":
            float(
                best_df[
                    "iou"
                ].median()
            ),

        "mean_source_coverage":
            float(
                best_df[
                    "source_coverage"
                ].mean()
            ),

        "median_source_coverage":
            float(
                best_df[
                    "source_coverage"
                ].median()
            ),

        "mean_target_coverage":
            float(
                best_df[
                    "target_coverage"
                ].mean()
            ),

        "median_target_coverage":
            float(
                best_df[
                    "target_coverage"
                ].median()
            ),

        "mean_geometry_distance_m":
            float(
                best_df[
                    "geometry_distance_m"
                ].mean()
            ),

        "median_geometry_distance_m":
            float(
                best_df[
                    "geometry_distance_m"
                ].median()
            ),

        "mean_centroid_distance_m":
            float(
                best_df[
                    "centroid_distance_m"
                ].mean()
            ),

        "overlap_matches":
            overlap_matches,

        "proximity_matches":
            proximity_matches,
    }

    return summary


# ============================================================
# DIAGNOSTIC OUTPUT
# ============================================================

def save_diagnostic_geopackage(
    output_path,
    source,
    target,
    best_df
):

    if output_path.exists():
        output_path.unlink()

    source_output = (
        source.copy()
    )

    source_output[
        "source_idx"
    ] = np.arange(
        len(source_output)
    )

    target_output = (
        target.copy()
    )

    target_output[
        "target_idx"
    ] = np.arange(
        len(target_output)
    )

    if not best_df.empty:

        source_match = (
            best_df[
                [
                    "source_idx",
                    "target_idx",
                    "match_type",
                    "match_score",
                    "iou",
                    "source_coverage",
                    "target_coverage",
                    "geometry_distance_m",
                    "centroid_distance_m",
                    "area_ratio"
                ]
            ]
            .copy()
        )

        source_output = (
            source_output.merge(
                source_match,
                on="source_idx",
                how="left"
            )
        )

        target_counts = (
            best_df
            .groupby(
                "target_idx"
            )
            .size()
            .rename(
                "source_count"
            )
            .reset_index()
        )

        target_output = (
            target_output.merge(
                target_counts,
                on="target_idx",
                how="left"
            )
        )

        target_output[
            "source_count"
        ] = (
            target_output[
                "source_count"
            ]
            .fillna(0)
            .astype(int)
        )

    source_output.to_file(
        output_path,
        layer="source_buildings",
        driver="GPKG",
        engine="pyogrio"
    )

    target_output.to_file(
        output_path,
        layer="target_buildings",
        driver="GPKG",
        engine="pyogrio"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 120)
    print(
        "CARTOGEN — BUILD LITHUANIAN "
        "BUILDING CORRESPONDENCE"
    )
    print("=" * 120)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        WINDOW_JSON,
        "r",
        encoding="utf-8"
    ) as file:

        windows = json.load(
            file
        )

    global_summary = []

    for window in windows:

        name = window["window"]
        city = window["city"]

        print()
        print("=" * 120)
        print(
            f"{name} ({city})"
        )
        print("=" * 120)

        input_path = (
            INPUT_ROOT
            / f"{name}_validated.gpkg"
        )

        source = gpd.read_file(
            input_path,
            layer=SOURCE_LAYER,
            engine="pyogrio"
        )

        target = gpd.read_file(
            input_path,
            layer=TARGET_LAYER,
            engine="pyogrio"
        )

        print(
            f"Raw source buildings: "
            f"{len(source):,}"
        )

        print(
            f"Raw target buildings: "
            f"{len(target):,}"
        )

        source = prepare_geodataframe(
            source
        )

        target = prepare_geodataframe(
            target
        )

        bbox = get_window_bbox(
            window
        )

        interior = create_interior_polygon(
            bbox
        )

        source = filter_interior(
            source,
            interior
        )

        target = filter_interior(
            target,
            interior
        )

        print(
            f"Interior source buildings: "
            f"{len(source):,}"
        )

        print(
            f"Interior target buildings: "
            f"{len(target):,}"
        )

        (
            candidate_df,
            best_df
        ) = build_correspondence(
            source,
            target
        )

        print()
        print(
            f"Candidate pairs: "
            f"{len(candidate_df):,}"
        )

        summary = summarize_correspondence(
            source,
            target,
            candidate_df,
            best_df
        )

        summary[
            "window"
        ] = name

        summary[
            "city"
        ] = city

        global_summary.append(
            summary
        )

        print()
        print("-" * 120)
        print(
            "BUILDING CORRESPONDENCE SUMMARY"
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
                    f"{key:<40} "
                    f"{value:,.3f}"
                )

            else:

                print(
                    f"{key:<40} "
                    f"{value:,}"
                )

        # ----------------------------------------------------
        # SAVE CSV
        # ----------------------------------------------------

        csv_path = (
            OUTPUT_ROOT
            / (
                f"{name}_"
                f"building_pairs.csv"
            )
        )

        best_df.to_csv(
            csv_path,
            index=False
        )

        # Save all candidate pairs too.
        candidate_path = (
            OUTPUT_ROOT
            / (
                f"{name}_"
                f"building_candidates.csv"
            )
        )

        candidate_df.to_csv(
            candidate_path,
            index=False
        )

        # ----------------------------------------------------
        # SAVE DIAGNOSTIC GPKG
        # ----------------------------------------------------

        gpkg_path = (
            OUTPUT_ROOT
            / (
                f"{name}_"
                f"building_correspondence.gpkg"
            )
        )

        save_diagnostic_geopackage(
            gpkg_path,
            source,
            target,
            best_df
        )

        print()
        print("Best correspondence CSV:")
        print(csv_path)

        print()
        print("Candidate correspondence CSV:")
        print(candidate_path)

        print()
        print("Diagnostic GPKG:")
        print(gpkg_path)

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        global_summary
    )

    summary_csv = (
        OUTPUT_ROOT
        / "building_correspondence_summary.csv"
    )

    summary_json = (
        OUTPUT_ROOT
        / "building_correspondence_summary.json"
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
            global_summary,
            file,
            indent=2
        )

    print()
    print("=" * 120)
    print(
        "GLOBAL BUILDING CORRESPONDENCE"
    )
    print("=" * 120)

    display_columns = [
        "city",
        "source_count",
        "target_count",
        "matched_source_pct",
        "matched_target_pct",
        "multi_source_pct",
        "mean_best_match_iou",
        "mean_source_coverage",
        "mean_geometry_distance_m"
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("SCRIPT 37 COMPLETE")
    print("=" * 120)

    print()
    print("Summary CSV:")
    print(summary_csv)

    print()
    print("Summary JSON:")
    print(summary_json)


if __name__ == "__main__":
    main()