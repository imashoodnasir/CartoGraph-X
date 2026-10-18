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
    / "building_correspondence"
)

SUMMARY_FILE = (
    OUTPUT_ROOT
    / "building_correspondence_summary.json"
)


WINDOWS = [
    "london_central",
    "manchester_central",
    "birmingham_central",
]


# ------------------------------------------------------------
# Matching parameters
# ------------------------------------------------------------

# A generalized building may be slightly displaced relative
# to its detailed source geometry.
SEARCH_BUFFER_M = 20.0

# Minimum intersection area to accept a true overlap.
MIN_INTERSECTION_M2 = 1.0

# Minimum fraction of the LOCAL building covered by a
# generalized building.
MIN_LOCAL_COVERAGE = 0.05

# If there is no useful overlap, permit a nearby generalized
# building to become a proximity candidate.
MAX_PROXIMITY_M = 20.0

# Prevent boundary-clipped geometries from contaminating
# correspondence statistics.
BOUNDARY_MARGIN_M = 25.0


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ensure_id_column(
    gdf,
    prefix
):

    """
    Create our own stable row identifier.

    We deliberately do not rely on OS IDs because previous
    experiments showed that identifiers are not preserved
    across representations.
    """

    gdf = gdf.reset_index(
        drop=True
    ).copy()

    gdf[
        f"{prefix}_idx"
    ] = np.arange(
        len(gdf),
        dtype=np.int64
    )

    return gdf


def remove_boundary_features(
    gdf,
    bounds,
    margin
):

    """
    Exclude features touching the edge of the extracted
    6x6 km development window.

    Those geometries were clipped during extraction and
    therefore should not be used for correspondence labels.
    """

    min_x, min_y, max_x, max_y = bounds

    inner_min_x = min_x + margin
    inner_min_y = min_y + margin
    inner_max_x = max_x - margin
    inner_max_y = max_y - margin

    centroids = gdf.geometry.centroid

    mask = (
        (centroids.x >= inner_min_x)
        & (centroids.x <= inner_max_x)
        & (centroids.y >= inner_min_y)
        & (centroids.y <= inner_max_y)
    )

    return gdf[
        mask
    ].copy()


def calculate_pair_metrics(
    local_geom,
    district_geom
):

    intersection = (
        local_geom.intersection(
            district_geom
        )
    )

    intersection_area = (
        intersection.area
    )

    local_area = (
        local_geom.area
    )

    district_area = (
        district_geom.area
    )

    union_area = (
        local_geom.union(
            district_geom
        ).area
    )

    local_coverage = (
        intersection_area
        / local_area
        if local_area > 0
        else 0.0
    )

    district_coverage = (
        intersection_area
        / district_area
        if district_area > 0
        else 0.0
    )

    iou = (
        intersection_area
        / union_area
        if union_area > 0
        else 0.0
    )

    centroid_distance = (
        local_geom.centroid.distance(
            district_geom.centroid
        )
    )

    boundary_distance = (
        local_geom.distance(
            district_geom
        )
    )

    return {
        "intersection_area_m2":
            float(intersection_area),

        "local_area_m2":
            float(local_area),

        "district_area_m2":
            float(district_area),

        "local_coverage":
            float(local_coverage),

        "district_coverage":
            float(district_coverage),

        "iou":
            float(iou),

        "centroid_distance_m":
            float(centroid_distance),

        "geometry_distance_m":
            float(boundary_distance),
    }


# ============================================================
# CORRESPONDENCE
# ============================================================

def build_correspondence(
    local,
    district
):

    print(
        "  Building spatial index..."
    )

    district_sindex = (
        district.sindex
    )

    pair_records = []

    local_with_candidates = set()
    district_with_candidates = set()

    total_local = len(local)

    print(
        f"  Matching "
        f"{total_local:,} "
        f"local buildings..."
    )

    for position, local_row in enumerate(
        local.itertuples(),
        start=1
    ):

        local_idx = (
            local_row.local_idx
        )

        local_geom = (
            local_row.geometry
        )

        if (
            local_geom is None
            or local_geom.is_empty
        ):
            continue

        search_geom = (
            local_geom.buffer(
                SEARCH_BUFFER_M
            )
        )

        candidate_positions = list(
            district_sindex.intersection(
                search_geom.bounds
            )
        )

        accepted = []

        for district_position in (
            candidate_positions
        ):

            district_row = (
                district.iloc[
                    district_position
                ]
            )

            district_idx = int(
                district_row[
                    "district_idx"
                ]
            )

            district_geom = (
                district_row.geometry
            )

            if (
                district_geom is None
                or district_geom.is_empty
            ):
                continue

            metrics = (
                calculate_pair_metrics(
                    local_geom,
                    district_geom
                )
            )

            overlap_accept = (
                metrics[
                    "intersection_area_m2"
                ]
                >= MIN_INTERSECTION_M2

                and

                metrics[
                    "local_coverage"
                ]
                >= MIN_LOCAL_COVERAGE
            )

            proximity_accept = (
                metrics[
                    "geometry_distance_m"
                ]
                <= MAX_PROXIMITY_M
            )

            if not (
                overlap_accept
                or proximity_accept
            ):
                continue

            # -----------------------------------------
            # Matching score
            #
            # Strong overlap is preferred.
            # Proximity contributes only a smaller
            # secondary component.
            # -----------------------------------------

            distance_score = max(
                0.0,
                1.0
                - (
                    metrics[
                        "geometry_distance_m"
                    ]
                    / MAX_PROXIMITY_M
                )
            )

            score = (
                0.55
                * metrics[
                    "local_coverage"
                ]
                +
                0.25
                * metrics["iou"]
                +
                0.20
                * distance_score
            )

            accepted.append({
                "local_idx":
                    int(local_idx),

                "district_idx":
                    district_idx,

                **metrics,

                "match_score":
                    float(score),

                "match_basis":
                    (
                        "overlap"
                        if overlap_accept
                        else "proximity"
                    ),
            })

        if accepted:

            # Sort candidates for each detailed building.
            accepted.sort(
                key=lambda x:
                    x["match_score"],
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

                local_with_candidates.add(
                    item["local_idx"]
                )

                district_with_candidates.add(
                    item["district_idx"]
                )

        if (
            position % 2000 == 0
            or position == total_local
        ):

            print(
                f"    "
                f"{position:,}/"
                f"{total_local:,} "
                f"processed"
            )

    pairs = pd.DataFrame(
        pair_records
    )

    return (
        pairs,
        local_with_candidates,
        district_with_candidates,
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize_correspondence(
    local,
    district,
    pairs
):

    if pairs.empty:

        return {
            "local_count":
                len(local),

            "district_count":
                len(district),

            "candidate_pairs":
                0,
        }

    best_pairs = pairs[
        pairs[
            "is_best_local_match"
        ]
    ].copy()

    local_matched = (
        best_pairs[
            "local_idx"
        ]
        .nunique()
    )

    district_matched = (
        best_pairs[
            "district_idx"
        ]
        .nunique()
    )

    local_unmatched = (
        len(local)
        - local_matched
    )

    district_unmatched = (
        len(district)
        - district_matched
    )

    contributors = (
        best_pairs
        .groupby(
            "district_idx"
        )[
            "local_idx"
        ]
        .nunique()
    )

    one_source = int(
        (
            contributors == 1
        ).sum()
    )

    multi_source = int(
        (
            contributors > 1
        ).sum()
    )

    summary = {

        "local_count":
            int(len(local)),

        "district_count":
            int(len(district)),

        "candidate_pairs":
            int(len(pairs)),

        "best_match_pairs":
            int(len(best_pairs)),

        "matched_local":
            int(local_matched),

        "unmatched_local":
            int(local_unmatched),

        "matched_local_pct":
            float(
                100.0
                * local_matched
                / len(local)
                if len(local)
                else 0
            ),

        "matched_district":
            int(district_matched),

        "unmatched_district":
            int(district_unmatched),

        "matched_district_pct":
            float(
                100.0
                * district_matched
                / len(district)
                if len(district)
                else 0
            ),

        "district_with_one_source":
            one_source,

        "district_with_multiple_sources":
            multi_source,

        "multi_source_pct":
            float(
                100.0
                * multi_source
                / district_matched
                if district_matched
                else 0
            ),

        "mean_sources_per_district":
            float(
                contributors.mean()
            ),

        "median_sources_per_district":
            float(
                contributors.median()
            ),

        "max_sources_per_district":
            int(
                contributors.max()
            ),

        "mean_best_match_iou":
            float(
                best_pairs[
                    "iou"
                ].mean()
            ),

        "median_best_match_iou":
            float(
                best_pairs[
                    "iou"
                ].median()
            ),

        "mean_local_coverage":
            float(
                best_pairs[
                    "local_coverage"
                ].mean()
            ),

        "median_local_coverage":
            float(
                best_pairs[
                    "local_coverage"
                ].median()
            ),

        "mean_geometry_distance_m":
            float(
                best_pairs[
                    "geometry_distance_m"
                ].mean()
            ),

        "overlap_matches":
            int(
                (
                    best_pairs[
                        "match_basis"
                    ]
                    == "overlap"
                ).sum()
            ),

        "proximity_matches":
            int(
                (
                    best_pairs[
                        "match_basis"
                    ]
                    == "proximity"
                ).sum()
            ),
    }

    return summary


# ============================================================
# OUTPUT DIAGNOSTICS
# ============================================================

def save_diagnostic_layers(
    output_gpkg,
    local,
    district,
    pairs
):

    if output_gpkg.exists():
        output_gpkg.unlink()

    best_pairs = pairs[
        pairs[
            "is_best_local_match"
        ]
    ].copy()

    matched_local_ids = set(
        best_pairs[
            "local_idx"
        ].tolist()
    )

    matched_district_ids = set(
        best_pairs[
            "district_idx"
        ].tolist()
    )

    local_output = (
        local.copy()
    )

    local_output[
        "matched"
    ] = (
        local_output[
            "local_idx"
        ]
        .isin(
            matched_local_ids
        )
    )

    district_output = (
        district.copy()
    )

    district_output[
        "matched"
    ] = (
        district_output[
            "district_idx"
        ]
        .isin(
            matched_district_ids
        )
    )

    contributor_counts = (
        best_pairs
        .groupby(
            "district_idx"
        )[
            "local_idx"
        ]
        .nunique()
        .to_dict()
    )

    district_output[
        "source_count"
    ] = (
        district_output[
            "district_idx"
        ]
        .map(
            contributor_counts
        )
        .fillna(0)
        .astype(int)
    )

    local_output.to_file(
        output_gpkg,
        layer="local_buildings",
        driver="GPKG",
        engine="pyogrio"
    )

    district_output.to_file(
        output_gpkg,
        layer="district_buildings",
        driver="GPKG",
        engine="pyogrio"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 120)
    print(
        "CARTOGEN — BUILD BUILDING "
        "SPATIAL CORRESPONDENCE"
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

        source_gpkg = (
            DATA_ROOT
            / f"{window}.gpkg"
        )

        if not source_gpkg.exists():

            raise FileNotFoundError(
                source_gpkg
            )

        local = gpd.read_file(
            source_gpkg,
            layer="local_buildings",
            engine="pyogrio"
        )

        district = gpd.read_file(
            source_gpkg,
            layer="district_buildings",
            engine="pyogrio"
        )

        print(
            f"  Raw local buildings: "
            f"{len(local):,}"
        )

        print(
            f"  Raw district buildings: "
            f"{len(district):,}"
        )

        if (
            local.crs is None
            or district.crs is None
        ):

            raise RuntimeError(
                "Missing CRS."
            )

        if (
            local.crs.to_epsg()
            != 27700
            or
            district.crs.to_epsg()
            != 27700
        ):

            raise RuntimeError(
                "Expected EPSG:27700."
            )

        # --------------------------------------------
        # Determine shared extraction extent
        # --------------------------------------------

        total_bounds = local.total_bounds

        # --------------------------------------------
        # Remove boundary-contaminated centroids
        # --------------------------------------------

        local = (
            remove_boundary_features(
                local,
                total_bounds,
                BOUNDARY_MARGIN_M
            )
        )

        district = (
            remove_boundary_features(
                district,
                total_bounds,
                BOUNDARY_MARGIN_M
            )
        )

        local = ensure_id_column(
            local,
            "local"
        )

        district = ensure_id_column(
            district,
            "district"
        )

        print(
            f"  Interior local buildings: "
            f"{len(local):,}"
        )

        print(
            f"  Interior district buildings: "
            f"{len(district):,}"
        )

        # --------------------------------------------
        # Build correspondence
        # --------------------------------------------

        (
            pairs,
            _,
            _
        ) = build_correspondence(
            local,
            district
        )

        print(
            f"\n  Candidate pairs: "
            f"{len(pairs):,}"
        )

        # --------------------------------------------
        # Save pair table
        # --------------------------------------------

        csv_file = (
            OUTPUT_ROOT
            / f"{window}_building_pairs.csv"
        )

        pairs.to_csv(
            csv_file,
            index=False
        )

        # --------------------------------------------
        # Summary
        # --------------------------------------------

        summary = (
            summarize_correspondence(
                local,
                district,
                pairs
            )
        )

        all_summaries[
            window
        ] = summary

        print()
        print("-" * 120)
        print("CORRESPONDENCE SUMMARY")
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
                    f"{key:<35}"
                    f"{value:,.3f}"
                )

            else:

                print(
                    f"  "
                    f"{key:<35}"
                    f"{value:,}"
                )

        # --------------------------------------------
        # Save diagnostic GeoPackage
        # --------------------------------------------

        diagnostic_gpkg = (
            OUTPUT_ROOT
            / f"{window}_building_correspondence.gpkg"
        )

        if not pairs.empty:

            save_diagnostic_layers(
                diagnostic_gpkg,
                local,
                district,
                pairs
            )

        print(
            f"\n  CSV:\n"
            f"  {csv_file}"
        )

        print(
            f"\n  Diagnostic GPKG:\n"
            f"  {diagnostic_gpkg}"
        )

    # ------------------------------------------------
    # Save global summary
    # ------------------------------------------------

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
    print("SCRIPT 22 COMPLETE")
    print("=" * 120)

    print(
        f"Summary:\n"
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()