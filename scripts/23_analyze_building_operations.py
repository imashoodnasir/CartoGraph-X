from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.ops import unary_union


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WINDOW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "os"
    / "development_windows"
)

PAIR_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cartogen"
    / "building_correspondence"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cartogen"
    / "building_operations"
)

SUMMARY_FILE = (
    OUTPUT_ROOT
    / "building_operation_summary.json"
)

GLOBAL_CSV = (
    OUTPUT_ROOT
    / "building_operation_summary.csv"
)


WINDOWS = [
    "london_central",
    "manchester_central",
    "birmingham_central",
]


# ============================================================
# OPERATION THRESHOLDS
# ============================================================

# Similarity between union of detailed sources
# and generalized target.
HIGH_IOU = 0.70
MODERATE_IOU = 0.45

# Area-ratio thresholds.
AREA_SIMILAR_LOW = 0.80
AREA_SIMILAR_HIGH = 1.25

EXAGGERATION_RATIO = 1.25
STRONG_EXAGGERATION_RATIO = 1.75

# Centroid displacement.
LOW_DISPLACEMENT_M = 8.0
HIGH_DISPLACEMENT_M = 20.0

# Boundary simplification:
# fewer vertices than source geometry.
SIMPLIFICATION_RATIO = 0.80


# ============================================================
# GEOMETRY UTILITIES
# ============================================================

def vertex_count(geometry):

    if geometry is None:
        return 0

    if geometry.is_empty:
        return 0

    geom_type = geometry.geom_type

    if geom_type == "Polygon":

        total = len(
            geometry.exterior.coords
        )

        for interior in geometry.interiors:
            total += len(
                interior.coords
            )

        return total

    if geom_type == "MultiPolygon":

        return sum(
            vertex_count(part)
            for part in geometry.geoms
        )

    return 0


def compactness(geometry):

    if (
        geometry is None
        or geometry.is_empty
        or geometry.area <= 0
        or geometry.length <= 0
    ):
        return 0.0

    return float(
        4.0
        * np.pi
        * geometry.area
        / (
            geometry.length ** 2
        )
    )


def safe_iou(
    geometry_a,
    geometry_b
):

    union = (
        geometry_a.union(
            geometry_b
        )
    )

    if union.area <= 0:
        return 0.0

    intersection = (
        geometry_a.intersection(
            geometry_b
        )
    )

    return float(
        intersection.area
        / union.area
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_operation(
    source_count,
    union_iou,
    area_ratio,
    centroid_shift,
    vertex_ratio
):

    # --------------------------------------------------------
    # MULTIPLE SOURCE BUILDINGS
    # --------------------------------------------------------

    if source_count > 1:

        if (
            area_ratio
            >= STRONG_EXAGGERATION_RATIO
        ):
            return (
                "aggregation_plus_exaggeration"
            )

        return "aggregation"

    # --------------------------------------------------------
    # SINGLE SOURCE BUILDING
    # --------------------------------------------------------

    area_similar = (
        AREA_SIMILAR_LOW
        <= area_ratio
        <= AREA_SIMILAR_HIGH
    )

    if (
        union_iou >= HIGH_IOU
        and area_similar
        and centroid_shift
        <= LOW_DISPLACEMENT_M
    ):

        if (
            vertex_ratio
            < SIMPLIFICATION_RATIO
        ):
            return "simplification"

        return "preservation"

    if (
        area_ratio
        >= EXAGGERATION_RATIO
        and centroid_shift
        <= HIGH_DISPLACEMENT_M
    ):
        return "exaggeration"

    if (
        centroid_shift
        > HIGH_DISPLACEMENT_M
        and union_iou
        < MODERATE_IOU
    ):
        return "displacement"

    if (
        vertex_ratio
        < SIMPLIFICATION_RATIO
        and union_iou
        >= MODERATE_IOU
    ):
        return "simplification"

    return "complex_transformation"


# ============================================================
# ANALYSIS
# ============================================================

def analyze_window(window):

    print()
    print("=" * 120)
    print(window)
    print("=" * 120)

    source_gpkg = (
        WINDOW_ROOT
        / f"{window}.gpkg"
    )

    pair_csv = (
        PAIR_ROOT
        / f"{window}_building_pairs.csv"
    )

    if not source_gpkg.exists():
        raise FileNotFoundError(
            source_gpkg
        )

    if not pair_csv.exists():
        raise FileNotFoundError(
            pair_csv
        )

    local = gpd.read_file(
        source_gpkg,
        layer="local_buildings",
        engine="pyogrio"
    ).reset_index(
        drop=True
    )

    district = gpd.read_file(
        source_gpkg,
        layer="district_buildings",
        engine="pyogrio"
    ).reset_index(
        drop=True
    )

    # Script 22 creates local_idx and district_idx
    # AFTER boundary filtering. Therefore use its diagnostic
    # GPKG rather than assuming the original row numbers
    # still correspond.

    diagnostic_gpkg = (
        PAIR_ROOT
        / (
            f"{window}_"
            f"building_correspondence.gpkg"
        )
    )

    local = gpd.read_file(
        diagnostic_gpkg,
        layer="local_buildings",
        engine="pyogrio"
    )

    district = gpd.read_file(
        diagnostic_gpkg,
        layer="district_buildings",
        engine="pyogrio"
    )

    pairs = pd.read_csv(
        pair_csv
    )

    pairs = pairs[
        pairs[
            "is_best_local_match"
        ].astype(str).str.lower().isin(
            ["true", "1"]
        )
    ].copy()

    print(
        f"Interior local buildings: "
        f"{len(local):,}"
    )

    print(
        f"Interior district buildings: "
        f"{len(district):,}"
    )

    print(
        f"Best correspondence pairs: "
        f"{len(pairs):,}"
    )

    local_geometry = {
        int(row.local_idx):
            row.geometry

        for row in (
            local.itertuples()
        )
    }

    district_geometry = {
        int(row.district_idx):
            row.geometry

        for row in (
            district.itertuples()
        )
    }

    grouped = (
        pairs
        .groupby(
            "district_idx"
        )
    )

    records = []

    total_groups = (
        grouped.ngroups
    )

    print(
        f"Analyzing "
        f"{total_groups:,} "
        f"generalized buildings..."
    )

    for counter, (
        district_idx,
        group
    ) in enumerate(
        grouped,
        start=1
    ):

        district_idx = int(
            district_idx
        )

        target = (
            district_geometry.get(
                district_idx
            )
        )

        if target is None:
            continue

        source_ids = (
            group[
                "local_idx"
            ]
            .astype(int)
            .tolist()
        )

        source_geometries = [
            local_geometry[source_id]

            for source_id
            in source_ids

            if source_id
            in local_geometry
        ]

        if not source_geometries:
            continue

        source_union = (
            unary_union(
                source_geometries
            )
        )

        source_count = (
            len(
                source_geometries
            )
        )

        source_area = float(
            source_union.area
        )

        target_area = float(
            target.area
        )

        area_ratio = (
            target_area
            / source_area
            if source_area > 0
            else 0.0
        )

        union_iou = (
            safe_iou(
                source_union,
                target
            )
        )

        centroid_shift = float(
            source_union.centroid.distance(
                target.centroid
            )
        )

        source_vertices = (
            vertex_count(
                source_union
            )
        )

        target_vertices = (
            vertex_count(
                target
            )
        )

        vertex_ratio = (
            target_vertices
            / source_vertices
            if source_vertices > 0
            else 0.0
        )

        source_compactness = (
            compactness(
                source_union
            )
        )

        target_compactness = (
            compactness(
                target
            )
        )

        operation = (
            classify_operation(
                source_count,
                union_iou,
                area_ratio,
                centroid_shift,
                vertex_ratio
            )
        )

        record = {

            "window":
                window,

            "district_idx":
                district_idx,

            "source_count":
                source_count,

            "source_ids":
                ";".join(
                    str(value)
                    for value in source_ids
                ),

            "source_union_area_m2":
                source_area,

            "target_area_m2":
                target_area,

            "area_ratio":
                area_ratio,

            "union_iou":
                union_iou,

            "centroid_shift_m":
                centroid_shift,

            "source_vertices":
                source_vertices,

            "target_vertices":
                target_vertices,

            "vertex_ratio":
                vertex_ratio,

            "source_compactness":
                source_compactness,

            "target_compactness":
                target_compactness,

            "compactness_change":
                (
                    target_compactness
                    - source_compactness
                ),

            "operation":
                operation,
        }

        records.append(
            record
        )

        if (
            counter % 1000 == 0
            or counter == total_groups
        ):

            print(
                f"  "
                f"{counter:,}/"
                f"{total_groups:,} "
                f"analyzed"
            )

    result = pd.DataFrame(
        records
    )

    return (
        result,
        local,
        district,
        pairs
    )


# ============================================================
# ELIMINATION ANALYSIS
# ============================================================

def analyze_elimination(
    local,
    pairs
):

    matched_ids = set(
        pairs[
            "local_idx"
        ]
        .astype(int)
        .tolist()
    )

    elimination = (
        local[
            ~local[
                "local_idx"
            ].isin(
                matched_ids
            )
        ]
        .copy()
    )

    elimination[
        "operation"
    ] = "candidate_elimination"

    return elimination


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 120)
    print(
        "CARTOGEN — ANALYZE BUILDING "
        "GENERALIZATION OPERATIONS"
    )
    print("=" * 120)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    global_records = []

    global_summary = {}

    for window in WINDOWS:

        (
            result,
            local,
            district,
            pairs
        ) = analyze_window(
            window
        )

        # ----------------------------------------------------
        # Save detailed operation table
        # ----------------------------------------------------

        csv_file = (
            OUTPUT_ROOT
            / f"{window}_building_operations.csv"
        )

        result.to_csv(
            csv_file,
            index=False
        )

        # ----------------------------------------------------
        # Candidate eliminations
        # ----------------------------------------------------

        elimination = (
            analyze_elimination(
                local,
                pairs
            )
        )

        elimination_file = (
            OUTPUT_ROOT
            / (
                f"{window}_"
                f"candidate_eliminations.gpkg"
            )
        )

        if (
            elimination_file.exists()
        ):
            elimination_file.unlink()

        if not elimination.empty:

            elimination.to_file(
                elimination_file,
                layer="candidate_eliminations",
                driver="GPKG",
                engine="pyogrio"
            )

        # ----------------------------------------------------
        # Attach operation to district geometry
        # ----------------------------------------------------

        district_output = (
            district.merge(
                result[
                    [
                        "district_idx",
                        "source_count",
                        "area_ratio",
                        "union_iou",
                        "centroid_shift_m",
                        "vertex_ratio",
                        "operation",
                    ]
                ],
                on="district_idx",
                how="left"
            )
        )

        gpkg_file = (
            OUTPUT_ROOT
            / (
                f"{window}_"
                f"building_operations.gpkg"
            )
        )

        if gpkg_file.exists():
            gpkg_file.unlink()

        district_output.to_file(
            gpkg_file,
            layer="generalized_building_operations",
            driver="GPKG",
            engine="pyogrio"
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        counts = (
            result[
                "operation"
            ]
            .value_counts()
        )

        percentages = (
            result[
                "operation"
            ]
            .value_counts(
                normalize=True
            )
            * 100.0
        )

        print()
        print("-" * 120)
        print("OPERATION DISTRIBUTION")
        print("-" * 120)

        summary = {}

        for operation in sorted(
            counts.index
        ):

            count = int(
                counts[
                    operation
                ]
            )

            pct = float(
                percentages[
                    operation
                ]
            )

            summary[
                operation
            ] = {
                "count":
                    count,

                "percentage":
                    pct,
            }

            print(
                f"  "
                f"{operation:<35}"
                f"{count:>7,} "
                f"({pct:>6.2f}%)"
            )

        print()
        print(
            f"Candidate eliminations: "
            f"{len(elimination):,}"
        )

        print()
        print(
            f"Mean union IoU: "
            f"{result['union_iou'].mean():.3f}"
        )

        print(
            f"Median union IoU: "
            f"{result['union_iou'].median():.3f}"
        )

        print(
            f"Mean area ratio: "
            f"{result['area_ratio'].mean():.3f}"
        )

        print(
            f"Median area ratio: "
            f"{result['area_ratio'].median():.3f}"
        )

        print(
            f"Mean centroid shift: "
            f"{result['centroid_shift_m'].mean():.3f} m"
        )

        print(
            f"Median centroid shift: "
            f"{result['centroid_shift_m'].median():.3f} m"
        )

        global_summary[
            window
        ] = {

            "generalized_buildings_analyzed":
                int(len(result)),

            "candidate_eliminations":
                int(len(elimination)),

            "mean_union_iou":
                float(
                    result[
                        "union_iou"
                    ].mean()
                ),

            "median_union_iou":
                float(
                    result[
                        "union_iou"
                    ].median()
                ),

            "mean_area_ratio":
                float(
                    result[
                        "area_ratio"
                    ].mean()
                ),

            "median_area_ratio":
                float(
                    result[
                        "area_ratio"
                    ].median()
                ),

            "mean_centroid_shift_m":
                float(
                    result[
                        "centroid_shift_m"
                    ].mean()
                ),

            "median_centroid_shift_m":
                float(
                    result[
                        "centroid_shift_m"
                    ].median()
                ),

            "operations":
                summary,
        }

        # ----------------------------------------------------
        # Add to global table
        # ----------------------------------------------------

        for operation, values in (
            summary.items()
        ):

            global_records.append({
                "window":
                    window,

                "operation":
                    operation,

                "count":
                    values["count"],

                "percentage":
                    values[
                        "percentage"
                    ],
            })

        global_records.append({
            "window":
                window,

            "operation":
                "candidate_elimination",

            "count":
                int(
                    len(elimination)
                ),

            "percentage":
                float(
                    100.0
                    * len(elimination)
                    / len(local)
                    if len(local)
                    else 0.0
                ),
        })

        print()
        print(
            f"Operations CSV:\n"
            f"{csv_file}"
        )

        print(
            f"\nOperations GPKG:\n"
            f"{gpkg_file}"
        )

    # ========================================================
    # GLOBAL OUTPUT
    # ========================================================

    pd.DataFrame(
        global_records
    ).to_csv(
        GLOBAL_CSV,
        index=False
    )

    with open(
        SUMMARY_FILE,
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
    print("SCRIPT 23 COMPLETE")
    print("=" * 120)

    print(
        f"Global CSV:\n"
        f"{GLOBAL_CSV}"
    )

    print(
        f"\nSummary JSON:\n"
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()