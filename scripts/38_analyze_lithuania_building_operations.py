from pathlib import Path
import json
import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAIR_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "building_correspondence"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "building_operations"
)

WINDOWS = [
    ("vilnius_central", "Vilnius"),
    ("kaunas_central", "Kaunas"),
    ("klaipeda_central", "Klaipeda"),
]

PROGRESS_INTERVAL = 1000


# ============================================================
# GEOMETRIC FUNCTIONS
# ============================================================

def safe_divide(a, b):

    if b is None or b == 0:
        return 0.0

    return float(a) / float(b)


def vertex_count(geom):

    if geom is None or geom.is_empty:
        return 0

    if geom.geom_type == "Polygon":

        count = len(
            geom.exterior.coords
        )

        for ring in geom.interiors:
            count += len(ring.coords)

        return count

    if geom.geom_type == "MultiPolygon":

        return sum(
            vertex_count(part)
            for part in geom.geoms
        )

    return 0


def compactness(geom):

    if geom is None or geom.is_empty:
        return 0.0

    area = geom.area
    perimeter = geom.length

    if perimeter == 0:
        return 0.0

    return (
        4.0
        * math.pi
        * area
        / (perimeter ** 2)
    )


def calculate_iou(
    source_geom,
    target_geom
):

    intersection = (
        source_geom
        .intersection(target_geom)
        .area
    )

    union = (
        source_geom.area
        + target_geom.area
        - intersection
    )

    return safe_divide(
        intersection,
        union
    )


# ============================================================
# OPERATION CLASSIFICATION
# ============================================================

def classify_operation(
    source_count,
    iou,
    area_ratio,
    centroid_shift,
    vertex_ratio,
    compactness_change
):

    # --------------------------------------------------------
    # MULTIPLE DETAILED BUILDINGS -> ONE TARGET
    # --------------------------------------------------------

    if source_count > 1:

        if area_ratio >= 1.20:
            return (
                "aggregation_plus_exaggeration"
            )

        return "aggregation"

    # --------------------------------------------------------
    # ONE -> ONE
    # --------------------------------------------------------

    if (
        iou >= 0.80
        and 0.85 <= area_ratio <= 1.15
        and centroid_shift <= 3.0
        and vertex_ratio >= 0.70
    ):
        return "preservation"

    if (
        centroid_shift >= 8.0
        and 0.75 <= area_ratio <= 1.30
    ):
        return "displacement"

    if (
        area_ratio >= 1.20
        and centroid_shift < 8.0
    ):
        return "exaggeration"

    if (
        vertex_ratio <= 0.70
        and 0.70 <= area_ratio <= 1.30
        and centroid_shift < 8.0
    ):
        return "simplification"

    return "complex_transformation"


# ============================================================
# ANALYSIS
# ============================================================

def analyze_city(
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
        PAIR_ROOT
        / f"{window}_building_correspondence.gpkg"
    )

    pair_csv = (
        PAIR_ROOT
        / f"{window}_building_pairs.csv"
    )

    source = gpd.read_file(
        gpkg_path,
        layer="source_buildings",
        engine="pyogrio"
    )

    target = gpd.read_file(
        gpkg_path,
        layer="target_buildings",
        engine="pyogrio"
    )

    pairs = pd.read_csv(
        pair_csv
    )

    print(
        f"Interior source buildings: "
        f"{len(source):,}"
    )

    print(
        f"Interior target buildings: "
        f"{len(target):,}"
    )

    print(
        f"Best correspondence pairs: "
        f"{len(pairs):,}"
    )

    # --------------------------------------------------------
    # GROUP SOURCE FEATURES BY TARGET
    # --------------------------------------------------------

    groups = (
        pairs
        .groupby("target_idx")
    )

    records = []

    total_groups = groups.ngroups

    print(
        f"Analyzing "
        f"{total_groups:,} generalized buildings..."
    )

    for number, (
        target_idx,
        group
    ) in enumerate(
        groups,
        start=1
    ):

        target_idx = int(
            target_idx
        )

        target_geom = (
            target.iloc[
                target_idx
            ].geometry
        )

        source_indices = (
            group[
                "source_idx"
            ]
            .astype(int)
            .tolist()
        )

        source_geometries = (
            source.iloc[
                source_indices
            ].geometry.tolist()
        )

        # Combine all detailed buildings assigned
        # to this generalized target.
        source_union = unary_union(
            source_geometries
        )

        if (
            source_union is None
            or source_union.is_empty
            or target_geom is None
            or target_geom.is_empty
        ):
            continue

        source_count = len(
            source_indices
        )

        source_area = (
            source_union.area
        )

        target_area = (
            target_geom.area
        )

        area_ratio = safe_divide(
            target_area,
            source_area
        )

        iou = calculate_iou(
            source_union,
            target_geom
        )

        centroid_shift = (
            source_union.centroid
            .distance(
                target_geom.centroid
            )
        )

        source_vertices = (
            vertex_count(
                source_union
            )
        )

        target_vertices = (
            vertex_count(
                target_geom
            )
        )

        vertex_ratio = safe_divide(
            target_vertices,
            source_vertices
        )

        source_compactness = (
            compactness(
                source_union
            )
        )

        target_compactness = (
            compactness(
                target_geom
            )
        )

        compactness_change = (
            target_compactness
            - source_compactness
        )

        operation = classify_operation(
            source_count,
            iou,
            area_ratio,
            centroid_shift,
            vertex_ratio,
            compactness_change
        )

        records.append(
            {
                "target_idx":
                    target_idx,

                "source_count":
                    source_count,

                "source_indices":
                    ";".join(
                        str(x)
                        for x in source_indices
                    ),

                "operation":
                    operation,

                "source_area_m2":
                    source_area,

                "target_area_m2":
                    target_area,

                "area_ratio":
                    area_ratio,

                "iou":
                    iou,

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
                    compactness_change,
            }
        )

        if (
            number % PROGRESS_INTERVAL == 0
            or number == total_groups
        ):

            print(
                f"  "
                f"{number:,}/"
                f"{total_groups:,} analyzed"
            )

    operations = pd.DataFrame(
        records
    )

    # ========================================================
    # ELIMINATION LABELS
    # ========================================================

    matched_source_indices = set(
        pairs[
            "source_idx"
        ]
        .astype(int)
        .tolist()
    )

    all_source_indices = set(
        range(
            len(source)
        )
    )

    eliminated_indices = sorted(
        all_source_indices
        - matched_source_indices
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    operation_counts = (
        operations[
            "operation"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print("-" * 120)
    print(
        "OPERATION DISTRIBUTION"
    )
    print("-" * 120)

    for operation, count in (
        operation_counts.items()
    ):

        percentage = (
            100.0
            * count
            / len(operations)
        )

        print(
            f"{operation:<35}"
            f"{count:>10,} "
            f"({percentage:6.2f}%)"
        )

    print()
    print(
        f"Candidate eliminations: "
        f"{len(eliminated_indices):,}"
    )

    print()
    print(
        f"Mean union IoU: "
        f"{operations['iou'].mean():.3f}"
    )

    print(
        f"Median union IoU: "
        f"{operations['iou'].median():.3f}"
    )

    print(
        f"Mean area ratio: "
        f"{operations['area_ratio'].mean():.3f}"
    )

    print(
        f"Median area ratio: "
        f"{operations['area_ratio'].median():.3f}"
    )

    print(
        f"Mean centroid shift: "
        f"{operations['centroid_shift_m'].mean():.3f} m"
    )

    print(
        f"Median centroid shift: "
        f"{operations['centroid_shift_m'].median():.3f} m"
    )

    print(
        f"Mean vertex ratio: "
        f"{operations['vertex_ratio'].mean():.3f}"
    )

    print(
        f"Median vertex ratio: "
        f"{operations['vertex_ratio'].median():.3f}"
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = (
        OUTPUT_ROOT
        / f"{window}_building_operations.csv"
    )

    operations.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------------
    # ELIMINATIONS CSV
    # --------------------------------------------------------

    elimination_df = pd.DataFrame(
        {
            "source_idx":
                eliminated_indices,

            "operation":
                "elimination"
        }
    )

    elimination_csv = (
        OUTPUT_ROOT
        / f"{window}_building_eliminations.csv"
    )

    elimination_df.to_csv(
        elimination_csv,
        index=False
    )

    # ========================================================
    # SAVE GPKG
    # ========================================================

    gpkg_output = (
        OUTPUT_ROOT
        / f"{window}_building_operations.gpkg"
    )

    if gpkg_output.exists():
        gpkg_output.unlink()

    target_output = (
        target.copy()
    )

    target_output[
        "target_idx"
    ] = np.arange(
        len(target_output)
    )

    target_output = (
        target_output.merge(
            operations[
                [
                    "target_idx",
                    "source_count",
                    "operation",
                    "area_ratio",
                    "iou",
                    "centroid_shift_m",
                    "vertex_ratio"
                ]
            ],
            on="target_idx",
            how="left"
        )
    )

    target_output.to_file(
        gpkg_output,
        layer="target_operations",
        driver="GPKG",
        engine="pyogrio"
    )

    source_output = (
        source.copy()
    )

    source_output[
        "source_idx"
    ] = np.arange(
        len(source_output)
    )

    source_output[
        "eliminated"
    ] = (
        source_output[
            "source_idx"
        ]
        .isin(
            eliminated_indices
        )
        .astype(int)
    )

    source_output.to_file(
        gpkg_output,
        layer="source_elimination",
        driver="GPKG",
        engine="pyogrio"
    )

    # ========================================================
    # CITY SUMMARY
    # ========================================================

    summary = {
        "window":
            window,

        "city":
            city,

        "source_count":
            len(source),

        "target_count":
            len(target),

        "matched_targets":
            len(operations),

        "candidate_eliminations":
            len(eliminated_indices),

        "elimination_pct":
            100.0
            * safe_divide(
                len(eliminated_indices),
                len(source)
            ),

        "mean_iou":
            float(
                operations[
                    "iou"
                ].mean()
            ),

        "median_iou":
            float(
                operations[
                    "iou"
                ].median()
            ),

        "mean_area_ratio":
            float(
                operations[
                    "area_ratio"
                ].mean()
            ),

        "median_area_ratio":
            float(
                operations[
                    "area_ratio"
                ].median()
            ),

        "mean_centroid_shift_m":
            float(
                operations[
                    "centroid_shift_m"
                ].mean()
            ),

        "median_centroid_shift_m":
            float(
                operations[
                    "centroid_shift_m"
                ].median()
            ),

        "mean_vertex_ratio":
            float(
                operations[
                    "vertex_ratio"
                ].mean()
            ),

        "median_vertex_ratio":
            float(
                operations[
                    "vertex_ratio"
                ].median()
            ),
    }

    for operation, count in (
        operation_counts.items()
    ):

        summary[
            f"n_{operation}"
        ] = int(
            count
        )

        summary[
            f"pct_{operation}"
        ] = (
            100.0
            * count
            / len(operations)
        )

    print()
    print("Operations CSV:")
    print(csv_path)

    print()
    print("Eliminations CSV:")
    print(elimination_csv)

    print()
    print("Operations GPKG:")
    print(gpkg_output)

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 120)
    print(
        "CARTOGEN — ANALYZE LITHUANIAN "
        "BUILDING GENERALIZATION OPERATIONS"
    )
    print("=" * 120)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    summaries = []

    for window, city in WINDOWS:

        summary = analyze_city(
            window,
            city
        )

        summaries.append(
            summary
        )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_csv = (
        OUTPUT_ROOT
        / "building_operation_summary.csv"
    )

    summary_json = (
        OUTPUT_ROOT
        / "building_operation_summary.json"
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
            indent=2
        )

    print()
    print("=" * 120)
    print(
        "GLOBAL BUILDING OPERATION SUMMARY"
    )
    print("=" * 120)

    base_columns = [
        "city",
        "source_count",
        "target_count",
        "matched_targets",
        "candidate_eliminations",
        "elimination_pct",
        "mean_iou",
        "mean_area_ratio",
        "mean_centroid_shift_m",
        "mean_vertex_ratio",
    ]

    print(
        summary_df[
            base_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("SCRIPT 38 COMPLETE")
    print("=" * 120)

    print()
    print("Global CSV:")
    print(summary_csv)

    print()
    print("Summary JSON:")
    print(summary_json)


if __name__ == "__main__":
    main()