from pathlib import Path
import json
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


# ======================================================================
# CONFIG
# ======================================================================

ROOT = Path(__file__).resolve().parents[1]

TILE_DIR = (
    ROOT / "data" / "processed" /
    "lithuania" / "spatial_tiles"
)

MANIFEST = TILE_DIR / "lithuania_spatial_tile_manifest.csv"
TILE_INDEX = TILE_DIR / "lithuania_spatial_tile_index.gpkg"

OUT_DIR = (
    ROOT / "data" / "processed" /
    "lithuania" / "benchmark_audit"
)

FIG_DIR = OUT_DIR / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# Sparse thresholds.
# These are audit thresholds, not deletion rules.

MIN_BUILDINGS = 25
MIN_ROADS = 20

EXPECTED_SPLITS = {
    "train",
    "validation",
    "test",
}


# ======================================================================
# HELPERS
# ======================================================================

def header(text):
    print()
    print("=" * 120)
    print(text)
    print("=" * 120)


def subheader(text):
    print()
    print("-" * 120)
    print(text)
    print("-" * 120)


def save_figure(fig, name):
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"PNG: {png}")
    print(f"PDF: {pdf}")


def safe_ratio(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    result = np.full(
        len(a),
        np.nan,
        dtype=float
    )

    valid = b > 0

    result[valid] = (
        a[valid] / b[valid]
    )

    return result


# ======================================================================
# MAIN
# ======================================================================

def main():

    header(
        "CARTOGEN — FINAL LITHUANIAN BENCHMARK AUDIT"
    )

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    if not TILE_INDEX.exists():
        raise FileNotFoundError(TILE_INDEX)

    df = pd.read_csv(MANIFEST)

    tiles = gpd.read_file(
        TILE_INDEX,
        layer="tiles"
    )

    print(f"Manifest tiles: {len(df)}")
    print(f"Spatial tiles:  {len(tiles)}")
    print(f"CRS:            {tiles.crs}")

    # ==================================================================
    # BASIC STRUCTURAL VALIDATION
    # ==================================================================

    subheader("1. STRUCTURAL VALIDATION")

    duplicate_ids = int(
        df["tile_id"].duplicated().sum()
    )

    split_values = set(
        df["split"].dropna().unique()
    )

    print(
        f"Duplicate tile IDs: {duplicate_ids}"
    )

    print(
        f"Splits: {sorted(split_values)}"
    )

    if duplicate_ids == 0:
        print("[PASS] Tile identifiers unique.")
    else:
        print("[FAIL] Duplicate tile identifiers.")

    if split_values == EXPECTED_SPLITS:
        print("[PASS] All expected splits present.")
    else:
        print("[FAIL] Unexpected/missing splits.")

    if len(df) == len(tiles):
        print("[PASS] Manifest/index counts agree.")
    else:
        print("[FAIL] Manifest/index count mismatch.")

    # ==================================================================
    # POSITIVE AREA OVERLAP
    # ==================================================================

    subheader("2. CORE TILE LEAKAGE")

    overlap_records = []

    for city in tiles["city"].unique():

        city_tiles = (
            tiles[
                tiles["city"] == city
            ]
            .reset_index(drop=True)
        )

        sindex = city_tiles.sindex

        for i, row in city_tiles.iterrows():

            candidates = list(
                sindex.intersection(
                    row.geometry.bounds
                )
            )

            for j in candidates:

                if j <= i:
                    continue

                other = city_tiles.iloc[j]

                if row["split"] == other["split"]:
                    continue

                inter = row.geometry.intersection(
                    other.geometry
                )

                if (
                    not inter.is_empty
                    and inter.area > 0
                ):

                    overlap_records.append({
                        "city": city,
                        "tile_a": row["tile_id"],
                        "split_a": row["split"],
                        "tile_b": other["tile_id"],
                        "split_b": other["split"],
                        "overlap_area_m2":
                            float(inter.area),
                    })

    overlap_df = pd.DataFrame(
        overlap_records
    )

    overlap_csv = (
        OUT_DIR /
        "cross_split_core_overlap.csv"
    )

    overlap_df.to_csv(
        overlap_csv,
        index=False
    )

    print(
        f"Cross-split positive-area overlaps: "
        f"{len(overlap_df)}"
    )

    if len(overlap_df) == 0:
        print(
            "[PASS] No positive-area core leakage."
        )
    else:
        print(
            "[FAIL] Spatial core leakage detected."
        )

    # ==================================================================
    # TILE STATISTICS
    # ==================================================================

    subheader("3. TILE DENSITY AND RETENTION")

    df["building_retention"] = safe_ratio(
        df["target_buildings_core"],
        df["source_buildings_core"]
    )

    df["road_retention"] = safe_ratio(
        df["target_roads_core"],
        df["source_roads_core"]
    )

    df["building_reduction"] = (
        1.0 - df["building_retention"]
    )

    df["road_reduction"] = (
        1.0 - df["road_retention"]
    )

    df["building_sparse"] = (
        df["source_buildings_core"]
        < MIN_BUILDINGS
    )

    df["road_sparse"] = (
        df["source_roads_core"]
        < MIN_ROADS
    )

    df["empty_building"] = (
        df["source_buildings_core"] == 0
    )

    df["empty_road"] = (
        df["source_roads_core"] == 0
    )

    # Branch suitability.
    df["building_usable"] = (
        df["source_buildings_core"]
        >= MIN_BUILDINGS
    )

    df["road_usable"] = (
        df["source_roads_core"]
        >= MIN_ROADS
    )

    df["joint_usable"] = (
        df["building_usable"]
        & df["road_usable"]
    )

    print(
        f"Sparse building tiles: "
        f"{int(df['building_sparse'].sum())}"
    )

    print(
        f"Sparse road tiles:     "
        f"{int(df['road_sparse'].sum())}"
    )

    print(
        f"Empty building tiles:  "
        f"{int(df['empty_building'].sum())}"
    )

    print(
        f"Empty road tiles:      "
        f"{int(df['empty_road'].sum())}"
    )

    print(
        f"Building-usable tiles: "
        f"{int(df['building_usable'].sum())}"
    )

    print(
        f"Road-usable tiles:     "
        f"{int(df['road_usable'].sum())}"
    )

    print(
        f"Joint-usable tiles:    "
        f"{int(df['joint_usable'].sum())}"
    )

    audit_csv = (
        OUT_DIR /
        "lithuania_tile_audit.csv"
    )

    df.to_csv(
        audit_csv,
        index=False
    )

    # ==================================================================
    # SPLIT STATISTICS
    # ==================================================================

    subheader("4. SPLIT DISTRIBUTION")

    split_rows = []

    for split in [
        "train",
        "validation",
        "test"
    ]:

        x = df[
            df["split"] == split
        ]

        row = {
            "split": split,
            "tiles": len(x),

            "source_buildings":
                int(
                    x[
                        "source_buildings_core"
                    ].sum()
                ),

            "target_buildings":
                int(
                    x[
                        "target_buildings_core"
                    ].sum()
                ),

            "source_roads":
                int(
                    x[
                        "source_roads_core"
                    ].sum()
                ),

            "target_roads":
                int(
                    x[
                        "target_roads_core"
                    ].sum()
                ),

            "mean_building_retention":
                float(
                    x[
                        "building_retention"
                    ].mean()
                ),

            "median_building_retention":
                float(
                    x[
                        "building_retention"
                    ].median()
                ),

            "mean_road_retention":
                float(
                    x[
                        "road_retention"
                    ].mean()
                ),

            "median_road_retention":
                float(
                    x[
                        "road_retention"
                    ].median()
                ),

            "building_usable_tiles":
                int(
                    x[
                        "building_usable"
                    ].sum()
                ),

            "road_usable_tiles":
                int(
                    x[
                        "road_usable"
                    ].sum()
                ),
        }

        split_rows.append(row)

    split_df = pd.DataFrame(
        split_rows
    )

    print(
        split_df.to_string(
            index=False
        )
    )

    split_csv = (
        OUT_DIR /
        "split_distribution_summary.csv"
    )

    split_df.to_csv(
        split_csv,
        index=False
    )

    # ==================================================================
    # CITY x SPLIT
    # ==================================================================

    subheader("5. CITY × SPLIT REPRESENTATION")

    city_split = (
        df.groupby(
            ["city", "split"],
            as_index=False
        )
        .agg(
            tiles=("tile_id", "count"),

            source_buildings=(
                "source_buildings_core",
                "sum"
            ),

            target_buildings=(
                "target_buildings_core",
                "sum"
            ),

            source_roads=(
                "source_roads_core",
                "sum"
            ),

            target_roads=(
                "target_roads_core",
                "sum"
            ),

            building_usable=(
                "building_usable",
                "sum"
            ),

            road_usable=(
                "road_usable",
                "sum"
            ),
        )
    )

    print(
        city_split.to_string(
            index=False
        )
    )

    city_split_csv = (
        OUT_DIR /
        "city_split_representation.csv"
    )

    city_split.to_csv(
        city_split_csv,
        index=False
    )

    # ==================================================================
    # FIGURE 1 — SPATIAL SPLIT MAP
    # ==================================================================

    subheader("6. GENERATING FIGURES")

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    split_order = [
        "train",
        "validation",
        "test"
    ]

    for split in split_order:

        subset = tiles[
            tiles["split"] == split
        ]

        subset.plot(
            ax=ax,
            alpha=0.55,
            edgecolor="black",
            linewidth=0.5,
            label=split
        )

    ax.set_title(
        "Lithuanian CartoGen Spatial Benchmark Splits"
    )

    ax.set_xlabel(
        "Easting (EPSG:3346)"
    )

    ax.set_ylabel(
        "Northing (EPSG:3346)"
    )

    ax.legend(
        title="Split"
    )

    ax.set_aspect("equal")

    fig.tight_layout()

    save_figure(
        fig,
        "figure_01_spatial_splits"
    )

    # ==================================================================
    # FIGURE 2 — BUILDING DENSITY
    # ==================================================================

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    plot_df = (
        df.sort_values(
            "source_buildings_core",
            ascending=False
        )
        .reset_index(drop=True)
    )

    ax.bar(
        np.arange(len(plot_df)),
        plot_df[
            "source_buildings_core"
        ]
    )

    ax.axhline(
        MIN_BUILDINGS,
        linestyle="--",
        linewidth=1.2,
        label=(
            f"Sparse threshold "
            f"({MIN_BUILDINGS})"
        )
    )

    ax.set_title(
        "Source Building Density Across Spatial Tiles"
    )

    ax.set_xlabel(
        "Tiles ranked by source-building count"
    )

    ax.set_ylabel(
        "Source buildings per 1 km² core tile"
    )

    ax.legend()

    fig.tight_layout()

    save_figure(
        fig,
        "figure_02_building_density"
    )

    # ==================================================================
    # FIGURE 3 — ROAD DENSITY
    # ==================================================================

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    plot_df = (
        df.sort_values(
            "source_roads_core",
            ascending=False
        )
        .reset_index(drop=True)
    )

    ax.bar(
        np.arange(len(plot_df)),
        plot_df[
            "source_roads_core"
        ]
    )

    ax.axhline(
        MIN_ROADS,
        linestyle="--",
        linewidth=1.2,
        label=(
            f"Sparse threshold "
            f"({MIN_ROADS})"
        )
    )

    ax.set_title(
        "Source Road Density Across Spatial Tiles"
    )

    ax.set_xlabel(
        "Tiles ranked by source-road count"
    )

    ax.set_ylabel(
        "Source road features per 1 km² core tile"
    )

    ax.legend()

    fig.tight_layout()

    save_figure(
        fig,
        "figure_03_road_density"
    )

    # ==================================================================
    # FIGURE 4 — RETENTION BY SPLIT
    # ==================================================================

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    data = []

    labels = []

    for split in split_order:

        values = (
            df.loc[
                df["split"] == split,
                "building_retention"
            ]
            .dropna()
            .values
        )

        data.append(values)
        labels.append(split)

    ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=True
    )

    ax.set_title(
        "Building Feature Retention by Spatial Split"
    )

    ax.set_xlabel("Split")
    ax.set_ylabel(
        "Target / source building count"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure_04_building_retention_by_split"
    )

    # ==================================================================
    # FIGURE 5 — ROAD RETENTION BY SPLIT
    # ==================================================================

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    data = []

    labels = []

    for split in split_order:

        values = (
            df.loc[
                df["split"] == split,
                "road_retention"
            ]
            .dropna()
            .values
        )

        data.append(values)
        labels.append(split)

    ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=True
    )

    ax.set_title(
        "Road Feature Retention by Spatial Split"
    )

    ax.set_xlabel("Split")
    ax.set_ylabel(
        "Target / source road count"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure_05_road_retention_by_split"
    )

    # ==================================================================
    # FIGURE 6 — SOURCE/TARGET BUILDINGS BY CITY
    # ==================================================================

    city_summary = (
        df.groupby(
            "city",
            as_index=False
        )
        .agg(
            source_buildings=(
                "source_buildings_core",
                "sum"
            ),
            target_buildings=(
                "target_buildings_core",
                "sum"
            ),
            source_roads=(
                "source_roads_core",
                "sum"
            ),
            target_roads=(
                "target_roads_core",
                "sum"
            ),
        )
    )

    x = np.arange(
        len(city_summary)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        x - width / 2,
        city_summary[
            "source_buildings"
        ],
        width,
        label="Source"
    )

    ax.bar(
        x + width / 2,
        city_summary[
            "target_buildings"
        ],
        width,
        label="Target"
    )

    ax.set_xticks(
        x,
        city_summary["city"]
    )

    ax.set_title(
        "Building Samples by Lithuanian City"
    )

    ax.set_ylabel(
        "Building features"
    )

    ax.legend()

    fig.tight_layout()

    save_figure(
        fig,
        "figure_06_buildings_by_city"
    )

    # ==================================================================
    # FIGURE 7 — SOURCE/TARGET ROADS BY CITY
    # ==================================================================

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        x - width / 2,
        city_summary[
            "source_roads"
        ],
        width,
        label="Source"
    )

    ax.bar(
        x + width / 2,
        city_summary[
            "target_roads"
        ],
        width,
        label="Target"
    )

    ax.set_xticks(
        x,
        city_summary["city"]
    )

    ax.set_title(
        "Road Samples by Lithuanian City"
    )

    ax.set_ylabel(
        "Road features"
    )

    ax.legend()

    fig.tight_layout()

    save_figure(
        fig,
        "figure_07_roads_by_city"
    )

    # ==================================================================
    # READINESS
    # ==================================================================

    subheader("7. FINAL READINESS ASSESSMENT")

    checks = {
        "unique_tile_ids":
            duplicate_ids == 0,

        "all_splits_present":
            split_values == EXPECTED_SPLITS,

        "manifest_index_agree":
            len(df) == len(tiles),

        "no_core_spatial_leakage":
            len(overlap_df) == 0,

        "train_exists":
            (
                df["split"] == "train"
            ).any(),

        "validation_exists":
            (
                df["split"] == "validation"
            ).any(),

        "test_exists":
            (
                df["split"] == "test"
            ).any(),
    }

    for key, value in checks.items():

        print(
            f"{key:<35} "
            f"{'PASS' if value else 'FAIL'}"
        )

    ready = all(
        checks.values()
    )

    # Sparse tiles are intentionally warnings,
    # not readiness failures.

    warnings_list = []

    sparse_building_count = int(
        df["building_sparse"].sum()
    )

    sparse_road_count = int(
        df["road_sparse"].sum()
    )

    if sparse_building_count > 0:

        warnings_list.append(
            f"{sparse_building_count} tiles "
            f"contain fewer than "
            f"{MIN_BUILDINGS} source buildings."
        )

    if sparse_road_count > 0:

        warnings_list.append(
            f"{sparse_road_count} tiles "
            f"contain fewer than "
            f"{MIN_ROADS} source roads."
        )

    print()

    if ready:

        print(
            "[PASS] LITHUANIAN BENCHMARK "
            "IS STRUCTURALLY READY."
        )

    else:

        print(
            "[FAIL] LITHUANIAN BENCHMARK "
            "REQUIRES CORRECTION."
        )

    if warnings_list:

        print()
        print("Warnings:")

        for item in warnings_list:
            print(f"  - {item}")

    # ==================================================================
    # SAVE JSON
    # ==================================================================

    report = {
        "dataset":
            "CartoGen Lithuania",

        "total_tiles":
            int(len(df)),

        "cities":
            sorted(
                df["city"]
                .dropna()
                .unique()
                .tolist()
            ),

        "splits": {
            str(key): int(value)
            for key, value in
            df["split"]
            .value_counts()
            .to_dict()
            .items()
        },

        "thresholds": {
            "minimum_source_buildings":
                int(MIN_BUILDINGS),
            "minimum_source_roads":
                int(MIN_ROADS),
        },

        "sparse_tiles": {
            "building":
                int(sparse_building_count),
            "road":
                int(sparse_road_count),
            "empty_building":
                int(df["empty_building"].sum()),
            "empty_road":
                int(df["empty_road"].sum()),
        },

        "usable_tiles": {
            "building":
                int(df["building_usable"].sum()),
            "road":
                int(df["road_usable"].sum()),
            "joint":
                int(df["joint_usable"].sum()),
        },

        "checks": {
            str(key): bool(value)
            for key, value in checks.items()
        },

        "warnings":
            [str(x) for x in warnings_list],

        "ready_for_poland":
            bool(ready),
    }

    report_path = (
        OUT_DIR /
        "lithuania_benchmark_audit.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # ==================================================================
    # COMPLETE
    # ==================================================================

    header("SCRIPT 44 COMPLETE")

    print()
    print("Tile audit:")
    print(audit_csv)

    print()
    print("Split summary:")
    print(split_csv)

    print()
    print("City/split summary:")
    print(city_split_csv)

    print()
    print("Leakage audit:")
    print(overlap_csv)

    print()
    print("Audit JSON:")
    print(report_path)

    print()
    print("Figures (PNG + PDF):")
    print(FIG_DIR)

    print()

    if ready:
        print(
            "NEXT STAGE: POLAND"
        )
    else:
        print(
            "NEXT STAGE: FIX FAILED "
            "LITHUANIA AUDIT CHECKS"
        )


if __name__ == "__main__":
    main()