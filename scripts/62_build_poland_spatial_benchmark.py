#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CARTOGEN — SCRIPT 62
Build leakage-safe Polish spatial road benchmark tiles and train/validation/test splits.

Inputs
------
data/processed/poland/learning_dataset/poland_road_learning_dataset.csv
data/processed/poland/road_target_validation/{city}_validated_road_pair.gpkg
    expected layer: source_roads

Outputs
-------
data/processed/poland/spatial_benchmark/
    poland_spatial_tile_index.gpkg
    poland_spatial_tile_manifest.csv
    poland_spatial_tile_manifest.json
    poland_road_benchmark.csv
    poland_road_benchmark.gpkg
    split_distribution_summary.csv
    city_split_representation.csv
    benchmark_manifest.json

Design
------
* 1 km x 1 km core tiles in EPSG:2180.
* Exactly 36 core tiles (6 x 6) per city, centred on the actual source-road
  extent used by the validated pair.
* 25 train, 5 validation and 6 test tiles per city (75/15/18 globally).
* Core tiles never overlap with positive area.
* Each source road is assigned to exactly one tile using its representative
  midpoint; boundary ties are resolved deterministically.
* Full source geometry is retained; geometry is NOT clipped at tile borders.
* Script-61 supervision is joined by (city, source_idx).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

# ----------------------------------------------------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "poland"

LEARNING_CSV = (
    PROCESSED
    / "learning_dataset"
    / "poland_road_learning_dataset.csv"
)

VALIDATED_DIR = PROCESSED / "road_target_validation"

OUT_DIR = PROCESSED / "spatial_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TILE_SIZE_M = 1000.0
GRID_N = 6
EXPECTED_TILES_PER_CITY = GRID_N * GRID_N
EXPECTED_SPLITS = {"train": 25, "validation": 5, "test": 6}
EXPECTED_CRS = "EPSG:2180"

CITY_CONFIG = {
    "warsaw": "Warsaw",
    "krakow": "Krakow",
    "gdansk": "Gdansk",
}

# Sparse thresholds are audit flags, not deletion criteria.
MIN_SOURCE_ROADS = 20
MIN_RETAINED_ROADS = 3

# Fixed 6 x 6 split mask. It deliberately gives every city the same counts:
# train=25, validation=5, test=6.
#
# The mask is spatially deterministic. Core tiles are non-overlapping.
# Rows are indexed south -> north; columns west -> east.
SPLIT_MASK = [
    ["train", "train",      "test",       "train",      "validation", "train"],
    ["train", "validation", "train",      "train",      "train",      "test"],
    ["test",  "train",      "train",      "validation", "train",      "train"],
    ["train", "train",      "validation", "train",      "test",       "train"],
    ["train", "test",       "train",      "train",      "train",      "validation"],
    ["train", "train",      "train",      "test",       "train",      "train"],
]

# ----------------------------------------------------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------------------------------------------------

def banner(text: str) -> None:
    print("\n" + "=" * 120)
    print(text)
    print("=" * 120)


def section(text: str) -> None:
    print("\n" + "-" * 120)
    print(text)
    print("-" * 120)


def json_safe(obj):
    """Recursively convert numpy/pandas scalars into JSON-safe Python values."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if pd.isna(obj) if not isinstance(obj, (str, bytes)) else False:
        return None
    return obj


def find_source_layer(city_key: str) -> tuple[Path, str]:
    gpkg = VALIDATED_DIR / f"{city_key}_validated_road_pair.gpkg"
    if not gpkg.exists():
        raise FileNotFoundError(f"Validated road pair not found: {gpkg}")

    layers = list(gpd.list_layers(gpkg)["name"])
    preferred = ["source_roads", "source_road", "roads_source"]
    for name in preferred:
        if name in layers:
            return gpkg, name

    candidates = [x for x in layers if "source" in x.lower() and "road" in x.lower()]
    if len(candidates) == 1:
        return gpkg, candidates[0]

    raise RuntimeError(
        f"Could not uniquely identify source-road layer in {gpkg}\n"
        f"Available layers: {layers}"
    )


def normalize_city(series: pd.Series) -> pd.Series:
    lookup = {
        "warsaw": "Warsaw",
        "krakow": "Krakow",
        "gdansk": "Gdansk",
    }
    return series.astype(str).str.strip().str.lower().map(lookup)


def representative_xy(geom):
    """
    Deterministic representative point for a road.
    interpolate(0.5, normalized=True) gives the along-line midpoint.
    """
    if geom is None or geom.is_empty:
        return np.nan, np.nan
    try:
        p = geom.interpolate(0.5, normalized=True)
    except Exception:
        p = geom.representative_point()
    return float(p.x), float(p.y)


def make_grid(city_key: str, city_name: str, roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Build a 6 x 6, 1-km grid centred on the source-road extent midpoint.

    Snapping the lower-left corner to whole kilometres makes the grid stable
    across reruns and easy to reproduce.
    """
    minx, miny, maxx, maxy = roads.total_bounds
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0

    half = (GRID_N * TILE_SIZE_M) / 2.0
    x0 = math.floor((cx - half) / TILE_SIZE_M) * TILE_SIZE_M
    y0 = math.floor((cy - half) / TILE_SIZE_M) * TILE_SIZE_M

    rows = []
    for row in range(GRID_N):
        for col in range(GRID_N):
            xmin = x0 + col * TILE_SIZE_M
            ymin = y0 + row * TILE_SIZE_M
            xmax = xmin + TILE_SIZE_M
            ymax = ymin + TILE_SIZE_M
            split = SPLIT_MASK[row][col]
            tile_id = f"{city_key}_r{row:02d}_c{col:02d}"
            rows.append(
                {
                    "tile_id": tile_id,
                    "city": city_name,
                    "city_key": city_key,
                    "row": row,
                    "col": col,
                    "split": split,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "tile_size_m": TILE_SIZE_M,
                    "geometry": box(xmin, ymin, xmax, ymax),
                }
            )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=EXPECTED_CRS)


def assign_tiles(
    roads: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Assign each road to at most one core tile using its line midpoint.

    A half-open [xmin,xmax) x [ymin,ymax) convention eliminates ambiguity.
    The outer north/east edge is included for the last row/column.
    """
    xy = roads.geometry.apply(representative_xy)
    x = np.array([v[0] for v in xy], dtype=float)
    y = np.array([v[1] for v in xy], dtype=float)

    x0 = float(grid["xmin"].min())
    y0 = float(grid["ymin"].min())
    x1 = float(grid["xmax"].max())
    y1 = float(grid["ymax"].max())

    inside = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= x0)
        & (x <= x1)
        & (y >= y0)
        & (y <= y1)
    )

    col = np.floor((x - x0) / TILE_SIZE_M).astype(np.float64)
    row = np.floor((y - y0) / TILE_SIZE_M).astype(np.float64)

    # Points exactly on the outer max edge belong to the final tile.
    col[np.isclose(x, x1)] = GRID_N - 1
    row[np.isclose(y, y1)] = GRID_N - 1

    valid = (
        inside
        & (col >= 0)
        & (col < GRID_N)
        & (row >= 0)
        & (row < GRID_N)
    )

    result = pd.DataFrame(
        {
            "source_idx": roads["source_idx"].to_numpy(),
            "rep_x": x,
            "rep_y": y,
            "row": pd.Series(row).where(valid).astype("Int64"),
            "col": pd.Series(col).where(valid).astype("Int64"),
            "inside_benchmark": valid,
        }
    )

    tile_lookup = grid.set_index(["row", "col"])[["tile_id", "split"]]
    valid_rows = result.loc[result["inside_benchmark"], ["row", "col"]].copy()
    keys = list(zip(valid_rows["row"].astype(int), valid_rows["col"].astype(int)))
    result.loc[result["inside_benchmark"], "tile_id"] = [
        tile_lookup.loc[k, "tile_id"] for k in keys
    ]
    result.loc[result["inside_benchmark"], "split"] = [
        tile_lookup.loc[k, "split"] for k in keys
    ]
    return result


def positive_area_cross_split_overlaps(grid: gpd.GeoDataFrame) -> list[dict]:
    """
    Explicitly audit positive-area overlaps between tiles assigned to different splits.
    Touching edges/corners are allowed and have zero intersection area.
    """
    out = []
    sindex = grid.sindex
    geoms = grid.geometry.to_list()

    for i, geom in enumerate(geoms):
        for j in sindex.query(geom, predicate="intersects"):
            j = int(j)
            if j <= i:
                continue
            if grid.iloc[i]["split"] == grid.iloc[j]["split"]:
                continue
            area = float(geom.intersection(geoms[j]).area)
            if area > 1e-9:
                out.append(
                    {
                        "tile_a": grid.iloc[i]["tile_id"],
                        "split_a": grid.iloc[i]["split"],
                        "tile_b": grid.iloc[j]["tile_id"],
                        "split_b": grid.iloc[j]["split"],
                        "overlap_area_m2": area,
                    }
                )
    return out


def validate_learning_schema(df: pd.DataFrame) -> None:
    required = {
        "city",
        "source_idx",
        "selection_label",
        "is_selected",
        "is_eliminated",
        "operation_label",
        "target_operation",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "Script-61 global learning CSV is missing required columns:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )


# ----------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------

def main() -> None:
    banner("CARTOGEN — BUILD LEAKAGE-SAFE POLISH SPATIAL ROAD BENCHMARK")

    if not LEARNING_CSV.exists():
        raise FileNotFoundError(
            f"Script-61 global learning dataset not found:\n{LEARNING_CSV}"
        )

    learning = pd.read_csv(LEARNING_CSV)
    validate_learning_schema(learning)
    learning["city"] = normalize_city(learning["city"])

    if learning["city"].isna().any():
        raise RuntimeError("Unrecognized city values found in Script-61 learning dataset.")

    all_grids = []
    all_benchmark = []
    all_tile_rows = []
    city_reports = []
    leakage_rows = []

    for city_key, city_name in CITY_CONFIG.items():
        banner(f"{city_name.upper()} ({city_name})")

        gpkg, source_layer = find_source_layer(city_key)

        section("READING VALIDATED SOURCE ROADS")
        roads = gpd.read_file(gpkg, layer=source_layer)
        if roads.crs is None:
            raise RuntimeError(f"{city_name}: source roads have no CRS.")
        roads = roads.to_crs(EXPECTED_CRS)

        city_learning = learning.loc[learning["city"].eq(city_name)].copy()
        city_learning["source_idx"] = pd.to_numeric(
            city_learning["source_idx"], errors="raise"
        ).astype(int)

        # Script 61 used source_idx as the row index of the validated geometry.
        roads = roads.reset_index(drop=True)
        roads["source_idx"] = np.arange(len(roads), dtype=int)

        print(f"Layer:              {source_layer}")
        print(f"Source roads:       {len(roads):,}")
        print(f"Learning rows:      {len(city_learning):,}")
        print(f"CRS:                {roads.crs}")

        if len(roads) != len(city_learning):
            raise RuntimeError(
                f"{city_name}: geometry/learning count mismatch "
                f"({len(roads):,} vs {len(city_learning):,})."
            )

        if city_learning["source_idx"].duplicated().any():
            raise RuntimeError(f"{city_name}: duplicate source_idx values in learning data.")

        expected_idx = np.arange(len(roads), dtype=int)
        observed_idx = np.sort(city_learning["source_idx"].to_numpy())
        if not np.array_equal(expected_idx, observed_idx):
            raise RuntimeError(f"{city_name}: source_idx is not a complete 0..N-1 sequence.")

        section("CONSTRUCTING 6 × 6 CORE TILE GRID")
        grid = make_grid(city_key, city_name, roads)
        counts = grid["split"].value_counts().to_dict()
        print(f"Tiles:              {len(grid)}")
        print(f"Train:              {counts.get('train', 0)}")
        print(f"Validation:         {counts.get('validation', 0)}")
        print(f"Test:               {counts.get('test', 0)}")
        print(f"Tile size:          {TILE_SIZE_M/1000:.1f} × {TILE_SIZE_M/1000:.1f} km")

        if len(grid) != EXPECTED_TILES_PER_CITY:
            raise RuntimeError(f"{city_name}: expected 36 tiles.")
        if counts != EXPECTED_SPLITS:
            raise RuntimeError(
                f"{city_name}: split counts are wrong. "
                f"Observed={counts}, expected={EXPECTED_SPLITS}"
            )

        overlaps = positive_area_cross_split_overlaps(grid)
        leakage_rows.extend(overlaps)
        print(f"Cross-split positive-area overlaps: {len(overlaps)}")
        if overlaps:
            raise RuntimeError(f"{city_name}: positive-area cross-split tile leakage detected.")

        section("ASSIGNING SOURCE ROADS TO CORE TILES")
        assignments = assign_tiles(roads, grid)
        inside_n = int(assignments["inside_benchmark"].sum())
        outside_n = int((~assignments["inside_benchmark"]).sum())
        print(f"Inside benchmark:   {inside_n:,}")
        print(f"Outside 6×6 grid:   {outside_n:,}")

        # Join Script-61 supervision to source geometry.
        city_learning = city_learning.set_index("source_idx", drop=False)
        roads = roads.set_index("source_idx", drop=False)

        assignment_inside = assignments.loc[
            assignments["inside_benchmark"]
        ].copy()
        assignment_inside["source_idx"] = assignment_inside["source_idx"].astype(int)
        assignment_inside = assignment_inside.set_index("source_idx", drop=False)

        bench = city_learning.loc[assignment_inside.index].copy()
        bench["tile_id"] = assignment_inside["tile_id"]
        bench["split"] = assignment_inside["split"]
        bench["rep_x"] = assignment_inside["rep_x"]
        bench["rep_y"] = assignment_inside["rep_y"]
        bench["geometry"] = roads.loc[assignment_inside.index, "geometry"]

        bench_gdf = gpd.GeoDataFrame(
            bench.reset_index(drop=True),
            geometry="geometry",
            crs=EXPECTED_CRS,
        )

        # No road can occur in more than one core tile.
        duplicate_assignment = bool(
            bench_gdf.duplicated(subset=["city", "source_idx"]).any()
        )
        if duplicate_assignment:
            raise RuntimeError(f"{city_name}: duplicate road-to-core-tile assignment detected.")

        section("COMPUTING TILE STATISTICS")
        stats = (
            bench_gdf.groupby("tile_id", as_index=False)
            .agg(
                source_roads=("source_idx", "size"),
                retained_roads=("is_selected", "sum"),
                eliminated_roads=("is_eliminated", "sum"),
                mean_source_length_m=("length_m", "mean"),
            )
        )

        op_counts = (
            bench_gdf.groupby(["tile_id", "operation_label"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        op_counts.columns = [
            "tile_id" if c == "tile_id" else f"op_{c}"
            for c in op_counts.columns
        ]
        stats = stats.merge(op_counts, on="tile_id", how="left")

        grid = grid.merge(stats, on="tile_id", how="left")
        for c in ["source_roads", "retained_roads", "eliminated_roads"]:
            grid[c] = grid[c].fillna(0).astype(int)

        grid["retention_rate"] = np.where(
            grid["source_roads"] > 0,
            grid["retained_roads"] / grid["source_roads"],
            np.nan,
        )
        grid["source_road_density_km2"] = (
            grid["source_roads"] / (TILE_SIZE_M * TILE_SIZE_M / 1_000_000.0)
        )
        grid["retained_road_density_km2"] = (
            grid["retained_roads"] / (TILE_SIZE_M * TILE_SIZE_M / 1_000_000.0)
        )
        grid["empty_road_tile"] = grid["source_roads"].eq(0)
        grid["sparse_source_tile"] = grid["source_roads"].lt(MIN_SOURCE_ROADS)
        grid["sparse_retained_tile"] = grid["retained_roads"].lt(MIN_RETAINED_ROADS)
        grid["road_usable"] = (
            (~grid["empty_road_tile"])
            & (~grid["sparse_source_tile"])
            & (~grid["sparse_retained_tile"])
        )

        print(f"Empty road tiles:   {int(grid['empty_road_tile'].sum())}")
        print(f"Sparse source:      {int(grid['sparse_source_tile'].sum())}")
        print(f"Sparse retained:    {int(grid['sparse_retained_tile'].sum())}")
        print(f"Road-usable tiles:  {int(grid['road_usable'].sum())}/{len(grid)}")

        split_stats = (
            grid.groupby("split", as_index=False)
            .agg(
                tiles=("tile_id", "size"),
                source_roads=("source_roads", "sum"),
                retained_roads=("retained_roads", "sum"),
                road_usable_tiles=("road_usable", "sum"),
                mean_retention=("retention_rate", "mean"),
                median_retention=("retention_rate", "median"),
            )
        )
        print("\nSplit summary:")
        print(split_stats.to_string(index=False))

        city_reports.append(
            {
                "city": city_name,
                "source_roads_total": len(roads),
                "benchmark_roads": len(bench_gdf),
                "roads_outside_grid": outside_n,
                "tiles": len(grid),
                "train_tiles": int((grid["split"] == "train").sum()),
                "validation_tiles": int((grid["split"] == "validation").sum()),
                "test_tiles": int((grid["split"] == "test").sum()),
                "empty_tiles": int(grid["empty_road_tile"].sum()),
                "road_usable_tiles": int(grid["road_usable"].sum()),
            }
        )

        all_grids.append(grid)
        all_benchmark.append(bench_gdf)

        manifest_cols = [
            "tile_id", "city", "city_key", "row", "col", "split",
            "xmin", "ymin", "xmax", "ymax", "tile_size_m",
            "source_roads", "retained_roads", "eliminated_roads",
            "retention_rate", "source_road_density_km2",
            "retained_road_density_km2", "empty_road_tile",
            "sparse_source_tile", "sparse_retained_tile", "road_usable",
        ]
        all_tile_rows.append(pd.DataFrame(grid.drop(columns="geometry"))[manifest_cols])

    # ------------------------------------------------------------------------------------------------------------------
    # GLOBAL ASSEMBLY
    # ------------------------------------------------------------------------------------------------------------------

    banner("GLOBAL POLISH SPATIAL ROAD BENCHMARK")

    tile_index = gpd.GeoDataFrame(
        pd.concat(all_grids, ignore_index=True),
        geometry="geometry",
        crs=EXPECTED_CRS,
    )
    benchmark = gpd.GeoDataFrame(
        pd.concat(all_benchmark, ignore_index=True),
        geometry="geometry",
        crs=EXPECTED_CRS,
    )
    tile_manifest = pd.concat(all_tile_rows, ignore_index=True)

    print(f"Tiles:              {len(tile_index):,}")
    print(f"Benchmark roads:    {len(benchmark):,}")
    print(f"CRS:                {tile_index.crs}")

    section("GLOBAL SPLIT DISTRIBUTION")
    split_summary = (
        tile_index.groupby("split", as_index=False)
        .agg(
            tiles=("tile_id", "size"),
            source_roads=("source_roads", "sum"),
            retained_roads=("retained_roads", "sum"),
            eliminated_roads=("eliminated_roads", "sum"),
            mean_retention=("retention_rate", "mean"),
            median_retention=("retention_rate", "median"),
            road_usable_tiles=("road_usable", "sum"),
        )
    )
    split_order = pd.Categorical(
        split_summary["split"],
        categories=["train", "validation", "test"],
        ordered=True,
    )
    split_summary = (
        split_summary.assign(_order=split_order)
        .sort_values("_order")
        .drop(columns="_order")
    )
    print(split_summary.to_string(index=False))

    section("CITY × SPLIT REPRESENTATION")
    city_split = (
        tile_index.groupby(["city", "split"], as_index=False)
        .agg(
            tiles=("tile_id", "size"),
            source_roads=("source_roads", "sum"),
            retained_roads=("retained_roads", "sum"),
            eliminated_roads=("eliminated_roads", "sum"),
            road_usable=("road_usable", "sum"),
        )
    )
    print(city_split.to_string(index=False))

    # ------------------------------------------------------------------------------------------------------------------
    # FINAL VALIDATION
    # ------------------------------------------------------------------------------------------------------------------

    banner("FINAL VALIDATION")

    global_overlap = positive_area_cross_split_overlaps(tile_index)
    leakage_rows.extend(global_overlap)

    split_counts = tile_index["split"].value_counts().to_dict()
    expected_global = {
        k: v * len(CITY_CONFIG)
        for k, v in EXPECTED_SPLITS.items()
    }

    validations = {
        "tile_count_108": len(tile_index) == 108,
        "unique_tile_ids": not tile_index["tile_id"].duplicated().any(),
        "all_splits_present": set(tile_index["split"].unique())
            == {"train", "validation", "test"},
        "global_split_counts_correct": split_counts == expected_global,
        "three_cities_present": set(tile_index["city"].unique())
            == set(CITY_CONFIG.values()),
        "crs_2180": tile_index.crs is not None
            and tile_index.crs.to_epsg() == 2180,
        "no_positive_area_cross_split_overlap": len(global_overlap) == 0,
        "unique_city_source_idx": not benchmark.duplicated(
            subset=["city", "source_idx"]
        ).any(),
        "benchmark_rows_have_tile": benchmark["tile_id"].notna().all(),
        "benchmark_rows_have_split": benchmark["split"].notna().all(),
        "selection_partition_consistent": (
            benchmark["is_selected"].astype(int)
            + benchmark["is_eliminated"].astype(int)
            == 1
        ).all(),
        "retained_have_operation": (
            benchmark.loc[
                benchmark["is_selected"].astype(int).eq(1),
                "target_operation",
            ].astype(str) != "none_eliminated"
        ).all(),
    }

    for key, value in validations.items():
        print(f"{key:<50} {'PASS' if bool(value) else 'FAIL'}")

    if not all(bool(v) for v in validations.values()):
        failed = [k for k, v in validations.items() if not bool(v)]
        raise RuntimeError(
            "Polish spatial benchmark validation failed:\n"
            + "\n".join(f"  - {x}" for x in failed)
        )

    # ------------------------------------------------------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------------------------------------------------------

    section("SAVING BENCHMARK")

    tile_gpkg = OUT_DIR / "poland_spatial_tile_index.gpkg"
    benchmark_gpkg = OUT_DIR / "poland_road_benchmark.gpkg"
    benchmark_csv = OUT_DIR / "poland_road_benchmark.csv"
    tile_csv = OUT_DIR / "poland_spatial_tile_manifest.csv"
    tile_json = OUT_DIR / "poland_spatial_tile_manifest.json"
    split_csv = OUT_DIR / "split_distribution_summary.csv"
    city_split_csv = OUT_DIR / "city_split_representation.csv"
    leakage_csv = OUT_DIR / "cross_split_core_overlap.csv"
    manifest_json = OUT_DIR / "benchmark_manifest.json"

    if tile_gpkg.exists():
        tile_gpkg.unlink()
    if benchmark_gpkg.exists():
        benchmark_gpkg.unlink()

    tile_index.to_file(
        tile_gpkg,
        layer="spatial_tiles",
        driver="GPKG",
        engine="pyogrio",
    )
    benchmark.to_file(
        benchmark_gpkg,
        layer="road_benchmark",
        driver="GPKG",
        engine="pyogrio",
    )

    benchmark.drop(columns="geometry").to_csv(benchmark_csv, index=False)
    tile_manifest.to_csv(tile_csv, index=False)
    split_summary.to_csv(split_csv, index=False)
    city_split.to_csv(city_split_csv, index=False)

    leakage_df = pd.DataFrame(
        leakage_rows,
        columns=[
            "tile_a", "split_a", "tile_b", "split_b", "overlap_area_m2"
        ],
    ).drop_duplicates()
    leakage_df.to_csv(leakage_csv, index=False)

    tile_records = tile_manifest.replace({np.nan: None}).to_dict("records")
    with tile_json.open("w", encoding="utf-8") as f:
        json.dump(json_safe(tile_records), f, indent=2, ensure_ascii=False)

    manifest = {
        "script": "62_build_poland_spatial_benchmark.py",
        "benchmark": "CartoGen Polish road spatial benchmark",
        "crs": EXPECTED_CRS,
        "tile_size_m": TILE_SIZE_M,
        "grid_per_city": f"{GRID_N}x{GRID_N}",
        "tiles_per_city": EXPECTED_TILES_PER_CITY,
        "total_tiles": len(tile_index),
        "split_target_per_city": EXPECTED_SPLITS,
        "global_split_counts": split_counts,
        "benchmark_road_rows": len(benchmark),
        "sparse_thresholds": {
            "minimum_source_roads": MIN_SOURCE_ROADS,
            "minimum_retained_roads": MIN_RETAINED_ROADS,
        },
        "city_reports": city_reports,
        "validation": {k: bool(v) for k, v in validations.items()},
        "outputs": {
            "tile_index_gpkg": str(tile_gpkg),
            "benchmark_gpkg": str(benchmark_gpkg),
            "benchmark_csv": str(benchmark_csv),
            "tile_manifest_csv": str(tile_csv),
            "tile_manifest_json": str(tile_json),
            "split_summary_csv": str(split_csv),
            "city_split_summary_csv": str(city_split_csv),
            "leakage_audit_csv": str(leakage_csv),
        },
    }

    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(json_safe(manifest), f, indent=2, ensure_ascii=False)

    banner("SCRIPT 62 COMPLETE")
    print(f"Tile index:          {tile_gpkg}")
    print(f"Tile manifest:       {tile_csv}")
    print(f"Road benchmark CSV:  {benchmark_csv}")
    print(f"Road benchmark GPKG: {benchmark_gpkg}")
    print(f"Split summary:       {split_csv}")
    print(f"City/split summary:  {city_split_csv}")
    print(f"Leakage audit:       {leakage_csv}")
    print(f"Manifest:            {manifest_json}")
    print()
    print("[PASS] Polish leakage-safe spatial road benchmark created successfully.")
    print()
    print("NEXT: Script 63 — final Polish benchmark audit and publication-quality PNG/PDF figures.")


if __name__ == "__main__":
    main()
