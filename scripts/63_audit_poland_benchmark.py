"""
CartoGen Script 63
Final audit of the Polish leakage-safe spatial road benchmark.

Inputs (from Script 62):
    data/processed/poland/spatial_benchmark/
        poland_spatial_tile_manifest.csv
        poland_spatial_tile_index.gpkg
        poland_road_benchmark.csv
        poland_road_benchmark.gpkg
        split_distribution_summary.csv
        city_split_representation.csv

Outputs:
    data/processed/poland/benchmark_audit/
        poland_tile_audit.csv
        split_distribution_summary.csv
        city_split_representation.csv
        operation_distribution.csv
        selection_distribution.csv
        cross_split_core_overlap.csv
        poland_benchmark_audit.json
        figures/*.png
        figures/*.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "processed" / "poland" / "spatial_benchmark"
OUTPUT_DIR = ROOT / "data" / "processed" / "poland" / "benchmark_audit"
FIG_DIR = OUTPUT_DIR / "figures"

EXPECTED_CITIES = ["Warsaw", "Krakow", "Gdansk"]
EXPECTED_SPLITS = ["train", "validation", "test"]
EXPECTED_SPLIT_COUNTS = {"train": 75, "validation": 15, "test": 18}
EXPECTED_TILE_COUNT = 108
EXPECTED_CRS = "EPSG:2180"

# Keep these aligned with Script 62.
MIN_SOURCE_ROADS = 20
MIN_RETAINED_ROADS = 5


def banner(text, char="="):
    print()
    print(char * 120)
    print(text)
    print(char * 120)


def subbanner(text):
    print()
    print("-" * 120)
    print(text)
    print("-" * 120)


def first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(
        "None of the expected files exists:\n" + "\n".join(str(p) for p in paths)
    )


def normalize_city(x):
    s = str(x).strip()
    lut = {
        "warsaw": "Warsaw",
        "krakow": "Krakow",
        "kraków": "Krakow",
        "gdansk": "Gdansk",
        "gdańsk": "Gdansk",
    }
    return lut.get(s.lower(), s)


def to_builtin(obj):
    """Recursively convert NumPy/Pandas scalars into JSON-safe Python values."""
    if obj is None:
        return None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_builtin(v) for v in obj]
    if pd.isna(obj):
        return None
    return obj


def save_figure(fig, stem):
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {png}")
    print(f"PDF: {pdf}")


def detect_column(df, candidates, required=True):
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(
            f"Could not identify required column. Tried {candidates}. "
            f"Available: {list(df.columns)}"
        )
    return None


def detect_columns(tiles, roads):
    cols = {}

    cols["tile_id"] = detect_column(
        tiles, ["tile_id", "tile", "tileid", "core_tile_id"]
    )
    cols["tile_city"] = detect_column(tiles, ["city", "city_name"])
    cols["tile_split"] = detect_column(tiles, ["split", "dataset_split"])

    cols["road_city"] = detect_column(roads, ["city", "city_name"])
    cols["road_split"] = detect_column(roads, ["split", "dataset_split"])
    cols["road_tile"] = detect_column(
        roads, ["tile_id", "tile", "tileid", "core_tile_id"]
    )
    cols["source_idx"] = detect_column(
        roads, ["source_idx", "source_id", "road_id", "source_road_id"]
    )
    cols["selection"] = detect_column(
        roads,
        ["selection_label", "selection", "selected_label", "retention_label"],
    )
    cols["operation"] = detect_column(
        roads, ["operation", "operation_label", "generalization_operation"],
        required=False,
    )

    # Prefer statistics written by Script 62. If absent, recompute from road rows.
    cols["tile_source"] = detect_column(
        tiles, ["source_roads", "source_count", "n_source_roads"], required=False
    )
    cols["tile_retained"] = detect_column(
        tiles,
        ["retained_roads", "selected_roads", "target_roads", "n_retained_roads"],
        required=False,
    )
    cols["tile_eliminated"] = detect_column(
        tiles, ["eliminated_roads", "elimination_roads", "n_eliminated_roads"],
        required=False,
    )
    cols["tile_retention"] = detect_column(
        tiles, ["retention", "retention_ratio", "road_retention"], required=False
    )
    cols["tile_usable"] = detect_column(
        tiles, ["road_usable", "road_usable_tile", "usable"], required=False
    )
    return cols


def prepare_tile_audit(tiles, roads, c):
    audit = tiles.copy()
    audit[c["tile_city"]] = audit[c["tile_city"]].map(normalize_city)
    roads = roads.copy()
    roads[c["road_city"]] = roads[c["road_city"]].map(normalize_city)

    # Recompute counts from benchmark rows so the audit is independent of
    # summary columns stored by Script 62.
    sel = roads[c["selection"]].astype(str).str.strip().str.lower()
    road_tmp = pd.DataFrame({
        "tile_id": roads[c["road_tile"]].astype(str),
        "selection": sel,
    })

    counts = (
        road_tmp.groupby("tile_id", dropna=False)
        .agg(
            audit_source_roads=("selection", "size"),
            audit_retained_roads=("selection", lambda s: int(s.isin(["selected", "weak_selected"]).sum())),
            audit_eliminated_roads=("selection", lambda s: int((s == "elimination").sum())),
        )
        .reset_index()
        .rename(columns={"tile_id": "_tile_key"})
    )
    counts["audit_retention"] = np.where(
        counts["audit_source_roads"] > 0,
        counts["audit_retained_roads"] / counts["audit_source_roads"],
        np.nan,
    )

    # Merge on a temporary key only.  Do not introduce a second tile_id
    # column, otherwise pandas suffixes the original tile_id to tile_id_x
    # and downstream validation can no longer find the canonical field.
    audit["_tile_key"] = audit[c["tile_id"]].astype(str)
    audit = audit.merge(counts, on="_tile_key", how="left")
    audit["audit_source_roads"] = audit["audit_source_roads"].fillna(0).astype(int)
    audit["audit_retained_roads"] = audit["audit_retained_roads"].fillna(0).astype(int)
    audit["audit_eliminated_roads"] = audit["audit_eliminated_roads"].fillna(0).astype(int)
    audit["audit_retention"] = np.where(
        audit["audit_source_roads"] > 0,
        audit["audit_retained_roads"] / audit["audit_source_roads"],
        np.nan,
    )
    audit["empty_road_tile"] = audit["audit_source_roads"].eq(0)
    audit["sparse_source"] = audit["audit_source_roads"].lt(MIN_SOURCE_ROADS)
    audit["sparse_retained"] = audit["audit_retained_roads"].lt(MIN_RETAINED_ROADS)
    audit["road_usable"] = ~(audit["sparse_source"] | audit["sparse_retained"])

    return audit, roads


def compute_cross_split_overlap(tiles, c):
    rows = []
    work = tiles[[c["tile_id"], c["tile_city"], c["tile_split"], "geometry"]].copy()
    work[c["tile_city"]] = work[c["tile_city"]].map(normalize_city)

    # Only comparisons within each city are meaningful.
    for city, grp in work.groupby(c["tile_city"]):
        grp = grp.reset_index(drop=True)
        sindex = grp.sindex
        for i, row in grp.iterrows():
            candidates = list(sindex.query(row.geometry, predicate="intersects"))
            for j in candidates:
                if j <= i:
                    continue
                other = grp.iloc[j]
                if row[c["tile_split"]] == other[c["tile_split"]]:
                    continue
                inter = row.geometry.intersection(other.geometry)
                area = float(inter.area) if not inter.is_empty else 0.0
                if area > 0:
                    rows.append({
                        "city": city,
                        "tile_id_1": row[c["tile_id"]],
                        "split_1": row[c["tile_split"]],
                        "tile_id_2": other[c["tile_id"]],
                        "split_2": other[c["tile_split"]],
                        "overlap_area_m2": area,
                    })
    return pd.DataFrame(
        rows,
        columns=[
            "city", "tile_id_1", "split_1", "tile_id_2",
            "split_2", "overlap_area_m2"
        ],
    )


def make_summaries(audit, roads, c):
    split_summary = (
        audit.groupby(c["tile_split"], observed=False)
        .agg(
            tiles=(c["tile_id"], "count"),
            source_roads=("audit_source_roads", "sum"),
            retained_roads=("audit_retained_roads", "sum"),
            eliminated_roads=("audit_eliminated_roads", "sum"),
            mean_retention=("audit_retention", "mean"),
            median_retention=("audit_retention", "median"),
            road_usable_tiles=("road_usable", "sum"),
        )
        .reset_index()
        .rename(columns={c["tile_split"]: "split"})
    )

    city_split = (
        audit.groupby([c["tile_city"], c["tile_split"]], observed=False)
        .agg(
            tiles=(c["tile_id"], "count"),
            source_roads=("audit_source_roads", "sum"),
            retained_roads=("audit_retained_roads", "sum"),
            eliminated_roads=("audit_eliminated_roads", "sum"),
            road_usable=("road_usable", "sum"),
        )
        .reset_index()
        .rename(columns={c["tile_city"]: "city", c["tile_split"]: "split"})
    )

    selection_distribution = (
        roads.groupby([c["road_city"], c["road_split"], c["selection"]], dropna=False)
        .size()
        .reset_index(name="roads")
        .rename(columns={
            c["road_city"]: "city",
            c["road_split"]: "split",
            c["selection"]: "selection_label",
        })
    )

    if c["operation"] is not None:
        retained = roads[
            roads[c["selection"]].astype(str).str.strip().str.lower().isin(["selected", "weak_selected"])
        ].copy()
        operation_distribution = (
            retained.groupby(
                [c["road_city"], c["road_split"], c["operation"]], dropna=False
            )
            .size()
            .reset_index(name="roads")
            .rename(columns={
                c["road_city"]: "city",
                c["road_split"]: "split",
                c["operation"]: "operation",
            })
        )
    else:
        operation_distribution = pd.DataFrame(
            columns=["city", "split", "operation", "roads"]
        )

    return split_summary, city_split, selection_distribution, operation_distribution


def plot_spatial_splits(tiles, c):
    fig, ax = plt.subplots(figsize=(10, 8))
    # Matplotlib default color cycle; no hard-coded colors.
    for split in EXPECTED_SPLITS:
        x = tiles[tiles[c["tile_split"]].astype(str) == split]
        if len(x):
            x.plot(ax=ax, alpha=0.55, edgecolor="black", linewidth=0.35, label=split)
    for city, grp in tiles.groupby(c["tile_city"]):
        pt = grp.geometry.unary_union.centroid
        ax.text(pt.x, pt.y, city, ha="center", va="center", fontsize=10)
    ax.set_title("Polish CartoGen benchmark: spatial train/validation/test tiles")
    ax.set_xlabel("Easting (m), EPSG:2180")
    ax.set_ylabel("Northing (m), EPSG:2180")
    ax.legend(title="Split")
    ax.set_aspect("equal")
    fig.tight_layout()
    save_figure(fig, "figure_01_spatial_splits")


def plot_density(audit, c):
    order = EXPECTED_CITIES
    data = [
        audit.loc[audit[c["tile_city"]] == city, "audit_source_roads"].values
        for city in order
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=order, showmeans=True)
    ax.axhline(MIN_SOURCE_ROADS, linestyle="--", linewidth=1)
    ax.set_title("Source-road density across Polish benchmark tiles")
    ax.set_ylabel("Source roads per 1 km² tile")
    fig.tight_layout()
    save_figure(fig, "figure_02_road_density")


def plot_retention_by_split(audit, c):
    data = [
        audit.loc[
            audit[c["tile_split"]].astype(str) == split, "audit_retention"
        ].dropna().values
        for split in EXPECTED_SPLITS
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=EXPECTED_SPLITS, showmeans=True)
    ax.set_title("Road retention by benchmark split")
    ax.set_ylabel("Retained / source roads")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_figure(fig, "figure_03_road_retention_by_split")


def plot_roads_by_city(audit, c):
    x = (
        audit.groupby(c["tile_city"])["audit_source_roads"]
        .sum()
        .reindex(EXPECTED_CITIES)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x.index, x.values)
    ax.set_title("Benchmark source roads by city")
    ax.set_ylabel("Road samples")
    for i, v in enumerate(x.values):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom")
    fig.tight_layout()
    save_figure(fig, "figure_04_roads_by_city")


def plot_selection_distribution(roads, c):
    tmp = roads.copy()
    tmp["selection_norm"] = tmp[c["selection"]].astype(str).str.lower()
    tab = (
        tmp.groupby([c["road_city"], "selection_norm"])
        .size()
        .unstack(fill_value=0)
        .reindex(EXPECTED_CITIES)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(tab))
    for label in tab.columns:
        vals = tab[label].to_numpy()
        ax.bar(tab.index, vals, bottom=bottom, label=str(label))
        bottom += vals
    ax.set_title("Road selection supervision by city")
    ax.set_ylabel("Road samples")
    ax.legend(title="Selection")
    fig.tight_layout()
    save_figure(fig, "figure_05_selection_distribution")


def plot_operation_distribution(roads, c):
    if c["operation"] is None:
        return
    retained = roads[
        roads[c["selection"]].astype(str).str.strip().str.lower().isin(["selected", "weak_selected"])
    ].copy()
    if retained.empty:
        return

    tab = (
        retained.groupby([c["road_city"], c["operation"]])
        .size()
        .unstack(fill_value=0)
        .reindex(EXPECTED_CITIES)
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(tab))
    for op in tab.columns:
        vals = tab[op].to_numpy()
        ax.bar(tab.index, vals, bottom=bottom, label=str(op))
        bottom += vals
    ax.set_title("Generalization-operation labels for retained roads")
    ax.set_ylabel("Retained road samples")
    ax.legend(title="Operation", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_figure(fig, "figure_06_operation_distribution")


def plot_city_split_retention(audit, c):
    table = (
        audit.groupby([c["tile_city"], c["tile_split"]])["audit_retention"]
        .mean()
        .unstack()
        .reindex(index=EXPECTED_CITIES, columns=EXPECTED_SPLITS)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(table.index))
    width = 0.24
    offsets = np.linspace(-width, width, len(EXPECTED_SPLITS))
    for off, split in zip(offsets, EXPECTED_SPLITS):
        vals = table[split].values
        ax.bar(x + off, vals, width=width, label=split)
    ax.set_xticks(x)
    ax.set_xticklabels(table.index)
    ax.set_ylabel("Mean road retention")
    ax.set_title("Mean road retention by city and split")
    ax.legend(title="Split")
    fig.tight_layout()
    save_figure(fig, "figure_07_city_split_retention")


def main():
    banner("CARTOGEN — FINAL POLISH BENCHMARK AUDIT")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    tile_path = first_existing(
        INPUT_DIR / "poland_spatial_tile_index.gpkg",
    )
    road_csv = first_existing(
        INPUT_DIR / "poland_road_benchmark.csv",
    )

    tiles = gpd.read_file(tile_path)
    roads = pd.read_csv(road_csv)

    c = detect_columns(tiles, roads)

    tiles[c["tile_city"]] = tiles[c["tile_city"]].map(normalize_city)
    roads[c["road_city"]] = roads[c["road_city"]].map(normalize_city)

    print(f"Manifest tiles: {len(tiles):,}")
    print(f"Benchmark roads: {len(roads):,}")
    print(f"CRS:            {tiles.crs}")

    audit, roads = prepare_tile_audit(tiles, roads, c)

    subbanner("1. STRUCTURAL VALIDATION")
    duplicate_tiles = int(audit[c["tile_id"]].duplicated().sum())
    splits = sorted(audit[c["tile_split"]].astype(str).unique().tolist())
    cities = sorted(audit[c["tile_city"]].astype(str).unique().tolist())

    print(f"Duplicate tile IDs: {duplicate_tiles}")
    print(f"Splits: {splits}")
    print(f"Cities: {cities}")

    # Every benchmark row should point to exactly one known tile.
    known_tiles = set(audit[c["tile_id"]].astype(str))
    road_tiles = roads[c["road_tile"]].astype(str)
    missing_tile_refs = int((~road_tiles.isin(known_tiles)).sum())

    duplicate_city_source = int(
        roads.duplicated(subset=[c["road_city"], c["source_idx"]]).sum()
    )

    print(f"Road rows with unknown tile: {missing_tile_refs}")
    print(f"Duplicate city/source_idx:   {duplicate_city_source}")

    subbanner("2. CORE TILE LEAKAGE")
    leakage = compute_cross_split_overlap(audit, c)
    print(f"Cross-split positive-area overlaps: {len(leakage)}")
    if leakage.empty:
        print("[PASS] No positive-area core leakage.")
    else:
        print("[FAIL] Cross-split positive-area core leakage detected.")

    subbanner("3. TILE DENSITY AND RETENTION")
    sparse_source = int(audit["sparse_source"].sum())
    sparse_retained = int(audit["sparse_retained"].sum())
    empty = int(audit["empty_road_tile"].sum())
    usable = int(audit["road_usable"].sum())

    print(f"Sparse source-road tiles:   {sparse_source}")
    print(f"Sparse retained-road tiles: {sparse_retained}")
    print(f"Empty road tiles:           {empty}")
    print(f"Road-usable tiles:          {usable}/{len(audit)}")

    split_summary, city_split, selection_dist, operation_dist = make_summaries(
        audit, roads, c
    )

    subbanner("4. SPLIT DISTRIBUTION")
    print(split_summary.to_string(index=False))

    subbanner("5. CITY × SPLIT REPRESENTATION")
    print(city_split.to_string(index=False))

    subbanner("6. SUPERVISION INTEGRITY")
    selection_norm = roads[c["selection"]].astype(str).str.strip().str.lower()
    allowed_selection = {"selected", "weak_selected", "elimination"}
    bad_selection = int((~selection_norm.isin(allowed_selection)).sum())

    strong_selected = selection_norm.eq("selected")
    weak_selected = selection_norm.eq("weak_selected")
    selected = selection_norm.isin(["selected", "weak_selected"])
    eliminated = selection_norm.eq("elimination")

    if c["operation"] is not None:
        op = roads[c["operation"]]
        op_missing = op.isna() | op.astype(str).str.strip().str.lower().isin(["", "nan", "none"])
        selected_missing_op = int((selected & op_missing).sum())
    else:
        selected_missing_op = int(selected.sum())

    # Validate the explicit binary supervision written by Script 61/62 when present.
    binary_partition_ok = True
    label_binary_agreement = True
    if "is_selected" in roads.columns and "is_eliminated" in roads.columns:
        binary_selected = pd.to_numeric(roads["is_selected"], errors="coerce").fillna(0).astype(int).eq(1)
        binary_eliminated = pd.to_numeric(roads["is_eliminated"], errors="coerce").fillna(0).astype(int).eq(1)
        binary_partition_ok = bool(
            ((binary_selected.astype(int) + binary_eliminated.astype(int)) == 1).all()
        )
        label_binary_agreement = bool(
            (selected == binary_selected).all()
            and (eliminated == binary_eliminated).all()
        )

    partition_ok = bool(
        bad_selection == 0
        and int(selected.sum() + eliminated.sum()) == len(roads)
        and binary_partition_ok
        and label_binary_agreement
    )

    print(f"Strong selected roads:           {int(strong_selected.sum()):,}")
    print(f"Weak selected roads:             {int(weak_selected.sum()):,}")
    print(f"Total retained roads:            {int(selected.sum()):,}")
    print(f"Eliminated roads:                {int(eliminated.sum()):,}")
    print(f"Unexpected selection labels:     {bad_selection:,}")
    print(f"Retained roads missing operation:{selected_missing_op:,}")
    print(f"Selection partition consistent:  {partition_ok}")
    print(f"Binary partition consistent:     {binary_partition_ok}")
    print(f"Label/binary agreement:          {label_binary_agreement}")

    if not operation_dist.empty:
        print()
        print("Retained-road operation distribution:")
        global_ops = (
            operation_dist.groupby("operation", dropna=False)["roads"]
            .sum()
            .sort_values(ascending=False)
        )
        for op_name, n in global_ops.items():
            print(f"  {str(op_name):35s} {int(n):6,d}")

    subbanner("7. GENERATING FIGURES")
    plot_spatial_splits(audit, c)
    plot_density(audit, c)
    plot_retention_by_split(audit, c)
    plot_roads_by_city(audit, c)
    plot_selection_distribution(roads, c)
    plot_operation_distribution(roads, c)
    plot_city_split_retention(audit, c)

    subbanner("8. FINAL READINESS ASSESSMENT")

    split_counts = audit[c["tile_split"]].astype(str).value_counts().to_dict()

    checks = {
        "tile_count_108": len(audit) == EXPECTED_TILE_COUNT,
        "unique_tile_ids": duplicate_tiles == 0,
        "all_splits_present": set(EXPECTED_SPLITS).issubset(set(splits)),
        "global_split_counts_correct": all(
            int(split_counts.get(k, 0)) == v
            for k, v in EXPECTED_SPLIT_COUNTS.items()
        ),
        "three_cities_present": set(EXPECTED_CITIES) == set(cities),
        "crs_2180": str(audit.crs).upper() == EXPECTED_CRS,
        "no_positive_area_cross_split_overlap": leakage.empty,
        "unique_city_source_idx": duplicate_city_source == 0,
        "benchmark_rows_have_known_tile": missing_tile_refs == 0,
        "benchmark_rows_have_split": roads[c["road_split"]].notna().all(),
        "selection_partition_consistent": partition_ok,
        "retained_have_operation": selected_missing_op == 0,
    }

    for key, value in checks.items():
        print(f"{key:45s} {'PASS' if bool(value) else 'FAIL'}")

    ready = all(bool(v) for v in checks.values())

    if ready:
        print()
        print("[PASS] POLISH BENCHMARK IS STRUCTURALLY READY.")
    else:
        print()
        print("[FAIL] POLISH BENCHMARK REQUIRES ATTENTION.")

    warnings = []
    if sparse_source:
        warnings.append(
            f"{sparse_source} tiles contain fewer than {MIN_SOURCE_ROADS} source roads."
        )
    if sparse_retained:
        warnings.append(
            f"{sparse_retained} tiles contain fewer than {MIN_RETAINED_ROADS} retained roads."
        )
    if empty:
        warnings.append(f"{empty} tiles contain no benchmark source roads.")

    if warnings:
        print()
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")

    subbanner("9. SAVING AUDIT")

    # Drop geometry for CSV; preserve full spatial index separately in Script 62.
    audit_csv = OUTPUT_DIR / "poland_tile_audit.csv"
    pd.DataFrame(audit.drop(columns="geometry")).to_csv(audit_csv, index=False)

    split_csv = OUTPUT_DIR / "split_distribution_summary.csv"
    split_summary.to_csv(split_csv, index=False)

    city_csv = OUTPUT_DIR / "city_split_representation.csv"
    city_split.to_csv(city_csv, index=False)

    selection_csv = OUTPUT_DIR / "selection_distribution.csv"
    selection_dist.to_csv(selection_csv, index=False)

    operation_csv = OUTPUT_DIR / "operation_distribution.csv"
    operation_dist.to_csv(operation_csv, index=False)

    leakage_csv = OUTPUT_DIR / "cross_split_core_overlap.csv"
    leakage.to_csv(leakage_csv, index=False)

    report = {
        "benchmark": "CartoGen Polish spatial road benchmark",
        "input_directory": str(INPUT_DIR),
        "crs": str(audit.crs),
        "tile_count": int(len(audit)),
        "benchmark_road_rows": int(len(roads)),
        "cities": cities,
        "splits": splits,
        "thresholds": {
            "minimum_source_roads_per_tile": MIN_SOURCE_ROADS,
            "minimum_retained_roads_per_tile": MIN_RETAINED_ROADS,
        },
        "density": {
            "empty_road_tiles": empty,
            "sparse_source_tiles": sparse_source,
            "sparse_retained_tiles": sparse_retained,
            "road_usable_tiles": usable,
        },
        "supervision": {
            "strong_selected_roads": int(strong_selected.sum()),
            "weak_selected_roads": int(weak_selected.sum()),
            "retained_roads": int(selected.sum()),
            "eliminated_roads": int(eliminated.sum()),
            "unexpected_selection_labels": bad_selection,
            "retained_missing_operation": selected_missing_op,
            "binary_partition_consistent": binary_partition_ok,
            "label_binary_agreement": label_binary_agreement,
        },
        "cross_split_positive_area_overlaps": int(len(leakage)),
        "checks": checks,
        "ready": ready,
        "warnings": warnings,
        "outputs": {
            "tile_audit": str(audit_csv),
            "split_summary": str(split_csv),
            "city_split_summary": str(city_csv),
            "selection_distribution": str(selection_csv),
            "operation_distribution": str(operation_csv),
            "leakage_audit": str(leakage_csv),
            "figures": str(FIG_DIR),
        },
    }

    report_json = OUTPUT_DIR / "poland_benchmark_audit.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(to_builtin(report), f, indent=2, ensure_ascii=False)

    banner("SCRIPT 63 COMPLETE")
    print(f"Tile audit:             {audit_csv}")
    print(f"Split summary:          {split_csv}")
    print(f"City/split summary:     {city_csv}")
    print(f"Selection distribution:{selection_csv}")
    print(f"Operation distribution:{operation_csv}")
    print(f"Leakage audit:          {leakage_csv}")
    print(f"Audit JSON:             {report_json}")
    print(f"Figures (PNG + PDF):    {FIG_DIR}")
    print()
    if ready:
        print("[PASS] FINAL POLISH BENCHMARK AUDIT PASSED.")
        print("POLAND STAGE COMPLETE.")
    else:
        print("[CHECK] Resolve failed audit checks before closing the Poland stage.")


if __name__ == "__main__":
    main()
