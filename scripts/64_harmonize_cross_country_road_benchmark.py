#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
64_harmonize_cross_country_road_benchmark.py

CartoGen — harmonize Lithuania and Poland road-learning benchmarks.

Purpose
-------
1. Load the completed Lithuania and Poland road learning datasets.
2. Inspect schema/label compatibility without modifying upstream files.
3. Standardize country/city/split/selection semantics.
4. Preserve strong vs weak Polish retained supervision.
5. Build a common numeric feature space using only genuinely shared columns.
6. Detect duplicate sample identifiers and obvious split inconsistencies.
7. Export harmonized per-country tables, a combined table, schema reports,
   summary CSV/JSON, and publication-quality PNG/PDF figures.

This script deliberately does NOT normalize feature values globally. Scaling must
be fitted on training data only in the later modelling script to avoid leakage.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LT_CRS = "EPSG:3346"

ROOT = Path(r"D:\Research\CartoGen")

LT_TILE_INDEX = (
    ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "spatial_tiles"
    / "lithuania_spatial_tile_index.gpkg"
)

LT_CANDIDATES = [
    ROOT / "data/processed/lithuania/learning_dataset/lithuania_road_learning_dataset.csv",
]

PL_CANDIDATES = [
    ROOT / "data/processed/poland/spatial_benchmark/poland_road_benchmark.csv",
    ROOT / "data/processed/poland/learning_dataset/poland_road_learning_dataset.csv",
]

OUT = ROOT / "data/processed/cross_country/harmonized_roads"
FIG = OUT / "figures"

VALID_SPLITS = {"train", "validation", "test"}
POSITIVE_LABELS = {"selected", "weak_selected", "retained"}
NEGATIVE_LABELS = {"elimination", "eliminated"}

# Explicit transferable feature set for cross-country modelling.
# Raw coordinates are retained only as spatial metadata, because Lithuania and
# Poland use different national projected CRSs. orientation_deg and
# vertex_count are excluded from modelling because the current Polish benchmark
# has no useful variation in those fields.
MODEL_FEATURES = [
    "length_m",
    "intersecting_road_count",
    "neighbor_count_150m",
    "mean_neighbor_distance_m",
]

SPATIAL_METADATA = ["centroid_x", "centroid_y"]


def banner(s: str) -> None:
    print("\n" + "=" * 120)
    print(s)
    print("=" * 120)


def section(s: str) -> None:
    print("\n" + "-" * 120)
    print(s)
    print("-" * 120)


def first_existing(paths: list[Path], name: str) -> Path:
    found = [p for p in paths if p.exists()]
    if not found:
        raise FileNotFoundError(
            f"Could not locate {name}.\nChecked:\n" +
            "\n".join(f"  {p}" for p in paths)
        )
    return found[0]


def assign_lithuania_spatial_splits(df):
    """
    Assign the already-audited Lithuanian spatial tile/split to every
    Lithuanian road using its centroid coordinates.

    No new split is generated here.
    """
    import geopandas as gpd

    required = {"centroid_x", "centroid_y", "city", "window"}
    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            f"Cannot assign Lithuanian spatial splits. "
            f"Missing columns: {sorted(missing)}"
        )

    print("Loading audited Lithuanian spatial tile index...")
    if not LT_TILE_INDEX.exists():
        raise FileNotFoundError(f"Lithuanian tile index not found: {LT_TILE_INDEX}")
    tiles = gpd.read_file(LT_TILE_INDEX, layer="tiles")

    required_tile_cols = {"tile_id", "city", "city_key", "split", "geometry"}
    missing_tiles = required_tile_cols - set(tiles.columns)

    if missing_tiles:
        raise KeyError(
            f"Lithuanian tile index is missing columns: "
            f"{sorted(missing_tiles)}"
        )

    if tiles.crs is None:
        raise ValueError("Lithuanian tile index has no CRS.")

    if str(tiles.crs).upper() != LT_CRS:
        raise ValueError(
            f"Unexpected Lithuanian tile CRS: {tiles.crs}. "
            f"Expected {LT_CRS}."
        )

    work = df.copy()

    points = gpd.GeoDataFrame(
        work,
        geometry=gpd.points_from_xy(
            work["centroid_x"],
            work["centroid_y"],
        ),
        crs=LT_CRS,
    )

    # Use city_key/window as an additional safeguard so a road can only
    # inherit a tile belonging to its own analysis window.
    tiles_join = tiles[
        ["tile_id", "city_key", "split", "geometry"]
    ].copy()

    joined = gpd.sjoin(
        points,
        tiles_join,
        how="left",
        predicate="within",
    )

    # Boundary points are theoretically possible. Resolve unmatched
    # points with intersects rather than inventing a nearest tile.
    unmatched = joined["tile_id"].isna()

    if unmatched.any():
        boundary_points = points.loc[unmatched].copy()

        boundary_join = gpd.sjoin(
            boundary_points,
            tiles_join,
            how="left",
            predicate="intersects",
        )

        boundary_join = boundary_join[
            boundary_join["city_key"] == boundary_join["window"]
        ]

        if boundary_join.index.duplicated().any():
            duplicates = boundary_join[
                boundary_join.index.duplicated(keep=False)
            ]

            raise RuntimeError(
                "Some Lithuanian road centroids intersect multiple tiles. "
                f"Ambiguous road rows: {duplicates.index.nunique()}"
            )

        joined.loc[
            boundary_join.index, "tile_id"
        ] = boundary_join["tile_id"]

        joined.loc[
            boundary_join.index, "city_key"
        ] = boundary_join["city_key"]

        joined.loc[
            boundary_join.index, "split"
        ] = boundary_join["split"]

    # Verify road window agrees with tile city_key.
    assigned = joined["tile_id"].notna()

    mismatch = (
        assigned
        & (
            joined["window"].astype(str).str.lower()
            != joined["city_key"].astype(str).str.lower()
        )
    )

    if mismatch.any():
        raise RuntimeError(
            "Lithuanian road-to-tile city mismatch detected: "
            f"{int(mismatch.sum())} rows."
        )

    unassigned = int(joined["tile_id"].isna().sum())

    print(f"Lithuanian roads:       {len(joined):,}")
    print(f"Assigned to tile:       {len(joined)-unassigned:,}")
    print(f"Unassigned to tile:     {unassigned:,}")

    if unassigned:
        print(
            f"[INFO] Excluding {unassigned:,} Lithuanian roads "
            "whose centroids fall outside the audited core tile extent."
        )

        print(
            f"Core retention: "
            f"{100.0 * (len(joined) - unassigned) / len(joined):.2f}%"
        )

        keep = joined["tile_id"].notna()

        joined = joined.loc[keep].copy()
        work = work.loc[keep].copy()

        # Restore simple sequential indices so assignment below is positional.
        joined = joined.reset_index(drop=True)
        work = work.reset_index(drop=True)

    print("\nLithuanian split distribution:")
    print(joined["split"].value_counts().to_string())

    print("\nLithuanian tile count by split:")
    print(
        joined.groupby("split")["tile_id"]
        .nunique()
        .to_string()
    )

    work["tile_id"] = joined["tile_id"].values
    work["tile_id_harmonized"] = joined["tile_id"].astype(str).values
    work["split_harmonized"] = joined["split"].map(normalize_split).values

    return work

def norm_col(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mapping = {}
    used = set()
    for c in out.columns:
        n = norm_col(c)
        base = n
        i = 2
        while n in used:
            n = f"{base}_{i}"
            i += 1
        used.add(n)
        mapping[c] = n
    return out.rename(columns=mapping)


def find_col(df: pd.DataFrame, candidates: list[str], required=False) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    if required:
        raise KeyError(
            f"Required column not found. Tried {candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )
    return None


def normalize_split(x: Any) -> str:
    s = str(x).strip().lower()
    aliases = {
        "val": "validation",
        "valid": "validation",
        "dev": "validation",
        "training": "train",
        "testing": "test",
    }
    return aliases.get(s, s)


def normalize_label(x: Any) -> str:
    if pd.isna(x):
        return "unknown"
    s = str(x).strip().lower()
    aliases = {
        "eliminated": "elimination",
        "retain": "selected",
        "retained": "selected",
        "keep": "selected",
        "kept": "selected",
    }
    return aliases.get(s, s)


def infer_binary(df: pd.DataFrame, label_col: str | None) -> tuple[pd.Series, pd.Series]:
    """Return retained binary label and harmonized detailed label."""
    if label_col is not None:
        detail = df[label_col].map(normalize_label)
        retained = pd.Series(
            np.where(detail.isin(POSITIVE_LABELS), 1,
                     np.where(detail.isin(NEGATIVE_LABELS), 0, np.nan)),
            index=df.index,
            dtype="float64",
        )
        return retained, detail

    sel = find_col(df, ["is_selected", "selected", "retained"])
    elim = find_col(df, ["is_eliminated", "eliminated"])
    if sel:
        retained = pd.to_numeric(df[sel], errors="coerce")
        detail = retained.map({1.0: "selected", 0.0: "elimination"}).fillna("unknown")
        return retained, detail
    if elim:
        e = pd.to_numeric(df[elim], errors="coerce")
        retained = 1 - e
        detail = retained.map({1.0: "selected", 0.0: "elimination"}).fillna("unknown")
        return retained, detail

    # Lithuania Script 42 stores binary supervision in road_operation.
    if "road_operation" in df.columns:
        s = df["road_operation"].astype(str).str.strip().str.lower()

        valid = s.isin(["selected", "elimination"])

        if valid.all():
            retained = s.map({
                "selected": 1,
                "elimination": 0,
            }).astype("int8")

            detail = s.copy()

            return retained, detail

    raise KeyError(
        "Could not identify road selection supervision. "
        f"Available columns: {list(df.columns)}"
    )


def standardize(df: pd.DataFrame, country: str) -> pd.DataFrame:
    d = normalize_columns(df)

    city_col = find_col(d, ["city", "city_name"], required=True)
    split_col = find_col(d, ["split", "dataset_split", "benchmark_split"])
    source_col = find_col(
        d, ["source_idx", "source_id", "road_id", "feature_id", "feature_index", "id"],
        required=True,
    )
    tile_col = find_col(d, ["tile_id", "tile", "tile_name"])
    label_col = find_col(d, ["selection_label", "label", "selection"])

    retained, detail = infer_binary(d, label_col)

    d["country"] = country
    d["city_harmonized"] = d[city_col].astype(str).str.strip()
    d["source_idx_harmonized"] = d[source_col].astype(str)
    d["selection_label_harmonized"] = detail
    d["retained"] = retained

    if split_col:
        d["split_harmonized"] = d[split_col].map(normalize_split)
    else:
        d["split_harmonized"] = "unassigned"

    if tile_col:
        d["tile_id_harmonized"] = d[tile_col].astype(str)
    else:
        d["tile_id_harmonized"] = ""

    # Preserve operation semantics where available.
    op_col = find_col(
        d,
        ["operation", "operation_label", "generalization_operation",
         "road_operation", "target_operation"],
    )
    if op_col:
        d["operation_harmonized"] = d[op_col].fillna("").astype(str).str.strip()
    else:
        d["operation_harmonized"] = ""

    d["sample_uid"] = (
        d["country"].str.lower() + "::" +
        d["city_harmonized"].str.lower() + "::" +
        d["source_idx_harmonized"]
    )
    return d


def numeric_feature_candidates(df: pd.DataFrame) -> set[str]:
    exclude_exact = {
        "retained", "is_selected", "is_eliminated", "target_idx",
        "source_idx", "source_id", "id", "tile_id",
    }
    exclude_tokens = {
        "label", "split", "country", "city", "operation", "confidence",
        "uid", "target_id", "target_idx", "match_score", "coverage",
        "accepted",
    }

    ans = set()
    for c in df.columns:
        if c in exclude_exact:
            continue
        if any(tok in c for tok in exclude_tokens):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            ans.add(c)
    return ans


def save_fig(fig, stem: str) -> None:
    png = FIG / f"{stem}.png"
    pdf = FIG / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {png}")
    print(f"PDF: {pdf}")


def json_safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, Path):
        return str(v)
    if pd.isna(v) if not isinstance(v, (dict, list, tuple)) else False:
        return None
    return v


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    banner("CARTOGEN — HARMONIZE LITHUANIA + POLAND ROAD BENCHMARKS")

    lt_path = first_existing(LT_CANDIDATES, "Lithuanian road learning dataset")
    pl_path = first_existing(PL_CANDIDATES, "Polish road benchmark")

    print(f"Lithuania: {lt_path}")
    print(f"Poland:    {pl_path}")

    section("1. LOAD SOURCE DATASETS")
    lt_raw = pd.read_csv(lt_path, low_memory=False)
    pl_raw = pd.read_csv(pl_path, low_memory=False)
    print(f"Lithuania rows: {len(lt_raw):,} | columns: {len(lt_raw.columns)}")
    print(f"Poland rows:    {len(pl_raw):,} | columns: {len(pl_raw.columns)}")

    section("2. STANDARDIZE SUPERVISION AND IDENTIFIERS")
    lt = standardize(lt_raw, "Lithuania")
    pl = standardize(pl_raw, "Poland")

    print("\nAssigning Lithuania's existing audited spatial splits...")
    lt = assign_lithuania_spatial_splits(lt)

    for name, d in [("Lithuania", lt), ("Poland", pl)]:
        print(f"\n{name}")
        print("Cities:", sorted(d["city_harmonized"].dropna().unique().tolist()))
        print("Splits:", sorted(d["split_harmonized"].dropna().unique().tolist()))
        print("Labels:")
        print(d["selection_label_harmonized"].value_counts(dropna=False).to_string())
        print("Binary retained:")
        print(d["retained"].value_counts(dropna=False).sort_index().to_string())

    section("3. DISCOVER COMMON NUMERIC FEATURE SPACE")
    lt_features = numeric_feature_candidates(lt)
    pl_features = numeric_feature_candidates(pl)
    discovered_common = sorted(lt_features & pl_features)
    lt_only = sorted(lt_features - pl_features)
    pl_only = sorted(pl_features - lt_features)

    missing_model_features = [
        c for c in MODEL_FEATURES if c not in discovered_common
    ]
    if missing_model_features:
        raise RuntimeError(
            "Required cross-country model features are missing: "
            f"{missing_model_features}. Discovered common numeric columns: "
            f"{discovered_common}"
        )

    common = MODEL_FEATURES.copy()

    print(f"Lithuanian numeric candidates: {len(lt_features)}")
    print(f"Polish numeric candidates:     {len(pl_features)}")
    print(f"Discovered common numeric:     {len(discovered_common)}")
    print(f"Model-ready common features:   {len(common)}")
    print("\nMODEL-READY FEATURES")
    for c in common:
        print(f"  {c}")
    print("\nSPATIAL METADATA (NOT MODEL INPUT)")
    for c in SPATIAL_METADATA:
        if c in discovered_common:
            print(f"  {c}")

    section("4. BUILD HARMONIZED TABLES")
    meta = [
        "sample_uid", "country", "city_harmonized", "source_idx_harmonized",
        "tile_id_harmonized", "split_harmonized",
        "selection_label_harmonized", "retained", "operation_harmonized",
    ]
    spatial_meta_shared = [
        c for c in SPATIAL_METADATA if c in lt.columns and c in pl.columns
    ]
    keep = meta + spatial_meta_shared + common

    lt_h = lt[keep].copy()
    pl_h = pl[keep].copy()
    combined = pd.concat([lt_h, pl_h], ignore_index=True)

    # Coerce common features consistently.
    for c in common:
        combined[c] = pd.to_numeric(combined[c], errors="coerce")
        lt_h[c] = pd.to_numeric(lt_h[c], errors="coerce")
        pl_h[c] = pd.to_numeric(pl_h[c], errors="coerce")

    duplicate_uid = int(combined["sample_uid"].duplicated().sum())
    unknown_binary = int(combined["retained"].isna().sum())
    invalid_splits = sorted(
        set(combined["split_harmonized"].dropna().unique()) - VALID_SPLITS
    )

    print(f"Combined samples:        {len(combined):,}")
    print(f"Duplicate sample UIDs:   {duplicate_uid}")
    print(f"Unknown binary labels:   {unknown_binary}")
    print(f"Unexpected split values: {invalid_splits}")

    section("5. FEATURE COMPLETENESS")
    rows = []
    for c in common:
        for country, d in [("Lithuania", lt_h), ("Poland", pl_h)]:
            rows.append({
                "feature": c,
                "country": country,
                "rows": len(d),
                "non_null": int(d[c].notna().sum()),
                "missing": int(d[c].isna().sum()),
                "missing_pct": float(d[c].isna().mean() * 100.0),
                "n_unique": int(d[c].nunique(dropna=True)),
                "mean": float(d[c].mean()) if d[c].notna().any() else np.nan,
                "median": float(d[c].median()) if d[c].notna().any() else np.nan,
                "std": float(d[c].std()) if d[c].notna().sum() > 1 else np.nan,
            })
    completeness = pd.DataFrame(rows)
    print(completeness[["feature", "country", "missing_pct", "n_unique"]].to_string(index=False))

    section("6. COUNTRY / SPLIT / LABEL SUMMARY")
    summary = (
        combined.groupby(["country", "split_harmonized"], dropna=False)
        .agg(
            samples=("sample_uid", "size"),
            retained=("retained", lambda s: int((s == 1).sum())),
            eliminated=("retained", lambda s: int((s == 0).sum())),
        )
        .reset_index()
    )
    summary["retention_pct"] = np.where(
        summary["samples"] > 0, summary["retained"] / summary["samples"] * 100, np.nan
    )
    print(summary.to_string(index=False))

    label_summary = (
        combined.groupby(["country", "selection_label_harmonized"], dropna=False)
        .size().reset_index(name="samples")
    )

    section("7. SCHEMA REPORT")
    all_cols = sorted(set(lt.columns) | set(pl.columns))
    schema_rows = []
    for c in all_cols:
        schema_rows.append({
            "column": c,
            "in_lithuania": c in lt.columns,
            "in_poland": c in pl.columns,
            "lithuania_dtype": str(lt[c].dtype) if c in lt.columns else "",
            "poland_dtype": str(pl[c].dtype) if c in pl.columns else "",
            "common_numeric_feature": c in discovered_common,
            "model_ready_feature": c in common,
            "spatial_metadata_only": c in SPATIAL_METADATA,
        })
    schema = pd.DataFrame(schema_rows)
    print(f"Union of normalized columns: {len(schema)}")
    print(f"Shared columns: {int((schema.in_lithuania & schema.in_poland).sum())}")
    print(f"Discovered common numeric features: {len(discovered_common)}")
    print(f"Model-ready common numeric features: {len(common)}")

    section("8. GENERATING FIGURES")

    # Figure 1 — samples by country and split
    p = summary.pivot(index="split_harmonized", columns="country", values="samples").fillna(0)
    p = p.reindex([x for x in ["train", "validation", "test"] if x in p.index])
    fig, ax = plt.subplots(figsize=(8, 5))
    p.plot(kind="bar", ax=ax)
    ax.set_xlabel("Split")
    ax.set_ylabel("Road samples")
    ax.set_title("Cross-country road samples by split")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    save_fig(fig, "figure_01_samples_by_country_split")

    # Figure 2 — retention percentage
    p = summary.pivot(index="split_harmonized", columns="country", values="retention_pct").fillna(0)
    p = p.reindex([x for x in ["train", "validation", "test"] if x in p.index])
    fig, ax = plt.subplots(figsize=(8, 5))
    p.plot(kind="bar", ax=ax)
    ax.set_xlabel("Split")
    ax.set_ylabel("Retained roads (%)")
    ax.set_title("Road retention by country and split")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    save_fig(fig, "figure_02_retention_by_country_split")

    # Figure 3 — detailed label distribution
    p = label_summary.pivot(
        index="selection_label_harmonized", columns="country", values="samples"
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 5))
    p.plot(kind="bar", ax=ax)
    ax.set_xlabel("Selection label")
    ax.set_ylabel("Samples")
    ax.set_title("Selection-label distribution across countries")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    save_fig(fig, "figure_03_selection_labels")

    # Figure 4 — missingness of common features
    p = completeness.pivot(index="feature", columns="country", values="missing_pct").fillna(0)
    fig_h = max(5, min(14, 0.32 * len(p) + 2))
    fig, ax = plt.subplots(figsize=(9, fig_h))
    p.plot(kind="barh", ax=ax)
    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel("Feature")
    ax.set_title("Common-feature completeness")
    fig.tight_layout()
    save_fig(fig, "figure_04_common_feature_missingness")

    section("9. SAVE HARMONIZED BENCHMARK")
    lt_out = OUT / "lithuania_roads_harmonized.csv"
    pl_out = OUT / "poland_roads_harmonized.csv"
    combined_out = OUT / "lithuania_poland_road_benchmark.csv"
    schema_out = OUT / "cross_country_schema_comparison.csv"
    completeness_out = OUT / "common_feature_completeness.csv"
    summary_out = OUT / "country_split_summary.csv"
    labels_out = OUT / "selection_label_summary.csv"

    lt_h.to_csv(lt_out, index=False)
    pl_h.to_csv(pl_out, index=False)
    combined.to_csv(combined_out, index=False)
    schema.to_csv(schema_out, index=False)
    completeness.to_csv(completeness_out, index=False)
    summary.to_csv(summary_out, index=False)
    label_summary.to_csv(labels_out, index=False)

    checks = {
        "both_countries_present": set(combined["country"].unique()) == {"Lithuania", "Poland"},
        "common_feature_space_nonempty": len(common) > 0,
        "unique_sample_uids": duplicate_uid == 0,
        "all_binary_labels_resolved": unknown_binary == 0,
        "valid_split_labels": len(invalid_splits) == 0,
        "train_present_both": all(
            "train" in set(d["split_harmonized"]) for d in [lt_h, pl_h]
        ),
        "validation_present_both": all(
            "validation" in set(d["split_harmonized"]) for d in [lt_h, pl_h]
        ),
        "test_present_both": all(
            "test" in set(d["split_harmonized"]) for d in [lt_h, pl_h]
        ),
    }

    report = {
        "script": "64_harmonize_cross_country_road_benchmark.py",
        "inputs": {
            "lithuania": str(lt_path),
            "poland": str(pl_path),
            "lithuania_tile_index": str(LT_TILE_INDEX),
        },
        "counts": {
            "lithuania": len(lt_h),
            "poland": len(pl_h),
            "combined": len(combined),
            "common_numeric_features": len(common),
        },
        "common_numeric_features": common,
        "discovered_common_numeric_features": discovered_common,
        "spatial_metadata_not_model_input": spatial_meta_shared,
        "lithuania_only_numeric_candidates": lt_only,
        "poland_only_numeric_candidates": pl_only,
        "checks": checks,
        "ready": all(checks.values()),
    }
    report_path = OUT / "cross_country_harmonization_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, indent=2, ensure_ascii=False)

    section("10. FINAL HARMONIZATION ASSESSMENT")
    for k, v in checks.items():
        print(f"{k:45s} {'PASS' if v else 'FAIL'}")

    if all(checks.values()):
        print("\n[PASS] LITHUANIA + POLAND ROAD BENCHMARK IS HARMONIZED.")
    else:
        print("\n[CHECK] CROSS-COUNTRY HARMONIZATION REQUIRES ATTENTION.")

    banner("SCRIPT 64 COMPLETE")
    print(f"Lithuania harmonized: {lt_out}")
    print(f"Poland harmonized:    {pl_out}")
    print(f"Combined benchmark:   {combined_out}")
    print(f"Schema comparison:    {schema_out}")
    print(f"Feature completeness: {completeness_out}")
    print(f"Summary:              {summary_out}")
    print(f"Report:               {report_path}")
    print(f"Figures (PNG + PDF):  {FIG}")
    print("\nNEXT: Script 65 — define leakage-safe cross-country modelling protocols and feature preprocessing.")


if __name__ == "__main__":
    main()
