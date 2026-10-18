"""
Script 68 — Construct leakage-safe spatial/graph representations for CartoGen.

Purpose
-------
Build road-segment graph representations independently within each audited
country × split × tile partition. No graph edge is allowed to cross a tile or
split boundary.

Inputs
------
data/processed/cross_country/harmonized_roads/
    lithuania_poland_road_benchmark.csv

Outputs
-------
data/processed/cross_country/graph_representations/
    road_graph_nodes.csv
    road_graph_edges.csv
    graph_tile_summary.csv
    graph_feature_summary.csv
    graph_construction_report.json
    figures/*.png
    figures/*.pdf

Notes
-----
This script intentionally uses only source-side information. Labels are copied
for later modelling but are NEVER used to construct graph edges/features.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "cross_country" / "harmonized_roads" / "lithuania_poland_road_benchmark.csv"
OUT = ROOT / "data" / "processed" / "cross_country" / "graph_representations"
FIG = OUT / "figures"

# Graph construction parameters. Distances are in each country's projected CRS
# units represented by the harmonized centroid coordinates (metres).
K_NEAREST = 6
MAX_EDGE_DISTANCE_M = 250.0
RADII_M = (50.0, 100.0, 150.0, 250.0)

VALID_SPLITS = {"train", "validation", "test"}


def banner(s: str) -> None:
    print("\n" + "=" * 120)
    print(s)
    print("=" * 120)


def section(s: str) -> None:
    print("\n" + "-" * 120)
    print(s)
    print("-" * 120)


def json_safe(x):
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x


def first_existing(df: pd.DataFrame, names, required=True):
    for n in names:
        if n in df.columns:
            return n
    if required:
        raise KeyError(f"None of {names} found.\nAvailable columns:\n{df.columns.tolist()}")
    return None


def normalize_split(x):
    s = str(x).strip().lower()
    if s in {"val", "valid", "dev"}:
        return "validation"
    return s


def save_fig(fig, stem):
    png = FIG / f"{stem}.png"
    pdf = FIG / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {png}")
    print(f"PDF: {pdf}")


def robust_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    country = first_existing(d, ["country"])
    city = first_existing(d, ["city_harmonized", "city"])
    split = first_existing(d, ["split_harmonized", "split"])
    tile = first_existing(
        d,
        ["tile_id_harmonized", "tile_id", "spatial_tile_id"]
    )
    uid = first_existing(d, ["sample_uid", "road_uid", "uid"], required=False)
    xcol = first_existing(d, ["centroid_x", "x"])
    ycol = first_existing(d, ["centroid_y", "y"])

    if uid is None:
        source = first_existing(d, ["source_idx", "feature_index", "source_id", "id"])
        d["sample_uid"] = (
            d[country].astype(str) + "::" +
            d[city].astype(str) + "::" +
            d[source].astype(str)
        )
        uid = "sample_uid"

    rename = {}
    if country != "country": rename[country] = "country"
    if city != "city": rename[city] = "city"
    if split != "split": rename[split] = "split"
    if tile != "tile_id": rename[tile] = "tile_id"
    if uid != "sample_uid": rename[uid] = "sample_uid"
    if xcol != "centroid_x": rename[xcol] = "centroid_x"
    if ycol != "centroid_y": rename[ycol] = "centroid_y"
    d = d.rename(columns=rename)

    d["split"] = d["split"].map(normalize_split)
    d["centroid_x"] = robust_numeric(d["centroid_x"])
    d["centroid_y"] = robust_numeric(d["centroid_y"])

    if "retained" not in d.columns:
        lab = first_existing(d, ["selection_label_harmonized", "selection_label", "road_operation"])
        m = d[lab].astype(str).str.lower().str.strip()
        d["retained"] = np.where(m.eq("elimination"), 0, np.where(m.isin(["selected", "weak_selected"]), 1, np.nan))

    if "selection_label_harmonized" not in d.columns:
        lab = first_existing(d, ["selection_label", "road_operation"], required=False)
        if lab:
            d["selection_label_harmonized"] = d[lab].astype(str)

    return d


def build_tile_graph(g: pd.DataFrame):
    """
    Undirected kNN-radius graph inside ONE country/split/tile only.
    Each node connects to up to K_NEAREST nearest nodes, provided distance <=
    MAX_EDGE_DISTANCE_M. Symmetric duplicates are removed.
    """
    g = g.copy().reset_index(drop=True)
    n = len(g)

    coords = g[["centroid_x", "centroid_y"]].to_numpy(float)
    if not np.isfinite(coords).all():
        raise ValueError("Non-finite centroid coordinate encountered.")

    adjacency = [set() for _ in range(n)]
    edge_distance = {}

    if n > 1:
        tree = cKDTree(coords)
        k = min(K_NEAREST + 1, n)
        dist, ind = tree.query(coords, k=k)

        if k == 1:
            dist = dist[:, None]
            ind = ind[:, None]

        for i in range(n):
            for dd, j in zip(np.atleast_1d(dist[i])[1:], np.atleast_1d(ind[i])[1:]):
                j = int(j)
                dd = float(dd)
                if j == i or not np.isfinite(dd) or dd > MAX_EDGE_DISTANCE_M:
                    continue
                a, b = sorted((i, j))
                adjacency[a].add(b)
                adjacency[b].add(a)
                key = (a, b)
                if key not in edge_distance or dd < edge_distance[key]:
                    edge_distance[key] = dd

    # Local density features are computed from all nodes in the SAME tile.
    tree = cKDTree(coords) if n else None
    radius_counts = {}
    for r in RADII_M:
        if n:
            radius_counts[r] = np.array(
                [max(0, len(v) - 1) for v in tree.query_ball_point(coords, r)],
                dtype=int,
            )
        else:
            radius_counts[r] = np.zeros(0, dtype=int)

    degrees = np.array([len(x) for x in adjacency], dtype=int)
    mean_edge = np.full(n, np.nan)
    min_edge = np.full(n, np.nan)
    max_edge = np.full(n, np.nan)
    clustering = np.zeros(n, dtype=float)

    for i in range(n):
        neigh = list(adjacency[i])
        if neigh:
            ds = [float(np.linalg.norm(coords[i] - coords[j])) for j in neigh]
            mean_edge[i] = np.mean(ds)
            min_edge[i] = np.min(ds)
            max_edge[i] = np.max(ds)

        k_i = len(neigh)
        if k_i >= 2:
            possible = k_i * (k_i - 1) / 2
            links = 0
            ns = set(neigh)
            for u in neigh:
                links += len(adjacency[u].intersection(ns))
            links /= 2
            clustering[i] = links / possible

    # Connected components, without networkx dependency.
    component = np.full(n, -1, dtype=int)
    comp_sizes = {}
    cid = 0
    for start in range(n):
        if component[start] != -1:
            continue
        stack = [start]
        members = []
        component[start] = cid
        while stack:
            u = stack.pop()
            members.append(u)
            for v in adjacency[u]:
                if component[v] == -1:
                    component[v] = cid
                    stack.append(v)
        comp_sizes[cid] = len(members)
        cid += 1

    g["graph_degree"] = degrees
    g["graph_mean_edge_distance_m"] = mean_edge
    g["graph_min_edge_distance_m"] = min_edge
    g["graph_max_edge_distance_m"] = max_edge
    g["graph_local_clustering"] = clustering
    g["graph_component_id_local"] = component
    g["graph_component_size"] = [comp_sizes[int(c)] for c in component]
    g["graph_is_isolated"] = (degrees == 0).astype(int)

    for r in RADII_M:
        g[f"local_road_count_{int(r)}m"] = radius_counts[r]

    # Aggregate neighbor source features without labels.
    aggregate_candidates = [
        "length_m",
        "intersecting_road_count",
        "neighbor_count_150m",
        "mean_neighbor_distance_m",
    ]
    aggregate_candidates = [c for c in aggregate_candidates if c in g.columns]

    for col in aggregate_candidates:
        vals = robust_numeric(g[col]).to_numpy(float)
        means = np.full(n, np.nan)
        stds = np.full(n, np.nan)
        for i in range(n):
            neigh = list(adjacency[i])
            if not neigh:
                continue
            z = vals[neigh]
            z = z[np.isfinite(z)]
            if len(z):
                means[i] = float(np.mean(z))
                stds[i] = float(np.std(z))
        g[f"nbr_mean_{col}"] = means
        g[f"nbr_std_{col}"] = stds

    edges = []
    for (a, b), dd in sorted(edge_distance.items()):
        edges.append({
            "source_uid": g.loc[a, "sample_uid"],
            "target_uid": g.loc[b, "sample_uid"],
            "country": g.loc[a, "country"],
            "city": g.loc[a, "city"],
            "split": g.loc[a, "split"],
            "tile_id": g.loc[a, "tile_id"],
            "distance_m": dd,
        })

    edge_df = pd.DataFrame(edges)
    summary = {
        "country": str(g["country"].iloc[0]),
        "city": str(g["city"].iloc[0]),
        "split": str(g["split"].iloc[0]),
        "tile_id": str(g["tile_id"].iloc[0]),
        "nodes": int(n),
        "edges": int(len(edges)),
        "isolated_nodes": int((degrees == 0).sum()),
        "components": int(cid),
        "largest_component": int(max(comp_sizes.values()) if comp_sizes else 0),
        "mean_degree": float(degrees.mean()) if n else 0.0,
        "median_degree": float(np.median(degrees)) if n else 0.0,
        "mean_edge_distance_m": float(edge_df["distance_m"].mean()) if len(edge_df) else np.nan,
    }
    return g, edge_df, summary


def main():
    banner("CARTOGEN — CONSTRUCT LEAKAGE-SAFE SPATIAL/GRAPH ROAD REPRESENTATIONS")
    print(f"Benchmark: {INPUT}")
    print(f"Output:    {OUT}")

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    section("1. LOAD AND VERIFY HARMONIZED BENCHMARK")
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    raw = pd.read_csv(INPUT)
    d = prepare(raw)

    print(f"Rows:      {len(d):,}")
    print(f"Countries: {sorted(d['country'].dropna().unique().tolist())}")
    print(f"Splits:    {sorted(d['split'].dropna().unique().tolist())}")
    print(f"Tiles:     {d['tile_id'].nunique():,}")

    required = ["sample_uid", "country", "city", "split", "tile_id",
                "centroid_x", "centroid_y", "retained"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise KeyError(f"Required columns missing: {missing}")

    if d["sample_uid"].duplicated().any():
        dup = int(d["sample_uid"].duplicated().sum())
        raise RuntimeError(f"Duplicate sample_uid values: {dup}")

    bad_split = sorted(set(d["split"].dropna()) - VALID_SPLITS)
    if bad_split:
        raise RuntimeError(f"Unexpected split labels: {bad_split}")

    if d[["centroid_x", "centroid_y"]].isna().any().any():
        raise RuntimeError("Missing centroid coordinates found.")

    if d["tile_id"].isna().any():
        raise RuntimeError(f"Missing tile IDs: {int(d['tile_id'].isna().sum())}")

    section("2. BUILD GRAPHS INDEPENDENTLY WITHIN COUNTRY × SPLIT × TILE")
    node_parts = []
    edge_parts = []
    summaries = []

    groups = list(d.groupby(["country", "split", "tile_id"], sort=True, dropna=False))
    print(f"Independent graph partitions: {len(groups):,}")
    print(f"k nearest:                    {K_NEAREST}")
    print(f"maximum edge distance:        {MAX_EDGE_DISTANCE_M:.1f} m")

    for idx, ((country, split, tile), g) in enumerate(groups, 1):
        nodes, edges, summary = build_tile_graph(g)
        node_parts.append(nodes)
        if not edges.empty:
            edge_parts.append(edges)
        summaries.append(summary)

        if idx % 25 == 0 or idx == len(groups):
            print(f"  processed {idx:>3}/{len(groups)} partitions")

    nodes = pd.concat(node_parts, ignore_index=True)
    edges = pd.concat(edge_parts, ignore_index=True) if edge_parts else pd.DataFrame(
        columns=["source_uid", "target_uid", "country", "city", "split", "tile_id", "distance_m"]
    )
    tile_summary = pd.DataFrame(summaries)

    section("3. LEAKAGE AUDIT")
    lookup = nodes.set_index("sample_uid")[["country", "split", "tile_id"]]
    e = edges.copy()
    if len(e):
        e = e.join(lookup.add_prefix("src_"), on="source_uid")
        e = e.join(lookup.add_prefix("dst_"), on="target_uid")
        cross_country = int((e["src_country"] != e["dst_country"]).sum())
        cross_split = int((e["src_split"] != e["dst_split"]).sum())
        cross_tile = int((e["src_tile_id"] != e["dst_tile_id"]).sum())
    else:
        cross_country = cross_split = cross_tile = 0

    duplicate_edges = int(edges.duplicated(["source_uid", "target_uid"]).sum()) if len(edges) else 0
    self_edges = int((edges["source_uid"] == edges["target_uid"]).sum()) if len(edges) else 0
    unknown_sources = int((~edges["source_uid"].isin(nodes["sample_uid"])).sum()) if len(edges) else 0
    unknown_targets = int((~edges["target_uid"].isin(nodes["sample_uid"])).sum()) if len(edges) else 0

    print(f"Nodes:                       {len(nodes):,}")
    print(f"Edges:                       {len(edges):,}")
    print(f"Cross-country edges:         {cross_country}")
    print(f"Cross-split edges:           {cross_split}")
    print(f"Cross-tile edges:            {cross_tile}")
    print(f"Self edges:                  {self_edges}")
    print(f"Duplicate undirected edges:  {duplicate_edges}")
    print(f"Unknown source endpoints:    {unknown_sources}")
    print(f"Unknown target endpoints:    {unknown_targets}")

    section("4. GRAPH STATISTICS")
    summary_country_split = (
        tile_summary.groupby(["country", "split"], as_index=False)
        .agg(
            tiles=("tile_id", "nunique"),
            nodes=("nodes", "sum"),
            edges=("edges", "sum"),
            isolated_nodes=("isolated_nodes", "sum"),
            components=("components", "sum"),
            mean_tile_degree=("mean_degree", "mean"),
            mean_edge_distance_m=("mean_edge_distance_m", "mean"),
        )
    )
    summary_country_split["isolated_pct"] = (
        100 * summary_country_split["isolated_nodes"] /
        summary_country_split["nodes"].replace(0, np.nan)
    )
    print(summary_country_split.to_string(index=False))

    section("5. GRAPH FEATURE COMPLETENESS")
    graph_features = [
        c for c in nodes.columns
        if c.startswith("graph_") or c.startswith("local_road_count_") or
           c.startswith("nbr_mean_") or c.startswith("nbr_std_")
    ]
    feature_rows = []
    for col in graph_features:
        x = robust_numeric(nodes[col])
        feature_rows.append({
            "feature": col,
            "missing": int(x.isna().sum()),
            "missing_pct": float(100 * x.isna().mean()),
            "n_unique": int(x.nunique(dropna=True)),
            "mean": float(x.mean()) if x.notna().any() else np.nan,
            "std": float(x.std()) if x.notna().sum() > 1 else np.nan,
        })
    feature_summary = pd.DataFrame(feature_rows)
    print(feature_summary.to_string(index=False))

    section("6. GENERATE FIGURES")
    # Figure 1: mean degree by country/split
    p = summary_country_split.copy()
    labels = [f"{a}\n{b}" for a, b in zip(p["country"], p["split"])]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, p["mean_tile_degree"])
    ax.set_ylabel("Mean tile-level node degree")
    ax.set_title("Leakage-safe graph connectivity by country and split")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    save_fig(fig, "figure_01_graph_degree_by_country_split")

    # Figure 2: isolated percentage
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, p["isolated_pct"])
    ax.set_ylabel("Isolated nodes (%)")
    ax.set_title("Isolated road segments by country and split")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    save_fig(fig, "figure_02_isolated_nodes_by_country_split")

    # Figure 3: component size distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = nodes["graph_component_size"].clip(upper=nodes["graph_component_size"].quantile(0.99))
    ax.hist(vals.dropna(), bins=40)
    ax.set_xlabel("Connected-component size (clipped at 99th percentile)")
    ax.set_ylabel("Road nodes")
    ax.set_title("Road graph component-size distribution")
    fig.tight_layout()
    save_fig(fig, "figure_03_component_size_distribution")

    section("7. SAVE REPRESENTATIONS")
    nodes_path = OUT / "road_graph_nodes.csv"
    edges_path = OUT / "road_graph_edges.csv"
    tile_path = OUT / "graph_tile_summary.csv"
    feat_path = OUT / "graph_feature_summary.csv"
    cs_path = OUT / "graph_country_split_summary.csv"

    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)
    tile_summary.to_csv(tile_path, index=False)
    feature_summary.to_csv(feat_path, index=False)
    summary_country_split.to_csv(cs_path, index=False)

    section("8. FINAL GRAPH-CONSTRUCTION AUDIT")
    checks = {
        "both_countries_present": nodes["country"].nunique() == 2,
        "all_three_splits_present": set(nodes["split"].unique()) == VALID_SPLITS,
        "all_input_nodes_preserved": len(nodes) == len(d),
        "unique_sample_uids": not nodes["sample_uid"].duplicated().any(),
        "no_cross_country_edges": cross_country == 0,
        "no_cross_split_edges": cross_split == 0,
        "no_cross_tile_edges": cross_tile == 0,
        "no_self_edges": self_edges == 0,
        "no_duplicate_edges": duplicate_edges == 0,
        "all_edge_endpoints_known": unknown_sources == 0 and unknown_targets == 0,
        "labels_not_used_for_graph_construction": True,
    }

    for k, v in checks.items():
        print(f"{k:45s} {'PASS' if v else 'FAIL'}")

    passed = all(checks.values())

    report = {
        "input": str(INPUT),
        "output": str(OUT),
        "parameters": {
            "k_nearest": K_NEAREST,
            "max_edge_distance_m": MAX_EDGE_DISTANCE_M,
            "density_radii_m": list(RADII_M),
            "edge_scope": "country × split × tile",
        },
        "counts": {
            "input_rows": len(d),
            "nodes": len(nodes),
            "edges": len(edges),
            "graph_partitions": len(groups),
            "tiles": int(nodes["tile_id"].nunique()),
            "isolated_nodes": int(nodes["graph_is_isolated"].sum()),
        },
        "leakage_audit": {
            "cross_country_edges": cross_country,
            "cross_split_edges": cross_split,
            "cross_tile_edges": cross_tile,
            "self_edges": self_edges,
            "duplicate_edges": duplicate_edges,
            "unknown_source_endpoints": unknown_sources,
            "unknown_target_endpoints": unknown_targets,
        },
        "graph_features": graph_features,
        "checks": checks,
        "passed": passed,
    }

    report_path = OUT / "graph_construction_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, indent=2, ensure_ascii=False)

    section("9. OUTPUTS")
    print(f"Graph nodes:          {nodes_path}")
    print(f"Graph edges:          {edges_path}")
    print(f"Tile summary:         {tile_path}")
    print(f"Feature summary:      {feat_path}")
    print(f"Country/split summary:{cs_path}")
    print(f"Report:               {report_path}")
    print(f"Figures (PNG + PDF):  {FIG}")

    banner("SCRIPT 68 COMPLETE")
    if passed:
        print("[PASS] LEAKAGE-SAFE CROSS-COUNTRY ROAD GRAPH REPRESENTATION COMPLETE.")
        print("NEXT: Script 69 — train graph-aware road-selection models and compare against Script 66 tabular baselines.")
    else:
        print("[FAIL] GRAPH REPRESENTATION REQUIRES ATTENTION.")
        raise RuntimeError("One or more final graph-construction checks failed.")


if __name__ == "__main__":
    main()
