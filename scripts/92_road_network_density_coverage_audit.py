#!/usr/bin/env python3
"""Script 92: read-only road network density, extent and length proximity audit.

Run: python scripts/92_road_network_density_coverage_audit.py
Optional: --buffers 5 15 30 --output data/processed/cross_country/road_density_coverage_audit_v1
All outputs are diagnostics, NOT verified target coverage or retention labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("Lithuania", "vilnius_central", "lithuania/validated_pairs/vilnius_central_validated.gpkg"),
    ("Lithuania", "kaunas_central", "lithuania/validated_pairs/kaunas_central_validated.gpkg"),
    ("Lithuania", "klaipeda_central", "lithuania/validated_pairs/klaipeda_central_validated.gpkg"),
    ("Poland", "warsaw", "poland/road_target_validation/warsaw_validated_road_pair.gpkg"),
    ("Poland", "krakow", "poland/road_target_validation/krakow_validated_road_pair.gpkg"),
    ("Poland", "gdansk", "poland/road_target_validation/gdansk_validated_road_pair.gpkg"),
]


def clean_lines(gdf: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    """Preserve original row index and multipart source identities."""
    out = gdf.copy()
    out["original_row_index"] = np.arange(len(out), dtype=np.int64)
    valid = out.geometry.notna() & ~out.geometry.is_empty
    out = out.loc[valid].copy()
    valid_types = out.geom_type.isin(["LineString", "MultiLineString"])
    if not valid_types.all():
        print(f"  WARNING {label}: excluding {(~valid_types).sum()} non-line geometries")
    out = out.loc[valid_types].copy()
    if (~out.geometry.is_valid).any():
        raise ValueError(f"{label} contains invalid line geometries; audit aborted to avoid silent repairs")
    out = out.loc[out.geometry.length > 0].copy()
    return out.reset_index(drop=True)


def load_pair(path: Path):
    s = gpd.read_file(path, layer="source_roads")
    t = gpd.read_file(path, layer="target_roads")
    if s.crs is None or t.crs is None:
        raise ValueError(f"Missing CRS: {path}")
    if s.crs != t.crs:
        t = t.to_crs(s.crs)
    if not s.crs.is_projected or not any(x in s.crs.axis_info[0].unit_name.lower() for x in ("metre", "meter")):
        raise ValueError(f"Expected projected metric CRS, got {s.crs}")
    return clean_lines(s, "source"), clean_lines(t, "target")


def lengths_stats(gdf):
    a = gdf.geometry.length.to_numpy(dtype=float)
    return {"count": len(a), "total_m": float(a.sum()), "mean_m": float(a.mean()),
            "median_m": float(np.median(a)), "p90_m": float(np.percentile(a, 90))}


def length_in_region(gdf, polygon):
    # Polygon is an extent PROXY, not a certified study area.
    return np.asarray([float(g.intersection(polygon).length) for g in gdf.geometry], dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/cross_country/road_density_coverage_audit_v1")
    parser.add_argument("--buffers", nargs="+", type=float, default=[5.0, 15.0, 30.0])
    args = parser.parse_args()
    buffers = sorted(set(args.buffers))
    if not buffers or any(not np.isfinite(b) or b <= 0 for b in buffers):
        raise ValueError("All proximity buffers must be finite positive metre values")
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "configuration.json").write_text(json.dumps({"buffers_m": buffers,
        "warning": "Bounding rectangles and target-road buffers are diagnostic proxies, not verified geographic coverage"}, indent=2), encoding="utf-8")
    summaries, source_rows = [], []
    for country, city, rel in CASES:
        path = ROOT / "data/processed" / rel
        print(f"\n{'='*90}\n{country} / {city}\n{path}", flush=True)
        if not path.exists():
            raise FileNotFoundError(path)
        s, t = load_pair(path)
        if s.empty or t.empty:
            raise ValueError(f"Empty source or target: {path}")
        sl, tl = lengths_stats(s), lengths_stats(t)
        source_box = box(*s.total_bounds)
        target_box = box(*t.total_bounds)
        overlap_box = source_box.intersection(target_box)
        s_lengths = s.geometry.length.to_numpy(dtype=float)
        t_lengths = t.geometry.length.to_numpy(dtype=float)
        s_inside_target_box = length_in_region(s, target_box)
        t_inside_source_box = length_in_region(t, source_box)
        s_inside_overlap = length_in_region(s, overlap_box)
        t_inside_overlap = length_in_region(t, overlap_box)
        # A buffered union of the target network is only a geometric proximity measure.
        target_union = unary_union(list(t.geometry))
        row_data = {"country": country, "city": city,
                    "source_index": s.original_row_index.to_numpy(dtype=int),
                    "source_length_m": s_lengths,
                    "length_inside_target_bbox_m": s_inside_target_box,
                    "fraction_inside_target_bbox": np.divide(s_inside_target_box, s_lengths, out=np.zeros_like(s_lengths), where=s_lengths>0)}
        row_summary = {"country": country, "city": city, "crs": str(s.crs),
            "source_count": sl["count"], "target_count": tl["count"],
            "source_total_length_km": sl["total_m"] / 1000,
            "target_total_length_km": tl["total_m"] / 1000,
            "target_to_source_length_ratio": tl["total_m"] / sl["total_m"],
            "source_median_feature_length_m": sl["median_m"],
            "target_median_feature_length_m": tl["median_m"],
            "source_p90_feature_length_m": sl["p90_m"],
            "target_p90_feature_length_m": tl["p90_m"],
            "source_bbox_area_km2": source_box.area / 1e6,
            "target_bbox_area_km2": target_box.area / 1e6,
            "bbox_intersection_area_km2": overlap_box.area / 1e6,
            "source_length_inside_target_bbox_pct": 100 * s_inside_target_box.sum() / sl["total_m"],
            "target_length_inside_source_bbox_pct": 100 * t_inside_source_box.sum() / tl["total_m"],
            "source_length_inside_shared_bbox_km": s_inside_overlap.sum() / 1000,
            "target_length_inside_shared_bbox_km": t_inside_overlap.sum() / 1000,
            "source_density_km_per_source_bbox_km2": (sl["total_m"] / 1000) / (source_box.area / 1e6),
            "target_density_km_per_target_bbox_km2": (tl["total_m"] / 1000) / (target_box.area / 1e6),
            "source_density_in_shared_bbox_km_per_km2": (s_inside_overlap.sum() / 1000) / (overlap_box.area / 1e6) if overlap_box.area else np.nan,
            "target_density_in_shared_bbox_km_per_km2": (t_inside_overlap.sum() / 1000) / (overlap_box.area / 1e6) if overlap_box.area else np.nan,
            "verified_target_coverage": False,
        }
        for b in buffers:
            print(f"  Computing source length within {b:g} m of target network...", flush=True)
            corridor = target_union.buffer(b)
            covered = np.asarray([float(g.intersection(corridor).length) for g in s.geometry], dtype=float)
            covered = np.minimum(covered, s_lengths)
            frac = np.divide(covered, s_lengths, out=np.zeros_like(covered), where=s_lengths>0)
            key = f"near_target_{b:g}m".replace(".", "p")
            row_data[f"{key}_length_m"] = covered
            row_data[f"{key}_fraction"] = frac
            row_summary[f"{key}_source_length_pct"] = 100 * covered.sum() / sl["total_m"]
            row_summary[f"{key}_source_features_ge50pct_pct"] = 100 * np.mean(frac >= 0.5)
            row_summary[f"{key}_source_features_any_pct"] = 100 * np.mean(frac > 0)
            # Same proximity metric restricted to shared bounding rectangle.
            restricted = np.asarray([float(g.intersection(corridor).intersection(overlap_box).length)
                                     for g in s.geometry], dtype=float)
            row_summary[f"{key}_source_length_in_shared_bbox_pct"] = (
                100 * restricted.sum() / s_inside_overlap.sum() if s_inside_overlap.sum() > 0 else np.nan)
        pd.DataFrame(row_data).to_csv(out / f"{country.lower()}_{city}_source_length_audit.csv", index=False)
        summaries.append(row_summary)
        source_rows.append((country, city, sl["total_m"], tl["total_m"]))
        print(f"  Source={sl['total_m']/1000:.1f} km; target={tl['total_m']/1000:.1f} km; "
              f"target/source={tl['total_m']/sl['total_m']:.3f}; "
              f"source length inside target bbox={row_summary['source_length_inside_target_bbox_pct']:.1f}%", flush=True)
    df = pd.DataFrame(summaries)
    prev = ROOT / "data/processed/cross_country/unified_correspondence_audit_v1/city_configuration_summary.csv"
    if prev.exists():
        old = pd.read_csv(prev)
        old = old.loc[old.configuration == "unified_primary", ["country", "city", "matched_source_rate"]]
        df = df.merge(old.rename(columns={"matched_source_rate": "script91_primary_matched_source_rate"}),
                      on=["country", "city"], how="left", validate="one_to_one")
    df.to_csv(out / "city_network_density_coverage_summary.csv", index=False)
    print("\n" + df[["country", "city", "source_total_length_km", "target_total_length_km",
        "target_to_source_length_ratio", "source_length_inside_target_bbox_pct"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nRead-only outputs: {out}")
    print("WARNING: Bounding-box overlap is NOT verified geographic coverage.")
    print("WARNING: Proximity to target roads is NOT a confirmed retention/elimination label.")
    print("WARNING: Network length ratios are NOT road retention rates; overlapping linework may be double-counted.")


if __name__ == "__main__":
    main()
