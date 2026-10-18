#!/usr/bin/env python3
"""Script 91: read-only cross-country road-correspondence sensitivity audit.

Requires: geopandas, shapely, pandas, numpy, pyproj.
Run: python scripts/91_unified_road_correspondence_audit.py
Does not modify original GeoPackages, benchmark labels or splits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString
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
# Predeclared sensitivity settings, not tuned to target labels or test MCC.
CONFIGS = {
    "unified_primary": dict(search_m=15., buffer_m=5., min_source_cov=.20,
                            min_target_cov=.03, min_score=.20),
    "strict": dict(search_m=10., buffer_m=4., min_source_cov=.30,
                   min_target_cov=.05, min_score=.25),
    "permissive": dict(search_m=20., buffer_m=8., min_source_cov=.10,
                       min_target_cov=.03, min_score=.15),
}


def line_parts(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom] if geom.length > 0 else []
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if g.length > 0]
    return []


def endpoint_angle(geom):
    parts = line_parts(geom)
    if not parts:
        return float("nan")
    g = max(parts, key=lambda x: x.length)
    xy = list(g.coords)
    if len(xy) < 2:
        return float("nan")
    dx, dy = xy[-1][0] - xy[0][0], xy[-1][1] - xy[0][1]
    if abs(dx) + abs(dy) < 1e-10:
        return float("nan")
    return float(np.degrees(np.arctan2(dy, dx)) % 180.)


def coverage(line, other, buffer_m):
    if line.length <= 0:
        return 0.
    try:
        return float(np.clip(line.intersection(other.buffer(buffer_m)).length / line.length, 0., 1.))
    except Exception:
        return 0.


def load_pair(path):
    source = gpd.read_file(path, layer="source_roads")
    target = gpd.read_file(path, layer="target_roads")
    if source.crs is None or target.crs is None:
        raise ValueError(f"Missing CRS in {path}")
    if source.crs != target.crs:
        target = target.to_crs(source.crs)
    if not source.crs.is_projected:
        raise ValueError(f"Geographic CRS, cannot use metre thresholds: {source.crs}")
    unit = source.crs.axis_info[0].unit_name.lower()
    if "metre" not in unit and "meter" not in unit:
        raise ValueError(f"CRS units not metres: {source.crs}: {unit}")
    for name, gdf in (("source", source), ("target", target)):
        if not gdf.geom_type.isin(["LineString", "MultiLineString"]).all():
            raise ValueError(f"Unexpected {name} geometries in {path}")
        if (~gdf.geometry.is_valid).any():
            print(f"  WARNING: {name} contains {(~gdf.geometry.is_valid).sum()} invalid geometries")
    source = source.reset_index(drop=True)
    target = target.reset_index(drop=True)
    source["audit_source_index"] = np.arange(len(source), dtype=int)
    target["audit_target_index"] = np.arange(len(target), dtype=int)
    return source, target


def matching(source, target, cfg, country, city, config_name):
    # Spatial index returns target positional indices. Candidate geometry operations
    # occur in a metric CRS. Source identities are NEVER split or renumbered.
    sindex = target.sindex
    target_geoms = target.geometry.tolist()
    target_angles = [endpoint_angle(g) for g in target_geoms]
    pairs = []
    source_records = []
    for i, src in enumerate(source.geometry):
        if src is None or src.is_empty or src.length <= 0:
            source_records.append((i, 0, 0, 0))
            continue
        candidates = sindex.query(src.buffer(cfg["search_m"]), predicate="intersects")
        accepted = []
        n_candidates = 0
        src_angle = endpoint_angle(src)
        for j in candidates:
            j = int(j)
            tgt = target_geoms[j]
            if tgt is None or tgt.is_empty or tgt.length <= 0:
                continue
            distance = float(src.distance(tgt))
            if distance > cfg["search_m"]:
                continue
            n_candidates += 1
            sc = coverage(src, tgt, cfg["buffer_m"])
            tc = coverage(tgt, src, cfg["buffer_m"])
            ta = target_angles[j]
            angle_score = (1. - min(abs(src_angle - ta), 180. - abs(src_angle - ta)) / 90.) if np.isfinite(src_angle) and np.isfinite(ta) else 0.
            distance_score = max(0., 1. - distance / cfg["search_m"])
            score = .50 * sc + .20 * tc + .20 * distance_score + .10 * angle_score
            if sc >= cfg["min_source_cov"] and tc >= cfg["min_target_cov"] and score >= cfg["min_score"]:
                accepted.append((j, score, sc, tc, distance, angle_score))
        # One best accepted target per source; many sources can share a target.
        if accepted:
            j, score, sc, tc, distance, angle_score = sorted(accepted, key=lambda z: (-z[1], -z[2], -z[3], z[0]))[0]
            pairs.append(dict(country=country, city=city, configuration=config_name,
                              source_index=i, target_index=j, match_score=score,
                              source_coverage=sc, target_coverage=tc,
                              distance_m=distance, orientation_score=angle_score,
                              accepted_candidates=len(accepted)))
        source_records.append((i, n_candidates, len(accepted), int(bool(accepted))))
    pairs_df = pd.DataFrame(pairs, columns=["country", "city", "configuration", "source_index", "target_index",
                                           "match_score", "source_coverage", "target_coverage", "distance_m",
                                           "orientation_score", "accepted_candidates"])
    labels = pd.DataFrame(source_records, columns=["source_index", "candidate_count", "accepted_candidate_count", "matched"])
    labels.insert(0, "configuration", config_name)
    labels.insert(0, "city", city)
    labels.insert(0, "country", country)
    return pairs_df, labels


def coverage_proxy(source, target, buffer_m=15.):
    """Heuristic spatial support ONLY, never ground-truth coverage or elimination."""
    # Target-road buffer can miss genuinely eliminated streets and must NOT
    # be used as a hard eligibility filter. It is a proximity diagnostic.
    union = unary_union([g for g in target.geometry if g is not None and not g.is_empty])
    proximity = union.buffer(buffer_m)
    lengths = []
    for g in source.geometry:
        if g is None or g.is_empty or g.length <= 0:
            lengths.append(0.)
        else:
            lengths.append(float(np.clip(g.intersection(proximity).length / g.length, 0., 1.)))
    return np.asarray(lengths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/cross_country/unified_correspondence_audit_v1")
    parser.add_argument("--config", choices=["all", *CONFIGS], default="all")
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    configs = CONFIGS if args.config == "all" else {args.config: CONFIGS[args.config]}
    (out / "audit_configuration.json").write_text(json.dumps(configs, indent=2), encoding="utf-8")
    summaries = []
    for country, city, relative in CASES:
        path = ROOT / "data/processed" / relative
        print(f"\n{'=' * 90}\n{country} / {city}\n{path}", flush=True)
        if not path.exists():
            raise FileNotFoundError(path)
        source, target = load_pair(path)
        print(f"CRS={source.crs}, source={len(source)}, target={len(target)}", flush=True)
        # This is explicitly NOT a valid geographic coverage polygon.
        proximity_fraction = coverage_proxy(source, target)
        for config_name, cfg in configs.items():
            print(f"  Running {config_name}...", flush=True)
            pairs, labels = matching(source, target, cfg, country, city, config_name)
            labels["near_target_road_fraction_15m"] = proximity_fraction
            labels["coverage_status"] = "UNVERIFIED_TARGET_COVERAGE"
            prefix = f"{country.lower()}_{city}_{config_name}"
            pairs.to_csv(out / f"{prefix}_best_pairs.csv", index=False)
            labels.to_csv(out / f"{prefix}_source_audit.csv", index=False)
            matched = int(labels.matched.sum())
            summaries.append(dict(country=country, city=city, configuration=config_name,
                                  source_count=len(source), target_count=len(target),
                                  matched_source_count=matched,
                                  unmatched_source_count=len(source) - matched,
                                  matched_source_rate=matched / len(source) if len(source) else np.nan,
                                  matched_target_count=int(pairs.target_index.nunique()),
                                  total_candidate_count=int(labels.candidate_count.sum()),
                                  total_accepted_candidate_count=int(labels.accepted_candidate_count.sum()),
                                  near_target_road_fraction_mean=float(proximity_fraction.mean()),
                                  median_match_distance_m=float(pairs.distance_m.median()) if len(pairs) else np.nan,
                                  median_source_coverage=float(pairs.source_coverage.median()) if len(pairs) else np.nan,
                                  median_target_coverage=float(pairs.target_coverage.median()) if len(pairs) else np.nan,
                                  median_match_score=float(pairs.match_score.median()) if len(pairs) else np.nan,
                                  input_path=str(path.relative_to(ROOT)),
                                  crs=str(source.crs),
                                  target_coverage_verified=False))
            print(f"    Matched {matched:,}/{len(source):,} ({100 * matched / max(1, len(source)):.2f}%)", flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "city_configuration_summary.csv", index=False)
    print("\n" + summary[["country", "city", "configuration", "matched_source_rate"]].to_string(index=False))
    print(f"\nRead-only audit outputs: {out}")
    print("WARNING: Unmatched does NOT mean confirmed elimination; target coverage is unverified.")
    print("WARNING: These rates are NOT directly comparable to benchmark labels until source ID alignment is verified.")


if __name__ == "__main__":
    main()
