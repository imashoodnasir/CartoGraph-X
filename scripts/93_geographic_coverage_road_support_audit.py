#!/usr/bin/env python3
"""CartoGraph-X Script 93: read-only source-road geometric support audit.

Usage:
    python scripts/93_geographic_coverage_road_support_audit.py
    python scripts/93_geographic_coverage_road_support_audit.py --cities klaipeda_central

This script NEVER interprets unmatched roads as confirmed eliminations. Target
coverage cannot be verified from target geometry bounds. The bbox flag is a
screening diagnostic, not a valid-coverage label.
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
OUTPUT = ROOT / 'data' / 'processed' / 'cross_country' / 'geographic_coverage_road_support_audit_v1'
CASES = [
    ('Lithuania', 'vilnius_central', 'lithuania/validated_pairs/vilnius_central_validated.gpkg'),
    ('Lithuania', 'kaunas_central', 'lithuania/validated_pairs/kaunas_central_validated.gpkg'),
    ('Lithuania', 'klaipeda_central', 'lithuania/validated_pairs/klaipeda_central_validated.gpkg'),
    ('Poland', 'warsaw', 'poland/road_target_validation/warsaw_validated_road_pair.gpkg'),
    ('Poland', 'krakow', 'poland/road_target_validation/krakow_validated_road_pair.gpkg'),
    ('Poland', 'gdansk', 'poland/road_target_validation/gdansk_validated_road_pair.gpkg'),
]
BUFFERS = (5, 15, 30)


def clean_lines(gdf: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    """Retain original feature rows; do not explode multilines."""
    gdf = gdf.copy()
    gdf['_input_row'] = np.arange(len(gdf), dtype=np.int64)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geom_type.isin(['LineString', 'MultiLineString'])].copy()
    bad = ~gdf.geometry.is_valid
    if bad.any():
        try:
            gdf.loc[bad, gdf.geometry.name] = gdf.loc[bad].geometry.make_valid()
        except Exception as exc:
            raise RuntimeError(f'Cannot repair {label} geometries: {exc}') from exc
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geom_type.isin(['LineString', 'MultiLineString'])].copy()
    gdf.reset_index(drop=True, inplace=True)
    if gdf.empty:
        raise ValueError(f'{label}: no valid line geometries')
    return gdf


def metric_crs(gdf: gpd.GeoDataFrame, label: str) -> None:
    if gdf.crs is None or not gdf.crs.is_projected:
        raise ValueError(f'{label}: projected metric CRS required, found {gdf.crs}')
    axis = gdf.crs.axis_info
    if axis and any((getattr(a, 'unit_conversion_factor', None) or 0) != 1 for a in axis[:2]):
        raise ValueError(f'{label}: CRS axis unit is not metre: {gdf.crs}')


def pct(numerator: float, denominator: float) -> float:
    return float(100 * numerator / denominator) if denominator > 0 else float('nan')


def process(country: str, city: str, relative: str) -> dict:
    path = ROOT / 'data' / 'processed' / relative
    if not path.exists():
        raise FileNotFoundError(path)
    source = clean_lines(gpd.read_file(path, layer='source_roads'), f'{city}/source')
    target = clean_lines(gpd.read_file(path, layer='target_roads'), f'{city}/target')
    metric_crs(source, f'{city}/source')
    metric_crs(target, f'{city}/target')
    if target.crs != source.crs:
        target = target.to_crs(source.crs)
    print(f'\n{country} / {city}: {len(source):,} source, {len(target):,} target, CRS {source.crs}', flush=True)

    # This bounding rectangle is NOT a verified target data-coverage polygon.
    target_bbox = box(*target.total_bounds)
    target_lines = unary_union(list(target.geometry))
    buffered = {}
    for radius in BUFFERS:
        print(f'  Preparing {radius}m target-network buffer...', flush=True)
        buffered[radius] = target_lines.buffer(radius)

    rows = []
    for i, src in source.iterrows():
        geom = src.geometry
        length = float(geom.length)
        inside_bbox = float(geom.intersection(target_bbox).length) if length > 0 else 0.0
        row = {
            'country': country,
            'city': city,
            'source_input_row': int(src['_input_row']),
            'source_feature_length_m': length,
            'length_inside_target_bbox_m': inside_bbox,
            'length_inside_target_bbox_fraction': min(1.0, inside_bbox / length) if length else np.nan,
            'bbox_outside_any': bool(inside_bbox < length - 1e-6),
            'bbox_outside_all': bool(inside_bbox <= 1e-6),
            'coverage_verified': False,
            'confirmed_elimination': False,
        }
        for id_field in ('cartogen_source_id', 'source_id', 'road_id', 'id'):
            if id_field in source.columns:
                value = src[id_field]
                row['source_identifier_field'] = id_field
                row['source_identifier_value'] = str(value) if pd.notna(value) else ''
                break
        else:
            row['source_identifier_field'] = '_input_row'
            row['source_identifier_value'] = str(int(src['_input_row']))
        for radius in BUFFERS:
            supported = float(geom.intersection(buffered[radius]).length) if length else 0.0
            row[f'supported_length_{radius}m_m'] = supported
            row[f'supported_fraction_{radius}m'] = min(1.0, supported / length) if length else np.nan
            row[f'ge50pct_supported_{radius}m'] = bool(length > 0 and supported / length >= .5)
        # This is an evidence tier, not a road-retention label.
        s5, s15, s30 = [row[f'supported_fraction_{r}m'] for r in BUFFERS]
        if row['bbox_outside_any']:
            row['evidence_tier'] = 'bbox_extent_uncertain'
        elif s5 >= .5:
            row['evidence_tier'] = 'strong_geometric_support'
        elif s15 >= .5:
            row['evidence_tier'] = 'moderate_geometric_support'
        elif s30 >= .5:
            row['evidence_tier'] = 'weak_geometric_support'
        else:
            row['evidence_tier'] = 'little_geometric_support_unverified'
        rows.append(row)
        if (i + 1) % 1500 == 0:
            print(f'  Assessed {i + 1:,}/{len(source):,} source roads', flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(OUTPUT / f'{country.lower()}_{city}_source_road_support.csv', index=False)
    total_length = detail['source_feature_length_m'].sum()
    inside_length = detail['length_inside_target_bbox_m'].sum()
    summary = {
        'country': country, 'city': city, 'source_count': len(source), 'target_count': len(target),
        'crs': str(source.crs), 'source_length_km': total_length / 1000,
        'target_length_km': float(target.geometry.length.sum()) / 1000,
        'source_length_inside_target_bbox_pct': pct(inside_length, total_length),
        'bbox_outside_any_count': int(detail['bbox_outside_any'].sum()),
        'bbox_outside_all_count': int(detail['bbox_outside_all'].sum()),
        'coverage_verified': False,
    }
    for radius in BUFFERS:
        summary[f'supported_source_length_{radius}m_pct'] = pct(detail[f'supported_length_{radius}m_m'].sum(), total_length)
        summary[f'supported_source_length_inside_bbox_{radius}m_pct'] = pct(
            np.minimum(detail[f'supported_length_{radius}m_m'], detail['length_inside_target_bbox_m']).sum(),
            inside_length,
        )
        summary[f'source_features_ge50pct_supported_{radius}m_pct'] = pct(
            detail[f'ge50pct_supported_{radius}m'].sum(), len(detail)
        )
    for tier, count in detail['evidence_tier'].value_counts().items():
        summary[f'count_{tier}'] = int(count)
    print(f"  15m supported source length: {summary['supported_source_length_15m_pct']:.2f}%")
    print(f"  Source features with any length outside target bbox: {summary['bbox_outside_any_count']:,}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cities', nargs='*', help='Optional subset of city keys')
    args = parser.parse_args()
    selected = [case for case in CASES if not args.cities or case[1] in args.cities]
    if not selected:
        parser.error('No city matched --cities')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = [process(*case) for case in selected]
    df = pd.DataFrame(summaries).fillna(0)
    df.to_csv(OUTPUT / 'city_road_support_summary.csv', index=False)
    metadata = {
        'purpose': 'read-only geographic extent and geometric support diagnostics',
        'buffers_m': list(BUFFERS),
        'target_coverage_verified': False,
        'warning': 'Target bbox is not verified geographic coverage; geometric support is not a confirmed retention label.',
        'source_identifier_note': 'Identifiers may not align with benchmark IDs; source_input_row indexes the raw GeoPackage layer.',
        'input_files': [str(ROOT / 'data' / 'processed' / rel) for _, _, rel in selected],
    }
    (OUTPUT / 'audit_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    cols = ['country', 'city', 'source_count', 'bbox_outside_any_count', 'supported_source_length_5m_pct', 'supported_source_length_15m_pct', 'supported_source_length_30m_pct']
    print('\n' + df[cols].to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print(f'\nRead-only outputs: {OUTPUT}')
    print('WARNING: Bounding-box flags do not verify target coverage.')
    print('WARNING: Geometric support is not ground-truth retention or elimination.')


if __name__ == '__main__':
    main()
