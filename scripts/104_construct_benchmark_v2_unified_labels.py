#!/usr/bin/env python3
"""Script 104 — construct provisional CartoGraph-X Benchmark V2 from frozen C2.

Uses the record-level *geometric correspondence* output of Script 99 (which
computed directional line coverage for all six validated source/target pairs).
Verifies the C2 freeze and source-row/geometry lineage; does not recompute or
change the frozen thresholds. Outputs a separate full-source audit and a
26,034-node graph-aligned V2 dataset with original splits and features intact.

The V1 test roads have been used during earlier development: all V2 test
results on these roads remain exploratory, not confirmatory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data' / 'processed' / 'cross_country'
DEFAULT_FREEZE = BASE / 'benchmark_v2_provisional_configuration_v1' / 'provisional_correspondence_config.json'
DEFAULT_RECORDS = BASE / 'benchmark_v2_unified_correspondence_audit_v1' / 'record_level_correspondence_sensitivity.csv'
DEFAULT_GRAPH = BASE / 'graph_representations' / 'road_graph_nodes.csv'
DEFAULT_OUT = BASE / 'benchmark_v2_unified_labels_v1'
EXPECTED = dict(config='C2_intermediate', search_m=15.0, buffer_m=5.0, source_min=0.20, target_min=0.03)
EXPECTED_CITY = {'Lithuania': {'vilnius_central': 'Vilnius', 'kaunas_central': 'Kaunas', 'klaipeda_central': 'Klaipeda'},
                 'Poland': {'warsaw': 'Warsaw', 'krakow': 'Krakow', 'gdansk': 'Gdansk'}}
EXPECTED_SOURCE_N = {'Lithuania': 11129, 'Poland': 18326}
EXPECTED_GRAPH_N = {'Lithuania': 10926, 'Poland': 15108}
EXPECTED_TRAIN = {'Lithuania': (7465, 5103), 'Poland': (10267, 2075)}


def require(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Required file not found: {path}')
    return path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fail(msg):
    raise ValueError('BENCHMARK V2 INTEGRITY ERROR: ' + msg)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze', type=Path, default=DEFAULT_FREEZE)
    p.add_argument('--records', type=Path, default=DEFAULT_RECORDS)
    p.add_argument('--graph', type=Path, default=DEFAULT_GRAPH)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--verify-geometry', action='store_true', help='Read all six GPKGs and verify original row lengths (slower)')
    args = p.parse_args()
    freeze_path, records_path, graph_path = map(require, [args.freeze, args.records, args.graph])
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    if freeze.get('status') != 'PROVISIONAL_NOT_MANUALLY_VALIDATED':
        fail('Unexpected freeze status; this script requires the provisional C2 freeze')
    cfg = freeze.get('configuration', {})
    for key, expected in EXPECTED.items():
        got = cfg.get(key)
        if (got != expected if isinstance(expected, str) else got is None or abs(float(got) - expected) > 1e-10):
            fail(f'Frozen parameter {key}: expected {expected!r}, got {got!r}')

    print('Reading frozen C2 correspondence records and existing graph nodes...', flush=True)
    usecols = ['country', 'city', 'source_input_row', 'config', 'subset', 'tile_id', 'original_label',
               'unified_proxy_label', 'source_length_m', 'best_source_coverage', 'best_target_coverage',
               'best_distance_m', 'candidate_count', 'invalid_or_empty_geometry']
    records = pd.read_csv(records_path, usecols=usecols, low_memory=False)
    graph = pd.read_csv(graph_path, low_memory=False)
    required_graph = {'sample_uid','country','city','source_idx_harmonized','tile_id','split','length_m','retained','selection_label_harmonized'}
    if missing := required_graph - set(graph.columns):
        fail(f'Graph missing columns {sorted(missing)}')
    records = records.loc[records['config'].eq(EXPECTED['config'])].copy()
    if len(records) != sum(EXPECTED_SOURCE_N.values()):
        fail(f'Expected {sum(EXPECTED_SOURCE_N.values())} C2 source records, found {len(records)}')
    if records[['country','city','source_input_row']].isna().any().any():
        fail('Null source identity in Script 99 records')
    if records.duplicated(['country','city','source_input_row']).any():
        fail('Duplicate C2 source identities')
    if graph['sample_uid'].isna().any() or graph['sample_uid'].duplicated().any():
        fail('Duplicate or missing sample_uid in graph')
    if graph.duplicated(['country','city','source_idx_harmonized']).any():
        fail('Duplicate graph source identities')
    for country, cities in EXPECTED_CITY.items():
        rec_country = records.loc[records.country.eq(country)]
        graph_country = graph.loc[graph.country.eq(country)]
        if len(rec_country) != EXPECTED_SOURCE_N[country] or len(graph_country) != EXPECTED_GRAPH_N[country]:
            fail(f'{country}: source or graph count changed')
        if set(rec_country.city) != set(cities) or set(graph_country.city) != set(cities.values()):
            fail(f'{country}: unexpected city names')
        for key, city in cities.items():
            a = rec_country.loc[rec_country.city.eq(key)].source_input_row
            if not np.array_equal(np.sort(a.to_numpy()), np.arange(len(a))):
                fail(f'{country}/{key}: source rows not exactly 0..N-1')

    records['source_input_row'] = pd.to_numeric(records['source_input_row'], errors='raise').astype('int64')
    records['v2_retained'] = pd.to_numeric(records['unified_proxy_label'], errors='raise').astype('int64')
    if not records.v2_retained.isin([0,1]).all():
        fail('V2 labels are not binary')
    if records['invalid_or_empty_geometry'].fillna(False).astype(bool).any():
        fail('Invalid source geometries present; investigate before labeling')
    if ((records.v2_retained.eq(1)) & (records.best_source_coverage < cfg['source_min'] - 1e-9)).any():
        fail('A positive road violates frozen minimum source coverage')
    if ((records.v2_retained.eq(1)) & (records.best_target_coverage < cfg['target_min'] - 1e-9)).any():
        fail('A positive road violates frozen minimum target coverage')
    if ((records.v2_retained.eq(1)) & (records.best_distance_m > cfg['search_m'] + 1e-9)).any():
        fail('A positive road violates frozen search distance')

    # Join by authoritative original source row, not by row order, tile order or geometry proximity.
    records['graph_city'] = records.apply(lambda r: EXPECTED_CITY[r['country']][r['city']], axis=1)
    graph['source_input_row'] = pd.to_numeric(graph['source_idx_harmonized'], errors='raise').astype('int64')
    merged = graph.merge(records.drop(columns=['config','city']).rename(columns={'graph_city':'city'}),
                         on=['country','city','source_input_row'], how='left', validate='one_to_one', indicator=True,
                         suffixes=('_graph','_record'), sort=False)
    if len(merged) != len(graph) or not merged['_merge'].eq('both').all():
        fail(f'Graph-to-source join failed for {int(merged._merge.ne("both").sum())} graph roads')
    if not np.allclose(pd.to_numeric(merged.length_m, errors='raise'),
                       pd.to_numeric(merged.source_length_m, errors='raise'), rtol=1e-3, atol=.1):
        fail('Graph/source road geometry lengths disagree')
    if not merged['split'].astype(str).eq(merged['subset'].astype(str)).all():
        fail('Graph/source split mismatch')
    if not merged['tile_id_graph'].astype(str).eq(merged['tile_id_record'].astype(str)).all():
        fail('Graph/source tile mismatch')
    if not merged['retained'].astype(int).eq(pd.to_numeric(merged.original_label).astype(int)).all():
        fail('Graph V1 labels differ from Script 99 source lineage')
    if not merged.split.isin(['train','validation','test']).all():
        fail('Unexpected split')

    if args.verify_geometry:
        import geopandas as gpd
        for country, cities in EXPECTED_CITY.items():
            for citykey, city in cities.items():
                gpkg = (ROOT/'data'/'processed'/'lithuania'/'validated_pairs'/f'{citykey}_validated.gpkg'
                        if country == 'Lithuania' else
                        ROOT/'data'/'processed'/'poland'/'road_target_validation'/f'{citykey}_validated_road_pair.gpkg')
                src = gpd.read_file(require(gpkg), layer='source_roads').reset_index(drop=True)
                a = records.loc[(records.country == country) & (records.city == citykey)].sort_values('source_input_row')
                if len(src) != len(a) or not np.allclose(src.geometry.length.to_numpy(),
                        a.source_length_m.to_numpy(), rtol=1e-3, atol=.1):
                    fail(f'Original GPKG source geometry mismatch: {country}/{citykey}')
                print(f'  Geometry verified: {country}/{citykey} ({len(src):,})')

    for country, (n, matched) in EXPECTED_TRAIN.items():
        part = merged[(merged.country == country) & (merged.split == 'train')]
        if len(part) != n or int(part.v2_retained.sum()) != matched:
            fail(f'{country}: C2 training counts differ from frozen training-only evidence')

    # Restore all original graph columns unchanged except label-specific V2 columns.
    # The merge suffixes shared metadata (e.g. tile_id_graph/tile_id_record).
    # Start from the authoritative original graph instead of reconstructing it
    # from suffixed merged columns; this preserves all features, tiles, splits,
    # identities and row order byte-for-byte at the dataframe-value level.
    original_cols = [c for c in graph.columns if c != 'source_input_row']
    v2 = graph[original_cols].copy().reset_index(drop=True)
    if len(v2) != len(merged) or not v2['sample_uid'].equals(merged['sample_uid'].reset_index(drop=True)):
        fail('Merged correspondence rows do not align with authoritative graph rows')
    v2['v1_retained'] = v2['retained'].astype('int64')
    v2['v1_selection_label_harmonized'] = v2['selection_label_harmonized'].astype(str)
    v2['retained'] = merged['v2_retained'].to_numpy(dtype='int64')
    v2['selection_label_harmonized'] = np.where(v2.retained.eq(1), 'selected', 'elimination')
    # Original operation categories are country-specific V1 interpretations and must NOT be reused as V2 operations.
    if 'operation_harmonized' in v2.columns:
        v2['v1_operation_harmonized'] = v2['operation_harmonized'].astype(str)
        v2['operation_harmonized'] = np.where(v2.retained.eq(1), 'geometric_correspondence', 'no_geometric_correspondence')
    v2['v2_correspondence_config'] = EXPECTED['config']
    v2['v2_best_source_coverage'] = merged.best_source_coverage.to_numpy()
    v2['v2_best_target_coverage'] = merged.best_target_coverage.to_numpy()
    v2['v2_best_distance_m'] = merged.best_distance_m.to_numpy()
    v2['v2_proximity_candidate_count'] = merged.candidate_count.to_numpy()
    v2['v2_label_changed_from_v1'] = v2.retained.ne(v2.v1_retained)
    v2['v2_label_status'] = 'PROVISIONAL_GEOMETRIC_PROXY_NOT_MANUALLY_VALIDATED'
    if not v2.sample_uid.equals(graph.sample_uid):
        fail('Graph row ordering changed')

    full = records.copy()
    full['graph_included'] = full['subset'].isin(['train','validation','test'])
    full['v2_label_status'] = 'PROVISIONAL_GEOMETRIC_PROXY_NOT_MANUALLY_VALIDATED'
    full['v1_retained_if_in_graph'] = full['original_label']
    full['v2_label_changed_if_in_graph'] = np.where(full.graph_included,
                                                    full.v2_retained.ne(pd.to_numeric(full.original_label)), np.nan)
    full['original_label'] = full['original_label'].where(full.graph_included, np.nan)
    # Excluded roads have no authoritative V1 graph split or V1 graph label; don't invent them.
    if full.loc[~full.graph_included, 'original_label'].notna().any():
        fail('Excluded road incorrectly received a graph label')
    summary = (v2.groupby(['country','city','split'], as_index=False)
               .agg(n=('retained','size'), v1_positive=('v1_retained','sum'),
                    v2_positive=('retained','sum'), changed=('v2_label_changed_from_v1','sum')))
    summary['v1_positive_pct'] = (100 * summary.v1_positive / summary.n).round(4)
    summary['v2_positive_pct'] = (100 * summary.v2_positive / summary.n).round(4)
    summary['changed_pct'] = (100 * summary.changed / summary.n).round(4)
    country = (summary.groupby(['country','split'], as_index=False)
               [['n','v1_positive','v2_positive','changed']].sum())
    country['v1_positive_pct'] = (100 * country.v1_positive / country.n).round(4)
    country['v2_positive_pct'] = (100 * country.v2_positive / country.n).round(4)
    country['changed_pct'] = (100 * country.changed / country.n).round(4)
    excluded = (full.groupby(['country','graph_included'], as_index=False)
                .agg(n=('v2_retained','size'), v2_positive=('v2_retained','sum')))
    out = args.out
    if out.exists() and any(out.iterdir()):
        fail(f'Output directory is not empty: {out}; use --out NEW_DIRECTORY to prevent accidental overwrites')
    out.mkdir(parents=True, exist_ok=True)
    v2.to_csv(out/'road_graph_nodes_v2.csv', index=False)
    full.to_csv(out/'all_source_roads_c2_labels.csv', index=False)
    summary.to_csv(out/'city_split_v1_v2_label_audit.csv', index=False)
    country.to_csv(out/'country_split_v1_v2_label_audit.csv', index=False)
    excluded.to_csv(out/'included_excluded_source_counts.csv', index=False)
    manifest = dict(benchmark='CartoGraph-X Benchmark V2 provisional', configuration=cfg,
                    configuration_status=freeze['status'], original_graph_node_count=len(graph),
                    source_road_count=len(full), graph_row_order_preserved=True,
                    graph_features_preserved=True,
                    graph_edges_reference=str((graph_path.parent/'road_graph_edges.csv').resolve()),
                    graph_edges_not_rebuilt=True,
                    source_to_graph_join='country + city + original zero-based source row, checked against geometry length, tile, split and V1 label',
                    geometry_files_reverified=bool(args.verify_geometry),
                    positive_label_definition='At least one source-target road pair meets frozen C2 distance and both directional coverage criteria',
                    excluded_roads='Recorded for lineage but not inserted into existing spatial tiles or graph',
                    manual_validation='NOT_COMPLETED',
                    evaluation_warning='Existing test geographic roads previously examined; V2 test metrics exploratory only. Require genuinely untouched holdout for confirmatory evaluation.',
                    input_sha256={str(p.resolve()): sha256(p) for p in [freeze_path, records_path, graph_path]},
                    output_sha256={f.name: sha256(f) for f in sorted(out.glob('*.csv'))},
                    generated_at_utc=datetime.now(timezone.utc).isoformat())
    (out/'benchmark_v2_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print('\nBENCHMARK V2 — PROVISIONAL C2 LABELS, GRAPH-ALIGNED')
    print(country.to_string(index=False))
    print('\nINCLUDED / EXCLUDED SOURCE ROADS')
    print(excluded.to_string(index=False))
    print(f'\nOutput: {out.resolve()}')
    print('Original V1 untouched. Existing graph features, spatial splits and graph edges preserved.')
    print('IMPORTANT: V2 labels are geometric proxies, not manually validated correspondence ground truth.')

if __name__ == '__main__':
    main()
