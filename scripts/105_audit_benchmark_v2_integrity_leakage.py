#!/usr/bin/env python3
"""Script 105: read-only CartoGraph-X Benchmark V2 integrity and leakage audit.

Run: python scripts/105_audit_benchmark_v2_integrity_leakage.py
Exit nonzero on any hard failure. Never edits V1/V2 source files.
This audits dataset-level leakage only, not the training/model-selection code.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data' / 'processed' / 'cross_country'
SPLITS = {'train', 'validation', 'test'}
COUNTRIES = {'Lithuania', 'Poland'}
EXPECTED = {('Lithuania','train'):7465,('Lithuania','validation'):1537,('Lithuania','test'):1924,
            ('Poland','train'):10267,('Poland','validation'):2289,('Poland','test'):2552}
EXPECTED_POS = {('Lithuania','train'):5103,('Lithuania','validation'):1037,('Lithuania','test'):1339,
                ('Poland','train'):2075,('Poland','validation'):404,('Poland','test'):565}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize_id(s):
    return s.astype('string').str.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--v1', type=Path, default=BASE/'graph_representations'/'road_graph_nodes.csv')
    ap.add_argument('--v2-dir', type=Path, default=BASE/'benchmark_v2_unified_labels_v1')
    ap.add_argument('--edges', type=Path, default=BASE/'graph_representations'/'road_graph_edges.csv')
    ap.add_argument('--out', type=Path, default=BASE/'benchmark_v2_integrity_audit_v1')
    ap.add_argument('--allow-unexpected-counts', action='store_true', help='Allow counts different from the frozen Script 104 run, but still check consistency')
    args = ap.parse_args()
    v2path = args.v2_dir/'road_graph_nodes_v2.csv'
    fullpath = args.v2_dir/'all_source_roads_c2_labels.csv'
    manifestpath = args.v2_dir/'benchmark_v2_manifest.json'
    required_paths = [args.v1,v2path,fullpath,manifestpath,args.edges]
    for p in required_paths:
        if not p.is_file():
            raise FileNotFoundError(f'Missing input: {p}')
    if args.out.resolve() == args.v2_dir.resolve() or args.out.resolve() == args.v1.parent.resolve():
        raise ValueError('Audit output must not overwrite benchmark inputs')

    checks=[]
    def check(name, passed, detail='', severity='ERROR'):
        ok = bool(passed)
        checks.append(dict(check=name,status='PASS' if ok else ('WARN' if severity=='WARN' else 'FAIL'),detail=str(detail)))
        print(f"[{'PASS' if ok else ('WARN' if severity=='WARN' else 'FAIL')}] {name}: {detail}",flush=True)

    print('Reading V1/V2 graph nodes, C2 source lineage, and graph edges...',flush=True)
    v1 = pd.read_csv(args.v1,low_memory=False)
    v2 = pd.read_csv(v2path,low_memory=False)
    full = pd.read_csv(fullpath,low_memory=False)
    edges = pd.read_csv(args.edges,low_memory=False)
    manifest = json.loads(manifestpath.read_text(encoding='utf-8'))
    check('row_count_26034',len(v1)==len(v2)==26034,f'V1={len(v1)}, V2={len(v2)}')
    check('all_source_29455',len(full)==29455,f'full source={len(full)}')
    for name,df,needed in [
        ('v1',v1,['sample_uid','country','city','split','tile_id','source_idx_harmonized','retained']),
        ('v2',v2,['sample_uid','country','city','split','tile_id','source_idx_harmonized','retained','v1_retained','v2_label_changed_from_v1','v2_correspondence_config']),
        ('full',full,['country','city','source_input_row','v2_retained','graph_included','subset']),
        ('edges',edges,['source_uid','target_uid','country','city','split','tile_id'])]:
        missing=set(needed)-set(df.columns)
        check(f'{name}_required_columns',not missing,f'missing={sorted(missing)}')
    if any(x['status']=='FAIL' for x in checks):
        print('Required columns absent; cannot safely continue.',file=sys.stderr)
        return save_report(args,checks,[],[],manifest,required_paths)

    for dfname,df in [('v1',v1),('v2',v2)]:
        check(f'{dfname}_unique_uid',df.sample_uid.notna().all() and df.sample_uid.is_unique,
              f'null={df.sample_uid.isna().sum()}, duplicate={df.sample_uid.duplicated().sum()}')
        check(f'{dfname}_unique_source',not df.duplicated(['country','city','source_idx_harmonized']).any(),
              f'duplicates={df.duplicated(["country","city","source_idx_harmonized"]).sum()}')
        check(f'{dfname}_valid_splits_countries',set(df.split.dropna())==SPLITS and set(df.country.dropna())==COUNTRIES,
              f'splits={sorted(df.split.dropna().unique())}, countries={sorted(df.country.dropna().unique())}')
        check(f'{dfname}_valid_binary',pd.to_numeric(df.retained,errors='coerce').isin([0,1]).all(),
              f'null/nonbinary={int((~pd.to_numeric(df.retained,errors="coerce").isin([0,1])).sum())}')
        check(f'{dfname}_complete_metadata',df[['country','city','split','tile_id','source_idx_harmonized']].notna().all().all())
    check('graph_uid_order_identical',v1.sample_uid.astype(str).equals(v2.sample_uid.astype(str)))
    shared = sorted(set(v1.columns)&set(v2.columns)-{'retained','selection_label_harmonized','operation_harmonized'})
    diffs=[]
    for col in shared:
        a,b=v1[col],v2[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            # Floating graph measurements may differ at ~1e-14 after CSV
            # serialization. Discrete identifiers/counts must remain exact.
            if pd.api.types.is_float_dtype(a) or pd.api.types.is_float_dtype(b):
                equal=np.allclose(
                    a.to_numpy(dtype=np.float64),
                    b.to_numpy(dtype=np.float64),
                    rtol=1e-9, atol=1e-9, equal_nan=True,
                )
            else:
                equal=np.array_equal(a.to_numpy(),b.to_numpy(),equal_nan=True)
        else:
            equal=a.fillna('__NA__').astype(str).equals(b.fillna('__NA__').astype(str))
        if not equal:
            diffs.append(col)
    check('unchanged_original_graph_metadata_features',not diffs,f'checked={len(shared)}, modified={diffs[:20]}')
    check('v1_label_backups',np.array_equal(pd.to_numeric(v1.retained).to_numpy(),pd.to_numeric(v2.v1_retained).to_numpy()),'V1 retained backup')
    changed=(pd.to_numeric(v2.retained)!=pd.to_numeric(v1.retained))
    recorded=v2.v2_label_changed_from_v1.astype(str).str.lower().isin(['true','1'])
    check('changed_flags_exact',np.array_equal(changed.to_numpy(),recorded.to_numpy()),f'changed={int(changed.sum())}')
    check('frozen_config_applied',v2.v2_correspondence_config.eq('C2_intermediate').all())
    check('status_provisional', 'v2_label_status' in v2 and v2.v2_label_status.astype(str).str.contains('NOT_MANUALLY_VALIDATED').all())
    check('v2_label_text_consistent', 'selection_label_harmonized' in v2 and np.array_equal(v2.selection_label_harmonized.eq('selected').to_numpy(),pd.to_numeric(v2.retained).eq(1).to_numpy()))

    count_rows=[]
    for (country,split),expected_n in EXPECTED.items():
        sub=v2[(v2.country==country)&(v2.split==split)]
        pos=int(pd.to_numeric(sub.retained,errors='coerce').fillna(-1).eq(1).sum())
        count_rows.append(dict(country=country,split=split,n=len(sub),v1_positive=int(v1.loc[sub.index,'retained'].sum()),v2_positive=pos,changed=int(changed.loc[sub.index].sum()),prevalence=round(pos/len(sub),6) if len(sub) else None))
        if not args.allow_unexpected_counts:
            check(f'count_{country}_{split}',len(sub)==expected_n and pos==EXPECTED_POS[(country,split)],f'n={len(sub)}, positive={pos}; expected n={expected_n}, positive={EXPECTED_POS[(country,split)]}')
    check('all_graph_roads_accounted',sum(x['n'] for x in count_rows)==len(v2))
    # Tiles may legitimately repeat across cities. Only country+city+tile_id defines a spatial tile.
    tile_splits=v2.groupby(['country','city','tile_id'],dropna=False)['split'].nunique(dropna=False)
    check('no_tile_cross_split_overlap',(tile_splits<=1).all(),f'overlapping tiles={int((tile_splits>1).sum())}')
    uid_meta=v2.set_index('sample_uid')[['country','city','split','tile_id']]
    src=edges.source_uid.astype(str)
    dst=edges.target_uid.astype(str)
    node_uids=pd.Index(v2.sample_uid.astype(str))
    check('edge_endpoints_exist',src.isin(node_uids).all() and dst.isin(node_uids).all(),f'edges={len(edges)}, missing endpoints={int((~src.isin(node_uids)|~dst.isin(node_uids)).sum())}')
    if src.isin(node_uids).all() and dst.isin(node_uids).all():
        uid_meta.index=uid_meta.index.astype(str)
        for field in ['country','city','split','tile_id']:
            left=src.map(uid_meta[field]).to_numpy()
            right=dst.map(uid_meta[field]).to_numpy()
            same=np.asarray(left==right)
            edge_same=(edges[field].astype(str).to_numpy()==left.astype(str))
            check(f'edge_same_{field}',same.all() and edge_same.all(),f'cross={int((~same).sum())}, edge-metadata-mismatch={int((~edge_same).sum())}')
        check('no_self_loops',(~src.eq(dst)).all(),f'self_loops={int(src.eq(dst).sum())}')
    # Check full source lineage against the V2 graph using exact source row, not positional matching.
    graph_key=v2[['country','city','source_idx_harmonized','retained','split','tile_id']].copy()
    graph_key=graph_key.rename(columns={'source_idx_harmonized':'source_input_row','retained':'graph_v2_retained'})
    city_names={'Vilnius':'vilnius_central','Kaunas':'kaunas_central','Klaipeda':'klaipeda_central',
                'Warsaw':'warsaw','Krakow':'krakow','Gdansk':'gdansk'}
    graph_key['city']=graph_key['city'].replace(city_names)
    full_key=full[['country','city','source_input_row','v2_retained','graph_included','subset']].copy()
    for df in [graph_key,full_key]:
        df['source_input_row']=pd.to_numeric(df.source_input_row,errors='coerce')
    check('full_source_unique_key',not full_key.duplicated(['country','city','source_input_row']).any())
    lineage=graph_key.merge(full_key,on=['country','city','source_input_row'],how='left',validate='one_to_one',indicator=True)
    check('graph_to_full_source_identity',len(lineage)==len(v2) and lineage._merge.eq('both').all(),f'unmatched={int(lineage._merge.ne("both").sum())}')
    if lineage._merge.eq('both').all():
        check('graph_v2_labels_match_c2_source',pd.to_numeric(lineage.graph_v2_retained).eq(pd.to_numeric(lineage.v2_retained)).all())
        check('full_source_inclusion_matches',lineage.graph_included.astype(str).str.lower().isin(['true','1']).all())
        check('full_source_split_matches',lineage.split.astype(str).eq(lineage.subset.astype(str)).all())
    included=full.graph_included.astype(str).str.lower().isin(['true','1'])
    check('full_inclusion_count',int(included.sum())==len(v2),f'included={int(included.sum())}, excluded={int((~included).sum())}')
    # Manifest provenance: verify hashes of source inputs/outputs recorded by Script 104.
    manifest_hashes=manifest.get('output_sha256',{})
    for filename in ['road_graph_nodes_v2.csv','all_source_roads_c2_labels.csv']:
        check(f'manifest_output_hash_{filename}',manifest_hashes.get(filename)==sha256(args.v2_dir/filename),f'hash present={filename in manifest_hashes}')
    for name,expected_hash in manifest.get('input_sha256',{}).items():
        source=Path(name)
        if source.exists():
            check(f'manifest_input_hash_{source.name}',sha256(source)==expected_hash)
        else:
            check(f'manifest_input_exists_{source.name}',False,f'cannot verify {source}',severity='WARN')
    check('provisional_manifest_status',manifest.get('configuration_status')=='PROVISIONAL_NOT_MANUALLY_VALIDATED')
    check('test_not_untouched',False,'Historical test roads were examined in V1–V10; use an independent holdout for confirmatory results',severity='WARN')
    check('training_pipeline_not_audited',False,'Dataset checks cannot prove source-only model selection; audit training scripts separately',severity='WARN')
    return save_report(args,checks,count_rows,[{'country':str(k[0]),'city':str(k[1]),'tile_id':str(k[2]),'split_count':int(v)} for k,v in tile_splits.items() if v>1],manifest,required_paths)


def save_report(args,checks,counts,overlap,manifest,inputs):
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(checks,columns=['check','status','detail']).to_csv(out/'integrity_checks.csv',index=False)
    pd.DataFrame(counts,columns=['country','split','n','v1_positive','v2_positive','changed','prevalence']).to_csv(out/'country_split_label_summary.csv',index=False)
    pd.DataFrame(overlap,columns=['country','city','tile_id','split_count']).to_csv(out/'cross_split_tiles.csv',index=False)
    failures=[c for c in checks if c['status']=='FAIL']
    warnings=[c for c in checks if c['status']=='WARN']
    report={'audit':'CartoGraph-X Benchmark V2 integrity and dataset-level leakage',
            'generated_utc':datetime.now(timezone.utc).isoformat(),
            'status':'FAIL' if failures else 'PASS_WITH_WARNINGS' if warnings else 'PASS',
            'checks':len(checks),'failures':len(failures),'warnings':len(warnings),
            'manual_validation':'NOT_COMPLETED',
            'holdout':'NOT_INDEPENDENT_FROM_PREVIOUS_EXPLORATORY_MODEL_DEVELOPMENT',
            'training_code_source_only_selection':'NOT_AUDITED',
            'input_sha256':{str(p.resolve()):sha256(p) for p in inputs}}
    (out/'audit_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('\n'+'='*76)
    print(f"SCRIPT 105 STATUS: {report['status']} | checks={len(checks)} failures={len(failures)} warnings={len(warnings)}")
    print(f'Output: {out.resolve()}')
    if failures:
        print('FAILED CHECKS: '+', '.join(x['check'] for x in failures))
    print('No benchmark files modified. Source-only training selection still requires a separate code audit.')
    return 2 if failures else 0

if __name__=='__main__':
    sys.exit(main())
