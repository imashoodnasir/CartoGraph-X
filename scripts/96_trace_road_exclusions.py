#!/usr/bin/env python3
"""CartoGraph-X Script 96: read-only road-record lineage / exclusion-stage audit.

Run from CartoGen root: python scripts/96_trace_road_exclusions.py
Only identifies the first observable stage where a record is absent. It does NOT
infer a filtering cause, assign missing splits, change labels, or change files.
"""
from pathlib import Path
import argparse
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
CROSS = DATA / 'cross_country'
OUT = CROSS / 'road_exclusion_lineage_audit_v1'
SOURCE95 = CROSS / 'benchmark_integrity_exclusion_audit_v1' / 'source_inclusion_exclusion_audit.csv'
GRAPH = CROSS / 'graph_representations' / 'road_graph_nodes.csv'
HARM = CROSS / 'harmonized_roads' / 'lithuania_poland_road_benchmark.csv'
CITIES = [('Lithuania','vilnius_central','Vilnius'),('Lithuania','kaunas_central','Kaunas'),('Lithuania','klaipeda_central','Klaipeda'),('Poland','warsaw','Warsaw'),('Poland','krakow','Krakow'),('Poland','gdansk','Gdansk')]

def norm_city(x):
    s=re.sub(r'[^a-z0-9]','',str(x).casefold())
    for k,v in [('vilniuscentral','vilnius'),('kaunascentral','kaunas'),('klaipedacentral','klaipeda')]:
        if s==k:return v
    return s

def num_id(series, context):
    n=pd.to_numeric(series,errors='coerce')
    if n.isna().any() or not np.isfinite(n.to_numpy(dtype=float)).all() or not (n==np.floor(n)).all():
        raise ValueError(f'{context}: source index missing or not an integer')
    return n.astype('int64')

def stage_frame(path, stage, key_candidates, country=None, city_col_candidates=('city','city_harmonized')):
    if not path.is_file():return None, f'file not found: {path}'
    d=pd.read_csv(path,low_memory=False)
    key=next((k for k in key_candidates if k in d.columns),None)
    if key is None:return None,f'no usable source-index column: {path}; columns={list(d.columns)[:25]}'
    city_col=next((k for k in city_col_candidates if k in d.columns),None)
    if city_col is None:return None,f'no city column: {path}'
    if country is None:
        if 'country' not in d.columns:return None,f'no country column: {path}'
        d['country_key']=d['country'].astype(str).str.casefold().str.strip()
    else:d['country_key']=country.casefold()
    d['city_key']=d[city_col].map(norm_city)
    d['source_input_row']=num_id(d[key],f'{stage}: {path}')
    cols=['country_key','city_key','source_input_row']
    if d.duplicated(cols).any():
        raise ValueError(f'{stage}: duplicate country/city/source index: {path}; {int(d.duplicated(cols).sum())} duplicates')
    if 'length_m' in d.columns:
        d[f'{stage}_length_m']=pd.to_numeric(d['length_m'],errors='coerce')
    else:
        d[f'{stage}_length_m']=np.nan
    if 'retained' in d.columns:
        d[f'{stage}_label']=pd.to_numeric(d['retained'],errors='coerce')
    elif 'is_selected' in d.columns:
        d[f'{stage}_label']=pd.to_numeric(d['is_selected'],errors='coerce')
    else:d[f'{stage}_label']=np.nan
    if 'split_harmonized' in d.columns:split='split_harmonized'
    elif 'split' in d.columns:split='split'
    else:split=None
    d[f'{stage}_split']=d[split].astype('string') if split else pd.Series(pd.NA,index=d.index,dtype='string')
    tile=next((c for c in ['tile_id','tile_id_harmonized','spatial_tile_id'] if c in d.columns),None)
    d[f'{stage}_tile']=d[tile].astype('string') if tile else pd.Series(pd.NA,index=d.index,dtype='string')
    out=d[cols+[f'{stage}_{c}' for c in ['length_m','label','split','tile']]].copy()
    out[f'in_{stage}']=True
    return out,None

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cities',nargs='*',help='Optional city keys')
    args=ap.parse_args()
    if not SOURCE95.is_file():raise FileNotFoundError(f'Run Script 95 first: {SOURCE95}')
    if not GRAPH.is_file():raise FileNotFoundError(GRAPH)
    base=pd.read_csv(SOURCE95,low_memory=False)
    required=['country','city','source_input_row','inclusion_status','original_label','source_feature_length_m']
    if set(required)-set(base.columns):raise ValueError('Script95 missing '+str(set(required)-set(base.columns)))
    base['country_key']=base.country.astype(str).str.casefold().str.strip()
    base['city_key']=base.city.map(norm_city)
    base['source_input_row']=num_id(base.source_input_row,'Script95')
    if args.cities:
        wanted={norm_city(x) for x in args.cities}
        base=base[base.city_key.isin(wanted)].copy()
    if base.empty:raise ValueError('No selected source records')
    keys=['country_key','city_key','source_input_row']
    if base.duplicated(keys).any():raise ValueError('Script95 duplicate source identifiers')
    notes=[]
    stages=[]
    # Country-specific upstream spatial benchmarks. These are optional: do not assume their schemas.
    candidates=[
        ('lt_spatial',DATA/'lithuania'/'spatial_benchmark'/'lithuania_road_benchmark.csv','Lithuania', ['feature_index','source_idx','source_idx_harmonized']),
        ('pl_spatial',DATA/'poland'/'spatial_benchmark'/'poland_road_benchmark.csv','Poland',['source_idx','feature_index','source_idx_harmonized']),
        ('harmonized',HARM,None,['source_idx_harmonized','source_idx','feature_index']),
        ('graph',GRAPH,None,['source_idx_harmonized','source_idx','feature_index']),
    ]
    for name,path,country,kc in candidates:
        frame,error=stage_frame(path,name,kc,country)
        if error:
            notes.append({'stage':name,'status':'UNAVAILABLE','path':str(path),'note':error})
            print(f'{name:12s} unavailable: {error}')
            continue
        if country:
            frame=frame[frame.country_key.eq(country.casefold())].copy()
        stages.append((name,frame))
        notes.append({'stage':name,'status':'AVAILABLE','path':str(path),'note':f'{len(frame)} indexed records'})
        print(f'{name:12s} indexed={len(frame):,}')
    if not any(name=='harmonized' for name,_ in stages):
        print('WARNING: harmonized stage unavailable; cannot localize pre-graph exclusions.')
    for name,frame in stages:
        base=base.merge(frame,on=keys,how='left',validate='one_to_one')
        base[f'in_{name}']=base[f'in_{name}'].fillna(False).astype(bool)
        a=pd.to_numeric(base.source_feature_length_m,errors='coerce')
        b=base[f'{name}_length_m']
        delta=(a-b).abs()
        base[f'{name}_length_verified']=base[f'in_{name}'] & b.notna() & ((delta<=.10)|(delta/a.clip(lower=1)<=.001))
        base[f'{name}_length_mismatch']=base[f'in_{name}'] & b.notna() & ~base[f'{name}_length_verified']
        base[f'{name}_label_disagreement']=base[f'in_{name}'] & base[f'{name}_label'].notna() & pd.to_numeric(base.original_label,errors='coerce').ne(base[f'{name}_label'])
    has_harm='in_harmonized' in base
    has_graph='in_graph' in base
    if has_graph:
        expected=base.inclusion_status.eq('included')
        discrepancy=base.in_graph.ne(expected)
        if discrepancy.any():raise ValueError(f'Graph membership differs from Script95 for {int(discrepancy.sum())} records; unsafe lineage')
    def classify(r):
        if has_harm:
            if not r['in_harmonized']:return 'absent_before_or_at_harmonization'
            if has_graph and not r['in_graph']:return 'present_harmonized_absent_graph'
            if has_graph:return 'present_graph'
            return 'present_harmonized_graph_unavailable'
        return 'stage_unresolved_harmonized_unavailable'
    base['first_observed_absence']=base.apply(classify,axis=1)
    if 'in_pl_spatial' in base:
        mask=base.country_key.eq('poland') & base.first_observed_absence.eq('absent_before_or_at_harmonization')
        base.loc[mask & base.in_pl_spatial,'first_observed_absence']='present_pl_spatial_absent_harmonized'
        base.loc[mask & ~base.in_pl_spatial,'first_observed_absence']='absent_at_or_before_pl_spatial'
    if 'in_lt_spatial' in base:
        mask=base.country_key.eq('lithuania') & base.first_observed_absence.eq('absent_before_or_at_harmonization')
        base.loc[mask & base.in_lt_spatial,'first_observed_absence']='present_lt_spatial_absent_harmonized'
        base.loc[mask & ~base.in_lt_spatial,'first_observed_absence']='absent_at_or_before_lt_spatial'
    # The stage is observable, the specific filtering reason is not.
    base['confirmed_exclusion_cause']='NOT_ESTABLISHED'
    OUT.mkdir(parents=True,exist_ok=True)
    base.to_csv(OUT/'source_record_stage_lineage.csv',index=False)
    excluded=base[base.inclusion_status.eq('excluded')].copy()
    excluded.to_csv(OUT/'excluded_record_stage_lineage.csv',index=False)
    summary=base.groupby(['country','city','first_observed_absence'],dropna=False).agg(rows=('source_input_row','size'),original_positive=('original_label',lambda x:int(pd.to_numeric(x,errors='coerce').eq(1).sum())),total_source_length_m=('source_feature_length_m','sum')).reset_index()
    summary.to_csv(OUT/'city_stage_lineage_summary.csv',index=False)
    country_summary=excluded.groupby(['country','first_observed_absence'],dropna=False).size().reset_index(name='excluded_roads')
    country_summary.to_csv(OUT/'country_exclusion_stage_summary.csv',index=False)
    checks=[]
    for name,_ in stages:
        present=base[f'in_{name}']
        checks.append({'stage':name,'matched':int(present.sum()),'length_verified':int(base[f'{name}_length_verified'].sum()),'length_mismatch':int(base[f'{name}_length_mismatch'].sum()),'label_disagreement_when_available':int(base[f'{name}_label_disagreement'].sum())})
    pd.DataFrame(checks).to_csv(OUT/'stage_identity_checks.csv',index=False)
    pd.DataFrame(notes).to_csv(OUT/'stage_input_inventory.csv',index=False)
    (OUT/'audit_metadata.json').write_text(json.dumps({'script':'96','purpose':'read-only first-observable-exclusion-stage tracing','excluded_split':'not inferred','exclusion_cause':'not established by stage absence','stage_availability':notes,'warning':'Stage membership is not evidence of the precise upstream filtering reason. Row index plus length is not an immutable feature identifier.'},indent=2),encoding='utf-8')
    print('\nEXCLUDED ROADS BY FIRST OBSERVED ABSENCE\n'+country_summary.to_string(index=False))
    print('\nSTAGE IDENTITY CHECKS\n'+pd.DataFrame(checks).to_string(index=False))
    print('\nOutputs:',OUT)
    print('CAUTION: Missing from a stage does not establish WHY a road was removed.')

if __name__=='__main__':main()
