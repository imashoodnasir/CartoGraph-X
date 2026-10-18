#!/usr/bin/env python3
"""CartoGraph-X Script 100: training-only geometric correspondence assessment.

READ-ONLY. Consumes Script 99 record-level output. Does not inspect validation/test
labels, select a winning configuration, or rewrite existing benchmark artifacts.
Best-accepted-pair statistics are conditional on matching; candidate_count counts
proximity candidates, not accepted alternatives. This cannot validate cartographic
correctness, topology, orientation, or semantic correspondence.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data'/'processed'/'cross_country'
INPUT=BASE/'benchmark_v2_unified_correspondence_audit_v1'/'record_level_correspondence_sensitivity.csv'
CONFIG_FILE=INPUT.parent/'configurations.json'
OUT=BASE/'benchmark_v2_configuration_assessment_v1'
EXPECTED_COUNTRIES={'Lithuania','Poland'}
EXPECTED_CONFIGS={'C1_conservative','C2_intermediate','C3_permissive'}
REQUIRED={'country','city','source_input_row','config','subset','unified_proxy_label',
          'source_length_m','best_source_coverage','best_target_coverage',
          'best_distance_m','candidate_count','invalid_or_empty_geometry'}

def quant(s,q):
    v=pd.to_numeric(s,errors='coerce').dropna()
    return float(v.quantile(q)) if len(v) else np.nan

def assessment(g):
    n=len(g)
    match=g[g.unified_proxy_label.eq(1)]
    unmatched=g[g.unified_proxy_label.eq(0)]
    sc=pd.to_numeric(match.best_source_coverage,errors='coerce')
    tc=pd.to_numeric(match.best_target_coverage,errors='coerce')
    dist=pd.to_numeric(match.best_distance_m,errors='coerce')
    count=pd.to_numeric(g.candidate_count,errors='coerce')
    src_len=pd.to_numeric(g.source_length_m,errors='coerce')
    return pd.Series({
        'n':n,'matched':len(match),'unmatched':len(unmatched),
        'match_rate_pct':100*len(match)/n if n else np.nan,
        'source_length_km':float(src_len.sum()/1000),
        'matched_source_length_km':float(src_len[g.unified_proxy_label.eq(1)].sum()/1000),
        'matched_source_length_pct':100*float(src_len[g.unified_proxy_label.eq(1)].sum())/float(src_len.sum()) if src_len.sum()>0 else np.nan,
        'best_source_cov_median':quant(sc,.5),'best_source_cov_p10':quant(sc,.1),
        'best_source_cov_p90':quant(sc,.9),'best_target_cov_median':quant(tc,.5),
        'best_target_cov_p10':quant(tc,.1),'best_target_cov_p90':quant(tc,.9),
        'best_distance_median_m':quant(dist,.5),'best_distance_p90_m':quant(dist,.9),
        'candidate_count_median':quant(count,.5),'candidate_count_p90':quant(count,.9),
        'proximity_ambiguous_n':int((count>1).sum()),
        'proximity_ambiguous_pct':100*float((count>1).mean()) if n else np.nan,
        'matched_with_multiple_proximity_candidates_n':int((pd.to_numeric(match.candidate_count,errors='coerce')>1).sum()),
        'invalid_or_empty_n':int(g.invalid_or_empty_geometry.sum()),
    })

def main():
    if not INPUT.exists(): raise FileNotFoundError(f'Run Script 99 first: {INPUT}')
    OUT.mkdir(parents=True,exist_ok=True)
    header=pd.read_csv(INPUT,nrows=0).columns.tolist()
    missing=REQUIRED-set(header)
    if missing: raise ValueError(f'Script 99 output lacks columns: {sorted(missing)}')
    # Deliberately exclude original_label/disagreement and validation/test rows from analysis.
    fields=sorted(REQUIRED)
    frames=[]
    for chunk in pd.read_csv(INPUT,usecols=fields,chunksize=150000,low_memory=False):
        frames.append(chunk.loc[chunk['subset'].eq('train')].copy())
    if not frames: raise ValueError('Empty Script 99 record-level file')
    train=pd.concat(frames,ignore_index=True)
    if train.empty: raise ValueError('No train rows')
    if set(train.country.unique())!=EXPECTED_COUNTRIES: raise ValueError('Unexpected countries')
    if set(train.config.unique())!=EXPECTED_CONFIGS: raise ValueError('Unexpected configs')
    key=['country','city','source_input_row','config']
    if train.duplicated(key).any(): raise ValueError('Duplicate train source/config rows')
    for col in ['unified_proxy_label','source_length_m','best_source_coverage','best_target_coverage','best_distance_m','candidate_count']:
        train[col]=pd.to_numeric(train[col],errors='coerce')
    if not train.unified_proxy_label.isin([0,1]).all(): raise ValueError('Invalid proxy labels')
    if train.source_length_m.isna().any() or (train.source_length_m<=0).any(): raise ValueError('Invalid source lengths')
    train['invalid_or_empty_geometry']=train.invalid_or_empty_geometry.astype(str).str.lower().isin(['true','1'])
    if train.invalid_or_empty_geometry.any(): raise ValueError('Unexpected invalid training geometry; inspect Script 99')
    matched=train.unified_proxy_label.eq(1)
    if train.loc[matched,['best_source_coverage','best_target_coverage','best_distance_m']].isna().any().any():
        raise ValueError('Matched source missing geometric quality fields')
    for c in ['best_source_coverage','best_target_coverage']:
        if ((train.loc[matched,c]<0)|(train.loc[matched,c]>1.00001)).any(): raise ValueError('Coverage out of range')
    if (train.candidate_count<0).any(): raise ValueError('Negative candidate count')
    per_city=train.groupby(['country','city','config'],sort=True).apply(assessment,include_groups=False).reset_index()
    per_country=train.groupby(['country','config'],sort=True).apply(assessment,include_groups=False).reset_index()
    # Verify each configuration is evaluated on the exact same training source IDs.
    for (country,city),g in train.groupby(['country','city']):
        ids={cfg:set(zip(part.city,part.source_input_row)) for cfg,part in g.groupby('config')}
        if any(v!=next(iter(ids.values())) for v in ids.values()):
            raise ValueError(f'{country}/{city}: config populations differ')
    # Source-level transitions under increasing permissiveness; flag non-nested results.
    wide=train.pivot(index=['country','city','source_input_row'],columns='config',values='unified_proxy_label').reset_index()
    a=wide['C1_conservative']; b=wide['C2_intermediate']; c=wide['C3_permissive']
    if ((a>b)|(b>c)).any():
        raise ValueError('Matching not nested C1<=C2<=C3; inspect algorithm or input')
    wide['first_matched_config']=np.select([a.eq(1),b.eq(1),c.eq(1)],['C1_conservative','C2_intermediate','C3_permissive'],default='none')
    transitions=wide.groupby(['country','city','first_matched_config']).size().rename('roads').reset_index()
    country_transitions=wide.groupby(['country','first_matched_config']).size().rename('roads').reset_index()
    # A descriptive quality/coverage frontier, not an automated winner.
    # Increased match rate can reduce median coverage because extra weak matches enter.
    columns=['country','config','n','matched','match_rate_pct','matched_source_length_pct',
             'best_source_cov_median','best_target_cov_median','best_distance_p90_m',
             'proximity_ambiguous_pct']
    print('\nTRAIN-ONLY CORRESPONDENCE QUALITY / COVERAGE TRADE-OFF')
    print(per_country[columns].round(4).to_string(index=False))
    print('\nFIRST CONFIGURATION THAT MATCHES A ROAD (TRAIN ONLY)')
    print(country_transitions.to_string(index=False))
    per_city.to_csv(OUT/'train_city_geometric_quality.csv',index=False)
    per_country.to_csv(OUT/'train_country_geometric_quality.csv',index=False)
    transitions.to_csv(OUT/'train_city_matching_transitions.csv',index=False)
    country_transitions.to_csv(OUT/'train_country_matching_transitions.csv',index=False)
    # No source-level export with original labels, and no evaluation on held-out subsets.
    source_cfg=json.loads(CONFIG_FILE.read_text(encoding='utf-8')) if CONFIG_FILE.exists() else {}
    report={
        'script':'100_unified_correspondence_configuration_assessment.py',
        'input':str(INPUT),'scope':'train only',
        'configurations':source_cfg.get('configs',[]),
        'selected_configuration':None,
        'decision_status':'NOT SELECTED: requires predeclared cartographic criteria and independent validation',
        'limitations':[
            'Best coverage/distance statistics describe the selected accepted pair, not every candidate.',
            'candidate_count measures nearby target candidates, not the number of accepted correspondences.',
            'Geometric proximity does not establish correct topology, orientation or semantic identity.',
            'Matching-rate or label-agreement optimization alone cannot establish ground truth.',
            'The existing historical test data have been used in model development and are not untouched holdout data.',
        ]}
    (OUT/'assessment_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nOutputs:',OUT)
    print('NO CONFIGURATION AUTO-SELECTED. READ-ONLY. TEST/VALIDATION LABELS NOT READ.')

if __name__=='__main__': main()
