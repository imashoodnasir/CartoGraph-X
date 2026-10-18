#!/usr/bin/env python3
"""CartoGraph-X Script 101: TRAIN-ONLY correspondence verification and review sample.

READ-ONLY. Uses Script 99 records and the six validated source GPKGs.
Does not choose thresholds, change labels, or inspect validation/test records.
Candidate_count measures proximity candidates, NOT accepted-pair ambiguity.
Source-only GIS review cannot establish a correct source-to-target pairing.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data'/'processed'
INPUT=BASE/'cross_country'/'benchmark_v2_unified_correspondence_audit_v1'/'record_level_correspondence_sensitivity.csv'
OUT=BASE/'cross_country'/'benchmark_v2_correspondence_verification_v1'
CONFIGS=['C1_conservative','C2_intermediate','C3_permissive']
CITY_PATHS={
 ('Lithuania','vilnius_central'):BASE/'lithuania'/'validated_pairs'/'vilnius_central_validated.gpkg',
 ('Lithuania','kaunas_central'):BASE/'lithuania'/'validated_pairs'/'kaunas_central_validated.gpkg',
 ('Lithuania','klaipeda_central'):BASE/'lithuania'/'validated_pairs'/'klaipeda_central_validated.gpkg',
 ('Poland','warsaw'):BASE/'poland'/'road_target_validation'/'warsaw_validated_road_pair.gpkg',
 ('Poland','krakow'):BASE/'poland'/'road_target_validation'/'krakow_validated_road_pair.gpkg',
 ('Poland','gdansk'):BASE/'poland'/'road_target_validation'/'gdansk_validated_road_pair.gpkg',
}
COLUMNS=['country','city','source_input_row','subset','config','unified_proxy_label',
         'best_source_coverage','best_target_coverage','best_distance_m','candidate_count',
         'source_length_m','invalid_or_empty_geometry']
SEED=101
SAMPLE_PER_CITY_CATEGORY=15

def q(x,p):
    s=pd.to_numeric(x,errors='coerce').dropna()
    return float(s.quantile(p)) if len(s) else None

def main():
    if not INPUT.is_file(): raise FileNotFoundError(f'Run Script 99 first: {INPUT}')
    OUT.mkdir(parents=True,exist_ok=True)
    header=set(pd.read_csv(INPUT,nrows=0).columns)
    missing=set(COLUMNS)-header
    if missing: raise ValueError(f'Missing required Script 99 fields: {sorted(missing)}')
    parts=[]
    for chunk in pd.read_csv(INPUT,usecols=COLUMNS,chunksize=120000,low_memory=False):
        parts.append(chunk.loc[chunk.subset.eq('train')].copy())
    if not parts: raise ValueError('No records')
    d=pd.concat(parts,ignore_index=True)
    if d.empty: raise ValueError('No train records')
    d['source_input_row']=pd.to_numeric(d.source_input_row,errors='raise').astype(int)
    d['unified_proxy_label']=pd.to_numeric(d.unified_proxy_label,errors='raise').astype(int)
    if d.duplicated(['country','city','source_input_row','config']).any(): raise ValueError('Duplicate road/config rows')
    if set(d.config.unique())!=set(CONFIGS): raise ValueError('Unexpected or missing configurations')
    key=['country','city','source_input_row']
    counts=d.groupby(key).config.nunique()
    if (counts!=3).any(): raise ValueError('Incomplete per-road configurations')
    wide=d.pivot(index=key,columns='config',values='unified_proxy_label')
    if not ((wide[CONFIGS[0]]<=wide[CONFIGS[1]]) & (wide[CONFIGS[1]]<=wide[CONFIGS[2]])).all():
        raise ValueError('Nonmonotonic matching: review Script 99 configuration nesting')
    wide['transition']=np.select([
        wide[CONFIGS[0]].eq(1),
        wide[CONFIGS[0]].eq(0)&wide[CONFIGS[1]].eq(1),
        wide[CONFIGS[1]].eq(0)&wide[CONFIGS[2]].eq(1),
    ],['C1_already_matched','C2_new_match','C3_only_new_match'],default='unmatched_all')
    base=d[d.config.eq(CONFIGS[1])].set_index(key)
    wide=wide.join(base[['source_length_m','candidate_count','best_source_coverage',
                         'best_target_coverage','best_distance_m']],validate='one_to_one')
    wide=wide.reset_index()
    wide['review_category']=wide['transition']
    # High proximity candidate count is an inspection priority, not accepted-pair ambiguity.
    wide['proximity_multiple_C2']=pd.to_numeric(wide.candidate_count,errors='coerce').gt(1)
    wide.to_csv(OUT/'train_road_transitions.csv',index=False)
    stats=[]
    for (country,city,config),g in d.groupby(['country','city','config'],sort=True):
        m=g[g.unified_proxy_label.eq(1)]
        stats.append(dict(country=country,city=city,config=config,n=len(g),matched=len(m),
            matched_pct=100*len(m)/len(g),source_coverage_p10=q(m.best_source_coverage,.1),
            source_coverage_median=q(m.best_source_coverage,.5),
            target_coverage_p10=q(m.best_target_coverage,.1),
            target_coverage_median=q(m.best_target_coverage,.5),
            distance_p90_m=q(m.best_distance_m,.9),
            proximity_multiple_pct=100*float(pd.to_numeric(g.candidate_count,errors='coerce').gt(1).mean()),
            matched_proximity_multiple_pct=100*float(pd.to_numeric(m.candidate_count,errors='coerce').gt(1).mean()) if len(m) else None))
    pd.DataFrame(stats).to_csv(OUT/'train_city_configuration_quality.csv',index=False)
    trans=(wide.groupby(['country','city','transition'],as_index=False).size().rename(columns={'size':'roads'}))
    trans.to_csv(OUT/'train_city_transition_counts.csv',index=False)
    # Deterministic, stratified sample with all categories represented when available.
    # Priority sorting for C2 matches uses low target coverage and multiple proximity candidates.
    rng=np.random.default_rng(SEED)
    sampled=[]
    for (country,city,cat),g in wide.groupby(['country','city','review_category'],sort=True):
        n=min(SAMPLE_PER_CITY_CATEGORY,len(g))
        if n==0: continue
        if cat in ('C1_already_matched','C2_new_match'):
            g=g.sort_values(['best_target_coverage','candidate_count','source_input_row'],ascending=[True,False,True])
            priority=g.head(min((n+1)//2,len(g)))
            rest=g.drop(priority.index)
            random_rows=rest.iloc[rng.choice(len(rest),size=min(n-len(priority),len(rest)),replace=False)] if len(rest) else rest
            sample=pd.concat([priority,random_rows])
        else:
            sample=g.iloc[rng.choice(len(g),size=n,replace=False)]
        sampled.append(sample.copy())
    review=pd.concat(sampled,ignore_index=True)
    review['review_status']='unreviewed'
    review['source_target_match_valid']=''
    review['review_notes']=''
    review.to_csv(OUT/'manual_review_queue.csv',index=False)
    # A single source-road geometry layer per country avoids mixing projected CRSs.
    # Review alongside the target_roads layer of each original validated GPKG.
    try:
        import geopandas as gpd
        for (country,city),group in review.groupby(['country','city'],sort=True):
            path=CITY_PATHS.get((country,city))
            if path is None or not path.exists(): raise FileNotFoundError(path)
            source=gpd.read_file(path,layer='source_roads').reset_index(drop=True)
            idx=group.source_input_row.to_numpy(dtype=int)
            if (idx<0).any() or (idx>=len(source)).any(): raise ValueError(f'Invalid source index for {city}')
            geometry=source.geometry.iloc[idx].reset_index(drop=True)
            attrs=group.reset_index(drop=True).copy()
            geo=gpd.GeoDataFrame(attrs,geometry=geometry,crs=source.crs)
            geo.to_file(OUT/f'{country.lower()}_{city}_review_sources.gpkg',layer='review_sources',driver='GPKG')
    except ImportError:
        print('geopandas unavailable: CSV review queue created; GIS layers skipped.')
    manifest={'script':101,'seed':SEED,'train_only':True,'sample_per_city_category':SAMPLE_PER_CITY_CATEGORY,
       'rows_train':int(len(wide)),'review_rows':int(len(review)),
       'limitations':['Candidate count is proximity-based, not accepted-match ambiguity.',
                      'Best-pair statistics do not identify the matched target feature.',
                      'GIS layers contain source roads only; open original validated target_roads layer for comparison.',
                      'Manual validation is needed before freezing correspondence rules.',
                      'No validation/test records or labels were read.']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('\nTRAIN-ONLY MATCHING TRANSITIONS')
    print(wide.groupby(['country','transition']).size().to_string())
    print('\nTRAIN-ONLY GEOMETRIC QUALITY')
    print(pd.DataFrame(stats).groupby(['country','config'],as_index=False).agg(
        roads=('n','sum'),matched=('matched','sum'),
        median_city_target_coverage=('target_coverage_median','median'),
        median_city_proximity_multiple_pct=('proximity_multiple_pct','median')).to_string(index=False))
    print(f'\nManual review source roads: {len(review):,}')
    print('Outputs:',OUT)
    print('READ-ONLY. TRAIN ONLY. No configuration selected; no label modifications.')

if __name__=='__main__': main()
