#!/usr/bin/env python3
"""CartoGraph-X Script 99: read-only cross-country road correspondence sensitivity.

No existing benchmark, labels, splits or model outputs are changed. Test labels are
reported for audit only; they must NOT be used to select matching thresholds.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data' / 'processed'
OUT = BASE / 'cross_country' / 'benchmark_v2_unified_correspondence_audit_v1'
GRAPH = BASE / 'cross_country' / 'graph_representations' / 'road_graph_nodes.csv'
CITIES = [
 ('Lithuania','vilnius_central','Vilnius',BASE/'lithuania'/'validated_pairs'/'vilnius_central_validated.gpkg'),
 ('Lithuania','kaunas_central','Kaunas',BASE/'lithuania'/'validated_pairs'/'kaunas_central_validated.gpkg'),
 ('Lithuania','klaipeda_central','Klaipeda',BASE/'lithuania'/'validated_pairs'/'klaipeda_central_validated.gpkg'),
 ('Poland','warsaw','Warsaw',BASE/'poland'/'road_target_validation'/'warsaw_validated_road_pair.gpkg'),
 ('Poland','krakow','Krakow',BASE/'poland'/'road_target_validation'/'krakow_validated_road_pair.gpkg'),
 ('Poland','gdansk','Gdansk',BASE/'poland'/'road_target_validation'/'gdansk_validated_road_pair.gpkg'),
]
# Unified geometric correspondence rules. Not equivalent to either original
# country's correspondence scoring (which may include orientation/semantics).
CONFIGS = [
 dict(config='C1_conservative',search_m=10.,buffer_m=4.,source_min=.35,target_min=.05),
 dict(config='C2_intermediate',search_m=15.,buffer_m=5.,source_min=.20,target_min=.03),
 dict(config='C3_permissive',search_m=25.,buffer_m=8.,source_min=.08,target_min=.02),
]

def num(s): return pd.to_numeric(s,errors='coerce')

def clean_lines(gdf):
    # Preserve original row positions; invalid/empty geometries are flagged, never silently reindexed.
    if gdf.crs is None or not gdf.crs.is_projected:
        raise ValueError('Expected a projected CRS with metric units; got '+str(gdf.crs))
    if gdf.crs.axis_info and any(a.unit_name.lower() not in ('metre','meter','metres','meters') for a in gdf.crs.axis_info[:2]):
        raise ValueError('Expected metre CRS: '+str(gdf.crs))
    return gdf.reset_index(drop=True)

def audit_city(country, citykey, citydisplay, gpkg, graph):
    if not gpkg.exists(): raise FileNotFoundError(gpkg)
    src=clean_lines(gpd.read_file(gpkg,layer='source_roads'))
    tgt=clean_lines(gpd.read_file(gpkg,layer='target_roads'))
    if src.crs != tgt.crs: tgt=tgt.to_crs(src.crs)
    gg=graph[(graph.country==country)&(graph.city.astype(str).str.lower()==citydisplay.lower())].copy()
    gg['source_input_row']=num(gg.source_idx_harmonized)
    if gg.source_input_row.isna().any() or gg.source_input_row.duplicated().any():
        raise ValueError(f'{citykey}: nonunique/non-numeric graph source IDs')
    gg['source_input_row']=gg.source_input_row.astype(int)
    if (gg.source_input_row<0).any() or (gg.source_input_row>=len(src)).any():
        raise ValueError(f'{citykey}: graph IDs outside geometry row range')
    lookup=gg.set_index('source_input_row')
    # Train/validation/test metadata are joined only by verified original source row.
    lengths=np.asarray(src.geometry.length,dtype=float)
    check=lookup['length_m'].astype(float).to_numpy()
    actual=lengths[lookup.index.to_numpy()]
    if not np.allclose(check,actual,rtol=1e-3,atol=.1):
        raise ValueError(f'{citykey}: graph/source geometry length mismatch; audit stopped')
    labels=np.full(len(src),np.nan)
    labels[lookup.index.to_numpy()]=num(lookup.retained).to_numpy()
    splits=np.full(len(src),'excluded',dtype=object)
    splits[lookup.index.to_numpy()]=lookup.split.astype(str).to_numpy()
    tiles=np.full(len(src),'',dtype=object)
    tiles[lookup.index.to_numpy()]=lookup.tile_id.astype(str).to_numpy()
    # Use the same spatial candidate pool for all configs, then apply per-config rules.
    max_search=max(c['search_m'] for c in CONFIGS)
    max_buffer=max(c['buffer_m'] for c in CONFIGS)
    target_geoms=tgt.geometry.to_list()
    sindex=tgt.sindex
    pair_records=[]
    empty_invalid=0
    for i,geom in enumerate(src.geometry):
        if geom is None or geom.is_empty or not geom.is_valid or geom.length<=0:
            empty_invalid+=1
            continue
        # Candidate search uses bounding-box intersection with source buffer.
        candidates=sindex.query(geom.buffer(max_search+max_buffer))
        for j in candidates:
            j=int(j)
            t=target_geoms[j]
            if t is None or t.is_empty or not t.is_valid or t.length<=0: continue
            distance=float(geom.distance(t))
            if distance>max_search: continue
            # Length of each road lying within the opposite road's buffer.
            # Coverage is directional; never use polygon area for line matching.
            sc=float(geom.intersection(t.buffer(max_buffer)).length / geom.length)
            tc=float(t.intersection(geom.buffer(max_buffer)).length / t.length)
            pair_records.append((i,j,distance,sc,tc))
        if (i+1)%1500==0:
            print(f'  {citykey}: scanned {i+1:,}/{len(src):,} sources',flush=True)
    pairs=pd.DataFrame(pair_records,columns=['source_input_row','target_input_row','distance_m','source_cov_maxbuffer','target_cov_maxbuffer'])
    # Recompute coverage at each config buffer; maximum-buffer coverage cannot be reused.
    config_rows=[]; record_rows=[]
    for c in CONFIGS:
        candidate=pairs[pairs.distance_m<=c['search_m']].copy()
        matched=np.zeros(len(src),dtype=bool)
        best_sc=np.zeros(len(src)); best_tc=np.zeros(len(src)); best_distance=np.full(len(src),np.nan)
        candidate_n=np.zeros(len(src),dtype=int)
        if len(candidate):
            # Per-source best accepted candidate by source coverage then target coverage.
            for k,(i,j,dist) in enumerate(candidate[['source_input_row','target_input_row','distance_m']].itertuples(index=False,name=None)):
                i=int(i);j=int(j)
                sg=src.geometry.iloc[i]; tg=target_geoms[j]
                sc=min(1.,float(sg.intersection(tg.buffer(c['buffer_m'])).length/sg.length))
                tc=min(1.,float(tg.intersection(sg.buffer(c['buffer_m'])).length/tg.length))
                candidate_n[i]+=1
                if sc>=c['source_min'] and tc>=c['target_min']:
                    if (not matched[i]) or (sc,tc,-dist)>(best_sc[i],best_tc[i],-best_distance[i]):
                        matched[i]=True;best_sc[i]=sc;best_tc[i]=tc;best_distance[i]=dist
        for i in range(len(src)):
            record_rows.append(dict(country=country,city=citykey,source_input_row=i,config=c['config'],
                subset=splits[i],tile_id=tiles[i],original_label=labels[i],unified_proxy_label=int(matched[i]),
                disagreement=(int(labels[i])!=int(matched[i])) if np.isfinite(labels[i]) else np.nan,
                source_length_m=lengths[i],best_source_coverage=best_sc[i],best_target_coverage=best_tc[i],
                best_distance_m=best_distance[i],candidate_count=int(candidate_n[i]),
                invalid_or_empty_geometry=bool(src.geometry.iloc[i] is None or src.geometry.iloc[i].is_empty or not src.geometry.iloc[i].is_valid)))
        for subset in ['train','validation','test','excluded']:
            ix=splits==subset
            n=int(ix.sum()); valid=ix & np.isfinite(labels)
            disagreement=int((matched[valid]!=labels[valid]).sum())
            config_rows.append(dict(country=country,city=citykey,config=c['config'],subset=subset,n=n,
                original_positive=int((labels[valid]==1).sum()),unified_proxy_positive=int(matched[ix].sum()),
                disagreements=disagreement if valid.sum() else np.nan,
                disagreement_pct=100*disagreement/int(valid.sum()) if valid.sum() else np.nan,
                source_positive_proxy_negative=int(((labels[valid]==1)&(~matched[valid])).sum()),
                source_negative_proxy_positive=int(((labels[valid]==0)&matched[valid]).sum())))
    print(f'{country:10s} {citykey:18s} source={len(src):5,d} target={len(tgt):5,d} graph={len(gg):5,d} invalid_source={empty_invalid}',flush=True)
    return pd.DataFrame(record_rows),pd.DataFrame(config_rows),dict(country=country,city=citykey,source=len(src),target=len(tgt),graph=len(gg),invalid_source=empty_invalid,pair_candidates=len(pairs))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if not GRAPH.exists(): raise FileNotFoundError(GRAPH)
    graph=pd.read_csv(GRAPH,low_memory=False)
    needed={'country','city','source_idx_harmonized','length_m','retained','split','tile_id'}
    if needed-set(graph.columns): raise ValueError(f'Missing graph columns: {needed-set(graph.columns)}')
    all_records=[];all_summaries=[];inventory=[]
    for args in CITIES:
        r,s,inv=audit_city(*args,graph)
        all_records.append(r);all_summaries.append(s);inventory.append(inv)
        r.to_csv(OUT/f'{args[0].lower()}_{args[1]}_correspondence_sensitivity.csv',index=False)
    records=pd.concat(all_records,ignore_index=True)
    summary=pd.concat(all_summaries,ignore_index=True)
    records.to_csv(OUT/'record_level_correspondence_sensitivity.csv',index=False)
    summary.to_csv(OUT/'city_split_correspondence_summary.csv',index=False)
    country=(summary.groupby(['country','config','subset'],as_index=False)[['n','original_positive','unified_proxy_positive','disagreements','source_positive_proxy_negative','source_negative_proxy_positive']].sum(min_count=1))
    country['disagreement_pct']=100*country['disagreements']/country['n'].replace(0,np.nan)
    country.to_csv(OUT/'country_split_correspondence_summary.csv',index=False)
    pd.DataFrame(inventory).to_csv(OUT/'input_geometry_inventory.csv',index=False)
    (OUT/'configurations.json').write_text(json.dumps({'configs':CONFIGS,'warning':'Geometric correspondence proxy only; no original label changes. Do not choose thresholds using test labels.'},indent=2),encoding='utf-8')
    print('\nCOUNTRY SUMMARY (TRAIN ONLY; threshold selection must not use test labels)')
    print(country[country.subset=='train'][['country','config','n','original_positive','unified_proxy_positive','disagreements','disagreement_pct']].to_string(index=False))
    print('\nOutputs:',OUT)
    print('READ-ONLY: no label edits, no train/test threshold selection, no changes to existing benchmark.')
if __name__=='__main__': main()
