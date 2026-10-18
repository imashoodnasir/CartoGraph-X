#!/usr/bin/env python3
"""CartoGraph-X Script 94: read-only original-label versus geometric-support audit.

Run from project root: python scripts/94_original_vs_geometric_support_label_audit.py
No source files or labels are changed. Source-row alignment is explicitly checked.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
SUPPORT = DATA / 'cross_country' / 'geographic_coverage_road_support_audit_v1'
GRAPH = DATA / 'cross_country' / 'graph_representations' / 'road_graph_nodes.csv'
OUT = DATA / 'cross_country' / 'original_geometric_label_audit_v1'
CITIES = [('Lithuania','vilnius_central','Vilnius'),('Lithuania','kaunas_central','Kaunas'),('Lithuania','klaipeda_central','Klaipeda'),('Poland','warsaw','Warsaw'),('Poland','krakow','Krakow'),('Poland','gdansk','Gdansk')]

def must_unique(df, key, desc):
    if df[key].isna().any() or df[key].duplicated().any():
        raise ValueError(f'{desc}: null or duplicate key {key}; aborting unsafe join')

def safe_numeric(s): return pd.to_numeric(s, errors='coerce')

def normalize_label(series):
    return series.astype(str).str.lower().str.strip().map({'selected':1,'weak_selected':1,'elimination':0})

def run_city(country, city, city_label, graph):
    slug = f'{country.lower()}_{city}'
    sp = SUPPORT / f'{slug}_source_road_support.csv'
    if not sp.exists():
        raise FileNotFoundError(f'Missing Script 93 output: {sp}')
    s = pd.read_csv(sp)
    must_unique(s,'source_input_row',f'{slug} support')
    s['source_input_row'] = safe_numeric(s['source_input_row']).astype('Int64')
    if country == 'Lithuania':
        lp = DATA/'lithuania'/'learning_dataset'/'lithuania_road_learning_dataset.csv'
        l = pd.read_csv(lp)
        # Window is the unambiguous city key in this dataset.
        if 'window' not in l: raise ValueError('Lithuanian learning dataset lacks window')
        l = l[l['window'].astype(str).str.lower() == city.lower()].copy()
        idx='feature_index'; label='road_operation'
        if l.empty: raise ValueError(f'No learning records for {city} using window column')
        l['original_label'] = l[label].astype(str).str.lower().map({'selected':1,'elimination':0})
    else:
        lp = DATA/'poland'/'learning_dataset'/f'{city}_road_learning_dataset.csv'
        l = pd.read_csv(lp)
        idx='source_idx'; label='selection_label'
        l['original_label'] = normalize_label(l[label])
    l[idx] = safe_numeric(l[idx]).astype('Int64')
    must_unique(l,idx,f'{slug} learning')
    l = l.rename(columns={idx:'source_input_row',label:'original_label_text'})
    fields=['source_input_row','original_label','original_label_text']
    fields += [c for c in ['length_m','centroid_x','centroid_y'] if c in l.columns]
    merged=s.merge(l[fields],on='source_input_row',how='outer',indicator='learning_join',validate='one_to_one')
    # Check source index alignment against original source geometry length.
    merged['length_abs_error_m'] = (safe_numeric(merged['source_feature_length_m']) - safe_numeric(merged['length_m'])).abs() if 'length_m' in merged else np.nan
    merged['length_relative_error'] = merged['length_abs_error_m'] / safe_numeric(merged['source_feature_length_m']).clip(lower=1.0)
    merged['length_consistent'] = (merged['length_abs_error_m'] <= 0.10) | (merged['length_relative_error'] <= 0.001)
    merged['length_consistent'] = merged['length_consistent'].fillna(False)
    merged['learning_identity_verified_by_length'] = merged['learning_join'].eq('both') & merged['length_consistent']
    # A graph-node join is independently audited; no assumption that graph includes all source roads.
    g=graph[(graph['country'].astype(str).str.lower()==country.lower()) & (graph['city'].astype(str).str.lower()==city_label.lower())].copy()
    g['source_input_row']=safe_numeric(g['source_idx_harmonized']).astype('Int64')
    must_unique(g,'source_input_row',f'{slug} graph')
    gfields=['source_input_row','sample_uid','retained','selection_label_harmonized','split','length_m','centroid_x','centroid_y']
    g=g[gfields].rename(columns={'length_m':'graph_length_m','centroid_x':'graph_centroid_x','centroid_y':'graph_centroid_y'})
    merged=merged.merge(g,on='source_input_row',how='left',validate='one_to_one',indicator='graph_join')
    merged['graph_length_abs_error_m']=(safe_numeric(merged['source_feature_length_m'])-safe_numeric(merged['graph_length_m'])).abs()
    merged['graph_length_consistent']=((merged['graph_length_abs_error_m']<=0.10)|(merged['graph_length_abs_error_m']/safe_numeric(merged['source_feature_length_m']).clip(lower=1)<=0.001)).fillna(False)
    merged['graph_label_agrees_learning']=safe_numeric(merged['retained']).eq(safe_numeric(merged['original_label']))
    # Geometric evidence is NOT ground truth. Avoid calling these confusion matrices.
    frac=safe_numeric(merged['supported_fraction_15m'])
    merged['support_15m_band']=pd.cut(frac,[-0.001,0.05,0.20,0.50,0.80,1.000001],labels=['0-5%','5-20%','20-50%','50-80%','80-100%'])
    merged['potential_disagreement']=np.select([
        merged['original_label'].eq(1)&frac.lt(0.05),
        merged['original_label'].eq(0)&frac.ge(0.80),
        merged['original_label'].eq(1)&frac.lt(0.20),
        merged['original_label'].eq(0)&frac.ge(0.50),
    ],['positive_with_very_low_support','negative_with_very_high_support','positive_with_low_support','negative_with_high_support'],default='no_priority_flag')
    merged['safe_to_compare']=merged['learning_identity_verified_by_length'] & merged['original_label'].isin([0,1])
    merged.loc[~merged['safe_to_compare'],'potential_disagreement']='identity_or_label_unverified'
    merged['country']=country;merged['city']=city
    merged.to_csv(OUT/f'{slug}_original_vs_support.csv',index=False)
    eligible=merged[merged['safe_to_compare']]
    summary={'country':country,'city':city,'support_rows':int(len(s)),'learning_rows':int(len(l)),
             'graph_rows':int(len(g)),'learning_join_both':int(merged['learning_join'].eq('both').sum()),
             'learning_length_consistent':int(merged['learning_identity_verified_by_length'].sum()),
             'graph_join_both':int(merged['graph_join'].eq('both').sum()),
             'graph_length_consistent':int((merged['graph_join'].eq('both')&merged['graph_length_consistent']).sum()),
             'graph_label_disagreements':int((merged['graph_join'].eq('both')&merged['safe_to_compare']&~merged['graph_label_agrees_learning']).sum()),
             'eligible_for_label_support_comparison':int(len(eligible)),
             'original_positive':int(eligible['original_label'].eq(1).sum()),
             'original_negative':int(eligible['original_label'].eq(0).sum()),
             'positive_very_low_support':int(eligible['potential_disagreement'].eq('positive_with_very_low_support').sum()),
             'negative_very_high_support':int(eligible['potential_disagreement'].eq('negative_with_very_high_support').sum()),
             'bbox_outside_any':int(s['bbox_outside_any'].fillna(False).astype(bool).sum()),
             'coverage_verified':bool(s['coverage_verified'].fillna(False).astype(bool).all())}
    cross=eligible.groupby(['original_label','support_15m_band'],observed=True).size().reset_index(name='count')
    cross.insert(0,'city',city);cross.insert(0,'country',country)
    priority=eligible[eligible['potential_disagreement'].ne('no_priority_flag')].copy()
    priority=priority.sort_values(['potential_disagreement','source_input_row'])
    priority.to_csv(OUT/f'{slug}_manual_review_candidates.csv',index=False)
    print(f"{country:10s} {city:18s} source={len(s):5d} learning={len(l):5d} verified_by_length={summary['learning_length_consistent']:5d} graph={len(g):5d} graph_label_disagree={summary['graph_label_disagreements']:4d} flagged={len(priority):5d}")
    return summary,cross

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cities',nargs='*',help='Optional city keys, e.g. kaunas_central warsaw')
    args=ap.parse_args()
    if not GRAPH.exists(): raise FileNotFoundError(GRAPH)
    OUT.mkdir(parents=True,exist_ok=True)
    graph=pd.read_csv(GRAPH)
    summaries=[];crosses=[]
    for country,city,city_label in CITIES:
        if args.cities and city not in args.cities: continue
        a,b=run_city(country,city,city_label,graph);summaries.append(a);crosses.append(b)
    if not summaries: raise ValueError('No cities selected')
    pd.DataFrame(summaries).to_csv(OUT/'city_label_identity_summary.csv',index=False)
    pd.concat(crosses,ignore_index=True).to_csv(OUT/'original_label_vs_15m_support_bands.csv',index=False)
    (OUT/'audit_metadata.json').write_text(json.dumps({'purpose':'diagnostic only; no relabeling','identity_check':'source row index plus road length, absolute <=0.10m or relative <=0.1%','coverage_verified':'not inferred from bounding boxes','support_15m':'geometric proximity, not ground truth','input_graph':str(GRAPH),'input_support':str(SUPPORT)},indent=2),encoding='utf-8')
    print('\nRead-only outputs:',OUT)
    print('WARNING: Low support does not prove elimination; high support does not prove retention.')
    print('WARNING: Row+length agreement is a diagnostic, not a substitute for immutable source feature IDs.')

if __name__=='__main__': main()
