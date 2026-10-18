#!/usr/bin/env python3
"""CartoGraph-X Script 95: read-only benchmark inclusion/exclusion integrity audit.

Run: python scripts/95_benchmark_integrity_exclusion_audit.py
Inputs: Script 94 per-city CSVs and graph_representations/road_graph_nodes.csv.
Does not modify labels, graph, partitions, or any original data.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data' / 'processed' / 'cross_country'
AUDIT94 = BASE / 'original_geometric_label_audit_v1'
GRAPH_PATH = BASE / 'graph_representations' / 'road_graph_nodes.csv'
OUT = BASE / 'benchmark_integrity_exclusion_audit_v1'
CITIES = [('Lithuania','vilnius_central','Vilnius'),('Lithuania','kaunas_central','Kaunas'),('Lithuania','klaipeda_central','Klaipeda'),('Poland','warsaw','Warsaw'),('Poland','krakow','Krakow'),('Poland','gdansk','Gdansk')]

def numeric(s): return pd.to_numeric(s,errors='coerce')
def check_unique(df,cols,what):
    if df[cols].isna().any().any() or df.duplicated(cols).any():
        raise ValueError(f'{what}: null/duplicate identifier {cols}')
def safe_bool(s): return s.astype(str).str.lower().eq('true')
def rate(n,d): return round(100*n/d,3) if d else np.nan

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cities',nargs='*',help='City keys; default all six')
    args=ap.parse_args()
    if not GRAPH_PATH.is_file(): raise FileNotFoundError(GRAPH_PATH)
    graph=pd.read_csv(GRAPH_PATH)
    required=['sample_uid','country','city','source_idx_harmonized','split','tile_id','retained','length_m','centroid_x','centroid_y']
    missing=set(required)-set(graph.columns)
    if missing: raise ValueError(f'Graph missing columns: {sorted(missing)}')
    check_unique(graph,['sample_uid'],'graph')
    OUT.mkdir(parents=True,exist_ok=True)
    city_summaries=[]; details=[]; bylabel=[]; bysupport=[]; bysplit=[]; bytile=[]; split_issues=[]
    for country,city,city_display in CITIES:
        if args.cities and city not in args.cities: continue
        slug=f'{country.lower()}_{city}'
        path=AUDIT94/f'{slug}_original_vs_support.csv'
        if not path.is_file(): raise FileNotFoundError(f'Run Script 94 first: {path}')
        d=pd.read_csv(path)
        for col in ['source_input_row','original_label','source_feature_length_m','supported_fraction_15m','potential_disagreement','learning_identity_verified_by_length','graph_join','graph_length_consistent','graph_label_agrees_learning','bbox_outside_any']:
            if col not in d: raise ValueError(f'{path}: missing {col}')
        check_unique(d,['source_input_row'],slug)
        d['included_in_graph']=d['graph_join'].eq('both')
        d['original_label']=numeric(d['original_label'])
        d['support_fraction_15m']=numeric(d['supported_fraction_15m'])
        d['source_feature_length_m']=numeric(d['source_feature_length_m'])
        d['review_flag']=~d['potential_disagreement'].isin(['no_priority_flag','identity_or_label_unverified'])
        d['length_identity_ok']=safe_bool(d['learning_identity_verified_by_length'])
        d['graph_length_ok']=safe_bool(d['graph_length_consistent'])
        d['graph_label_ok']=safe_bool(d['graph_label_agrees_learning'])
        d['bbox_outside_any_bool']=safe_bool(d['bbox_outside_any'])
        d['support_band']=pd.cut(d['support_fraction_15m'],[-0.001,0.05,0.2,0.5,0.8,1.000001],labels=['0-5%','5-20%','20-50%','50-80%','80-100%'])
        # Graph metadata from Script94's audited one-to-one join; not fabricated for excluded records.
        d['country']=country;d['city']=city
        included=d['included_in_graph']; excluded=~included
        if included.sum() != d['sample_uid'].notna().sum(): raise ValueError(f'{slug}: graph join / sample UID mismatch')
        if (included & ~d['graph_length_ok']).any():
            raise ValueError(f'{slug}: included graph length mismatch; investigate before downstream work')
        if (included & ~d['graph_label_ok']).any():
            raise ValueError(f'{slug}: graph labels disagree with learning dataset')
        if not d['length_identity_ok'].all():
            raise ValueError(f'{slug}: original source-learning identity check failed')
        # Check independently against graph table, including split and tile fields.
        g=graph[(graph.country.astype(str).str.casefold()==country.casefold()) & (graph.city.astype(str).str.casefold()==city_display.casefold())].copy()
        check_unique(g,['source_idx_harmonized'],slug+' graph')
        ids=set(numeric(g.source_idx_harmonized).dropna().astype(int))
        observed=set(numeric(d.loc[included,'source_input_row']).astype(int))
        if ids!=observed: raise ValueError(f'{slug}: Script94 join differs from graph node identifiers')
        # Script 94 may omit tile_id (and sometimes split) from its per-road CSV.
        # Recover authoritative partition metadata from the graph using validated source IDs.
        graph_meta=g[['source_idx_harmonized','sample_uid','split','tile_id']].copy()
        graph_meta['source_idx_harmonized']=numeric(graph_meta['source_idx_harmonized']).astype('int64')
        d['_audit_source_key']=numeric(d['source_input_row']).astype('int64')
        graph_meta=graph_meta.rename(columns={'source_idx_harmonized':'_audit_source_key'})
        d=d.merge(graph_meta, on='_audit_source_key', how='left', validate='one_to_one', suffixes=('','_graph95'))
        for field in ['sample_uid','split','tile_id']:
            verified_field=field+'_graph95'
            if verified_field in d.columns:
                if field in d.columns:
                    both=d[field].notna() & d[verified_field].notna()
                    inconsistent=both & (d[field].astype(str)!=d[verified_field].astype(str))
                    if inconsistent.any():
                        raise ValueError(f'{slug}: {field} conflicts with authoritative graph for {int(inconsistent.sum())} rows')
                    d[field]=d[verified_field].combine_first(d[field])
                else:
                    d[field]=d[verified_field]
                d=d.drop(columns=[verified_field])
        if d.loc[included,['sample_uid','split','tile_id']].isna().any().any():
            raise ValueError(f'{slug}: missing authoritative graph metadata on included rows')
        if d.loc[excluded,['split','tile_id']].notna().any().any():
            raise ValueError(f'{slug}: excluded rows unexpectedly have graph split/tile assignments')
        d['inclusion_status']=np.where(included,'included','excluded')
        d['original_label_name']=d['original_label'].map({0:'negative',1:'positive'}).fillna('unverified')
        # Record-level reports preserve only already observed source and graph fields.
        keep=['country','city','source_input_row','inclusion_status','original_label','original_label_text',
              'source_feature_length_m','support_fraction_15m','support_band','potential_disagreement',
              'review_flag','bbox_outside_any_bool','evidence_tier','sample_uid','split','tile_id',
              'centroid_x','centroid_y','length_identity_ok','graph_length_ok','graph_label_ok']
        keep=[c for c in keep if c in d]
        details.append(d[keep].copy())
        n=len(d); ni=int(included.sum()); ne=int(excluded.sum())
        row={'country':country,'city':city,'source_rows':n,'included_graph_rows':ni,'excluded_rows':ne,
             'excluded_pct':rate(ne,n),'positive_all':int(d.original_label.eq(1).sum()),
             'positive_included':int((included & d.original_label.eq(1)).sum()),
             'positive_excluded':int((excluded & d.original_label.eq(1)).sum()),
             'positive_rate_all_pct':rate(int(d.original_label.eq(1).sum()),n),
             'positive_rate_included_pct':rate(int((included & d.original_label.eq(1)).sum()),ni),
             'positive_rate_excluded_pct':rate(int((excluded & d.original_label.eq(1)).sum()),ne),
             'review_all':int(d.review_flag.sum()),'review_included':int((included & d.review_flag).sum()),
             'review_excluded':int((excluded & d.review_flag).sum()),
             'review_rate_included_pct':rate(int((included & d.review_flag).sum()),ni),
             'review_rate_excluded_pct':rate(int((excluded & d.review_flag).sum()),ne),
             'bbox_outside_excluded':int((excluded & d.bbox_outside_any_bool).sum()),
             'source_length_excluded_pct':rate(float(d.loc[excluded,'source_feature_length_m'].sum()),float(d.source_feature_length_m.sum())),
             'mean_support_included':float(d.loc[included,'support_fraction_15m'].mean()),
             'mean_support_excluded':float(d.loc[excluded,'support_fraction_15m'].mean())}
        city_summaries.append(row)
        for label,part in d.groupby('original_label_name',dropna=False):
            bylabel.append({'country':country,'city':city,'label':label,'total':len(part),
                            'included':int(part.included_in_graph.sum()),'excluded':int((~part.included_in_graph).sum()),
                            'excluded_pct':rate(int((~part.included_in_graph).sum()),len(part))})
        for band,part in d.groupby('support_band',observed=True):
            bysupport.append({'country':country,'city':city,'support_band':str(band),'total':len(part),
                              'included':int(part.included_in_graph.sum()),'excluded':int((~part.included_in_graph).sum()),
                              'review_count':int(part.review_flag.sum())})
        for split,part in d.loc[included].groupby('split',dropna=False):
            bysplit.append({'country':country,'city':city,'split':split,'rows':len(part),
                            'positive':int(part.original_label.eq(1).sum()),'positive_rate_pct':rate(int(part.original_label.eq(1).sum()),len(part)),
                            'review':int(part.review_flag.sum()),'review_rate_pct':rate(int(part.review_flag.sum()),len(part)),
                            'mean_support_15m':float(part.support_fraction_15m.mean())})
        for (split,tile),part in d.loc[included].groupby(['split','tile_id'],dropna=False):
            bytile.append({'country':country,'city':city,'split':split,'tile_id':tile,'rows':len(part),
                           'positive':int(part.original_label.eq(1).sum()),'review':int(part.review_flag.sum()),
                           'review_rate_pct':rate(int(part.review_flag.sum()),len(part))})
        # Leakage check: a tile must not span different partitions within the same country/city.
        tile_splits=g.groupby('tile_id',dropna=False)['split'].nunique(dropna=False)
        for tile in tile_splits[tile_splits>1].index:
            split_issues.append({'country':country,'city':city,'tile_id':tile,'splits':';'.join(sorted(g.loc[g.tile_id.eq(tile),'split'].astype(str).unique()))})
        print(f'{country:10s} {city:18s} included={ni:5d} excluded={ne:5d} ({rate(ne,n):5.1f}%) '
              f'positive[in/ex]={row["positive_rate_included_pct"]:5.1f}/{row["positive_rate_excluded_pct"]:5.1f}% '
              f'review[in/ex]={row["review_included"]:3d}/{row["review_excluded"]:3d}')
    if not city_summaries: raise ValueError('No matching cities')
    allrows=pd.concat(details,ignore_index=True)
    allrows.to_csv(OUT/'source_inclusion_exclusion_audit.csv',index=False)
    allrows[allrows.inclusion_status.eq('excluded')].to_csv(OUT/'excluded_source_roads.csv',index=False)
    allrows[allrows.review_flag].to_csv(OUT/'flagged_source_roads.csv',index=False)
    pd.DataFrame(city_summaries).to_csv(OUT/'city_inclusion_exclusion_summary.csv',index=False)
    pd.DataFrame(bylabel).to_csv(OUT/'inclusion_by_original_label.csv',index=False)
    pd.DataFrame(bysupport).to_csv(OUT/'inclusion_by_geometric_support.csv',index=False)
    pd.DataFrame(bysplit).to_csv(OUT/'included_split_label_support_summary.csv',index=False)
    pd.DataFrame(bytile).to_csv(OUT/'included_tile_review_summary.csv',index=False)
    pd.DataFrame(split_issues,columns=['country','city','tile_id','splits']).to_csv(OUT/'tile_split_leakage_flags.csv',index=False)
    country_summary=allrows.groupby('country').agg(total=('source_input_row','size'),included=('inclusion_status',lambda s:int(s.eq('included').sum())),
        excluded=('inclusion_status',lambda s:int(s.eq('excluded').sum())),flagged=('review_flag','sum')).reset_index()
    country_summary['excluded_pct']=100*country_summary.excluded/country_summary.total
    country_summary.to_csv(OUT/'country_inclusion_exclusion_summary.csv',index=False)
    metadata={'script':'95','purpose':'read-only inclusion, exclusion, and spatial partition integrity audit',
              'exclusion_cause':'NOT ESTABLISHED: requires inspection of upstream filtering/selection scripts',
              'excluded_split':'UNKNOWN: absent graph rows do not have authoritative graph split/tile assignments',
              'geometry_support':'NOT ground truth; target network was used for diagnostic only, not training features',
              'coverage':'NOT independently verified','tile_check':'within-country/city tile does not cross split',
              'input_script94':str(AUDIT94),'input_graph':str(GRAPH_PATH)}
    (OUT/'audit_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('\nCOUNTRY TOTALS\n'+country_summary.to_string(index=False))
    print(f'\nTile split overlap flags: {len(split_issues)}')
    print('Read-only outputs:',OUT)
    print('NOTE: Exclusion causes are NOT inferred; excluded records have no authoritative graph split or tile.')
    print('NOTE: Geometric-support disagreement is not proof of incorrect original labels.')

if __name__=='__main__': main()
