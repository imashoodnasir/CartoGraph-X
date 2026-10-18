#!/usr/bin/env python3
"""CartoGraph-X Script 98: read-only cross-country label/support sensitivity audit.

Uses the original per-road tables produced by Script 94 and Script 93. A geometric
support threshold is a diagnostic proxy, NOT a source-target correspondence label.
Does not alter existing labels, partitions, source data or test results.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data' / 'processed' / 'cross_country'
A94 = BASE / 'original_geometric_label_audit_v1'
A93 = BASE / 'geographic_coverage_road_support_audit_v1'
OUT = BASE / 'benchmark_v2_label_support_audit_v1'
CITIES = [('Lithuania','vilnius_central'),('Lithuania','kaunas_central'),('Lithuania','klaipeda_central'),('Poland','warsaw'),('Poland','krakow'),('Poland','gdansk')]
RADII = [5,15,30]
THRESHOLDS = [0.10,0.20,0.35,0.50]

def number(s): return pd.to_numeric(s,errors='coerce')
def require(df, cols, path):
    missing=set(cols)-set(df.columns)
    if missing: raise ValueError(f'{path}: missing {sorted(missing)}; available={list(df.columns)}')

def binary_labels(d,path):
    if 'original_label' in d:
        s=d['original_label'].astype(str).str.lower().str.strip()
        mapping={'1':1,'1.0':1,'true':1,'selected':1,'weak_selected':1,'retained':1,
                 '0':0,'0.0':0,'false':0,'elimination':0,'eliminated':0}
        b=s.map(mapping)
        if b.notna().all(): return b.astype('int8')
    for c in ['retained','is_selected']:
        if c in d:
            b=number(d[c]);
            if b.isin([0,1]).all(): return b.astype('int8')
    raise ValueError(f'{path}: unable to decode original labels; sample={d.original_label.value_counts(dropna=False).head(12).to_dict()}')

def find93(country,city):
    candidates=[A93/f'{country.lower()}_{city}_source_road_support.csv', A93/f'{country.lower()}_{city}_road_support.csv']
    for p in candidates:
        if p.exists(): return p
    matches=sorted(A93.glob(f'*{city}*source*support*.csv'))
    if len(matches)==1: return matches[0]
    return None

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]; summaries=[]; bands=[]; missing93=[]
    for country,city in CITIES:
        path=A94/f'{country.lower()}_{city}_original_vs_support.csv'
        if not path.exists(): raise FileNotFoundError(f'Missing Script94 output: {path}')
        d=pd.read_csv(path,low_memory=False)
        require(d,['source_input_row','source_feature_length_m','original_label'],path)
        if d.source_input_row.isna().any() or d.source_input_row.duplicated().any():
            raise ValueError(f'{path}: invalid/duplicate source_input_row')
        d['source_input_row']=number(d.source_input_row).astype('int64')
        d['original_binary']=binary_labels(d,path)
        d['country']=country; d['city']=city
        d['graph_included']=d.graph_join.astype(str).eq('both') if 'graph_join' in d else False
        p93=find93(country,city)
        if p93:
            s=pd.read_csv(p93,low_memory=False)
            cols=['source_input_row']+[f'supported_fraction_{r}m' for r in RADII if f'supported_fraction_{r}m' in s]
            require(s,['source_input_row'],p93)
            if s.source_input_row.duplicated().any(): raise ValueError(f'{p93}: duplicate source ids')
            s=s[cols].copy(); s['source_input_row']=number(s.source_input_row).astype('int64')
            d=d.merge(s,on='source_input_row',how='left',validate='one_to_one',suffixes=('','_from93'),indicator='_support_merge')
            if (d._support_merge!='both').any(): raise ValueError(f'{city}: {int((d._support_merge!="both").sum())} roads missing Script93 support')
            for r in RADII:
                c=f'supported_fraction_{r}m'; alt=c+'_from93'
                if alt in d:
                    if c in d:
                        left=number(d[c]); right=number(d[alt]); both=left.notna() & right.notna()
                        if ((left[both]-right[both]).abs()>1e-5).any(): raise ValueError(f'{city}: support mismatch at {r}m')
                        d[c]=left.combine_first(right)
                    else: d[c]=d[alt]
                    d=d.drop(columns=alt)
            d=d.drop(columns='_support_merge')
        else: missing93.append(city)
        for r in RADII:
            col=f'supported_fraction_{r}m'
            if col not in d: continue
            d[col]=number(d[col]); valid=d[col].notna() & d[col].between(0,1)
            if not valid.all(): raise ValueError(f'{city}: {col} has {int((~valid).sum())} missing/out-of-range values')
            for t in THRESHOLDS:
                proxy=d[col]>=t
                disagree=proxy.ne(d.original_binary.astype(bool))
                for subset,mask in [('all',pd.Series(True,index=d.index)),('graph_included',d.graph_included),('graph_excluded',~d.graph_included)]:
                    x=d.loc[mask]; p=proxy.loc[mask]; z=disagree.loc[mask]
                    summaries.append(dict(country=country,city=city,subset=subset,radius_m=r,threshold=t,
                        n=len(x),positive_original=int(x.original_binary.sum()),positive_proxy=int(p.sum()),
                        proxy_disagreement_n=int(z.sum()),proxy_disagreement_pct=round(100*z.mean(),3) if len(x) else np.nan,
                        original_positive_proxy_negative=int((x.original_binary.eq(1)&~p).sum()),
                        original_negative_proxy_positive=int((x.original_binary.eq(0)&p).sum())))
                if r==15 and t in [0.20,0.50]:
                    bands.append(dict(country=country,city=city,threshold=t,original_positive_proxy_negative=int((d.original_binary.eq(1)&~proxy).sum()),original_negative_proxy_positive=int((d.original_binary.eq(0)&proxy).sum())))
        keep=['country','city','source_input_row','original_binary','graph_included','source_feature_length_m']
        keep += [c for c in ['original_label','evidence_tier','potential_disagreement','split','sample_uid','bbox_outside_any'] if c in d]
        keep += [f'supported_fraction_{r}m' for r in RADII if f'supported_fraction_{r}m' in d]
        for r in RADII:
            c=f'supported_fraction_{r}m'
            if c in d:
                for t in THRESHOLDS:
                    label=f'proxy_disagree_{r}m_{int(t*100):02d}pct'
                    d[label]=(d[c].ge(t)!=d.original_binary.eq(1))
                    keep.append(label)
        rows.append(d[keep].copy())
        print(f'{country:10s} {city:18s} rows={len(d):5d} graph={int(d.graph_included.sum()):5d} original_positive={d.original_binary.mean():.3f}')
    detail=pd.concat(rows,ignore_index=True)
    detail.to_csv(OUT/'source_road_label_support_sensitivity.csv',index=False)
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUT/'city_radius_threshold_sensitivity.csv',index=False)
    pd.DataFrame(bands).to_csv(OUT/'selected_support_comparison_15m.csv',index=False)
    agg=[]
    for (country,subset,r,t),group in summary.groupby(['country','subset','radius_m','threshold']):
        n=group.n.sum(); mismatch=group.proxy_disagreement_n.sum()
        agg.append(dict(country=country,subset=subset,radius_m=r,threshold=t,n=int(n),
                        original_positive=int(group.positive_original.sum()),proxy_positive=int(group.positive_proxy.sum()),
                        proxy_disagreement_n=int(mismatch),proxy_disagreement_pct=round(100*mismatch/n,3) if n else np.nan,
                        original_positive_proxy_negative=int(group.original_positive_proxy_negative.sum()),
                        original_negative_proxy_positive=int(group.original_negative_proxy_positive.sum())))
    country=pd.DataFrame(agg)
    country.to_csv(OUT/'country_radius_threshold_sensitivity.csv',index=False)
    manifest={'script':'98_benchmark_v2_label_support_audit.py','read_only_inputs':True,
        'source_records':len(detail),'graph_included':int(detail.graph_included.sum()),
        'graph_excluded':int((~detail.graph_included).sum()),'missing_script93_city_files':missing93,
        'warning':'Geometric proximity/support is NOT correspondence ground truth. Proxy disagreements must not be used to relabel roads.',
        'next_step':'Compare original Script40 and Script59 correspondence acceptance under unified matching rules, without modifying current test labels.'}
    (OUT/'audit_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('\nCOUNTRY SUMMARY: 15m support, 20% threshold (diagnostic only)')
    print(country[(country.subset=='graph_included')&(country.radius_m==15)&(country.threshold==0.20)].to_string(index=False))
    print('\nSaved to:',OUT)
    print('CAUTION: These are geometric-support proxies, not validated correspondence labels.')

if __name__=='__main__': main()
