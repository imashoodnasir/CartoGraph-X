#!/usr/bin/env python3
"""CartoGraph-X 97: read-only upstream spatial-filter investigation.

Run: python scripts/97_spatial_filter_root_cause_audit.py
Uses Script95's full source-level inclusion audit. Finds actual pipeline scripts,
extracts filtering-related code context, and tests measurable explanations for
exclusions. Does not claim a root cause from correlation or overwrite data.
"""
from pathlib import Path
import argparse, json, re, hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CROSS = ROOT/'data'/'processed'/'cross_country'
DEFAULT_OUT = CROSS/'spatial_filter_root_cause_audit_v1'
SOURCE95 = CROSS/'benchmark_integrity_exclusion_audit_v1'/'source_inclusion_exclusion_audit.csv'
KEYWORDS = re.compile(r'\b(filter|dropna|drop_duplicates|is_valid|is_empty|within|intersects|sjoin|overlay|clip|buffer|tile|split|boundary|length|minimum|min_length|sample|spatial|exclude|remove|keep|select|intersection|centroid|geometry)\b',re.I)
PIPELINE_NAMES = re.compile(r'(spatial|harmoniz|road|graph|benchmark|tile|split|learning|correspondence)',re.I)

def norm_city(x):
    s=re.sub('[^a-z0-9]','',str(x).lower())
    return {'vilniuscentral':'vilnius','kaunascentral':'kaunas','klaipedacentral':'klaipeda'}.get(s,s)

def detect_included(d):
    if 'inclusion_status' in d:
        x=d['inclusion_status'].astype(str).str.lower().str.strip()
        print('Observed inclusion_status values:',x.value_counts(dropna=False).to_dict())
        if x.isin(['included','excluded']).all():return x.eq('included')
    for col in ('included_in_graph','in_graph','graph_included'):
        if col in d:
            x=d[col].astype(str).str.lower().str.strip()
            if x.isin(['true','false','1','0']).all():return x.isin(['true','1'])
    raise ValueError('Cannot interpret Script95 inclusion_status; inspect source CSV')

def discover_data():
    roots=[ROOT/'data'/'processed'/'poland',ROOT/'data'/'processed'/'lithuania',CROSS]
    inventory=[]
    for root in roots:
        if not root.exists():continue
        for p in root.rglob('*.csv'):
            name=p.name.lower()
            if not PIPELINE_NAMES.search(name+' '+str(p.parent.name)):continue
            try:
                head=pd.read_csv(p,nrows=0).columns.tolist()
                stat=p.stat()
                inventory.append({'path':str(p.relative_to(ROOT)),'bytes':stat.st_size,'columns':';'.join(head),'has_source_index':any(c in head for c in ['source_idx','feature_index','source_idx_harmonized','source_input_row']),'has_split':any(c in head for c in ['split','split_harmonized']),'has_tile':any(c in head for c in ['tile_id','tile_id_harmonized','spatial_tile_id'])})
            except Exception as exc:inventory.append({'path':str(p.relative_to(ROOT)),'read_error':str(exc)})
    return pd.DataFrame(inventory)

def code_inventory():
    rows=[]
    script_dir=ROOT/'scripts'
    if not script_dir.is_dir():return pd.DataFrame(rows)
    for p in sorted(script_dir.glob('*.py')):
        if not PIPELINE_NAMES.search(p.stem):continue
        try:lines=p.read_text(encoding='utf-8-sig',errors='replace').splitlines()
        except Exception:continue
        for i,line in enumerate(lines):
            if not KEYWORDS.search(line):continue
            # Exclude boilerplate comments with no filter-relevant context only if script is very long
            start=max(0,i-2);end=min(len(lines),i+3)
            rows.append({'script':p.name,'line':i+1,'code':line.strip()[:450], 'context':'\n'.join(f'{j+1}: {lines[j]}' for j in range(start,end))[:1600]})
    return pd.DataFrame(rows)

def summarize(d,cols):
    if not all(c in d for c in cols):return pd.DataFrame()
    return d.groupby(cols+['audit_included'],dropna=False).size().unstack(fill_value=0).reset_index().rename(columns={False:'excluded',True:'included'})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    if not SOURCE95.exists():raise FileNotFoundError('Run Script95 first: '+str(SOURCE95))
    d=pd.read_csv(SOURCE95,low_memory=False)
    d['audit_included']=detect_included(d)
    d['country_key']=d['country'].astype(str).str.casefold().str.strip()
    d['city_key']=d['city'].map(norm_city)
    d['source_input_row']=pd.to_numeric(d['source_input_row'],errors='raise').astype('int64')
    keys=['country_key','city_key','source_input_row']
    if d.duplicated(keys).any():raise ValueError('Duplicate source identity in Script95')
    print('Source records:',len(d),'included:',int(d.audit_included.sum()),'excluded:',int((~d.audit_included).sum()))
    summary=d.groupby(['country','city','audit_included'],dropna=False).size().unstack(fill_value=0).rename(columns={False:'excluded',True:'included'}).reset_index()
    summary['excluded_pct']=100*summary['excluded']/(summary['included']+summary['excluded'])
    summary.to_csv(out/'city_exclusion_summary.csv',index=False)
    print(summary.to_string(index=False))
    inv=discover_data();inv.to_csv(out/'available_intermediate_csv_inventory.csv',index=False)
    print('Discovered intermediate CSV files:',len(inv))
    code=code_inventory();code.to_csv(out/'upstream_filter_code_context.csv',index=False)
    print('Filtering-related code lines:',len(code))
    # Search full intermediate CSVs for explicit source-index membership. Never assume an index is stable
    # unless length and original label corroborate the row identity.
    candidates=[]
    for _,entry in inv.iterrows():
        if not entry.get('has_source_index',False):continue
        p=ROOT/entry['path']
        if p.resolve()==SOURCE95.resolve():continue
        try:
            h=pd.read_csv(p,nrows=0).columns
            idx=next((c for c in ('source_idx_harmonized','source_idx','feature_index','source_input_row') if c in h),None)
            city=next((c for c in ('city','city_harmonized','city_key','window') if c in h),None)
            if idx is None or city is None:continue
            country='country' if 'country' in h else None
            length=next((c for c in ('length_m','source_feature_length_m') if c in h),None)
            use=[idx,city]+([country] if country else [])+([length] if length else [])
            x=pd.read_csv(p,usecols=list(dict.fromkeys(use)),low_memory=False)
            x['city_key']=x[city].map(norm_city)
            x['source_input_row']=pd.to_numeric(x[idx],errors='coerce')
            x=x.dropna(subset=['source_input_row']);x['source_input_row']=x.source_input_row.astype('int64')
            if country:x['country_key']=x[country].astype(str).str.casefold().str.strip()
            else:
                sp=str(p).lower()
                inferred='poland' if '/poland/' in sp.replace('\\','/') else 'lithuania' if '/lithuania/' in sp.replace('\\','/') else None
                if inferred is None:continue
                x['country_key']=inferred
            if x.duplicated(keys).any():
                candidates.append({'path':entry['path'],'status':'ambiguous_duplicate_keys','rows':len(x)});continue
            join=d[keys+['audit_included','source_feature_length_m']].merge(x[keys+([length] if length else [])],on=keys,how='left',indicator=True,validate='one_to_one')
            found=join['_merge'].eq('both')
            if length:
                a=pd.to_numeric(join['source_feature_length_m'],errors='coerce')
                b=pd.to_numeric(join[length],errors='coerce')
                length_ok=found & np.isclose(a,b,rtol=1e-5,atol=0.05,equal_nan=False)
                verified=int(length_ok.sum())
                mismatches=int((found & ~length_ok).sum())
            else:verified=0;mismatches=0
            candidates.append({'path':entry['path'],'status':'inspected','rows':len(x),'matched_source_records':int(found.sum()),'excluded_present':int((found & ~join.audit_included).sum()),'included_present':int((found & join.audit_included).sum()),'length_verified':verified,'length_mismatch':mismatches,'length_available':bool(length)})
        except Exception as exc:candidates.append({'path':entry['path'],'status':'read_error','detail':str(exc)[:350]})
    stages=pd.DataFrame(candidates);stages.to_csv(out/'candidate_stage_membership.csv',index=False)
    if not stages.empty:
        cols=[c for c in ('path','status','matched_source_records','excluded_present','length_verified','length_mismatch') if c in stages]
        print('\nSTAGE CANDIDATES (largest excluded membership):')
        print(stages.sort_values('excluded_present',ascending=False,na_position='last')[cols].head(20).to_string(index=False))
    # Investigate correlations, but avoid causal attribution.
    for col in ('original_label','evidence_tier','bbox_outside_any','bbox_outside_all'):
        if col in d:
            s=summarize(d,['country',col]);s.to_csv(out/f'exclusion_by_{col}.csv',index=False)
    numeric=['source_feature_length_m','supported_fraction_5m','supported_fraction_15m','supported_fraction_30m','length_inside_target_bbox_fraction']
    ncols=[c for c in numeric if c in d]
    if ncols:
        z=d.groupby(['country','audit_included'])[ncols].agg(['count','median','mean']).reset_index()
        z.columns=['_'.join(str(k) for k in t if k!='') if isinstance(t,tuple) else str(t) for t in z.columns]
        z.to_csv(out/'exclusion_numeric_characteristics.csv',index=False)
    with (out/'audit_notes.json').open('w',encoding='utf-8') as f:
        json.dump({'source':'Script95','excluded_total':int((~d.audit_included).sum()),'coverage_verified':False,'interpretation':'CSV stage membership and code snippets are diagnostics, not proven filtering causes. Verify code execution paths and feature IDs before claiming a root cause.','output_files':sorted(p.name for p in out.iterdir() if p.is_file())},f,indent=2)
    print('\nOutputs:',out)
    print('CAUTION: Do not infer exclusion causes from code keywords or missing stage records alone.')

if __name__=='__main__':main()
