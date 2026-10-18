#!/usr/bin/env python3
"""CartoGraph-X Script 102: training-only cartographic manual review package.

Reads Script 101's stratified review queue and the six original validated GPKGs.
Draws source road and nearby target-road geometries, NOT asserted matched pairs.
Does not read validation/test labels, relabel roads, or choose a configuration.
Dependencies: pandas, numpy, geopandas, shapely, matplotlib.
"""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data' / 'processed'
IN = BASE / 'cross_country' / 'benchmark_v2_correspondence_verification_v1'
OUT = BASE / 'cross_country' / 'benchmark_v2_manual_review_package_v1'
CITY_PATHS = {
 ('Lithuania','vilnius_central'): BASE/'lithuania'/'validated_pairs'/'vilnius_central_validated.gpkg',
 ('Lithuania','kaunas_central'): BASE/'lithuania'/'validated_pairs'/'kaunas_central_validated.gpkg',
 ('Lithuania','klaipeda_central'): BASE/'lithuania'/'validated_pairs'/'klaipeda_central_validated.gpkg',
 ('Poland','warsaw'): BASE/'poland'/'road_target_validation'/'warsaw_validated_road_pair.gpkg',
 ('Poland','krakow'): BASE/'poland'/'road_target_validation'/'krakow_validated_road_pair.gpkg',
 ('Poland','gdansk'): BASE/'poland'/'road_target_validation'/'gdansk_validated_road_pair.gpkg',
}
CONFIGS = ['C1_conservative','C2_intermediate','C3_permissive']
TRANSITIONS = ['C1_already_matched','C2_new_match','C3_only_new_match','unmatched_all']


def get_neighbors(gdf, geometry, radius):
    region = geometry.buffer(radius)
    try:
        indices = list(gdf.sindex.query(region, predicate='intersects'))
        return gdf.iloc[indices]
    except Exception:
        return gdf.loc[gdf.geometry.intersects(region)]


def plot_sample(row, source, targets, destination, radius):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    geom = source.geometry.iloc[int(row.source_input_row)]
    if geom is None or geom.is_empty:
        return 'empty source geometry'
    near = get_neighbors(targets, geom, radius)
    minx,miny,maxx,maxy = geom.bounds
    width=max(maxx-minx,maxy-miny,40.0)
    pad=max(25.0,min(110.0,width*.35))
    bounds=(minx-pad,maxx+pad,miny-pad,maxy+pad)
    fig, axes=plt.subplots(1,2,figsize=(12,5.3),constrained_layout=True)
    for ax in axes:
        ax.set_aspect('equal',adjustable='box')
        ax.set_xlim(bounds[0],bounds[1]); ax.set_ylim(bounds[2],bounds[3])
        ax.tick_params(labelsize=7)
        ax.grid(alpha=.12)
    nearby_sources=get_neighbors(source,geom,radius)
    if len(nearby_sources): nearby_sources.plot(ax=axes[0],color='#bcbcbc',linewidth=1.1)
    axes[0].set_title('Source road in local source network',fontsize=10)
    if len(near): near.plot(ax=axes[1],color='#1f77b4',linewidth=2)
    axes[1].set_title('Source road vs nearby TARGET network',fontsize=10)
    # Overlay the selected source feature on both panels.
    import geopandas as gpd
    focal=gpd.GeoSeries([geom],crs=source.crs)
    for ax in axes: focal.plot(ax=ax,color='#d62728',linewidth=3.2)
    flags=' / '.join(f'{c}: {"YES" if int(row[c]) else "NO"}' for c in CONFIGS)
    fig.suptitle(f'{row.country} | {row.city} | source row {int(row.source_input_row)}\n'
                 f'{row.transition} | {flags} | nearby target features: {len(near)}',fontsize=11)
    fig.legend(handles=[Line2D([0],[0],color='#d62728',lw=3,label='Focal source road'),
                        Line2D([0],[0],color='#1f77b4',lw=2,label='Nearby target roads'),
                        Line2D([0],[0],color='#bcbcbc',lw=1,label='Other source roads')],
               loc='lower center',ncol=3,fontsize=8)
    fig.savefig(destination,dpi=135,bbox_inches='tight')
    plt.close(fig)
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-panels',type=int,default=120,
                   help='Total deterministic stratified image panels (default 120, 0 for all 360)')
    p.add_argument('--radius-m',type=float,default=50.0,
                   help='Target proximity visualization radius (default 50 m)')
    args=p.parse_args()
    if args.max_panels<0 or args.radius_m<=0: p.error('Invalid max-panels or radius-m')
    queue=IN/'manual_review_queue.csv'
    if not queue.exists(): raise FileNotFoundError(f'Run Script 101 first: {queue}')
    d=pd.read_csv(queue,low_memory=False)
    required={'country','city','source_input_row','transition',*CONFIGS}
    if missing:=required-set(d.columns): raise ValueError(f'Missing columns: {sorted(missing)}')
    if d.duplicated(['country','city','source_input_row']).any(): raise ValueError('Duplicate review IDs')
    if d.empty: raise ValueError('Empty review queue')
    if set(d.transition.unique())-set(TRANSITIONS): raise ValueError('Unexpected transitions')
    # Validate Script 101 labels before proceeding.
    for c in CONFIGS:
        d[c]=pd.to_numeric(d[c],errors='raise').astype(int)
        if not d[c].isin([0,1]).all(): raise ValueError(f'Nonbinary {c}')
    if not ((d[CONFIGS[0]]<=d[CONFIGS[1]]) & (d[CONFIGS[1]]<=d[CONFIGS[2]])).all():
        raise ValueError('Nonmonotonic matching flags')
    d['source_input_row']=pd.to_numeric(d.source_input_row,errors='raise').astype(int)
    d=d.sort_values(['country','city','transition','source_input_row']).reset_index(drop=True)
    # Preserve every review row, including rows without a rendered panel.
    OUT.mkdir(parents=True,exist_ok=True)
    image_dir=OUT/'panels'; image_dir.mkdir(exist_ok=True)
    d['image_file']=''; d['review_validity']=''; d['review_confidence']=''; d['review_target_feature_ids']=''
    d['reviewer']=''; d['review_date']=''; d['review_comments']=''
    # Round-robin sampling across 6 cities x 4 transition groups.
    groups=[g.index.to_list() for _,g in d.groupby(['country','city','transition'],sort=True)]
    chosen=[]; i=0
    budget=len(d) if args.max_panels==0 else min(args.max_panels,len(d))
    while len(chosen)<budget:
        any_added=False
        for group in groups:
            if i<len(group) and len(chosen)<budget:
                chosen.append(group[i]); any_added=True
        if not any_added: break
        i+=1
    chosen=set(chosen)
    try:
        import geopandas as gpd
        import matplotlib
    except ImportError as exc:
        raise SystemExit('Missing package. Install: pip install geopandas matplotlib') from exc
    errors=[]; panels=0
    for (country,city),part in d.loc[sorted(chosen)].groupby(['country','city'],sort=True):
        path=CITY_PATHS.get((country,city))
        if path is None or not path.is_file(): raise FileNotFoundError(f'Validated GPKG missing: {path}')
        source=gpd.read_file(path,layer='source_roads').reset_index(drop=True)
        targets=gpd.read_file(path,layer='target_roads').reset_index(drop=True)
        if source.crs is None or targets.crs is None: raise ValueError(f'CRS missing for {city}')
        if source.crs!=targets.crs: targets=targets.to_crs(source.crs)
        if source.crs.is_geographic: raise ValueError(f'{city}: expected projected metre CRS, found {source.crs}')
        for idx,row in part.iterrows():
            r=int(row.source_input_row)
            if r<0 or r>=len(source): raise ValueError(f'{city}: source index {r} outside [0,{len(source)})')
            filename=f'{country.lower()}_{city}_{r:06d}.png'
            err=plot_sample(row,source,targets,image_dir/filename,args.radius_m)
            if err: errors.append({'country':country,'city':city,'source_input_row':r,'error':err})
            else: d.at[idx,'image_file']='panels/'+filename; panels+=1
        print(f'{country:10} {city:18} rendered={len(part):3} source={len(source):5} target={len(targets):5}',flush=True)
    d.to_csv(OUT/'manual_review_sheet.csv',index=False)
    pd.DataFrame(errors,columns=['country','city','source_input_row','error']).to_csv(OUT/'render_errors.csv',index=False)
    # Static HTML index works offline. No script, external assets, or network dependencies.
    html_rows=[]
    for row in d.itertuples(index=False):
        if not row.image_file: continue
        caption=f'{row.country} / {row.city} / source #{row.source_input_row} / {row.transition}'
        html_rows.append(f'<article><img loading="lazy" src="{html.escape(row.image_file,quote=True)}" alt="Road comparison">'
                         f'<p>{html.escape(caption)}</p></article>')
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>CartoGraph-X correspondence review</title>
<style>body{font:15px system-ui;margin:24px;max-width:1450px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:18px}article{border:1px solid #bbb;padding:10px}img{width:100%;height:auto}p{margin:5px 0}</style></head>
<body><h1>CartoGraph-X: training-only correspondence review</h1><p>Red: focal source road. Blue: nearby target network. This is NOT a verified source-target pair. Record manual assessments in manual_review_sheet.csv. Blank panels indicate unrendered queue items.</p><main>'''+''.join(html_rows)+'</main></body></html>'
    (OUT/'index.html').write_text(page,encoding='utf-8')
    manifest={'script':102,'train_only':True,'queue_rows':len(d),'panels':panels,
              'radius_m':args.radius_m,'requested_max_panels':args.max_panels,
              'limitations':['Near-target visualization does not establish a valid correspondence.',
                             'Source road row IDs are positional within original validated source layer.',
                             'Reviewer must inspect network connectivity, parallel roads, scale effects and merge/aggregation.',
                             'No automatic threshold selection or label modification.',
                             'All queue rows exported; subset rendered using deterministic round-robin stratification.']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'\nReview queue: {len(d)} | panels rendered: {panels} | errors: {len(errors)}')
    print('Review HTML:',OUT/'index.html')
    print('Review sheet:',OUT/'manual_review_sheet.csv')
    print('READ-ONLY. TRAIN ONLY. Existing benchmark unchanged.')

if __name__=='__main__': main()
