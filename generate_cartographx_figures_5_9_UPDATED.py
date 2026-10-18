#!/usr/bin/env python3
"""Generate CartoGraph-X publication Figures 5-9 from the 16-table workbook.
Usage: python generate_cartographx_figures_5_9.py --workbook CartoGraphX_Master_Results.xlsx
"""
import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.stats import t

SEED_COLS = [f'Seed {s}' for s in range(42, 52)]
PROTOCOL_NAMES = {'joint_to_lt':'Joint → Lithuania','joint_to_pl':'Joint → Poland',
 'lt_to_lt':'Lithuania → Lithuania','pl_to_pl':'Poland → Poland',
 'lt_to_pl_zero_shot':'Lithuania → Poland','pl_to_lt_zero_shot':'Poland → Lithuania'}

def read_sheet(path, name):
    df = pd.read_excel(path, sheet_name=name, header=1)
    for col in SEED_COLS:
        if col not in df.columns:
            raise ValueError(f'{name} missing {col}')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def stats(row):
    v = row[SEED_COLS].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0: return np.nan,np.nan,np.nan,0
    m = float(v.mean())
    if n < 2: return m,np.nan,np.nan,n
    sd = float(v.std(ddof=1))
    ci = float(t.ppf(0.975,n-1)*sd/math.sqrt(n))
    return m,sd,ci,n

def parse_fields(value):
    return dict((m.group(1).strip(),m.group(2).strip()) for m in
        re.finditer(r'([A-Za-z_]+)\s*=\s*([^|]+)',str(value)))

def save(fig, output, stem):
    fig.savefig(output/f'{stem}.png', dpi=350, bbox_inches='tight')
    fig.savefig(output/f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'SAVED {stem}.png and .pdf')

def fig5(wb,out):
    df=read_sheet(wb,'T06_Graph_Val')
    df=df[df['Metric'].astype(str).str.lower().eq('mcc')].copy()
    records=[]
    for _,row in df.iterrows():
        f=parse_fields(row['Protocol / Group'])
        try: k=int(float(f['k'])); radius=float(f['radius_m']); proto=f['protocol']
        except (KeyError,ValueError): continue
        m,sd,ci,n=stats(row)
        if n: records.append(dict(k=k,radius=radius,protocol=proto,mcc=m,n=n))
    d=pd.DataFrame(records)
    if d.empty: raise ValueError('T06 has no graph MCC data')
    protocols=[p for p in PROTOCOL_NAMES if p in set(d.protocol)]
    protocols += sorted(set(d.protocol)-set(protocols))
    fig,axs=plt.subplots(2,3,figsize=(15,9),squeeze=False,layout='constrained')
    lo,hi=float(d.mcc.min()),float(d.mcc.max())
    if lo==hi: lo-=0.01;hi+=0.01
    for ax,proto in zip(axs.flat,protocols):
        p=d[d.protocol.eq(proto)].pivot_table(index='k',columns='radius',values='mcc',aggfunc='mean').sort_index(ascending=True)
        im=ax.imshow(p.to_numpy(),cmap='viridis',vmin=lo,vmax=hi,aspect='auto',origin='lower')
        ax.set_xticks(range(len(p.columns)),[f'{x:g}' for x in p.columns])
        ax.set_yticks(range(len(p.index)),[str(x) for x in p.index])
        ax.set_xlabel('Maximum graph radius (m)');ax.set_ylabel('Neighbors (k)')
        ax.set_title(PROTOCOL_NAMES.get(proto,proto))
        for i in range(len(p.index)):
            for j in range(len(p.columns)):
                v=p.iloc[i,j]
                if np.isfinite(v): ax.text(j,i,f'{v:.3f}',ha='center',va='center',fontsize=8,
                   color='white' if (v-lo)/(hi-lo)<0.48 else 'black')
    for ax in axs.flat[len(protocols):]:ax.axis('off')
    fig.colorbar(im,ax=axs.ravel().tolist(),label='Validation MCC (seed mean)',shrink=0.80)
    save(fig,out,'Fig05_Graph_Construction_Sensitivity')

def fig6(wb,out):
    df=read_sheet(wb,'T07_Fraction_Val')
    df=df[df['Metric'].astype(str).str.lower().eq('mcc')]
    rows=[]
    for _,row in df.iterrows():
        f=parse_fields(row['Protocol / Group'])
        try: frac=float(f['fraction']);proto=f['protocol']
        except (KeyError,ValueError):continue
        m,sd,ci,n=stats(row)
        if n:rows.append((proto,frac,m,ci,n))
    d=pd.DataFrame(rows,columns=['protocol','fraction','mean','ci','n'])
    if d.empty:raise ValueError('T07 has no fraction MCC data')
    fig,ax=plt.subplots(figsize=(10,6.3))
    for proto,g in d.groupby('protocol'):
        g=g.sort_values('fraction')
        x=g.fraction.to_numpy(dtype=float)*100
        y=g['mean'].to_numpy(dtype=float)
        err=g.ci.to_numpy(dtype=float)
        ax.plot(x,y,marker='o',lw=2,label=PROTOCOL_NAMES.get(proto,proto))
        if np.isfinite(err).any():ax.fill_between(x,y-np.nan_to_num(err),y+np.nan_to_num(err),alpha=0.14)
    ax.set_xlabel('Training data used (%)');ax.set_ylabel('Validation MCC')
    ax.set_title('Sensitivity to training data availability')
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5,0.015),
               ncol=6, frameon=False, fontsize=8, columnspacing=1.0, handlelength=1.5)
    fig.subplots_adjust(bottom=0.18,top=0.91,left=0.10,right=0.98)
    save(fig,out,'Fig06_Training_Fraction_Learning_Curves')

def fig7(wb,out):
    df=read_sheet(wb,'T12_Tiles')
    df=df[df['Metric'].astype(str).str.lower().eq('mcc')]
    rows=[]
    for _,row in df.iterrows():
        # The same tile may appear in both seed_tile_confusion and tile_seed_metrics.
        # Keep the former as canonical when both exist.
        if str(row['Model / Variant'])!='seed_tile_confusion':continue
        f=parse_fields(row['Protocol / Group'])
        tile=f.get('tile_id','')
        m=re.search(r'_r(\d+)_c(\d+)$',tile)
        if not m:continue
        mean,sd,ci,n=stats(row)
        rows.append(dict(city=f.get('city','unknown'),protocol=f.get('protocol','unknown'),
            r=int(m.group(1)),c=int(m.group(2)),mean=mean,n=n))
    d=pd.DataFrame(rows)
    if d.empty:raise ValueError('T12 has no parseable tile coordinates in seed_tile_confusion')
    groups=list(d.groupby(['protocol','city']))
    # There are 18 city-protocol panels in the current workbook. Preserve every
    # panel: two requested six-panel figures, plus a third six-panel supplement.
    for batch_start in range(0,len(groups),6):
        chunk=groups[batch_start:batch_start+6]
        fig,axs=plt.subplots(2,3,figsize=(13.8,8.4),squeeze=False,layout='constrained')
        im=None
        for ax,((proto,city),g) in zip(axs.flat,chunk):
            grid=np.full((int(g.r.max())+1,int(g.c.max())+1),np.nan)
            for _,item in g.iterrows():
                if item.n: grid[int(item.r),int(item.c)]=item['mean']
            im=ax.imshow(np.ma.masked_invalid(grid),vmin=-1,vmax=1,
                         cmap='RdYlGn',origin='upper')
            ax.set_title(f'{city} | {PROTOCOL_NAMES.get(proto,proto)}',fontsize=10)
            ax.set_xlabel('Tile column (grid index)')
            ax.set_ylabel('Tile row (grid index)')
            ax.set_xticks(range(grid.shape[1]))
            ax.set_yticks(range(grid.shape[0]))
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if np.isfinite(grid[i,j]):
                        ax.text(j,i,f'{grid[i,j]:.2f}',ha='center',va='center',fontsize=8)
        for ax in axs.flat[len(chunk):]:ax.axis('off')
        fig.colorbar(im,ax=axs.ravel().tolist(),
                     label='Tile MCC (seed mean); blank = undefined',shrink=0.8)
        letter=chr(ord('A')+batch_start//6)
        save(fig,out,f'Fig07{letter}_Spatial_Tile_MCC_Heatmaps')
    print(f'Fig07 contains {len(groups)} city/protocol panels in '
          f'{math.ceil(len(groups)/6)} six-panel figures.')

def fig8(wb,out):
    df=read_sheet(wb,'T08_Paired_Diffs')
    records=[]
    for _,row in df.iterrows():
        m,sd,ci,n=stats(row)
        if n<3 or not np.isfinite(ci):continue
        model=str(row['Model / Variant']).replace('CartoGraph-X minus ','')
        protocol=str(row['Protocol / Group'])
        records.append((protocol,model,m,ci,n))
    d=pd.DataFrame(records,columns=['protocol','model','mean','ci','n'])
    if d.empty:raise ValueError('T08 has no paired comparisons with at least 3 seeds')
    # Limit the main figure to 10-run GNN comparisons, avoiding partial classical comparisons.
    full=d[d.n.eq(10)].copy()
    if full.empty:full=d
    full=full.sort_values(['protocol','model'])
    fig,ax=plt.subplots(figsize=(11,max(5,0.25*len(full)+1.5)),layout='constrained')
    y=np.arange(len(full))
    ax.errorbar(full['mean'],y,xerr=full.ci,fmt='o',capsize=2,markersize=3.7,elinewidth=1)
    ax.axvline(0,color='black',lw=1,ls='--')
    labels=[f"{PROTOCOL_NAMES.get(p,p)} | {m} (n={n})" for p,m,n in
        zip(full.protocol,full.model,full.n)]
    ax.set_yticks(y,labels,fontsize=7.7);ax.invert_yaxis()
    ax.set_xlabel('Paired test MCC difference: CartoGraph-X minus baseline (95% t CI)')
    ax.set_title('Matched-seed model comparisons')
    ax.grid(axis='x',alpha=0.22)
    save(fig,out,'Fig08_Paired_MCC_Forest_Plot')

def fig9(wb,out):
    perf=read_sheet(wb,'T03_Baseline_Test')
    eff=read_sheet(wb,'T09_Efficiency')
    perf=perf[perf['Metric'].astype(str).str.lower().eq('mcc')]
    records=[]
    for _,row in eff.iterrows():
        metric=str(row['Metric'])
        if metric not in ('inference_graph_seconds_mean','parameter_count'):continue
        f=parse_fields(row['Protocol / Group'])
        model=f.get('model','')
        pop=f.get('population','')
        if not model or not pop:continue
        mean,sd,ci,n=stats(row)
        if n>=2 and np.isfinite(mean):records.append(dict(model=model,pop=pop,metric=metric,value=mean,n=n))
    ed=pd.DataFrame(records)
    if ed.empty:raise ValueError('T09 has no ten-seed efficiency measurements')
    pmap={'Joint':'joint_to_lt','Lithuania':'lt_to_lt','Poland':'pl_to_pl'}
    points=[]
    for (pop,model),g in ed.groupby(['pop','model']):
        lookup=dict(zip(g.metric,g.value))
        if 'inference_graph_seconds_mean' not in lookup or 'parameter_count' not in lookup:continue
        proto=pmap.get(pop)
        if proto is None:continue
        # Performance can be compared only for matching evaluation protocol.
        candidates=perf[perf['Protocol / Group'].eq(proto)]
        if model.lower().replace('-','') in ('cartographx','cartographxv2'):
            candidates=candidates[candidates['Model / Variant'].astype(str).str.contains('CartoGraph-X V2',regex=False)]
        else:
            candidates=candidates[candidates['Model / Variant'].astype(str).str.contains('model='+model,regex=False)]
            # Weighted/unweighted candidates are distinct; do not silently average.
            candidates=candidates[candidates['Model / Variant'].astype(str).str.contains('variant=unweighted',regex=False)]
        if len(candidates)!=1:
            print(f'SKIP Fig09 ambiguous/missing test MCC: {pop} {model} ({len(candidates)} matches)')
            continue
        m,sd,ci,n=stats(candidates.iloc[0])
        if n<2:continue
        points.append(dict(pop=pop,model=model,mcc=m,latency_ms=lookup['inference_graph_seconds_mean']*1000,
                           parameters=lookup['parameter_count']))
    if not points:raise ValueError('No directly matched efficiency/test pairs; do not infer cross-protocol values')
    d=pd.DataFrame(points)
    fig,ax=plt.subplots(figsize=(9,5.5),layout='constrained')
    for pop,g in d.groupby('pop'):
        ax.scatter(g.latency_ms,g.mcc,s=np.clip(g.parameters/110,35,850),alpha=0.7,
                   label=pop,edgecolors='black',linewidths=0.5)
        for _,r in g.iterrows():ax.annotate(r.model,(r.latency_ms,r.mcc),xytext=(5,5),
                                           textcoords='offset points',fontsize=8)
    ax.set_xlabel('Mean full-graph inference time (ms)')
    ax.set_ylabel('Test MCC (matched protocol)')
    ax.set_title('Predictive performance and computational cost')
    ax.grid(alpha=0.2);ax.legend(title='Training population',frameon=False)
    save(fig,out,'Fig09_Accuracy_Latency_Tradeoff')
    d.to_csv(out/'Fig09_Plot_Data.csv',index=False)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workbook',default='CartoGraphX_Master_Results.xlsx')
    p.add_argument('--output',default='figures_5_9')
    a=p.parse_args()
    wb=Path(a.workbook).resolve();out=Path(a.output).resolve()
    if not wb.exists():p.error(f'Workbook not found: {wb}')
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
    for num,fn in [(5,fig5),(6,fig6),(7,fig7),(8,fig8),(9,fig9)]:
        try:fn(wb,out)
        except Exception as exc:
            print(f'ERROR Fig{num}: {type(exc).__name__}: {exc}')
            raise
    print(f'FIGURES 5, 6, 7A-C, 8, 9 GENERATED: {out}')

if __name__=='__main__':main()
