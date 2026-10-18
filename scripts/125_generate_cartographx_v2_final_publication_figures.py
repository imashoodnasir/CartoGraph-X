#!/usr/bin/env python3
"""SCRIPT 125 (FINAL): Publication cartographic plates, scientific figures and tables.

Frozen CartoGraph-X V2 results ONLY. No training, tuning, new predictions or
model selection. Reads Script 124 real vector map scenes and Script 115-123
existing analysis CSVs. Exports editable SVG/PDF and 600-dpi PNG; maps are
true projected-metre cartography, with same extents across comparison panels.

Requirements: pandas, numpy, matplotlib, geopandas, shapely, pyproj, openpyxl.
Run: python scripts/125_generate_cartographx_v2_final_publication_figures.py
Quick: python scripts/125_generate_cartographx_v2_final_publication_figures.py --maps-only --dpi 220
No scripts follow 125.

Interpretation: predicted geometry = original source segments selected by a
binary road-retention classifier. It is NOT newly generated road geometry.
Ground truth target network is an independently sourced vector layer; error
TP/FP/FN uses provisional C2 correspondence labels (not geometric comparison).
The historical test has been explored; all reported test metrics exploratory.
"""
from __future__ import annotations
import argparse, hashlib, json, math, re, warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, FancyArrowPatch
from matplotlib.ticker import MaxNLocator
from matplotlib import patheffects

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/processed/cross_country'
SCENES=BASE/'cartographx_v2_publication_map_scenes_v1'
STATS=BASE/'cartographx_v2_statistics_v1'
GNN=BASE/'benchmark_v2_ten_seed_gnn_baselines_v1'
CX=BASE/'cartographx_v2_ten_seed_summary_v1'
ABLA=BASE/'cartographx_v2_ablations_v1'
GEO=BASE/'cartographx_v2_geographic_robustness_v1'
SHIFT=BASE/'cartographx_v2_domain_shift_diagnostics_v1'
GRAPH=BASE/'cartographx_v2_graph_sensitivity_v1'
FRACT=BASE/'cartographx_v2_training_fraction_sensitivity_v1'
CAL=BASE/'cartographx_v2_calibration_reliability_v1'
ERROR=BASE/'cartographx_v2_error_failure_analysis_v1'
OUT=BASE/'cartographx_v2_final_publication_package_v1'
COL={'ink':'#17273B','sub':'#566678','paper':'#FFFFFF','background':'#F4F6F2',
     'source':'#ADB7C0','reference':'#154D78','cx':'#D14A36','gnn':'#6B5194',
     'tp':'#137F6D','fp':'#E19B31','fn':'#D23E63','building':'#B7B8AF',
     'green':'#B5D4A4','river':'#B4DCE6','grid':'#DCE4E8'}
PROT=['lt_to_lt','pl_to_pl','lt_to_pl_zero_shot','pl_to_lt_zero_shot','joint_to_lt','joint_to_pl']
NAMES={'lt_to_lt':'LT → LT','pl_to_pl':'PL → PL','lt_to_pl_zero_shot':'LT → PL','pl_to_lt_zero_shot':'PL → LT',
       'joint_to_lt':'Joint → LT','joint_to_pl':'Joint → PL'}
CITY={'vilnius_central':'Vilnius','kaunas_central':'Kaunas','klaipeda_central':'Klaipėda',
      'warsaw':'Warsaw','krakow':'Kraków','gdansk':'Gdańsk'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelcolor':COL['ink'],
 'axes.edgecolor':'#C7D0D8','text.color':COL['ink'],'xtick.color':COL['sub'],
 'ytick.color':COL['sub'],'axes.titleweight':'bold','pdf.fonttype':42,'ps.fonttype':42,
 'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white'})

MANIFEST={'script':'125','created_utc':datetime.now(timezone.utc).isoformat(),
 'scientific_status':'HISTORICAL_TEST_EXPLORATORY; C2_CORRESPONDENCES_PROVISIONAL_NOT_MANUALLY_VALIDATED',
 'figure_files':[],'tables':[],'skipped':[],'warnings':[]}

def load(folder,name,required=True):
 p=folder/name
 if not p.exists():
  if required:raise FileNotFoundError(f'Missing required Script 125 input: {p}')
  MANIFEST['skipped'].append(f'Missing optional table: {p}')
  return None
 return pd.read_csv(p,low_memory=False)

def savefig(fig,name,dpi,formats):
 d=OUT/'figures';d.mkdir(parents=True,exist_ok=True)
 for fmt in formats:
  p=d/f'{name}.{fmt}'
  fig.savefig(p,format=fmt,dpi=dpi if fmt=='png' else None,
              facecolor='white',bbox_inches='tight',pad_inches=.13)
  MANIFEST['figure_files'].append(str(p.relative_to(OUT)))
 plt.close(fig)
 print('  FIGURE:',name,flush=True)

def export_table(df,name):
 if df is None or df.empty:return
 d=OUT/'tables';d.mkdir(parents=True,exist_ok=True)
 p=d/f'{name}.csv';df.to_csv(p,index=False);MANIFEST['tables'].append(str(p.relative_to(OUT)))

def title(fig,title,subtitle=None):
 fig.suptitle(title,x=.055,y=.992,ha='left',fontsize=16,fontweight='bold',color=COL['ink'])
 if subtitle:fig.text(.055,.962,subtitle,ha='left',va='top',fontsize=8.5,color=COL['sub'])

def axis_style(ax,ygrid=True):
 ax.spines[['top','right']].set_visible(False)
 if ygrid:ax.grid(axis='y',color=COL['grid'],linewidth=.65,zorder=0)
 ax.set_axisbelow(True)

def optional_metric(frame,*candidates):
 for c in candidates:
  if c in frame.columns:return c
 return None

def aggregate_seed(df,group,value):
 if value is None:return pd.DataFrame()
 d=df.groupby(group,dropna=False)[value].agg(['mean','std','count']).reset_index()
 return d

def read_layer(path,name):
 try:
  import pyogrio
  if name not in pyogrio.list_layers(path)[:,0]:return None
 except ImportError:
  import fiona
  if name not in fiona.listlayers(path):return None
 g=gpd.read_file(path,layer=name)
 return g if not g.empty else None

def linework(ax,g,color,lw=1,alpha=1,z=3,casing=False):
 if g is None or g.empty:return
 geoms=g.loc[g.geom_type.isin(['LineString','MultiLineString'])]
 if geoms.empty:return
 if casing:geoms.plot(ax=ax,color='#F9FAF7',linewidth=lw+1.15,alpha=alpha,zorder=z-.1)
 geoms.plot(ax=ax,color=color,linewidth=lw,alpha=alpha,zorder=z)

def polygons(ax,g):
 if g is None or g.empty:return
 sub=g.loc[g.geom_type.isin(['Polygon','MultiPolygon'])]
 if not sub.empty:sub.plot(ax=ax,facecolor=COL['building'],edgecolor='#8C9698',
                          linewidth=.19,alpha=.74,zorder=1,rasterized=False)

def scalebar(ax,bounds):
 xmin,ymin,xmax,ymax=bounds;span=xmax-xmin
 wanted=span*.24
 power=10**math.floor(math.log10(max(wanted,1)))
 dist=min([1,2,5,10],key=lambda t:abs(t*power-wanted))*power
 x=xmin+.045*span;y=ymin+.055*(ymax-ymin)
 ax.plot([x,x+dist],[y,y],color='white',lw=5.0,zorder=20,solid_capstyle='butt')
 ax.plot([x,x+dist],[y,y],color=COL['ink'],lw=2.1,zorder=21,solid_capstyle='butt')
 ax.plot([x,x],[y-.007*span,y+.007*span],color=COL['ink'],lw=1.4,zorder=21)
 ax.plot([x+dist,x+dist],[y-.007*span,y+.007*span],color=COL['ink'],lw=1.4,zorder=21)
 ax.text(x+dist/2,y+.014*span,f'{dist:g} m',ha='center',va='bottom',fontsize=7.3,
         weight='bold',color=COL['ink'],zorder=22,
         path_effects=[patheffects.withStroke(linewidth=3,foreground='white')])

def north(ax):
 ax.annotate('N',xy=(.94,.92),xycoords='axes fraction',ha='center',va='bottom',
             fontsize=8,weight='bold',color=COL['ink'])
 ax.annotate('',xy=(.94,.916),xytext=(.94,.84),xycoords='axes fraction',
             arrowprops={'arrowstyle':'-|>','color':COL['ink'],'lw':1.5})

def map_axis(ax,layers,kind,bounds,with_buildings=True):
 ax.set_facecolor(COL['background'])
 if with_buildings:polygons(ax,layers.get('buildings'))
 if kind=='source':
  linework(ax,layers.get('source_roads'),COL['source'],.8,.97,3,casing=True)
 elif kind=='reference':
  linework(ax,layers.get('source_roads'),COL['source'],.42,.33,2)
  linework(ax,layers.get('reference_target_roads'),COL['reference'],1.8,1,5,casing=True)
 elif kind=='cx':
  linework(ax,layers.get('source_roads'),COL['source'],.53,.47,2)
  linework(ax,layers.get('predicted_cartographx_roads'),COL['cx'],1.85,1,5,casing=True)
 elif kind=='gnn':
  linework(ax,layers.get('source_roads'),COL['source'],.53,.47,2)
  linework(ax,layers.get('predicted_gnn_roads'),COL['gnn'],1.85,1,5,casing=True)
 elif kind=='error':
  linework(ax,layers.get('source_roads'),COL['source'],.49,.5,2)
  for key,color in [('tp_roads',COL['tp']),('fp_roads',COL['fp']),('fn_roads',COL['fn'])]:
   linework(ax,layers.get(key),color,1.9,1,5,casing=True)
 else:raise ValueError(kind)
 ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3]);ax.set_aspect('equal',adjustable='box')
 ax.set_xticks([]);ax.set_yticks([])
 for spine in ax.spines.values():spine.set_color('#B6C3C8');spine.set_linewidth(.7)
 scalebar(ax,bounds);north(ax)

def scene_layers(path):
 names=['source_roads','reference_target_roads','predicted_cartographx_roads','predicted_gnn_roads',
        'tp_roads','fp_roads','fn_roads','tn_roads','buildings']
 return {n:read_layer(path,n) for n in names}

def cartographic_plates(args):
 catalog=load(SCENES,'scene_catalog.csv')
 if len(catalog)!=12:raise RuntimeError(f'Expected 12 scenes, got {len(catalog)}')
 # All map panels use the same clipped projected extent; no visual distortion.
 scene_cache={}
 for _,r in catalog.iterrows():
  path=Path(r.output_gpkg)
  if not path.is_file():raise FileNotFoundError(path)
  scene_cache[str(r.scene_id)]=(r,scene_layers(path))
 kinds=['source','reference','cx','gnn','error']
 names=['A  Detailed source','B  Reference target','C  CartoGraph-X','D  Selected GNN','E  TP / FP / FN']
 for city in CITY:
  rows=[(r,lay) for r,lay in scene_cache.values() if r.city==city]
  rows.sort(key=lambda x:0 if x[0].protocol.startswith(('lt_to_lt','pl_to_pl')) else 1)
  if len(rows)!=2:raise RuntimeError(f'{city}: expected two protocols')
  fig,axes=plt.subplots(2,5,figsize=(21.0,9.25),gridspec_kw={'wspace':.055,'hspace':.13})
  for j,(r,layers) in enumerate(rows):
   bound=layers['source_roads'].total_bounds
   # Script124 clips the same square across every layer. Recover authoritative
   # original square extent from source scene metadata, never per-layer bounds.
   meta=json.loads((SCENES/'scenes'/f'{r.scene_id}.json').read_text(encoding='utf-8'))
   bounds=meta['bbox_projected_m']
   for k,kind in enumerate(kinds):
    ax=axes[j,k];map_axis(ax,layers,kind,bounds)
    if j==0:ax.set_title(names[k],fontsize=11,pad=10)
    if k==0:
     ax.text(-.04,.5,NAMES[r.protocol],rotation=90,transform=ax.transAxes,
             ha='right',va='center',fontsize=11,weight='bold')
    if k==4:
     counts=meta.get('error_counts',{})
     ax.text(.025,.98,f"TP {counts.get('TP',0)}  |  FP {counts.get('FP',0)}  |  FN {counts.get('FN',0)}",
             transform=ax.transAxes,ha='left',va='top',fontsize=7.5,
             bbox={'boxstyle':'round,pad=.28','fc':'white','ec':'#DDE3E6','alpha':.94})
  fig.suptitle(f'{CITY[city]}  |  Real-vector cartographic road-selection comparison',
               fontsize=17,fontweight='bold',y=.993)
  fig.text(.5,.038,'Reference: independently sourced generalized road geometries  •  '
           'Predictions: retained original source segments  •  '
           'Errors: provisional C2 labels  •  Same extent within each row',
           ha='center',fontsize=9,color=COL['sub'])
  handles=[Line2D([0],[0],color=COL['source'],lw=3,label='Detailed source'),
           Line2D([0],[0],color=COL['reference'],lw=3,label='Target reference'),
           Line2D([0],[0],color=COL['cx'],lw=3,label='CartoGraph-X'),
           Line2D([0],[0],color=COL['gnn'],lw=3,label='Selected GNN'),
           Line2D([0],[0],color=COL['tp'],lw=3,label='TP'),
           Line2D([0],[0],color=COL['fp'],lw=3,label='FP'),
           Line2D([0],[0],color=COL['fn'],lw=3,label='FN')]
  fig.legend(handles=handles,loc='lower center',ncol=7,frameon=False,bbox_to_anchor=(.5,.052),fontsize=8.5)
  fig.subplots_adjust(left=.043,right=.994,top=.92,bottom=.12)
  savefig(fig,f'F01_map_{city}',args.dpi,args.formats)
 # A separate six-city atlas at the SAME fixed within-row extent, one city per
 # row, source / reference / CartoGraph-X, inspired by the user's reference.
 fig,axes=plt.subplots(6,3,figsize=(14.2,23.4),gridspec_kw={'wspace':.045,'hspace':.10})
 for i,city in enumerate(CITY):
  r,layers=next((r,lay) for r,lay in scene_cache.values() if r.city==city and r.protocol in ('lt_to_lt','pl_to_pl'))
  meta=json.loads((SCENES/'scenes'/f'{r.scene_id}.json').read_text(encoding='utf-8'))
  for j,kind in enumerate(['source','reference','cx']):
   map_axis(axes[i,j],layers,kind,meta['bbox_projected_m'])
   if i==0:axes[i,j].set_title(['Detailed source','Generalized target','CartoGraph-X retained network'][j],fontsize=12,pad=12)
   if j==0:axes[i,j].text(-.035,.5,CITY[city],rotation=90,transform=axes[i,j].transAxes,
                           ha='right',va='center',fontsize=11,weight='bold')
 fig.suptitle('Six-city cartographic atlas  |  Actual source, reference and predicted road geometries',
              y=.995,fontsize=16,weight='bold')
 fig.text(.5,.016,'Source and target geometries are independently supplied; predictions are source-road selection only.',
          ha='center',fontsize=9,color=COL['sub'])
 fig.subplots_adjust(left=.07,right=.99,top=.955,bottom=.034)
 savefig(fig,'F02_six_city_source_reference_prediction_atlas',args.dpi,args.formats)


def metric_comparison(args):
 c=load(CX,'all_seed_protocol_results.csv')
 g=load(GNN,'all_seed_selected_protocol_results.csv')
 # Do not infer model score fields from means: require seed-level metrics.
 for d,label in [(c,'CartoGraph-X'),(g,'GNN')]:
  if not {'seed','protocol','test_mcc'}.issubset(d):raise RuntimeError(f'{label} missing seed-level MCC')
  if set(d.seed)!=set(range(42,52)):raise RuntimeError(f'{label}: incomplete seed set')
 c['model_family']='CartoGraph-X';g['model_family']='GNN OOF-selected'
 all_d=pd.concat([c,g],ignore_index=True)
 metric_list=[('test_mcc','MCC'),('test_balanced_accuracy','Balanced accuracy'),('test_roc_auc','ROC-AUC'),('test_pr_auc','PR-AUC')]
 available=[(col,name) for col,name in metric_list if col in c and col in g]
 if len(available)<2:raise RuntimeError('Missing secondary metrics in selected runs')
 export_table(all_d,'T01_selected_seed_level_results')
 fig,axes=plt.subplots(2,2,figsize=(15,9.5))
 x=np.arange(6);w=.35
 for ax,(metric,label) in zip(axes.flat,available):
  for j,(family,color) in enumerate([('CartoGraph-X',COL['cx']),('GNN OOF-selected',COL['gnn'])]):
   sub=all_d.loc[all_d.model_family.eq(family)]
   grouped=sub.groupby('protocol')[metric].agg(['mean','std']).reindex(PROT)
   ax.bar(x+(j-.5)*w,grouped['mean'],w,yerr=grouped['std'].fillna(0),capsize=3,
          color=color,edgecolor='white',linewidth=.7,zorder=3,label=family)
  ax.set_title(label,pad=11);ax.set_xticks(x,[NAMES[p] for p in PROT],rotation=17,ha='right')
  ax.set_ylim(bottom=min(0,float(all_d[metric].min())-.03));axis_style(ax)
  if label=='MCC':ax.axhline(0,color=COL['ink'],lw=.8)
 for ax in list(axes.flat)[len(available):]:ax.axis('off')
 fig.legend(handles=[Patch(facecolor=COL['cx'],label='CartoGraph-X'),Patch(facecolor=COL['gnn'],label='Selected GNN')],
            loc='upper right',bbox_to_anchor=(.95,.98),ncol=2,frameon=False)
 title(fig,'Discrimination across six evaluation protocols',
       'Matched seeds 42–51; mean ± SD. Historical test is exploratory.')
 fig.subplots_adjust(left=.075,right=.965,top=.88,bottom=.10,wspace=.22,hspace=.40)
 savefig(fig,'F03_six_protocol_discrimination',args.dpi,args.formats)


def calibration(args):
 scores=load(CAL,'seed_protocol_calibration_metrics.csv')
 bins=load(CAL,'reliability_bins_by_seed.csv')
 export_table(scores,'T02_probability_calibration_by_seed')
 fig,axes=plt.subplots(2,3,figsize=(15.5,9.5),sharex=True,sharey=True)
 for ax,prot in zip(axes.flat,PROT):
  for fam,color in [('CartoGraph-X',COL['cx']),('GNN OOF-selected',COL['gnn'])]:
   q=bins.loc[bins.protocol.eq(prot)&bins.model_family.eq(fam)].copy()
   # Script122 includes equal-width AND equal-frequency reliability bins.
   if 'strategy' in q:q=q.loc[q.strategy.eq('equal_width')]
   elif 'binning' in q:q=q.loc[q.binning.eq('equal_width')]
   if q.empty:continue
   q=q.loc[q.n.gt(0)]
   if q.empty:continue
   agg=q.groupby('bin',as_index=False).apply(lambda d:pd.Series({
     'x':np.average(d.mean_confidence,weights=d.n),
     'y':np.average(d.observed_positive_rate,weights=d.n)}),include_groups=False)
   agg=agg.sort_values('x')
   ax.plot(agg.x,agg.y,'-o',color=color,lw=1.9,ms=4,label=fam,zorder=3)
  ax.plot([0,1],[0,1],ls='--',color='#7C8A93',lw=1,zorder=1)
  ax.set_title(NAMES[prot]);ax.set_xlim(0,1);ax.set_ylim(0,1)
  ax.set_aspect('equal');ax.grid(color=COL['grid'],lw=.6)
  ax.set_xlabel('Predicted retention probability')
  ax.set_ylabel('Observed C2 retention fraction')
 fig.legend(handles=[Line2D([0],[0],color=COL['cx'],lw=2,label='CartoGraph-X'),
                     Line2D([0],[0],color=COL['gnn'],lw=2,label='Selected GNN'),
                     Line2D([0],[0],color='#7C8A93',ls='--',label='Perfect calibration')],
            ncol=3,loc='lower center',bbox_to_anchor=(.5,.015),frameon=False)
 title(fig,'Probability reliability  |  Six protocols',
       'Ten-seed aggregated, sample-weighted reliability bins; retrospective diagnostic, no calibrator fitted.')
 fig.subplots_adjust(left=.075,right=.98,top=.89,bottom=.12,wspace=.25,hspace=.38)
 savefig(fig,'F04_reliability_diagrams',args.dpi,args.formats)


def domain_shift(args):
 d=load(SHIFT,'table_country_split_prevalence.csv')
 transfer=load(SHIFT,'transfer_seed_diagnostics.csv')
 export_table(d,'T03_country_prevalence')
 export_table(transfer,'T04_domain_shift_seed_diagnostics')
 fig,axes=plt.subplots(1,2,figsize=(13.8,5.5))
 if {'country','split','prevalence'}.issubset(d):
  pivot=d.pivot(index='split',columns='country',values='prevalence')
  pivot=pivot.reindex(['train','validation','test'])
  for j,(country,color) in enumerate([('Lithuania',COL['reference']),('Poland',COL['fp'])]):
   axes[0].bar(np.arange(3)+(j-.5)*.35,pivot[country],.35,color=color,label=country)
  axes[0].set_xticks(range(3),['Train','Validation','Test']);axes[0].set_ylim(0,1)
  axes[0].set_ylabel('Positive road-selection prevalence');axes[0].legend(frameon=False)
 else:
  axes[0].text(.5,.5,'Prevalence schema unavailable',transform=axes[0].transAxes,ha='center')
 required={'protocol','predicted_positive_rate','prevalence'}
 if required.issubset(transfer):
  m=transfer.groupby('protocol')[['predicted_positive_rate','prevalence']].mean().reindex(PROT)
  x=np.arange(6);axes[1].bar(x-.17,m.prevalence,.34,color=COL['reference'],label='True C2 prevalence')
  axes[1].bar(x+.17,m.predicted_positive_rate,.34,color=COL['cx'],label='Predicted retained')
  axes[1].set_xticks(x,[NAMES[p] for p in PROT],rotation=30,ha='right');axes[1].set_ylim(0,1)
  axes[1].legend(frameon=False)
 axes[0].set_title('A  Country label-prevalence shift');axes[1].set_title('B  Prediction-rate mismatch')
 for ax in axes:axis_style(ax)
 title(fig,'Cross-country domain shift and selection-rate asymmetry',
       'Descriptive label and prediction differences; not evidence of causal domain-shift mechanisms.')
 fig.subplots_adjust(left=.075,right=.98,top=.84,bottom=.22,wspace=.26)
 savefig(fig,'F05_domain_shift_asymmetry',args.dpi,args.formats)


def ablation(args):
 d=load(ABLA,'ablation_matched_reference_differences.csv')
 export_table(d,'T05_architecture_ablation_validation')
 needed={'variant','protocol','delta_validation_mcc'}
 if not needed.issubset(d):raise RuntimeError(f'Ablation schema missing {needed-set(d.columns)}')
 z=d.pivot_table(index='variant',columns='protocol',values='delta_validation_mcc',aggfunc='mean').reindex(columns=PROT)
 if z.empty:return
 fig,ax=plt.subplots(figsize=(12.5,max(4.7,1.0*len(z)+1.8)))
 vmax=max(.06,float(np.nanmax(np.abs(z.to_numpy()))))
 im=ax.imshow(z.to_numpy(),cmap='RdBu',vmin=-vmax,vmax=vmax,aspect='auto')
 ax.set_xticks(range(len(z.columns)),[NAMES.get(v,v) for v in z.columns],rotation=25,ha='right')
 ax.set_yticks(range(len(z.index)),[str(v).replace('_',' ') for v in z.index])
 for i in range(len(z.index)):
  for j in range(len(z.columns)):
   v=z.iloc[i,j]
   if pd.notna(v):ax.text(j,i,f'{v:+.3f}',ha='center',va='center',fontsize=9,
                           color='white' if abs(v)>vmax*.55 else COL['ink'])
 fig.colorbar(im,ax=ax,shrink=.8,label='Validation MCC change versus frozen full model')
 title(fig,'Architecture ablation  |  Matched-seed validation differences',
       'Negative values indicate degradation when removing the component.')
 fig.subplots_adjust(left=.25,right=.96,top=.85,bottom=.20)
 savefig(fig,'F06_architecture_ablation_heatmap',args.dpi,args.formats)


def sensitivity(args):
 d=load(FRACT,'all_seed_fraction_validation.csv')
 export_table(d,'T06_training_fraction_validation')
 frac=optional_metric(d,'fraction','training_fraction')
 metric=optional_metric(d,'validation_mcc','val_mcc')
 if not frac or not metric:raise RuntimeError('Training fraction CSV missing fraction/MCC')
 fig,axes=plt.subplots(1,2,figsize=(14.2,5.8))
 for ax,prots in zip(axes,[PROT[:2]+PROT[4:],PROT[2:4]]):
  for p in prots:
   s=d.loc[d.protocol.eq(p)].groupby(frac)[metric].agg(['mean','std']).sort_index()
   ax.plot(s.index*100,s['mean'],marker='o',lw=2,label=NAMES[p])
   ax.fill_between(s.index*100,(s['mean']-s['std']).to_numpy(),(s['mean']+s['std']).to_numpy(),alpha=.12)
  ax.set_xlabel('Training nodes retained (%)');ax.set_ylabel('Validation MCC')
  ax.set_xticks([10,25,50,75,100]);ax.legend(frameon=False,fontsize=8)
  axis_style(ax)
 axes[0].set_title('A  Within-country and joint training')
 axes[1].set_title('B  Cross-country zero-shot transfer')
 title(fig,'Training-data fraction sensitivity',
       'Node-level subsampling alters both sample count and graph density; ten matched seeds.')
 fig.subplots_adjust(left=.08,right=.98,top=.84,bottom=.16,wspace=.23)
 savefig(fig,'F07_training_fraction_sensitivity',args.dpi,args.formats)


def graph_sensitivity(args):
 d=load(GRAPH,'all_seed_graph_sensitivity_validation.csv')
 export_table(d,'T07_graph_sensitivity_validation')
 k=optional_metric(d,'k','knn_k','graph_k')
 radius=optional_metric(d,'radius_m','max_distance_m','radius','graph_radius_m')
 metric=optional_metric(d,'validation_mcc','val_mcc')
 if not all((k,radius,metric)):raise RuntimeError(f'Graph sensitivity schema: {d.columns.tolist()}')
 fig,axes=plt.subplots(2,3,figsize=(14.7,9))
 for ax,p in zip(axes.flat,PROT):
  q=d.loc[d.protocol.eq(p)].pivot_table(index=k,columns=radius,values=metric,aggfunc='mean')
  im=ax.imshow(q.to_numpy(),cmap='viridis',aspect='auto',vmin=0,vmax=max(.36,float(d[metric].max())))
  ax.set_xticks(range(len(q.columns)),[f'{v:g} m' for v in q.columns]);ax.set_yticks(range(len(q.index)),[f'k={v:g}' for v in q.index])
  ax.set_title(NAMES[p]);ax.set_xlabel('Maximum neighbor radius')
  for i in range(len(q.index)):
   for j in range(len(q.columns)):
    ax.text(j,i,f'{q.iloc[i,j]:.3f}',ha='center',va='center',fontsize=8,
            color='white' if q.iloc[i,j]>.22 else COL['ink'])
  if 6 in q.index and 250 in q.columns:
   i=list(q.index).index(6);j=list(q.columns).index(250)
   ax.add_patch(Rectangle((j-.48,i-.48),.96,.96,fill=False,edgecolor='#F5A623',lw=2.6))
 fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.67,label='Validation MCC (ten-seed mean)',pad=.015)
 title(fig,'Geometric proximity-graph sensitivity',
       'k = 6, radius = 250 m is the frozen reference (gold outline), not selected retrospectively.')
 fig.subplots_adjust(left=.075,right=.87,top=.89,bottom=.08,wspace=.30,hspace=.35)
 savefig(fig,'F08_graph_topology_sensitivity',args.dpi,args.formats)


def city_robustness(args):
 d=load(GEO,'city_seed_metrics.csv')
 export_table(d,'T08_city_seed_robustness')
 metric=optional_metric(d,'mcc','test_mcc')
 if not metric:raise RuntimeError('City robustness missing MCC')
 cities=list(CITY.values())
 # City column names in CSV may use either canonical or display names.
 d['city_display']=d.city.map(lambda v:CITY.get(str(v).lower(),str(v).replace('_central','').title()))
 fig,axes=plt.subplots(1,2,figsize=(15.5,6.2))
 for ax,(country,prots) in zip(axes,[('Lithuania',['lt_to_lt','joint_to_lt','pl_to_lt_zero_shot']),
                                 ('Poland',['pl_to_pl','joint_to_pl','lt_to_pl_zero_shot'])]):
  s=d.loc[d.protocol.isin(prots)]
  order=[c for c in cities if c in s.city_display.unique()]
  x=np.arange(len(order));w=.25
  for j,p in enumerate(prots):
   q=s.loc[s.protocol.eq(p)&s.model_family.eq('CartoGraph-X')].groupby('city_display')[metric].agg(['mean','std']).reindex(order)
   ax.bar(x+(j-1)*w,q['mean'],w,yerr=q['std'],capsize=2.5,label=NAMES[p],alpha=.92)
  ax.set_xticks(x,order);ax.set_ylabel('City-level MCC');ax.set_title(country)
  ax.legend(frameon=False,fontsize=8);axis_style(ax)
 title(fig,'Geographic robustness across six held-out city test partitions',
       'Ten seeds; city subsets are not newly untouched holdouts, and C2 labels remain provisional.')
 fig.subplots_adjust(left=.075,right=.98,top=.84,bottom=.15,wspace=.21)
 savefig(fig,'F09_city_geographic_robustness',args.dpi,args.formats)


def error_fig(args):
 d=load(ERROR,'seed_protocol_confusion.csv')
 export_table(d,'T09_error_confusion_by_seed')
 required={'model_family','protocol','fpr','fnr'}
 if not required.issubset(d):raise RuntimeError(f'Confusion table missing {required-set(d.columns)}')
 fig,axes=plt.subplots(1,2,figsize=(14.4,5.7))
 for ax,(metric,label) in zip(axes,[('fpr','False-positive rate'),('fnr','False-negative rate')]):
  x=np.arange(6);w=.36
  for j,(fam,color) in enumerate([('CartoGraph-X',COL['cx']),('GNN OOF-selected',COL['gnn'])]):
   q=d.loc[d.model_family.eq(fam)].groupby('protocol')[metric].agg(['mean','std']).reindex(PROT)
   ax.bar(x+(j-.5)*w,q['mean'],w,yerr=q['std'],capsize=2.5,color=color,label=fam)
  ax.set_xticks(x,[NAMES[p] for p in PROT],rotation=25,ha='right')
  ax.set_ylim(0,1);ax.set_ylabel(label);ax.legend(frameon=False,fontsize=8);axis_style(ax)
 title(fig,'Asymmetric road-selection failures',
       'LT → PL over-selects; PL → LT under-selects. Rates computed from provisional C2 labels.')
 fig.subplots_adjust(left=.075,right=.98,top=.83,bottom=.24,wspace=.25)
 savefig(fig,'F10_false_positive_negative_profiles',args.dpi,args.formats)


def summary_tables():
 specs=[(STATS,'paired_seed_classical_vs_cartographx_mcc.csv','T10_classical_paired_statistics'),
        (GNN,'table_cartographx_vs_gnn_mcc.csv','T11_gnn_comparison'),
        (CAL,'table_calibration_mean_std.csv','T12_calibration_summary'),
        (ERROR,'table_protocol_errors_mean_std.csv','T13_error_summary'),
        (GRAPH,'table_graph_sensitivity_deltas.csv','T14_graph_sensitivity_deltas'),
        (FRACT,'table_fraction_mcc_deltas.csv','T15_fraction_deltas'),
        (GEO,'table_spatial_tile_bootstrap_intervals.csv','T16_spatial_bootstrap_intervals'),
        (SHIFT,'table_transfer_asymmetry_mean_std.csv','T17_transfer_asymmetry')]
 for folder,filename,out in specs:
  d=load(folder,filename,required=False)
  if d is not None:export_table(d,out)
 # Excel workbook is an index of already-computed, unmodified CSV tables.
 try:
  from openpyxl import Workbook
  wb=Workbook();ws=wb.active;ws.title='Contents'
  ws.append(['Table','CSV path','Rows','Columns'])
  for item in MANIFEST['tables']:
   d=pd.read_csv(OUT/item)
   ws.append([Path(item).stem,item,len(d),len(d.columns)])
  ws.column_dimensions['A'].width=48;ws.column_dimensions['B'].width=68
  ws.column_dimensions['C'].width=12;ws.column_dimensions['D'].width=12
  wb.save(OUT/'tables'/'publication_table_index.xlsx')
 except ImportError:
  MANIFEST['warnings'].append('openpyxl unavailable; CSV tables exported, Excel index skipped')


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--dpi',type=int,default=600,help='PNG DPI; vector SVG/PDF remain resolution-independent')
 ap.add_argument('--formats',nargs='+',choices=['pdf','svg','png'],default=['pdf','svg','png'])
 ap.add_argument('--maps-only',action='store_true')
 ap.add_argument('--charts-only',action='store_true')
 args=ap.parse_args()
 if args.maps_only and args.charts_only:ap.error('Choose maps-only OR charts-only')
 if args.dpi<100 or args.dpi>1200:ap.error('DPI must be 100..1200')
 OUT.mkdir(parents=True,exist_ok=True)
 if not args.charts_only:
  print('Rendering genuine georeferenced road-network map plates...',flush=True)
  cartographic_plates(args)
 if not args.maps_only:
  for func in [metric_comparison,calibration,domain_shift,ablation,sensitivity,
               graph_sensitivity,city_robustness,error_fig]:
   print('Rendering',func.__name__,flush=True)
   func(args)
  summary_tables()
 MANIFEST['completed_utc']=datetime.now(timezone.utc).isoformat()
 MANIFEST['figure_count']=len(set(Path(x).stem for x in MANIFEST['figure_files']))
 MANIFEST['table_count']=len(MANIFEST['tables'])
 (OUT/'final_publication_manifest.json').write_text(json.dumps(MANIFEST,indent=2,ensure_ascii=False),encoding='utf-8')
 (OUT/'README_PUBLICATION_STATUS.txt').write_text(
  'FINAL SCRIPT 125 — CARTOGRAPH-X V2 PUBLICATION FIGURES\n\n'
  'Figures: PDF + SVG editable vector graphics; PNG high-resolution raster.\n'
  'Maps: original projected road geometry, independently sourced reference target,\n'
  'and selected original road segments from CartoGraph-X / selected GNN.\n'
  'Buildings are original contextual geometries where available.\n\n'
  'CRITICAL: CartoGraph-X V2 predicts road retention only, not line simplification,\n'
  'displacement, road merging, topology reconstruction or building generalization.\n'
  'C2 correspondence labels are provisional and NOT manually validated.\n'
  'Historical test partition was explored in earlier experiments: all test\n'
  'comparisons are exploratory, not independent confirmatory claims.\n'
  'Graph edges represent geometric proximity kNN, NOT actual road connectivity.\n'
  'Training-fraction sensitivity uses node-level subsampling.\n'
  'Seed-level paired statistical tests reuse the same geographic test roads.\n'
  'NO additional scripts are planned after Script 125.\n',encoding='utf-8')
 print(f'\nSCRIPT 125 PASS: {MANIFEST["figure_count"]} distinct figures, {MANIFEST["table_count"]} CSV tables')
 print('Saved:',OUT)
 print('This is the FINAL numbered code in the CartoGraph-X V2 pipeline.')

if __name__=='__main__':main()
