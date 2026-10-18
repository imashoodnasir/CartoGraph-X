#!/usr/bin/env python3
"""Script 120 — Prespecified geometric kNN graph sensitivity, validation only.

Grid: k={3,6,10} x radius={150,250,400} meters. Fixed CartoGraph-X V2
architecture and hyperparameters, seeds 42–51. Fits each country-only model
and each joint model once per seed/config; evaluates six protocols on validation.
For zero-shot: threshold is chosen ONLY from source-country validation, and
applied unchanged to target-country validation (target labels are metrics only).
Joint models use each target country's validation for threshold selection.
No historical test predictions or test labels accessed; no publication figures.
Geometric proximity kNN is NOT true road-network topology.

Run:
 python scripts/120_cartographx_v2_graph_sensitivity.py --seeds 42
 python scripts/120_cartographx_v2_graph_sensitivity.py
 python scripts/120_cartographx_v2_graph_sensitivity.py --summarize-only
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
S111=ROOT/'scripts/111_train_benchmark_v2_weighted_gnn_baselines.py'
S112=ROOT/'scripts/112_train_cartographx_v2_architecture.py'
OUT=ROOT/'data/processed/cross_country/cartographx_v2_graph_sensitivity_v1'
KS=(3,6,10); RADII=(150,250,400); SEEDS=tuple(range(42,52))
PROTOCOLS=('lt_to_lt','pl_to_pl','lt_to_pl_zero_shot','pl_to_lt_zero_shot','joint_to_lt','joint_to_pl')
COLS=('sample_uid','country','city','split','tile_id','retained','centroid_x','centroid_y')
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def atomic(df,path):
 tmp=path.with_suffix('.tmp');df.to_csv(tmp,index=False);tmp.replace(path)
def module(path,name):
 if not path.is_file():raise FileNotFoundError(path)
 spec=importlib.util.spec_from_file_location(name,path)
 mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
 return mod

def graph_edges(df,k,radius):
 """Deterministic tile-local undirected kNN, capped by projected metric distance."""
 pairs=set()
 for _,g in df.groupby(['country','split','tile_id'],sort=False):
  ids=g.index.to_numpy(dtype=int)
  if len(ids)<2:continue
  coords=g[['centroid_x','centroid_y']].to_numpy(dtype=float)
  if not np.isfinite(coords).all():raise RuntimeError('Missing/nonfinite projected centroid')
  tree=cKDTree(coords)
  dist,near=tree.query(coords,k=min(k+1,len(ids)),distance_upper_bound=radius)
  dist=np.atleast_2d(dist) if dist.ndim==1 else dist
  near=np.atleast_2d(near) if near.ndim==1 else near
  for i in range(len(ids)):
   for d,j in zip(dist[i],near[i]):
    if not np.isfinite(d) or j>=len(ids) or i==j:continue
    a,b=sorted((int(ids[i]),int(ids[j])))
    pairs.add((a,b))
 if not pairs:raise RuntimeError('No graph edges constructed')
 uv=np.asarray(sorted(pairs),dtype=int)
 for field in ['country','split','tile_id']:
  v=df[field].astype(str).to_numpy()
  if not np.array_equal(v[uv[:,0]],v[uv[:,1]]):raise RuntimeError(f'Edge crosses {field}')
 xy=df[['centroid_x','centroid_y']].to_numpy(dtype=float)
 distances=np.linalg.norm(xy[uv[:,0]]-xy[uv[:,1]],axis=1)
 if np.any(distances>radius+1e-7):raise RuntimeError('Radius cap violation')
 return pd.DataFrame({'source_uid':df.sample_uid.to_numpy()[uv[:,0]],
                      'target_uid':df.sample_uid.to_numpy()[uv[:,1]],
                      'country':df.country.to_numpy()[uv[:,0]],
                      'city':df.city.to_numpy()[uv[:,0]],
                      'split':df.split.to_numpy()[uv[:,0]],
                      'tile_id':df.tile_id.to_numpy()[uv[:,0]],
                      'distance_m':distances})

def verify_seed(seed,config_hash):
 folder=OUT/f'seed_{seed}'
 manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
 if manifest['config_hash']!=config_hash or manifest['seed']!=seed:raise RuntimeError('Manifest configuration mismatch')
 path=folder/'validation_results.csv'
 if digest(path)!=manifest['results_sha256']:raise RuntimeError('Result checksum mismatch')
 frame=pd.read_csv(path)
 if len(frame)!=len(KS)*len(RADII)*len(PROTOCOLS):raise RuntimeError('Missing sensitivity runs')
 if frame.duplicated(['k','radius_m','protocol']).any():raise RuntimeError('Duplicate sensitivity runs')
 if not set(frame.protocol)==set(PROTOCOLS):raise RuntimeError('Missing protocol')
 if not frame.seed.eq(seed).all():raise RuntimeError('Seed mismatch')
 return frame

def run_seed(seed,df,b,m,args,config_hash):
 folder=OUT/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
 lttrain=np.flatnonzero((df.split=='train')&(df.country=='Lithuania'))
 pltrain=np.flatnonzero((df.split=='train')&(df.country=='Poland'))
 jointtrain=np.flatnonzero(df.split=='train')
 ltval=np.flatnonzero((df.split=='validation')&(df.country=='Lithuania'))
 plval=np.flatnonzero((df.split=='validation')&(df.country=='Poland'))
 if min(map(len,[lttrain,pltrain,jointtrain,ltval,plval]))<1:raise RuntimeError('Missing partition')
 rows=[]
 for k in KS:
  for radius in RADII:
   edges=graph_edges(df,k,radius)
   graph=b.GraphData(df,edges)
   print(f'\nseed={seed} k={k} radius={radius}m edges={len(edges):,}',flush=True)
   # Fixed seeds match Script112 full-training seed policy.
   (ltp,lt_to_plp)=m.train_predict(graph,lttrain,[ltval,plval],args,seed+99173,False)
   (plp,pl_to_ltp)=m.train_predict(graph,pltrain,[plval,ltval],args,seed+99173,False)
   (jointlt,jointpl)=m.train_predict(graph,jointtrain,[ltval,plval],args,seed+99173,True)
   # Strict source-only threshold for zero-shot. No target validation tuning.
   lt_t=b.threshold_for(graph.labels[ltval],ltp)
   pl_t=b.threshold_for(graph.labels[plval],plp)
   jointlt_t=b.threshold_for(graph.labels[ltval],jointlt)
   jointpl_t=b.threshold_for(graph.labels[plval],jointpl)
   items=[('lt_to_lt','Lithuania','Lithuania',ltval,ltp,lt_t,'Lithuania'),
          ('pl_to_pl','Poland','Poland',plval,plp,pl_t,'Poland'),
          ('lt_to_pl_zero_shot','Lithuania','Poland',plval,lt_to_plp,lt_t,'Lithuania'),
          ('pl_to_lt_zero_shot','Poland','Lithuania',ltval,pl_to_ltp,pl_t,'Poland'),
          ('joint_to_lt','joint','Lithuania',ltval,jointlt,jointlt_t,'Lithuania'),
          ('joint_to_pl','joint','Poland',plval,jointpl,jointpl_t,'Poland')]
   for protocol,source,target,ids,p,t,threshold_country in items:
    met=b.metrics(graph.labels[ids],p,t)
    rows.append(dict(seed=seed,k=k,radius_m=radius,protocol=protocol,source=source,target=target,
     validation_n=len(ids),validation_prevalence=float(graph.labels[ids].mean()),
     predicted_positive_rate=float(np.mean(p>=t)),threshold=t,
     threshold_selection_country=threshold_country,
     graph_edges=len(edges),**{'validation_'+x:y for x,y in met.items()}))
    print(f'  {protocol:22s} VAL MCC={met["mcc"]:.4f} threshold={t:.2f}',flush=True)
 frame=pd.DataFrame(rows)
 atomic(frame,folder/'validation_results.csv')
 manifest={'seed':seed,'config_hash':config_hash,'results_sha256':digest(folder/'validation_results.csv'),
           'script120_sha256':digest(Path(__file__)),
           'script111_sha256':digest(S111),'script112_sha256':digest(S112),
           'nodes_sha256':digest(b.DATA),'timestamp_utc':datetime.now(timezone.utc).isoformat(),
           'scope':'validation-only prespecified graph kNN sensitivity; no test data loaded',
           'zero_shot':'source-validation threshold only; target-validation labels used for reporting',
           'labels':'C2 provisional','figures_generated':0}
 (folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 return verify_seed(seed,config_hash)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS))
 ap.add_argument('--epochs',type=int,default=60)
 ap.add_argument('--hidden',type=int,default=64)
 ap.add_argument('--dropout',type=float,default=.2)
 ap.add_argument('--lr',type=float,default=.001)
 ap.add_argument('--weight-decay',type=float,default=1e-4)
 ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
 ap.add_argument('--force',action='store_true')
 ap.add_argument('--summarize-only',action='store_true')
 args=ap.parse_args()
 if len(set(args.seeds))!=len(args.seeds) or not args.seeds:ap.error('Seeds must be unique')
 if args.force and args.summarize_only:ap.error('--force conflicts with --summarize-only')
 if args.epochs<1 or args.hidden<1:ap.error('Invalid epochs/hidden')
 import torch
 if args.device=='auto':args.device='cuda' if torch.cuda.is_available() else 'cpu'
 if args.device=='cuda' and not torch.cuda.is_available():ap.error('CUDA unavailable')
 b=module(S111,'cartographx_s120_base')
 m=module(S112,'cartographx_s120_arch')
 if not b.DATA.is_file():raise FileNotFoundError(b.DATA)
 df=pd.read_csv(b.DATA,low_memory=False)
 if not set(COLS).issubset(df.columns):raise RuntimeError(f'Missing required node columns: {set(COLS)-set(df.columns)}')
 if df.sample_uid.duplicated().any():raise RuntimeError('Duplicate UIDs')
 if not df.retained.isin([0,1]).all():raise RuntimeError('Invalid labels')
 if (df.groupby(['country','tile_id']).split.nunique()>1).any():raise RuntimeError('Tile crosses split')
 # IMPORTANT: test rows are explicitly excluded from ALL sensitivity operations.
 df=df.loc[df.split.isin(['train','validation'])].reset_index(drop=True)
 for field in ['centroid_x','centroid_y',*b.FEATURES]:
  df[field]=pd.to_numeric(df[field],errors='raise')
 if not np.isfinite(df[['centroid_x','centroid_y']].to_numpy()).all():
  raise RuntimeError('Missing centroids')
 config={'k':KS,'radius_m':RADII,'epochs':args.epochs,'hidden':args.hidden,
  'dropout':args.dropout,'lr':args.lr,'weight_decay':args.weight_decay,
  'device':args.device,'script120_sha256':digest(Path(__file__)),
  'script111_sha256':digest(S111),'script112_sha256':digest(S112),
  'nodes_sha256':digest(b.DATA)}
 config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
 OUT.mkdir(parents=True,exist_ok=True)
 results=[]
 for seed in args.seeds:
  print(f'\n===== SCRIPT 120 SEED {seed} =====',flush=True)
  if not args.force:
   try:
    frame=verify_seed(seed,config_hash)
    print(f'REUSED VERIFIED seed {seed}',flush=True)
    results.append(frame);continue
   except (OSError,KeyError,RuntimeError,ValueError,TypeError) as e:
    if args.summarize_only:raise RuntimeError(f'Seed {seed} not verified: {e}') from e
    print(f'TRAINING seed {seed}: {e}',flush=True)
  elif args.summarize_only:raise RuntimeError('Incompatible flags')
  results.append(run_seed(seed,df,b,m,args,config_hash))
 combined=pd.concat(results,ignore_index=True)
 atomic(combined,OUT/'all_seed_graph_sensitivity_validation.csv')
 keys=['k','radius_m','protocol']
 metric_cols=['validation_mcc','validation_balanced_accuracy','validation_f1',
              'validation_roc_auc','validation_pr_auc','predicted_positive_rate']
 summary=combined.groupby(keys,sort=False)[metric_cols].agg(['mean','std','min','max'])
 summary.columns=['_'.join(c) for c in summary.columns]
 summary=summary.reset_index()
 summary.insert(3,'n_seeds',len(args.seeds))
 atomic(summary,OUT/'table_graph_sensitivity_mean_std.csv')
 reference=combined.loc[(combined.k==6)&(combined.radius_m==250),
                         ['seed','protocol','validation_mcc']].rename(columns={'validation_mcc':'reference_k6_r250_mcc'})
 matched=combined.merge(reference,on=['seed','protocol'],how='left',validate='many_to_one')
 if matched.reference_k6_r250_mcc.isna().any():raise RuntimeError('Missing reference')
 matched['delta_mcc_vs_k6_r250']=matched.validation_mcc-matched.reference_k6_r250_mcc
 atomic(matched,OUT/'paired_seed_graph_sensitivity_vs_reference.csv')
 delta=matched.groupby(keys,sort=False).agg(
  delta_mcc_mean=('delta_mcc_vs_k6_r250','mean'),
  delta_mcc_std=('delta_mcc_vs_k6_r250','std'),
  n_seeds=('seed','nunique')).reset_index()
 atomic(delta,OUT/'table_graph_sensitivity_deltas.csv')
 manifest={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'config':config,
   'config_hash':config_hash,'seeds':args.seeds,'test_partition_used':False,
   'threshold_policy':'source validation for zero shot, target validation for within/joint',
   'graph_semantics':'geometric proximity kNN, not actual road topology',
   'labels':'C2 provisional; manual review pending',
   'no_hyperparameter_selection':'sensitivity only; frozen k6/r250 reference',
   'figures_generated':0,'outputs_sha256':{f.name:digest(f) for f in OUT.glob('*.csv')}}
 (OUT/'summary_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print('\nGRAPH SENSITIVITY DELTA MCC (versus frozen k=6, radius=250m):')
 print(delta.to_string(index=False,float_format=lambda v:f'{v:.4f}'))
 print(f'\nSCRIPT 120 PASS: {len(args.seeds)} seeds, {len(combined)} validation evaluations.')
 print('Saved:',OUT)
 print('No test data, no figures. NEXT: Script 121 training fractions. Script 125 final figure code.')

if __name__=='__main__':main()
