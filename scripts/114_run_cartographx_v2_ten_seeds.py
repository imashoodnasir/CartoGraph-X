#!/usr/bin/env python3
"""Run fixed Script 112 across matched seeds 42..51 and verify/reduce outputs.

Usage:
 python scripts/114_run_cartographx_v2_ten_seeds.py
 python scripts/114_run_cartographx_v2_ten_seeds.py --seeds 42 43 --force
 python scripts/114_run_cartographx_v2_ten_seeds.py --summarize-only

Existing runs are reused only when source/dependency hashes and configuration
are verified. Seed 42 predating this runner must be rerun once to establish
runner provenance. Historical test results remain exploratory.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[1]
S112=ROOT/'scripts/112_train_cartographx_v2_architecture.py'
S111=ROOT/'scripts/111_train_benchmark_v2_weighted_gnn_baselines.py'
NODES=ROOT/'data/processed/cross_country/benchmark_v2_unified_labels_v1/road_graph_nodes_v2.csv'
EDGES=ROOT/'data/processed/cross_country/graph_representations/road_graph_edges.csv'
BASE=ROOT/'data/processed/cross_country/cartographx_v2_architecture_v1'
OUT=ROOT/'data/processed/cross_country/cartographx_v2_ten_seed_summary_v1'
PROTOCOLS=('lt_to_lt','pl_to_pl','lt_to_pl_zero_shot','pl_to_lt_zero_shot','joint_to_lt','joint_to_pl')
ARTIFACTS=('spatial_cv_country_diagnostics.csv','selected_protocol_results.csv','selected_test_predictions.csv')
METRICS=('mcc','balanced_accuracy','f1','roc_auc','pr_auc')

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def verify(seed, cfg, hashes, strict=True):
 folder=BASE/f'seed_{seed}'; manifest_file=folder/'experiment_manifest.json'
 if not manifest_file.exists():raise RuntimeError(f'Seed {seed}: missing manifest')
 m=json.loads(manifest_file.read_text(encoding='utf-8'))
 for k,v in [('seed',seed),*cfg.items()]:
  if k not in m or (isinstance(v,float) and not np.isclose(float(m[k]),v,atol=1e-12)) or (not isinstance(v,float) and m[k]!=v):
   raise RuntimeError(f'Seed {seed}: manifest {k} mismatch: {m.get(k)} != {v}')
 for key,h in [('script_sha256',hashes['script112']),('dependency_script_111_sha256',hashes['script111']),('node_sha256',hashes['nodes']),('edge_sha256',hashes['edges'])]:
  if m.get(key)!=h:raise RuntimeError(f'Seed {seed}: {key} mismatch')
 marker=folder/'script114_run_config.json'
 if strict:
  if not marker.exists():raise RuntimeError(f'Seed {seed}: missing runner provenance marker; use --force to regenerate')
  old=json.loads(marker.read_text(encoding='utf-8'))
  if old.get('config')!=cfg or old.get('hashes')!=hashes:raise RuntimeError(f'Seed {seed}: runner provenance mismatch')
 frames=[]
 for name in ARTIFACTS:
  p=folder/name
  if not p.exists() or m.get('outputs_sha256',{}).get(name)!=digest(p):raise RuntimeError(f'Seed {seed}: output missing/hash mismatch: {name}')
  frames.append(pd.read_csv(p,low_memory=False))
 cv,res,pred=frames
 if set(res.protocol)!=set(PROTOCOLS) or len(res)!=6:raise RuntimeError(f'Seed {seed}: protocols invalid')
 if set(pred.protocol)!=set(PROTOCOLS) or pred.duplicated(['protocol','sample_uid']).any():raise RuntimeError(f'Seed {seed}: prediction UIDs invalid')
 if set(cv.protocol)!=set(PROTOCOLS):raise RuntimeError(f'Seed {seed}: CV protocols invalid')
 for f in frames:
  if not f.seed.eq(seed).all():raise RuntimeError(f'Seed {seed}: inconsistent seed')
 for _,r in res.iterrows():
  p=pred.loc[pred.protocol.eq(r.protocol)]
  if len(p)!=int(r.test_n) or not np.allclose(p.threshold,float(r.threshold),atol=1e-10):raise RuntimeError(f'Seed {seed}/{r.protocol}: prediction metadata invalid')
  y=p.label.to_numpy(dtype=int);yh=p.prediction.to_numpy(dtype=int);probs=p.probability.to_numpy(dtype=float)
  if not np.isfinite(probs).all() or (probs<0).any() or (probs>1).any():raise RuntimeError(f'Seed {seed}/{r.protocol}: invalid probabilities')
  if not np.array_equal(yh,(probs>=float(r.threshold)).astype(int)):raise RuntimeError(f'Seed {seed}/{r.protocol}: threshold mismatch')
  met=dict(mcc=matthews_corrcoef(y,yh),balanced_accuracy=balanced_accuracy_score(y,yh),f1=f1_score(y,yh,zero_division=0),roc_auc=roc_auc_score(y,probs),pr_auc=average_precision_score(y,probs))
  for k,v in met.items():
   if not np.isclose(float(r['test_'+k]),v,atol=1e-9,rtol=1e-9):raise RuntimeError(f'Seed {seed}/{r.protocol}: {k} mismatch')
  if r.protocol.endswith('zero_shot') and r.threshold_selection_country!=r.source:raise RuntimeError(f'Seed {seed}/{r.protocol}: zero-shot selection violation')
  if not r.protocol.endswith('zero_shot') and r.threshold_selection_country!=r.target:raise RuntimeError(f'Seed {seed}/{r.protocol}: threshold country mismatch')
 return cv,res,pred

def summarize(all_cv,all_res,all_pred,seeds,hashes,cfg):
 OUT.mkdir(parents=True,exist_ok=True)
 cv=pd.concat(all_cv,ignore_index=True);res=pd.concat(all_res,ignore_index=True);pred=pd.concat(all_pred,ignore_index=True)
 cv.to_csv(OUT/'all_seed_spatial_cv_country_diagnostics.csv',index=False)
 res.to_csv(OUT/'all_seed_protocol_results.csv',index=False)
 pred.to_csv(OUT/'all_seed_selected_test_predictions.csv',index=False)
 fields=['test_'+k for k in METRICS]+['validation_mcc','cv_mcc','threshold']
 agg=res.groupby('protocol',sort=False)[fields].agg(['mean','std','min','max'])
 agg.columns=['_'.join(x) for x in agg.columns]
 agg=agg.reset_index();agg.insert(1,'n_seeds',len(seeds))
 agg.to_csv(OUT/'protocol_mean_std_min_max.csv',index=False)
 # Country-specific joint OOF diagnostics, not pooled MCC.
 joint_cv=cv[cv.protocol.isin(['joint_to_lt','joint_to_pl'])].copy()
 joint_cv.groupby(['protocol','country'],as_index=False).agg(cv_mcc_mean=('cv_mcc','mean'),cv_mcc_std=('cv_mcc','std'),cv_auc_mean=('cv_roc_auc','mean'),cv_auc_std=('cv_roc_auc','std'),n_seeds=('seed','nunique')).to_csv(OUT/'joint_country_cv_summary.csv',index=False)
 pred.groupby(['protocol','seed'],as_index=False).agg(actual_positive_rate=('label','mean'),predicted_positive_rate=('prediction','mean'),mean_probability=('probability','mean')).to_csv(OUT/'prediction_distribution_by_seed.csv',index=False)
 # Paired seed differences; not statistical independence of test data.
 baseline=ROOT/'data/processed/cross_country/benchmark_v2_ten_seed_summary_v1/all_seed_protocol_results.csv'
 if baseline.is_file():
  b=pd.read_csv(baseline)
  if {'seed','protocol','test_mcc'}.issubset(b.columns):
   a=res[['seed','protocol','test_mcc']].rename(columns={'test_mcc':'cartographx_test_mcc'})
   b=b[['seed','protocol','test_mcc']].rename(columns={'test_mcc':'classical_test_mcc'})
   joined=a.merge(b,on=['seed','protocol'],how='inner',validate='one_to_one')
   if len(joined)!=len(res):raise RuntimeError('Matched classical comparison incomplete; refusing partial comparison')
   joined['delta_mcc']=joined.cartographx_test_mcc-joined.classical_test_mcc
   joined.to_csv(OUT/'paired_seed_classical_mcc_differences.csv',index=False)
   joined.groupby('protocol',as_index=False).agg(delta_mean=('delta_mcc','mean'),delta_std=('delta_mcc','std'),n_seeds=('seed','nunique')).to_csv(OUT/'paired_mcc_difference_summary.csv',index=False)
 else:print('Classical Script 108 summary not found; paired comparison skipped.',flush=True)
 manifest={'script':Path(__file__).name,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'seeds':seeds,'config':cfg,'hashes':hashes,'status':'EXPLORATORY_NOT_UNTOUCHED_HOLDOUT','c2_labels':'PROVISIONAL_NOT_MANUALLY_VALIDATED','note':'Seed-level SD reflects training randomness with fixed splits; seeds are not independent test samples','output_sha256':{p.name:digest(p) for p in OUT.glob('*.csv')}}
 (OUT/'ten_seed_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print('\nTEN-SEED EXPLORATORY TEST SUMMARY',flush=True)
 print(agg[['protocol','n_seeds','test_mcc_mean','test_mcc_std','test_balanced_accuracy_mean','test_roc_auc_mean','test_pr_auc_mean']].to_string(index=False,float_format=lambda x:f'{x:.4f}'),flush=True)
 print(f'\nSaved: {OUT}\nEXPLORATORY ONLY: historical test previously examined; C2 labels provisional.',flush=True)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--seeds',type=int,nargs='+',default=list(range(42,52)))
 ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--folds',type=int,default=3)
 ap.add_argument('--hidden',type=int,default=64);ap.add_argument('--dropout',type=float,default=.2)
 ap.add_argument('--lr',type=float,default=.001);ap.add_argument('--weight-decay',type=float,default=1e-4)
 ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
 ap.add_argument('--force',action='store_true');ap.add_argument('--summarize-only',action='store_true')
 args=ap.parse_args()
 if len(set(args.seeds))!=len(args.seeds) or not args.seeds:ap.error('Seeds must be unique and nonempty')
 if args.epochs<1 or args.folds<2 or args.hidden<1:ap.error('Invalid training configuration')
 for p in (S112,S111,NODES,EDGES):
  if not p.is_file():ap.error(f'Missing required file: {p}')
 import torch
 device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
 if device=='cuda' and not torch.cuda.is_available():ap.error('CUDA unavailable')
 cfg={'epochs':args.epochs,'folds':args.folds,'hidden':args.hidden,'dropout':args.dropout,'lr':args.lr,'weight_decay':args.weight_decay,'device':device}
 hashes={'script112':digest(S112),'script111':digest(S111),'nodes':digest(NODES),'edges':digest(EDGES)}
 cvs=[];results=[];preds=[]
 for seed in args.seeds:
  folder=BASE/f'seed_{seed}'
  try:
   if args.force and not args.summarize_only:raise RuntimeError('Forced regeneration')
   cv,res,pred=verify(seed,cfg,hashes,strict=True)
   print(f'REUSED verified seed {seed}',flush=True)
  except Exception as exc:
   if args.summarize_only:raise RuntimeError(f'Seed {seed} cannot be summarized: {exc}') from exc
   print(f'\n=== Training seed {seed} ({exc}) ===',flush=True)
   cmd=[sys.executable,str(S112),'--seed',str(seed),'--epochs',str(args.epochs),'--folds',str(args.folds),'--hidden',str(args.hidden),'--dropout',str(args.dropout),'--lr',str(args.lr),'--weight-decay',str(args.weight_decay),'--device',device]
   subprocess.run(cmd,cwd=ROOT,check=True)
   # Verify generated outputs before stamping provenance.
   cv,res,pred=verify(seed,cfg,hashes,strict=False)
   (folder/'script114_run_config.json').write_text(json.dumps({'config':cfg,'hashes':hashes},indent=2),encoding='utf-8')
   cv,res,pred=verify(seed,cfg,hashes,strict=True)
   print(f'VERIFIED seed {seed}: six protocols, country CV, test predictions',flush=True)
  cvs.append(cv);results.append(res);preds.append(pred)
 summarize(cvs,results,preds,args.seeds,hashes,cfg)

if __name__=='__main__':main()
