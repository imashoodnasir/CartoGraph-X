#!/usr/bin/env python3
"""Script 117: locked ten-seed GNN baselines (GCN, GraphSAGE, GAT x weighted/unweighted).

Each of the six fixed candidates is evaluated under all six protocols.
Train-only spatial OOF scores are diagnostic; validation thresholds follow
the original source-only zero-shot and target-country joint rules.
Historical test is EXPLORATORY, never used to select models or thresholds.
No change to Script 111 is required.

Usage:
 python scripts/117_run_benchmark_v2_ten_seed_gnn_baselines.py --seeds 42
 python scripts/117_run_benchmark_v2_ten_seed_gnn_baselines.py
 python scripts/117_run_benchmark_v2_ten_seed_gnn_baselines.py --summarize-only
 python scripts/117_run_benchmark_v2_ten_seed_gnn_baselines.py --seeds 42 --force
"""
from __future__ import annotations
import argparse, importlib.util, json, hashlib, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import matthews_corrcoef

ROOT=Path(__file__).resolve().parents[1]
S111=ROOT/'scripts/111_train_benchmark_v2_weighted_gnn_baselines.py'
OUT=ROOT/'data/processed/cross_country/benchmark_v2_ten_seed_gnn_baselines_v1'
SEEDS=tuple(range(42,52))
PROTOCOLS=('lt_to_lt','pl_to_pl','lt_to_pl_zero_shot','pl_to_lt_zero_shot','joint_to_lt','joint_to_pl')
CANDIDATES=tuple((kind,variant) for kind in ('GCN','GraphSAGE','GAT') for variant in ('unweighted','weighted'))
PER_SEED=('all_candidate_protocol_results.csv','spatial_oof_candidate_results.csv',
          'all_candidate_test_predictions.csv','selected_protocol_results.csv')

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def atomic_csv(frame,path):
 tmp=path.with_suffix('.tmp')
 frame.to_csv(tmp,index=False)
 tmp.replace(path)

def load_dependency():
 if not S111.is_file():raise FileNotFoundError(f'Required unchanged Script 111: {S111}')
 spec=importlib.util.spec_from_file_location('cartographx_script111',S111)
 mod=importlib.util.module_from_spec(spec)
 sys.modules[spec.name]=mod
 spec.loader.exec_module(mod)
 return mod

def train_seed(seed,args,b,df,graph):
 folder=OUT/f'seed_{seed}'
 folder.mkdir(parents=True,exist_ok=True)
 b.seed_all(seed)
 runs=[];oof_rows=[];pred_rows=[];selected=[]
 train_cache={}
 for protocol,(source,target,joint) in b.PROTOCOLS.items():
  key='joint' if joint else source
  train_ids=np.flatnonzero((df.split=='train') & (df.country.isin(['Lithuania','Poland']) if joint else df.country.eq(source)))
  validation_country=target if joint or source==target else source
  val_ids=np.flatnonzero((df.split=='validation') & df.country.eq(validation_country))
  test_ids=np.flatnonzero((df.split=='test') & df.country.eq(target))
  if min(len(train_ids),len(val_ids),len(test_ids))==0:raise RuntimeError(f'Empty partition: {protocol}')
  if key not in train_cache:
   y=graph.labels[train_ids]
   groups=(df.iloc[train_ids].country.astype(str)+'::'+df.iloc[train_ids].tile_id.astype(str)).to_numpy()
   splitter=StratifiedGroupKFold(n_splits=args.folds,shuffle=True,random_state=seed)
   folds=list(splitter.split(train_ids,y,groups))
   cv_data={}
   for kind,variant in CANDIDATES:
    oof=np.full(len(train_ids),np.nan,dtype=float)
    for fold,(tr,va) in enumerate(folds):
     if len(np.unique(y[tr]))<2 or len(np.unique(y[va]))<2:raise RuntimeError(f'One-class fold: {key}/{fold}')
     if set(groups[tr]) & set(groups[va]):raise RuntimeError('Spatial CV group leakage')
     (prob,)=b.train_predict(graph,train_ids[tr],[train_ids[va]],kind,args,seed+fold*1009,variant)
     oof[va]=prob
    if not np.isfinite(oof).all():raise RuntimeError('Incomplete OOF')
    t=b.threshold_for(y,oof)
    met=b.metrics(y,oof,t)
    cv_data[(kind,variant)]={'oof_mcc':met['mcc'],'oof_roc_auc':met['roc_auc'],
                            'oof_threshold':t,'oof_mcc_at_0_5':float(matthews_corrcoef(y,oof>=.5))}
    print(f'  {key:10s} {kind:10s} {variant:10s} OOF_MCC={met["mcc"]:.4f}',flush=True)
   train_cache[key]=(train_ids,cv_data)
  _,cv_data=train_cache[key]
  for kind,variant in CANDIDATES:
   cv=cv_data[(kind,variant)]
   oof_rows.append({'seed':seed,'protocol':protocol,'training_population':key,'model':kind,
                    'variant':variant,'folds':args.folds,'train_n':len(train_ids),**cv})
   # Train fixed candidate on complete training partition; validation calibrates threshold.
   val_prob,test_prob=b.train_predict(graph,train_ids,[val_ids,test_ids],kind,args,seed+99173,variant)
   t=b.threshold_for(graph.labels[val_ids],val_prob)
   vm=b.metrics(graph.labels[val_ids],val_prob,t)
   tm=b.metrics(graph.labels[test_ids],test_prob,t)
   record={'seed':seed,'protocol':protocol,'source':source,'target':target,'joint_training':joint,
           'model':kind,'variant':variant,'training_population':key,'train_n':len(train_ids),
           'validation_n':len(val_ids),'test_n':len(test_ids),'threshold':t,
           'threshold_selection_country':validation_country,**cv,
           **{'validation_'+k:v for k,v in vm.items()},
           **{'test_'+k:v for k,v in tm.items()}}
   runs.append(record)
   pred_rows.append(pd.DataFrame({'seed':seed,'protocol':protocol,'model':kind,'variant':variant,
       'sample_uid':df.iloc[test_ids].sample_uid.to_numpy(),
       'country':df.iloc[test_ids].country.to_numpy(),'city':df.iloc[test_ids].city.to_numpy(),
       'tile_id':df.iloc[test_ids].tile_id.to_numpy(),'split':'test',
       'label':graph.labels[test_ids],'probability':test_prob,
       'prediction':(test_prob>=t).astype(int),'threshold':t}))
   print(f'  {protocol:22s} {kind:10s} {variant:10s} VAL={vm["mcc"]:.4f} TEST={tm["mcc"]:.4f} t={t:.2f}',flush=True)
  # Select model ONLY by training-only OOF, with AUC and stable candidate order tie-breaks.
  winner=max(CANDIDATES,key=lambda c:(cv_data[c]['oof_mcc'],cv_data[c]['oof_roc_auc'],-CANDIDATES.index(c)))
  match=[r for r in runs if r['protocol']==protocol and (r['model'],r['variant'])==winner]
  if len(match)!=1:raise RuntimeError('Model selection mismatch')
  selected.append(match[0])
  print(f'SELECTED {protocol}: {winner[0]}/{winner[1]} (train-only OOF)',flush=True)
 frames={'all_candidate_protocol_results.csv':pd.DataFrame(runs),
         'spatial_oof_candidate_results.csv':pd.DataFrame(oof_rows),
         'all_candidate_test_predictions.csv':pd.concat(pred_rows,ignore_index=True),
         'selected_protocol_results.csv':pd.DataFrame(selected)}
 for name,frame in frames.items():atomic_csv(frame,folder/name)
 manifest={'script':Path(__file__).name,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
           'seed':seed,'configuration':vars(args)|{'seeds':None,'force':None,'summarize_only':None},
           'script117_sha256':digest(Path(__file__)),'script111_sha256':digest(S111),
           'nodes_sha256':digest(b.DATA),'edges_sha256':digest(b.EDGES),
           'artifacts_sha256':{name:digest(folder/name) for name in frames},
           'candidate_count':len(CANDIDATES),'protocol_count':len(PROTOCOLS),
           'model_selection':'train-only spatial OOF MCC; ROC-AUC tie-break',
           'threshold_policy':'source validation for zero-shot, target validation for within/joint',
           'test_policy':'historical exploratory test, not used in model/threshold selection',
           'label_status':'C2 PROVISIONAL_NOT_MANUALLY_VALIDATED'}
 (folder/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
 return verify_seed(seed,args,b)

def verify_seed(seed,args,b):
 folder=OUT/f'seed_{seed}'
 m=json.loads((folder/'experiment_manifest.json').read_text(encoding='utf-8'))
 if m['seed']!=seed:raise RuntimeError('Seed manifest mismatch')
 for name,h in [('script117_sha256',digest(Path(__file__))),('script111_sha256',digest(S111)),
                ('nodes_sha256',digest(b.DATA)),('edges_sha256',digest(b.EDGES))]:
  if m.get(name)!=h:raise RuntimeError(f'Provenance mismatch: {name}')
 cfg=m['configuration']
 for k in ('epochs','folds','hidden','dropout','lr','weight_decay','device'):
  if cfg.get(k)!=getattr(args,k):raise RuntimeError(f'Configuration mismatch: {k}')
 frames={}
 for name in PER_SEED:
  path=folder/name
  if not path.is_file() or digest(path)!=m['artifacts_sha256'].get(name):
   raise RuntimeError(f'Artifact integrity failure: {path}')
  frames[name]=pd.read_csv(path,low_memory=False)
 runs=frames['all_candidate_protocol_results.csv']
 selected=frames['selected_protocol_results.csv']
 cv=frames['spatial_oof_candidate_results.csv']
 pred=frames['all_candidate_test_predictions.csv']
 if len(runs)!=36 or len(cv)!=36 or len(selected)!=6:raise RuntimeError('Incorrect candidate/protocol counts')
 if set(runs.protocol)!=set(PROTOCOLS) or set(selected.protocol)!=set(PROTOCOLS):raise RuntimeError('Missing protocol')
 if runs.duplicated(['protocol','model','variant']).any() or selected.duplicated('protocol').any():raise RuntimeError('Duplicate results')
 if pred.duplicated(['protocol','model','variant','sample_uid']).any():raise RuntimeError('Duplicate predictions')
 if not all(frame.seed.eq(seed).all() for frame in frames.values()):raise RuntimeError('Seed mismatch')
 for _,r in runs.iterrows():
  p=pred.loc[(pred.protocol==r.protocol)&(pred.model==r.model)&(pred.variant==r.variant)]
  if len(p)!=int(r.test_n):raise RuntimeError('Missing test prediction rows')
  y=p.label.to_numpy(dtype=int);prob=p.probability.to_numpy(dtype=float)
  if not np.isfinite(prob).all() or np.any((prob<0)|(prob>1)):raise RuntimeError('Invalid probabilities')
  if not np.array_equal(p.prediction.to_numpy(dtype=int),(prob>=float(r.threshold)).astype(int)):
   raise RuntimeError('Prediction threshold mismatch')
  test=b.metrics(y,prob,float(r.threshold))
  for k,v in test.items():
   if not np.isclose(float(r['test_'+k]),v,atol=1e-8,rtol=1e-8):
    raise RuntimeError(f'Metric mismatch {r.protocol}/{r.model}/{r.variant}/{k}')
  if r.threshold_selection_country!=(r.source if r.protocol.endswith('zero_shot') else r.target):
   raise RuntimeError('Invalid validation-country policy')
 for _,r in selected.iterrows():
  candidates=cv.loc[cv.protocol.eq(r.protocol)]
  winner=max(CANDIDATES,key=lambda c:(float(candidates.loc[(candidates.model==c[0])&(candidates.variant==c[1]),'oof_mcc'].iloc[0]),
           float(candidates.loc[(candidates.model==c[0])&(candidates.variant==c[1]),'oof_roc_auc'].iloc[0]),-CANDIDATES.index(c)))
  if (r.model,r.variant)!=winner:raise RuntimeError('Selected model not train-only OOF winner')
 return frames

def summarize(results,seeds):
 OUT.mkdir(parents=True,exist_ok=True)
 runs=pd.concat([r['all_candidate_protocol_results.csv'] for r in results],ignore_index=True)
 selected=pd.concat([r['selected_protocol_results.csv'] for r in results],ignore_index=True)
 cv=pd.concat([r['spatial_oof_candidate_results.csv'] for r in results],ignore_index=True)
 if len(runs)!=36*len(seeds) or len(selected)!=6*len(seeds):raise RuntimeError('Incomplete summary')
 fields=['test_mcc','test_balanced_accuracy','test_f1','test_roc_auc','test_pr_auc','validation_mcc','oof_mcc']
 candidate=runs.groupby(['protocol','model','variant'],sort=False)[fields].agg(['mean','std','min','max'])
 candidate.columns=['_'.join(c) for c in candidate.columns]
 candidate=candidate.reset_index()
 candidate.insert(3,'n_seeds',len(seeds))
 selected_summary=selected.groupby('protocol',sort=False)[fields].agg(['mean','std','min','max'])
 selected_summary.columns=['_'.join(c) for c in selected_summary.columns]
 selected_summary=selected_summary.reset_index()
 selected_summary.insert(1,'n_seeds',len(seeds))
 atomic_csv(runs,OUT/'all_seed_candidate_protocol_results.csv')
 atomic_csv(cv,OUT/'all_seed_spatial_oof_results.csv')
 atomic_csv(selected,OUT/'all_seed_selected_protocol_results.csv')
 atomic_csv(candidate,OUT/'table_gnn_six_candidate_mean_std.csv')
 atomic_csv(selected_summary,OUT/'table_gnn_train_oof_selected_mean_std.csv')
 selection_counts=selected.groupby(['protocol','model','variant'],as_index=False).size().rename(columns={'size':'selected_seed_count'})
 atomic_csv(selection_counts,OUT/'table_gnn_model_selection_frequency.csv')
 # Compare with frozen CartoGraph-X, but never use test scores to choose a new model.
 s114=ROOT/'data/processed/cross_country/cartographx_v2_ten_seed_summary_v1/all_seed_protocol_results.csv'
 if s114.is_file():
  proposed=pd.read_csv(s114,low_memory=False)
  if {'seed','protocol','test_mcc'}.issubset(proposed.columns):
   cmp=selected[['seed','protocol','model','variant','test_mcc']].rename(columns={'test_mcc':'gnn_selected_test_mcc'})
   cmp=cmp.merge(proposed[['seed','protocol','test_mcc']].rename(columns={'test_mcc':'cartographx_test_mcc'}),
                 on=['seed','protocol'],how='inner',validate='one_to_one')
   if len(cmp)!=6*len(seeds):raise RuntimeError('Incomplete matched CartoGraph-X comparison')
   cmp['delta_cartographx_minus_gnn_mcc']=cmp.cartographx_test_mcc-cmp.gnn_selected_test_mcc
   atomic_csv(cmp,OUT/'paired_seed_cartographx_vs_selected_gnn_mcc.csv')
   agg=cmp.groupby('protocol',sort=False).agg(gnn_mcc_mean=('gnn_selected_test_mcc','mean'),
        cartographx_mcc_mean=('cartographx_test_mcc','mean'),
        delta_mcc_mean=('delta_cartographx_minus_gnn_mcc','mean'),
        delta_mcc_std=('delta_cartographx_minus_gnn_mcc','std'),n_seeds=('seed','nunique')).reset_index()
   atomic_csv(agg,OUT/'table_cartographx_vs_gnn_mcc.csv')
   print('\nMATCHED-SEED CARTOGRAPH-X vs TRAIN-OOF-SELECTED GNN (EXPLORATORY)',flush=True)
   print(agg.to_string(index=False,float_format=lambda v:f'{v:.4f}'),flush=True)
 manifest={'script':Path(__file__).name,'seeds':list(seeds),'timestamp_utc':datetime.now(timezone.utc).isoformat(),
  'script117_sha256':digest(Path(__file__)),'script111_sha256':digest(S111),
  'policy':'all six fixed GNN candidates, train-only OOF selection, validation-only threshold',
  'historical_test':'EXPLORATORY','c2_labels':'PROVISIONAL',
  'outputs_sha256':{p.name:digest(p) for p in OUT.glob('*.csv')}}
 (OUT/'summary_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print(f'\nVERIFIED {len(seeds)} seeds, {len(runs)} candidate-protocol results. Saved: {OUT}')
 print('Historical test is exploratory. Script 125 remains final figure-generation code.')

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
 ap.add_argument('--epochs',type=int,default=60)
 ap.add_argument('--folds',type=int,default=3)
 ap.add_argument('--hidden',type=int,default=64)
 ap.add_argument('--dropout',type=float,default=.2)
 ap.add_argument('--lr',type=float,default=.001)
 ap.add_argument('--weight-decay',type=float,default=1e-4)
 ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
 ap.add_argument('--force',action='store_true')
 ap.add_argument('--summarize-only',action='store_true')
 args=ap.parse_args()
 if not args.seeds or len(set(args.seeds))!=len(args.seeds):ap.error('Seeds must be nonempty and unique')
 if args.epochs<1 or args.folds<2 or args.hidden<1:ap.error('Invalid training configuration')
 b=load_dependency()
 import torch
 if args.device=='auto':args.device='cuda' if torch.cuda.is_available() else 'cpu'
 if args.device=='cuda' and not torch.cuda.is_available():ap.error('CUDA unavailable')
 if not b.DATA.is_file() or not b.EDGES.is_file():ap.error('Missing V2 nodes or graph edges')
 df=pd.read_csv(b.DATA,low_memory=False)
 edges=pd.read_csv(b.EDGES,low_memory=False)
 if df.sample_uid.isna().any() or df.sample_uid.duplicated().any():raise RuntimeError('Invalid sample_uid')
 if not df.retained.isin([0,1]).all():raise RuntimeError('Invalid labels')
 if (df.groupby(['country','tile_id']).split.nunique()>1).any():raise RuntimeError('Spatial tile crosses split')
 for c in b.FEATURES:
  df[c]=pd.to_numeric(df[c],errors='raise')
  if np.isinf(df[c].to_numpy(dtype=float)).any():raise RuntimeError(f'Nonfinite {c}')
 graph=b.GraphData(df,edges)
 outputs=[]
 for seed in args.seeds:
  print(f'\n=== SCRIPT 117 SEED {seed} ===',flush=True)
  if args.force and args.summarize_only:ap.error('Cannot combine --force and --summarize-only')
  try:
   if args.force:raise RuntimeError('Force requested')
   r=verify_seed(seed,args,b)
   print(f'REUSED VERIFIED seed {seed}',flush=True)
  except (FileNotFoundError,RuntimeError,KeyError,ValueError) as exc:
   if args.summarize_only:raise RuntimeError(f'Seed {seed} unavailable or invalid: {exc}') from exc
   print(f'TRAINING seed {seed}: {exc}',flush=True)
   r=train_seed(seed,args,b,df,graph)
   print(f'VERIFIED seed {seed}',flush=True)
  outputs.append(r)
 summarize(outputs,args.seeds)

if __name__=='__main__':main()
