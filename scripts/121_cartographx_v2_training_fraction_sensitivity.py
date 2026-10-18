#!/usr/bin/env python3
"""Script 121: Frozen CartoGraph-X V2 training-data fraction sensitivity.

Fractions: 10%, 25%, 50%, 75%, 100%. Seeds: 42..51.
Uses original k=6, 250m graph; train/validation rows ONLY (no test data).
Deterministic nested, country/city/class-stratified TRAIN node subsamples.
Train induced graphs contain ONLY selected training nodes. The full validation
graph is evaluated independently, with no train-validation edges.
For zero-shot transfer, the decision threshold is chosen on SOURCE validation
only; target validation labels are used only for descriptive metrics.
Joint thresholds are selected on their respective target validation sets.
No model selection, no final figures, no modification to primary model.

Run:
 python scripts/121_cartographx_v2_training_fraction_sensitivity.py --seeds 42
 python scripts/121_cartographx_v2_training_fraction_sensitivity.py
 python scripts/121_cartographx_v2_training_fraction_sensitivity.py --summarize-only
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SCRIPT111=ROOT/'scripts/111_train_benchmark_v2_weighted_gnn_baselines.py'
SCRIPT112=ROOT/'scripts/112_train_cartographx_v2_architecture.py'
OUT=ROOT/'data/processed/cross_country/cartographx_v2_training_fraction_sensitivity_v1'
FRACTIONS=(0.10,0.25,0.50,0.75,1.00)
SEEDS=tuple(range(42,52))
PROTOCOLS=('lt_to_lt','pl_to_pl','lt_to_pl_zero_shot','pl_to_lt_zero_shot','joint_to_lt','joint_to_pl')
COUNTRIES=('Lithuania','Poland')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def atomic_csv(df,path):
 tmp=path.with_suffix('.tmp')
 df.to_csv(tmp,index=False)
 tmp.replace(path)
def load_module(path,name):
 if not path.exists():raise FileNotFoundError(path)
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec)
 sys.modules[name]=module
 spec.loader.exec_module(module)
 return module

def make_nested_indices(df,seed):
 """One random order per country/city/class stratum, nested across fractions.

 Each country/city/class stratum contributes at least one item for any fraction.
 A fixed order means lower-fraction training sets are subsets of higher ones.
 """
 train=df.loc[df.split.eq('train')]
 if not len(train):raise RuntimeError('No training nodes')
 rng=np.random.default_rng(seed+121000)
 groups=[]
 for (country,city,label),g in train.groupby(['country','city','retained'],sort=True):
  idx=g.index.to_numpy(dtype=int)
  order=rng.permutation(idx)
  groups.append((country,city,int(label),order))
 selected={}
 for frac in FRACTIONS:
  ids=[]
  for country,city,label,order in groups:
   count=len(order) if frac==1.0 else max(1,int(np.floor(len(order)*frac+0.5)))
   ids.extend(order[:count].tolist())
  arr=np.asarray(sorted(ids),dtype=int)
  if len(arr)!=len(np.unique(arr)):raise RuntimeError('Duplicate sampled IDs')
  for country in COUNTRIES:
   labels=df.iloc[arr[arr_country(df,arr,country)]].retained
   if len(labels)<2 or labels.nunique()!=2:raise RuntimeError('Missing class in training fraction')
  selected[frac]=arr
 for low,high in zip(FRACTIONS[:-1],FRACTIONS[1:]):
  if not set(selected[low]).issubset(set(selected[high])):
   raise RuntimeError('Sampling is not nested')
 if not np.array_equal(selected[1.0],np.flatnonzero(df.split=='train')):
  raise RuntimeError('Full fraction does not recover complete training set')
 return selected

def arr_country(df,ids,country):
 return df.country.to_numpy()[ids]==country

def verify(seed,config_hash):
 folder=OUT/f'seed_{seed}'
 manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
 if manifest['config_hash']!=config_hash or manifest['seed']!=seed:
  raise RuntimeError('Configuration/seed mismatch')
 path=folder/'validation_results.csv'
 if sha(path)!=manifest['result_sha256']:raise RuntimeError('Result checksum mismatch')
 d=pd.read_csv(path)
 if len(d)!=len(FRACTIONS)*len(PROTOCOLS):raise RuntimeError('Incomplete seed output')
 if d.duplicated(['fraction','protocol']).any():raise RuntimeError('Duplicate fraction/protocol')
 if set(d.protocol)!=set(PROTOCOLS):raise RuntimeError('Missing protocols')
 if set(np.round(d.fraction,6))!=set(FRACTIONS):raise RuntimeError('Missing fractions')
 if not d.seed.eq(seed).all():raise RuntimeError('Incorrect seed')
 return d

def run_seed(seed,df,graph,base,arch,args,config_hash):
 folder=OUT/f'seed_{seed}'
 folder.mkdir(parents=True,exist_ok=True)
 selected=make_nested_indices(df,seed)
 val_lt=np.flatnonzero((df.split=='validation')&(df.country=='Lithuania'))
 val_pl=np.flatnonzero((df.split=='validation')&(df.country=='Poland'))
 if not len(val_lt) or not len(val_pl):raise RuntimeError('Missing validation country')
 rows=[];sampling=[]
 for frac in FRACTIONS:
  ids=selected[frac]
  lt_train=ids[arr_country(df,ids,'Lithuania')]
  pl_train=ids[arr_country(df,ids,'Poland')]
  for country,train_ids in [('Lithuania',lt_train),('Poland',pl_train)]:
   y=graph.labels[train_ids]
   sampling.append(dict(seed=seed,fraction=frac,country=country,n_train=len(train_ids),
    n_positive=int(y.sum()),prevalence=float(y.mean()),
    n_tiles=df.iloc[train_ids].tile_id.nunique(),
    n_cities=df.iloc[train_ids].city.nunique()))
  print(f'\nseed={seed} fraction={frac:.0%} LT={len(lt_train)} PL={len(pl_train)}',flush=True)
  # Each country-only model is trained ONCE and evaluated in both countries.
  lt_val,lt_to_pl=arch.train_predict(graph,lt_train,[val_lt,val_pl],args,seed+99173,False)
  pl_val,pl_to_lt=arch.train_predict(graph,pl_train,[val_pl,val_lt],args,seed+99173,False)
  joint_lt,joint_pl=arch.train_predict(graph,ids,[val_lt,val_pl],args,seed+99173,True)
  t_lt=base.threshold_for(graph.labels[val_lt],lt_val)
  t_pl=base.threshold_for(graph.labels[val_pl],pl_val)
  t_joint_lt=base.threshold_for(graph.labels[val_lt],joint_lt)
  t_joint_pl=base.threshold_for(graph.labels[val_pl],joint_pl)
  cases=[
   ('lt_to_lt','Lithuania','Lithuania',val_lt,lt_val,t_lt,'Lithuania',lt_train),
   ('pl_to_pl','Poland','Poland',val_pl,pl_val,t_pl,'Poland',pl_train),
   ('lt_to_pl_zero_shot','Lithuania','Poland',val_pl,lt_to_pl,t_lt,'Lithuania',lt_train),
   ('pl_to_lt_zero_shot','Poland','Lithuania',val_lt,pl_to_lt,t_pl,'Poland',pl_train),
   ('joint_to_lt','joint','Lithuania',val_lt,joint_lt,t_joint_lt,'Lithuania',ids),
   ('joint_to_pl','joint','Poland',val_pl,joint_pl,t_joint_pl,'Poland',ids)]
  for protocol,source,target,val_ids,p,t,threshold_country,training_ids in cases:
   met=base.metrics(graph.labels[val_ids],p,t)
   rows.append(dict(seed=seed,fraction=frac,protocol=protocol,source=source,target=target,
    n_train=len(training_ids),n_validation=len(val_ids),
    train_positive_prevalence=float(graph.labels[training_ids].mean()),
    validation_positive_prevalence=float(graph.labels[val_ids].mean()),
    predicted_positive_rate=float(np.mean(p>=t)),
    threshold_selection_country=threshold_country,threshold=float(t),
    **{'validation_'+key:value for key,value in met.items()}))
   print(f'  {protocol:22s} VAL MCC={met["mcc"]:.4f} threshold={t:.2f}',flush=True)
 result=pd.DataFrame(rows)
 atomic_csv(result,folder/'validation_results.csv')
 atomic_csv(pd.DataFrame(sampling),folder/'sample_composition.csv')
 manifest={'seed':seed,'config_hash':config_hash,
  'result_sha256':sha(folder/'validation_results.csv'),
  'sample_composition_sha256':sha(folder/'sample_composition.csv'),
  'timestamp_utc':datetime.now(timezone.utc).isoformat(),
  'scope':'validation only, no test partition loaded',
  'sampling':'nested country/city/class stratified TRAIN nodes; induced training subgraph',
  'threshold':'source validation for zero-shot, country validation for within/joint',
  'figures_generated':0,'label_status':'C2 provisional'}
 (folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 return verify(seed,config_hash)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
 ap.add_argument('--epochs',type=int,default=60)
 ap.add_argument('--hidden',type=int,default=64)
 ap.add_argument('--dropout',type=float,default=.2)
 ap.add_argument('--lr',type=float,default=.001)
 ap.add_argument('--weight-decay',type=float,default=1e-4)
 ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
 ap.add_argument('--force',action='store_true')
 ap.add_argument('--summarize-only',action='store_true')
 args=ap.parse_args()
 if len(set(args.seeds))!=len(args.seeds) or not args.seeds:ap.error('Seeds must be unique and nonempty')
 if args.force and args.summarize_only:ap.error('Incompatible --force and --summarize-only')
 if args.epochs<1 or args.hidden<1:ap.error('Invalid model hyperparameters')
 import torch
 if args.device=='auto':args.device='cuda' if torch.cuda.is_available() else 'cpu'
 if args.device=='cuda' and not torch.cuda.is_available():ap.error('CUDA unavailable')
 base=load_module(SCRIPT111,'cartographx121_base')
 arch=load_module(SCRIPT112,'cartographx121_arch')
 if not base.DATA.is_file() or not base.EDGES.is_file():raise FileNotFoundError('V2 graph files missing')
 # Deliberately EXCLUDE test rows and test edges before creating the graph.
 df=pd.read_csv(base.DATA,low_memory=False)
 df=df.loc[df.split.isin(['train','validation'])].reset_index(drop=True)
 if df.sample_uid.isna().any() or df.sample_uid.duplicated().any():raise RuntimeError('Invalid node UIDs')
 if not df.retained.isin([0,1]).all():raise RuntimeError('Invalid labels')
 if (df.groupby(['country','tile_id']).split.nunique()>1).any():raise RuntimeError('Tile crosses splits')
 for col in base.FEATURES:df[col]=pd.to_numeric(df[col],errors='raise')
 edges=pd.read_csv(base.EDGES,low_memory=False)
 edges=edges.loc[edges.split.isin(['train','validation'])].reset_index(drop=True)
 graph=base.GraphData(df,edges)
 config={'fractions':FRACTIONS,'epochs':args.epochs,'hidden':args.hidden,
  'dropout':args.dropout,'lr':args.lr,'weight_decay':args.weight_decay,
  'device':args.device,'script121_sha256':sha(Path(__file__)),
  'script111_sha256':sha(SCRIPT111),'script112_sha256':sha(SCRIPT112),
  'nodes_sha256':sha(base.DATA),'edges_sha256':sha(base.EDGES)}
 config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
 OUT.mkdir(parents=True,exist_ok=True)
 frames=[]
 for seed in args.seeds:
  print(f'\n===== SCRIPT 121 SEED {seed} =====',flush=True)
  if not args.force:
   try:
    existing=verify(seed,config_hash)
    print(f'REUSED VERIFIED seed {seed}',flush=True)
    frames.append(existing);continue
   except (OSError,ValueError,KeyError,RuntimeError,TypeError) as ex:
    if args.summarize_only:raise RuntimeError(f'Cannot verify seed {seed}: {ex}') from ex
    print(f'TRAINING seed {seed}: {ex}',flush=True)
  frames.append(run_seed(seed,df,graph,base,arch,args,config_hash))
 all_df=pd.concat(frames,ignore_index=True)
 atomic_csv(all_df,OUT/'all_seed_fraction_validation.csv')
 cols=['validation_mcc','validation_balanced_accuracy','validation_f1',
       'validation_roc_auc','validation_pr_auc','predicted_positive_rate','n_train']
 summary=all_df.groupby(['fraction','protocol'],sort=False)[cols].agg(['mean','std','min','max'])
 summary.columns=['_'.join(x) for x in summary.columns]
 summary=summary.reset_index()
 summary.insert(2,'n_seeds',len(args.seeds))
 atomic_csv(summary,OUT/'table_training_fraction_mean_std.csv')
 full=all_df.loc[all_df.fraction==1.0,['seed','protocol','validation_mcc']]
 full=full.rename(columns={'validation_mcc':'full_fraction_validation_mcc'})
 paired=all_df.merge(full,on=['seed','protocol'],how='left',validate='many_to_one')
 if paired.full_fraction_validation_mcc.isna().any():raise RuntimeError('Missing full reference')
 paired['delta_mcc_vs_full']=paired.validation_mcc-paired.full_fraction_validation_mcc
 atomic_csv(paired,OUT/'paired_seed_fraction_vs_full.csv')
 deltas=paired.groupby(['fraction','protocol'],sort=False).agg(
  delta_mcc_mean=('delta_mcc_vs_full','mean'),
  delta_mcc_std=('delta_mcc_vs_full','std'),
  n_seeds=('seed','nunique')).reset_index()
 atomic_csv(deltas,OUT/'table_fraction_mcc_deltas.csv')
 composition=pd.concat([pd.read_csv(OUT/f'seed_{seed}'/'sample_composition.csv') for seed in args.seeds],
                       ignore_index=True)
 atomic_csv(composition,OUT/'all_seed_sample_composition.csv')
 manifest={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'config':config,
  'config_hash':config_hash,'seeds':args.seeds,'test_partition_used':False,
  'sampling':'nested class/city/country stratified train-only, not tile-level holdout',
  'validation_threshold_policy':'source-only for zero-shot',
  'full_fraction_reference':'original k6 250m graph and frozen Script112 architecture',
  'caveat':'No new untouched test; C2 correspondence labels provisional',
  'figures_generated':0,
  'outputs_sha256':{f.name:sha(f) for f in OUT.glob('*.csv')}}
 (OUT/'summary_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print('\nTRAINING FRACTION MCC DELTAS (versus 100% training):')
 print(deltas.to_string(index=False,float_format=lambda v:f'{v:.4f}'))
 print(f'\nSCRIPT 121 PASS: {len(args.seeds)} seeds, {len(all_df)} protocol-level validation evaluations')
 print('Saved:',OUT)
 print('No test data used, no figures. NEXT: Script 122 calibration. Script 125 is final figure code.')

if __name__=='__main__':main()
