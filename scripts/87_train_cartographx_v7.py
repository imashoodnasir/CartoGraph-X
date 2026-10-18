#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CartoGraph-X V7 — source-only structural model selection and ensembles.

No neural residual and no transductive adaptation. All candidates use training
spatial GroupKFold predictions for CV model ranking. Source validation chooses
an individual candidate or a prespecified equal-weight probability ensemble,
and its classification threshold. Target test is evaluated only after selection.
Road graph features come from Script 68 (centroid proximity, not physical topology).

Run from project root:
 python scripts/87_train_cartographx_v7.py --mode development --protocols lt_to_pl_zero_shot pl_to_lt_zero_shot
 python scripts/87_train_cartographx_v7.py --mode development
 python scripts/87_train_cartographx_v7.py --mode full
"""
from __future__ import annotations
import argparse, copy, json, os, random, time, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (matthews_corrcoef, balanced_accuracy_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
warnings.filterwarnings('ignore', category=FutureWarning)
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'processed'/'cross_country'
NODES=DATA/'graph_representations'/'road_graph_nodes.csv'
EDGES=DATA/'graph_representations'/'road_graph_edges.csv'
OUT=DATA/'cartographx_v7_models'
TAB=['length_m','intersecting_road_count','neighbor_count_150m','mean_neighbor_distance_m']
GRAPH=['graph_degree','graph_mean_edge_distance_m','graph_min_edge_distance_m',
       'graph_max_edge_distance_m','graph_local_clustering','graph_component_size',
       'graph_is_isolated','local_road_count_50m','local_road_count_100m',
       'local_road_count_150m','local_road_count_250m','nbr_mean_length_m',
       'nbr_std_length_m','nbr_mean_intersecting_road_count',
       'nbr_std_intersecting_road_count','nbr_mean_neighbor_count_150m',
       'nbr_std_neighbor_count_150m','nbr_mean_mean_neighbor_distance_m',
       'nbr_std_mean_neighbor_distance_m']
# Structural features are computed without road-selection labels, within each
# country/split/tile. They describe local relative geometry and density.
STRUCT_BASE=['length_m','intersecting_road_count','neighbor_count_150m',
             'mean_neighbor_distance_m','graph_degree','graph_component_size',
             'local_road_count_50m','local_road_count_100m',
             'local_road_count_150m','local_road_count_250m']
STRUCT=[f'struct_{c}' for c in STRUCT_BASE]
FEATURE_SETS={'tabular':TAB,'tabular_graph':TAB+GRAPH}

def add_structural_features(nodes):
 # This is a label-free local, per-graph transformation. No test labels or
 # global target statistics are involved. Each tile is a deployment unit.
 nodes=nodes.copy()
 group_cols=['country','split','tile_id']
 for col,out in zip(STRUCT_BASE,STRUCT):
  v=pd.to_numeric(nodes[col],errors='coerce').replace([np.inf,-np.inf],np.nan)
  v=np.log1p(v.clip(lower=0))
  grp=v.groupby([nodes[c] for c in group_cols],sort=False)
  med=grp.transform('median').fillna(0)
  spread=(v-med).abs().groupby([nodes[c] for c in group_cols],sort=False).transform('median')
  z=((v-med)/(1.4826*spread+0.25)).clip(-6,6)
  nodes[out]=z.fillna(0).astype('float32')
 return nodes

def configure_representation(kind):
 global FEATURE_SETS, RESIDUAL_FEATURES
 if kind=='raw':
  FEATURE_SETS={'tabular':TAB,'tabular_graph':TAB+GRAPH}
  RESIDUAL_FEATURES=TAB+GRAPH
 else:
  FEATURE_SETS={'structural':STRUCT,
                'tabular_structural':TAB+STRUCT,
                'tabular_graph_structural':TAB+GRAPH+STRUCT}
  RESIDUAL_FEATURES=TAB+GRAPH+STRUCT

PROTOCOLS={
 'lt_to_lt':(['Lithuania'],'Lithuania','Lithuania'),
 'pl_to_pl':(['Poland'],'Poland','Poland'),
 'lt_to_pl_zero_shot':(['Lithuania'],'Lithuania','Poland'),
 'pl_to_lt_zero_shot':(['Poland'],'Poland','Lithuania'),
 'joint_to_lt':(['Lithuania','Poland'],'Lithuania','Lithuania'),
 'joint_to_pl':(['Lithuania','Poland'],'Poland','Poland')}
# Predeclared baseline configurations from Script 69; selection uses train-only spatial CV.
CONFIGS={
 'logistic_regression':[{'C':c,'class_weight':'balanced'} for c in (0.1,1.,10.)],
 'random_forest':[{'n_estimators':400,'max_depth':12,'min_samples_leaf':3,'max_features':'sqrt','class_weight':'balanced'},
                  {'n_estimators':500,'max_depth':None,'min_samples_leaf':5,'max_features':'sqrt','class_weight':'balanced'}],
 'extra_trees':[{'n_estimators':400,'max_depth':12,'min_samples_leaf':3,'max_features':'sqrt','class_weight':'balanced'},
                {'n_estimators':500,'max_depth':None,'min_samples_leaf':5,'max_features':'sqrt','class_weight':'balanced'}],
 'hist_gradient_boosting':[{'learning_rate':.05,'max_iter':250,'max_leaf_nodes':31,'l2_regularization':1.},
                           {'learning_rate':.08,'max_iter':300,'max_leaf_nodes':15,'l2_regularization':2.}]}
# Predeclared ablations. All model and variant selection uses source validation only.
# The anchor remains an explicit candidate, allowing a no-correction fallback.
VARIANTS={
 'V6-A_anchor': {'graph_dropout':0., 'max_delta':0., 'confidence_penalty':0.},
 'V6-B_bounded': {'graph_dropout':0., 'max_delta':0.35, 'confidence_penalty':0.},
 'V6-C_confidence': {'graph_dropout':0., 'max_delta':0.35, 'confidence_penalty':0.15},
 'V6-D_dropout': {'graph_dropout':0.25, 'max_delta':0.35, 'confidence_penalty':0.15},
 'V6-E_strict': {'graph_dropout':0.25, 'max_delta':0.18, 'confidence_penalty':0.30},
}
ANCHOR_NAME='V6-A_anchor'

def seed_everything(seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def model_factory(name,params,seed):
 if name=='logistic_regression':
  clf=LogisticRegression(**params,solver='liblinear',max_iter=3000,random_state=seed)
  return Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True)),('scale',RobustScaler()),('model',clf)])
 if name=='random_forest': clf=RandomForestClassifier(**params,random_state=seed,n_jobs=-1)
 elif name=='extra_trees': clf=ExtraTreesClassifier(**params,random_state=seed,n_jobs=-1)
 else: clf=HistGradientBoostingClassifier(**params,random_state=seed)
 return Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True)),('model',clf)])

def prob(model,x): return np.asarray(model.predict_proba(x)[:,1],dtype=np.float64)
def logit(p):
 p=np.clip(np.asarray(p),1e-5,1-1e-5)
 return np.log(p/(1-p))
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-40,40)))

def metrics(y,p,t):
 y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float); yp=(p>=t).astype(int)
 tn,fp,fn,tp=confusion_matrix(y,yp,labels=[0,1]).ravel()
 return {'mcc':float(matthews_corrcoef(y,yp)),
         'balanced_accuracy':float(balanced_accuracy_score(y,yp)),
         'f1':float(f1_score(y,yp,zero_division=0)),
         'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
         'pr_auc':float(average_precision_score(y,p)) if len(np.unique(y))==2 else None,
         'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def threshold_select(y,p):
 choices=[]
 for t in np.round(np.arange(.05,.951,.01),2):
  m=metrics(y,p,t)
  choices.append((m['mcc'],m['balanced_accuracy'],m['f1'],-abs(t-.5),float(t)))
 best=max(choices)
 return best[-1],metrics(y,p,best[-1])

def spatial_groups(frame):
 # Spatially isolated tiles are the grouping unit. Prefix country to avoid collisions.
 return (frame['country'].astype(str)+'::'+frame['tile_id'].astype(str)).to_numpy()

def choose_anchor(train,val,seed,fast=False,cv_folds=3):
 """Select baseline family/config/representation by TRAIN-only spatial OOF MCC.

 The source validation partition is reserved for threshold calibration and
 residual-variant selection. This avoids selecting a base model solely because
 it happens to fit one validation partition. No target-country labels are used.
 """
 y=train.retained.to_numpy(dtype=int)
 vy=val.retained.to_numpy(dtype=int)
 groups=spatial_groups(train)
 ng=len(np.unique(groups))
 if ng<2: raise RuntimeError('At least two training spatial groups required')
 nfold=min(cv_folds,ng)
 cv=list(GroupKFold(n_splits=nfold).split(train,y,groups))
 results=[]; winner=None
 for rep,features in FEATURE_SETS.items():
  X=train[features]
  for name,configs in CONFIGS.items():
   if fast and name not in ('extra_trees','hist_gradient_boosting'): continue
   for j,params in enumerate(configs,1):
    oof=np.full(len(train),np.nan,dtype=np.float64)
    for k,(fit_idx,held_idx) in enumerate(cv,1):
     if len(np.unique(y[fit_idx]))<2:
      raise RuntimeError(f'Anchor CV fold {k} training has only one class')
     mdl=model_factory(name,params,seed+k)
     mdl.fit(X.iloc[fit_idx],y[fit_idx])
     oof[held_idx]=prob(mdl,X.iloc[held_idx])
    if not np.isfinite(oof).all(): raise RuntimeError('Incomplete anchor OOF')
    cv_threshold,cm=threshold_select(y,oof)
    # Prevalence-insensitive ranking: MCC first, then BA and AUC.
    key=(cm['mcc'],cm['balanced_accuracy'],cm['roc_auc'] or -1)
    row={'representation':rep,'model':name,'config_id':j,
         'params':json.dumps(params,sort_keys=True),
         'spatial_cv_folds':nfold,'spatial_cv_mcc':cm['mcc'],
         'spatial_cv_balanced_accuracy':cm['balanced_accuracy'],
         'spatial_cv_roc_auc':cm['roc_auc'],'spatial_cv_threshold':cv_threshold}
    results.append(row)
    print(f'  anchor CV {rep:14s} {name:23s} config={j} CV_MCC={cm["mcc"]:.4f}',flush=True)
    if winner is None or key>winner['key']:
     winner={'key':key,'representation':rep,'features':features,
             'model_name':name,'params':params,'config_id':j,
             'cv_metrics':cm,'cv_threshold':cv_threshold,
             'oof':oof.copy()}
 # Refit ONLY on the training split; calibration uses source validation.
 mdl=model_factory(winner['model_name'],winner['params'],seed)
 mdl.fit(train[winner['features']],y)
 vp=prob(mdl,val[winner['features']])
 threshold,vm=threshold_select(vy,vp)
 winner.update({'model':mdl,'threshold':threshold,'val_metrics':vm})
 return winner,pd.DataFrame(results)

def oof_predictions(train,best,seed,folds):
 # Each spatial tile is a whole group. No row in a held-out tile is used
 # to fit the model predicting that tile's labels.
 groups=(train['country'].astype(str)+'::'+train['tile_id'].astype(str)).to_numpy()
 ng=len(np.unique(groups))
 if ng<2: raise RuntimeError('At least two independent training tiles required for OOF')
 nfold=min(folds,ng)
 cv=GroupKFold(n_splits=nfold); oof=np.full(len(train),np.nan)
 X=train[best['features']]; y=train.retained.to_numpy(dtype=int)
 for k,(tr,ho) in enumerate(cv.split(X,y,groups),1):
  if len(np.unique(y[tr]))!=2: raise RuntimeError(f'OOF fold {k} has only one class in training')
  mdl=model_factory(best['model_name'],best['params'],seed+k)
  mdl.fit(X.iloc[tr],y[tr]); oof[ho]=prob(mdl,X.iloc[ho])
  print(f'    OOF fold {k}/{nfold}: fit={len(tr):,}, heldout={len(ho):,}',flush=True)
 if not np.isfinite(oof).all(): raise RuntimeError('Incomplete OOF predictions')
 return oof

def load_data():
 if not NODES.exists() or not EDGES.exists():
  raise FileNotFoundError(f'Run Script 68 first. Missing {NODES if not NODES.exists() else EDGES}')
 nodes=pd.read_csv(NODES,low_memory=False); edges=pd.read_csv(EDGES,low_memory=False)
 required=['sample_uid','country','split','tile_id','retained']+TAB+GRAPH
 missing=[x for x in required if x not in nodes.columns]
 if missing: raise KeyError(f'Graph nodes missing: {missing}')
 if nodes.sample_uid.isna().any() or nodes.sample_uid.duplicated().any():
  raise RuntimeError('Non-unique/missing sample_uid in nodes')
 if not set(nodes.split.unique())=={'train','validation','test'}:
  raise RuntimeError('Unexpected split values')
 if not set(nodes.retained.dropna().unique()).issubset({0,1}):
  raise RuntimeError('Labels must be binary')
 if edges.duplicated(['source_uid','target_uid']).any(): raise RuntimeError('Duplicate edges')
 lookup=nodes.set_index('sample_uid')[['country','split','tile_id']]
 for endpoint in ('source_uid','target_uid'):
  if not edges[endpoint].isin(lookup.index).all(): raise RuntimeError(f'Unknown {endpoint}')
 if len(edges):
  a=lookup.loc[edges.source_uid.to_numpy()].reset_index(drop=True)
  b=lookup.loc[edges.target_uid.to_numpy()].reset_index(drop=True)
  for col in ('country','split','tile_id'):
   if not a[col].astype(str).equals(b[col].astype(str)):
    raise RuntimeError(f'Cross-{col} edges detected')
  if (edges.source_uid==edges.target_uid).any(): raise RuntimeError('Self edges detected')
 return nodes,edges

def get_partitions(nodes,protocol):
 countries,vc,tc=PROTOCOLS[protocol]
 tr=nodes.loc[nodes.country.isin(countries)&(nodes.split=='train')].copy().reset_index(drop=True)
 va=nodes.loc[(nodes.country==vc)&(nodes.split=='validation')].copy().reset_index(drop=True)
 te=nodes.loc[(nodes.country==tc)&(nodes.split=='test')].copy().reset_index(drop=True)
 if min(len(tr),len(va),len(te))==0: raise RuntimeError(f'Empty partition: {protocol}')
 for a,b in ((tr,va),(tr,te),(va,te)):
  if set(a.sample_uid)&set(b.sample_uid): raise RuntimeError('Overlapping sample UIDs')
 return tr,va,te



def safe_json(path,obj):
 path.write_text(json.dumps(obj,indent=2,default=lambda v:float(v) if isinstance(v,np.floating) else int(v) if isinstance(v,np.integer) else str(v)),encoding='utf-8')

def cv_candidates(train,seed,nfold,fast=False):
 y=train.retained.to_numpy(dtype=int)
 groups=spatial_groups(train)
 if len(np.unique(groups))<nfold: raise RuntimeError('Insufficient independent training tiles')
 folds=list(GroupKFold(n_splits=nfold).split(train,y,groups))
 rows=[]; candidates=[]
 for rep,features in FEATURE_SETS.items():
  for name,config_list in CONFIGS.items():
   if fast and name not in ('logistic_regression','extra_trees'): continue
   for config_id,params in enumerate(config_list,1):
    oof=np.empty(len(train),dtype=float); oof[:]=np.nan
    for fold_id,(fit_idx,held_idx) in enumerate(folds,1):
     if len(np.unique(y[fit_idx]))!=2: raise RuntimeError('Single-class training fold')
     model=model_factory(name,params,seed+fold_id)
     model.fit(train.iloc[fit_idx][features],y[fit_idx])
     oof[held_idx]=prob(model,train.iloc[held_idx][features])
    if not np.isfinite(oof).all(): raise RuntimeError('Missing OOF predictions')
    t,m=threshold_select(y,oof)
    ident=f'{rep}__{name}__{config_id}'
    row={'candidate':ident,'representation':rep,'model':name,'config_id':config_id,'cv_mcc':m['mcc'],'cv_balanced_accuracy':m['balanced_accuracy'],'cv_roc_auc':m['roc_auc'],'cv_threshold':t}
    rows.append(row)
    candidates.append({'id':ident,'rep':rep,'features':features,'name':name,'config_id':config_id,'params':params,'oof':oof,'cv':m})
    print(f'  CV {ident:56s} MCC={m["mcc"]:.4f}',flush=True)
 return candidates,pd.DataFrame(rows)

def run_protocol(nodes,protocol,seed,args):
 run_dir=OUT/args.mode/protocol/f'seed_{seed}'
 done=run_dir/'complete.json'
 if done.exists() and not args.force:
  print(f'[SKIP] {protocol} seed={seed}',flush=True);return
 run_dir.mkdir(parents=True,exist_ok=True)
 print('\n'+'='*88+f'\n{protocol} seed={seed}\n'+'='*88,flush=True)
 tr,va,te=get_partitions(nodes,protocol)
 print(f'Partitions train={len(tr):,}, val={len(va):,}, test={len(te):,}',flush=True)
 candidates,cv_df=cv_candidates(tr,seed,args.cv_folds,args.fast)
 cv_df.to_csv(run_dir/'candidate_spatial_cv.csv',index=False)
 # Rank by TRAIN-only spatial CV, avoiding target-country information.
 ranked=sorted(candidates,key=lambda c:(c['cv']['mcc'],c['cv']['balanced_accuracy'],c['cv']['roc_auc'] or 0),reverse=True)
 # Top three unique model families, plus top-ranked LR to preserve linear transfer.
 selected=[]; families=set()
 for c in ranked:
  if c['name'] not in families:
   selected.append(c);families.add(c['name'])
  if len(selected)>=args.top_k:break
 if not any(c['name']=='logistic_regression' for c in selected):
  lr=next((c for c in ranked if c['name']=='logistic_regression'),None)
  if lr is not None: selected.append(lr)
 # Independently refit all members using TRAIN only.
 vy=va.retained.to_numpy(dtype=int)
 fitted={}; val_probs={};test_probs={}
 for c in selected:
  m=model_factory(c['name'],c['params'],seed)
  m.fit(tr[c['features']],tr.retained.to_numpy(dtype=int))
  fitted[c['id']]=m
  val_probs[c['id']]=prob(m,va[c['features']])
  test_probs[c['id']]=prob(m,te[c['features']])
  joblib.dump(m,run_dir/f'{c["id"]}.joblib',compress=3)
  print(f'  fitted {c["id"]}',flush=True)
 # Prespecified ensemble recipes: no weights fitted on labels.
 # Selecting between these recipes uses SOURCE validation only.
 specs=[]
 for c in selected:
  specs.append((f'single__{c["id"]}',[c['id']],[1.]))
 if len(selected)>=2:
  ids=[c['id'] for c in selected]
  specs.append(('ensemble__equal',ids,[1/len(ids)]*len(ids)))
  specs.append(('ensemble__top2_equal',ids[:2],[.5,.5]))
  specs.append(('ensemble__top1_dominant',ids,[.7]+[.3/(len(ids)-1)]*(len(ids)-1)))
 results=[]; probability_cache={}
 for variant,ids,weights in specs:
  pv=sum(w*val_probs[i] for i,w in zip(ids,weights))
  pt=sum(w*test_probs[i] for i,w in zip(ids,weights))
  threshold,vm=threshold_select(vy,pv)
  probability_cache[variant]=(pt,threshold)
  results.append({'protocol':protocol,'seed':seed,'variant':variant,'members':json.dumps(ids),'weights':json.dumps(weights),'threshold':threshold,
                  **{'val_'+k:v for k,v in vm.items()}})
  print(f'  validation {variant:52s} MCC={vm["mcc"]:.4f}',flush=True)
 # Use source validation only; no target labels have been accessed for selection.
 # Require ensemble improvement margin over best individual to avoid weak blending.
 singles=[r for r in results if r['variant'].startswith('single__')]
 winner_single=max(singles,key=lambda r:(r['val_mcc'],r['val_balanced_accuracy'],r['val_roc_auc'] or 0))
 ensembles=[r for r in results if r['variant'].startswith('ensemble__')]
 winner_ens=max(ensembles,key=lambda r:(r['val_mcc'],r['val_balanced_accuracy'],r['val_roc_auc'] or 0)) if ensembles else None
 winner=winner_ens if winner_ens and winner_ens['val_mcc']>=winner_single['val_mcc']+args.ensemble_margin else winner_single
 print(f'  SELECTED {winner["variant"]} (source val MCC={winner["val_mcc"]:.4f})',flush=True)
 yt=te.retained.to_numpy(dtype=int)  # Read target labels only after selection is frozen.
 for row in results:
  pt,t=probability_cache[row['variant']]
  mt=metrics(yt,pt,t)
  row['selected_by_validation']=row['variant']==winner['variant']
  row.update({'test_'+k:v for k,v in mt.items()})
  pred=te[['sample_uid','country','city','split','tile_id','retained']].copy()
  pred['protocol']=protocol;pred['seed']=seed;pred['variant']=row['variant']
  pred['probability_retained']=pt;pred['predicted_retained']=(pt>=t).astype(int);pred['threshold']=t
  pred.to_csv(run_dir/f'{row["variant"]}_test_predictions.csv',index=False)
  print(f'  TEST {row["variant"]:55s} MCC={mt["mcc"]:.4f}',flush=True)
 pd.DataFrame(results).to_csv(run_dir/'variant_test_metrics.csv',index=False)
 safe_json(run_dir/'experiment_metadata.json',{'protocol':protocol,'seed':seed,'n_train':len(tr),'n_validation':len(va),'n_test':len(te),
  'cv_folds':args.cv_folds,'source_only':True,'target_test_labels_for_selection':False,
  'candidate_ranking':'training spatial GroupKFold MCC','selection':'source-validation MCC with ensemble margin',
  'selected_variant':winner['variant'],'members':[c['id'] for c in selected],
  'graph_semantics':'centroid proximity kNN, not physical road topology'})
 safe_json(done,{'status':'COMPLETE','selected_variant':winner['variant']})

def aggregate(mode):
 paths=sorted((OUT/mode).glob('*/seed_*/variant_test_metrics.csv'))
 if not paths:return
 df=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
 df.to_csv(OUT/mode/'all_runs_metrics.csv',index=False)
 df.loc[df.selected_by_validation.astype(str).str.lower().eq('true')].to_csv(OUT/mode/'validation_selected_runs.csv',index=False)
 metrics_cols=['test_mcc','test_balanced_accuracy','test_f1','test_roc_auc','test_pr_auc']
 agg=df.groupby(['protocol','variant'])[metrics_cols].agg(['mean','std','count'])
 agg.columns=['_'.join(col) for col in agg.columns]
 agg.reset_index().to_csv(OUT/mode/'aggregate_metrics.csv',index=False)
 print(f'Wrote aggregate results: {OUT/mode}',flush=True)

def main():
 ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 ap.add_argument('--mode',choices=['development','full'],default='development')
 ap.add_argument('--protocols',nargs='+',choices=list(PROTOCOLS),default=list(PROTOCOLS))
 ap.add_argument('--seed',type=int,default=42)
 ap.add_argument('--cv-folds',type=int,default=3)
 ap.add_argument('--top-k',type=int,default=3)
 ap.add_argument('--ensemble-margin',type=float,default=.003)
 ap.add_argument('--fast',action='store_true',help='Only LR and ExtraTrees for quick smoke tests')
 ap.add_argument('--force',action='store_true')
 args=ap.parse_args()
 if args.cv_folds<2 or args.top_k<2 or args.ensemble_margin<0:ap.error('Require cv-folds>=2, top-k>=2, ensemble-margin>=0')
 configure_representation('structural')
 nodes,edges=load_data()
 nodes=add_structural_features(nodes)
 print(f'CartoGraph-X V7 | mode={args.mode} | nodes={len(nodes):,} | edges={len(edges):,}',flush=True)
 (OUT/args.mode).mkdir(parents=True,exist_ok=True)
 seeds=[args.seed] if args.mode=='development' else list(range(42,52))
 for seed in seeds:
  for protocol in args.protocols:
   run_protocol(nodes,protocol,seed,args)
   aggregate(args.mode)
 print('[PASS] V7 completed',flush=True)

if __name__=='__main__':main()
