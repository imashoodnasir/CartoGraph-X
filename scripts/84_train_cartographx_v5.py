#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CartoGen Script 84 — CartoGraph-X V5, spatial-CV-selected anchor + reliability-gated
spatial graph residual learning.

Usage (from D:\\Research\\CartoGen):
  python scripts\\84_train_cartographx_v5.py --mode development
  python scripts\\84_train_cartographx_v5.py --mode full
  python scripts\\84_train_cartographx_v5.py --mode development --force

Requires: numpy, pandas, scipy, scikit-learn, torch, joblib.
Inputs (unchanged): graph_representations/road_graph_nodes.csv and
road_graph_edges.csv from Script 68. All output goes to cartographx_v5_models.

Design: select an anchor from LR/RF/ET/HGB x tabular/combined features via
TRAIN-only spatial GroupKFold OOF MCC; reuse those OOF train probabilities;
train residual message-passing correction with OOF anchor logits on training
nodes, full-train anchor logits on validation/test nodes. Independent graph
partitions prevent cross-country/split/tile edges. Select anchor threshold, residual epoch and variant
using source validation only. No target labels are read for zero-shot selection.

Caveat: spatial kNN edges represent proximity, not verified road connectivity.
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
OUT=DATA/'cartographx_v5_models'
TAB=['length_m','intersecting_road_count','neighbor_count_150m','mean_neighbor_distance_m']
GRAPH=['graph_degree','graph_mean_edge_distance_m','graph_min_edge_distance_m',
       'graph_max_edge_distance_m','graph_local_clustering','graph_component_size',
       'graph_is_isolated','local_road_count_50m','local_road_count_100m',
       'local_road_count_150m','local_road_count_250m','nbr_mean_length_m',
       'nbr_std_length_m','nbr_mean_intersecting_road_count',
       'nbr_std_intersecting_road_count','nbr_mean_neighbor_count_150m',
       'nbr_std_neighbor_count_150m','nbr_mean_mean_neighbor_distance_m',
       'nbr_std_mean_neighbor_distance_m']
FEATURE_SETS={'tabular':TAB,'tabular_graph':TAB+GRAPH}
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
 'V5-A_anchor': {'graph_dropout':0., 'max_delta':0., 'confidence_penalty':0.},
 'V5-B_bounded': {'graph_dropout':0., 'max_delta':0.35, 'confidence_penalty':0.},
 'V5-C_confidence': {'graph_dropout':0., 'max_delta':0.35, 'confidence_penalty':0.15},
 'V5-D_dropout': {'graph_dropout':0.25, 'max_delta':0.35, 'confidence_penalty':0.15},
 'V5-E_strict': {'graph_dropout':0.25, 'max_delta':0.18, 'confidence_penalty':0.30},
}
ANCHOR_NAME='V5-A_anchor'

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

def graph_indices(frame,edges):
 # The graph is constructed once by Script 68, with labels excluded.
 index=pd.Series(np.arange(len(frame),dtype=np.int64),index=frame.sample_uid)
 relevant=edges.loc[edges.source_uid.isin(index.index)&edges.target_uid.isin(index.index)]
 if len(relevant):
  s=index.loc[relevant.source_uid].to_numpy(dtype=np.int64)
  t=index.loc[relevant.target_uid].to_numpy(dtype=np.int64)
  # undirected edges, no self edges; add self loops in message-passing module.
  ei=np.vstack((np.r_[s,t],np.r_[t,s]))
 else: ei=np.zeros((2,0),dtype=np.int64)
 return ei

class GraphResidual(nn.Module):
 def __init__(self,d,hidden=64,dropout=.15,gated=True,graph_drop=0.,max_delta=.5):
  super().__init__(); self.gated=gated; self.graph_drop=graph_drop; self.max_delta=max_delta
  self.self_encoder=nn.Sequential(nn.Linear(d,hidden),nn.ReLU(),nn.LayerNorm(hidden))
  self.neighbor_encoder=nn.Sequential(nn.Linear(d,hidden),nn.ReLU(),nn.LayerNorm(hidden))
  self.residual=nn.Sequential(nn.Linear(hidden*2+2,hidden),nn.ReLU(),
                              nn.Dropout(dropout),nn.Linear(hidden,1))
  self.gate=nn.Sequential(nn.Linear(hidden*2+2,hidden//2),nn.ReLU(),
                          nn.Linear(hidden//2,1)) if gated else None
  # Zero residual at initialization; gate learns whether corrections help.
  nn.init.zeros_(self.residual[-1].weight); nn.init.zeros_(self.residual[-1].bias)
  if gated: nn.init.constant_(self.gate[-1].bias,-2.)
 def forward(self,x,anchor_logits,edge_index):
  n=x.shape[0]; src,dst=edge_index
  # Spatial aggregation on unlabeled node features, no label message passing.
  # Neighbors' anchor logits are deliberately excluded: train OOF versus
  # validation/test full-fit anchors would otherwise amplify a shift.
  neigh=torch.zeros_like(x); count=torch.zeros((n,1),device=x.device)
  if src.numel():
   neigh.index_add_(0,dst,x[src]); count.index_add_(0,dst,torch.ones((len(dst),1),device=x.device))
  neigh=neigh/count.clamp(min=1.)
  if self.training and self.graph_drop:
   keep=(torch.rand((n,1),device=x.device)>=self.graph_drop).float()
   neigh=neigh*keep
  h=self.self_encoder(x); hn=self.neighbor_encoder(neigh)
  extra=torch.cat((anchor_logits[:,None],torch.log1p(count)),dim=1)
  features=torch.cat((h,hn,extra),dim=1)
  delta=self.max_delta*torch.tanh(self.residual(features).squeeze(-1))
  gate=torch.sigmoid(self.gate(features).squeeze(-1)) if self.gated else torch.ones_like(delta)
  return anchor_logits+gate*delta,gate,delta

def tensor_bundle(frame,anchor_probs,edge_index,scaler,device):
 x=frame[TAB+GRAPH].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float64)
 x=np.where(np.isfinite(x),x,np.nan)
 x=scaler.transform(x).astype('float32')
 return (torch.as_tensor(x,device=device),
         torch.as_tensor(logit(anchor_probs),dtype=torch.float32,device=device),
         torch.as_tensor(edge_index,dtype=torch.long,device=device))

def run_variant(variant,train,val,train_bundle,val_bundle,ytrain,yval,seed,args,device):
 config=VARIANTS[variant]; dropout=config['graph_dropout']; gated=True
 net=GraphResidual(train_bundle[0].shape[1],args.hidden,args.dropout,gated,dropout,config['max_delta']).to(device)
 optim=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
 target=torch.as_tensor(ytrain,dtype=torch.float32,device=device)
 # Train supervision is only from training labels. Validation labels choose epoch/threshold.
 # Zero-shot validation is SOURCE-country validation, never target labels.
 pos=max(1.,float((len(ytrain)-np.sum(ytrain))/max(1,np.sum(ytrain))))
 pos_weight=torch.tensor(min(pos,8.),dtype=torch.float32,device=device)
 criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
 best=None; bad=0; history=[]
 for epoch in range(1,args.epochs+1):
  net.train(); optim.zero_grad(set_to_none=True)
  logits,gate,delta=net(*train_bundle)
  # Penalize actual gated correction, especially on confident anchor predictions.
  # This encourages transfer-safe corrections instead of overriding reliable anchors.
  correction=logits-train_bundle[1]
  confidence=(torch.sigmoid(train_bundle[1])-.5).abs()*2.
  loss=(criterion(logits,target)
        +args.residual_penalty*correction.square().mean()
        +config['confidence_penalty']*(confidence*correction.square()).mean())
  loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
  optim.step()
  if epoch%args.eval_every and epoch!=args.epochs: continue
  net.eval()
  with torch.no_grad(): v_logits,v_gate,v_delta=net(*val_bundle)
  vp=torch.sigmoid(v_logits).cpu().numpy()
  threshold,vm=threshold_select(yval,vp)
  key=(vm['mcc'],vm['balanced_accuracy'],vm['f1'])
  history.append({'epoch':epoch,'train_loss':float(loss.item()),
                  'validation_mcc':vm['mcc'],'validation_balanced_accuracy':vm['balanced_accuracy'],
                  'threshold':threshold,'mean_gate':float(v_gate.mean().item())})
  if best is None or key>best['key']:
   best={'key':key,'state':copy.deepcopy({k:v.detach().cpu() for k,v in net.state_dict().items()}),
         'epoch':epoch,'threshold':threshold,'val_metrics':vm};bad=0
  else: bad+=1
  if bad>=args.patience: break
 net.load_state_dict(best['state']);net.eval()
 return net,best,pd.DataFrame(history)

def write_json(path,data):
 with open(path,'w',encoding='utf-8') as f:json.dump(data,f,indent=2,default=str)

def run_protocol(nodes,edges,protocol,seed,args,device):
 run_dir=OUT/args.mode/protocol/f'seed_{seed}'
 complete=run_dir/'COMPLETE.json'
 if complete.exists() and not args.force:
  print(f'[SKIP] {protocol} seed={seed} complete',flush=True);return
 run_dir.mkdir(parents=True,exist_ok=True)
 print(f'\n{"="*90}\n{protocol} seed={seed}\n{"="*90}',flush=True)
 seed_everything(seed)
 tr,va,te=get_partitions(nodes,protocol)
 print(f'Partitions train={len(tr):,}, val={len(va):,}, test={len(te):,}',flush=True)
 # Anchor selection uses source validation even for zero-shot protocols.
 anchor,search=choose_anchor(tr,va,seed,args.fast_anchor,args.anchor_cv_folds)
 search.to_csv(run_dir/'anchor_validation_search.csv',index=False)
 print(f'Selected anchor by TRAIN spatial CV: {anchor["model_name"]}/{anchor["representation"]}, '
       f'CV MCC={anchor["cv_metrics"]["mcc"]:.4f}, source-val MCC={anchor["val_metrics"]["mcc"]:.4f}',flush=True)
 oof=anchor['oof']  # exact training-only spatial OOF scores used to select anchor
 vp=prob(anchor['model'],va[anchor['features']]); tp=prob(anchor['model'],te[anchor['features']])
 # The full-fit model is never trained on validation/test data.
 anchor_t=anchor['threshold']; ytr=tr.retained.to_numpy(dtype=int)
 yv=va.retained.to_numpy(dtype=int)
 # Train-only median imputation and standardization of graph node features.
 Xtr=tr[TAB+GRAPH].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float64)
 Xtr=np.where(np.isfinite(Xtr),Xtr,np.nan)
 imputer=SimpleImputer(strategy='median',keep_empty_features=True)
 xtrain=imputer.fit_transform(Xtr)
 scaler=StandardScaler();scaler.fit(xtrain)
 class FeatureTransformer:
  def transform(self,x):return scaler.transform(imputer.transform(x))
 transform=FeatureTransformer()
 # tensor_bundle expects .transform to produce final features.
 train_e=graph_indices(tr,edges); val_e=graph_indices(va,edges); test_e=graph_indices(te,edges)
 def bundle(frame,p,e):
  x=frame[TAB+GRAPH].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float64)
  x=np.where(np.isfinite(x),x,np.nan)
  xx=transform.transform(x).astype('float32')
  return (torch.as_tensor(xx,device=device),
          torch.as_tensor(logit(p),dtype=torch.float32,device=device),
          torch.as_tensor(e,dtype=torch.long,device=device))
 train_b=bundle(tr,oof,train_e);val_b=bundle(va,vp,val_e)
 # Keep test predictions strictly after all validation-based choices.
 joblib.dump({'anchor':anchor['model'],'imputer':imputer,'scaler':scaler,
             'features':anchor['features'],'params':anchor['params']},run_dir/'anchor_and_scalers.joblib')
 results=[]; allpred=[]; candidate_records=[]
 for variant in VARIANTS:
  if variant==ANCHOR_NAME:
   threshold=anchor_t; valmet=metrics(yv,vp,threshold)
   candidate_records.append((valmet['mcc'],valmet['balanced_accuracy'],valmet['f1'],variant))
   # Test is evaluated once below, after all validation candidates have been finalized.
   results.append({'variant':variant,'selected_epoch':0,'threshold':threshold,'validation_mcc':valmet['mcc'],
                   'validation_balanced_accuracy':valmet['balanced_accuracy'],'model_path':None})
   continue
  print(f'  Training {variant} ...',flush=True)
  net,best,hist=run_variant(variant,tr,va,train_b,val_b,ytr,yv,seed,args,device)
  hist.to_csv(run_dir/f'{variant}_training_history.csv',index=False)
  path=run_dir/f'{variant}_model.pt'
  torch.save({'state_dict':best['state'],'architecture':{'input_dim':len(TAB+GRAPH),
              'hidden':args.hidden,'dropout':args.dropout,'gated':True,
              'graph_dropout':VARIANTS[variant]['graph_dropout'],
              'max_delta':VARIANTS[variant]['max_delta']},'epoch':best['epoch'],
              'threshold':best['threshold'],'seed':seed},path)
  vm=best['val_metrics']
  candidate_records.append((vm['mcc'],vm['balanced_accuracy'],vm['f1'],variant))
  results.append({'variant':variant,'selected_epoch':best['epoch'],
                  'threshold':best['threshold'],'validation_mcc':vm['mcc'],
                  'validation_balanced_accuracy':vm['balanced_accuracy'],'model_path':str(path)})
  print(f'    epoch={best["epoch"]} val_MCC={vm["mcc"]:.4f} threshold={best["threshold"]:.2f}',flush=True)
 # Select winning variant exclusively from validation, before reading test labels.
 # Ties prefer the anchor and then smaller corrections; never optimize on test.
 # A small preregistered margin rejects near-ties; no test-based selection.
 anchor_score=next(z[0] for z in candidate_records if z[-1]==ANCHOR_NAME)
 eligible=[z for z in candidate_records if z[-1]!=ANCHOR_NAME and z[0]>=anchor_score+args.selection_margin]
 selected=max(eligible)[-1] if eligible else ANCHOR_NAME
 print(f'  Validation-selected variant: {selected}',flush=True)
 test_b=bundle(te,tp,test_e)
 yt=te.retained.to_numpy(dtype=int)  # strictly after validation-only selection
 for row in results:
  variant=row['variant']; threshold=row['threshold']
  if variant==ANCHOR_NAME: p=tp.copy();g=np.zeros(len(te));delta=np.zeros(len(te))
  else:
   ck=torch.load(row['model_path'],map_location='cpu',weights_only=False)
   conf=ck['architecture']
   net=GraphResidual(d=conf['input_dim'],hidden=conf['hidden'],dropout=conf['dropout'],gated=conf['gated'],graph_drop=conf['graph_dropout'],max_delta=conf['max_delta']).to(device);net.load_state_dict(ck['state_dict']);net.eval()
   with torch.no_grad(): logits,gg,dd=net(*test_b)
   p=torch.sigmoid(logits).cpu().numpy();g=gg.cpu().numpy();delta=dd.cpu().numpy()
   del net
  testmet=metrics(yt,p,threshold)
  row.update({'protocol':protocol,'seed':seed,'selected_by_validation':variant==selected,
              'anchor_model':anchor['model_name'],'anchor_representation':anchor['representation'],
              'anchor_config':json.dumps(anchor['params']),
              **{'test_'+k:v for k,v in testmet.items()}})
  pred=te[['sample_uid','country','city','split','tile_id','retained']].copy()
  pred['protocol']=protocol;pred['seed']=seed;pred['variant']=variant
  pred['probability_retained']=p;pred['predicted_retained']=(p>=threshold).astype(int)
  pred['threshold']=threshold;pred['anchor_probability']=tp
  pred['gate']=g;pred['graph_residual_logit']=delta
  pred.to_csv(run_dir/f'{variant}_test_predictions.csv',index=False)
  allpred.append(pred)
  print(f'    TEST {variant:21s} MCC={testmet["mcc"]:.4f} BA={testmet["balanced_accuracy"]:.4f}',flush=True)
 pd.DataFrame(results).to_csv(run_dir/'variant_test_metrics.csv',index=False)
 # Do not aggregate test outcomes into a decision to change selected variant.
 write_json(run_dir/'experiment_metadata.json',{'protocol':protocol,'seed':seed,
  'train_countries':PROTOCOLS[protocol][0], 'validation_country':PROTOCOLS[protocol][1],
  'test_country':PROTOCOLS[protocol][2], 'n_train':len(tr),'n_validation':len(va),'n_test':len(te),
  'train_graph_edges_directed':int(train_e.shape[1]),'validation_graph_edges_directed':int(val_e.shape[1]),
  'test_graph_edges_directed':int(test_e.shape[1]),'oof_folds':min(args.anchor_cv_folds,len(np.unique(spatial_groups(tr)))),
  'anchor':{'model':anchor['model_name'],'representation':anchor['representation'],
            'params':anchor['params'],'validation_threshold':anchor_t,
            'training_spatial_cv_mcc':anchor['cv_metrics']['mcc']},
  'validation_selected_variant':selected,'test_not_used_for_selection':True,
  'conservative_correction':True,'anchor_selection':'train-only spatial OOF MCC',
  'source_validation_only_for_threshold_and_variant':True,'candidate_specs':VARIANTS,
  'graph_semantics':'centroid proximity kNN, not verified physical road topology'})
 write_json(complete,{'status':'COMPLETE','protocol':protocol,'seed':seed,'selected_variant':selected})
 del train_b,val_b,test_b
 if device.type=='cuda':torch.cuda.empty_cache()

def aggregate(mode):
 files=sorted((OUT/mode).glob('*/seed_*/variant_test_metrics.csv'))
 if not files:return
 df=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
 df.to_csv(OUT/mode/'all_runs_metrics.csv',index=False)
 cols=['test_mcc','test_balanced_accuracy','test_f1','test_roc_auc','test_pr_auc']
 summary=df.groupby(['protocol','variant'],as_index=False)[cols].agg(['mean','std','count'])
 summary.columns=['_'.join([str(x) for x in col if str(x)]) for col in summary.columns.to_flat_index()]
 summary.to_csv(OUT/mode/'aggregate_metrics.csv',index=False)
 selected=df.loc[df.selected_by_validation.astype(str).str.lower().eq('true')]
 selected.to_csv(OUT/mode/'validation_selected_runs.csv',index=False)
 print(f'\nWrote aggregate results: {OUT/mode}',flush=True)

def main():
 ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 ap.add_argument('--mode',choices=['development','full'],default='development')
 ap.add_argument('--protocols',nargs='+',choices=list(PROTOCOLS),default=list(PROTOCOLS))
 ap.add_argument('--seed',type=int,default=42)
 ap.add_argument('--force',action='store_true')
 ap.add_argument('--fast-anchor',action='store_true',help='Only ET/HGB anchor candidates (faster smoke test)')
 ap.add_argument('--anchor-cv-folds',type=int,default=3,help='Training-only spatial GroupKFold for anchor selection and OOF')
 ap.add_argument('--epochs',type=int,default=100)
 ap.add_argument('--eval-every',type=int,default=2)
 ap.add_argument('--patience',type=int,default=10,help='Validation checks without improvement')
 ap.add_argument('--hidden',type=int,default=64)
 ap.add_argument('--dropout',type=float,default=.15)
 ap.add_argument('--lr',type=float,default=.001)
 ap.add_argument('--weight-decay',type=float,default=.001)
 ap.add_argument('--residual-penalty',type=float,default=.20)
 ap.add_argument('--selection-margin',type=float,default=.002,help='Minimum source-validation MCC improvement required to select a correction over anchor')
 ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
 args=ap.parse_args()
 if args.anchor_cv_folds<2 or args.eval_every<1 or args.selection_margin<0:ap.error('--anchor-cv-folds >=2, --eval-every >=1, --selection-margin >=0 required')
 device=torch.device('cuda' if args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available()) else 'cpu')
 if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
 print(f'CartoGraph-X V5 | mode={args.mode} | device={device}',flush=True)
 nodes,edges=load_data(); print(f'Input nodes={len(nodes):,}, undirected edges={len(edges):,}',flush=True)
 seeds=[args.seed] if args.mode=='development' else list(range(42,52))
 (OUT/args.mode).mkdir(parents=True,exist_ok=True)
 for seed in seeds:
  for protocol in args.protocols:
   run_protocol(nodes,edges,protocol,seed,args,device)
   aggregate(args.mode)
 print('\n[PASS] V5 completed. Compare validation-selected variants with matched-seed baselines.',flush=True)
if __name__=='__main__':main()
