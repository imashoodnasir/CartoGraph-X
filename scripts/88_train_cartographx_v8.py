#!/usr/bin/env python
"""CartoGraph-X V8: source-safe, domain-aware feature selection.

Place beside 87_train_cartographx_v7.py in scripts/. Uses its graph loader,
structural features, model factory, metrics, and protocol definitions.

Run:
 python scripts/88_train_cartographx_v8.py --mode development --protocols lt_to_pl_zero_shot pl_to_lt_zero_shot
 python scripts/88_train_cartographx_v8.py --mode full

Target-country training FEATURES (not labels) are used to assess country
separability. No target labels are used in fitting/selection/calibration.
Target test labels are accessed once, for final diagnostic evaluation.
"""
from __future__ import annotations
import argparse, importlib.util, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
warnings.filterwarnings('ignore', category=FutureWarning)
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cartographx_v7',HERE/'87_train_cartographx_v7.py')
if spec is None or spec.loader is None: raise RuntimeError('Place Script 87 next to Script 88')
v7=importlib.util.module_from_spec(spec); spec.loader.exec_module(v7)
OUT=v7.DATA/'cartographx_v8_models'
SEEDS=list(range(42,52))


def jdump(p,data):
 p.write_text(json.dumps(data,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')


def audit(nodes,out):
 # Descriptive audit, not model selection. Do not assume prevalence equivalence.
 rows=[]
 for (country,split),g in nodes.groupby(['country','split']):
  y=g['retained'].dropna().astype(int)
  rows.append(dict(country=country,split=split,n=len(g),retained=int(y.sum()),retained_rate=float(y.mean()),
                   n_tiles=int(g.tile_id.nunique()),n_missing_labels=int(g.retained.isna().sum())))
 pd.DataFrame(rows).to_csv(out/'label_prevalence_audit.csv',index=False)
 # Detect feature availability and basic schema inconsistencies.
 feats=list(dict.fromkeys(v7.TAB+v7.GRAPH+v7.STRUCT))
 schema=[]
 for country,g in nodes.groupby('country'):
  for f in feats:
   x=pd.to_numeric(g[f],errors='coerce').replace([np.inf,-np.inf],np.nan)
   schema.append(dict(country=country,feature=f,n=len(x),missing_rate=float(x.isna().mean()),
                      median=float(x.median()) if x.notna().any() else None,
                      p05=float(x.quantile(.05)) if x.notna().any() else None,
                      p95=float(x.quantile(.95)) if x.notna().any() else None))
 pd.DataFrame(schema).to_csv(out/'feature_provenance_proxy_audit.csv',index=False)


def source_feature_scores(source,domain_source,domain_target,features):
 """Score predictive value using SOURCE TRAIN labels, domain shift using TRAIN features only.

 No target labels. Univariate source ROC-AUC is descriptive/in-sample; feature
 selection is subsequently validated with source-only spatial OOF models.
 """
 y=source.retained.to_numpy(dtype=int)
 rows=[]
 for f in features:
  a=pd.to_numeric(domain_source[f],errors='coerce').replace([np.inf,-np.inf],np.nan)
  b=pd.to_numeric(domain_target[f],errors='coerce').replace([np.inf,-np.inf],np.nan)
  s=pd.to_numeric(source[f],errors='coerce').replace([np.inf,-np.inf],np.nan)
  pred_auc=.5
  if s.notna().sum()>10 and s.nunique()>1 and len(np.unique(y[s.notna()]))==2:
   pred_auc=roc_auc_score(y[s.notna()],s[s.notna()])
  pred_strength=abs(pred_auc-.5)*2
  domain_auc=.5
  if a.notna().sum()>10 and b.notna().sum()>10:
   z=np.r_[a.dropna().to_numpy(),b.dropna().to_numpy()]
   d=np.r_[np.zeros(a.notna().sum()),np.ones(b.notna().sum())]
   if np.unique(z).size>1: domain_auc=roc_auc_score(d,z)
  domain_strength=abs(domain_auc-.5)*2
  # Ranking heuristic, not a calibrated estimate of causal invariance.
  score=pred_strength-.50*domain_strength
  rows.append(dict(feature=f,source_predictive_auc=float(max(pred_auc,1-pred_auc)),
                   country_auc=float(max(domain_auc,1-domain_auc)),
                   predictive_strength=float(pred_strength),country_strength=float(domain_strength),
                   invariant_score=float(score),
                   source_missing=float(s.isna().mean()),target_missing=float(b.isna().mean())))
 return pd.DataFrame(rows).sort_values('invariant_score',ascending=False).reset_index(drop=True)


def get_feature_sets(scores):
 all_feats=list(dict.fromkeys(v7.TAB+v7.GRAPH+v7.STRUCT))
 rank=scores.feature.tolist()
 sets={'original_tabular_graph':list(v7.TAB+v7.GRAPH),
       'structural_only':list(v7.STRUCT),
       'full_structural':all_feats,
       'invariant_top8':rank[:8],
       'invariant_top12':rank[:12],
       'invariant_top16':rank[:16],
       'invariant_top20':rank[:20]}
 # Domain-screened subset: only source-predictive features with relatively
 # low univariate country separability. Fallback is rank-based.
 screen=scores[(scores.country_auc<=.70)&(scores.source_predictive_auc>=.54)].feature.tolist()
 sets['domain_screened']=screen[:20] if len(screen)>=4 else rank[:12]
 return {k:list(dict.fromkeys(v)) for k,v in sets.items() if len(v)>=2}


def oof_model(train,features,name,params,seed,folds=3):
 y=train.retained.to_numpy(dtype=int)
 groups=v7.spatial_groups(train)
 n=min(folds,len(np.unique(groups)))
 if n<2: raise RuntimeError('Need >=2 training tiles')
 oof=np.full(len(train),np.nan)
 cv=GroupKFold(n_splits=n)
 for k,(it,iv) in enumerate(cv.split(train,y,groups),1):
  if np.unique(y[it]).size<2: raise RuntimeError('Single-class spatial training fold')
  model=v7.model_factory(name,params,seed+k)
  model.fit(train.iloc[it][features],y[it]); oof[iv]=v7.prob(model,train.iloc[iv][features])
 if not np.isfinite(oof).all(): raise RuntimeError('Incomplete OOF')
 t,m=v7.threshold_select(y,oof)
 return t,m


def run_one(nodes,protocol,seed,out,fast=False):
 train,val,test=v7.get_partitions(nodes,protocol)
 source_c=val.country.iloc[0]; target_c=test.country.iloc[0]
 # Country discriminator uses only country-specific TRAIN features, regardless
 # of which country is the held-out evaluation target.
 domain_a=nodes[(nodes.country==source_c)&(nodes.split=='train')]
 domain_b=nodes[(nodes.country==target_c)&(nodes.split=='train')]
 feats=list(dict.fromkeys(v7.TAB+v7.GRAPH+v7.STRUCT))
 if source_c==target_c:
  # For same-country protocols, there is no cross-country adaptation target.
  # Feature scores are predictive-only, with domain penalty set to zero.
  scores=source_feature_scores(train,domain_a,domain_a,feats)
  scores['country_auc']=.5; scores['country_strength']=0.
  scores['invariant_score']=scores['predictive_strength']
  scores=scores.sort_values('invariant_score',ascending=False).reset_index(drop=True)
 else:
  scores=source_feature_scores(train,domain_a,domain_b,feats)
 run_dir=out/protocol/f'seed_{seed}'
 run_dir.mkdir(parents=True,exist_ok=True)
 scores.to_csv(run_dir/'feature_invariance_scores.csv',index=False)
 sets=get_feature_sets(scores)
 jdump(run_dir/'feature_sets.json',sets)
 models=[('logistic_regression',v7.CONFIGS['logistic_regression'][0]),
         ('random_forest',v7.CONFIGS['random_forest'][1]),
         ('extra_trees',v7.CONFIGS['extra_trees'][1])]
 if not fast: models.append(('hist_gradient_boosting',v7.CONFIGS['hist_gradient_boosting'][0]))
 cv_rows=[]
 print(f'\n{protocol} seed={seed} | train={len(train):,} val={len(val):,} test={len(test):,}',flush=True)
 for fs,features in sets.items():
  for name,params in models:
   t,m=oof_model(train,features,name,params,seed)
   cv_rows.append(dict(feature_set=fs,model=name,cv_threshold=t,cv_mcc=m['mcc'],cv_ba=m['balanced_accuracy'],cv_auc=m['roc_auc']))
   print(f'  CV {fs:24s} {name:23s} MCC={m["mcc"]:.4f}',flush=True)
 cv_df=pd.DataFrame(cv_rows).sort_values(['cv_mcc','cv_ba','cv_auc'],ascending=False)
 cv_df.to_csv(run_dir/'source_train_spatial_cv.csv',index=False)
 # Only the top three CV candidates are evaluated on source validation.
 # Source validation then selects one candidate and its decision threshold.
 candidates=[]
 for _,r in cv_df.head(3).iterrows():
  fs,name=str(r.feature_set),str(r.model)
  params=next(p for n,p in models if n==name)
  model=v7.model_factory(name,params,seed)
  model.fit(train[sets[fs]],train.retained.to_numpy(dtype=int))
  pv=v7.prob(model,val[sets[fs]])
  threshold,vm=v7.threshold_select(val.retained.to_numpy(dtype=int),pv)
  candidates.append((vm['mcc'],vm['balanced_accuracy'],vm['roc_auc'] or -1,fs,name,threshold,vm,model))
  print(f'  VAL {fs:24s} {name:23s} MCC={vm["mcc"]:.4f}',flush=True)
 selected=max(candidates,key=lambda r:r[:3])
 _,_,_,fs,name,t,vm,model=selected
 # Evaluate test exactly once AFTER model/feature/threshold selection.
 pt=v7.prob(model,test[sets[fs]])
 tm=v7.metrics(test.retained.to_numpy(dtype=int),pt,t)
 print(f'  SELECTED {fs}/{name} threshold={t:.2f} val_MCC={vm["mcc"]:.4f} test_MCC={tm["mcc"]:.4f}',flush=True)
 result=dict(protocol=protocol,seed=seed,feature_set=fs,model=name,threshold=t,
             train_n=len(train),val_n=len(val),test_n=len(test),
             val_mcc=vm['mcc'],test_mcc=tm['mcc'],test_ba=tm['balanced_accuracy'],
             test_roc_auc=tm['roc_auc'],test_pr_auc=tm['pr_auc'],
             source_country=source_c,target_country=target_c,
             source_val_prevalence=float(val.retained.mean()),target_test_prevalence=float(test.retained.mean()))
 pd.DataFrame([result]).to_csv(run_dir/'selected_result.csv',index=False)
 pd.DataFrame({'sample_uid':test.sample_uid,'retained':test.retained,'probability':pt,
               'predicted':(pt>=t).astype(int)}).to_csv(run_dir/'final_test_predictions.csv',index=False)
 return result


def main():
 p=argparse.ArgumentParser()
 p.add_argument('--mode',choices=['development','full'],default='development')
 p.add_argument('--protocols',nargs='+',choices=list(v7.PROTOCOLS),default=list(v7.PROTOCOLS))
 p.add_argument('--fast',action='store_true',help='Omit HGB candidates')
 args=p.parse_args()
 nodes,_=v7.load_data()
 nodes=v7.add_structural_features(nodes)
 out=OUT/args.mode; out.mkdir(parents=True,exist_ok=True)
 audit(nodes,out)
 seeds=[42] if args.mode=='development' else SEEDS
 results=[]
 for seed in seeds:
  for protocol in args.protocols:
   results.append(run_one(nodes,protocol,seed,out,args.fast))
   pd.DataFrame(results).to_csv(out/'all_runs_metrics.csv',index=False)
 df=pd.DataFrame(results)
 summary=df.groupby('protocol',as_index=False).agg(n_seeds=('seed','size'),
         mean_test_mcc=('test_mcc','mean'),std_test_mcc=('test_mcc','std'),
         mean_test_ba=('test_ba','mean'),mean_test_roc_auc=('test_roc_auc','mean'))
 summary.to_csv(out/'aggregate_metrics.csv',index=False)
 print('\n[PASS] V8 complete. Review label_prevalence_audit.csv, feature scores, and aggregate_metrics.csv')

if __name__=='__main__': main()
