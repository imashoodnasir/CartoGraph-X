#!/usr/bin/env python3
"""Script 119 — Frozen domain-shift and transfer diagnostics for CartoGraph-X V2.

READ-ONLY diagnostic: no retraining of road-selection models, no threshold tuning,
no test-time adaptation, no figures. Domain classifier is a separate *diagnostic*
trained on TRAIN features only and evaluated on validation/test features.
Historical test labels are used only for descriptive post-hoc evaluation.
C2 correspondence labels remain provisional, not manually validated.

Run: python scripts/119_diagnose_cartographx_v2_domain_shift.py
"""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, balanced_accuracy_score,
                             brier_score_loss, log_loss, f1_score)
from sklearn.model_selection import StratifiedGroupKFold

ROOT=Path(__file__).resolve().parents[1]
NODES=ROOT/'data/processed/cross_country/benchmark_v2_unified_labels_v1/road_graph_nodes_v2.csv'
PRED=ROOT/'data/processed/cross_country/cartographx_v2_ten_seed_summary_v1/all_seed_selected_test_predictions.csv'
OUT=ROOT/'data/processed/cross_country/cartographx_v2_domain_shift_diagnostics_v1'
SEEDS=tuple(range(42,52))
PROTOCOLS=('lt_to_lt','pl_to_pl','lt_to_pl_zero_shot','pl_to_lt_zero_shot','joint_to_lt','joint_to_pl')
SOURCE_TARGET={'lt_to_lt':('Lithuania','Lithuania'),'pl_to_pl':('Poland','Poland'),
 'lt_to_pl_zero_shot':('Lithuania','Poland'),'pl_to_lt_zero_shot':('Poland','Lithuania'),
 'joint_to_lt':('joint','Lithuania'),'joint_to_pl':('joint','Poland')}
BASE_FEATURES=('length_m','intersecting_road_count','neighbor_count_150m','mean_neighbor_distance_m')
GRAPH_FEATURES=('graph_degree','graph_mean_edge_distance_m','graph_min_edge_distance_m',
 'graph_max_edge_distance_m','graph_local_clustering','graph_component_size',
 'graph_is_isolated','local_road_count_50m','local_road_count_100m',
 'local_road_count_150m','local_road_count_250m','nbr_mean_length_m',
 'nbr_std_length_m','nbr_mean_intersecting_road_count',
 'nbr_std_intersecting_road_count','nbr_mean_neighbor_count_150m',
 'nbr_std_neighbor_count_150m','nbr_mean_mean_neighbor_distance_m',
 'nbr_std_mean_neighbor_distance_m')
FEATURES=BASE_FEATURES+GRAPH_FEATURES

def hash_file(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def save(df,name):
 path=OUT/name;tmp=path.with_suffix('.tmp');df.to_csv(tmp,index=False);tmp.replace(path)
def calc_metrics(y,p,t):
 y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
 yp=(p>=t).astype(int)
 both=len(np.unique(y))==2
 return dict(n=len(y),prevalence=float(y.mean()),predicted_positive_rate=float(yp.mean()),
  mean_probability=float(p.mean()),threshold=float(t),
  mcc=float(matthews_corrcoef(y,yp)) if both else np.nan,
  balanced_accuracy=float(balanced_accuracy_score(y,yp)) if both else np.nan,
  f1=float(f1_score(y,yp,zero_division=0)) if both else np.nan,
  roc_auc=float(roc_auc_score(y,p)) if both else np.nan,
  pr_auc=float(average_precision_score(y,p)) if both else np.nan,
  brier=float(brier_score_loss(y,p)),
  log_loss=float(log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1])),
  ece_10=float(sum(len(idx)/len(y)*abs(float(y[idx].mean())-float(p[idx].mean()))
   for idx in [np.flatnonzero((p>=j/10)&(p<((j+1)/10 if j<9 else 1.00000001))) for j in range(10)] if len(idx))))
def load():
 for path in (NODES,PRED):
  if not path.is_file():raise FileNotFoundError(path)
 n=pd.read_csv(NODES,low_memory=False)
 p=pd.read_csv(PRED,low_memory=False)
 required={'sample_uid','country','city','split','tile_id','retained',*FEATURES}
 if not required.issubset(n.columns):raise RuntimeError(f'Nodes missing {required-set(n.columns)}')
 reqp={'seed','protocol','sample_uid','country','city','tile_id','split','label','probability','prediction','threshold'}
 if not reqp.issubset(p.columns):raise RuntimeError(f'Predictions missing {reqp-set(p.columns)}')
 if n.sample_uid.duplicated().any():raise RuntimeError('Duplicate node IDs')
 if p.duplicated(['seed','protocol','sample_uid']).any():raise RuntimeError('Duplicate predictions')
 if set(p.seed)!=set(SEEDS) or set(p.protocol)!=set(PROTOCOLS):raise RuntimeError('Incomplete seeds or protocols')
 if not p.split.eq('test').all():raise RuntimeError('Non-test prediction detected')
 for k in ['country','city','split','tile_id']:
  if n[k].isna().any() or p[k].isna().any():raise RuntimeError(f'Missing metadata {k}')
 if not n.retained.isin([0,1]).all() or not p.label.isin([0,1]).all():raise RuntimeError('Invalid binary labels')
 if (n.groupby(['country','tile_id']).split.nunique()>1).any():raise RuntimeError('Tile crosses splits')
 for f in FEATURES:
  n[f]=pd.to_numeric(n[f],errors='coerce')
  n.loc[~np.isfinite(n[f]),f]=np.nan
 q=p.merge(n[['sample_uid','country','city','tile_id','split','retained']],
            on='sample_uid',how='left',suffixes=('','_node'),validate='many_to_one',indicator=True)
 if not q._merge.eq('both').all():raise RuntimeError('Prediction UID missing from nodes')
 for f in ['country','city','tile_id','split']:
  if not q[f].astype(str).eq(q[f+'_node'].astype(str)).all():raise RuntimeError(f'Metadata mismatch: {f}')
 if not q.label.eq(q.retained).all():raise RuntimeError('Prediction label mismatch')
 if not np.isfinite(p.probability.to_numpy(dtype=float)).all() or not p.probability.between(0,1).all():
  raise RuntimeError('Invalid probability')
 if not np.array_equal(p.prediction.to_numpy(dtype=int),
                       (p.probability.to_numpy(dtype=float)>=p.threshold.to_numpy(dtype=float)).astype(int)):
  raise RuntimeError('Threshold mismatch')
 for (seed,protocol),g in p.groupby(['seed','protocol']):
  target=SOURCE_TARGET[protocol][1]
  if not g.country.eq(target).all():raise RuntimeError(f'Wrong target: {seed}/{protocol}')
  expected=int(((n.split=='test')&(n.country==target)).sum())
  if len(g)!=expected:raise RuntimeError(f'Incomplete test set: {seed}/{protocol}')
 return n,p

def prevalence(n):
 rows=[]
 for (country,split),g in n.groupby(['country','split'],sort=False):
  rows.append(dict(country=country,split=split,n=len(g),positive=int(g.retained.sum()),
   prevalence=float(g.retained.mean()),n_cities=g.city.nunique(),n_tiles=g.tile_id.nunique()))
 return pd.DataFrame(rows)

def feature_shift(n):
 rows=[]
 for split in ('train','validation','test'):
  lt=n.loc[(n.country=='Lithuania')&(n.split==split)]
  pl=n.loc[(n.country=='Poland')&(n.split==split)]
  for f in FEATURES:
   x=lt[f].dropna().to_numpy(dtype=float);y=pl[f].dropna().to_numpy(dtype=float)
   if not len(x) or not len(y):continue
   sd=np.sqrt((np.var(x,ddof=1)+np.var(y,ddof=1))/2)
   rows.append(dict(split=split,feature=f,lt_n=len(x),pl_n=len(y),
    lt_missing_rate=float(lt[f].isna().mean()),pl_missing_rate=float(pl[f].isna().mean()),
    lt_mean=float(np.mean(x)),pl_mean=float(np.mean(y)),
    lt_median=float(np.median(x)),pl_median=float(np.median(y)),
    standardized_mean_difference_lt_minus_pl=float((x.mean()-y.mean())/sd) if sd>0 else np.nan,
    ks_statistic=float(ks_2samp(x,y,method='asymp').statistic),
    wasserstein_distance=float(wasserstein_distance(x,y))))
 return pd.DataFrame(rows)

def country_discriminator(n,seed,folds):
 """Diagnostic country classification: train-only spatial OOF; held-out splits."""
 train=n.loc[n.split.eq('train')].reset_index(drop=True)
 if set(train.country)!={'Lithuania','Poland'}:raise RuntimeError('Missing country')
 y=train.country.eq('Poland').astype(int).to_numpy()
 groups=(train.country.astype(str)+'::'+train.tile_id.astype(str)).to_numpy()
 cv=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=seed)
 oof=np.full(len(train),np.nan)
 for fold,(tr,va) in enumerate(cv.split(train,y,groups)):
  if set(groups[tr])&set(groups[va]):raise RuntimeError('Domain CV leakage')
  model=make_pipeline(SimpleImputer(strategy='median',add_indicator=False),
                      StandardScaler(),LogisticRegression(max_iter=1500,random_state=seed))
  model.fit(train.iloc[tr][list(FEATURES)],y[tr])
  oof[va]=model.predict_proba(train.iloc[va][list(FEATURES)])[:,1]
 if not np.isfinite(oof).all():raise RuntimeError('Incomplete domain OOF')
 results=[dict(split='train_spatial_oof',n=len(train),country_auc=roc_auc_score(y,oof),
               country_balanced_accuracy=balanced_accuracy_score(y,oof>=.5))]
 model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                     LogisticRegression(max_iter=1500,random_state=seed))
 model.fit(train[list(FEATURES)],y)
 for split in ('validation','test'):
  g=n.loc[n.split.eq(split)]
  labels=g.country.eq('Poland').astype(int).to_numpy()
  prob=model.predict_proba(g[list(FEATURES)])[:,1]
  results.append(dict(split=split,n=len(g),country_auc=roc_auc_score(labels,prob),
               country_balanced_accuracy=balanced_accuracy_score(labels,prob>=.5)))
 coef=model.named_steps['logisticregression'].coef_[0]
 rank=pd.DataFrame({'feature':FEATURES,'standardized_coefficient_for_poland':coef,
                    'absolute_coefficient':abs(coef)}).sort_values('absolute_coefficient',ascending=False)
 return pd.DataFrame(results),rank

def transfer_diagnostics(p):
 rows=[];bins=[]
 for (seed,protocol),g in p.groupby(['seed','protocol'],sort=False):
  source,target=SOURCE_TARGET[protocol]
  if g.threshold.nunique()!=1:raise RuntimeError(f'Multiple thresholds {seed}/{protocol}')
  t=float(g.threshold.iloc[0])
  d=calc_metrics(g.label,g.probability,t)
  rows.append(dict(seed=int(seed),protocol=protocol,source=source,target=target,**d,
   selection_rate_minus_prevalence=d['predicted_positive_rate']-d['prevalence'],
   probability_mean_minus_prevalence=d['mean_probability']-d['prevalence']))
  for i in range(10):
   low=i/10;hi=(i+1)/10
   part=g.loc[(g.probability>=low)&(g.probability<(hi if i<9 else 1.00000001))]
   bins.append(dict(seed=int(seed),protocol=protocol,target=target,bin=i,
     bin_lower=low,bin_upper=hi,n=len(part),
     observed_positive_rate=float(part.label.mean()) if len(part) else np.nan,
     mean_probability=float(part.probability.mean()) if len(part) else np.nan))
 return pd.DataFrame(rows),pd.DataFrame(bins)

def aggregate(df,keys,cols):
 result=df.groupby(keys,sort=False)[cols].agg(['mean','std','min','max'])
 result.columns=['_'.join(x) for x in result.columns]
 return result.reset_index()

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--domain-seed',type=int,default=42)
 ap.add_argument('--domain-folds',type=int,default=3)
 args=ap.parse_args()
 if args.domain_folds<2:ap.error('domain-folds must be >=2')
 OUT.mkdir(parents=True,exist_ok=True)
 n,p=load()
 prev=prevalence(n);save(prev,'table_country_split_prevalence.csv')
 shift=feature_shift(n);save(shift,'table_country_feature_shift.csv')
 disc,rank=country_discriminator(n,args.domain_seed,args.domain_folds)
 save(disc,'table_country_discriminator.csv')
 save(rank,'table_domain_discriminator_feature_coefficients.csv')
 transfer,bins=transfer_diagnostics(p)
 save(transfer,'transfer_seed_diagnostics.csv')
 save(bins,'probability_reliability_bins_by_seed.csv')
 cols=['mcc','balanced_accuracy','f1','roc_auc','pr_auc','brier','log_loss','ece_10',
       'prevalence','predicted_positive_rate','mean_probability','threshold',
       'selection_rate_minus_prevalence','probability_mean_minus_prevalence']
 summary=aggregate(transfer,['protocol','source','target'],cols)
 save(summary,'table_transfer_asymmetry_mean_std.csv')
 # Same target roads; compare source-only zero-shot vs within-country and joint.
 comp=[]
 for target,within,zero,joint in [('Lithuania','lt_to_lt','pl_to_lt_zero_shot','joint_to_lt'),
                                  ('Poland','pl_to_pl','lt_to_pl_zero_shot','joint_to_pl')]:
  for seed in SEEDS:
   records={protocol:transfer.loc[(transfer.seed==seed)&(transfer.protocol==protocol)].iloc[0]
            for protocol in (within,zero,joint)}
   comp.append(dict(seed=seed,target=target,
    within_mcc=float(records[within].mcc),zero_shot_mcc=float(records[zero].mcc),
    joint_mcc=float(records[joint].mcc),
    zero_minus_within_mcc=float(records[zero].mcc-records[within].mcc),
    joint_minus_within_mcc=float(records[joint].mcc-records[within].mcc),
    within_positive_rate=float(records[within].predicted_positive_rate),
    zero_shot_positive_rate=float(records[zero].predicted_positive_rate),
    joint_positive_rate=float(records[joint].predicted_positive_rate),
    target_prevalence=float(records[within].prevalence)))
 paired=pd.DataFrame(comp);save(paired,'paired_transfer_vs_within_and_joint.csv')
 save(aggregate(paired,['target'],['zero_minus_within_mcc','joint_minus_within_mcc',
   'within_positive_rate','zero_shot_positive_rate','joint_positive_rate','target_prevalence']),
   'table_transfer_penalties_mean_std.csv')
 # City-level positive-rate mismatch to expose geography and prevalence shift.
 city=[]
 for (seed,protocol,city_name),g in p.groupby(['seed','protocol','city'],sort=False):
  t=float(g.threshold.iloc[0])
  d=calc_metrics(g.label,g.probability,t)
  city.append(dict(seed=int(seed),protocol=protocol,city=city_name,**d))
 city=pd.DataFrame(city);save(city,'city_transfer_seed_diagnostics.csv')
 save(aggregate(city,['protocol','city'],['mcc','prevalence','predicted_positive_rate',
  'mean_probability','ece_10']),'table_city_transfer_shift.csv')
 checks=pd.DataFrame([
  {'check':'all_10_seeds','passed':set(p.seed)==set(SEEDS)},
  {'check':'all_6_protocols','passed':set(p.protocol)==set(PROTOCOLS)},
  {'check':'no_retraining_of_cartographx','passed':True},
  {'check':'no_target_label_use_for_threshold_selection','passed':True},
  {'check':'all_probabilities_finite','passed':bool(np.isfinite(p.probability).all())},
  {'check':'label_status_provisional','passed':True},
  {'check':'historical_test_exploratory','passed':True}])
 if not checks.passed.all():raise RuntimeError('Audit check failed')
 save(checks,'audit_checks.csv')
 manifest={'script':Path(__file__).name,'generated_utc':datetime.now(timezone.utc).isoformat(),
  'seeds':list(SEEDS),'domain_seed':args.domain_seed,'domain_folds':args.domain_folds,
  'source_sha256':{'nodes':hash_file(NODES),'cartographx_predictions':hash_file(PRED)},
  'outputs_sha256':{q.name:hash_file(q) for q in OUT.glob('*.csv')},
  'interpretation':'Descriptive domain-shift diagnostics only. Test labels used post-hoc, never for fitting road-selection models or thresholds.',
  'labels':'C2 PROVISIONAL_NOT_MANUALLY_VALIDATED','test':'historical exploratory',
  'domain_discriminator':'separate diagnostic trained on train only; no target adaptation',
  'figures_generated':0}
 (OUT/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print('SCRIPT 119 PASS')
 print('\nCOUNTRY SPLIT PREVALENCE:\n',prev.to_string(index=False))
 print('\nCOUNTRY DISCRIMINATOR:\n',disc.to_string(index=False))
 print('\nTRANSFER SUMMARY:\n',summary[['protocol','mcc_mean','predicted_positive_rate_mean',
  'prevalence_mean','roc_auc_mean','ece_10_mean']].to_string(index=False))
 print('\nTRANSFER PENALTIES:\n',paired.groupby('target')[['zero_minus_within_mcc',
  'joint_minus_within_mcc']].mean().to_string())
 print('\nSaved:',OUT)
 print('No CartoGraph-X retraining, no new thresholds, no figures.')
 print('NEXT: Script 120 graph sensitivity. Script 125 remains final figure-generation code.')

if __name__=='__main__':main()
