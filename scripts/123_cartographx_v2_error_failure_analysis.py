#!/usr/bin/env python3
"""Script 123: frozen-prediction road-selection error/failure analysis.

Retrospective analysis ONLY: 10 seeds x six protocols, CartoGraph-X and
OOF-selected GNN. Uses historical test predictions for descriptive diagnostics,
never training, threshold selection, model modification or new figures.

Outputs per-seed confusion, city/tile errors, road-feature error strata,
confidence-related errors, paired road disagreements and representative road
IDs for Script 124 map preparation. Does not treat road labels as manually
validated (C2 provisional). Run: python scripts/123_cartographx_v2_error_failure_analysis.py
"""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

ROOT=Path(__file__).resolve().parents[1]
CX=ROOT/'data/processed/cross_country/cartographx_v2_ten_seed_summary_v1/all_seed_selected_test_predictions.csv'
GNN=ROOT/'data/processed/cross_country/benchmark_v2_ten_seed_gnn_baselines_v1'
NODES=ROOT/'data/processed/cross_country/benchmark_v2_unified_labels_v1/road_graph_nodes_v2.csv'
OUT=ROOT/'data/processed/cross_country/cartographx_v2_error_failure_analysis_v1'
SEEDS=set(range(42,52))
PROTOCOLS={'lt_to_lt','pl_to_pl','lt_to_pl_zero_shot','pl_to_lt_zero_shot','joint_to_lt','joint_to_pl'}
TARGET={'lt_to_lt':'Lithuania','pl_to_pl':'Poland','lt_to_pl_zero_shot':'Poland',
        'pl_to_lt_zero_shot':'Lithuania','joint_to_lt':'Lithuania','joint_to_pl':'Poland'}
FEATURES=('length_m','intersecting_road_count','neighbor_count_150m','mean_neighbor_distance_m')
NEEDED={'seed','protocol','sample_uid','country','city','tile_id','split','label','probability','prediction','threshold'}

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
def save(df,name):
 p=OUT/name;tmp=p.with_suffix('.tmp');df.to_csv(tmp,index=False);tmp.replace(p)
def validate(d,model):
 if not NEEDED.issubset(d.columns):raise RuntimeError(f'{model}: missing {NEEDED-set(d.columns)}')
 if set(d.seed)!=SEEDS or set(d.protocol)!=PROTOCOLS:raise RuntimeError(f'{model}: seeds/protocols incomplete')
 if d.duplicated(['seed','protocol','sample_uid']).any():raise RuntimeError(f'{model}: duplicate roads')
 if not d.split.eq('test').all():raise RuntimeError(f'{model}: non-test rows')
 if not d.label.isin([0,1]).all() or not d.prediction.isin([0,1]).all():raise RuntimeError(f'{model}: invalid binary data')
 if not np.isfinite(d.probability).all() or not d.probability.between(0,1).all():raise RuntimeError(f'{model}: invalid probability')
 if not np.allclose(d.prediction.to_numpy(),(d.probability>=d.threshold).astype(int).to_numpy()):
  raise RuntimeError(f'{model}: prediction does not match frozen threshold')
 for (seed,protocol),g in d.groupby(['seed','protocol']):
  if not g.country.eq(TARGET[protocol]).all() or g.threshold.nunique()!=1:raise RuntimeError('Target or threshold inconsistency')
  if g.label.nunique()!=2:raise RuntimeError('Missing class')
 return d

def load_gnn():
 sel_path=GNN/'all_seed_selected_protocol_results.csv'
 if not sel_path.exists():return None
 sel=pd.read_csv(sel_path)
 frames=[]
 for seed in sorted(SEEDS):
  f=GNN/f'seed_{seed}'/'all_candidate_test_predictions.csv'
  if not f.exists():raise FileNotFoundError(f)
  d=pd.read_csv(f,low_memory=False)
  d=d.merge(sel.loc[sel.seed.eq(seed),['seed','protocol','model','variant']],
            on=['seed','protocol','model','variant'],how='inner',validate='many_to_one')
  frames.append(d)
 return validate(pd.concat(frames,ignore_index=True),'GNN OOF-selected')

def confusion(g,extra=None):
 y=g.label.to_numpy(dtype=int);p=g.prediction.to_numpy(dtype=int)
 tp=int(((y==1)&(p==1)).sum());tn=int(((y==0)&(p==0)).sum())
 fp=int(((y==0)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum())
 return dict(n=len(g),positives=int(y.sum()),negatives=int((1-y).sum()),tp=tp,tn=tn,fp=fp,fn=fn,
  accuracy=float((tp+tn)/len(y)),mcc=float(matthews_corrcoef(y,p)) if len(np.unique(y))==2 else np.nan,
  balanced_accuracy=float(balanced_accuracy_score(y,p)) if len(np.unique(y))==2 else np.nan,
  precision=tp/(tp+fp) if tp+fp else np.nan,recall=tp/(tp+fn) if tp+fn else np.nan,
  specificity=tn/(tn+fp) if tn+fp else np.nan,
  fpr=fp/(fp+tn) if fp+tn else np.nan,fnr=fn/(fn+tp) if fn+tp else np.nan,
  predicted_positive_rate=float(p.mean()),prevalence=float(y.mean()))

def annotate(df):
 d=df.copy()
 d['error_type']=np.select([d.label.eq(0)&d.prediction.eq(1),d.label.eq(1)&d.prediction.eq(0),
   d.label.eq(1)&d.prediction.eq(1)],['FP','FN','TP'],default='TN')
 d['is_error']=d.label.ne(d.prediction)
 d['probability_confidence']=np.maximum(d.probability,1-d.probability)
 d['decision_margin']=abs(d.probability-d.threshold)
 d['confident_wrong_80']=d.is_error & ((d.probability>=.8)|(d.probability<=.2))
 d['high_margin_wrong_20']=d.is_error & (d.decision_margin>=.20)
 return d

def feature_strata(d):
 # Bins frozen from the country-specific historical TEST road feature distribution.
 # Purely descriptive; no thresholds, model parameters or prediction values are altered.
 rows=[]
 for country,feature in [(c,f) for c in sorted(d.country.unique()) for f in FEATURES]:
  vals=d.loc[d.country.eq(country),['sample_uid',feature]].drop_duplicates('sample_uid')
  vals[feature]=pd.to_numeric(vals[feature],errors='coerce')
  finite=vals[feature].replace([np.inf,-np.inf],np.nan).dropna()
  if len(finite)<4:continue
  q=np.quantile(finite,[0,.25,.50,.75,1.]);unique=np.unique(q)
  if len(unique)<2:continue
  # rank-based quartiles tolerate tied integer-valued attributes.
  ranks=finite.rank(method='average',pct=True)
  vals['feature_bin']='missing'
  vals.loc[finite.index,'feature_bin']=np.select([ranks<=.25,ranks<=.50,ranks<=.75],
   ['Q1','Q2','Q3'],default='Q4')
  merged=d.loc[d.country.eq(country)].merge(vals[['sample_uid','feature_bin']],on='sample_uid',validate='many_to_one')
  for (model,seed,protocol,bin_name),g in merged.groupby(['model_family','seed','protocol','feature_bin'],sort=False):
   cm=confusion(g)
   rows.append(dict(model_family=model,seed=seed,protocol=protocol,country=country,
    feature=feature,feature_bin=bin_name,**cm))
 return pd.DataFrame(rows)

def paired_disagreements(cx,gnn):
 keys=['seed','protocol','sample_uid']
 a=cx[keys+['country','city','tile_id','label','prediction','probability','threshold']]
 b=gnn[keys+['country','city','tile_id','label','prediction','probability','threshold']]
 j=a.merge(b,on=keys,how='outer',suffixes=('_cx','_gnn'),indicator=True,validate='one_to_one')
 if not j._merge.eq('both').all():raise RuntimeError('CX and GNN sample IDs differ')
 for col in ('country','city','tile_id','label'):
  if not j[f'{col}_cx'].eq(j[f'{col}_gnn']).all():raise RuntimeError(f'CX/GNN mismatch: {col}')
 j['cx_error']=j.prediction_cx.ne(j.label_cx)
 j['gnn_error']=j.prediction_gnn.ne(j.label_gnn)
 j['outcome']=np.select([j.cx_error&~j.gnn_error,~j.cx_error&j.gnn_error,j.cx_error&j.gnn_error],
   ['GNN_correct_CX_wrong','CX_correct_GNN_wrong','both_wrong'],default='both_correct')
 result=j.groupby(['seed','protocol','city_cx','outcome'],sort=False).size().rename('n_roads').reset_index()
 result=result.rename(columns={'city_cx':'city'})
 return result,j

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--no-gnn',action='store_true',help='Skip optional GNN comparison')
 ap.add_argument('--examples-per-class-city',type=int,default=12)
 args=ap.parse_args()
 if args.examples_per_class_city<1:ap.error('Example count must be positive')
 for f in (CX,NODES):
  if not f.exists():raise FileNotFoundError(f)
 OUT.mkdir(parents=True,exist_ok=True)
 cx=validate(pd.read_csv(CX,low_memory=False),'CartoGraph-X')
 models=[('CartoGraph-X',cx)]
 if not args.no_gnn:
  gnn=load_gnn()
  if gnn is not None:models.append(('GNN OOF-selected',gnn))
 nodes=pd.read_csv(NODES,low_memory=False)
 if nodes.sample_uid.duplicated().any():raise RuntimeError('Duplicate graph node UIDs')
 if not set(FEATURES).issubset(nodes.columns):raise RuntimeError('Road features missing')
 nodes=nodes.loc[nodes.split.eq('test'),['sample_uid','country','city','tile_id','retained',*FEATURES]].copy()
 frames=[]
 for model,pred in models:
  j=pred.merge(nodes,on='sample_uid',how='left',suffixes=('','_node'),validate='many_to_one',indicator=True)
  if not j._merge.eq('both').all():raise RuntimeError('Predictions lack graph node matches')
  for field in ('country','city','tile_id'):
   if not j[field].astype(str).eq(j[field+'_node'].astype(str)).all():raise RuntimeError(f'Metadata mismatch: {field}')
  if not j.label.eq(j.retained).all():raise RuntimeError('C2 label mismatch')
  j=j.drop(columns=['_merge','country_node','city_node','tile_id_node','retained'])
  j['model_family']=model
  frames.append(annotate(j))
 d=pd.concat(frames,ignore_index=True)
 expected=10*sum([1924 if TARGET[p]=='Lithuania' else 2552 for p in PROTOCOLS])
 if len(cx)!=expected:raise RuntimeError(f'Unexpected CartoGraph-X prediction count {len(cx)} vs {expected}')
 cm=[];city=[];tiles=[];confidence=[];examples=[]
 for (model,seed,protocol),g in d.groupby(['model_family','seed','protocol'],sort=False):
  cm.append(dict(model_family=model,seed=seed,protocol=protocol,country=TARGET[protocol],**confusion(g)))
  for city_name,part in g.groupby('city',sort=False):
   city.append(dict(model_family=model,seed=seed,protocol=protocol,city=city_name,**confusion(part)))
   for etype in ('FP','FN','TP','TN'):
    candidates=part.loc[part.error_type.eq(etype)].copy()
    if etype in ('FP','FN'):
     candidates=candidates.sort_values(['decision_margin','sample_uid'],ascending=[False,True])
    else:candidates=candidates.sort_values(['decision_margin','sample_uid'],ascending=[True,True])
    for _,r in candidates.head(args.examples_per_class_city).iterrows():
     examples.append(dict(model_family=model,seed=seed,protocol=protocol,country=r.country,
      city=city_name,tile_id=r.tile_id,sample_uid=r.sample_uid,error_type=etype,
      label=int(r.label),prediction=int(r.prediction),probability=float(r.probability),
      threshold=float(r.threshold),decision_margin=float(r.decision_margin),
      **{f:r[f] for f in FEATURES}))
  for tile,part in g.groupby(['city','tile_id'],sort=False):
   city_name,tile_id=tile
   tiles.append(dict(model_family=model,seed=seed,protocol=protocol,city=city_name,tile_id=tile_id,**confusion(part)))
  for low,high in ((0,.1),(.1,.2),(.2,.3),(.3,.4),(.4,.5),(.5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,1.00000001)):
   part=g.loc[g.probability.ge(low)&g.probability.lt(high)]
   if part.empty:continue
   confidence.append(dict(model_family=model,seed=seed,protocol=protocol,
    prob_bin_lower=low,prob_bin_upper=min(high,1.0),n=len(part),
    observed_positive_rate=float(part.label.mean()),predicted_positive_rate=float(part.prediction.mean()),
    error_rate=float(part.is_error.mean()),fp=int(part.error_type.eq('FP').sum()),
    fn=int(part.error_type.eq('FN').sum()),mean_probability=float(part.probability.mean())))
 cm=pd.DataFrame(cm);city=pd.DataFrame(city);tiles=pd.DataFrame(tiles)
 save(cm,'seed_protocol_confusion.csv')
 save(city,'seed_city_confusion.csv')
 save(tiles,'seed_tile_confusion.csv')
 save(pd.DataFrame(confidence),'probability_bin_error_profiles.csv')
 save(pd.DataFrame(examples),'road_failure_example_candidates.csv')
 strata=feature_strata(d)
 save(strata,'road_feature_quartile_error_strata.csv')
 metrics=['mcc','balanced_accuracy','precision','recall','specificity','fpr','fnr','predicted_positive_rate']
 for frame,keys,name in ((cm,['model_family','protocol'],'table_protocol_errors_mean_std.csv'),
                         (city,['model_family','protocol','city'],'table_city_errors_mean_std.csv'),
                         (strata,['model_family','protocol','country','feature','feature_bin'],'table_feature_strata_mean_std.csv')):
  summary=frame.groupby(keys,sort=False)[metrics].agg(['mean','std'])
  summary.columns=['_'.join(t) for t in summary.columns]
  save(summary.reset_index(),name)
 if len(models)==2:
  comparison,matched=paired_disagreements(cx,models[1][1])
  save(comparison,'paired_road_outcome_by_seed_city.csv')
  summary=comparison.pivot_table(index=['seed','protocol','city'],columns='outcome',values='n_roads',fill_value=0).reset_index()
  for col in ('CX_correct_GNN_wrong','GNN_correct_CX_wrong','both_wrong','both_correct'):
   if col not in summary.columns:summary[col]=0
  summary['net_CX_correct_advantage']=summary.CX_correct_GNN_wrong-summary.GNN_correct_CX_wrong
  save(summary,'table_paired_error_disagreement.csv')
  # Store compact per-road matched outcomes to allow map candidate screening.
  compact=matched[['seed','protocol','sample_uid','country_cx','city_cx','tile_id_cx','label_cx',
                   'probability_cx','probability_gnn','prediction_cx','prediction_gnn','outcome']].copy()
  compact.columns=['seed','protocol','sample_uid','country','city','tile_id','label',
                   'cartographx_probability','gnn_probability','cartographx_prediction','gnn_prediction','outcome']
  save(compact,'paired_road_error_outcomes.csv')
 manifest={'script':Path(__file__).name,'utc':datetime.now(timezone.utc).isoformat(),
  'models':[x[0] for x in models],'seeds':sorted(SEEDS),'protocols':sorted(PROTOCOLS),
  'historical_test_is_exploratory':True,'c2_labels_provisional':True,
  'retraining':False,'threshold_tuning':False,'figures_generated':0,
  'interpretation':'road features and error strata are associations, not causal failure explanations',
  'input_sha256':{'cx':digest(CX),'nodes':digest(NODES)},
  'output_sha256':{f.name:digest(f) for f in OUT.glob('*.csv')}}
 (OUT/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print('SCRIPT 123 PASS')
 print('Model families:',', '.join(manifest['models']))
 print('Road-prediction records:',len(d))
 print('\nMEAN ERROR METRICS:')
 print(cm.groupby(['model_family','protocol'],sort=False)[['mcc','fpr','fnr','recall']].mean().round(4).to_string())
 print('\nSaved:',OUT)
 print('No retraining, no threshold tuning, no figures.')
 print('NEXT: Script 124 map-example preparation; Script 125 is final figure-generation code.')

if __name__=='__main__':main()
