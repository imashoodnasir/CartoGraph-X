#!/usr/bin/env python3
"""Script 122: Retrospective probability calibration and reliability audit.

Reads frozen ten-seed CartoGraph-X predictions; optionally includes selected
GNN baseline predictions. NO retraining, recalibration, threshold search, or
new figures. Test labels are used solely for descriptive evaluation.

Run: python scripts/122_cartographx_v2_calibration_reliability.py
"""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[1]
CX=ROOT/'data/processed/cross_country/cartographx_v2_ten_seed_summary_v1/all_seed_selected_test_predictions.csv'
GNN=ROOT/'data/processed/cross_country/benchmark_v2_ten_seed_gnn_baselines_v1'
OUT=ROOT/'data/processed/cross_country/cartographx_v2_calibration_reliability_v1'
SEEDS=set(range(42,52))
PROTOCOLS={'lt_to_lt','pl_to_pl','lt_to_pl_zero_shot','pl_to_lt_zero_shot','joint_to_lt','joint_to_pl'}
TARGET={'lt_to_lt':'Lithuania','pl_to_pl':'Poland','lt_to_pl_zero_shot':'Poland',
        'pl_to_lt_zero_shot':'Lithuania','joint_to_lt':'Lithuania','joint_to_pl':'Poland'}

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(df,name):
 tmp=OUT/(name+'.tmp');df.to_csv(tmp,index=False);tmp.replace(OUT/name)
def check(df,model):
 needed={'seed','protocol','sample_uid','country','city','tile_id','split','label','probability','prediction','threshold'}
 if not needed.issubset(df.columns):raise RuntimeError(f'{model} missing {needed-set(df.columns)}')
 if set(df.seed)!=SEEDS or set(df.protocol)!=PROTOCOLS:raise RuntimeError(f'{model} missing seeds/protocols')
 if df.duplicated(['seed','protocol','sample_uid']).any():raise RuntimeError(f'{model} duplicate predictions')
 if not df.split.eq('test').all():raise RuntimeError('Non-test predictions found')
 if not df.label.isin([0,1]).all():raise RuntimeError('Invalid label')
 if not np.isfinite(df.probability).all() or not df.probability.between(0,1).all():raise RuntimeError('Invalid probability')
 if not np.array_equal(df.prediction.to_numpy(dtype=int),
                       (df.probability.to_numpy()>=df.threshold.to_numpy()).astype(int)):
  raise RuntimeError(f'{model} threshold/prediction mismatch')
 for (seed,protocol),g in df.groupby(['seed','protocol']):
  if not g.country.eq(TARGET[protocol]).all():raise RuntimeError('Incorrect target country')
  if g.threshold.nunique()!=1:raise RuntimeError('Inconsistent threshold')
  if g.label.nunique()!=2:raise RuntimeError('Single-class test partition')
 return df

def load_gnn():
 selected=GNN/'all_seed_selected_protocol_results.csv'
 if not selected.is_file():return None
 sel=pd.read_csv(selected,low_memory=False)
 if not {'seed','protocol','model','variant'}.issubset(sel.columns):raise RuntimeError('Selected GNN schema changed')
 frames=[]
 for seed in sorted(SEEDS):
  path=GNN/f'seed_{seed}'/'all_candidate_test_predictions.csv'
  if not path.is_file():raise FileNotFoundError(path)
  d=pd.read_csv(path,low_memory=False)
  d=d.merge(sel.loc[sel.seed.eq(seed),['seed','protocol','model','variant']],
            on=['seed','protocol','model','variant'],how='inner',validate='many_to_one')
  frames.append(d)
 return check(pd.concat(frames,ignore_index=True),'GNN')

def ece(y,p,bins=10,strategy='equal_width'):
 if strategy=='equal_width':
  edges=np.linspace(0,1,bins+1)
 elif strategy=='equal_frequency':
  edges=np.quantile(p,np.linspace(0,1,bins+1))
  edges[0]=0;edges[-1]=1
 else:raise ValueError(strategy)
 groups=np.searchsorted(edges[1:-1],p,side='right')
 rows=[];total=0.
 for j in range(bins):
  mask=groups==j;n=int(mask.sum())
  if not n:
   rows.append(dict(bin=j,bin_lower=float(edges[j]),bin_upper=float(edges[j+1]),
                    n=0,mean_confidence=np.nan,observed_positive_rate=np.nan,abs_gap=np.nan))
   continue
  confidence=float(p[mask].mean());rate=float(y[mask].mean());gap=abs(confidence-rate)
  total+=n/len(y)*gap
  rows.append(dict(bin=j,bin_lower=float(edges[j]),bin_upper=float(edges[j+1]),
                   n=n,mean_confidence=confidence,observed_positive_rate=rate,abs_gap=gap))
 return float(total),rows

def metrics(g,model):
 y=g.label.to_numpy(dtype=int);p=g.probability.to_numpy(dtype=float)
 eps=np.clip(p,1e-7,1-1e-7)
 e10,_=ece(y,p,10);e15,_=ece(y,p,15);eq,_=ece(y,p,10,'equal_frequency')
 pos=p[y==1];neg=p[y==0]
 return dict(model_family=model,seed=int(g.seed.iloc[0]),protocol=g.protocol.iloc[0],
  target=g.country.iloc[0],n=len(g),prevalence=float(y.mean()),
  predicted_positive_rate=float(g.prediction.mean()),mean_probability=float(p.mean()),
  brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,eps,labels=[0,1])),
  ece10=e10,ece15=e15,ece10_equal_frequency=eq,
  roc_auc=float(roc_auc_score(y,p)),pr_auc=float(average_precision_score(y,p)),
  positive_mean_probability=float(pos.mean()),negative_mean_probability=float(neg.mean()),
  probability_bias=float(p.mean()-y.mean()),
  confident_error_rate_80=float(np.mean(((p>=.8)&(y==0))|((p<=.2)&(y==1)))),
  confident_coverage_80=float(np.mean((p>=.8)|(p<=.2))))

def risk_coverage(g,model):
 # Confidence relative to a probability of 0.5; classification uses frozen threshold.
 p=g.probability.to_numpy(dtype=float);y=g.label.to_numpy(dtype=int)
 pred=g.prediction.to_numpy(dtype=int)
 confidence=np.maximum(p,1-p)
 order=np.argsort(-confidence,kind='stable')
 rows=[]
 for coverage in (.10,.25,.50,.75,1.):
  m=max(1,int(round(len(y)*coverage)));idx=order[:m]
  rows.append(dict(model_family=model,seed=int(g.seed.iloc[0]),protocol=g.protocol.iloc[0],
   coverage_target=coverage,n_selected=m,realized_coverage=m/len(y),
   selective_error=float(np.mean(pred[idx]!=y[idx])),
   mean_probability_confidence=float(np.mean(confidence[idx]))))
 return rows

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--no-gnn',action='store_true',help='Skip optional selected-GNN comparison')
 args=ap.parse_args()
 if not CX.is_file():raise FileNotFoundError(CX)
 OUT.mkdir(parents=True,exist_ok=True)
 cx=check(pd.read_csv(CX,low_memory=False),'CartoGraph-X')
 sources=[('CartoGraph-X',cx)]
 if not args.no_gnn:
  gnn=load_gnn()
  if gnn is not None:
   # Ensure the same road labels and IDs before comparing model families.
   a=cx[['seed','protocol','sample_uid','label','country']]
   b=gnn[['seed','protocol','sample_uid','label','country']]
   joined=a.merge(b,on=['seed','protocol','sample_uid'],how='outer',suffixes=('_cx','_gnn'),indicator=True)
   if not joined._merge.eq('both').all() or not joined.label_cx.eq(joined.label_gnn).all() or not joined.country_cx.eq(joined.country_gnn).all():
    raise RuntimeError('GNN/CX test samples or labels differ')
   sources.append(('GNN OOF-selected',gnn))
 rows=[];reliability=[];risk=[];city=[]
 for model,df in sources:
  for (seed,protocol),g in df.groupby(['seed','protocol'],sort=False):
   rows.append(metrics(g,model))
   risk.extend(risk_coverage(g,model))
   y=g.label.to_numpy(dtype=int);p=g.probability.to_numpy(dtype=float)
   for strategy in ('equal_width','equal_frequency'):
    _,bins=ece(y,p,10,strategy)
    for row in bins:reliability.append(dict(model_family=model,seed=seed,protocol=protocol,
     binning=strategy,**row))
   for city_name,part in g.groupby('city',sort=False):
    # City calibration; small bins can be noisy and should be interpreted cautiously.
    yy=part.label.to_numpy(dtype=int);pp=part.probability.to_numpy(dtype=float)
    city.append(dict(model_family=model,seed=seed,protocol=protocol,city=city_name,
     n=len(part),prevalence=float(yy.mean()),mean_probability=float(pp.mean()),
     brier=float(brier_score_loss(yy,pp)),ece10=ece(yy,pp,10)[0]))
 scores=pd.DataFrame(rows);save(scores,'seed_protocol_calibration_metrics.csv')
 bins=pd.DataFrame(reliability);save(bins,'reliability_bins_by_seed.csv')
 risk=pd.DataFrame(risk);save(risk,'selective_risk_coverage_by_seed.csv')
 city=pd.DataFrame(city);save(city,'city_calibration_by_seed.csv')
 metric_cols=['brier','log_loss','ece10','ece15','ece10_equal_frequency','roc_auc','pr_auc',
              'probability_bias','confident_error_rate_80','confident_coverage_80']
 summ=scores.groupby(['model_family','protocol'],sort=False)[metric_cols].agg(['mean','std','min','max'])
 summ.columns=['_'.join(t) for t in summ.columns]
 save(summ.reset_index(),'table_calibration_mean_std.csv')
 selective=risk.groupby(['model_family','protocol','coverage_target'],sort=False)[
  ['selective_error','mean_probability_confidence']].agg(['mean','std'])
 selective.columns=['_'.join(t) for t in selective.columns]
 save(selective.reset_index(),'table_selective_risk_mean_std.csv')
 citysum=city.groupby(['model_family','protocol','city'],sort=False)[['brier','ece10']].agg(['mean','std'])
 citysum.columns=['_'.join(t) for t in citysum.columns]
 save(citysum.reset_index(),'table_city_calibration_mean_std.csv')
 if len(sources)==2:
  a=scores.loc[scores.model_family.eq('CartoGraph-X')]
  b=scores.loc[scores.model_family.eq('GNN OOF-selected')]
  paired=a.merge(b,on=['seed','protocol'],validate='one_to_one',suffixes=('_cx','_gnn'))
  for metric in ('brier','log_loss','ece10'):
   paired[f'{metric}_cx_minus_gnn']=paired[f'{metric}_cx']-paired[f'{metric}_gnn']
  save(paired,'paired_seed_calibration_vs_gnn.csv')
  cols=[f'{m}_cx_minus_gnn' for m in ('brier','log_loss','ece10')]
  table=paired.groupby('protocol',sort=False)[cols].agg(['mean','std'])
  table.columns=['_'.join(t) for t in table.columns]
  save(table.reset_index(),'table_calibration_vs_gnn.csv')
 manifest={'script':Path(__file__).name,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
  'models':[m for m,_ in sources],'seeds':sorted(SEEDS),'protocols':sorted(PROTOCOLS),
  'input_sha256':{'cartographx':sha(CX),**({'gnn_selected':sha(GNN/'all_seed_selected_protocol_results.csv')} if len(sources)==2 else {})},
  'output_sha256':{f.name:sha(f) for f in OUT.glob('*.csv')},
  'calibration_method':'descriptive only; no calibrator trained, no probability changes',
  'reliability_bins':'10 equal-width and 10 equal-frequency per seed/protocol',
  'selective_risk':'confidence=max(p,1-p), frozen decision threshold; descriptive',
  'historical_test':'EXPLORATORY_NOT_UNTOUCHED_HOLDOUT',
  'labels':'C2 PROVISIONAL_NOT_MANUALLY_VALIDATED','figures_generated':0}
 (OUT/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print('SCRIPT 122 PASS')
 print('Model families:',', '.join(m for m,_ in sources))
 print('\nMEAN CALIBRATION METRICS (lower Brier/ECE is better):')
 print(scores.groupby(['model_family','protocol'],sort=False)[['brier','ece10','log_loss']].mean().round(4).to_string())
 print('\nSaved:',OUT)
 print('No calibrator fitting, no threshold tuning, no figures.')
 print('NEXT: Script 123 error analysis. Script 125 is final figure-generation code.')

if __name__=='__main__':main()
