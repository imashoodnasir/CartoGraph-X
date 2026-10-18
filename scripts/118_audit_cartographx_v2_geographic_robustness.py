#!/usr/bin/env python3
"""Script 118: geographic robustness of frozen CartoGraph-X and GNN predictions.

No training, model selection, thresholds, or new figures. Historical test only:
exploratory, NOT an untouched geographic holdout. Repeated seeds share the
same roads, so tile uncertainty is computed over tiles, NOT 10x replicated roads.

Usage:
 python scripts/118_audit_cartographx_v2_geographic_robustness.py
 python scripts/118_audit_cartographx_v2_geographic_robustness.py --bootstrap 2000
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.metrics import (matthews_corrcoef, balanced_accuracy_score,
                             f1_score, roc_auc_score, average_precision_score)

ROOT=Path(__file__).resolve().parents[1]
CX=ROOT/'data/processed/cross_country/cartographx_v2_ten_seed_summary_v1/all_seed_selected_test_predictions.csv'
GNN=ROOT/'data/processed/cross_country/benchmark_v2_ten_seed_gnn_baselines_v1'
GNNS=GNN/'all_seed_selected_protocol_results.csv'
NODES=ROOT/'data/processed/cross_country/benchmark_v2_unified_labels_v1/road_graph_nodes_v2.csv'
OUT=ROOT/'data/processed/cross_country/cartographx_v2_geographic_robustness_v1'
PROTOCOLS=('lt_to_lt','pl_to_pl','lt_to_pl_zero_shot','pl_to_lt_zero_shot','joint_to_lt','joint_to_pl')
SEEDS=tuple(range(42,52))
FIELDS=['mcc','balanced_accuracy','f1','roc_auc','pr_auc','n','positives','positive_rate','predicted_positive_rate','mean_probability']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def save(df,name):
 path=OUT/name;tmp=path.with_suffix('.tmp');df.to_csv(tmp,index=False);tmp.replace(path)
def metrics(g):
 y=g.label.to_numpy(dtype=int);pred=g.prediction.to_numpy(dtype=int);p=g.probability.to_numpy(dtype=float)
 if not len(y):raise ValueError('Empty group')
 if not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('Invalid probabilities')
 both=len(np.unique(y))==2
 return dict(n=len(y),positives=int(y.sum()),positive_rate=float(y.mean()),
   predicted_positive_rate=float(pred.mean()),mean_probability=float(p.mean()),
   mcc=float(matthews_corrcoef(y,pred)) if both else np.nan,
   balanced_accuracy=float(balanced_accuracy_score(y,pred)) if both else np.nan,
   f1=float(f1_score(y,pred,zero_division=0)) if both else np.nan,
   roc_auc=float(roc_auc_score(y,p)) if both else np.nan,
   pr_auc=float(average_precision_score(y,p)) if both else np.nan,
   has_both_classes=bool(both))
def summarize(df,group_cols):
 metrics_cols=['mcc','balanced_accuracy','f1','roc_auc','pr_auc','positive_rate','predicted_positive_rate']
 rows=[]
 for key,g in df.groupby(group_cols,dropna=False,sort=False):
  if not isinstance(key,tuple):key=(key,)
  row=dict(zip(group_cols,key));row['n_seeds']=int(g.seed.nunique())
  row['n_roads_per_seed']=int(g.n.iloc[0])
  row['n_valid_mcc_seeds']=int(g.mcc.notna().sum())
  for col in metrics_cols:
   vals=g[col].dropna()
   row[col+'_mean']=float(vals.mean()) if len(vals) else np.nan
   row[col+'_std']=float(vals.std(ddof=1)) if len(vals)>1 else np.nan
   row[col+'_min']=float(vals.min()) if len(vals) else np.nan
   row[col+'_max']=float(vals.max()) if len(vals) else np.nan
  rows.append(row)
 return pd.DataFrame(rows)
def load():
 if not CX.is_file() or not GNNS.is_file() or not NODES.is_file():
  raise FileNotFoundError(f'Required files:\n{CX}\n{GNNS}\n{NODES}')
 a=pd.read_csv(CX,low_memory=False)
 required={'seed','protocol','sample_uid','country','city','tile_id','split','label','probability','prediction','threshold'}
 if not required.issubset(a.columns):raise RuntimeError(f'CartoGraph-X missing {required-set(a.columns)}')
 a['model_family']='CartoGraph-X'
 bres=pd.read_csv(GNNS,low_memory=False)
 required_sel={'seed','protocol','model','variant','threshold'}
 if not required_sel.issubset(bres.columns):raise RuntimeError(f'GNN selections missing {required_sel-set(bres.columns)}')
 bframes=[]
 for seed in SEEDS:
  path=GNN/f'seed_{seed}'/'all_candidate_test_predictions.csv'
  if not path.is_file():raise FileNotFoundError(path)
  p=pd.read_csv(path,low_memory=False)
  sel=bres.loc[bres.seed.eq(seed),['seed','protocol','model','variant','threshold']]
  if len(sel)!=6 or sel.duplicated(['seed','protocol']).any():raise RuntimeError(f'Invalid GNN selection seed {seed}')
  p=p.merge(sel,on=['seed','protocol','model','variant'],suffixes=('','_selected'),how='inner',validate='many_to_one')
  if 'threshold_selected' in p and not np.allclose(p.threshold,p.threshold_selected,atol=1e-9):
   raise RuntimeError('GNN threshold selection mismatch')
  p['model_family']='GNN OOF-selected'
  bframes.append(p)
 b=pd.concat(bframes,ignore_index=True)
 needed=['seed','protocol','sample_uid','country','city','tile_id','split','label','probability','prediction','threshold','model_family']
 a=a[needed].copy();b=b[needed].copy()
 combined=pd.concat([a,b],ignore_index=True)
 if set(combined.seed)!=set(SEEDS) or set(combined.protocol)!=set(PROTOCOLS):raise RuntimeError('Missing seed/protocol')
 if combined.duplicated(['model_family','seed','protocol','sample_uid']).any():raise RuntimeError('Duplicate road predictions')
 if not combined.split.eq('test').all():raise RuntimeError('Non-test predictions')
 if not np.array_equal(combined.prediction.to_numpy(dtype=int),
                       (combined.probability.to_numpy(dtype=float)>=combined.threshold.to_numpy(dtype=float)).astype(int)):
  raise RuntimeError('Predictions inconsistent with thresholds')
 nodes=pd.read_csv(NODES,usecols=['sample_uid','country','city','split','tile_id','retained'],low_memory=False)
 if nodes.sample_uid.duplicated().any():raise RuntimeError('Duplicate source node IDs')
 joined=combined.merge(nodes,on='sample_uid',how='left',suffixes=('','_node'),validate='many_to_one',indicator=True)
 if not joined._merge.eq('both').all():raise RuntimeError('Predictions missing source node')
 for k in ['country','city','split','tile_id']:
  if not joined[k].astype(str).eq(joined[k+'_node'].astype(str)).all():raise RuntimeError(f'Metadata mismatch: {k}')
 if not joined.label.eq(joined.retained).all():raise RuntimeError('Labels differ from frozen V2')
 # Verify both models evaluate identical sample IDs and labels for each seed/protocol.
 keys=['seed','protocol','sample_uid']
 pivot=a[keys+['label']].merge(b[keys+['label']],on=keys,how='outer',suffixes=('_cx','_gnn'),indicator=True,validate='one_to_one')
 if not pivot._merge.eq('both').all() or not pivot.label_cx.eq(pivot.label_gnn).all():
  raise RuntimeError('Unmatched CartoGraph-X/GNN evaluation sets')
 return combined

def group_metrics(data,cols):
 rows=[]
 for key,g in data.groupby(cols,sort=False,dropna=False):
  if not isinstance(key,tuple):key=(key,)
  rows.append({**dict(zip(cols,key)),**metrics(g)})
 return pd.DataFrame(rows)

def tile_cluster_ci(data,boot,seed):
 """Hierarchical summary: average model probabilities across seeds for each road;
 then resample spatial tiles. This is descriptive, not an independent holdout."""
 rng=np.random.default_rng(seed)
 rows=[]
 for (family,protocol,country),g in data.groupby(['model_family','protocol','country'],sort=False):
  # Each road appears exactly once per seed, and true label is identical.
  agg=g.groupby(['sample_uid','tile_id'],as_index=False).agg(
   label=('label','first'),probability=('probability','mean'),
   predicted_fraction=('prediction','mean'))
  agg['prediction']=(agg.predicted_fraction>=.5).astype(int)
  tiles=list(agg.tile_id.unique())
  if len(tiles)<2:
   rows.append(dict(model_family=family,protocol=protocol,country=country,n_tiles=len(tiles),
                    bootstrap_valid=0,mcc_ci_lower=np.nan,mcc_ci_upper=np.nan,
                    bootstrap_note='Too few tiles'))
   continue
  blocks=[agg.loc[agg.tile_id.eq(t)] for t in tiles]
  scores=[]
  for _ in range(boot):
   idx=rng.integers(0,len(blocks),len(blocks))
   y=np.concatenate([blocks[j].label.to_numpy(dtype=int) for j in idx])
   yp=np.concatenate([blocks[j].prediction.to_numpy(dtype=int) for j in idx])
   if len(np.unique(y))==2:scores.append(matthews_corrcoef(y,yp))
  rows.append(dict(model_family=family,protocol=protocol,country=country,n_tiles=len(tiles),
     bootstrap_valid=len(scores),
     mcc_ci_lower=float(np.percentile(scores,2.5)) if scores else np.nan,
     mcc_ci_upper=float(np.percentile(scores,97.5)) if scores else np.nan,
     bootstrap_note='Descriptive tile-resampling interval; mean-seed road decisions, not independent geographic validation'))
 return pd.DataFrame(rows)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--bootstrap',type=int,default=1000,help='Spatial tile bootstrap replicates')
 ap.add_argument('--seed',type=int,default=20261014,help='Bootstrap RNG seed, not model seed')
 args=ap.parse_args()
 if args.bootstrap<100:ap.error('--bootstrap must be >=100')
 OUT.mkdir(parents=True,exist_ok=True)
 data=load()
 print(f'Validated {len(data):,} prediction rows, {data.sample_uid.nunique():,} unique test roads')
 city=group_metrics(data,['model_family','seed','protocol','country','city'])
 tile=group_metrics(data,['model_family','seed','protocol','country','city','tile_id'])
 overall=group_metrics(data,['model_family','seed','protocol','country'])
 if city.n.lt(1).any() or tile.n.lt(1).any():raise RuntimeError('Empty groups')
 for col in ['model_family','seed','protocol']:
  if city[col].isna().any():raise RuntimeError('Invalid city metric metadata')
 # A city with one class has undefined MCC/AUC; preserve NA, do not set to zero.
 save(city,'city_seed_metrics.csv')
 save(tile,'tile_seed_metrics.csv')
 save(overall,'country_protocol_seed_metrics.csv')
 city_summary=summarize(city,['model_family','protocol','country','city'])
 tile_summary=summarize(tile,['model_family','protocol','country','city','tile_id'])
 save(city_summary,'table_city_robustness_mean_std.csv')
 save(tile_summary,'table_tile_robustness_mean_std.csv')
 save(summarize(overall,['model_family','protocol','country']),'table_protocol_robustness_mean_std.csv')
 # Matched seed-city differences: same roads for both models.
 cx=city.loc[city.model_family.eq('CartoGraph-X')].drop(columns='model_family')
 gn=city.loc[city.model_family.eq('GNN OOF-selected')].drop(columns='model_family')
 join_keys=['seed','protocol','country','city']
 paired=cx.merge(gn,on=join_keys,how='outer',suffixes=('_cx','_gnn'),indicator=True,validate='one_to_one')
 if not paired._merge.eq('both').all():raise RuntimeError('City mismatch')
 if not paired.n_cx.eq(paired.n_gnn).all():raise RuntimeError('City road-count mismatch')
 for met in ['mcc','balanced_accuracy','f1','roc_auc','pr_auc']:
  paired['delta_'+met+'_cx_minus_gnn']=paired[met+'_cx']-paired[met+'_gnn']
 save(paired.drop(columns='_merge'),'paired_seed_city_cartographx_vs_gnn.csv')
 # Report per-city, per-seed MCC spread; no pooled seed-road pseudo-replication.
 disparity=[]
 for (family,seed,protocol,country),g in city.groupby(['model_family','seed','protocol','country']):
  valid=g.loc[g.mcc.notna()]
  disparity.append(dict(model_family=family,seed=seed,protocol=protocol,country=country,
      n_cities=len(g),n_valid_cities=len(valid),
      worst_city_mcc=float(valid.mcc.min()) if len(valid) else np.nan,
      best_city_mcc=float(valid.mcc.max()) if len(valid) else np.nan,
      city_mcc_range=float(valid.mcc.max()-valid.mcc.min()) if len(valid)>1 else np.nan,
      unweighted_city_mcc_mean=float(valid.mcc.mean()) if len(valid) else np.nan))
 disparity=pd.DataFrame(disparity)
 save(disparity,'city_disparity_by_seed.csv')
 disparity_summary=disparity.groupby(['model_family','protocol','country'],as_index=False).agg(
     n_seeds=('seed','nunique'),worst_city_mcc_mean=('worst_city_mcc','mean'),
     worst_city_mcc_std=('worst_city_mcc','std'),
     best_city_mcc_mean=('best_city_mcc','mean'),
     city_mcc_range_mean=('city_mcc_range','mean'),
     unweighted_city_mcc_mean=('unweighted_city_mcc_mean','mean'))
 save(disparity_summary,'table_city_disparity_summary.csv')
 cis=tile_cluster_ci(data,args.bootstrap,args.seed)
 save(cis,'table_spatial_tile_bootstrap_intervals.csv')
 warning=pd.DataFrame([{'check':'single_class_city_groups','count':int(city.mcc.isna().sum())},
  {'check':'single_class_tile_groups','count':int(tile.mcc.isna().sum())},
  {'check':'historical_test_reused','count':1},
  {'check':'C2_labels_provisional','count':1},
  {'check':'same_test_roads_across_seeds_not_independent','count':1},
  {'check':'geometric_knn_not_road_topology','count':1}])
 save(warning,'robustness_warnings.csv')
 manifest={'script':Path(__file__).name,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
  'model_seeds':list(SEEDS),'bootstrap_replicates':args.bootstrap,'bootstrap_rng_seed':args.seed,
  'source_hashes':{'cartographx_predictions':sha(CX),'gnn_selected_results':sha(GNNS),'nodes':sha(NODES)},
  'scope':'descriptive geographic robustness, historical exploratory test, not an untouched holdout',
  'labels':'C2 provisional, not manually validated','figures_generated':0,
  'files_sha256':{f.name:sha(f) for f in OUT.glob('*.csv')}}
 (OUT/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print('\nCITY ROBUSTNESS (mean over seeds):')
 print(city_summary[['model_family','protocol','city','mcc_mean','mcc_std','n_roads_per_seed']].to_string(index=False))
 print(f'\nSCRIPT 118 PASS. Saved: {OUT}')
 print('No training, model selection or figures. Historical test exploratory; C2 provisional.')
 print('NEXT: Script 119 domain-shift diagnostics. Script 125 is FINAL figure code.')

if __name__=='__main__':main()
