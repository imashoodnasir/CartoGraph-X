#!/usr/bin/env python
"""Script 85: CartoGraph-X zero-shot transfer diagnostics (NO model selection).

Run from D:/Research/CartoGen:
 python scripts/85_diagnose_cartographx_transfer.py
 python scripts/85_diagnose_cartographx_transfer.py --protocols lt_to_pl_zero_shot

Reads Script 68 node features and Script 84/V5 saved anchor and test predictions.
All target-test-label analyses are DIAGNOSTIC ONLY, never for selecting thresholds,
models, training hyperparameters or an eventual zero-shot deployment.
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, balanced_accuracy_score, brier_score_loss)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore',category=FutureWarning)
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data'/'processed'/'cross_country'
NODES=BASE/'graph_representations'/'road_graph_nodes.csv'
V5=BASE/'cartographx_v5_models'/'development'
OUT=BASE/'cartographx_transfer_diagnostics'
PROTOCOLS={'lt_to_pl_zero_shot':('Lithuania','Poland'),
           'pl_to_lt_zero_shot':('Poland','Lithuania')}
TAB=['length_m','intersecting_road_count','neighbor_count_150m','mean_neighbor_distance_m']
GRAPH=['graph_degree','graph_mean_edge_distance_m','graph_min_edge_distance_m','graph_max_edge_distance_m',
'graph_local_clustering','graph_component_size','graph_is_isolated','local_road_count_50m',
'local_road_count_100m','local_road_count_150m','local_road_count_250m','nbr_mean_length_m',
'nbr_std_length_m','nbr_mean_intersecting_road_count','nbr_std_intersecting_road_count',
'nbr_mean_neighbor_count_150m','nbr_std_neighbor_count_150m',
'nbr_mean_mean_neighbor_distance_m','nbr_std_mean_neighbor_distance_m']
FEATURES=TAB+GRAPH

def safe_auc(y,p):
 return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan

def safe_pr(y,p):
 return float(average_precision_score(y,p)) if len(np.unique(y))==2 else np.nan

def ece(y,p,bins=10):
 ids=np.minimum((p*bins).astype(int),bins-1)
 return float(sum(np.mean(ids==k)*abs(y[ids==k].mean()-p[ids==k].mean())
                  for k in range(bins) if np.any(ids==k)))

def score(y,p,t):
 yp=(p>=t).astype(int)
 return {'mcc':float(matthews_corrcoef(y,yp)),
         'balanced_accuracy':float(balanced_accuracy_score(y,yp)),
         'predicted_positive_rate':float(yp.mean())}

def drift_table(source,target):
 rows=[]
 for col in FEATURES:
  a=pd.to_numeric(source[col],errors='coerce').replace([np.inf,-np.inf],np.nan).dropna().to_numpy(float)
  b=pd.to_numeric(target[col],errors='coerce').replace([np.inf,-np.inf],np.nan).dropna().to_numpy(float)
  if not len(a) or not len(b):continue
  pooled=np.sqrt((np.var(a)+np.var(b))/2)
  rows.append({'feature':col,'source_median':float(np.median(a)),
   'target_median':float(np.median(b)),
   'standardized_mean_difference':float((np.mean(b)-np.mean(a))/pooled) if pooled>0 else 0.,
   'source_missing_rate':1-len(a)/len(source),'target_missing_rate':1-len(b)/len(target)})
 return pd.DataFrame(rows).sort_values('standardized_mean_difference',key=lambda s:s.abs(),ascending=False)

def country_discrimination(nodes,seed):
 """Country separability from TRAIN features only; spatially grouped OOF."""
 x=nodes[(nodes.split=='train') & nodes.country.isin(['Lithuania','Poland'])].copy()
 y=(x.country=='Poland').astype(int).to_numpy()
 groups=(x.country.astype(str)+'::'+x.tile_id.astype(str)).to_numpy()
 splits=GroupKFold(n_splits=min(5,len(np.unique(groups))))
 oof=np.full(len(x),np.nan)
 for fold,(a,b) in enumerate(splits.split(x,y,groups),1):
  if len(np.unique(y[a]))<2: raise ValueError('Country CV fold has only one country in training')
  model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),
                      StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced',random_state=seed))
  model.fit(x.iloc[a][FEATURES],y[a]);oof[b]=model.predict_proba(x.iloc[b][FEATURES])[:,1]
 return {'country_cv_roc_auc':safe_auc(y,oof),'country_cv_pr_auc':safe_pr(y,oof),
         'country_cv_samples':int(len(x)), 'country_cv_tiles':int(len(np.unique(groups)))}

def run(nodes,protocol,seed):
 source_country,target_country=PROTOCOLS[protocol]
 folder=V5/protocol/f'seed_{seed}'
 anchor_path=folder/'anchor_and_scalers.joblib'
 predictions_path=folder/'V5-A_anchor_test_predictions.csv'
 if not anchor_path.exists() or not predictions_path.exists():
  print(f'[SKIP] {protocol}: missing V5 outputs in {folder}');return
 out=OUT/protocol/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
 source_train=nodes[(nodes.country==source_country)&(nodes.split=='train')].copy()
 source_val=nodes[(nodes.country==source_country)&(nodes.split=='validation')].copy()
 target=nodes[(nodes.country==target_country)&(nodes.split=='test')].copy()
 # The saved anchor is already fitted on source TRAIN ONLY.
 artifact=joblib.load(anchor_path)
 model=artifact['anchor']; features=artifact['features']
 val_p=model.predict_proba(source_val[features])[:,1]
 val_y=source_val.retained.to_numpy(int)
 pred=pd.read_csv(predictions_path)
 if pred.sample_uid.duplicated().any():raise ValueError('Duplicate test predictions')
 merged=target[['sample_uid','retained']].merge(pred[['sample_uid','probability_retained','threshold']],
                                                on='sample_uid',how='left',validate='one_to_one',indicator=True)
 if not (merged._merge=='both').all() or len(merged)!=len(target):
  raise ValueError('Saved test predictions do not match target-test sample UIDs')
 test_y=merged.retained.to_numpy(int)
 test_p=merged.probability_retained.to_numpy(float)
 saved_t=float(merged.threshold.iloc[0])
 if not np.allclose(merged.threshold,saved_t):raise ValueError('Inconsistent saved thresholds')
 thresholds=np.round(np.arange(.05,.951,.01),2)
 curve=[]
 for t in thresholds:
  vm=score(val_y,val_p,t);tm=score(test_y,test_p,t)
  curve.append({'threshold':t,'source_validation_mcc':vm['mcc'],
                'source_validation_balanced_accuracy':vm['balanced_accuracy'],
                'target_test_mcc_DIAGNOSTIC_ONLY':tm['mcc'],
                'target_test_balanced_accuracy_DIAGNOSTIC_ONLY':tm['balanced_accuracy'],
                'target_test_predicted_positive_rate_DIAGNOSTIC_ONLY':tm['predicted_positive_rate']})
 curve=pd.DataFrame(curve);curve.to_csv(out/'threshold_sensitivity_DIAGNOSTIC_ONLY.csv',index=False)
 # Oracle threshold is a descriptive diagnostic, NOT an admissible model-selection result.
 oracle=curve.loc[curve.target_test_mcc_DIAGNOSTIC_ONLY.idxmax()]
 report={'protocol':protocol,'seed':seed,'source_country':source_country,'target_country':target_country,
  'source_train_n':len(source_train),'source_validation_n':len(source_val),'target_test_n':len(target),
  'source_train_prevalence':float(source_train.retained.mean()),
  'source_validation_prevalence':float(source_val.retained.mean()),
  'target_test_prevalence_DIAGNOSTIC_ONLY':float(test_y.mean()),
  'saved_source_selected_threshold':saved_t,
  'source_validation_roc_auc':safe_auc(val_y,val_p),
  'target_test_roc_auc_DIAGNOSTIC_ONLY':safe_auc(test_y,test_p),
  'source_validation_pr_auc':safe_pr(val_y,val_p),
  'target_test_pr_auc_DIAGNOSTIC_ONLY':safe_pr(test_y,test_p),
  'source_validation_brier':float(brier_score_loss(val_y,val_p)),
  'target_test_brier_DIAGNOSTIC_ONLY':float(brier_score_loss(test_y,test_p)),
  'source_validation_ece':ece(val_y,val_p),
  'target_test_ece_DIAGNOSTIC_ONLY':ece(test_y,test_p),
  'source_validation_mean_probability':float(val_p.mean()),
  'target_test_mean_probability_DIAGNOSTIC_ONLY':float(test_p.mean()),
  'source_validation_mcc':score(val_y,val_p,saved_t)['mcc'],
  'target_test_mcc_DIAGNOSTIC_ONLY':score(test_y,test_p,saved_t)['mcc'],
  'target_oracle_threshold_DIAGNOSTIC_ONLY_NEVER_SELECT':float(oracle.threshold),
  'target_oracle_mcc_DIAGNOSTIC_ONLY_NEVER_SELECT':float(oracle.target_test_mcc_DIAGNOSTIC_ONLY)}
 drift_table(source_train,target).to_csv(out/'feature_shift.csv',index=False)
 # Probability bins: descriptive diagnostics, not threshold tuning.
 bins=np.linspace(0,1,11)
 calibration=[]
 for name,y,p in [('source_validation',val_y,val_p),('target_test_DIAGNOSTIC_ONLY',test_y,test_p)]:
  idx=np.minimum(np.digitize(p,bins[1:-1]),9)
  for k in range(10):
   mask=idx==k
   calibration.append({'partition':name,'bin':k,'count':int(mask.sum()),
                       'mean_predicted':float(p[mask].mean()) if mask.any() else np.nan,
                       'observed_positive_fraction':float(y[mask].mean()) if mask.any() else np.nan})
 pd.DataFrame(calibration).to_csv(out/'calibration_bins_DIAGNOSTIC_ONLY.csv',index=False)
 (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(f'\n{protocol}: source val MCC={report["source_validation_mcc"]:.4f}, '
       f'target test MCC={report["target_test_mcc_DIAGNOSTIC_ONLY"]:.4f}; '
       f'ROC-AUC source={report["source_validation_roc_auc"]:.4f}, '
       f'target={report["target_test_roc_auc_DIAGNOSTIC_ONLY"]:.4f}')
 print(f'  Saved threshold={saved_t:.2f}; target oracle threshold '
       f'{report["target_oracle_threshold_DIAGNOSTIC_ONLY_NEVER_SELECT"]:.2f} '
       '(DIAGNOSTIC ONLY, forbidden for selection)')
 return report

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--protocols',nargs='+',choices=list(PROTOCOLS),default=list(PROTOCOLS))
 parser.add_argument('--seed',type=int,default=42)
 args=parser.parse_args()
 if not NODES.exists():parser.error(f'Missing {NODES}')
 nodes=pd.read_csv(NODES,low_memory=False)
 missing=[c for c in ['sample_uid','country','split','tile_id','retained']+FEATURES if c not in nodes.columns]
 if missing:parser.error(f'Missing columns: {missing}')
 OUT.mkdir(parents=True,exist_ok=True)
 report=country_discrimination(nodes,args.seed)
 (OUT/'country_separability_train_only.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print('Spatially grouped TRAIN-only country classifier ROC-AUC:',round(report['country_cv_roc_auc'],4))
 reports=[r for p in args.protocols if (r:=run(nodes,p,args.seed)) is not None]
 if reports:pd.DataFrame(reports).to_csv(OUT/f'diagnostic_summary_seed_{args.seed}.csv',index=False)
 print(f'\n[PASS] Diagnostic reports saved to {OUT}')
 print('IMPORTANT: target oracle thresholds and target calibration are analysis-only; do NOT use them to select V6.')
if __name__=='__main__':main()
