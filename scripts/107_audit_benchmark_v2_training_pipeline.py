#!/usr/bin/env python3
"""Read-only static + output audit of CartoGraph-X Script 106 (seed 42 by default).

Usage: python scripts/107_audit_benchmark_v2_training_pipeline.py --seed 42
Does not train, alter data, or certify an untouched holdout.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'scripts/106_train_benchmark_v2_classical_baselines.py'
DATA=ROOT/'data/processed/cross_country/benchmark_v2_unified_labels_v1/road_graph_nodes_v2.csv'
BASE=ROOT/'data/processed/cross_country/benchmark_v2_classical_baselines_v1'
OUT=ROOT/'data/processed/cross_country/benchmark_v2_training_pipeline_audit_v1'
EXPECTED={
'lt_to_lt':('Lithuania','Lithuania',False),
'pl_to_pl':('Poland','Poland',False),
'lt_to_pl_zero_shot':('Lithuania','Poland',False),
'pl_to_lt_zero_shot':('Poland','Lithuania',False),
'joint_to_lt':('Lithuania','Lithuania',True),
'joint_to_pl':('Poland','Poland',True)}

def hashfile(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);args=p.parse_args()
 run=BASE/f'seed_{args.seed}';OUT.mkdir(parents=True,exist_ok=True)
 checks=[]
 def check(name,passed,details='',warn=False):
  status='PASS' if passed else ('WARN' if warn else 'FAIL')
  checks.append({'check':name,'status':status,'details':str(details)})
  print(f'[{status}] {name}: {details}',flush=True)
  return passed
 for f in [SRC,DATA,run/'spatial_cv_candidate_results.csv',run/'selected_protocol_results.csv',run/'selected_test_predictions.csv',run/'experiment_manifest.json']:
  if not f.exists():
   print(f'MISSING INPUT: {f}',file=sys.stderr);sys.exit(2)
 source=SRC.read_text(encoding='utf-8');tree=ast.parse(source)
 funcs={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 check('source_main_present','main' in funcs)
 check('source_cv_split_function','choose_cv_splits' in funcs)
 check('source_threshold_function','select_threshold' in funcs)
 check('source_scoring_function','scores' in funcs)
 check('pipeline_preprocessing', 'make_pipeline(' in source and 'SimpleImputer(' in source and 'StandardScaler(' in source,
       'imputation/scaling constructed inside estimators')
 check('spatial_cv_grouping','StratifiedGroupKFold' in source and "train.country.astype(str) + '::' + train.tile_id.astype(str)" in source)
 check('spatial_fold_disjoint_assertion','set(groups[tr]).intersection(groups[va])' in source)
 check('cv_fit_on_training_fold','estimator.fit(X[tr], y[tr])' in source and 'estimator.predict_proba(X[va])' in source)
 check('train_only_candidate_selection','key = (cv_mcc, cv_auc' in source and 'if best_key is None or key > best_key' in source)
 check('source_validation_selection',"validation_country = target if joint or source == target else source" in source)
 check('validation_threshold_only','select_threshold(validation.retained.to_numpy(dtype=int), val_probs)' in source)
 check('no_test_labels_in_threshold','select_threshold(evaluation.' not in source and 'select_threshold(test.' not in source)
 check('test_metrics_after_selection',source.find('test_metrics = scores(')>source.find('threshold = select_threshold(')>source.find('best_choice = (feature_set, model_name)'))
 # These are pattern-level safeguards, not a proof of noninterference.
 check('static_audit_limitations',False,'Static checks cannot prove absence of indirect leakage; separate negative-control tests needed',warn=True)
 df=pd.read_csv(DATA,low_memory=False)
 cv=pd.read_csv(run/'spatial_cv_candidate_results.csv');res=pd.read_csv(run/'selected_protocol_results.csv');pred=pd.read_csv(run/'selected_test_predictions.csv')
 manifest=json.loads((run/'experiment_manifest.json').read_text(encoding='utf-8'))
 check('input_hash_matches_manifest',manifest.get('input_sha256')==hashfile(DATA))
 for name in ['spatial_cv_candidate_results.csv','selected_protocol_results.csv','selected_test_predictions.csv']:
  check('output_hash_'+name,manifest.get('output_hashes',{}).get(name)==hashfile(run/name))
 check('seed_consistency',all(d.seed.eq(args.seed).all() for d in [cv,res,pred]))
 check('protocol_set_results',set(res.protocol)==set(EXPECTED) and len(res)==6)
 check('protocol_set_cv',set(cv.protocol)==set(EXPECTED))
 check('protocol_set_predictions',set(pred.protocol)==set(EXPECTED))
 check('cv_candidates_12_each',all(len(cv[cv.protocol==k])==12 for k in EXPECTED),cv.groupby('protocol').size().to_dict())
 check('graph_uid_unique',df.sample_uid.is_unique)
 check('test_rows_unique',df.loc[df.split.eq('test'),'sample_uid'].is_unique)
 check('country_tile_no_cross_split',not (df.groupby(['country','tile_id']).split.nunique()>1).any())
 checks_per=[]
 for protocol,(source_country,target_country,joint) in EXPECTED.items():
  rr=res[res.protocol.eq(protocol)]
  if len(rr)!=1:continue
  r=rr.iloc[0]; pp=pred[pred.protocol.eq(protocol)].copy(); cc=cv[cv.protocol.eq(protocol)].copy()
  exp_train=df[(df.split=='train') & (df.country.isin(['Lithuania','Poland']) if joint else df.country.eq(source_country))]
  val_country=target_country if joint or source_country==target_country else source_country
  exp_val=df[(df.split=='validation') & df.country.eq(val_country)]
  exp_test=df[(df.split=='test') & df.country.eq(target_country)]
  check(protocol+'_partition_metadata',str(r.source)==source_country and str(r.target)==target_country and bool(r.joint_training)==joint and str(r.threshold_selection_country)==val_country)
  check(protocol+'_partition_counts',int(r.train_n)==len(exp_train) and int(r.validation_n)==len(exp_val) and int(r.test_n)==len(exp_test))
  check(protocol+'_prediction_uid_set',len(pp)==len(exp_test) and pp.sample_uid.is_unique and set(pp.sample_uid)==set(exp_test.sample_uid))
  check(protocol+'_prediction_country_split',pp.country.eq(target_country).all() and pp.split.eq('test').all())
  check(protocol+'_prediction_labels',pp.set_index('sample_uid').label.reindex(exp_test.sample_uid).to_numpy().tolist()==exp_test.retained.to_numpy().tolist())
  threshold=float(r.threshold)
  check(protocol+'_threshold_predictions',np.allclose(pp.threshold.to_numpy(dtype=float),threshold,atol=1e-12,rtol=0) and np.array_equal(pp.prediction.to_numpy(dtype=int),(pp.probability.to_numpy(dtype=float)>=threshold).astype(int)))
  check(protocol+'_model_feature_match',pp.model.eq(r.model).all() and pp.feature_set.eq(r.feature_set).all())
  check(protocol+'_selected_candidate_exists',((cc.model==r.model)&(cc.feature_set==r.feature_set)).sum()==1)
  if len(cc)==12:
   models=['LogisticRegression','RandomForest','ExtraTrees','HistGradientBoosting']; features=['tabular','graph','tabular_graph']
   cc['model_rank']=cc.model.map({m:i for i,m in enumerate(models)});cc['feature_rank']=cc.feature_set.map({f:i for i,f in enumerate(features)})
   if cc[['model_rank','feature_rank']].notna().all().all():
    top=cc.sort_values(['cv_mcc_at_0_5','cv_roc_auc','feature_rank','model_rank'],ascending=[False,False,True,True]).iloc[0]
    check(protocol+'_selected_train_cv_argmax',str(top.model)==str(r.model) and str(top.feature_set)==str(r.feature_set))
   else:check(protocol+'_selected_train_cv_argmax',False,'Unexpected model/feature names')
  y=pp.label.to_numpy(dtype=int);pr=pp.probability.to_numpy(dtype=float);yh=pp.prediction.to_numpy(dtype=int)
  tn,fp,fn,tp=confusion_matrix(y,yh,labels=[0,1]).ravel()
  metric={'mcc':matthews_corrcoef(y,yh),'balanced_accuracy':balanced_accuracy_score(y,yh),'f1':f1_score(y,yh,zero_division=0),'roc_auc':roc_auc_score(y,pr) if len(np.unique(y))==2 else np.nan,'pr_auc':average_precision_score(y,pr) if len(np.unique(y))==2 else np.nan,'positive_prediction_rate':yh.mean(),'positive_label_rate':y.mean(),'tn':tn,'fp':fp,'fn':fn,'tp':tp}
  mismatches=[]
  for k,v in metric.items():
   stored=r.get('test_'+k,np.nan)
   if not np.isclose(float(stored),float(v),atol=1e-9,rtol=1e-9,equal_nan=True):mismatches.append((k,stored,v))
  check(protocol+'_recomputed_test_metrics',not mismatches,mismatches[:3])
  checks_per.append({'protocol':protocol,'source':source_country,'target':target_country,'selection_country':val_country,'train_n':len(exp_train),'validation_n':len(exp_val),'test_n':len(exp_test),'model':r.model,'feature_set':r.feature_set,'threshold':threshold,'test_mcc':float(metric['mcc'])})
 # Joint-country CV pooled MCC is not a per-target objective.
 check('joint_cv_country_balance_caveat',False,'Joint pooled CV MCC may mask weaker Poland performance; report per-country CV metrics separately',warn=True)
 check('holdout_not_untouched',False,'Historical test partition previously examined; exploratory evaluation only',warn=True)
 check('manual_validation_pending',False,'C2 labels remain provisional, without completed manual review',warn=True)
 pd.DataFrame(checks).to_csv(OUT/f'seed_{args.seed}_audit_checks.csv',index=False)
 pd.DataFrame(checks_per).to_csv(OUT/f'seed_{args.seed}_protocol_verification.csv',index=False)
 failures=sum(x['status']=='FAIL' for x in checks);warnings=sum(x['status']=='WARN' for x in checks)
 status='FAIL' if failures else 'PASS_WITH_WARNINGS' if warnings else 'PASS'
 report={'status':status,'seed':args.seed,'checks':len(checks),'failures':failures,'warnings':warnings,'source_sha256':hashfile(SRC),'input_sha256':hashfile(DATA),'caveat':'Static source-pattern checks and saved-result consistency do not establish causal noninterference or an untouched holdout.'}
 (OUT/f'seed_{args.seed}_audit_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(f'\nSCRIPT 107 STATUS: {status} | checks={len(checks)} failures={failures} warnings={warnings}')
 print(f'Output: {OUT}')
 if failures:sys.exit(1)

if __name__=='__main__':main()
