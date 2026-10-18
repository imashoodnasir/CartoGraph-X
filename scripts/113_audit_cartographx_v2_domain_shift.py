#!/usr/bin/env python3
"""Script 113: CartoGraph-X V2 implementation, leakage, and domain-shift audit.

Run: python scripts/113_audit_cartographx_v2_domain_shift.py --seed 42
Optional: --negative-controls (short training perturbation experiments; slower).
No new historical-test model fitting or threshold optimization is performed.
"""
from __future__ import annotations
import argparse, ast, hashlib, importlib.util, json, math, sys, traceback
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score, confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=Path(__file__).resolve().parent
DEFAULT=ROOT/'data/processed/cross_country/cartographx_v2_architecture_v1'
NODE=ROOT/'data/processed/cross_country/benchmark_v2_unified_labels_v1/road_graph_nodes_v2.csv'
EDGES=ROOT/'data/processed/cross_country/graph_representations/road_graph_edges.csv'
S112=SCRIPTS/'112_train_cartographx_v2_architecture.py'
S111=SCRIPTS/'111_train_benchmark_v2_weighted_gnn_baselines.py'
PROTOCOLS={'lt_to_lt':('Lithuania','Lithuania',False),'pl_to_pl':('Poland','Poland',False),'lt_to_pl_zero_shot':('Lithuania','Poland',False),'pl_to_lt_zero_shot':('Poland','Lithuania',False),'joint_to_lt':('Lithuania','Lithuania',True),'joint_to_pl':('Poland','Poland',True)}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def getcol(r, options):
 for k in options:
  if k in r.index:return r[k]
 raise KeyError('Missing any of '+str(options))

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--seed',type=int,default=42)
 ap.add_argument('--results-dir',type=Path)
 ap.add_argument('--output-dir',type=Path)
 ap.add_argument('--negative-controls',action='store_true',help='Run extra 2-epoch label perturbation controls; uses GPU if available')
 ap.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
 args=ap.parse_args()
 inp=args.results_dir or DEFAULT/f'seed_{args.seed}'
 out=args.output_dir or ROOT/'data/processed/cross_country/cartographx_v2_audit_v1'/f'seed_{args.seed}'
 out.mkdir(parents=True,exist_ok=True)
 checks=[];diagnostics=[];negative=[]
 def check(name,ok,detail='',warning=False):
  status='PASS' if ok else ('WARN' if warning else 'FAIL')
  checks.append(dict(check=name,status=status,detail=str(detail)))
  print(f'[{status}] {name}: {detail}',flush=True)
 def warn(name,detail):check(name,False,detail,True)
 def require(name,condition,detail=''):check(name,bool(condition),detail)
 for p in [S111,S112,NODE,EDGES,inp/'experiment_manifest.json',inp/'selected_protocol_results.csv',inp/'selected_test_predictions.csv',inp/'spatial_cv_country_diagnostics.csv']:
  require('file_exists_'+p.name,p.is_file(),str(p))
 if any(c['status']=='FAIL' for c in checks):
  finish(out,checks,diagnostics,negative,args);return 1
 manifest=json.loads((inp/'experiment_manifest.json').read_text(encoding='utf8'))
 require('manifest_seed',int(manifest['seed'])==args.seed)
 for field,path in [('script_sha256',S112),('dependency_script_111_sha256',S111),('node_sha256',NODE),('edge_sha256',EDGES)]:
  require('manifest_hash_'+field,manifest.get(field)==sha(path))
 for filename,hashval in manifest.get('outputs_sha256',{}).items():
  require('output_hash_'+filename,(inp/filename).exists() and sha(inp/filename)==hashval)
 src=S112.read_text(encoding='utf8');src111=S111.read_text(encoding='utf8')
 tree=ast.parse(src);tree111=ast.parse(src111)
 functions={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 def body(name):return ast.get_source_segment(src,functions[name]) if name in functions else ''
 train=body('train_predict');maincode=body('main');model=next((n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ResidualSAGE'),None)
 require('train_predict_present',bool(train))
 require('train_fitted_preprocessing', 'graph.subset(train_ids,fit=True' in train and 'imputer=imp,scaler=sc' in train)
 require('train_only_country_weights','torch.unique(c)' in train and 'y[mask]' in train)
 require('joint_only_residual_heads',model is not None and 'if self.joint:' in ast.get_source_segment(src,model))
 require('source_only_zero_shot_partition',"val_country=target if joint or source==target else source" in maincode and "if joint else df.country.eq(source)" in maincode)
 require('validation_threshold_only','threshold=base.threshold_for(graph.labels[val_ids],val_prob)' in maincode)
 require('test_scored_after_threshold','test_m=base.metrics(graph.labels[test_ids],test_prob,threshold)' in maincode)
 require('induced_graph_subsets','graph.subset(ids,imputer=imp,scaler=sc' in train)
 require('cv_group_disjoint_assertion',"set(groups[tr])&set(groups[va])" in maincode)
 require('no_early_stopping_on_validation','for epoch in range(args.epochs)' in train and 'optimizer.step()' in train)
 require('graph_subsets_training_fitted', 'if fit:' in src111 and 'scaler.fit_transform' in src111 and 'lookup[ids]' in src111)
 require('feature_set_primitive', "FEATURES = ['length_m','intersecting_road_count','neighbor_count_150m','mean_neighbor_distance_m']" in src111)
 warn('static_audit_limit','Static code inspection is not a formal proof of no indirect leakage')
 df=pd.read_csv(NODE,low_memory=False);df['sample_uid']=df['sample_uid'].astype(str)
 require('unique_uids',not df.sample_uid.duplicated().any())
 require('tile_split_disjoint',not (df.groupby(['country','tile_id']).split.nunique()>1).any())
 edges=pd.read_csv(EDGES,low_memory=False)
 lookup=df.set_index('sample_uid')
 require('all_edge_uids_exist',edges.source_uid.astype(str).isin(lookup.index).all() and edges.target_uid.astype(str).isin(lookup.index).all())
 for col in ['country','split','tile_id']:
  require('edges_do_not_cross_'+col,(edges.source_uid.astype(str).map(lookup[col]).to_numpy()==edges.target_uid.astype(str).map(lookup[col]).to_numpy()).all())
 results=pd.read_csv(inp/'selected_protocol_results.csv');pred=pd.read_csv(inp/'selected_test_predictions.csv');cv=pd.read_csv(inp/'spatial_cv_country_diagnostics.csv')
 require('six_protocols',set(results.protocol)==set(PROTOCOLS) and len(results)==6)
 require('prediction_protocols',set(pred.protocol)==set(PROTOCOLS))
 require('cv_protocols',set(cv.protocol)==set(PROTOCOLS))
 require('prediction_uid_protocol_unique',not pred.duplicated(['protocol','sample_uid']).any())
 require('results_seed_consistent',(results.seed==args.seed).all() and (pred.seed==args.seed).all())
 for protocol,(source,target,joint) in PROTOCOLS.items():
  r=results.loc[results.protocol==protocol]
  p=pred.loc[pred.protocol==protocol].copy()
  if len(r)!=1:
   require(protocol+'_one_result',False,len(r));continue
  r=r.iloc[0]
  tr=df[(df.split=='train') & (df.country.isin(['Lithuania','Poland']) if joint else df.country.eq(source))]
  vc=target if joint or source==target else source
  va=df[(df.split=='validation') & df.country.eq(vc)]
  te=df[(df.split=='test') & df.country.eq(target)]
  require(protocol+'_partition_metadata',str(r.source)==source and str(r.target)==target and bool(r.joint_training)==joint and str(r.threshold_selection_country)==vc)
  require(protocol+'_partition_counts',int(r.train_n)==len(tr) and int(r.validation_n)==len(va) and int(r.test_n)==len(te))
  require(protocol+'_test_uid_set',len(p)==len(te) and set(p.sample_uid.astype(str))==set(te.sample_uid.astype(str)))
  if len(p)!=len(te) or set(p.sample_uid.astype(str))!=set(te.sample_uid.astype(str)):continue
  p['sample_uid']=p.sample_uid.astype(str)
  true=te.set_index('sample_uid').loc[p.sample_uid]
  y=true.retained.to_numpy(dtype=int);prob=p.probability.to_numpy(dtype=float);t=float(r.threshold)
  require(protocol+'_labels_match',np.array_equal(y,p.label.to_numpy(dtype=int)))
  require(protocol+'_predictions_match',np.array_equal((prob>=t).astype(int),p.prediction.to_numpy(dtype=int)))
  require(protocol+'_probabilities_valid',np.isfinite(prob).all() and np.all((prob>=0)&(prob<=1)))
  tn,fp,fn,tp=confusion_matrix(y,prob>=t,labels=[0,1]).ravel()
  metrics={'mcc':matthews_corrcoef(y,prob>=t),'balanced_accuracy':balanced_accuracy_score(y,prob>=t),'f1':f1_score(y,prob>=t,zero_division=0),'roc_auc':roc_auc_score(y,prob),'pr_auc':average_precision_score(y,prob),'tn':tn,'fp':fp,'fn':fn,'tp':tp}
  diffs=[]
  for key,val in metrics.items():
   col='test_'+key
   if col not in r or not np.isclose(float(r[col]),float(val),atol=1e-7,rtol=1e-7):diffs.append(key)
  require(protocol+'_test_metrics_recomputed',not diffs,diffs)
  diag={'protocol':protocol,'source':source,'target':target,'joint':joint,'threshold':t,'n':len(p),'positive_rate':float(y.mean()),'prob_mean':float(prob.mean()),'prob_median':float(np.median(prob)),'prob_std':float(prob.std()),'pred_positive_rate':float((prob>=t).mean()),'prob_q05':float(np.quantile(prob,.05)),'prob_q25':float(np.quantile(prob,.25)),'prob_q75':float(np.quantile(prob,.75)),'prob_q95':float(np.quantile(prob,.95)),'test_mcc':float(metrics['mcc']),'test_auc':float(metrics['roc_auc']),'test_pr_auc':float(metrics['pr_auc']),'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn)}
  # Diagnostic only: never use target test labels to select or update a threshold.
  for threshold in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
   diag['descriptive_pred_rate_t'+str(threshold)]=float((prob>=threshold).mean())
  diagnostics.append(diag)
  if 'zero_shot' in protocol:
   require(protocol+'_source_only_validation',vc==source)
   require(protocol+'_source_only_train',set(tr.country)=={source})
  if joint:
   require(protocol+'_joint_both_countries_in_train',set(tr.country)=={'Lithuania','Poland'})
  print(f"  {protocol:22s} MCC={metrics['mcc']:.4f} positive_rate={y.mean():.3f} predicted_positive={diag['pred_positive_rate']:.3f} probability_mean={prob.mean():.3f}",flush=True)
 # Verify cross-country reuse of source-only models and thresholds.
 for a,b in [('lt_to_lt','lt_to_pl_zero_shot'),('pl_to_pl','pl_to_lt_zero_shot')]:
  ra=results.set_index('protocol').loc[a];rb=results.set_index('protocol').loc[b]
  for field in ['threshold','cv_mcc','cv_oof_threshold','validation_mcc','validation_roc_auc']:
   require(f'source_reuse_{a}_{b}_{field}',np.isclose(float(ra[field]),float(rb[field]),atol=1e-9))
 for a,b in [('joint_to_lt','joint_to_pl')]:
  ra=results.set_index('protocol').loc[a];rb=results.set_index('protocol').loc[b]
  require('joint_cv_shared',np.isclose(float(ra.cv_mcc),float(rb.cv_mcc)))
  require('joint_target_thresholds_separate',ra.threshold_selection_country!=rb.threshold_selection_country)
 for protocol in PROTOCOLS:
  rows=cv[cv.protocol==protocol]
  expected={'Lithuania','Poland'} if protocol.startswith('joint') else {PROTOCOLS[protocol][0]}
  require(protocol+'_cv_countries',set(rows.country)==expected)
  if len(rows):
   print('  CV '+protocol+': '+', '.join(f"{x.country} MCC={x.cv_mcc:.4f}" for x in rows.itertuples()),flush=True)
 warn('pooled_cv_not_target_cv','Pooled joint OOF MCC combines countries with different prevalences; use country-specific CV diagnostics')
 if args.negative_controls:
  try:
   import torch
   spec=importlib.util.spec_from_file_location('cartographx112audit',S112);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
   device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
   if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
   class Args:pass
   a=Args();a.device=device;a.epochs=2;a.hidden=16;a.dropout=.2;a.lr=.001;a.weight_decay=1e-4
   g=mod.base.GraphData(df,edges)
   for source,target in [('Lithuania','Poland'),('Poland','Lithuania')]:
    tr=np.flatnonzero((df.split=='train')&df.country.eq(source));target_val=np.flatnonzero((df.split=='validation')&df.country.eq(target));source_val=np.flatnonzero((df.split=='validation')&df.country.eq(source))
    original=mod.train_predict(g,tr,[source_val,target_val],a,args.seed+99173,False)
    old=g.labels.copy()
    try:
     g.labels[target_val]=1-g.labels[target_val]
     mutated=mod.train_predict(g,tr,[source_val,target_val],a,args.seed+99173,False)
    finally:g.labels[:]=old
    maxdiff=max(float(np.max(np.abs(x-y))) for x,y in zip(original,mutated))
    require('negative_control_target_validation_label_invariance_'+source,maxdiff<1e-7,f'max_probability_change={maxdiff:.3g}')
    negative.append({'source':source,'target':target,'control':'flip_target_validation_labels','max_abs_probability_change':maxdiff,'pass':maxdiff<1e-7})
   # Joint residual country head: gradient for unused country embedding must be zero.
   net=mod.ResidualSAGE(4,16,.0,True).to(device)
   xx=torch.randn(8,4,device=device);ee=torch.arange(8,device=device).repeat(2,1);cc=torch.zeros(8,dtype=torch.long,device=device)
   net(xx,ee,cc).sum().backward()
   for attr in ['country_delta','country_bias']:
    gradient=getattr(net,attr).weight.grad
    ok=gradient is not None and float(gradient[1].abs().max())==0.0 and float(gradient[0].abs().sum())>0
    require('joint_head_gradient_isolation_'+attr,ok)
  except Exception as exc:
   check('negative_controls_execution',False,repr(exc))
 else:warn('negative_controls_not_run','Optional --negative-controls runs target-label perturbation and head-gradient isolation')
 warn('manual_correspondence_pending','C2 geometric correspondence labels are provisional; manual review remains incomplete')
 warn('historical_test_exploratory','Historical test labels previously examined; test diagnostics must not guide further confirmatory model selection')
 return finish(out,checks,diagnostics,negative,args)

def finish(out,checks,diagnostics,negative,args):
 pd.DataFrame(checks).to_csv(out/'audit_checks.csv',index=False)
 pd.DataFrame(diagnostics).to_csv(out/'domain_shift_prediction_diagnostics.csv',index=False)
 pd.DataFrame(negative).to_csv(out/'negative_controls.csv',index=False)
 failures=sum(x['status']=='FAIL' for x in checks);warnings=sum(x['status']=='WARN' for x in checks)
 status='FAIL' if failures else ('PASS_WITH_WARNINGS' if warnings else 'PASS')
 manifest={'script':'113_audit_cartographx_v2_domain_shift.py','timestamp_utc':datetime.now(timezone.utc).isoformat(),'seed':args.seed,'status':status,'checks':len(checks),'failures':failures,'warnings':warnings,'negative_controls_requested':args.negative_controls,'limitations':['Source inspection cannot establish complete causal absence of leakage','Test diagnostics are descriptive and exploratory only','C2 correspondence labels require manual review']}
 (out/'audit_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
 print(f'\nSCRIPT 113 STATUS: {status} | checks={len(checks)} failures={failures} warnings={warnings}\nOutput: {out}',flush=True)
 return 1 if failures else 0

if __name__=='__main__':sys.exit(main())
