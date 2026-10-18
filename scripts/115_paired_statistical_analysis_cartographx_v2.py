#!/usr/bin/env python3
"""Script 115: paired ten-seed exploratory comparison, with provenance checks.

Run: python scripts/115_paired_statistical_analysis_cartographx_v2.py
Optional: --bootstrap-reps 20000 --bootstrap-seed 20261014

Statistical unit is the training seed on a FIXED geographic test set. Thus CIs,
p-values and effect sizes quantify seed-level variability, NOT independent
geographic generalization. Previously inspected test and provisional C2 labels
make all test-based comparisons EXPLORATORY ONLY.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
CARTO=ROOT/'data/processed/cross_country/cartographx_v2_ten_seed_summary_v1'
CLASSIC=ROOT/'data/processed/cross_country/benchmark_v2_ten_seed_summary_v1'
OUT=ROOT/'data/processed/cross_country/cartographx_v2_statistics_v1'
PROTOCOLS=('lt_to_lt','pl_to_pl','lt_to_pl_zero_shot','pl_to_lt_zero_shot','joint_to_lt','joint_to_pl')
METRICS=('mcc','balanced_accuracy','f1','roc_auc','pr_auc')


def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()


def require(ok,message):
 if not ok:raise RuntimeError(message)


def load_results(path,model,seeds):
 require(path.is_file(),f'Missing {model}: {path}')
 d=pd.read_csv(path,low_memory=False)
 cols={'seed','protocol'}|{'test_'+x for x in METRICS}
 require(cols.issubset(d.columns),f'{model}: missing columns {sorted(cols-set(d.columns))}')
 require(not d.duplicated(['seed','protocol']).any(),f'{model}: duplicated seed/protocol')
 require(set(d.seed.astype(int))==set(seeds),f'{model}: seed set mismatch: {sorted(d.seed.unique())}')
 require(set(d.protocol)==set(PROTOCOLS),f'{model}: protocol set mismatch')
 require(len(d)==len(seeds)*len(PROTOCOLS),f'{model}: incomplete seed/protocol grid')
 for m in METRICS:
  require(np.isfinite(pd.to_numeric(d['test_'+m],errors='coerce')).all(),f'{model}: nonfinite test_{m}')
 return d


def holm(pvals):
 """Holm step-down family-wise adjustment; conservative for related tests."""
 p=np.asarray(pvals,float);out=np.full(len(p),np.nan)
 valid=np.flatnonzero(np.isfinite(p));order=valid[np.argsort(p[valid])]
 running=0.
 for i,idx in enumerate(order):
  running=max(running,min(1.,(len(order)-i)*p[idx]));out[idx]=running
 return out


def paired_stats(deltas,reps,rng):
 d=np.asarray(deltas,float);n=len(d)
 require(n>=3,'At least three paired seeds required')
 mean=float(d.mean());sd=float(d.std(ddof=1));sem=sd/math.sqrt(n)
 ci_t=stats.t.interval(.95,df=n-1,loc=mean,scale=sem) if sd>0 else (mean,mean)
 if sd>0:
  t_stat,t_p=stats.ttest_1samp(d,0.)
  t_stat=float(t_stat);t_p=float(t_p)
  dz=mean/sd
 else:
  t_stat=float('nan');t_p=(1. if mean==0 else 0.)
  dz=(0. if mean==0 else float('nan'))
 # Sign-flip randomization is exact for <= 16 paired seeds (2^n flips).
 # It assumes symmetric paired differences under the null.
 if n<=16:
  flips=np.array([[1 if ((mask>>j)&1) else -1 for j in range(n)] for mask in range(1<<n)],dtype=float)
  null_means=(flips@d)/n
  signflip_p=float(np.mean(np.abs(null_means)>=abs(mean)-1e-12))
 else:
  signflip_p=float('nan')
 # Wilcoxon is undefined if all differences are exactly zero.
 if np.all(d==0):
  wilcox_p=1.;wilcox_w=0.
 else:
  try:
   w=stats.wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='auto')
   wilcox_w=float(w.statistic);wilcox_p=float(w.pvalue)
  except ValueError:
   wilcox_w=float('nan');wilcox_p=float('nan')
 # Exact two-sided binomial sign test ignores ties.
 pos=int(np.count_nonzero(d>0));neg=int(np.count_nonzero(d<0));ties=n-pos-neg
 sign_p=float(stats.binomtest(pos,pos+neg,.5).pvalue) if pos+neg else 1.
 if reps>0:
  indices=rng.integers(0,n,size=(reps,n))
  boot=np.mean(d[indices],axis=1)
  b_lo,b_hi=np.quantile(boot,[.025,.975])
 else:b_lo,b_hi=float('nan'),float('nan')
 try:shapiro_p=float(stats.shapiro(d).pvalue) if sd>0 and n>=3 else float('nan')
 except Exception:shapiro_p=float('nan')
 return {'n_seeds':n,'delta_mean':mean,'delta_std':sd,'delta_median':float(np.median(d)),
  'delta_min':float(d.min()),'delta_max':float(d.max()),
  'ci95_t_low':float(ci_t[0]),'ci95_t_high':float(ci_t[1]),
  'ci95_seed_bootstrap_low':float(b_lo),'ci95_seed_bootstrap_high':float(b_hi),
  'paired_t_statistic':t_stat,'paired_t_p':t_p,'wilcoxon_statistic':wilcox_w,
  'wilcoxon_p':wilcox_p,'exact_signflip_p':signflip_p,'exact_sign_test_p':sign_p,
  'positive_seeds':pos,'negative_seeds':neg,'tied_seeds':ties,
  'cohens_dz':dz,'shapiro_difference_p':shapiro_p}


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--seeds',type=int,nargs='+',default=list(range(42,52)))
 ap.add_argument('--bootstrap-reps',type=int,default=20000)
 ap.add_argument('--bootstrap-seed',type=int,default=20261014)
 args=ap.parse_args()
 seeds=sorted(args.seeds)
 require(len(seeds)>=3 and len(seeds)==len(set(seeds)),'At least 3 unique seeds required')
 require(args.bootstrap_reps>=0,'Bootstrap repetitions cannot be negative')
 cp=CARTO/'all_seed_protocol_results.csv';bp=CLASSIC/'all_seed_protocol_results.csv'
 c=load_results(cp,'CartoGraph-X',seeds);b=load_results(bp,'classical baseline',seeds)
 # Verify original ten-seed runner provenance where available.
 manifest_path=CARTO/'ten_seed_manifest.json'
 require(manifest_path.is_file(),'Missing Script 114 manifest')
 manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
 require(sorted(manifest.get('seeds',[]))==seeds,'Script 114 manifest seed mismatch')
 hashes=manifest.get('output_sha256',{})
 require(hashes.get(cp.name)==sha256(cp),'CartoGraph-X result SHA256 differs from Script 114 manifest')
 # Classical baseline manifest may have a different schema; hash the file for provenance.
 a=c[['seed','protocol']+['test_'+m for m in METRICS]].copy()
 d=a.merge(b[['seed','protocol']+['test_'+m for m in METRICS]],on=['seed','protocol'],suffixes=('_cartographx','_classical'),validate='one_to_one')
 require(len(d)==len(PROTOCOLS)*len(seeds),'Incomplete paired comparison')
 d=d.sort_values(['protocol','seed']).reset_index(drop=True)
 for m in METRICS:d['delta_'+m]=d[f'test_{m}_cartographx']-d[f'test_{m}_classical']
 OUT.mkdir(parents=True,exist_ok=True)
 d.to_csv(OUT/'matched_seed_all_metric_differences.csv',index=False)
 rng=np.random.default_rng(args.bootstrap_seed)
 rows=[]
 for metric in METRICS:
  for protocol in PROTOCOLS:
   s=d[d.protocol.eq(protocol)]
   st=paired_stats(s['delta_'+metric].to_numpy(),args.bootstrap_reps,rng)
   st.update({'metric':metric,'protocol':protocol,
    'cartographx_mean':float(s['test_'+metric+'_cartographx'].mean()),
    'cartographx_sd':float(s['test_'+metric+'_cartographx'].std(ddof=1)),
    'classical_mean':float(s['test_'+metric+'_classical'].mean()),
    'classical_sd':float(s['test_'+metric+'_classical'].std(ddof=1))})
   rows.append(st)
 summary=pd.DataFrame(rows)
 # Six protocol tests constitute one family PER METRIC; also adjust all 30 tests.
 for method in ('paired_t_p','wilcoxon_p','exact_signflip_p','exact_sign_test_p'):
  summary[method+'_holm_within_metric']=np.nan
  for metric in METRICS:
   ix=summary.index[summary.metric.eq(metric)]
   summary.loc[ix,method+'_holm_within_metric']=holm(summary.loc[ix,method].to_numpy())
  summary[method+'_holm_all_30']=holm(summary[method].to_numpy())
 summary.to_csv(OUT/'paired_statistical_tests_all_metrics.csv',index=False)
 mcc=summary[summary.metric.eq('mcc')].copy()
 mcc.to_csv(OUT/'table_mcc_paired_statistical_comparison.csv',index=False)
 # Report validation and OOF descriptively, without claiming cross-model paired CV equivalence.
 other=[]
 for name,frame in [('CartoGraph-X',c),('Classical',b)]:
  cols=['validation_mcc','cv_mcc','cv_mcc_at_0_5','threshold']
  for col in cols:
   if col in frame.columns:
    for protocol in PROTOCOLS:
     vals=pd.to_numeric(frame.loc[frame.protocol.eq(protocol),col],errors='coerce').dropna()
     if len(vals):other.append({'model':name,'protocol':protocol,'field':col,'n':len(vals),'mean':float(vals.mean()),'sd':float(vals.std(ddof=1))})
 pd.DataFrame(other).to_csv(OUT/'validation_and_cv_descriptive_summary.csv',index=False)
 cvpath=CARTO/'all_seed_spatial_cv_country_diagnostics.csv'
 if cvpath.is_file():
  cv=pd.read_csv(cvpath)
  needed={'seed','protocol','country','cv_mcc','cv_roc_auc'}
  require(needed.issubset(cv.columns),'Country CV file missing required fields')
  require(not cv.duplicated(['seed','protocol','country']).any(),'Duplicate country CV records')
  require(set(cv.seed)==set(seeds),'Country CV seeds mismatch')
  country=cv.groupby(['protocol','country'],as_index=False).agg(
   n_seeds=('seed','nunique'),cv_mcc_mean=('cv_mcc','mean'),cv_mcc_sd=('cv_mcc','std'),
   cv_roc_auc_mean=('cv_roc_auc','mean'),cv_roc_auc_sd=('cv_roc_auc','std'))
  country.to_csv(OUT/'country_specific_spatial_cv_summary.csv',index=False)
  # Same joint trained model used for joint_to_lt and joint_to_pl; do not double-count.
  joint=cv[cv.protocol.eq('joint_to_lt')].pivot(index='seed',columns='country',values='cv_mcc')
  if {'Lithuania','Poland'}.issubset(joint.columns):
   country_gap=joint.reset_index();country_gap['lt_minus_pl_cv_mcc']=country_gap.Lithuania-country_gap.Poland
   country_gap.to_csv(OUT/'joint_country_cv_mcc_gap_by_seed.csv',index=False)
 # Directional asymmetry: within-model seed-matched, descriptive only.
 asym=[]
 for model,frame in [('CartoGraph-X',c),('Classical',b)]:
  p=frame.pivot(index='seed',columns='protocol',values='test_mcc')
  for label,left,right in [('zero_shot_lt_to_pl_minus_pl_to_lt','lt_to_pl_zero_shot','pl_to_lt_zero_shot'),
                           ('joint_lt_minus_pl','joint_to_lt','joint_to_pl')]:
   vals=(p[left]-p[right]).to_numpy()
   asym.append({'model':model,'comparison':label,'mean_mcc_gap':float(vals.mean()),'sd_mcc_gap':float(vals.std(ddof=1)),'n_seeds':len(vals)})
 pd.DataFrame(asym).to_csv(OUT/'directional_asymmetry_descriptive.csv',index=False)
 notes={
  'analysis_type':'EXPLORATORY_FIXED_TEST_SEED_PAIRED',
  'unit_of_analysis':'matched training initialization seed; NOT independent road, tile, city or country',
  'paired_design':'same seed indices, same test partition; classical and CartoGraph-X have different training/selection pipelines',
  'test_status':'historical test already used in previous experiments; NOT untouched confirmatory holdout',
  'labels_status':'C2 unified correspondence provisional; manual review incomplete',
  'p_value_caution':'p-values describe initialization-seed comparisons conditional on this test set and should not be interpreted as evidence of geographic generalization',
  'confidence_interval_caution':'t and bootstrap CIs quantify seed-level variability only; bootstrap resamples seeds, not geographic units',
  'multiplicity':'Holm adjustment across six protocols per metric and across all 30 tests',
  'signflip_assumption':'exact paired sign-flip randomization requires symmetry/exchangeability of differences under null',
  'normality':'Shapiro-Wilk diagnostic only; not a guarantee of normality with n=10',
  'cohens_dz':'paired mean difference / SD of paired differences; undefined when SD=0 and nonzero mean',
  'bootstrap_reps':args.bootstrap_reps,'bootstrap_seed':args.bootstrap_seed,'seeds':seeds,
  'inputs_sha256':{str(cp.relative_to(ROOT)):sha256(cp),str(bp.relative_to(ROOT)):sha256(bp),str(manifest_path.relative_to(ROOT)):sha256(manifest_path)},
  'timestamp_utc':datetime.now(timezone.utc).isoformat()}
 (OUT/'statistical_analysis_manifest.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
 (OUT/'STATISTICAL_INTERPRETATION.txt').write_text(
  'EXPLORATORY ONLY. These tests pair 10 random seeds on a FIXED, PREVIOUSLY INSPECTED test set.\n'
  'Seed-level uncertainty is NOT geographic sampling uncertainty. C2 labels are not manually validated.\n'
  'Do not claim confirmatory significance, independent replication, or generalization to new cities.\n'
  'Holm-adjusted p-values are provided as descriptive exploratory diagnostics.\n',encoding='utf-8')
 print('\nSCRIPT 115: EXPLORATORY PAIRED MCC ANALYSIS (10 matched seeds)')
 view=['protocol','classical_mean','cartographx_mean','delta_mean','ci95_t_low','ci95_t_high','exact_signflip_p','exact_signflip_p_holm_within_metric']
 print(mcc[view].to_string(index=False,float_format=lambda x:f'{x:.5f}'))
 print(f'\nSaved: {OUT}')
 print('WARNING: fixed historical test previously inspected; C2 labels provisional; seed-level p-values are NOT geographic generalization evidence.')

if __name__=='__main__':main()
