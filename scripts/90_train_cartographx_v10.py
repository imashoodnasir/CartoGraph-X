#!/usr/bin/env python
"""CartoGraph-X V10: source-safe feature robustness experiment.

Place next to 87_train_cartographx_v7.py under scripts/.
Development:
 python scripts/90_train_cartographx_v10.py --mode development --protocols lt_to_pl_zero_shot pl_to_lt_zero_shot
Full (after freezing design):
 python scripts/90_train_cartographx_v10.py --mode full

Selection: source training GroupKFold -> best feature set/model/config;
source validation -> MCC threshold; target test -> selected model only.
No target labels are read for feature ranking, model selection, or calibration.
Repeated seeds do not imply independent data splits.
"""
from __future__ import annotations
import argparse, importlib.util, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore', category=FutureWarning)

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cartographx_v7', HERE/'87_train_cartographx_v7.py')
if spec is None or spec.loader is None:
    raise RuntimeError('Script 87 must be in the same scripts directory')
v7=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)
OUT=v7.DATA/'cartographx_v10_models'
RAW=list(dict.fromkeys(v7.TAB+v7.GRAPH))
STRUCT=list(v7.STRUCT)
ALL=list(dict.fromkeys(RAW+STRUCT))
MODELS=['logistic_regression','random_forest','extra_trees','hist_gradient_boosting']
# Pre-registered ablations. Exclusion of distance features is a diagnostic
# hypothesis, not an assumption that all distance features are harmful.
DISTANCE_NAMES=['distance','edge_distance']

def is_distance(feature):
    return any(term in feature for term in DISTANCE_NAMES)

def get_feature_sets(train, source_validation, target_country, protocol):
    """Build feature sets; stability ranking uses source TRAIN only.

    Source validation is not used for ranking. Target data and target labels
    are never used to rank features. The extra args document that boundary.
    """
    del source_validation, target_country, protocol
    no_neighbor_distance=[f for f in ALL if f not in {
        'mean_neighbor_distance_m','nbr_mean_mean_neighbor_distance_m',
        'nbr_std_mean_neighbor_distance_m'}]
    no_raw_distance=[f for f in ALL if not (f in RAW and is_distance(f))]
    intersection_density=[f for f in ALL if (
        'intersect' in f or 'road_count' in f or 'neighbor_count' in f or
        f in ('graph_degree','graph_local_clustering','graph_is_isolated'))
        and not is_distance(f)]
    structural_only=list(STRUCT)
    # A source-only univariate stability proxy: predictive strength minus
    # missingness penalty, with no country-label comparison. Spatial CV below
    # makes the actual selection among representations.
    y=train.retained.to_numpy(dtype=int)
    ranks=[]
    for f in ALL:
        x=pd.to_numeric(train[f],errors='coerce').replace([np.inf,-np.inf],np.nan)
        mask=x.notna().to_numpy()
        auc=.5
        if mask.sum()>20 and x[mask].nunique()>1 and len(np.unique(y[mask]))==2:
            auc=float(roc_auc_score(y[mask],x[mask]))
        strength=abs(auc-.5)*2
        missing=float((~mask).mean())
        ranks.append({'feature':f,'source_train_auc':auc,'source_strength':strength,
                      'source_missing':missing,'rank_score':strength-0.2*missing})
    ranking=pd.DataFrame(ranks).sort_values(['rank_score','feature'],ascending=[False,True])
    source_stable=ranking.feature.head(12).tolist()
    sets={
        'V10_A_full_structural':ALL,
        'V10_B_no_neighbor_distance':no_neighbor_distance,
        'V10_C_no_raw_distance':no_raw_distance,
        'V10_D_intersection_density':intersection_density,
        'V10_E_structural_only':structural_only,
        'V10_F_source_top12':source_stable,
    }
    sets={k:list(dict.fromkeys(v)) for k,v in sets.items() if len(v)>=2}
    return sets,ranking

def select_by_spatial_cv(train,sets,seed,folds,fast=False):
    y=train.retained.to_numpy(dtype=int)
    groups=v7.spatial_groups(train)
    n=min(folds,len(np.unique(groups)))
    if n<2: raise RuntimeError('Need at least two spatial training groups')
    cv=list(GroupKFold(n_splits=n).split(train,y,groups))
    rows=[]
    for representation,features in sets.items():
        for name in MODELS:
            configurations=v7.CONFIGS[name][:1] if fast else v7.CONFIGS[name]
            for config_index,params in enumerate(configurations,1):
                oof=np.full(len(train),np.nan,dtype=float)
                for fold,(it,iv) in enumerate(cv,1):
                    if len(np.unique(y[it]))<2:
                        raise RuntimeError(f'Single-class spatial training fold {fold}')
                    model=v7.model_factory(name,params,seed+fold)
                    model.fit(train.iloc[it][features],y[it])
                    oof[iv]=v7.prob(model,train.iloc[iv][features])
                if not np.isfinite(oof).all(): raise RuntimeError('OOF prediction incomplete')
                threshold,score=v7.threshold_select(y,oof)
                rows.append(dict(representation=representation,model=name,
                                 config_index=config_index,config_json=json.dumps(params,sort_keys=True),
                                 n_features=len(features),cv_threshold=threshold,
                                 cv_mcc=score['mcc'],cv_ba=score['balanced_accuracy'],
                                 cv_f1=score['f1']))
                print(f'  {representation:31s} {name:23s} cfg={config_index} '
                      f'CV MCC={score["mcc"]:.4f}',flush=True)
    results=pd.DataFrame(rows).sort_values(
        ['cv_mcc','cv_ba','cv_f1','representation','model'],
        ascending=[False,False,False,True,True],kind='stable').reset_index(drop=True)
    return results

def run_one(nodes,protocol,seed,fast,folds):
    out=OUT/protocol/f'seed_{seed}'
    out.mkdir(parents=True,exist_ok=True)
    tr,va,te=v7.get_partitions(nodes,protocol)
    _,source_country,target_country=v7.PROTOCOLS[protocol]
    if protocol.endswith('zero_shot'):
        assert set(tr.country.unique())=={source_country}
        assert set(va.country.unique())=={source_country}
        assert set(te.country.unique())=={target_country}
    sets,ranking=get_feature_sets(tr,va,target_country,protocol)
    ranking.to_csv(out/'source_training_feature_ranking.csv',index=False)
    pd.DataFrame([{'representation':k,'features':';'.join(v),'n_features':len(v)}
                  for k,v in sets.items()]).to_csv(out/'feature_sets.csv',index=False)
    print(f'\n=== {protocol} seed={seed} train={len(tr)} source_val={len(va)} target_test={len(te)} ===',flush=True)
    cv=select_by_spatial_cv(tr,sets,seed,folds,fast)
    cv.to_csv(out/'spatial_cv_candidates.csv',index=False)
    best=cv.iloc[0]
    features=sets[best.representation]
    params=json.loads(best.config_json)
    model=v7.model_factory(best.model,params,seed)
    model.fit(tr[features],tr.retained.to_numpy(dtype=int))
    source_prob=v7.prob(model,va[features])
    threshold,source_metrics=v7.threshold_select(va.retained.to_numpy(dtype=int),source_prob)
    # Strictly after the selection and source threshold are frozen:
    test_prob=v7.prob(model,te[features])
    test_metrics=v7.metrics(te.retained.to_numpy(dtype=int),test_prob,threshold)
    pd.DataFrame({'sample_uid':te.sample_uid,'retained':te.retained,
                  'probability':test_prob,'prediction':(test_prob>=threshold).astype(int),
                  'threshold':threshold}).to_csv(out/'final_test_predictions.csv',index=False)
    result={'protocol':protocol,'seed':seed,'representation':best.representation,
            'model':best.model,'config_index':int(best.config_index),
            'features':';'.join(features),'n_features':len(features),
            'spatial_cv_mcc':float(best.cv_mcc),'source_val_mcc':source_metrics['mcc'],
            'source_val_ba':source_metrics['balanced_accuracy'],
            'threshold':threshold,'mcc':test_metrics['mcc'],
            'balanced_accuracy':test_metrics['balanced_accuracy'],
            'f1':test_metrics['f1'],'roc_auc':test_metrics['roc_auc'],
            'pr_auc':test_metrics['pr_auc'],
            'test_prevalence_DIAGNOSTIC_ONLY':float(te.retained.mean()),
            'predicted_positive_rate':float((test_prob>=threshold).mean())}
    pd.DataFrame([result]).to_csv(out/'selected_result.csv',index=False)
    (out/'selected_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f'[SELECTED] {best.representation} / {best.model} cfg={int(best.config_index)} '
          f'CV MCC={best.cv_mcc:.4f} source-val MCC={source_metrics["mcc"]:.4f} '
          f'threshold={threshold:.2f} TEST MCC={test_metrics["mcc"]:.4f} '
          f'BA={test_metrics["balanced_accuracy"]:.4f}',flush=True)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['development','full'],default='development')
    ap.add_argument('--protocols',nargs='+',choices=list(v7.PROTOCOLS))
    ap.add_argument('--seeds',nargs='+',type=int)
    ap.add_argument('--cv-folds',type=int,default=3)
    ap.add_argument('--fast',action='store_true',help='First configuration per model; smoke test only')
    args=ap.parse_args()
    if args.cv_folds<2: ap.error('--cv-folds must be >=2')
    protocols=args.protocols or (['lt_to_pl_zero_shot','pl_to_lt_zero_shot']
                                  if args.mode=='development' else list(v7.PROTOCOLS))
    seeds=args.seeds or ([42] if args.mode=='development' else list(range(42,52)))
    nodes,_=v7.load_data()
    nodes=v7.add_structural_features(nodes)
    OUT.mkdir(parents=True,exist_ok=True)
    collected=[]
    for protocol in protocols:
        for seed in seeds:
            collected.append(run_one(nodes,protocol,seed,args.fast,args.cv_folds))
            pd.DataFrame(collected).to_csv(OUT/'all_selected_results.csv',index=False)
    aggregate=pd.DataFrame(collected).groupby('protocol').agg(
        n_seeds=('seed','count'),mean_mcc=('mcc','mean'),sd_mcc=('mcc','std'),
        mean_ba=('balanced_accuracy','mean'),mean_roc_auc=('roc_auc','mean'),
        mean_pr_auc=('pr_auc','mean')).reset_index()
    aggregate.to_csv(OUT/'aggregate_metrics.csv',index=False)
    print('\n=== V10 SUMMARY (selected configurations only) ===')
    print(aggregate.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print(f'\n[PASS] Saved to {OUT}')
    print('NOTE: Prior V1-V9 test inspections make these existing test splits exploratory.')

if __name__=='__main__': main()
