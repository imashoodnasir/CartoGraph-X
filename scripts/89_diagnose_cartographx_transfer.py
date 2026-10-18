#!/usr/bin/env python
"""CartoGraph-X V9: descriptive cross-country audit; never use target labels for tuning.
Run: python scripts/89_diagnose_cartographx_transfer.py
Optional: --v8-dir data/processed/cross_country/cartographx_v8_models/full
"""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import roc_auc_score, matthews_corrcoef, balanced_accuracy_score

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cartographx_v7', HERE/'87_train_cartographx_v7.py')
v7=importlib.util.module_from_spec(spec); spec.loader.exec_module(v7)


def numeric(s):
    return pd.to_numeric(s,errors='coerce').replace([np.inf,-np.inf],np.nan)

def auc_univariate(x,y):
    mask=x.notna() & y.notna()
    if mask.sum()<20 or x[mask].nunique()<2 or y[mask].nunique()<2: return np.nan
    return float(roc_auc_score(y[mask].astype(int),x[mask]))

def prevalence_audit(nodes,out):
    rows=[]
    for (country,split),g in nodes.groupby(['country','split'],dropna=False):
        y=numeric(g['retained'])
        rows.append(dict(country=country,split=split,n=len(g),n_labeled=int(y.notna().sum()),
                         retained_rate=float(y.mean()),retained_n=int(y.sum()),
                         tiles=int(g['tile_id'].nunique())))
    pd.DataFrame(rows).to_csv(out/'label_prevalence_by_split.csv',index=False)

def feature_audit(nodes,features,out):
    # Only train/validation labels for label-association analysis. Test labels are
    # deliberately excluded from the feature ranking table.
    rows=[]
    for f in features:
        for split in ['train','validation','test']:
            a=nodes[(nodes.country=='Lithuania')&(nodes.split==split)]
            b=nodes[(nodes.country=='Poland')&(nodes.split==split)]
            if len(a)==0 or len(b)==0: continue
            xa,xb=numeric(a[f]),numeric(b[f]); va,vb=xa.dropna(),xb.dropna()
            pooled=np.r_[va.to_numpy(),vb.to_numpy()]
            scale=float(np.subtract(*np.nanpercentile(pooled,[75,25]))) if len(pooled) else np.nan
            shift=(float(vb.median()-va.median())/scale) if np.isfinite(scale) and scale>0 else np.nan
            ks=float(ks_2samp(va,vb).statistic) if len(va)>1 and len(vb)>1 else np.nan
            row=dict(feature=f,split=split,lt_n=len(a),pl_n=len(b),
                     lt_missing=float(xa.isna().mean()),pl_missing=float(xb.isna().mean()),
                     lt_median=float(va.median()) if len(va) else np.nan,
                     pl_median=float(vb.median()) if len(vb) else np.nan,
                     standardized_median_shift=shift,ks_statistic=ks)
            if split!='test':
                al=auc_univariate(xa,numeric(a.retained)); bl=auc_univariate(xb,numeric(b.retained))
                row.update(lt_label_auc=al,pl_label_auc=bl,
                           auc_direction_disagreement=bool((al-.5)*(bl-.5)<0) if np.isfinite(al) and np.isfinite(bl) else np.nan,
                           label_auc_difference=abs(al-bl) if np.isfinite(al) and np.isfinite(bl) else np.nan)
            rows.append(row)
    pd.DataFrame(rows).to_csv(out/'feature_shift_and_label_association.csv',index=False)

def prediction_audit(v8_dir,out):
    rows=[]; grid=np.round(np.arange(.05,.951,.01),2)
    for protocol in ['lt_to_pl_zero_shot','pl_to_lt_zero_shot','lt_to_lt','pl_to_pl','joint_to_lt','joint_to_pl']:
        for seed in range(42,52):
            d=v8_dir/protocol/f'seed_{seed}'
            p=d/'final_test_predictions.csv'; r=d/'selected_result.csv'
            if not (p.exists() and r.exists()): continue
            pred=pd.read_csv(p); meta=pd.read_csv(r).iloc[0]
            y=pred.retained.to_numpy(dtype=int); score=pred.probability.to_numpy(dtype=float)
            chosen=float(meta.threshold); selected=(score>=chosen).astype(int)
            # Oracle thresholds use TEST LABELS and are for retrospective diagnosis only.
            oracle_scores=np.array([matthews_corrcoef(y,(score>=t).astype(int)) for t in grid])
            best=int(np.argmax(oracle_scores)); oracle=float(oracle_scores[best])
            rows.append(dict(protocol=protocol,seed=seed,n=len(y),target_prevalence=float(y.mean()),
                             selected_threshold=chosen,selected_mcc=float(matthews_corrcoef(y,selected)),
                             selected_ba=float(balanced_accuracy_score(y,selected)),
                             predicted_positive_rate=float(selected.mean()),mean_probability=float(score.mean()),
                             oracle_test_threshold_DIAGNOSTIC_ONLY=float(grid[best]),
                             oracle_test_mcc_DIAGNOSTIC_ONLY=oracle,
                             oracle_gap_DIAGNOSTIC_ONLY=float(oracle-matthews_corrcoef(y,selected))))
    if not rows:
        print('[INFO] No V8 prediction files found; skipping threshold audit')
        return
    df=pd.DataFrame(rows); df.to_csv(out/'threshold_transfer_diagnostic_ONLY.csv',index=False)
    agg=df.groupby('protocol',as_index=False).agg(n_seeds=('seed','size'),
             selected_mcc=('selected_mcc','mean'),oracle_mcc_DIAGNOSTIC_ONLY=('oracle_test_mcc_DIAGNOSTIC_ONLY','mean'),
             oracle_gap_DIAGNOSTIC_ONLY=('oracle_gap_DIAGNOSTIC_ONLY','mean'),
             mean_predicted_positive_rate=('predicted_positive_rate','mean'),
             target_prevalence=('target_prevalence','mean'))
    agg.to_csv(out/'threshold_diagnostic_summary_ONLY.csv',index=False)
    print('\nTHRESHOLD DIAGNOSTIC (test-label oracle is NOT deployable):')
    print(agg.to_string(index=False,float_format=lambda x:f'{x:.4f}'))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default=None)
    ap.add_argument('--v8-dir',default=None)
    args=ap.parse_args()
    out=Path(args.out) if args.out else v7.DATA/'cartographx_v9_diagnostics'
    out.mkdir(parents=True,exist_ok=True)
    nodes,_=v7.load_data(); nodes=v7.add_structural_features(nodes)
    features=list(dict.fromkeys(v7.TAB+v7.GRAPH+v7.STRUCT))
    prevalence_audit(nodes,out)
    feature_audit(nodes,features,out)
    v8_dir=Path(args.v8_dir) if args.v8_dir else v7.DATA/'cartographx_v8_models'/'full'
    prediction_audit(v8_dir,out)
    (out/'READ_ME.txt').write_text('DIAGNOSTIC ONLY. Never use test-label oracle thresholds, target-label associations, or observed test gains to select or tune a future model. Label semantic equivalence requires provenance/source-map inspection.\n',encoding='utf-8')
    print(f'\n[PASS] V9 diagnostic reports saved to {out}')
    print('Feature label associations exclude test labels; oracle thresholds are explicitly retrospective only.')
if __name__=='__main__': main()
