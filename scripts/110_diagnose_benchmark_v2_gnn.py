#!/usr/bin/env python3
"""Script 110: leakage-aware GNN diagnostics and train/validation-only sensitivity.

Run: python scripts/110_diagnose_benchmark_v2_gnn.py --seed 42
Optional: --epochs 60 --device auto --skip-training
Uses Script 109 as an imported module; does not modify Script 109 or use test
labels for any diagnostic, selection, or optimization. Existing test predictions
are read ONLY to summarize frozen historical results, not to tune parameters.
Graph edges are geometric kNN, not actual road topology.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[1]
SCRIPT109=ROOT/'scripts/109_train_benchmark_v2_gnn_baselines.py'
GNNDIR=ROOT/'data/processed/cross_country/benchmark_v2_gnn_baselines_v1'
OUT=ROOT/'data/processed/cross_country/benchmark_v2_gnn_diagnostics_v1'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def load109():
    if not SCRIPT109.exists():raise FileNotFoundError(f'Script 109 not found: {SCRIPT109}')
    spec=importlib.util.spec_from_file_location('benchmark_gnn109',SCRIPT109)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m

def group_stats(df):
    rows=[]
    for (country,split),g in df.groupby(['country','split']):
        rows.append({'country':country,'split':split,'n':len(g),'positive_n':int(g.retained.sum()),
                     'positive_rate':float(g.retained.mean()),'n_tiles':g.tile_id.nunique(),
                     'majority_class_accuracy':float(max(g.retained.mean(),1-g.retained.mean()))})
    return pd.DataFrame(rows)

def graph_stats(df,edges):
    lookup=pd.Series(df.retained.to_numpy(dtype=np.int8),index=df.sample_uid.astype(str))
    meta=df.set_index(df.sample_uid.astype(str))[['country','split','tile_id']]
    e=edges[['source_uid','target_uid']].copy()
    e['source_uid']=e.source_uid.astype(str);e['target_uid']=e.target_uid.astype(str)
    if e.source_uid.map(lookup).isna().any() or e.target_uid.map(lookup).isna().any():
        raise RuntimeError('Edges reference unknown node IDs')
    e['a']=e.source_uid.map(lookup).astype(int);e['b']=e.target_uid.map(lookup).astype(int)
    e['country']=e.source_uid.map(meta.country);e['split']=e.source_uid.map(meta.split)
    if (e.country.to_numpy()!=e.target_uid.map(meta.country).to_numpy()).any() or (e.split.to_numpy()!=e.target_uid.map(meta.split).to_numpy()).any():
        raise RuntimeError('Cross-country or cross-split graph edges detected')
    if (e.source_uid.map(meta.tile_id).to_numpy()!=e.target_uid.map(meta.tile_id).to_numpy()).any():
        raise RuntimeError('Cross-tile graph edges detected')
    rows=[]
    for (country,split),g in e.groupby(['country','split']):
        p=float(df.loc[(df.country==country)&(df.split==split),'retained'].mean())
        observed=float((g.a==g.b).mean()); expected=p*p+(1-p)*(1-p)
        pp=int(((g.a==1)&(g.b==1)).sum());nn=int(((g.a==0)&(g.b==0)).sum())
        pn=int(len(g)-pp-nn)
        rows.append({'country':country,'split':split,'edges_undirected':len(g),
                     'observed_same_label_fraction':observed,'independent_label_expected_fraction':expected,
                     'homophily_excess':observed-expected,'positive_positive_edges':pp,
                     'negative_negative_edges':nn,'mixed_label_edges':pn})
    return pd.DataFrame(rows)

def baseline_report(m,seed):
    path=GNNDIR/f'seed_{seed}'
    r=pd.read_csv(path/'selected_protocol_results.csv')
    cv=pd.read_csv(path/'spatial_cv_candidate_results.csv')
    for needed in ['protocol','model','threshold','test_mcc','validation_mcc']:
        if needed not in r.columns:raise RuntimeError(f'Missing result column: {needed}')
    return r,cv

def threshold_table(y,p,country,variant,seed):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    rows=[]
    for t in np.round(np.arange(.05,.951,.01),2):
        pred=(p>=t).astype(int)
        rows.append({'seed':seed,'country':country,'variant':variant,'threshold':float(t),
                     'validation_mcc':float(matthews_corrcoef(y,pred)),
                     'predicted_positive_rate':float(pred.mean()),'observed_positive_rate':float(y.mean()),
                     'tp':int(((y==1)&(pred==1)).sum()),'fp':int(((y==0)&(pred==1)).sum()),
                     'fn':int(((y==1)&(pred==0)).sum()),'tn':int(((y==0)&(pred==0)).sum())})
    return rows

def train_variant(m,graph,train_ids,val_ids,seed,args,variant,tag):
    m.seed_all(seed)
    (x,y,e),imp,scaler=graph.subset(train_ids,fit=True,device=args.device)
    (vx,vy,ve),_,_=graph.subset(val_ids,imputer=imp,scaler=scaler,device=args.device)
    if y.min()==y.max() or vy.min()==vy.max():raise RuntimeError(f'One-class train/validation: {tag}')
    net=m.Net(x.shape[1],args.hidden,'GraphSAGE',args.dropout).to(args.device)
    optim=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    p=float(y.mean().item()); weight=(1-p)/max(p,1e-9)
    pos_weight=torch.tensor([weight],dtype=torch.float32,device=args.device) if variant=='weighted' else None
    history=[]
    for epoch in range(1,args.epochs+1):
        net.train();optim.zero_grad(set_to_none=True)
        logits=net(x,e)
        loss=F.binary_cross_entropy_with_logits(logits,y,pos_weight=pos_weight)
        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite training loss: {tag}, epoch {epoch}')
        loss.backward();optim.step()
        if epoch==1 or epoch%args.log_every==0 or epoch==args.epochs:
            net.eval()
            with torch.no_grad():
                tl=F.binary_cross_entropy_with_logits(net(x,e),y).item()
                v_logits=net(vx,ve)
                vl=F.binary_cross_entropy_with_logits(v_logits,vy).item()
                prob=torch.sigmoid(v_logits).cpu().numpy()
                auc=float(roc_auc_score(vy.cpu().numpy(),prob))
                pred=(prob>=.5).astype(int)
                mcc=float(matthews_corrcoef(vy.cpu().numpy(),pred))
            history.append({'country':tag,'variant':variant,'seed':seed,'epoch':epoch,
                            'train_unweighted_bce':float(tl),'validation_unweighted_bce':float(vl),
                            'validation_auc':auc,'validation_mcc_at_0_5':mcc,
                            'validation_predicted_positive_rate_at_0_5':float(pred.mean()),
                            'train_positive_rate':p,'training_pos_weight':float(weight if pos_weight is not None else 1.)})
    net.eval()
    with torch.no_grad():probs=torch.sigmoid(net(vx,ve)).cpu().numpy()
    return history,threshold_table(vy.cpu().numpy(),probs,tag,variant,seed),{
        'country':tag,'variant':variant,'seed':seed,'train_n':len(train_ids),'validation_n':len(val_ids),
        'validation_auc':float(roc_auc_score(vy.cpu().numpy(),probs)),
        'validation_pr_auc':float(average_precision_score(vy.cpu().numpy(),probs)),
        'validation_probability_min':float(np.min(probs)),'validation_probability_p10':float(np.quantile(probs,.1)),
        'validation_probability_median':float(np.median(probs)),
        'validation_probability_p90':float(np.quantile(probs,.9)),
        'validation_probability_max':float(np.max(probs)),
        'validation_predicted_positive_rate_at_0_5':float(np.mean(probs>=.5)),
        'validation_mcc_at_0_5':float(matthews_corrcoef(vy.cpu().numpy(),probs>=.5)),
        'validation_observed_positive_rate':float(vy.mean().item())}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--hidden',type=int,default=64);parser.add_argument('--dropout',type=float,default=.2)
    parser.add_argument('--lr',type=float,default=.001);parser.add_argument('--weight-decay',type=float,default=1e-4)
    parser.add_argument('--log-every',type=int,default=5)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--skip-training',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=None)
    args=parser.parse_args()
    if args.epochs<1 or args.log_every<1:parser.error('epochs and log-every must be positive')
    args.device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if args.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA is unavailable')
    m=load109()
    for p in [m.DATA,m.EDGES]:
        if not p.exists():raise FileNotFoundError(p)
    df=pd.read_csv(m.DATA,low_memory=False)
    edges=pd.read_csv(m.EDGES,low_memory=False)
    if df.sample_uid.isna().any() or df.sample_uid.duplicated().any():raise RuntimeError('Duplicate/missing sample UID')
    if not df.retained.isin([0,1]).all():raise RuntimeError('Invalid binary label')
    if (df.groupby(['country','tile_id']).split.nunique()>1).any():raise RuntimeError('Spatial tile spans splits')
    output=args.output_dir or OUT/f'seed_{args.seed}'
    output.mkdir(parents=True,exist_ok=True)
    class_stats=group_stats(df)
    homophily=graph_stats(df,edges)
    original,cv=baseline_report(m,args.seed)
    # Original results are read for documentation only. No test labels/predictions are used in any training decision.
    original[['protocol','model','cv_mcc_at_0_5','validation_mcc','test_mcc','threshold']].to_csv(output/'original_script109_reference.csv',index=False)
    cv.to_csv(output/'original_script109_cv_candidates.csv',index=False)
    class_stats.to_csv(output/'class_distribution_by_country_split.csv',index=False)
    homophily.to_csv(output/'geometric_graph_label_homophily.csv',index=False)
    print('\nClass distribution:\n'+class_stats.to_string(index=False),flush=True)
    print('\nGeometric graph label homophily:\n'+homophily.to_string(index=False),flush=True)
    histories=[];thresholds=[];summaries=[]
    if not args.skip_training:
        graph=m.GraphData(df,edges)
        for country in ['Lithuania','Poland']:
            train_ids=np.flatnonzero((df.country==country)&(df.split=='train'))
            val_ids=np.flatnonzero((df.country==country)&(df.split=='validation'))
            for variant in ['unweighted','weighted']:
                print(f'\nTraining diagnostic {country} / {variant} / {args.epochs} epochs',flush=True)
                h,t,s=train_variant(m,graph,train_ids,val_ids,args.seed,args,variant,country)
                histories.extend(h);thresholds.extend(t);summaries.append(s)
                tt=pd.DataFrame(t).sort_values(['validation_mcc','threshold'],ascending=[False,True]).iloc[0]
                print(f'  val AUC={s["validation_auc"]:.4f} MCC@0.5={s["validation_mcc_at_0_5"]:.4f} '
                      f'best validation MCC={tt.validation_mcc:.4f} at t={tt.threshold:.2f}; '
                      f'positive predictions@0.5={s["validation_predicted_positive_rate_at_0_5"]:.3f}',flush=True)
        pd.DataFrame(histories).to_csv(output/'training_validation_loss_curves.csv',index=False)
        pd.DataFrame(thresholds).to_csv(output/'validation_threshold_sensitivity.csv',index=False)
        pd.DataFrame(summaries).to_csv(output/'validation_probability_diagnostics.csv',index=False)
    manifest={'script':Path(__file__).name,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
              'seed':args.seed,'epochs':args.epochs,'device':args.device,'skip_training':args.skip_training,
              'training_variants':['unweighted','weighted'] if not args.skip_training else [],
              'training_scope':'country-specific train only; diagnostics on country-specific validation only',
              'model':'GraphSAGE from frozen Script 109; no modifications to Script 109',
              'selection_policy':'diagnostic only; no new model chosen for test evaluation',
              'test_policy':'previous historical test scores copied as reference only; no test labels used for fitting, tuning, or diagnostics',
              'limitations':['manual C2 label review incomplete','historical test previously examined',
                             'geometric proximity kNN edges are not road topology',
                             'training variants are exploratory; repeated validation inspection can overfit validation'],
              'inputs_sha256':{'nodes':sha(m.DATA),'edges':sha(m.EDGES),'script109':sha(SCRIPT109)},
              'outputs_sha256':{p.name:sha(p) for p in sorted(output.glob('*.csv'))}}
    (output/'diagnostic_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'\nSCRIPT 110 COMPLETED: {output}\nNo new test-set model evaluations performed.',flush=True)

if __name__=='__main__':main()
