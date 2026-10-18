#!/usr/bin/env python3
"""Script 116: locked CartoGraph-X V2 architecture ablations, TRAIN/VALIDATION ONLY.

Five one-factor removals, 10 matched seeds, six frozen protocols. Never reads
historical TEST labels or scores. Reference metrics read from Script 114/112
validation outputs. No test-guided selection or modification of Script 112.

Usage:
  python scripts/116_cartographx_v2_architecture_ablations.py
  python scripts/116_cartographx_v2_architecture_ablations.py --seeds 42
  python scripts/116_cartographx_v2_architecture_ablations.py --force

Requires Scripts 111 and 112 alongside this script, and existing Script 114
seed-level results. Training-only spatial OOF scores are diagnostic; validation
thresholds are calibrated using only the permitted validation country.
"""
from __future__ import annotations
import argparse, importlib.util, json, hashlib, random
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold

HERE=Path(__file__).resolve().parent
P111=HERE/'111_train_benchmark_v2_weighted_gnn_baselines.py'
P112=HERE/'112_train_cartographx_v2_architecture.py'
OUT=HERE.parent/'data/processed/cross_country/cartographx_v2_ablations_v1'
REFERENCE=HERE.parent/'data/processed/cross_country/cartographx_v2_architecture_v1'
VARIANTS=('no_graph_message_passing','no_gated_fusion','no_residual_connections',
          'no_country_conditioned_heads','no_class_balanced_loss')
METRICS=('mcc','balanced_accuracy','f1','roc_auc','pr_auc')

def load_module(path,name):
    if not path.is_file():raise FileNotFoundError(path)
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod
base=load_module(P111,'cartographx_ablation_dependency111')
ref=load_module(P112,'cartographx_ablation_dependency112')

class AblatedResidualSAGE(ref.ResidualSAGE):
    def __init__(self,n_features,hidden,dropout,joint,variant):
        super().__init__(n_features,hidden,dropout,joint and variant!='no_country_conditioned_heads')
        self.variant=variant
    def forward(self,x,edge,country):
        z=self.project(x)
        if self.variant=='no_graph_message_passing':
            h=z  # No neighborhood aggregation; keeps original primitive-feature path.
        elif self.variant=='no_residual_connections':
            h=self.norm1(F.dropout(F.relu(self.sage1(z,edge)),p=self.dropout,training=self.training))
            h=self.norm2(F.dropout(F.relu(self.sage2(h,edge)),p=self.dropout,training=self.training))
        else:
            h=self.norm1(z+F.dropout(F.relu(self.sage1(z,edge)),p=self.dropout,training=self.training))
            h=self.norm2(h+F.dropout(F.relu(self.sage2(h,edge)),p=self.dropout,training=self.training))
        if self.variant=='no_gated_fusion':
            fused=h
        else:
            gate=torch.sigmoid(self.gate(torch.cat([z,h],dim=1)))
            fused=gate*h+(1-gate)*z
        logits=self.shared(fused).flatten()
        if self.joint:
            logits=logits+(fused*self.country_delta(country)).sum(dim=1)/(fused.shape[1]**.5)
            logits=logits+self.country_bias(country).flatten()
        return logits

def train_predict(graph,train_ids,prediction_sets,args,seed,joint,variant):
    base.seed_all(seed)
    (x,y,e),imp,sc=graph.subset(train_ids,fit=True,device=args.device)
    if len(torch.unique(y))!=2:raise RuntimeError('One-class training fold')
    c=ref.country_tensor(graph,train_ids,args.device)
    net=AblatedResidualSAGE(x.shape[1],args.hidden,args.dropout,joint,variant).to(args.device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    weights=torch.ones_like(y)
    if variant!='no_class_balanced_loss':
        for k in torch.unique(c):
            mask=c==k
            p=float(y[mask].sum().item());n=int(mask.sum().item())
            if p==0 or p==n:raise RuntimeError('One-class country in training fold')
            weights[mask]=torch.where(y[mask]>0.5,torch.tensor(n/(2*p),device=y.device),
                                      torch.tensor(n/(2*(n-p)),device=y.device))
    for epoch in range(args.epochs):
        net.train();optimizer.zero_grad(set_to_none=True)
        logits=net(x,e,c)
        loss=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*weights).mean()
        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss epoch {epoch}')
        loss.backward();optimizer.step()
    net.eval();outputs=[]
    with torch.no_grad():
        for ids in prediction_sets:
            (px,_,pe),_,_=graph.subset(ids,imputer=imp,scaler=sc,device=args.device)
            pc=ref.country_tensor(graph,ids,args.device)
            outputs.append(torch.sigmoid(net(px,pe,pc)).cpu().numpy())
    return outputs

def hash_file(p):return base.digest(p)
def write_csv(df,path):
    tmp=path.with_suffix('.tmp')
    df.to_csv(tmp,index=False);tmp.replace(path)

def run_seed(seed,graph,df,args,run_dir):
    rows=[];cvrows=[];cache={}
    for variant in VARIANTS:
        for protocol,(source,target,joint) in base.PROTOCOLS.items():
            train_ids=np.flatnonzero((df.split=='train') & (df.country.isin(['Lithuania','Poland']) if joint else df.country.eq(source)))
            val_country=target if joint or source==target else source
            val_ids=np.flatnonzero((df.split=='validation') & df.country.eq(val_country))
            if not len(train_ids) or not len(val_ids):raise RuntimeError('Empty train/validation partition')
            key=(variant,'joint' if joint else source)
            if key not in cache:
                groups=(df.iloc[train_ids].country.astype(str)+'::'+df.iloc[train_ids].tile_id.astype(str)).to_numpy()
                y=graph.labels[train_ids]
                cv=StratifiedGroupKFold(n_splits=args.folds,shuffle=True,random_state=seed)
                oof=np.full(len(train_ids),np.nan)
                for fold,(tr,va) in enumerate(cv.split(train_ids,y,groups)):
                    if set(groups[tr])&set(groups[va]):raise RuntimeError('Spatial CV overlap')
                    if len(np.unique(y[tr]))!=2 or len(np.unique(y[va]))!=2:raise RuntimeError('One-class fold')
                    if joint:
                        for country in ('Lithuania','Poland'):
                            yc=y[tr][df.iloc[train_ids[tr]].country.to_numpy()==country]
                            if len(np.unique(yc))!=2:raise RuntimeError('One-class country fold')
                    (p,)=train_predict(graph,train_ids[tr],[train_ids[va]],args,seed+1009*fold,joint,variant)
                    oof[va]=p
                if not np.isfinite(oof).all():raise RuntimeError('Missing OOF scores')
                t=base.threshold_for(y,oof)
                pooled=base.metrics(y,oof,t)
                by_country={}
                for country in (('Lithuania','Poland') if joint else (source,)):
                    mask=df.iloc[train_ids].country.to_numpy()==country
                    by_country[country]=base.metrics(y[mask],oof[mask],t)
                cache[key]=(t,pooled,by_country)
            oof_t,pooled,by_country=cache[key]
            for country,m in by_country.items():
                cvrows.append({'seed':seed,'variant':variant,'protocol':protocol,'country':country,
                               'cv_mcc':m['mcc'],'cv_roc_auc':m['roc_auc'],
                               'pooled_cv_mcc':pooled['mcc'],'cv_oof_threshold':oof_t})
            (val_prob,)=train_predict(graph,train_ids,[val_ids],args,seed+99173,joint,variant)
            threshold=base.threshold_for(graph.labels[val_ids],val_prob)
            vm=base.metrics(graph.labels[val_ids],val_prob,threshold)
            row={'seed':seed,'variant':variant,'protocol':protocol,'source':source,'target':target,
                 'joint_training':joint,'threshold_selection_country':val_country,
                 'train_n':len(train_ids),'validation_n':len(val_ids),
                 'cv_mcc':pooled['mcc'],'cv_oof_threshold':oof_t,'threshold':threshold}
            row.update({f'validation_{k}':v for k,v in vm.items()})
            rows.append(row)
            print(f'  {seed} {variant:30s} {protocol:22s} VAL_MCC={vm["mcc"]:.4f} t={threshold:.2f}',flush=True)
    write_csv(pd.DataFrame(rows),run_dir/'ablation_validation_results.csv')
    write_csv(pd.DataFrame(cvrows),run_dir/'ablation_country_cv_results.csv')
    manifest={'script':Path(__file__).name,'seed':seed,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
              'epochs':args.epochs,'folds':args.folds,'hidden':args.hidden,'dropout':args.dropout,
              'lr':args.lr,'weight_decay':args.weight_decay,'device':args.device,
              'variants':list(VARIANTS),'protocols':list(base.PROTOCOLS),
              'script_sha256':hash_file(Path(__file__)),'script111_sha256':hash_file(P111),
              'script112_sha256':hash_file(P112),'node_sha256':hash_file(base.DATA),
              'edge_sha256':hash_file(base.EDGES),
              'outputs_sha256':{n:hash_file(run_dir/n) for n in ('ablation_validation_results.csv','ablation_country_cv_results.csv')},
              'limitations':['Training and validation only; historical test deliberately not accessed',
                             'C2 labels are provisional','One-factor removals are not parameter-count matched',
                             'Validation MCC uses validation-tuned threshold and is optimistic']}
    (run_dir/'ablation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

def valid_seed(seed,args):
    d=OUT/f'seed_{seed}';mp=d/'ablation_manifest.json'
    if not mp.exists():return False
    try:
        m=json.loads(mp.read_text(encoding='utf-8'))
        if m['seed']!=seed or m['variants']!=list(VARIANTS):return False
        for k,v in [('epochs',args.epochs),('folds',args.folds),('hidden',args.hidden),
                    ('dropout',args.dropout),('lr',args.lr),('weight_decay',args.weight_decay)]:
            if m[k]!=v:return False
        for k,p in [('script_sha256',Path(__file__)),('script111_sha256',P111),
                    ('script112_sha256',P112),('node_sha256',base.DATA),('edge_sha256',base.EDGES)]:
            if m[k]!=hash_file(p):return False
        for n,h in m['outputs_sha256'].items():
            if not (d/n).exists() or hash_file(d/n)!=h:return False
        r=pd.read_csv(d/'ablation_validation_results.csv')
        c=pd.read_csv(d/'ablation_country_cv_results.csv')
        return len(r)==len(VARIANTS)*len(base.PROTOCOLS) and not r.duplicated(['variant','protocol']).any() and len(c)==len(VARIANTS)*8
    except Exception:return False

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds',type=int,nargs='+',default=list(range(42,52)))
    ap.add_argument('--epochs',type=int,default=60)
    ap.add_argument('--folds',type=int,default=3)
    ap.add_argument('--hidden',type=int,default=64)
    ap.add_argument('--dropout',type=float,default=.2)
    ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--weight-decay',type=float,default=1e-4)
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--force',action='store_true')
    args=ap.parse_args()
    if len(set(args.seeds))!=len(args.seeds) or not args.seeds:ap.error('Seeds must be unique and nonempty')
    if args.epochs<1 or args.folds<2 or args.hidden<1:ap.error('Invalid training configuration')
    if args.device=='auto':args.device='cuda' if torch.cuda.is_available() else 'cpu'
    if args.device=='cuda' and not torch.cuda.is_available():ap.error('CUDA unavailable')
    # Reference experiment used exactly these values; prohibit silent budget drift.
    for seed in args.seeds:
        mpath=REFERENCE/f'seed_{seed}'/'experiment_manifest.json'
        rpath=REFERENCE/f'seed_{seed}'/'selected_protocol_results.csv'
        if not mpath.exists() or not rpath.exists():raise FileNotFoundError(f'Missing frozen reference seed {seed}: {mpath}')
        m=json.loads(mpath.read_text(encoding='utf-8'))
        for k in ('epochs','folds','hidden','dropout','lr','weight_decay'):
            if m[k]!=getattr(args,k):raise RuntimeError(f'Reference seed {seed} {k}={m[k]} differs from ablation {getattr(args,k)}')
        if m['node_sha256']!=hash_file(base.DATA) or m['edge_sha256']!=hash_file(base.EDGES):
            raise RuntimeError(f'Reference seed {seed} input hashes differ')
        if m['outputs_sha256']['selected_protocol_results.csv']!=hash_file(rpath):
            raise RuntimeError(f'Reference seed {seed} result hash mismatch')
    df=pd.read_csv(base.DATA,low_memory=False)
    edges=pd.read_csv(base.EDGES,low_memory=False)
    required={'sample_uid','country','city','tile_id','split','retained'}|set(base.FEATURES)
    if required-set(df):raise RuntimeError(f'Missing node columns: {sorted(required-set(df))}')
    if df.sample_uid.isna().any() or df.sample_uid.duplicated().any():raise RuntimeError('Invalid UIDs')
    if not df.retained.isin([0,1]).all():raise RuntimeError('Invalid labels')
    if (df.groupby(['country','tile_id']).split.nunique()>1).any():raise RuntimeError('Tile split overlap')
    for c in base.FEATURES:df[c]=pd.to_numeric(df[c],errors='raise')
    # IMPORTANT: do not read or use test labels in any training/validation operation.
    # Exclude test rows and edges altogether from the ablation graph.
    df=df[df.split.isin(['train','validation'])].copy().reset_index(drop=True)
    valid=set(df.sample_uid.astype(str))
    edges=edges[edges.source_uid.astype(str).isin(valid)&edges.target_uid.astype(str).isin(valid)].copy()
    graph=base.GraphData(df,edges)
    OUT.mkdir(parents=True,exist_ok=True)
    for seed in args.seeds:
        d=OUT/f'seed_{seed}';d.mkdir(parents=True,exist_ok=True)
        if not args.force and valid_seed(seed,args):
            print(f'VERIFIED existing seed {seed}; skipping',flush=True)
        else:
            print(f'\n=== ABLATIONS seed {seed} ===',flush=True)
            run_seed(seed,graph,df,args,d)
            if not valid_seed(seed,args):raise RuntimeError(f'Seed {seed} output verification failed')
            print(f'VERIFIED seed {seed}',flush=True)
    runs=pd.concat([pd.read_csv(OUT/f'seed_{s}'/'ablation_validation_results.csv') for s in args.seeds],ignore_index=True)
    cvs=pd.concat([pd.read_csv(OUT/f'seed_{s}'/'ablation_country_cv_results.csv') for s in args.seeds],ignore_index=True)
    refs=[]
    for seed in args.seeds:
        r=pd.read_csv(REFERENCE/f'seed_{seed}'/'selected_protocol_results.csv')
        if set(r.protocol)!=set(base.PROTOCOLS) or len(r)!=6:raise RuntimeError(f'Invalid reference seed {seed}')
        refs.append(r)
    references=pd.concat(refs,ignore_index=True)
    keep=['seed','protocol']+[f'validation_{m}' for m in METRICS]
    references=references[keep].rename(columns={f'validation_{m}':f'reference_validation_{m}' for m in METRICS})
    paired=runs.merge(references,on=['seed','protocol'],how='left',validate='many_to_one')
    if paired[[f'reference_validation_{m}' for m in METRICS]].isna().any().any():raise RuntimeError('Missing reference validation metrics')
    for metric in METRICS:
        paired[f'delta_validation_{metric}']=paired[f'validation_{metric}']-paired[f'reference_validation_{metric}']
    summary=paired.groupby(['variant','protocol'],as_index=False).agg(
        n_seeds=('seed','size'),ablation_validation_mcc_mean=('validation_mcc','mean'),
        ablation_validation_mcc_std=('validation_mcc','std'),
        reference_validation_mcc_mean=('reference_validation_mcc','mean'),
        delta_validation_mcc_mean=('delta_validation_mcc','mean'),
        delta_validation_mcc_std=('delta_validation_mcc','std'),
        delta_validation_mcc_min=('delta_validation_mcc','min'),
        delta_validation_mcc_max=('delta_validation_mcc','max'))
    country_summary=cvs.groupby(['variant','protocol','country'],as_index=False).agg(
        n_seeds=('seed','size'),cv_mcc_mean=('cv_mcc','mean'),cv_mcc_std=('cv_mcc','std'),
        cv_roc_auc_mean=('cv_roc_auc','mean'))
    files={'ablation_validation_all_seeds.csv':runs,'ablation_country_cv_all_seeds.csv':cvs,
           'ablation_matched_reference_differences.csv':paired,
           'table_ablation_validation_mcc_summary.csv':summary,
           'table_ablation_country_cv_summary.csv':country_summary}
    for name,frame in files.items():write_csv(frame,OUT/name)
    report={'script':'116_cartographx_v2_architecture_ablations.py','timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'seeds':args.seeds,'variants':list(VARIANTS),'protocols':list(base.PROTOCOLS),
            'script_sha256':hash_file(Path(__file__)),'reference_script_sha256':hash_file(P112),
            'node_sha256':hash_file(base.DATA),'edge_sha256':hash_file(base.EDGES),
            'outputs_sha256':{n:hash_file(OUT/n) for n in files},
            'evaluation_scope':'TRAIN/VALIDATION ONLY; no historical test rows, labels or predictions loaded',
            'caveats':['C2 labels provisional','Validation threshold tuned on same validation set',
                       'Ablations differ in effective parameter count',
                       'No confirmatory geographic generalization claims']}
    (OUT/'ablation_summary_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nSCRIPT 116: VALIDATION MCC DELTA (ABLATION MINUS FROZEN REFERENCE)',flush=True)
    print(summary[['variant','protocol','n_seeds','delta_validation_mcc_mean','delta_validation_mcc_std']].to_string(index=False,float_format=lambda v:f'{v:.4f}'),flush=True)
    print(f'\nSaved: {OUT}\nNO TEST EVALUATION PERFORMED. C2 labels remain provisional.',flush=True)

if __name__=='__main__':main()
