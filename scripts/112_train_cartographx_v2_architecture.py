#!/usr/bin/env python3
"""CartoGraph-X V2 architecture prototype, six leakage-conscious protocols.

Uses Script 111's graph construction and primitive road features. Geometric kNN
edges are NOT road-network topology. No held-out fold edges in training.
The architecture is fixed before evaluating the historical (exploratory) test.

Usage: python scripts/112_train_cartographx_v2_architecture.py --seed 42 --epochs 60
Requires scripts/111_train_benchmark_v2_weighted_gnn_baselines.py beside this file.
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
PARENT=HERE/'111_train_benchmark_v2_weighted_gnn_baselines.py'
if not PARENT.is_file():
    raise FileNotFoundError(f'Script 111 is required beside Script 112: {PARENT}')
spec=importlib.util.spec_from_file_location('benchmark_v2_script111',PARENT)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
OUT=HERE.parent/'data/processed/cross_country/cartographx_v2_architecture_v1'

class ResidualSAGE(nn.Module):
    """Two-layer residual GraphSAGE with gated tabular/graph feature fusion.

    Shared head supports genuine source-only zero shot. Joint training uses
    additional country residual logits; these heads only see their own country's
    TRAIN labels. No target-country head is created for zero-shot transfer.
    """
    def __init__(self,n_features,hidden=64,dropout=.2,joint=False):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(n_features,hidden),nn.LayerNorm(hidden),nn.ReLU())
        self.sage1=base.Layer(hidden,hidden,'GraphSAGE')
        self.sage2=base.Layer(hidden,hidden,'GraphSAGE')
        self.norm1=nn.LayerNorm(hidden);self.norm2=nn.LayerNorm(hidden)
        self.gate=nn.Linear(hidden*2,hidden)
        self.shared=nn.Linear(hidden,1)
        self.joint=joint
        self.country_delta=nn.Embedding(2,hidden) if joint else None
        self.country_bias=nn.Embedding(2,1) if joint else None
        if joint:
            nn.init.zeros_(self.country_delta.weight)
            nn.init.zeros_(self.country_bias.weight)
        self.dropout=dropout
    def forward(self,x,edge,country):
        z=self.project(x)
        h=self.norm1(z+F.dropout(F.relu(self.sage1(z,edge)),p=self.dropout,training=self.training))
        h=self.norm2(h+F.dropout(F.relu(self.sage2(h,edge)),p=self.dropout,training=self.training))
        gate=torch.sigmoid(self.gate(torch.cat([z,h],dim=1)))
        fused=gate*h+(1-gate)*z
        logits=self.shared(fused).flatten()
        if self.joint:
            logits=logits+(fused*self.country_delta(country)).sum(dim=1)/(fused.shape[1]**.5)
            logits=logits+self.country_bias(country).flatten()
        return logits

def country_tensor(graph,ids,device):
    countries=graph.df.iloc[np.asarray(ids,dtype=int)].country.to_numpy()
    return torch.tensor(np.where(countries=='Lithuania',0,1),dtype=torch.long,device=device)

def train_predict(graph,train_ids,predict_sets,args,seed,joint):
    base.seed_all(seed)
    (x,y,e),imp,sc=graph.subset(train_ids,fit=True,device=args.device)
    if len(torch.unique(y))!=2:raise RuntimeError('One-class training fold')
    c=country_tensor(graph,train_ids,args.device)
    model=ResidualSAGE(x.shape[1],args.hidden,args.dropout,joint).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    # Balanced within each TRAIN country, avoiding pooled majority domination.
    # For source-only experiments this is ordinary positive-class weighting.
    weights=torch.empty_like(y)
    for k in torch.unique(c):
        mask=c==k
        p=float(y[mask].sum().item());n=int(mask.sum().item())
        if p==0 or p==n:raise RuntimeError('One-class country in training fold')
        weights[mask]=torch.where(y[mask]>0.5,torch.tensor(n/(2*p),device=y.device),
                                   torch.tensor(n/(2*(n-p)),device=y.device))
    for epoch in range(args.epochs):
        model.train();optimizer.zero_grad(set_to_none=True)
        logits=model(x,e,c)
        loss=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*weights).mean()
        if not torch.isfinite(loss):raise RuntimeError(f'Non-finite loss at epoch {epoch}')
        loss.backward();optimizer.step()
    model.eval();outputs=[]
    with torch.no_grad():
        for ids in predict_sets:
            (px,_,pe),_,_=graph.subset(ids,imputer=imp,scaler=sc,device=args.device)
            pc=country_tensor(graph,ids,args.device)
            outputs.append(torch.sigmoid(model(px,pe,pc)).cpu().numpy())
    return outputs

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--epochs',type=int,default=60)
    ap.add_argument('--folds',type=int,default=3)
    ap.add_argument('--hidden',type=int,default=64)
    ap.add_argument('--dropout',type=float,default=.2)
    ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--weight-decay',type=float,default=1e-4)
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--output-dir',type=Path)
    args=ap.parse_args()
    if args.epochs<1 or args.folds<2 or args.hidden<1:ap.error('Invalid epochs/folds/hidden')
    if args.device=='auto':args.device='cuda' if torch.cuda.is_available() else 'cpu'
    if args.device=='cuda' and not torch.cuda.is_available():ap.error('CUDA unavailable')
    base.seed_all(args.seed)
    df=pd.read_csv(base.DATA,low_memory=False)
    edges=pd.read_csv(base.EDGES,low_memory=False)
    required={'sample_uid','country','city','tile_id','split','retained'}|set(base.FEATURES)
    if required-set(df.columns):raise RuntimeError(f'Missing columns: {sorted(required-set(df.columns))}')
    if df.sample_uid.isna().any() or df.sample_uid.duplicated().any():raise RuntimeError('Bad sample UIDs')
    if not df.retained.isin([0,1]).all():raise RuntimeError('Invalid labels')
    if (df.groupby(['country','tile_id']).split.nunique()>1).any():raise RuntimeError('Tile split leakage')
    for col in base.FEATURES:df[col]=pd.to_numeric(df[col],errors='raise')
    graph=base.GraphData(df,edges)
    out=args.output_dir or OUT/f'seed_{args.seed}'
    out.mkdir(parents=True,exist_ok=True)
    cv_rows=[];results=[];preds=[];cache={}
    for protocol,(source,target,joint) in base.PROTOCOLS.items():
        train_ids=np.flatnonzero((df.split=='train') & (df.country.isin(['Lithuania','Poland']) if joint else df.country.eq(source)))
        val_country=target if joint or source==target else source
        val_ids=np.flatnonzero((df.split=='validation') & df.country.eq(val_country))
        test_ids=np.flatnonzero((df.split=='test') & df.country.eq(target))
        if not len(train_ids) or not len(val_ids) or not len(test_ids):raise RuntimeError('Empty protocol partition')
        key='joint' if joint else source
        if key not in cache:
            groups=(df.iloc[train_ids].country.astype(str)+'::'+df.iloc[train_ids].tile_id.astype(str)).to_numpy()
            y=graph.labels[train_ids]
            cv=StratifiedGroupKFold(n_splits=args.folds,shuffle=True,random_state=args.seed)
            oof=np.full(len(train_ids),np.nan)
            for fold,(tr,va) in enumerate(cv.split(train_ids,y,groups)):
                if set(groups[tr])&set(groups[va]):raise RuntimeError('Spatial CV leakage')
                if len(np.unique(y[tr]))!=2 or len(np.unique(y[va]))!=2:raise RuntimeError('One-class fold')
                if joint:
                    for country in ('Lithuania','Poland'):
                        yc=y[tr][df.iloc[train_ids[tr]].country.to_numpy()==country]
                        if len(np.unique(yc))!=2:raise RuntimeError('One-class joint country fold')
                (p,)=train_predict(graph,train_ids[tr],[train_ids[va]],args,args.seed+1009*fold,joint)
                oof[va]=p
            if not np.isfinite(oof).all():raise RuntimeError('Incomplete OOF')
            # Train-only OOF score is diagnostic; architecture is fixed, not chosen on test.
            oof_t=base.threshold_for(y,oof)
            cv_m=base.metrics(y,oof,oof_t)
            by_country={}
            for country in (('Lithuania','Poland') if joint else (source,)):
                mask=df.iloc[train_ids].country.to_numpy()==country
                by_country[country]=base.metrics(y[mask],oof[mask],oof_t)
            cache[key]=(oof_t,cv_m,by_country)
            print(f'  {key:10s} fixed CartoGraph-X OOF_MCC={cv_m["mcc"]:.4f} OOF_AUC={cv_m["roc_auc"]:.4f} OOF_t={oof_t:.2f}',flush=True)
        oof_t,cv_m,by_country=cache[key]
        for country,m in by_country.items():
            cv_rows.append({'protocol':protocol,'seed':args.seed,'country':country,'model':'CartoGraphX_ResidualSAGE',
                            'cv_mcc':m['mcc'],'cv_roc_auc':m['roc_auc'],'pooled_cv_mcc':cv_m['mcc'],
                            'cv_oof_threshold':oof_t,'train_n':len(train_ids),'joint_training':joint})
        val_prob,test_prob=train_predict(graph,train_ids,[val_ids,test_ids],args,args.seed+99173,joint)
        threshold=base.threshold_for(graph.labels[val_ids],val_prob)
        val_m=base.metrics(graph.labels[val_ids],val_prob,threshold)
        # Historical test labels are read only after training and threshold freezing.
        test_m=base.metrics(graph.labels[test_ids],test_prob,threshold)
        row={'protocol':protocol,'seed':args.seed,'source':source,'target':target,'joint_training':joint,
             'model':'CartoGraphX_ResidualSAGE','variant':'country_balanced_loss',
             'feature_set':'primitive_tabular_plus_induced_graph',
             'cv_mcc':cv_m['mcc'],'cv_roc_auc':cv_m['roc_auc'],'cv_oof_threshold':oof_t,
             'threshold':threshold,'threshold_selection_country':val_country,
             'train_n':len(train_ids),'validation_n':len(val_ids),'test_n':len(test_ids)}
        row.update({f'validation_{k}':v for k,v in val_m.items()})
        row.update({f'test_{k}':v for k,v in test_m.items()})
        results.append(row)
        preds.append(pd.DataFrame({'protocol':protocol,'seed':args.seed,
           'sample_uid':df.iloc[test_ids].sample_uid.to_numpy(),
           'country':df.iloc[test_ids].country.to_numpy(),'city':df.iloc[test_ids].city.to_numpy(),
           'tile_id':df.iloc[test_ids].tile_id.to_numpy(),'split':'test','label':graph.labels[test_ids],
           'probability':test_prob,'prediction':(test_prob>=threshold).astype(int),
           'threshold':threshold,'model':'CartoGraphX_ResidualSAGE'}))
        print(f'{protocol:22s} CartoGraphX_ResidualSAGE CV_MCC={cv_m["mcc"]:.4f} '
              f'VAL_MCC={val_m["mcc"]:.4f} TEST_MCC={test_m["mcc"]:.4f} threshold={threshold:.2f}',flush=True)
    files={'spatial_cv_country_diagnostics.csv':pd.DataFrame(cv_rows),
           'selected_protocol_results.csv':pd.DataFrame(results),
           'selected_test_predictions.csv':pd.concat(preds,ignore_index=True)}
    for name,frame in files.items():frame.to_csv(out/name,index=False)
    manifest={'script':Path(__file__).name,'script_sha256':base.digest(Path(__file__)),
      'dependency_script_111_sha256':base.digest(PARENT),'timestamp_utc':datetime.now(timezone.utc).isoformat(),
      'seed':args.seed,'epochs':args.epochs,'folds':args.folds,'hidden':args.hidden,
      'dropout':args.dropout,'lr':args.lr,'weight_decay':args.weight_decay,'device':args.device,
      'torch_version':torch.__version__,'node_sha256':base.digest(base.DATA),'edge_sha256':base.digest(base.EDGES),
      'architecture':'fixed two-layer residual GraphSAGE, layer normalization, gated primitive-feature fusion, shared head, joint-only country residual heads',
      'training':'within-training-country class-balanced BCE; training-only spatial OOF diagnostics; fixed epochs',
      'zero_shot_policy':'source-only training and validation; shared head; target country not trained',
      'graph_semantics':'geometric proximity kNN, NOT road topology',
      'limitations':['C2 correspondence labels provisional and not manually validated',
                     'historical test already examined; exploratory only',
                     'CV OOF threshold tuned on OOF predictions, not an unbiased estimate',
                     'fixed hyperparameters; no architecture search or early stopping'],
      'outputs_sha256':{name:base.digest(out/name) for name in files}}
    (out/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'\nSaved: {out}\nEXPLORATORY ONLY: historical test is not untouched.',flush=True)

if __name__=='__main__':main()
