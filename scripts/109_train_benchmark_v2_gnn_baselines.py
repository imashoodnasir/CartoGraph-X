#!/usr/bin/env python3
"""CartoGraph-X Benchmark V2: leakage-controlled native-PyTorch GCN/SAGE/GAT baselines.

Usage: python scripts/109_train_benchmark_v2_gnn_baselines.py --seed 42
Optional: --epochs 60 --folds 3 --device auto --hidden 64 --patience 0

Graph = geometric proximity kNN, NOT true road-network topology. All graphs are
induced separately for training/CV folds, validation, and test; no edges cross
countries, splits, or spatial CV groups. Architecture selected by train-only
spatial OOF MCC @ 0.5; threshold by permitted validation country. Test labels
are accessed only after selection. Historical test is exploratory, not untouched.
Requires torch, numpy, pandas, scikit-learn. No torch-geometric dependency.
"""
from __future__ import annotations
import argparse, hashlib, json, random, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/cross_country/benchmark_v2_unified_labels_v1/road_graph_nodes_v2.csv'
EDGES = ROOT / 'data/processed/cross_country/graph_representations/road_graph_edges.csv'
OUT = ROOT / 'data/processed/cross_country/benchmark_v2_gnn_baselines_v1'
# Use primitive road attributes only: precomputed graph/neighborhood features could
# carry held-out fold information into spatial CV. Message passing uses induced edges.
FEATURES = ['length_m','intersecting_road_count','neighbor_count_150m','mean_neighbor_distance_m']

PROTOCOLS = {'lt_to_lt':('Lithuania','Lithuania',False), 'pl_to_pl':('Poland','Poland',False),
'lt_to_pl_zero_shot':('Lithuania','Poland',False), 'pl_to_lt_zero_shot':('Poland','Lithuania',False),
'joint_to_lt':('Lithuania','Lithuania',True), 'joint_to_pl':('Poland','Poland',True)}
MODELS = ('GCN','GraphSAGE','GAT')
THRESHOLDS = np.round(np.arange(.05,.951,.01),2)

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    try:torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:pass

def metrics(y,p,t):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    pred=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'mcc':float(matthews_corrcoef(y,pred)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'f1':float(f1_score(y,pred,zero_division=0)),
            'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan'),
            'pr_auc':float(average_precision_score(y,p)) if len(np.unique(y))==2 else float('nan'),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def threshold_for(y,p):
    # Tie break toward 0.5, then smaller threshold, matching a deterministic policy.
    return float(max(THRESHOLDS,key=lambda t:(matthews_corrcoef(y,p>=t),-abs(t-.5),-t)))

class GraphData:
    def __init__(self,df,edge_df):
        self.df=df.reset_index(drop=True)
        self.uid_to_i=pd.Series(np.arange(len(df),dtype=np.int64),index=df.sample_uid.astype(str)).to_dict()
        self.features=df[FEATURES].to_numpy(dtype=np.float64)
        self.labels=df.retained.to_numpy(dtype=np.int64)
        a=edge_df.source_uid.astype(str).map(self.uid_to_i)
        b=edge_df.target_uid.astype(str).map(self.uid_to_i)
        if a.isna().any() or b.isna().any():raise RuntimeError('Graph edges refer to unknown nodes')
        self.u=a.to_numpy(dtype=np.int64); self.v=b.to_numpy(dtype=np.int64)
        if np.any(self.u==self.v):raise RuntimeError('Unexpected self edges in original graph')
        for col in ['country','split','tile_id']:
            vals=df[col].astype(str).to_numpy()
            if np.any(vals[self.u]!=vals[self.v]):raise RuntimeError(f'Leakage: graph edge crosses {col}')
        self.n=len(df)

    def subset(self,ids,imputer=None,scaler=None,fit=False,device='cpu'):
        ids=np.asarray(ids,dtype=np.int64)
        if len(ids)==0:raise RuntimeError('Empty graph subset')
        if len(np.unique(ids))!=len(ids):raise RuntimeError('Repeated graph indices')
        lookup=np.full(self.n,-1,dtype=np.int64); lookup[ids]=np.arange(len(ids))
        keep=(lookup[self.u]>=0)&(lookup[self.v]>=0)
        src=np.r_[lookup[self.u[keep]],lookup[self.v[keep]],np.arange(len(ids))]
        dst=np.r_[lookup[self.v[keep]],lookup[self.u[keep]],np.arange(len(ids))]
        # Remove duplicate directed edges; self loops exactly once.
        pairs=np.unique(np.stack([src,dst],axis=1),axis=0)
        edges=torch.tensor(pairs.T.copy(),dtype=torch.long,device=device)
        x=self.features[ids]
        if fit:
            imputer=SimpleImputer(strategy='median',keep_empty_features=True)
            scaler=StandardScaler()
            x=scaler.fit_transform(imputer.fit_transform(x))
        else:
            if imputer is None or scaler is None:raise RuntimeError('Transformers must be training-fitted')
            x=scaler.transform(imputer.transform(x))
        x=np.nan_to_num(x,nan=0,posinf=0,neginf=0).astype(np.float32)
        y=self.labels[ids]
        return (torch.tensor(x,device=device),torch.tensor(y,dtype=torch.float32,device=device),edges),imputer,scaler

class Layer(nn.Module):
    def __init__(self,dim_in,dim_out,kind):
        super().__init__();self.kind=kind
        self.lin=nn.Linear(dim_in,dim_out,bias=False)
        if kind=='GraphSAGE':self.combine=nn.Linear(dim_in*2,dim_out)
        if kind=='GAT':
            self.att_src=nn.Parameter(torch.empty(dim_out));self.att_dst=nn.Parameter(torch.empty(dim_out))
            nn.init.xavier_uniform_(self.att_src.unsqueeze(0));nn.init.xavier_uniform_(self.att_dst.unsqueeze(0))
    def forward(self,x,edge):
        src,dst=edge; n=x.shape[0]
        if self.kind=='GraphSAGE':
            agg=torch.zeros_like(x);agg.index_add_(0,dst,x[src])
            deg=torch.bincount(dst,minlength=n).to(x.dtype).clamp(min=1).unsqueeze(1)
            return self.combine(torch.cat((x,agg/deg),dim=1))
        h=self.lin(x)
        if self.kind=='GCN':
            deg=torch.bincount(dst,minlength=n).to(h.dtype).clamp(min=1)
            weight=(deg[src]*deg[dst]).rsqrt()
            agg=torch.zeros_like(h);agg.index_add_(0,dst,h[src]*weight.unsqueeze(1))
            return agg
        logits=F.leaky_relu((h[src]*self.att_src).sum(1)+(h[dst]*self.att_dst).sum(1),.2)
        max_per_node=torch.full((n,),-float('inf'),dtype=logits.dtype,device=logits.device)
        max_per_node.scatter_reduce_(0,dst,logits,reduce='amax',include_self=True)
        exp=(logits-max_per_node[dst]).exp()
        den=torch.zeros(n,device=h.device,dtype=h.dtype);den.index_add_(0,dst,exp)
        alpha=exp/den[dst].clamp_min(1e-12)
        out=torch.zeros_like(h);out.index_add_(0,dst,h[src]*alpha.unsqueeze(1))
        return out

class Net(nn.Module):
    def __init__(self,dim,hidden,kind,dropout):
        super().__init__();self.a=Layer(dim,hidden,kind);self.b=Layer(hidden,hidden,kind)
        self.out=nn.Linear(hidden,1);self.dropout=dropout
    def forward(self,x,edges):
        h=F.relu(self.a(x,edges));h=F.dropout(h,p=self.dropout,training=self.training)
        h=F.relu(self.b(h,edges));h=F.dropout(h,p=self.dropout,training=self.training)
        return self.out(h).flatten()

def train_predict(graph,train_ids,prediction_sets,kind,args,run_seed):
    # All transforms fit on TRAIN nodes only; prediction graphs independently induced.
    seed_all(run_seed)
    train_data,imp,scale=graph.subset(train_ids,fit=True,device=args.device)
    x,y,edges=train_data
    if len(torch.unique(y))!=2:raise RuntimeError('Training fold contains only one class')
    model=Net(x.shape[1],args.hidden,kind,args.dropout).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    # No CV-held-out or validation labels used for early stopping or learning rate.
    for epoch in range(args.epochs):
        model.train();optimizer.zero_grad(set_to_none=True)
        logits=model(x,edges)
        loss=F.binary_cross_entropy_with_logits(logits,y)
        if not torch.isfinite(loss):raise RuntimeError(f'Non-finite loss for {kind} epoch {epoch}')
        loss.backward();optimizer.step()
    model.eval();preds=[]
    with torch.no_grad():
        for ids in prediction_sets:
            (px,_,pe),_,_=graph.subset(ids,imputer=imp,scaler=scale,device=args.device)
            preds.append(torch.sigmoid(model(px,pe)).cpu().numpy())
    del model,optimizer,train_data
    return preds

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--folds',type=int,default=3)
    ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--hidden',type=int,default=64)
    ap.add_argument('--dropout',type=float,default=.2);ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--weight-decay',type=float,default=1e-4)
    ap.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    ap.add_argument('--output-dir',type=Path)
    args=ap.parse_args()
    if args.folds<2 or args.epochs<1 or args.hidden<1:ap.error('Invalid folds/epochs/hidden')
    if args.device=='auto':args.device='cuda' if torch.cuda.is_available() else 'cpu'
    if args.device=='cuda' and not torch.cuda.is_available():ap.error('CUDA unavailable')
    seed_all(args.seed)
    if not DATA.exists() or not EDGES.exists():raise FileNotFoundError(f'Missing {DATA} or {EDGES}')
    df=pd.read_csv(DATA,low_memory=False);edge_df=pd.read_csv(EDGES,low_memory=False)
    required={'sample_uid','country','city','tile_id','split','retained'}|set(FEATURES)
    if required-set(df.columns):raise RuntimeError(f'Missing node columns {sorted(required-set(df.columns))}')
    if {'source_uid','target_uid'}-set(edge_df.columns):raise RuntimeError('Missing edge endpoint columns')
    if df.sample_uid.isna().any() or df.sample_uid.duplicated().any():raise RuntimeError('Duplicate/missing UIDs')
    if not df.retained.isin([0,1]).all():raise RuntimeError('Nonbinary labels')
    if set(df.country.unique())!={'Lithuania','Poland'}:raise RuntimeError('Unexpected countries')
    if set(df.split.unique())!={'train','validation','test'}:raise RuntimeError('Unexpected splits')
    if df.tile_id.isna().any() or (df.groupby(['country','tile_id']).split.nunique()>1).any():
        raise RuntimeError('Tile crosses split or missing tile')
    for c in FEATURES:
        df[c]=pd.to_numeric(df[c],errors='raise')
        if np.isinf(df[c].to_numpy(dtype=float)).any():raise RuntimeError(f'Infinite feature {c}')
    graph=GraphData(df,edge_df)
    out=args.output_dir or OUT/f'seed_{args.seed}';out.mkdir(parents=True,exist_ok=True)
    all_cv=[];all_results=[];all_preds=[];cache={}
    # Cache train-only OOF model selection across protocols with identical training populations.
    for protocol,(source,target,joint) in PROTOCOLS.items():
        train_ids=np.flatnonzero((df.split=='train') & (df.country.isin(['Lithuania','Poland']) if joint else df.country.eq(source)))
        validation_country=target if joint or source==target else source
        val_ids=np.flatnonzero((df.split=='validation') & df.country.eq(validation_country))
        test_ids=np.flatnonzero((df.split=='test') & df.country.eq(target))
        key='joint' if joint else source
        if min(len(train_ids),len(val_ids),len(test_ids))==0:raise RuntimeError(f'Empty partition {protocol}')
        if key not in cache:
            trdf=df.iloc[train_ids]
            groups=(trdf.country.astype(str)+'::'+trdf.tile_id.astype(str)).to_numpy()
            y=graph.labels[train_ids]
            n_groups=len(np.unique(groups));folds=min(args.folds,n_groups)
            if folds<2:raise RuntimeError('Insufficient groups for CV')
            cv=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=args.seed)
            splits=list(cv.split(train_ids,y,groups))
            best=None;best_kind=None;cv_by_kind={}
            for kind in MODELS:
                oof=np.full(len(train_ids),np.nan)
                for fold,(tr,va) in enumerate(splits):
                    if set(groups[tr])&set(groups[va]):raise RuntimeError('CV spatial group overlap')
                    if len(np.unique(y[tr]))<2 or len(np.unique(y[va]))<2:raise RuntimeError('One-class CV fold')
                    # The fold-held-out nodes and their incident edges are excluded from training.
                    (prob,)=train_predict(graph,train_ids[tr],[train_ids[va]],kind,args,args.seed+fold*1009)
                    oof[va]=prob
                if not np.isfinite(oof).all():raise RuntimeError('Missing OOF probabilities')
                score=float(matthews_corrcoef(y,oof>=.5));auc=float(roc_auc_score(y,oof))
                cv_by_kind[kind]=(score,auc)
                print(f'  {key:10s} {kind:10s} OOF_MCC={score:.4f} OOF_AUC={auc:.4f}',flush=True)
                candidate=(score,auc,-MODELS.index(kind))
                if best is None or candidate>best:best=candidate;best_kind=kind
            cache[key]=(best_kind,cv_by_kind,folds)
        best_kind,cv_by_kind,folds=cache[key]
        for kind,(mcc,auc) in cv_by_kind.items():
            all_cv.append({'protocol':protocol,'seed':args.seed,'model':kind,'feature_set':'primitive_tabular_plus_induced_graph',
                           'cv_mcc_at_0_5':mcc,'cv_roc_auc':auc,'folds':folds,'train_n':len(train_ids),
                           'train_positive_rate':float(graph.labels[train_ids].mean())})
        val_prob,test_prob=train_predict(graph,train_ids,[val_ids,test_ids],best_kind,args,args.seed+99173)
        threshold=threshold_for(graph.labels[val_ids],val_prob)
        val_m=metrics(graph.labels[val_ids],val_prob,threshold)
        # Test labels only used below after model, architecture, threshold are frozen.
        test_m=metrics(graph.labels[test_ids],test_prob,threshold)
        record={'protocol':protocol,'seed':args.seed,'source':source,'target':target,'joint_training':joint,
                'model':best_kind,'feature_set':'primitive_tabular_plus_induced_graph','cv_mcc_at_0_5':cv_by_kind[best_kind][0],
                'cv_roc_auc':cv_by_kind[best_kind][1],'threshold':threshold,
                'threshold_selection_country':validation_country,'train_n':len(train_ids),
                'validation_n':len(val_ids),'test_n':len(test_ids)}
        record.update({f'validation_{k}':v for k,v in val_m.items()})
        record.update({f'test_{k}':v for k,v in test_m.items()})
        all_results.append(record)
        all_preds.append(pd.DataFrame({'protocol':protocol,'seed':args.seed,
             'sample_uid':df.iloc[test_ids].sample_uid.to_numpy(),
             'country':df.iloc[test_ids].country.to_numpy(),
             'city':df.iloc[test_ids].city.to_numpy(),
             'tile_id':df.iloc[test_ids].tile_id.to_numpy(),
             'split':'test','label':graph.labels[test_ids],
             'probability':test_prob,'prediction':(test_prob>=threshold).astype(int),
             'threshold':threshold,'model':best_kind,'feature_set':'primitive_tabular_plus_induced_graph'}))
        print(f'{protocol:22s} {best_kind:10s} CV_MCC={cv_by_kind[best_kind][0]:.4f} '
              f'VAL_MCC={val_m["mcc"]:.4f} TEST_MCC={test_m["mcc"]:.4f} threshold={threshold:.2f}',flush=True)
    files={'spatial_cv_candidate_results.csv':pd.DataFrame(all_cv),
           'selected_protocol_results.csv':pd.DataFrame(all_results),
           'selected_test_predictions.csv':pd.concat(all_preds,ignore_index=True)}
    for name,frame in files.items():frame.to_csv(out/name,index=False)
    manifest={'script':Path(__file__).name,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
              'seed':args.seed,'folds':args.folds,'epochs':args.epochs,'hidden':args.hidden,
              'dropout':args.dropout,'lr':args.lr,'weight_decay':args.weight_decay,
              'device':args.device,'torch_version':torch.__version__,'input_nodes':str(DATA),
              'input_edges':str(EDGES),'node_sha256':digest(DATA),'edge_sha256':digest(EDGES),
              'graph_semantics':'geometric kNN, not topological road graph',
              'feature_policy':'primitive tabular attributes only; avoid fold-contaminated precomputed graph features',
              'cv_selection':'train-only spatial OOF MCC at 0.5; AUC tie-break',
              'threshold_selection':'permitted validation-country MCC',
              'test_policy':'exploratory historical test, never used for selection',
              'limitations':['provisional C2 labels, manual review incomplete',
                             'historical test previously examined','fixed-epoch training; no early stopping'],
              'outputs_sha256':{name:digest(out/name) for name in files}}
    (out/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
    print(f'\nSaved: {out}\nEXPLORATORY ONLY: historical test partition is not untouched.',flush=True)

if __name__=='__main__':main()
