#!/usr/bin/env python3
"""Frozen CartoGraph-X V2 and GCN/GraphSAGE/GAT efficiency benchmark.

Standalone RUNNER (imports the unchanged Script 111 and Script 112 alongside it).
Ten matched seeds, 60 epochs, three train populations, four architectures.
No test labels, no model selection, no accuracy claims, no figure generation.

Run: python scripts/cartographx_v2_efficiency_benchmark_10runs.py
Quick smoke test: python scripts/cartographx_v2_efficiency_benchmark_10runs.py --seeds 42 --epochs 2 --warmup 2 --repeats 5
Run with PyTorch operator FLOPs profiling: ... --profile-flops

Training time excludes loading CSVs and preprocessing but includes forward,
backward, optimizer steps and CUDA synchronization. Inference uses the complete
validation-country induced graph and reports latency and effective throughput.
PyTorch profiler FLOPs only count supported operators (often linear/matmul),
NOT all graph gather/scatter/softmax operations: treat as PARTIAL FLOPs.
"""
from __future__ import annotations
import argparse, importlib.util, json, math, os, platform, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent

def load_script(filename,modname):
    path=HERE/filename
    if not path.is_file(): raise FileNotFoundError(f'Missing required unchanged script: {path}')
    spec=importlib.util.spec_from_file_location(modname,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[modname]=module
    spec.loader.exec_module(module)
    return module

base=load_script('111_train_benchmark_v2_weighted_gnn_baselines.py','benchmark111_efficiency')
arch=load_script('112_train_cartographx_v2_architecture.py','benchmark112_efficiency')
ARCHS=('CartoGraph-X','GCN','GraphSAGE','GAT')
POPS=('Lithuania','Poland','Joint')

def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)

def memory_peak(device):
    return float(torch.cuda.max_memory_allocated(device)/1048576) if device.type=='cuda' else float('nan')

def percentile(a,p):return float(np.percentile(a,p))

def construct(name,nfeatures,args,joint,device):
    if name=='CartoGraph-X':
        return arch.ResidualSAGE(nfeatures,args.hidden,args.dropout,joint=joint).to(device)
    return base.Net(nfeatures,args.hidden,name,args.dropout).to(device)

def forward(model,name,x,e,c):
    return model(x,e,c) if name=='CartoGraph-X' else model(x,e)

def train_one(graph,train_ids,pred_ids,name,pop,seed,args,device):
    # identical fixed train and validation graph for all models in a population
    base.seed_all(seed+99173)
    train_data,imputer,scaler=graph.subset(train_ids,fit=True,device=str(device))
    x,y,e=train_data
    (vx,_,ve),_,_=graph.subset(pred_ids,imputer=imputer,scaler=scaler,device=str(device))
    c=arch.country_tensor(graph,train_ids,str(device))
    vc=arch.country_tensor(graph,pred_ids,str(device))
    model=construct(name,x.shape[1],args,pop=='Joint',device)
    params=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    if name=='CartoGraph-X':
        weights=torch.empty_like(y)
        for k in torch.unique(c):
            mask=c==k;p=float(y[mask].sum().item());n=int(mask.sum().item())
            if p<=0 or p>=n:raise RuntimeError('One-class training population')
            weights[mask]=torch.where(y[mask]>.5,n/(2*p),n/(2*(n-p)))
    else:
        p=float(y.sum().item());n=int(y.numel())
        if p<=0 or p>=n:raise RuntimeError('One-class training population')
        pos_weight=torch.tensor([float((n-p)/p)],dtype=torch.float32,device=device)

    # reset peak AFTER data/model/optimizer allocation: includes live allocation baseline
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    sync(device);start=time.perf_counter()
    for epoch in range(args.epochs):
        model.train();opt.zero_grad(set_to_none=True)
        logits=forward(model,name,x,e,c)
        if name=='CartoGraph-X':
            loss=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*weights).mean()
        else:
            loss=F.binary_cross_entropy_with_logits(logits,y,pos_weight=pos_weight)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward();opt.step()
    sync(device);train_seconds=time.perf_counter()-start
    peak_train=memory_peak(device)
    model.eval()
    with torch.inference_mode():
        for _ in range(args.warmup):
            _=forward(model,name,vx,ve,vc)
        sync(device)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        timings=[]
        for _ in range(args.repeats):
            sync(device);t0=time.perf_counter()
            logits=forward(model,name,vx,ve,vc)
            sync(device);timings.append(time.perf_counter()-t0)
        peak_infer=memory_peak(device)
        if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite inference logits')
    n_nodes=len(pred_ids)
    inference_mean=float(np.mean(timings))
    record=dict(seed=seed,population=pop,model=name,device=str(device),
        train_nodes=len(train_ids),validation_nodes=n_nodes,
        train_directed_edges=int(e.shape[1]),validation_directed_edges=int(ve.shape[1]),
        epochs=args.epochs,parameter_count=params,trainable_parameter_count=trainable,
        training_seconds=train_seconds,training_seconds_per_epoch=train_seconds/args.epochs,
        inference_graph_seconds_mean=inference_mean,
        inference_graph_seconds_std=float(np.std(timings,ddof=1)),
        inference_graph_seconds_median=float(np.median(timings)),
        inference_graph_seconds_p95=percentile(timings,95),
        inference_milliseconds_per_road=1000*inference_mean/n_nodes,
        inference_roads_per_second=n_nodes/inference_mean,
        gpu_peak_training_mib=peak_train,gpu_peak_inference_mib=peak_infer,
        timing_repetitions=args.repeats,timing_warmup=args.warmup)
    profile=None
    if args.profile_flops and seed==args.seeds[0]:
        try:
            from torch.profiler import profile as torch_profile, ProfilerActivity
            activities=[ProfilerActivity.CPU]
            if device.type=='cuda':activities.append(ProfilerActivity.CUDA)
            with torch.inference_mode():
                with torch_profile(activities=activities,with_flops=True,record_shapes=False) as prof:
                    _=forward(model,name,vx,ve,vc)
                    sync(device)
            events=prof.key_averages()
            flops=int(sum(int(evt.flops or 0) for evt in events))
            profile=dict(seed=seed,population=pop,model=name,
                profiler_supported_forward_flops=flops,
                profiler_supported_flops_per_road=flops/n_nodes,
                flops_completeness='PARTIAL: unsupported graph gather/scatter/softmax and other ops excluded',
                counted_ops=';'.join(f'{ev.key}:{int(ev.flops)}' for ev in events if ev.flops))
        except Exception as exc:
            profile=dict(seed=seed,population=pop,model=name,
                         profiler_supported_forward_flops=float('nan'),
                         profiler_supported_flops_per_road=float('nan'),
                         flops_completeness=f'PROFILER FAILED: {type(exc).__name__}: {exc}')
    del model,opt,train_data,x,y,e,vx,ve,c,vc,logits
    if device.type=='cuda':torch.cuda.empty_cache()
    return record,profile

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(42,52)))
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--dropout',type=float,default=.2)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--repeats',type=int,default=30)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--profile-flops',action='store_true',help='Profile supported forward FLOPs once per model/population (partial coverage)')
    p.add_argument('--output-dir',type=Path,default=HERE.parent/'data/processed/cross_country/cartographx_v2_efficiency_10runs_v1')
    p.add_argument('--force',action='store_true',help='Rerun already saved seed/population/model combinations')
    args=p.parse_args()
    if not args.seeds or len(set(args.seeds))!=len(args.seeds):p.error('Provide unique seeds')
    if args.epochs<1 or args.hidden<1 or args.warmup<0 or args.repeats<2:p.error('Invalid epochs/hidden/warmup/repeats')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if device.type=='cuda' and not torch.cuda.is_available():p.error('CUDA not available')
    if not base.DATA.is_file() or not base.EDGES.is_file():raise FileNotFoundError('Benchmark V2 graph inputs missing')
    df=pd.read_csv(base.DATA,low_memory=False)
    edges=pd.read_csv(base.EDGES,low_memory=False)
    graph=base.GraphData(df,edges)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    runs_file=args.output_dir/'efficiency_all_runs.csv'
    flops_file=args.output_dir/'forward_flops_profiler_partial.csv'
    existing=pd.read_csv(runs_file) if runs_file.is_file() and not args.force else pd.DataFrame()
    flops_existing=pd.read_csv(flops_file) if flops_file.is_file() and not args.force else pd.DataFrame()
    if len(existing):
        mismatched=existing[(existing.epochs!=args.epochs) | (existing.device!=str(device)) |
            (existing.timing_repetitions!=args.repeats) | (existing.timing_warmup!=args.warmup)]
        if len(mismatched):
            raise RuntimeError('Existing results use different settings. Use a separate --output-dir for smoke tests, or --force to replace prior results.')
    done={(int(r.seed),str(r.population),str(r.model)) for r in existing.itertuples()} if len(existing) else set()
    rows=existing.to_dict('records') if len(existing) else []
    profiles=flops_existing.to_dict('records') if len(flops_existing) else []
    print('BENCHMARK HARDWARE:',torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),flush=True)
    print(f'DEVICE={device}, epochs={args.epochs}, seeds={args.seeds}, repeats={args.repeats}',flush=True)
    for seed in args.seeds:
        for pop in POPS:
            train_ids=np.flatnonzero((df.split=='train') & (df.country.isin(['Lithuania','Poland']) if pop=='Joint' else df.country.eq(pop)))
            # Joint inference measured on both validation countries; country-specific on matching validation country
            pred_ids=np.flatnonzero((df.split=='validation') & (df.country.isin(['Lithuania','Poland']) if pop=='Joint' else df.country.eq(pop)))
            if not len(train_ids) or not len(pred_ids):raise RuntimeError(f'Empty population {pop}')
            for name in ARCHS:
                key=(seed,pop,name)
                if key in done:
                    print('SKIP',key,flush=True);continue
                print(f'RUN seed={seed} population={pop} model={name}',flush=True)
                rec,profile=train_one(graph,train_ids,pred_ids,name,pop,seed,args,device)
                rows.append(rec)
                if profile is not None:profiles.append(profile)
                pd.DataFrame(rows).to_csv(runs_file,index=False)
                if profiles:pd.DataFrame(profiles).to_csv(flops_file,index=False)
                print(f'  train={rec["training_seconds"]:.3f}s, inference={rec["inference_graph_seconds_mean"]*1000:.3f}ms, '
                      f'throughput={rec["inference_roads_per_second"]:.1f} roads/s, '
                      f'params={rec["parameter_count"]:,}, peakGPU={rec["gpu_peak_training_mib"]:.1f} MiB',flush=True)
    data=pd.DataFrame(rows)
    # Prevent mixing quick-test configurations with final benchmarks.
    current=data[data.seed.isin(args.seeds) & (data.epochs==args.epochs) & (data.device==str(device)) &
                 (data.timing_repetitions==args.repeats) & (data.timing_warmup==args.warmup)].copy()
    metric_cols=['parameter_count','trainable_parameter_count','training_seconds','training_seconds_per_epoch',
                 'inference_graph_seconds_mean','inference_milliseconds_per_road','inference_roads_per_second',
                 'gpu_peak_training_mib','gpu_peak_inference_mib']
    summary=current.groupby(['population','model'],as_index=False)[metric_cols].agg(['mean','std'])
    summary.columns=['_'.join(str(x) for x in col if x) for col in summary.columns.to_flat_index()]
    summary.to_csv(args.output_dir/'table_efficiency_mean_std.csv',index=False)
    manifest={'status':'PASS','timestamp_utc':datetime.now(timezone.utc).isoformat(),
      'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
      'device':str(device),'gpu_name':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
      'python':platform.python_version(),'seeds_requested':args.seeds,'epochs':args.epochs,
      'warmup':args.warmup,'repeats':args.repeats,'run_count_current':len(current),
      'graph_type':'geometric proximity kNN, not topological road connectivity',
      'inference':'complete induced validation-country graph; no test labels used',
      'training':'fixed 60 epochs by default; weighted BCE for baselines; country-balanced BCE for CartoGraph-X',
      'flops':'PyTorch profiler partial forward FLOPs only, not complete model FLOPs' if args.profile_flops else 'not measured; rerun with --profile-flops',
      'warning':'Hardware timing is environment-specific; GPU peak is torch CUDA allocated memory, not total device occupancy.'}
    (args.output_dir/'efficiency_manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
    print('\nEFFICIENCY BENCHMARK PASS:',len(current),'matching runs',flush=True)
    print('Saved:',args.output_dir,flush=True)
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
