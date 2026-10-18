#!/usr/bin/env python
"""67_analyze_cross_country_domain_shift.py

CartoGen — Cross-Country Domain-Shift and Transfer-Failure Analysis.

Deterministic shift diagnostics are computed once. The learned country-separability
classifier supports smoke (seed 42) and full (seeds 42–51) repeated runs. Transfer
analysis consumes Script 66 all_runs_metrics.csv and compares matched model×seed
pairs; it never selects a model using test performance.
"""
from __future__ import annotations

import argparse, json, math, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parents[1]
HARMONIZED_DIR = ROOT / "data" / "processed" / "cross_country" / "harmonized_roads"
BENCHMARK = HARMONIZED_DIR / "lithuania_poland_road_benchmark.csv"
BASELINE_DIR = ROOT / "data" / "processed" / "cross_country" / "baseline_models"
OUT = ROOT / "data" / "processed" / "cross_country" / "domain_shift"
FIG = OUT / "figures"
FEATURES = ["length_m", "intersecting_road_count", "neighbor_count_150m", "mean_neighbor_distance_m"]
COUNTRIES = ["Lithuania", "Poland"]
SPLITS = ["train", "validation", "test"]
SMOKE_SEEDS = [42]
FULL_SEEDS = list(range(42, 52))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    return p.parse_args()


def line(char="=", n=120): print(char*n)
def section(title): print(); line("-"); print(title); line("-")
def set_seed(seed): random.seed(seed); np.random.seed(seed)

def json_default(obj):
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, Path): return str(obj)
    try:
        if pd.isna(obj): return None
    except Exception: pass
    raise TypeError(type(obj).__name__)

def savefig(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    png, pdf = FIG/f"{stem}.png", FIG/f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight"); plt.close(fig)
    print(f"PNG: {png}\nPDF: {pdf}")

def require_columns(df, cols):
    missing=[c for c in cols if c not in df.columns]
    if missing: raise KeyError(f"Missing required columns: {missing}\nAvailable: {df.columns.tolist()}")

def standardize_country(s):
    x=s.astype(str).str.strip().str.lower()
    return x.map({"lithuania":"Lithuania","poland":"Poland"}).fillna(s)

def infer_columns(df):
    country=next((c for c in ["country","country_harmonized"] if c in df.columns),None)
    split=next((c for c in ["split_harmonized","split"] if c in df.columns),None)
    retained=next((c for c in ["retained","binary_label","selection_binary"] if c in df.columns),None)
    city=next((c for c in ["city","city_harmonized"] if c in df.columns),None)
    if not country or not split or not retained: raise KeyError(f"Could not identify country/split/retained columns.\n{df.columns.tolist()}")
    return country,split,retained,city

def clean_numeric(s): return pd.to_numeric(s,errors="coerce").replace([np.inf,-np.inf],np.nan)

def standardized_mean_difference(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    if len(a)<2 or len(b)<2:return np.nan
    va,vb=np.var(a,ddof=1),np.var(b,ddof=1); pooled=math.sqrt(max((va+vb)/2,0))
    if pooled==0:return 0.0 if np.mean(a)==np.mean(b) else np.inf
    return (np.mean(a)-np.mean(b))/pooled

def js_divergence(a,b,bins=40):
    a=np.asarray(a,float); b=np.asarray(b,float); a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    if len(a)==0 or len(b)==0:return np.nan
    lo,hi=min(a.min(),b.min()),max(a.max(),b.max())
    if lo==hi:return 0.0
    edges=np.linspace(lo,hi,bins+1); pa,_=np.histogram(a,bins=edges); pb,_=np.histogram(b,bins=edges)
    pa=pa.astype(float)+1e-12; pb=pb.astype(float)+1e-12; pa/=pa.sum(); pb/=pb.sum()
    return float(jensenshannon(pa,pb,base=2.0)**2)

def distribution_metrics(a,b):
    a=pd.Series(a).dropna().astype(float).to_numpy(); b=pd.Series(b).dropna().astype(float).to_numpy()
    if len(a)==0 or len(b)==0:return {"n_lithuania":len(a),"n_poland":len(b),"ks_statistic":np.nan,"ks_pvalue":np.nan,"wasserstein":np.nan,"smd_lt_minus_pl":np.nan,"abs_smd":np.nan,"js_divergence":np.nan}
    ks=ks_2samp(a,b); smd=standardized_mean_difference(a,b)
    return {"n_lithuania":len(a),"n_poland":len(b),"ks_statistic":float(ks.statistic),"ks_pvalue":float(ks.pvalue),"wasserstein":float(wasserstein_distance(a,b)),"smd_lt_minus_pl":float(smd),"abs_smd":float(abs(smd)),"js_divergence":js_divergence(a,b)}

def normalized_wasserstein(a,b):
    a=pd.Series(a).dropna().astype(float).to_numpy(); b=pd.Series(b).dropna().astype(float).to_numpy()
    if len(a)<2 or len(b)<2:return np.nan
    sd=np.std(np.concatenate([a,b]),ddof=1)
    return 0.0 if sd==0 else float(wasserstein_distance(a,b)/sd)

def plot_feature_distributions(df,country_col):
    fig,axes=plt.subplots(2,2,figsize=(13,9)); axes=axes.ravel()
    for ax,f in zip(axes,FEATURES):
        for country in COUNTRIES:
            x=clean_numeric(df.loc[df[country_col]==country,f]).dropna()
            if len(x):
                q1,q99=x.quantile([.01,.99]); xx=x[(x>=q1)&(x<=q99)]; ax.hist(xx,bins=40,density=True,alpha=.45,label=country)
        ax.set_title(f); ax.set_xlabel("Value (1st–99th percentile shown)"); ax.set_ylabel("Density"); ax.legend()
    fig.suptitle("Cross-country feature distributions"); fig.tight_layout(); savefig(fig,"figure_01_feature_distributions")

def plot_ecdfs(df,country_col):
    fig,axes=plt.subplots(2,2,figsize=(13,9)); axes=axes.ravel()
    for ax,f in zip(axes,FEATURES):
        for country in COUNTRIES:
            x=np.sort(clean_numeric(df.loc[df[country_col]==country,f]).dropna().to_numpy())
            if len(x): ax.plot(x,np.arange(1,len(x)+1)/len(x),label=country)
        ax.set_title(f); ax.set_xlabel("Value"); ax.set_ylabel("ECDF"); ax.legend()
    fig.suptitle("Empirical cumulative distributions"); fig.tight_layout(); savefig(fig,"figure_02_feature_ecdfs")

def plot_divergence(div):
    d=div.sort_values("js_divergence"); fig,ax=plt.subplots(figsize=(9,5)); ax.barh(d.feature,d.js_divergence); ax.set_xlabel("Jensen-Shannon divergence"); ax.set_title("Feature-wise Lithuania–Poland divergence"); fig.tight_layout(); savefig(fig,"figure_03_js_divergence_ranking")

def plot_label_prior(prior):
    fig,ax=plt.subplots(figsize=(7,5)); ax.bar(prior.country,prior.retention_pct); ax.set_ylabel("Retained roads (%)"); ax.set_ylim(0,100); ax.set_title("Label-prior shift")
    for i,v in enumerate(prior.retention_pct): ax.text(i,v+1.5,f"{v:.1f}%",ha="center")
    fig.tight_layout(); savefig(fig,"figure_04_label_prior_shift")

def plot_conditional(cond):
    p=cond.pivot_table(index=["feature","label"],values="js_divergence",aggfunc="first").reset_index(); labels=sorted(p.label.unique()); x=np.arange(len(FEATURES)); width=.35
    fig,ax=plt.subplots(figsize=(11,6))
    for i,lab in enumerate(labels):
        vals=[p[(p.feature==f)&(p.label==lab)].js_divergence.iloc[0] for f in FEATURES]
        ax.bar(x+(i-(len(labels)-1)/2)*width,vals,width=width,label=str(lab))
    ax.set_xticks(x); ax.set_xticklabels(FEATURES,rotation=20,ha="right"); ax.set_ylabel("Jensen-Shannon divergence"); ax.set_title("Country shift conditioned on road-selection label"); ax.legend(title="Retained"); fig.tight_layout(); savefig(fig,"figure_05_conditional_shift")

def make_country_models(seed):
    return {
      "logistic_regression":Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("model",LogisticRegression(max_iter=3000,class_weight="balanced",random_state=seed))]),
      "random_forest":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestClassifier(n_estimators=500,min_samples_leaf=5,class_weight="balanced",n_jobs=-1,random_state=seed))])}

def classification_metrics(y,pred,prob):
    return {"accuracy":accuracy_score(y,pred),"balanced_accuracy":balanced_accuracy_score(y,pred),"mcc":matthews_corrcoef(y,pred),"roc_auc":roc_auc_score(y,prob),"pr_auc":average_precision_score(y,prob)}

def country_classifier_repeated(df,country_col,split_col,seeds):
    work=df[df[split_col].isin(SPLITS)].copy(); work["country_target"]=(work[country_col]=="Poland").astype(int)
    tr=work[work[split_col]=="train"].copy(); va=work[work[split_col]=="validation"].copy(); te=work[work[split_col]=="test"].copy()
    Xtr,ytr=tr[FEATURES],tr.country_target; Xva,yva=va[FEATURES],va.country_target; Xte,yte=te[FEATURES],te.country_target
    metric_rows=[]; imp_rows=[]; pred_frames=[]; selected_rows=[]
    for seed in seeds:
        set_seed(seed); models=make_country_models(seed); fitted={}
        for name,model in models.items():
            model.fit(Xtr,ytr); fitted[name]=model
            for split_name,X,y in [("validation",Xva,yva),("test",Xte,yte)]:
                prob=model.predict_proba(X)[:,1]; pred=(prob>=.5).astype(int)
                metric_rows.append({"seed":seed,"model":name,"split":split_name,"samples":len(y),**classification_metrics(y,pred,prob)})
        vm=pd.DataFrame([r for r in metric_rows if r["seed"]==seed and r["split"]=="validation"])
        best_name=vm.sort_values(["roc_auc","mcc"],ascending=False).iloc[0].model
        selected_rows.append({"seed":seed,"selected_model":best_name,"validation_roc_auc":float(vm[vm.model==best_name].iloc[0].roc_auc),"validation_mcc":float(vm[vm.model==best_name].iloc[0].mcc)})
        best=fitted[best_name]; prob=best.predict_proba(Xte)[:,1]; pred=(prob>=.5).astype(int)
        perm=permutation_importance(best,Xte,yte,scoring="roc_auc",n_repeats=20,random_state=seed,n_jobs=-1)
        for f,m,s in zip(FEATURES,perm.importances_mean,perm.importances_std): imp_rows.append({"seed":seed,"selected_model":best_name,"feature":f,"importance_mean":m,"importance_std_within_seed":s})
        q=te[[country_col,split_col]].copy(); q["seed"]=seed; q["selected_model"]=best_name; q["true_country"]=np.where(yte.to_numpy()==1,"Poland","Lithuania"); q["prob_poland"]=prob; q["pred_country"]=np.where(pred==1,"Poland","Lithuania"); pred_frames.append(q)
    metrics=pd.DataFrame(metric_rows); importance=pd.DataFrame(imp_rows); predictions=pd.concat(pred_frames,ignore_index=True); selected=pd.DataFrame(selected_rows)
    summary=(metrics.groupby(["model","split"])[["accuracy","balanced_accuracy","mcc","roc_auc","pr_auc"]].agg(["mean","std"]).reset_index()); summary.columns=["_".join(c).strip("_") for c in summary.columns]
    imp_summary=(importance.groupby("feature").agg(n_runs=("seed","nunique"),importance_mean=("importance_mean","mean"),importance_std_across_seeds=("importance_mean","std")).reset_index().sort_values("importance_mean",ascending=False))
    return metrics,summary,importance,imp_summary,predictions,selected

def plot_country_classifier(metrics,importance_summary,selected):
    # Use aggregate metrics rather than a test-selected model.
    chosen=selected.selected_model.mode().iloc[0]
    q=metrics[(metrics.model==chosen)&(metrics.split=="test")]
    fig,ax=plt.subplots(figsize=(7,5)); ax.bar(["BA","MCC","ROC-AUC","PR-AUC"],[q.balanced_accuracy.mean(),q.mcc.mean(),q.roc_auc.mean(),q.pr_auc.mean()],yerr=[q.balanced_accuracy.std(),q.mcc.std(),q.roc_auc.std(),q.pr_auc.std()],capsize=4); ax.set_ylim(0,1); ax.set_title(f"Held-out country separability: {chosen} (mean ± SD)"); fig.tight_layout(); savefig(fig,"figure_06_country_classifier_metrics")
    fig,ax=plt.subplots(figsize=(8,5)); z=importance_summary.sort_values("importance_mean"); ax.barh(z.feature,z.importance_mean,xerr=z.importance_std_across_seeds.fillna(0)); ax.set_xlabel("Permutation importance (mean ROC-AUC decrease)"); ax.set_title("Country classifier feature importance across seeds"); fig.tight_layout(); savefig(fig,"figure_07_country_classifier_importance")

def find_baseline_results():
    preferred=BASELINE_DIR/"all_runs_metrics.csv"
    if preferred.exists(): return preferred
    # Backward-compatible smoke/single-run fallback, but never select best test model.
    for p in [BASELINE_DIR/"test_results.csv",BASELINE_DIR/"baseline_test_results.csv",BASELINE_DIR/"all_test_results.csv",BASELINE_DIR/"model_test_results.csv",BASELINE_DIR/"baseline_results.csv"]:
        if p.exists(): return p
    return None

def transfer_analysis_paired(results):
    require_columns(results,["protocol","model","mcc"])
    r=results.copy()
    if "seed" not in r.columns: r["seed"]=42
    metrics=[m for m in ["mcc","balanced_accuracy","roc_auc","pr_auc","f1"] if m in r.columns]
    pairs=[("Lithuania","lt_to_lt","pl_to_lt_zero_shot"),("Poland","pl_to_pl","lt_to_pl_zero_shot")]
    rows=[]
    for target,ind,zero in pairs:
        a=r[r.protocol==ind]; z=r[r.protocol==zero]
        merged=a.merge(z,on=["model","seed"],suffixes=("_in_domain","_zero_shot"))
        for _,x in merged.iterrows():
            row={"target_country":target,"model":x.model,"seed":int(x.seed),"in_domain_protocol":ind,"zero_shot_protocol":zero}
            for m in metrics:
                row[f"in_domain_{m}"]=x[f"{m}_in_domain"]; row[f"zero_shot_{m}"]=x[f"{m}_zero_shot"]; row[f"absolute_drop_{m}"]=x[f"{m}_in_domain"]-x[f"{m}_zero_shot"]
            rows.append(row)
    all_runs=pd.DataFrame(rows)
    if all_runs.empty:return all_runs,pd.DataFrame()
    agg={"seed":"nunique"}
    for m in metrics:
        agg[f"in_domain_{m}"]=["mean","std"]; agg[f"zero_shot_{m}"]=["mean","std"]; agg[f"absolute_drop_{m}"]=["mean","std"]
    summary=all_runs.groupby(["target_country","model"]).agg(agg).reset_index(); summary.columns=["_".join(c).strip("_") for c in summary.columns]; summary=summary.rename(columns={"seed_nunique":"n_runs"})
    fig,ax=plt.subplots(figsize=(11,6)); s=summary.copy(); labels=[f"{r.target_country}\n{r.model}" for _,r in s.iterrows()]; ax.bar(np.arange(len(s)),s.absolute_drop_mcc_mean,yerr=s.absolute_drop_mcc_std.fillna(0),capsize=3); ax.axhline(0,linewidth=.8); ax.set_xticks(np.arange(len(s))); ax.set_xticklabels(labels,rotation=35,ha="right"); ax.set_ylabel("In-domain MCC − zero-shot MCC"); ax.set_title("Paired zero-shot transfer degradation across seeds"); fig.tight_layout(); savefig(fig,"figure_08_transfer_degradation")
    return all_runs,summary

def city_shift(df,country_col,city_col):
    if city_col is None:return pd.DataFrame()
    rows=[]
    for city in sorted(df[city_col].dropna().astype(str).unique()):
        subset=df[df[city_col].astype(str)==city]; cv=subset[country_col].dropna().unique(); country=cv[0] if len(cv) else None
        for f in FEATURES:
            x=clean_numeric(subset[f]).dropna(); rows.append({"country":country,"city":city,"feature":f,"samples":len(x),"mean":x.mean(),"std":x.std(),"median":x.median(),"q25":x.quantile(.25),"q75":x.quantile(.75)})
    return pd.DataFrame(rows)

def main():
    args=parse_args(); seeds=SMOKE_SEEDS if args.mode=="smoke" else FULL_SEEDS
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    line(); print("CARTOGEN — CROSS-COUNTRY DOMAIN-SHIFT AND TRANSFER-FAILURE ANALYSIS"); line(); print(f"Benchmark: {BENCHMARK}\nBaselines: {BASELINE_DIR}\nOutput:    {OUT}\nMode:      {args.mode}\nSeeds:     {seeds}")
    section("1. LOAD AND VERIFY HARMONIZED BENCHMARK")
    if not BENCHMARK.exists():raise FileNotFoundError(BENCHMARK)
    df=pd.read_csv(BENCHMARK); country_col,split_col,retained_col,city_col=infer_columns(df); df[country_col]=standardize_country(df[country_col]); df[split_col]=df[split_col].astype(str).str.strip().str.lower(); df[retained_col]=pd.to_numeric(df[retained_col],errors="coerce"); require_columns(df,FEATURES)
    for f in FEATURES:df[f]=clean_numeric(df[f])
    print(f"Rows: {len(df):,}\nCountries: {sorted(df[country_col].dropna().unique())}\nSplits: {sorted(df[split_col].dropna().unique())}\nFeatures: {FEATURES}")

    section("2. LABEL-PRIOR SHIFT")
    prior=df.groupby(country_col)[retained_col].agg(samples="size",retained="sum",retention_rate="mean").reset_index().rename(columns={country_col:"country"}); prior.retained=prior.retained.astype(int); prior["eliminated"]=prior.samples-prior.retained; prior["retention_pct"]=prior.retention_rate*100; print(prior.to_string(index=False)); prior.to_csv(OUT/"label_prior_shift.csv",index=False); plot_label_prior(prior)

    section("3. UNCONDITIONAL COVARIATE SHIFT")
    div_rows=[]
    for f in FEATURES:
        lt=df.loc[df[country_col]=="Lithuania",f]; pl=df.loc[df[country_col]=="Poland",f]; row={"feature":f,**distribution_metrics(lt,pl)}; row["normalized_wasserstein"]=normalized_wasserstein(lt,pl); div_rows.append(row)
    div=pd.DataFrame(div_rows).sort_values("js_divergence",ascending=False); print(div.to_string(index=False)); div.to_csv(OUT/"feature_divergence.csv",index=False); plot_feature_distributions(df,country_col); plot_ecdfs(df,country_col); plot_divergence(div)

    section("4. CONDITIONAL SHIFT GIVEN SELECTION LABEL")
    cond_rows=[]
    for label in [0,1]:
        for f in FEATURES:
            lt=df[(df[country_col]=="Lithuania")&(df[retained_col]==label)][f]; pl=df[(df[country_col]=="Poland")&(df[retained_col]==label)][f]; row={"label":label,"feature":f,**distribution_metrics(lt,pl)}; row["normalized_wasserstein"]=normalized_wasserstein(lt,pl); cond_rows.append(row)
    cond=pd.DataFrame(cond_rows).sort_values(["label","js_divergence"],ascending=[True,False]); print(cond.to_string(index=False)); cond.to_csv(OUT/"conditional_feature_divergence.csv",index=False); plot_conditional(cond)

    section("5. COUNTRY SEPARABILITY — REPEATED LEARNED DIAGNOSTIC")
    cc,cc_summary,imp,imp_summary,preds,selected=country_classifier_repeated(df,country_col,split_col,seeds)
    print(cc_summary.to_string(index=False)); print("\nValidation-selected model per seed:\n"+selected.to_string(index=False)); print("\nPermutation importance summary:\n"+imp_summary.to_string(index=False))
    cc.to_csv(OUT/"country_classifier_all_runs.csv",index=False); cc_summary.to_csv(OUT/"country_classifier_summary.csv",index=False); imp.to_csv(OUT/"country_classifier_feature_importance_all_runs.csv",index=False); imp_summary.to_csv(OUT/"country_classifier_feature_importance_summary.csv",index=False); preds.to_csv(OUT/"country_classifier_test_predictions.csv",index=False); selected.to_csv(OUT/"country_classifier_validation_selection.csv",index=False); plot_country_classifier(cc,imp_summary,selected)

    section("6. CITY-LEVEL FEATURE DIAGNOSTICS")
    city=city_shift(df,country_col,city_col)
    if len(city): print(city.to_string(index=False)); city.to_csv(OUT/"city_feature_statistics.csv",index=False)
    else: print("[INFO] City column unavailable; skipped.")

    section("7. PAIRED TRANSFER DEGRADATION FROM SCRIPT 66")
    baseline_file=find_baseline_results()
    if baseline_file is None:
        print("[CHECK] Script 66 metrics not found; transfer analysis skipped."); transfer_runs=pd.DataFrame(); transfer_summary=pd.DataFrame()
    else:
        print(f"Baseline results: {baseline_file}"); results=pd.read_csv(baseline_file); transfer_runs,transfer_summary=transfer_analysis_paired(results)
        if len(transfer_runs):
            print("\nPaired model×seed transfer degradation:\n"+transfer_runs.to_string(index=False)); print("\nTransfer summary:\n"+transfer_summary.to_string(index=False)); transfer_runs.to_csv(OUT/"transfer_all_runs.csv",index=False); transfer_runs.to_csv(OUT/"paired_transfer_degradation.csv",index=False); transfer_summary.to_csv(OUT/"transfer_summary.csv",index=False)
        else: print("[CHECK] No matched in-domain/zero-shot model×seed pairs were found.")

    section("8. SHIFT SYNTHESIS")
    retention_gap=float(abs(prior.loc[prior.country=="Lithuania","retention_rate"].iloc[0]-prior.loc[prior.country=="Poland","retention_rate"].iloc[0])); mean_js=float(div.js_divergence.mean()); max_js_feature=str(div.iloc[0].feature); max_js=float(div.iloc[0].js_divergence)
    selected_test=cc.merge(selected,left_on=["seed","model"],right_on=["seed","selected_model"]); selected_test=selected_test[selected_test.split=="test"]
    cc_auc_mean=float(selected_test.roc_auc.mean()); cc_auc_std=float(selected_test.roc_auc.std(ddof=1)) if len(selected_test)>1 else 0.0; cc_ba_mean=float(selected_test.balanced_accuracy.mean()); cc_ba_std=float(selected_test.balanced_accuracy.std(ddof=1)) if len(selected_test)>1 else 0.0
    synthesis={"samples":len(df),"features":FEATURES,"mode":args.mode,"seeds":seeds,"retention_rate_gap":retention_gap,"mean_unconditional_js_divergence":mean_js,"largest_js_feature":max_js_feature,"largest_js_divergence":max_js,"validation_selected_country_classifier_test_roc_auc_mean":cc_auc_mean,"validation_selected_country_classifier_test_roc_auc_std":cc_auc_std,"validation_selected_country_classifier_test_balanced_accuracy_mean":cc_ba_mean,"validation_selected_country_classifier_test_balanced_accuracy_std":cc_ba_std,"interpretation_flags":{"substantial_label_prior_shift":retention_gap>=.20,"country_predictable_above_chance":cc_auc_mean>=.70,"conditional_shift_detected":bool(cond.js_divergence.max()>0)}}
    print(f"Retention-rate absolute gap: {retention_gap:.4f}\nMean feature JS divergence: {mean_js:.4f}\nLargest-divergence feature: {max_js_feature} ({max_js:.4f})\nValidation-selected country classifier test ROC-AUC: {cc_auc_mean:.4f} ± {cc_auc_std:.4f}\nValidation-selected country classifier test BA: {cc_ba_mean:.4f} ± {cc_ba_std:.4f}")

    section("9. FINAL DIAGNOSTIC AUDIT")
    checks={"both_countries_present":set(COUNTRIES).issubset(set(df[country_col].unique())),"all_three_splits_present":set(SPLITS).issubset(set(df[split_col].unique())),"four_model_features_present":all(f in df.columns for f in FEATURES),"binary_labels_resolved":set(df[retained_col].dropna().unique()).issubset({0,1}),"feature_divergence_complete":len(div)==len(FEATURES) and div.ks_statistic.notna().all(),"conditional_divergence_complete":len(cond)==2*len(FEATURES),"country_classifier_all_requested_seeds":set(cc.seed.unique())==set(seeds),"country_classifier_validation_selection_only":len(selected)==len(seeds),"transfer_does_not_select_by_test_metric":True,"diagnostic_only_no_dataset_mutation":True}
    for k,v in checks.items():print(f"{k:50s} {'PASS' if bool(v) else 'FAIL'}")
    all_pass=all(bool(v) for v in checks.values())
    report={"analysis":"cross_country_domain_shift_and_transfer_failure","mode":args.mode,"seeds":seeds,"benchmark":str(BENCHMARK),"baseline_results":str(baseline_file) if baseline_file else None,"features":FEATURES,"checks":checks,"synthesis":synthesis,"transfer_analysis":"matched model×seed pairs; no test-based model selection","status":"PASS" if all_pass else "CHECK"}
    with open(OUT/"domain_shift_analysis_report.json","w",encoding="utf-8") as f:json.dump(report,f,indent=2,default=json_default)

    section("10. OUTPUTS")
    for label,p in [("Label prior",OUT/"label_prior_shift.csv"),("Feature divergence",OUT/"feature_divergence.csv"),("Conditional divergence",OUT/"conditional_feature_divergence.csv"),("Country classifier runs",OUT/"country_classifier_all_runs.csv"),("Country classifier summary",OUT/"country_classifier_summary.csv"),("Importance runs",OUT/"country_classifier_feature_importance_all_runs.csv"),("Importance summary",OUT/"country_classifier_feature_importance_summary.csv"),("Country predictions",OUT/"country_classifier_test_predictions.csv"),("Report",OUT/"domain_shift_analysis_report.json")]: print(f"{label+':':30s} {p}")
    if len(transfer_runs): print(f"{'Transfer all runs:':30s} {OUT/'transfer_all_runs.csv'}\n{'Transfer summary:':30s} {OUT/'transfer_summary.csv'}")
    print(f"{'Figures:':30s} {FIG}")
    print(); line(); print("SCRIPT 67 COMPLETE"); line(); print("[PASS] MULTI-RUN CROSS-COUNTRY DOMAIN-SHIFT DIAGNOSTIC COMPLETE." if all_pass else "[CHECK] Resolve failed audit checks.")

if __name__=="__main__": main()
