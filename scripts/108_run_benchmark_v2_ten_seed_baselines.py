#!/usr/bin/env python3
"""Run and audit 10 matched-seed Benchmark V2 classical baseline experiments.

Runs the existing Script 106 without changing its model selection, folds, feature
sets, or source-only zero-shot protocol. Outputs exploratory statistics only.

Usage:
 python scripts/108_run_benchmark_v2_ten_seed_baselines.py
 python scripts/108_run_benchmark_v2_ten_seed_baselines.py --seeds 42 43 --force
 python scripts/108_run_benchmark_v2_ten_seed_baselines.py --summarize-only
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (matthews_corrcoef, balanced_accuracy_score,
                             f1_score, roc_auc_score, average_precision_score)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT106 = ROOT / 'scripts/106_train_benchmark_v2_classical_baselines.py'
INPUT = ROOT / 'data/processed/cross_country/benchmark_v2_unified_labels_v1/road_graph_nodes_v2.csv'
BASE = ROOT / 'data/processed/cross_country/benchmark_v2_classical_baselines_v1'
OUT = ROOT / 'data/processed/cross_country/benchmark_v2_ten_seed_summary_v1'
PROTOCOLS = ('lt_to_lt', 'pl_to_pl', 'lt_to_pl_zero_shot',
             'pl_to_lt_zero_shot', 'joint_to_lt', 'joint_to_pl')
METRICS = ('mcc', 'balanced_accuracy', 'f1', 'roc_auc', 'pr_auc')
ARTIFACTS = ('spatial_cv_candidate_results.csv', 'selected_protocol_results.csv',
             'selected_test_predictions.csv')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_seed(seed: int, folds: int, input_hash: str, code_hash: str,
                require_code_match: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folder = BASE / f'seed_{seed}'
    paths = [folder / p for p in ARTIFACTS]
    manifest_path = folder / 'experiment_manifest.json'
    for p in [*paths, manifest_path]:
        if not p.is_file():
            raise RuntimeError(f'Seed {seed}: missing {p}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if int(manifest['seed']) != seed or int(manifest['folds']) != folds:
        raise RuntimeError(f'Seed {seed}: manifest seed/folds mismatch')
    if manifest.get('input_sha256') != input_hash:
        raise RuntimeError(f'Seed {seed}: benchmark input hash mismatch')
    for p in paths:
        if manifest.get('output_hashes', {}).get(p.name) != sha256(p):
            raise RuntimeError(f'Seed {seed}: output hash mismatch for {p.name}')
    # Script 106 manifests do not record the source-code hash. This runner
    # records the hash for new runs and refuses to claim verification of old ones.
    marker = folder / 'script108_source_hash.json'
    if marker.is_file():
        prior = json.loads(marker.read_text(encoding='utf-8'))
        if prior.get('script106_sha256') != code_hash:
            raise RuntimeError(f'Seed {seed}: Script 106 source differs from recorded run')
    elif require_code_match:
        raise RuntimeError(f'Seed {seed}: no source hash marker; rerun with --force to establish code provenance')
    cv, res, pred = [pd.read_csv(p, low_memory=False) for p in paths]
    for name, table in [('CV', cv), ('results', res), ('predictions', pred)]:
        if not {'seed', 'protocol'}.issubset(table.columns):
            raise RuntimeError(f'Seed {seed}: {name} missing seed/protocol')
        if not table.seed.eq(seed).all():
            raise RuntimeError(f'Seed {seed}: {name} contains other seeds')
    if set(res.protocol) != set(PROTOCOLS) or len(res) != len(PROTOCOLS):
        raise RuntimeError(f'Seed {seed}: missing/duplicated protocols in results')
    if set(cv.protocol) != set(PROTOCOLS) or not cv.groupby('protocol').size().eq(12).all():
        raise RuntimeError(f'Seed {seed}: expected 12 CV candidates per protocol')
    if set(pred.protocol) != set(PROTOCOLS):
        raise RuntimeError(f'Seed {seed}: missing prediction protocols')
    if pred.duplicated(['protocol', 'sample_uid']).any():
        raise RuntimeError(f'Seed {seed}: duplicate prediction identifiers')
    for _, row in res.iterrows():
        p = pred.loc[pred.protocol.eq(row.protocol)]
        if len(p) != int(row.test_n):
            raise RuntimeError(f'Seed {seed} {row.protocol}: test count mismatch')
        if not p.threshold.eq(row.threshold).all():
            raise RuntimeError(f'Seed {seed} {row.protocol}: threshold mismatch')
        if not p.prediction.to_numpy(dtype=int).tolist() == (p.probability.to_numpy() >= row.threshold).astype(int).tolist():
            raise RuntimeError(f'Seed {seed} {row.protocol}: threshold predictions mismatch')
        y = p.label.to_numpy(dtype=int)
        yh = p.prediction.to_numpy(dtype=int)
        prob = p.probability.to_numpy(dtype=float)
        recalculated = dict(mcc=matthews_corrcoef(y, yh),
                            balanced_accuracy=balanced_accuracy_score(y, yh),
                            f1=f1_score(y, yh, zero_division=0),
                            roc_auc=roc_auc_score(y, prob),
                            pr_auc=average_precision_score(y, prob))
        for metric, value in recalculated.items():
            if not np.isclose(float(row['test_' + metric]), float(value), atol=1e-9, rtol=1e-9):
                raise RuntimeError(f'Seed {seed} {row.protocol}: {metric} not reproducible')
        if row.protocol.endswith('zero_shot') and row.threshold_selection_country != row.source:
            raise RuntimeError(f'Seed {seed} {row.protocol}: target validation used for zero-shot')
        if not row.protocol.endswith('zero_shot') and row.threshold_selection_country != row.target:
            raise RuntimeError(f'Seed {seed} {row.protocol}: unexpected validation country')
    return cv, res, pred


def summarize(results: pd.DataFrame, preds: pd.DataFrame, seeds: list[int]):
    metrics = ['test_' + m for m in METRICS] + ['validation_mcc', 'cv_mcc_at_0_5', 'threshold']
    summary = results.groupby('protocol', sort=False)[metrics].agg(['mean', 'std', 'min', 'max'])
    summary.columns = ['_'.join(pair) for pair in summary.columns]
    summary = summary.reset_index()
    summary.insert(1, 'n_seeds', len(seeds))
    summary.to_csv(OUT / 'protocol_mean_std_min_max.csv', index=False)

    # All protocols share the same seeds, so differences are matched by seed.
    pivot = results.pivot(index='seed', columns='protocol', values='test_mcc')
    pairs = []
    for a in PROTOCOLS:
        for b in PROTOCOLS:
            if a >= b:
                continue
            delta = (pivot[a] - pivot[b]).dropna()
            pairs.append(dict(protocol_a=a, protocol_b=b, n_seeds=len(delta),
                              mean_mcc_difference=float(delta.mean()),
                              sd_mcc_difference=float(delta.std(ddof=1)) if len(delta)>1 else np.nan,
                              wins_a=int((delta > 0).sum()), ties=int((delta == 0).sum()),
                              wins_b=int((delta < 0).sum())))
    pd.DataFrame(pairs).to_csv(OUT / 'paired_seed_mcc_differences.csv', index=False)

    selection = (results.groupby(['protocol', 'model', 'feature_set'], dropna=False)
                 .size().rename('selected_seeds').reset_index())
    selection.to_csv(OUT / 'model_selection_frequencies.csv', index=False)
    results.to_csv(OUT / 'all_seed_protocol_results.csv', index=False)
    preds.to_csv(OUT / 'all_seed_selected_test_predictions.csv', index=False)
    # Joint models share a training pool; report evaluation-country outcomes separately.
    joint = summary[summary.protocol.isin(['joint_to_lt', 'joint_to_pl'])].copy()
    joint.to_csv(OUT / 'joint_country_test_summary.csv', index=False)
    print('\nTEN-SEED EXPLORATORY TEST SUMMARY')
    print(summary[['protocol', 'n_seeds', 'test_mcc_mean', 'test_mcc_std',
                   'test_balanced_accuracy_mean', 'test_roc_auc_mean',
                   'test_pr_auc_mean']].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nNote: joint-country TEST summaries are not country-specific CV scores.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(42, 52)))
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--force', action='store_true', help='Rerun even completed seeds; overwrites those seed outputs')
    ap.add_argument('--summarize-only', action='store_true', help='Do not train; verify and aggregate existing runs')
    ap.add_argument('--allow-unverified-existing', action='store_true',
                    help='Accept existing Script 106 runs without a source-code hash marker (provenance warning)')
    args = ap.parse_args()
    seeds = args.seeds
    if len(seeds) != len(set(seeds)) or not seeds:
        ap.error('Seeds must be nonempty and unique')
    if args.folds < 2:
        ap.error('--folds must be >=2')
    if not SCRIPT106.is_file() or not INPUT.is_file():
        raise FileNotFoundError(f'Missing Script 106 or V2 input: {SCRIPT106} / {INPUT}')
    OUT.mkdir(parents=True, exist_ok=True)
    input_hash, code_hash = sha256(INPUT), sha256(SCRIPT106)
    all_cv, all_res, all_pred, provenance_warnings = [], [], [], []
    for seed in seeds:
        folder = BASE / f'seed_{seed}'
        complete = all((folder / p).exists() for p in [*ARTIFACTS, 'experiment_manifest.json'])
        marker = folder / 'script108_source_hash.json'
        if not args.summarize_only and (args.force or not complete or (complete and not marker.exists() and not args.allow_unverified_existing)):
            print(f'\n=== Training seed {seed} ===', flush=True)
            command = [sys.executable, str(SCRIPT106), '--seed', str(seed), '--folds', str(args.folds)]
            subprocess.run(command, cwd=ROOT, check=True)
            marker.write_text(json.dumps({'seed': seed, 'script106_sha256': code_hash,
                                          'input_sha256': input_hash,
                                          'runner': Path(__file__).name}, indent=2), encoding='utf-8')
        elif not marker.exists():
            if not args.allow_unverified_existing:
                raise RuntimeError(
                    f'Seed {seed} already exists without Script 106 code provenance. '
                    f'Run normally to regenerate, or --allow-unverified-existing to accept with a warning.')
            provenance_warnings.append(seed)
        cv, res, pred = verify_seed(seed, args.folds, input_hash, code_hash,
                                    require_code_match=seed not in provenance_warnings)
        all_cv.append(cv)
        all_res.append(res)
        all_pred.append(pred)
        print(f'Verified seed {seed}: 6 protocols, 72 CV candidates, predictions and metrics', flush=True)
    cv_all = pd.concat(all_cv, ignore_index=True)
    results = pd.concat(all_res, ignore_index=True)
    preds = pd.concat(all_pred, ignore_index=True)
    if results.duplicated(['seed', 'protocol']).any():
        raise RuntimeError('Duplicate seed/protocol results')
    cv_all.to_csv(OUT / 'all_seed_cv_candidate_results.csv', index=False)
    summarize(results, preds, seeds)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'script106': str(SCRIPT106), 'script106_sha256': code_hash,
                'benchmark_v2_sha256': input_hash, 'seeds': seeds, 'folds': args.folds,
                'candidate_models_per_protocol': 12, 'protocols': list(PROTOCOLS),
                'source_provenance_unverified_existing_seeds': provenance_warnings,
                'exploratory_only': True,
                'limitations': [
                    'Historical test roads were previously examined in V1-V10 and are not untouched.',
                    'C2 road correspondence labels are provisional and not manually validated.',
                    'Static Script 107 checks do not eliminate all potential indirect leakage.',
                    'Paired seed comparisons do not substitute for independent spatial test areas.',
                    'Joint pooled CV country imbalance has not been resolved; joint results are reported separately by target country.'
                ]}
    manifest['output_sha256'] = {p.name: sha256(p) for p in OUT.glob('*.csv')}
    (OUT / 'ten_seed_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'\nSaved: {OUT}\nEXPLORATORY ONLY. Do not use these results as untouched-holdout evidence.')
    if provenance_warnings:
        print('WARNING: Source code provenance unavailable for pre-existing seeds:', provenance_warnings)

if __name__ == '__main__':
    main()
