#!/usr/bin/env python3
"""CartoGraph-X Script 103: provisional training-only configuration freeze.

Does not relabel roads, modify V1, or evaluate validation/test labels.
C2 is a provisional geometric definition, NOT manually validated ground truth.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

CONFIG_NAME = 'C2_intermediate'
EXPECTED = {'config': CONFIG_NAME, 'search_m': 15.0, 'buffer_m': 5.0,
            'source_min': 0.20, 'target_min': 0.03}
BASE = Path('data/processed/cross_country')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f'Required input not found: {path}')
    return path


def audit_review(path: Path):
    review = pd.read_csv(require(path), keep_default_na=False)
    required = {'country','city','source_input_row','review_status','C1_conservative',
                'C2_intermediate','C3_permissive','transition'}
    if not required.issubset(review.columns):
        raise ValueError(f'Review sheet missing columns: {sorted(required - set(review.columns))}')
    if review.duplicated(['country','city','source_input_row']).any():
        raise ValueError('Duplicate road identities in manual review sheet')
    status = review['review_status'].astype(str).str.strip().str.lower()
    valid_values = {'valid','invalid','uncertain','yes','no','true','false','1','0'}
    validity_cols = [c for c in ('review_validity','source_target_match_valid') if c in review]
    valid_entries = pd.Series(False, index=review.index)
    for col in validity_cols:
        valid_entries |= review[col].astype(str).str.strip().str.lower().isin(valid_values)
    completed = status.isin({'reviewed','complete','completed','done'}) & valid_entries
    summary = {'total_candidates': int(len(review)),
               'review_status_completed': int(status.isin({'reviewed','complete','completed','done'}).sum()),
               'validity_decisions_recorded': int(valid_entries.sum()),
               'completed_with_decision': int(completed.sum()),
               'manual_validation_complete': bool(completed.all() and len(review)>0),
               'country_counts': {str(k):int(v) for k,v in review.groupby('country').size().items()}}
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--review-sheet', type=Path, default=None)
    p.add_argument('--out', type=Path, default=None)
    args = p.parse_args()
    base = args.base
    configs_file = require(base/'benchmark_v2_unified_correspondence_audit_v1'/'configurations.json')
    quality_file = require(base/'benchmark_v2_configuration_assessment_v1'/'train_country_geometric_quality.csv')
    transition_file = require(base/'benchmark_v2_correspondence_verification_v1'/'train_city_transition_counts.csv')
    review_file = args.review_sheet or base/'benchmark_v2_manual_review_package_v1'/'manual_review_sheet.csv'
    out = args.out or base/'benchmark_v2_provisional_configuration_v1'

    original = json.loads(configs_file.read_text(encoding='utf-8'))
    matches = [c for c in original.get('configs', []) if c.get('config') == CONFIG_NAME]
    if len(matches) != 1:
        raise ValueError('Script 99 must define exactly one C2_intermediate configuration')
    config = matches[0]
    for key, value in EXPECTED.items():
        if key not in config or (str(config[key]) != value if isinstance(value, str) else abs(float(config[key])-value)>1e-10):
            raise ValueError(f'Unexpected C2 parameter {key}: {config.get(key)}; expected {value}')

    quality = pd.read_csv(quality_file)
    # Script 100 country summary uses 'n' for the number of training roads.
    # Accept either schema without changing the underlying evidence.
    if 'roads' not in quality.columns and 'n' in quality.columns:
        quality = quality.rename(columns={'n': 'roads'})
    for col in ('country','config','roads','matched'):
        if col not in quality.columns:
            raise ValueError(f'Quality file missing {col}; available columns: {list(quality.columns)}')
    chosen = quality.loc[quality['config'].eq(CONFIG_NAME)].copy()
    if set(chosen['country']) != {'Lithuania','Poland'} or len(chosen) != 2:
        raise ValueError('Expected exactly Lithuania and Poland training-only C2 quality rows')
    chosen['matched'] = pd.to_numeric(chosen['matched'], errors='raise')
    chosen['roads'] = pd.to_numeric(chosen['roads'], errors='raise')
    if not ((chosen['matched']>=0)&(chosen['matched']<=chosen['roads'])).all():
        raise ValueError('Invalid matched/roads totals')
    if (chosen.set_index('country').loc['Lithuania','roads'] != 7465 or
        chosen.set_index('country').loc['Poland','roads'] != 10267):
        raise ValueError('Training road counts differ from audited Script 100 totals; stop and investigate')
    transitions = pd.read_csv(transition_file)
    if 'country' not in transitions:
        raise ValueError('Script 101 transition summary lacks country column')
    if not {'Lithuania','Poland'}.issubset(set(transitions['country'])):
        raise ValueError('Script 101 transition summary lacks a country')

    review = audit_review(review_file)
    if review['completed_with_decision']:
        print(f"WARNING: {review['completed_with_decision']} manually reviewed candidates detected; "
              'this script does not calculate validity rates or claim manual validation.')

    source_paths = [configs_file, quality_file, transition_file, review_file]
    provenance = {str(f.as_posix()):sha256(f) for f in source_paths}
    selection = {
        'benchmark_version': 'CartoGraph-X_Benchmark_V2_provisional',
        'configuration_name': CONFIG_NAME,
        'configuration': config,
        'status': 'PROVISIONAL_NOT_MANUALLY_VALIDATED',
        'decision_basis': 'Training-only geometric coverage/selectivity trade-off in Scripts 99-101; not optimality or label accuracy',
        'selection_policy': 'Do not optimize correspondence criteria using validation/test labels or V1 test performance',
        'source_target_semantics': 'Positive iff at least one target-road candidate meets both directional line-coverage thresholds and search distance; one-to-many source-to-target contributions permitted',
        'candidate_definition': 'Metric projected source/target LineString geometry; distance <= search_m; source coverage computed against target buffer_m; target coverage computed against source buffer_m',
        'positive_definition': 'Matched source road contributes geometrically to target network; not necessarily individually retained unchanged',
        'scope': 'Future Benchmark V2 label generation only; original V1 and existing spatial split assignments remain unchanged',
        'manual_review': review,
        'limitations': [
            'No human correspondence-validity verification completed by this freeze',
            'Different target-network density and label prevalence remain across countries',
            'Matching proximity alone may accept parallel roads or intersection-neighbor candidates',
            'Original test set has been repeatedly examined during model development and is not a fresh confirmatory holdout',
            'Any V2 result on those same geographic roads is exploratory unless a truly untouched holdout is created'
        ],
        'input_sha256': provenance,
        'generated_at_utc': datetime.now(timezone.utc).isoformat()
    }
    out.mkdir(parents=True, exist_ok=True)
    # Refuse silent mutation of an existing freeze, even if later review decisions change.
    freeze = out/'provisional_correspondence_config.json'
    if freeze.exists():
        previous = json.loads(freeze.read_text(encoding='utf-8'))
        if previous.get('input_sha256') != provenance or previous.get('configuration') != config:
            raise FileExistsError(f'Existing freeze has different inputs. Preserve it and specify --out NEW_DIRECTORY: {freeze}')
        print('Existing freeze verified: input hashes and C2 parameters unchanged.')
    else:
        freeze.write_text(json.dumps(selection, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    # Write summaries once; do not silently overwrite an existing freeze.
    summary_file = out/'training_only_c2_summary.csv'
    if not summary_file.exists():
        chosen.sort_values('country').to_csv(summary_file,index=False)
    readme = out/'NEXT_STEPS.txt'
    if not readme.exists():
        readme.write_text(
            'PROVISIONAL C2 CORRESPONDENCE FREEZE - NOT MANUALLY VALIDATED\n'
            'Next: implement Benchmark V2 source-road labels from the frozen JSON.\n'
            'Recompute exact source-target matches for all six cities with the frozen C2 rule.\n'
            'Preserve original spatial tile/split IDs and source-row identities.\n'
            'Keep V1 untouched. Verify geometry, feature, label, and graph lineage.\n'
            'Do not use V1 test results or target-domain test labels to choose thresholds.\n'
            'Create an independent holdout for confirmatory performance claims.\n',encoding='utf-8')

    print('\nPROVISIONAL C2 FREEZE: training-only evidence')
    print(chosen.to_string(index=False))
    print(f"\nManual review: {review['completed_with_decision']}/{review['total_candidates']} completed with a validity decision")
    print('Status: PROVISIONAL_NOT_MANUALLY_VALIDATED')
    print(f'Frozen configuration: {freeze.resolve()}')
    print(f'Summary: {summary_file.resolve()}')
    print('No V1 files, labels, graph records, or validation/test labels were changed or read.')

if __name__ == '__main__':
    main()
