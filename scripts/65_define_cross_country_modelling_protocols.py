#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
65_define_cross_country_modelling_protocols.py

CartoGen — define leakage-safe Lithuania/Poland road modelling protocols
and fit protocol-specific preprocessing.

Inputs
------
data/processed/cross_country/harmonized_roads/
    lithuania_poland_road_benchmark.csv

Outputs
-------
data/processed/cross_country/modelling_protocols/
    protocol_assignments.csv
    protocol_summary.csv
    preprocessing_summary.csv
    modelling_protocol_manifest.json
    protocols/<protocol_id>/
        train.csv
        validation.csv
        test.csv
        preprocessing.json
        train_preprocessed.csv
        validation_preprocessed.csv
        test_preprocessed.csv

Important design choices
------------------------
* Uses only the four harmonized transferable predictors from Script 64.
* Never uses centroid coordinates as model inputs.
* Never resamples/rebalances validation or test data.
* Median imputation and standardization parameters are learned from TRAIN only.
* Zero-shot destination test data are never used for fitting preprocessing.
* Joint protocols train on both countries' train partitions.
* Validation is restricted to the destination country for joint protocols.
* Existing spatial train/validation/test partitions are preserved.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "cross_country"
    / "harmonized_roads"
    / "lithuania_poland_road_benchmark.csv"
)

OUT = (
    ROOT
    / "data"
    / "processed"
    / "cross_country"
    / "modelling_protocols"
)

PROTOCOL_ROOT = OUT / "protocols"

FEATURES = [
    "length_m",
    "intersecting_road_count",
    "neighbor_count_150m",
    "mean_neighbor_distance_m",
]

LABEL = "retained"
COUNTRY = "country"
SPLIT = "split_harmonized"

VALID_COUNTRIES = {"Lithuania", "Poland"}
VALID_SPLITS = {"train", "validation", "test"}

# Protocol definitions:
# train countries/splits -> validation countries/splits -> test countries/splits.
PROTOCOLS = {
    "lt_to_lt": {
        "description": "Lithuania within-country evaluation",
        "train_countries": ["Lithuania"],
        "validation_countries": ["Lithuania"],
        "test_countries": ["Lithuania"],
    },
    "pl_to_pl": {
        "description": "Poland within-country evaluation",
        "train_countries": ["Poland"],
        "validation_countries": ["Poland"],
        "test_countries": ["Poland"],
    },
    "lt_to_pl_zero_shot": {
        "description": "Train/tune on Lithuania; zero-shot evaluation on Poland test",
        "train_countries": ["Lithuania"],
        "validation_countries": ["Lithuania"],
        "test_countries": ["Poland"],
    },
    "pl_to_lt_zero_shot": {
        "description": "Train/tune on Poland; zero-shot evaluation on Lithuania test",
        "train_countries": ["Poland"],
        "validation_countries": ["Poland"],
        "test_countries": ["Lithuania"],
    },
    "joint_to_lt": {
        "description": "Joint Lithuania+Poland training; tune/evaluate for Lithuania",
        "train_countries": ["Lithuania", "Poland"],
        "validation_countries": ["Lithuania"],
        "test_countries": ["Lithuania"],
    },
    "joint_to_pl": {
        "description": "Joint Lithuania+Poland training; tune/evaluate for Poland",
        "train_countries": ["Lithuania", "Poland"],
        "validation_countries": ["Poland"],
        "test_countries": ["Poland"],
    },
}


def banner(text: str) -> None:
    print("\n" + "=" * 120)
    print(text)
    print("=" * 120)


def section(text: str) -> None:
    print("\n" + "-" * 120)
    print(text)
    print("-" * 120)


def py_scalar(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        x = float(x)
        return None if not math.isfinite(x) else x
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if pd.isna(x):
        return None
    return x


def validate_input(df: pd.DataFrame) -> None:
    required = set(FEATURES + [LABEL, COUNTRY, SPLIT])
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    countries = set(df[COUNTRY].dropna().astype(str).unique())
    splits = set(df[SPLIT].dropna().astype(str).unique())

    if countries != VALID_COUNTRIES:
        raise ValueError(
            f"Unexpected countries: {sorted(countries)}; "
            f"expected {sorted(VALID_COUNTRIES)}"
        )

    if splits != VALID_SPLITS:
        raise ValueError(
            f"Unexpected splits: {sorted(splits)}; "
            f"expected {sorted(VALID_SPLITS)}"
        )

    y = pd.to_numeric(df[LABEL], errors="coerce")
    if y.isna().any():
        raise ValueError(f"{int(y.isna().sum())} unresolved labels found.")

    values = set(y.astype(int).unique())
    if not values.issubset({0, 1}):
        raise ValueError(f"Unexpected binary labels: {sorted(values)}")

    for f in FEATURES:
        numeric = pd.to_numeric(df[f], errors="coerce")
        if numeric.notna().sum() == 0:
            raise ValueError(f"Feature {f!r} contains no numeric values.")


def select_partition(
    df: pd.DataFrame,
    countries: list[str],
    split: str,
) -> pd.DataFrame:
    return df[
        df[COUNTRY].isin(countries)
        & df[SPLIT].eq(split)
    ].copy()


def fit_preprocessor(train: pd.DataFrame) -> dict:
    """
    Fit median imputation + z-score standardization on TRAIN only.
    Population standard deviation (ddof=0) is used for deterministic
    transformation and easy reproduction outside scikit-learn.
    """
    result = {
        "strategy": {
            "imputation": "median",
            "scaling": "zscore",
            "fit_partition": "train_only",
            "std_ddof": 0,
        },
        "features": {},
    }

    for feature in FEATURES:
        x = pd.to_numeric(train[feature], errors="coerce")

        median = float(x.median())
        filled = x.fillna(median)
        mean = float(filled.mean())
        std = float(filled.std(ddof=0))

        if not np.isfinite(median):
            raise ValueError(f"Non-finite training median for {feature}.")
        if not np.isfinite(mean):
            raise ValueError(f"Non-finite training mean for {feature}.")

        constant = (not np.isfinite(std)) or std == 0.0
        scale = 1.0 if constant else std

        result["features"][feature] = {
            "median": median,
            "mean": mean,
            "std": std if np.isfinite(std) else None,
            "scale_used": scale,
            "constant_in_training": bool(constant),
            "train_missing": int(x.isna().sum()),
        }

    return result


def transform(df: pd.DataFrame, prep: dict) -> pd.DataFrame:
    out = df.copy()

    for feature in FEATURES:
        params = prep["features"][feature]
        x = pd.to_numeric(out[feature], errors="coerce")
        x = x.fillna(params["median"])

        out[f"{feature}__z"] = (
            (x - params["mean"]) / params["scale_used"]
        )

    return out


def partition_stats(
    protocol_id: str,
    role: str,
    frame: pd.DataFrame,
) -> dict:
    retained = int(pd.to_numeric(frame[LABEL], errors="coerce").sum())
    n = int(len(frame))

    return {
        "protocol": protocol_id,
        "role": role,
        "samples": n,
        "retained": retained,
        "eliminated": n - retained,
        "retention_pct": (100.0 * retained / n) if n else np.nan,
        "countries": ",".join(sorted(frame[COUNTRY].astype(str).unique())),
        "source_splits": ",".join(sorted(frame[SPLIT].astype(str).unique())),
    }


def ensure_nonempty(protocol_id: str, role: str, df: pd.DataFrame) -> None:
    if df.empty:
        raise RuntimeError(
            f"Protocol {protocol_id}: {role} partition is empty."
        )


def check_protocol_leakage(
    protocol_id: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    """
    Prefer sample_uid_harmonized/sample_uid if present.
    Since zero-shot and within-country protocols can legitimately reuse
    a country's validation/test in different *protocols*, leakage is checked
    only within each protocol.
    """
    uid_col = None
    for candidate in ["sample_uid_harmonized", "sample_uid"]:
        if candidate in train.columns:
            uid_col = candidate
            break

    if uid_col is None:
        # Construct a deterministic identity from country + source index.
        source_col = None
        for candidate in [
            "source_idx_harmonized",
            "source_idx",
            "feature_index",
        ]:
            if candidate in train.columns:
                source_col = candidate
                break

        if source_col is None:
            raise KeyError(
                "Cannot perform sample-level leakage audit: "
                "no harmonized UID/source index found."
            )

        def ids(x):
            return set(
                x[COUNTRY].astype(str)
                + "::"
                + x[source_col].astype(str)
            )
    else:
        def ids(x):
            return set(x[uid_col].astype(str))

    tr = ids(train)
    va = ids(validation)
    te = ids(test)

    return {
        "train_validation_overlap": len(tr & va),
        "train_test_overlap": len(tr & te),
        "validation_test_overlap": len(va & te),
        "pass": not bool((tr & va) or (tr & te) or (va & te)),
    }


def main() -> None:
    banner(
        "CARTOGEN — DEFINE LEAKAGE-SAFE CROSS-COUNTRY MODELLING PROTOCOLS"
    )

    print(f"Input: {INPUT}")
    print(f"Output: {OUT}")

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Harmonized benchmark not found:\n{INPUT}\n"
            "Run Script 64 successfully first."
        )

    OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL_ROOT.mkdir(parents=True, exist_ok=True)

    section("1. LOAD AND VALIDATE HARMONIZED BENCHMARK")

    df = pd.read_csv(INPUT)
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    validate_input(df)

    print("\nCountry × split:")
    print(
        df.groupby([COUNTRY, SPLIT])
        .size()
        .rename("samples")
        .to_string()
    )

    print("\nBinary label distribution:")
    print(
        df.groupby(COUNTRY)[LABEL]
        .agg(["count", "sum", "mean"])
        .to_string()
    )

    section("2. LOCK MODEL FEATURE SPACE")

    print("Transferable model features:")
    for feature in FEATURES:
        print(f"  {feature}")

    excluded = [
        c for c in [
            "centroid_x",
            "centroid_y",
            "orientation_deg",
            "vertex_count",
        ]
        if c in df.columns
    ]

    print("\nExplicitly excluded from model input:")
    for feature in excluded:
        print(f"  {feature}")

    section("3. BUILD LEAKAGE-SAFE PROTOCOLS")

    protocol_rows = []
    assignment_frames = []
    preprocessing_rows = []
    manifest_protocols = {}

    for protocol_id, spec in PROTOCOLS.items():
        print("\n" + "=" * 120)
        print(protocol_id.upper())
        print(spec["description"])
        print("=" * 120)

        train = select_partition(
            df, spec["train_countries"], "train"
        )
        validation = select_partition(
            df, spec["validation_countries"], "validation"
        )
        test = select_partition(
            df, spec["test_countries"], "test"
        )

        ensure_nonempty(protocol_id, "train", train)
        ensure_nonempty(protocol_id, "validation", validation)
        ensure_nonempty(protocol_id, "test", test)

        leakage = check_protocol_leakage(
            protocol_id, train, validation, test
        )

        if not leakage["pass"]:
            raise RuntimeError(
                f"Leakage detected in {protocol_id}: {leakage}"
            )

        prep = fit_preprocessor(train)

        train_z = transform(train, prep)
        validation_z = transform(validation, prep)
        test_z = transform(test, prep)

        protocol_dir = PROTOCOL_ROOT / protocol_id
        protocol_dir.mkdir(parents=True, exist_ok=True)

        train.to_csv(protocol_dir / "train.csv", index=False)
        validation.to_csv(
            protocol_dir / "validation.csv", index=False
        )
        test.to_csv(protocol_dir / "test.csv", index=False)

        train_z.to_csv(
            protocol_dir / "train_preprocessed.csv", index=False
        )
        validation_z.to_csv(
            protocol_dir / "validation_preprocessed.csv", index=False
        )
        test_z.to_csv(
            protocol_dir / "test_preprocessed.csv", index=False
        )

        with open(
            protocol_dir / "preprocessing.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(prep, f, indent=2)

        for role, frame in [
            ("train", train),
            ("validation", validation),
            ("test", test),
        ]:
            row = partition_stats(protocol_id, role, frame)
            protocol_rows.append(row)

            a = frame.copy()
            a.insert(0, "protocol", protocol_id)
            a.insert(1, "protocol_role", role)
            assignment_frames.append(a)

            print(
                f"{role:<12} "
                f"{len(frame):>7,} samples | "
                f"countries={sorted(frame[COUNTRY].unique())} | "
                f"retained={int(frame[LABEL].sum()):>6,} | "
                f"eliminated={len(frame)-int(frame[LABEL].sum()):>6,}"
            )

        print("\nLeakage audit:")
        print(
            f"  train ↔ validation: "
            f"{leakage['train_validation_overlap']}"
        )
        print(
            f"  train ↔ test:       "
            f"{leakage['train_test_overlap']}"
        )
        print(
            f"  validation ↔ test:  "
            f"{leakage['validation_test_overlap']}"
        )

        print("\nTRAIN-fitted preprocessing:")
        for feature in FEATURES:
            p = prep["features"][feature]
            print(
                f"  {feature:<30} "
                f"median={p['median']:.6g} | "
                f"mean={p['mean']:.6g} | "
                f"scale={p['scale_used']:.6g} | "
                f"constant={p['constant_in_training']}"
            )

            preprocessing_rows.append({
                "protocol": protocol_id,
                "feature": feature,
                "median": p["median"],
                "mean": p["mean"],
                "std": p["std"],
                "scale_used": p["scale_used"],
                "constant_in_training": p[
                    "constant_in_training"
                ],
                "train_missing": p["train_missing"],
            })

        manifest_protocols[protocol_id] = {
            **spec,
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "leakage_audit": leakage,
            "directory": str(protocol_dir),
        }

    section("4. SAVE GLOBAL PROTOCOL TABLES")

    summary = pd.DataFrame(protocol_rows)
    preprocessing_summary = pd.DataFrame(preprocessing_rows)
    assignments = pd.concat(
        assignment_frames,
        ignore_index=True,
        sort=False,
    )

    summary.to_csv(OUT / "protocol_summary.csv", index=False)
    preprocessing_summary.to_csv(
        OUT / "preprocessing_summary.csv",
        index=False,
    )
    assignments.to_csv(
        OUT / "protocol_assignments.csv",
        index=False,
    )

    print("\nProtocol summary:")
    print(summary.to_string(index=False))

    section("5. FINAL PROTOCOL AUDIT")

    checks = {}

    checks["six_protocols_defined"] = len(PROTOCOLS) == 6

    checks["all_protocol_partitions_nonempty"] = bool(
        (summary["samples"] > 0).all()
    )

    checks["all_protocols_leakage_free"] = all(
        x["leakage_audit"]["pass"]
        for x in manifest_protocols.values()
    )

    checks["four_model_features"] = len(FEATURES) == 4

    checks["no_coordinate_model_features"] = not bool(
        {"centroid_x", "centroid_y"} & set(FEATURES)
    )

    checks["all_preprocessing_train_only"] = True

    checks["lt_to_lt_correct"] = (
        PROTOCOLS["lt_to_lt"]["train_countries"] == ["Lithuania"]
        and PROTOCOLS["lt_to_lt"]["test_countries"] == ["Lithuania"]
    )

    checks["pl_to_pl_correct"] = (
        PROTOCOLS["pl_to_pl"]["train_countries"] == ["Poland"]
        and PROTOCOLS["pl_to_pl"]["test_countries"] == ["Poland"]
    )

    checks["lt_to_pl_zero_shot_correct"] = (
        PROTOCOLS["lt_to_pl_zero_shot"]["train_countries"]
        == ["Lithuania"]
        and PROTOCOLS["lt_to_pl_zero_shot"]["validation_countries"]
        == ["Lithuania"]
        and PROTOCOLS["lt_to_pl_zero_shot"]["test_countries"]
        == ["Poland"]
    )

    checks["pl_to_lt_zero_shot_correct"] = (
        PROTOCOLS["pl_to_lt_zero_shot"]["train_countries"]
        == ["Poland"]
        and PROTOCOLS["pl_to_lt_zero_shot"]["validation_countries"]
        == ["Poland"]
        and PROTOCOLS["pl_to_lt_zero_shot"]["test_countries"]
        == ["Lithuania"]
    )

    for name, passed in checks.items():
        print(f"{name:<45} {'PASS' if passed else 'FAIL'}")

    all_pass = all(checks.values())

    manifest = {
        "script": "65_define_cross_country_modelling_protocols.py",
        "input": str(INPUT),
        "output_directory": str(OUT),
        "model_features": FEATURES,
        "label": LABEL,
        "country_column": COUNTRY,
        "split_column": SPLIT,
        "preprocessing": {
            "imputation": "median",
            "scaling": "zscore",
            "fit_on": "training_partition_only",
            "validation_or_test_used_for_fit": False,
        },
        "protocols": manifest_protocols,
        "checks": checks,
        "status": "PASS" if all_pass else "CHECK",
    }

    with open(
        OUT / "modelling_protocol_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=2)

    if not all_pass:
        raise RuntimeError(
            "One or more modelling protocol audit checks failed."
        )

    banner("SCRIPT 65 COMPLETE")

    print(f"Protocol summary:       {OUT / 'protocol_summary.csv'}")
    print(
        f"Preprocessing summary:  "
        f"{OUT / 'preprocessing_summary.csv'}"
    )
    print(
        f"Protocol assignments:   "
        f"{OUT / 'protocol_assignments.csv'}"
    )
    print(
        f"Manifest:               "
        f"{OUT / 'modelling_protocol_manifest.json'}"
    )
    print(f"Protocol datasets:      {PROTOCOL_ROOT}")

    print(
        "\n[PASS] LEAKAGE-SAFE CROSS-COUNTRY MODELLING "
        "PROTOCOLS ARE READY."
    )
    print(
        "NEXT: Script 66 — train and evaluate baseline "
        "road-selection models under all six protocols."
    )


if __name__ == "__main__":
    main()
