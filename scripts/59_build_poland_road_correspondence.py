from pathlib import Path
import json
import math
import warnings

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union

warnings.filterwarnings("ignore")


ROOT = Path(r"D:\Research\CartoGen")

INPUT_ROOT = (
    ROOT / "data" / "processed" / "poland" /
    "road_target_validation"
)

OUT_ROOT = (
    ROOT / "data" / "processed" / "poland" /
    "road_correspondence"
)

OUT_ROOT.mkdir(parents=True, exist_ok=True)


CITIES = {
    "warsaw": "Warsaw",
    "krakow": "Krakow",
    "gdansk": "Gdansk",
}


# Candidate search radius around BDOO target geometry.
SEARCH_DISTANCE_M = 12.0

# Source segment is considered genuinely represented when enough
# of its length lies near the target.
MIN_SOURCE_COVERAGE = 0.20

# Candidate target must also receive some geometric support.
MIN_TARGET_COVERAGE = 0.03

# Final acceptance threshold.
MIN_MATCH_SCORE = 0.18

# Strong source support threshold used for source-level selection.
SOURCE_SELECTED_COVERAGE = 0.35

# Geometry buffer used for length-overlap calculations.
COVERAGE_BUFFER_M = 5.0


def banner(text):
    print()
    print("=" * 120)
    print(text)
    print("=" * 120)


def section(text):
    print()
    print("-" * 120)
    print(text)
    print("-" * 120)


def safe_div(a, b):
    if b is None or b == 0:
        return 0.0
    return float(a) / float(b)


def safe_make_valid(gdf):

    out = gdf.copy()

    bad = ~out.geometry.is_valid

    if bad.any():
        try:
            out.loc[bad, "geometry"] = (
                out.loc[bad, "geometry"].make_valid()
            )
        except Exception:
            out.loc[bad, "geometry"] = (
                out.loc[bad, "geometry"].buffer(0)
            )

    out = out[
        out.geometry.notna()
        & ~out.geometry.is_empty
    ].copy()

    return out.reset_index(drop=True)


def endpoint_angle(geom):

    if geom is None or geom.is_empty:
        return np.nan

    try:
        if geom.geom_type == "MultiLineString":
            geom = max(
                geom.geoms,
                key=lambda x: x.length
            )

        coords = list(geom.coords)

        if len(coords) < 2:
            return np.nan

        x1, y1 = coords[0]
        x2, y2 = coords[-1]

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return np.nan

        angle = math.degrees(
            math.atan2(dy, dx)
        )

        return angle % 180.0

    except Exception:
        return np.nan


def angle_difference(a, b):

    if np.isnan(a) or np.isnan(b):
        return 90.0

    d = abs(a - b) % 180.0

    return min(d, 180.0 - d)


def buffered_length_coverage(line, reference, buffer_m):

    if (
        line is None
        or reference is None
        or line.is_empty
        or reference.is_empty
        or line.length <= 0
    ):
        return 0.0

    try:
        overlap = line.intersection(
            reference.buffer(buffer_m)
        )

        return min(
            1.0,
            max(
                0.0,
                overlap.length / line.length
            )
        )

    except Exception:
        return 0.0


def normalized_distance(source_geom, target_geom):

    try:
        d = source_geom.distance(target_geom)
    except Exception:
        return np.inf

    # 0 m -> 1
    # SEARCH_DISTANCE_M -> approximately 0
    return max(
        0.0,
        1.0 - d / SEARCH_DISTANCE_M
    )


def calculate_pair_metrics(source_geom, target_geom):

    source_len = float(source_geom.length)
    target_len = float(target_geom.length)

    distance = float(
        source_geom.distance(target_geom)
    )

    source_cov = buffered_length_coverage(
        source_geom,
        target_geom,
        COVERAGE_BUFFER_M
    )

    target_cov = buffered_length_coverage(
        target_geom,
        source_geom,
        COVERAGE_BUFFER_M
    )

    source_angle = endpoint_angle(source_geom)
    target_angle = endpoint_angle(target_geom)

    angle_diff = angle_difference(
        source_angle,
        target_angle
    )

    angle_score = max(
        0.0,
        1.0 - angle_diff / 90.0
    )

    distance_score = normalized_distance(
        source_geom,
        target_geom
    )

    # Individual source segments may represent only part of a
    # generalized target chain. Therefore source coverage receives
    # the largest weight; target coverage is informative but less
    # heavily weighted.
    match_score = (
        0.50 * source_cov
        + 0.20 * target_cov
        + 0.20 * distance_score
        + 0.10 * angle_score
    )

    return {
        "source_length_m": source_len,
        "target_length_m": target_len,
        "distance_m": distance,
        "source_coverage": source_cov,
        "target_coverage": target_cov,
        "source_angle_deg": source_angle,
        "target_angle_deg": target_angle,
        "angle_difference_deg": angle_diff,
        "distance_score": distance_score,
        "angle_score": angle_score,
        "match_score": match_score,
    }


def build_candidates(source, target):

    section("BUILDING SPATIAL CANDIDATES")

    print(
        f"Candidate search distance: "
        f"{SEARCH_DISTANCE_M:.1f} m"
    )

    print("Building source spatial index...")

    sindex = source.sindex

    rows = []

    print(
        f"Evaluating {len(target):,} "
        f"BDOO target road segments..."
    )

    for target_idx, target_row in target.iterrows():

        if (
            target_idx > 0
            and target_idx % 250 == 0
        ):
            print(
                f"  {target_idx:,}/{len(target):,}"
            )

        target_geom = target_row.geometry

        search_geom = target_geom.buffer(
            SEARCH_DISTANCE_M
        )

        candidate_positions = sindex.query(
            search_geom,
            predicate="intersects"
        )

        for source_idx in candidate_positions:

            source_idx = int(source_idx)

            source_geom = source.iloc[
                source_idx
            ].geometry

            metrics = calculate_pair_metrics(
                source_geom,
                target_geom
            )

            # Keep permissive candidates here. Final selection
            # occurs later.
            if (
                metrics["distance_m"]
                <= SEARCH_DISTANCE_M
            ):
                rows.append(
                    {
                        "source_idx": source_idx,
                        "target_idx": int(target_idx),
                        **metrics,
                    }
                )

    print(
        f"  {len(target):,}/{len(target):,}"
    )

    candidates = pd.DataFrame(rows)

    print()
    print(
        f"Candidate pairs: "
        f"{len(candidates):,}"
    )

    return candidates


def select_correspondences(candidates):

    section("SELECTING ROAD CORRESPONDENCES")

    if len(candidates) == 0:
        return candidates.copy()

    candidates = candidates.copy()

    candidates["accepted"] = (
        (
            candidates["source_coverage"]
            >= MIN_SOURCE_COVERAGE
        )
        &
        (
            candidates["target_coverage"]
            >= MIN_TARGET_COVERAGE
        )
        &
        (
            candidates["match_score"]
            >= MIN_MATCH_SCORE
        )
    )

    selected = candidates[
        candidates["accepted"]
    ].copy()

    # A detailed source segment should normally contribute to one
    # generalized target. If it passes for multiple targets, retain
    # its strongest correspondence.
    if len(selected):

        selected = selected.sort_values(
            [
                "source_idx",
                "match_score",
                "source_coverage",
                "target_coverage",
            ],
            ascending=[
                True,
                False,
                False,
                False,
            ]
        )

        selected = selected.drop_duplicates(
            subset=["source_idx"],
            keep="first"
        )

    selected = selected.reset_index(
        drop=True
    )

    print(
        f"Selected source-target pairs: "
        f"{len(selected):,}"
    )

    return selected


def target_group_statistics(
    source,
    target,
    selected
):

    section(
        "ANALYZING GENERALIZED TARGET CHAINS"
    )

    records = []

    grouped = {
        int(k): v
        for k, v in selected.groupby(
            "target_idx"
        )
    }

    print(
        f"Analyzing {len(target):,} "
        f"target road segments..."
    )

    for target_idx, target_row in target.iterrows():

        if (
            target_idx > 0
            and target_idx % 250 == 0
        ):
            print(
                f"  {target_idx:,}/{len(target):,}"
            )

        target_geom = target_row.geometry
        target_length = float(
            target_geom.length
        )

        group = grouped.get(
            int(target_idx)
        )

        if group is None or len(group) == 0:

            records.append(
                {
                    "target_idx":
                        int(target_idx),
                    "source_count": 0,
                    "source_indices": "",
                    "source_total_length_m":
                        0.0,
                    "target_length_m":
                        target_length,
                    "length_ratio":
                        0.0,
                    "target_coverage":
                        0.0,
                    "mean_source_coverage":
                        0.0,
                    "mean_distance_m":
                        np.nan,
                    "max_distance_m":
                        np.nan,
                    "mean_match_score":
                        0.0,
                    "correspondence_type":
                        "unmatched_target",
                }
            )

            continue

        source_ids = (
            group["source_idx"]
            .astype(int)
            .tolist()
        )

        source_geoms = (
            source.iloc[source_ids]
            .geometry
            .tolist()
        )

        source_union = unary_union(
            source_geoms
        )

        source_total_length = float(
            sum(
                geom.length
                for geom in source_geoms
            )
        )

        target_cov = (
            buffered_length_coverage(
                target_geom,
                source_union,
                COVERAGE_BUFFER_M
            )
        )

        source_count = len(source_ids)

        if source_count == 1:

            if target_cov >= 0.80:
                correspondence_type = (
                    "one_to_one"
                )
            else:
                correspondence_type = (
                    "one_source_partial"
                )

        else:

            if target_cov >= 0.80:
                correspondence_type = (
                    "many_to_one_chain"
                )
            else:
                correspondence_type = (
                    "many_to_one_partial"
                )

        records.append(
            {
                "target_idx":
                    int(target_idx),

                "source_count":
                    int(source_count),

                "source_indices":
                    ";".join(
                        str(x)
                        for x in source_ids
                    ),

                "source_total_length_m":
                    source_total_length,

                "target_length_m":
                    target_length,

                "length_ratio":
                    safe_div(
                        target_length,
                        source_total_length
                    ),

                "target_coverage":
                    target_cov,

                "mean_source_coverage":
                    float(
                        group[
                            "source_coverage"
                        ].mean()
                    ),

                "mean_distance_m":
                    float(
                        group[
                            "distance_m"
                        ].mean()
                    ),

                "max_distance_m":
                    float(
                        group[
                            "distance_m"
                        ].max()
                    ),

                "mean_match_score":
                    float(
                        group[
                            "match_score"
                        ].mean()
                    ),

                "correspondence_type":
                    correspondence_type,
            }
        )

    print(
        f"  {len(target):,}/{len(target):,}"
    )

    return pd.DataFrame(records)


def source_selection_table(
    source,
    selected
):

    selected_lookup = {}

    for _, row in selected.iterrows():

        selected_lookup[
            int(row["source_idx"])
        ] = row

    records = []

    for source_idx, source_row in source.iterrows():

        match = selected_lookup.get(
            int(source_idx)
        )

        if match is None:

            records.append(
                {
                    "source_idx":
                        int(source_idx),

                    "target_idx":
                        -1,

                    "selection_label":
                        "elimination",

                    "source_coverage":
                        0.0,

                    "target_coverage":
                        0.0,

                    "distance_m":
                        np.nan,

                    "match_score":
                        0.0,
                }
            )

        else:

            coverage = float(
                match["source_coverage"]
            )

            if (
                coverage
                >= SOURCE_SELECTED_COVERAGE
            ):
                label = "selected"
            else:
                label = "weak_selected"

            records.append(
                {
                    "source_idx":
                        int(source_idx),

                    "target_idx":
                        int(
                            match["target_idx"]
                        ),

                    "selection_label":
                        label,

                    "source_coverage":
                        coverage,

                    "target_coverage":
                        float(
                            match[
                                "target_coverage"
                            ]
                        ),

                    "distance_m":
                        float(
                            match[
                                "distance_m"
                            ]
                        ),

                    "match_score":
                        float(
                            match[
                                "match_score"
                            ]
                        ),
                }
            )

    return pd.DataFrame(records)


def save_diagnostic_gpkg(
    city_key,
    source,
    target,
    source_table,
    target_table
):

    gpkg = (
        OUT_ROOT /
        f"{city_key}_road_correspondence.gpkg"
    )

    if gpkg.exists():
        gpkg.unlink()

    source_out = source.copy()

    source_out["source_idx"] = np.arange(
        len(source_out)
    )

    source_out = source_out.merge(
        source_table,
        on="source_idx",
        how="left"
    )

    target_out = target.copy()

    target_out["target_idx"] = np.arange(
        len(target_out)
    )

    target_out = target_out.merge(
        target_table,
        on="target_idx",
        how="left"
    )

    source_out.to_file(
        gpkg,
        layer="source_correspondence",
        driver="GPKG"
    )

    target_out.to_file(
        gpkg,
        layer="target_correspondence",
        driver="GPKG"
    )

    return gpkg


def summarize_city(
    city_name,
    source,
    target,
    candidates,
    selected,
    source_table,
    target_table
):

    selected_source = source_table[
        source_table[
            "selection_label"
        ].isin(
            [
                "selected",
                "weak_selected",
            ]
        )
    ]

    strong_source = source_table[
        source_table[
            "selection_label"
        ] == "selected"
    ]

    eliminated = source_table[
        source_table[
            "selection_label"
        ] == "elimination"
    ]

    matched_targets = target_table[
        target_table[
            "source_count"
        ] > 0
    ]

    unmatched_targets = target_table[
        target_table[
            "source_count"
        ] == 0
    ]

    one_source = target_table[
        target_table[
            "source_count"
        ] == 1
    ]

    multi_source = target_table[
        target_table[
            "source_count"
        ] > 1
    ]

    summary = {
        "city":
            city_name,

        "source_count":
            int(len(source)),

        "target_count":
            int(len(target)),

        "candidate_pairs":
            int(len(candidates)),

        "selected_pairs":
            int(len(selected)),

        "matched_source":
            int(len(selected_source)),

        "strong_selected_source":
            int(len(strong_source)),

        "eliminated_source":
            int(len(eliminated)),

        "source_selection_pct":
            100.0 * safe_div(
                len(selected_source),
                len(source)
            ),

        "source_elimination_pct":
            100.0 * safe_div(
                len(eliminated),
                len(source)
            ),

        "matched_targets":
            int(len(matched_targets)),

        "unmatched_targets":
            int(len(unmatched_targets)),

        "target_match_pct":
            100.0 * safe_div(
                len(matched_targets),
                len(target)
            ),

        "one_source_targets":
            int(len(one_source)),

        "multi_source_targets":
            int(len(multi_source)),

        "multi_source_pct":
            100.0 * safe_div(
                len(multi_source),
                len(matched_targets)
            ),

        "mean_sources_per_target":
            float(
                matched_targets[
                    "source_count"
                ].mean()
            )
            if len(matched_targets)
            else 0.0,

        "median_sources_per_target":
            float(
                matched_targets[
                    "source_count"
                ].median()
            )
            if len(matched_targets)
            else 0.0,

        "max_sources_per_target":
            int(
                matched_targets[
                    "source_count"
                ].max()
            )
            if len(matched_targets)
            else 0,

        "mean_target_coverage":
            float(
                matched_targets[
                    "target_coverage"
                ].mean()
            )
            if len(matched_targets)
            else 0.0,

        "median_target_coverage":
            float(
                matched_targets[
                    "target_coverage"
                ].median()
            )
            if len(matched_targets)
            else 0.0,

        "mean_source_coverage":
            float(
                selected[
                    "source_coverage"
                ].mean()
            )
            if len(selected)
            else 0.0,

        "median_source_coverage":
            float(
                selected[
                    "source_coverage"
                ].median()
            )
            if len(selected)
            else 0.0,

        "mean_distance_m":
            float(
                selected[
                    "distance_m"
                ].mean()
            )
            if len(selected)
            else np.nan,

        "median_distance_m":
            float(
                selected[
                    "distance_m"
                ].median()
            )
            if len(selected)
            else np.nan,

        "mean_match_score":
            float(
                selected[
                    "match_score"
                ].mean()
            )
            if len(selected)
            else 0.0,
    }

    return summary


def main():

    banner(
        "CARTOGEN — BUILD POLISH BDOT10k → BDOO ROAD CORRESPONDENCE"
    )

    global_summaries = []

    for city_key, city_name in CITIES.items():

        banner(
            f"{city_key.upper()} ({city_name})"
        )

        input_gpkg = (
            INPUT_ROOT /
            f"{city_key}_validated_road_pair.gpkg"
        )

        if not input_gpkg.exists():
            raise FileNotFoundError(
                input_gpkg
            )

        section(
            "READING VALIDATED SOURCE/TARGET NETWORKS"
        )

        source = gpd.read_file(
            input_gpkg,
            layer="source_roads"
        )

        target = gpd.read_file(
            input_gpkg,
            layer="target_roads"
        )

        source = safe_make_valid(source)
        target = safe_make_valid(target)

        if source.crs != target.crs:
            target = target.to_crs(
                source.crs
            )

        print(
            f"BDOT10k source roads: "
            f"{len(source):,}"
        )

        print(
            f"BDOO target roads:    "
            f"{len(target):,}"
        )

        print(
            f"CRS:                  "
            f"{source.crs}"
        )

        print(
            f"Source length:         "
            f"{source.geometry.length.sum()/1000:.2f} km"
        )

        print(
            f"Target length:         "
            f"{target.geometry.length.sum()/1000:.2f} km"
        )

        # ------------------------------------------------------------------
        # CANDIDATES
        # ------------------------------------------------------------------

        candidates = build_candidates(
            source,
            target
        )

        # ------------------------------------------------------------------
        # SELECTION
        # ------------------------------------------------------------------

        selected = select_correspondences(
            candidates
        )

        # ------------------------------------------------------------------
        # TARGET GROUPS
        # ------------------------------------------------------------------

        target_table = (
            target_group_statistics(
                source,
                target,
                selected
            )
        )

        # ------------------------------------------------------------------
        # SOURCE SELECTION LABELS
        # ------------------------------------------------------------------

        section(
            "DERIVING SOURCE ROAD SELECTION / ELIMINATION LABELS"
        )

        source_table = (
            source_selection_table(
                source,
                selected
            )
        )

        print()
        print(
            source_table[
                "selection_label"
            ]
            .value_counts()
            .to_string()
        )

        # ------------------------------------------------------------------
        # OPERATION TYPES
        # ------------------------------------------------------------------

        section(
            "TARGET CORRESPONDENCE TYPES"
        )

        operation_counts = (
            target_table[
                "correspondence_type"
            ]
            .value_counts()
        )

        for operation, count in (
            operation_counts.items()
        ):

            pct = (
                100.0
                * count
                / len(target_table)
            )

            print(
                f"{operation:<30} "
                f"{count:>7,} "
                f"({pct:6.2f}%)"
            )

        # ------------------------------------------------------------------
        # SUMMARY
        # ------------------------------------------------------------------

        summary = summarize_city(
            city_name,
            source,
            target,
            candidates,
            selected,
            source_table,
            target_table
        )

        section(
            "ROAD CORRESPONDENCE SUMMARY"
        )

        for key, value in summary.items():

            if key == "city":
                continue

            if isinstance(
                value,
                (float, np.floating)
            ):
                print(
                    f"{key:<40} "
                    f"{value:.3f}"
                )
            else:
                print(
                    f"{key:<40} "
                    f"{value:,}"
                )

        # ------------------------------------------------------------------
        # SAVE
        # ------------------------------------------------------------------

        section("SAVING OUTPUTS")

        candidate_csv = (
            OUT_ROOT /
            f"{city_key}_road_candidate_pairs.csv"
        )

        selected_csv = (
            OUT_ROOT /
            f"{city_key}_road_correspondence_pairs.csv"
        )

        source_csv = (
            OUT_ROOT /
            f"{city_key}_source_road_labels.csv"
        )

        target_csv = (
            OUT_ROOT /
            f"{city_key}_target_road_correspondence.csv"
        )

        candidates.to_csv(
            candidate_csv,
            index=False
        )

        selected.to_csv(
            selected_csv,
            index=False
        )

        source_table.to_csv(
            source_csv,
            index=False
        )

        target_table.to_csv(
            target_csv,
            index=False
        )

        gpkg = save_diagnostic_gpkg(
            city_key,
            source,
            target,
            source_table,
            target_table
        )

        print(
            f"Candidates:\n{candidate_csv}"
        )

        print(
            f"\nSelected pairs:\n"
            f"{selected_csv}"
        )

        print(
            f"\nSource labels:\n"
            f"{source_csv}"
        )

        print(
            f"\nTarget correspondence:\n"
            f"{target_csv}"
        )

        print(
            f"\nDiagnostic GPKG:\n"
            f"{gpkg}"
        )

        global_summaries.append(
            summary
        )

    # ======================================================================
    # GLOBAL SUMMARY
    # ======================================================================

    banner(
        "GLOBAL POLISH ROAD CORRESPONDENCE SUMMARY"
    )

    summary_df = pd.DataFrame(
        global_summaries
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    summary_csv = (
        OUT_ROOT /
        "poland_road_correspondence_summary.csv"
    )

    summary_json = (
        OUT_ROOT /
        "poland_road_correspondence_summary.json"
    )

    summary_df.to_csv(
        summary_csv,
        index=False
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            global_summaries,
            f,
            indent=2,
            ensure_ascii=False,
            default=lambda x: (
                x.item()
                if isinstance(
                    x,
                    np.generic
                )
                else str(x)
            )
        )

    banner(
        "SCRIPT 59 COMPLETE"
    )

    print()
    print("Summary CSV:")
    print(summary_csv)

    print()
    print("Summary JSON:")
    print(summary_json)

    print()
    print(
        "NEXT: Script 60 — analyze Polish road "
        "generalization operations and topology."
    )


if __name__ == "__main__":
    main()