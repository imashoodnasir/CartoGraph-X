from pathlib import Path
import json
import re

import geopandas as gpd
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRPK_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "grpk_windows"
)

GDR_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "gdr50lt_windows"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "road_attribute_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


WINDOWS = {
    "vilnius_central": "Vilnius",
    "kaunas_central": "Kaunas",
    "klaipeda_central": "Klaipeda",
}


# Attributes that are particularly interesting for cross-scale matching.
#
# We inspect all common attributes, but these receive special reporting.

IMPORTANT_FIELDS = [
    "TOP_ID",
    "GKODAS",
    "FKODAS",
    "VARDAS",
    "NUMERIS",
    "ENUMERIS",
    "KATEGOR",
    "PASKIRTIS",
    "DANGA",
    "LYGMUO",
    "JPLOTIS",
    "PLOTIS",
    "KIEKIS",
]


# =============================================================================
# HELPERS
# =============================================================================

def normalize_column_name(name):
    return str(name).strip().upper()


def normalize_value(value):
    """
    Normalize values only for comparison.

    Original values are never modified in the source datasets.
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.upper()


def find_column(gdf, requested):
    """
    Case-insensitive column lookup.
    """

    requested = normalize_column_name(
        requested
    )

    for column in gdf.columns:

        if normalize_column_name(column) == requested:
            return column

    return None


def describe_field(gdf, column):
    """
    Basic descriptive statistics for one attribute.
    """

    if column is None:
        return {
            "exists": False,
            "non_null": 0,
            "unique": 0,
            "sample_values": [],
        }

    values = (
        gdf[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = values[
        values != ""
    ]

    samples = (
        values
        .drop_duplicates()
        .head(10)
        .tolist()
    )

    return {
        "exists": True,
        "non_null": int(len(values)),
        "unique": int(values.nunique()),
        "sample_values": samples,
    }


def normalized_value_set(series):

    values = set()

    for value in series:

        normalized = normalize_value(
            value
        )

        if normalized is not None:
            values.add(
                normalized
            )

    return values


def compare_attribute(
    source,
    target,
    source_column,
    target_column
):
    """
    Compare unique normalized attribute values between datasets.
    """

    source_values = normalized_value_set(
        source[source_column]
    )

    target_values = normalized_value_set(
        target[target_column]
    )

    shared = (
        source_values
        & target_values
    )

    source_overlap_pct = (
        100.0
        * len(shared)
        / len(source_values)
        if source_values
        else 0.0
    )

    target_overlap_pct = (
        100.0
        * len(shared)
        / len(target_values)
        if target_values
        else 0.0
    )

    return {
        "source_unique": len(source_values),
        "target_unique": len(target_values),
        "shared_unique": len(shared),
        "source_value_overlap_pct":
            source_overlap_pct,
        "target_value_overlap_pct":
            target_overlap_pct,
        "shared_examples":
            sorted(shared)[:20],
    }


def row_level_match_rate(
    source,
    target,
    source_column,
    target_column
):
    """
    Percentage of source and target features whose normalized value
    occurs somewhere in the opposite dataset.
    """

    source_target_values = normalized_value_set(
        target[target_column]
    )

    target_source_values = normalized_value_set(
        source[source_column]
    )

    source_norm = (
        source[source_column]
        .apply(normalize_value)
    )

    target_norm = (
        target[target_column]
        .apply(normalize_value)
    )

    source_non_null = (
        source_norm.notna()
    )

    target_non_null = (
        target_norm.notna()
    )

    source_matches = (
        source_norm[
            source_non_null
        ]
        .isin(source_target_values)
    )

    target_matches = (
        target_norm[
            target_non_null
        ]
        .isin(target_source_values)
    )

    source_rate = (
        100.0
        * source_matches.mean()
        if len(source_matches)
        else 0.0
    )

    target_rate = (
        100.0
        * target_matches.mean()
        if len(target_matches)
        else 0.0
    )

    return {
        "source_features_with_value":
            int(source_non_null.sum()),

        "target_features_with_value":
            int(target_non_null.sum()),

        "source_feature_match_pct":
            float(source_rate),

        "target_feature_match_pct":
            float(target_rate),
    }


def inspect_shared_value_multiplicity(
    source,
    target,
    source_column,
    target_column
):
    """
    Determine whether a shared identifier tends to be unique or repeated.

    This is important because TOP_ID might identify objects, groups,
    chains, or something else.
    """

    source_norm = (
        source[source_column]
        .apply(normalize_value)
        .dropna()
    )

    target_norm = (
        target[target_column]
        .apply(normalize_value)
        .dropna()
    )

    source_counts = (
        source_norm
        .value_counts()
    )

    target_counts = (
        target_norm
        .value_counts()
    )

    shared = (
        set(source_counts.index)
        & set(target_counts.index)
    )

    if not shared:

        return {
            "shared_values": 0,
            "one_to_one": 0,
            "one_to_many": 0,
            "many_to_one": 0,
            "many_to_many": 0,
        }

    one_to_one = 0
    one_to_many = 0
    many_to_one = 0
    many_to_many = 0

    for value in shared:

        source_n = int(
            source_counts[value]
        )

        target_n = int(
            target_counts[value]
        )

        if source_n == 1 and target_n == 1:
            one_to_one += 1

        elif source_n == 1 and target_n > 1:
            one_to_many += 1

        elif source_n > 1 and target_n == 1:
            many_to_one += 1

        else:
            many_to_many += 1

    return {
        "shared_values":
            len(shared),

        "one_to_one":
            one_to_one,

        "one_to_many":
            one_to_many,

        "many_to_one":
            many_to_one,

        "many_to_many":
            many_to_many,
    }


def find_common_columns(
    source,
    target
):

    source_lookup = {
        normalize_column_name(column):
            column
        for column in source.columns
        if column != source.geometry.name
    }

    target_lookup = {
        normalize_column_name(column):
            column
        for column in target.columns
        if column != target.geometry.name
    }

    common = sorted(
        set(source_lookup)
        & set(target_lookup)
    )

    return [
        (
            name,
            source_lookup[name],
            target_lookup[name]
        )
        for name in common
    ]


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 120)
    print(
        "CARTOGEN — INSPECT LITHUANIAN ROAD "
        "CROSS-SCALE ATTRIBUTES"
    )
    print("=" * 120)

    all_comparisons = []
    detailed_results = {}

    for window, city in WINDOWS.items():

        print()
        print("=" * 120)
        print(
            f"{window} ({city})"
        )
        print("=" * 120)

        source_path = (
            GRPK_DIR
            / f"{window}_grpk.gpkg"
        )

        target_path = (
            GDR_DIR
            / f"{window}_gdr50lt.gpkg"
        )

        if not source_path.exists():
            raise FileNotFoundError(
                source_path
            )

        if not target_path.exists():
            raise FileNotFoundError(
                target_path
            )

        print()
        print("Reading GRPK source roads...")

        source = gpd.read_file(
            source_path,
            layer="source_roads"
        )

        print(
            f"Source features: "
            f"{len(source):,}"
        )

        print("Reading GDR50LT target roads...")

        target = gpd.read_file(
            target_path,
            layer="target_roads"
        )

        print(
            f"Target features: "
            f"{len(target):,}"
        )

        print()
        print("-" * 120)
        print("SOURCE COLUMNS")
        print("-" * 120)

        for column in source.columns:

            if column == source.geometry.name:
                continue

            print(
                f"{column:<30} "
                f"{str(source[column].dtype):<20}"
            )

        print()
        print("-" * 120)
        print("TARGET COLUMNS")
        print("-" * 120)

        for column in target.columns:

            if column == target.geometry.name:
                continue

            print(
                f"{column:<30} "
                f"{str(target[column].dtype):<20}"
            )

        common_columns = find_common_columns(
            source,
            target
        )

        print()
        print("-" * 120)
        print("COMMON ATTRIBUTE COLUMNS")
        print("-" * 120)

        if not common_columns:

            print(
                "No common non-geometry "
                "columns found."
            )

        else:

            for normalized_name, _, _ in common_columns:

                print(
                    f"  {normalized_name}"
                )

        city_results = {
            "city": city,
            "window": window,
            "source_count": len(source),
            "target_count": len(target),
            "attributes": {},
        }

        # -------------------------------------------------------------
        # Important-field inspection
        # -------------------------------------------------------------

        print()
        print("=" * 120)
        print("IMPORTANT ATTRIBUTE INSPECTION")
        print("=" * 120)

        for requested in IMPORTANT_FIELDS:

            source_column = find_column(
                source,
                requested
            )

            target_column = find_column(
                target,
                requested
            )

            print()
            print("-" * 120)
            print(
                f"ATTRIBUTE: {requested}"
            )
            print("-" * 120)

            source_description = describe_field(
                source,
                source_column
            )

            target_description = describe_field(
                target,
                target_column
            )

            print(
                f"Source column: "
                f"{source_column}"
            )

            print(
                f"Target column: "
                f"{target_column}"
            )

            print(
                f"Source non-null: "
                f"{source_description['non_null']:,}"
            )

            print(
                f"Target non-null: "
                f"{target_description['non_null']:,}"
            )

            print(
                f"Source unique: "
                f"{source_description['unique']:,}"
            )

            print(
                f"Target unique: "
                f"{target_description['unique']:,}"
            )

            attribute_result = {
                "source":
                    source_description,
                "target":
                    target_description,
            }

            if (
                source_column is not None
                and target_column is not None
            ):

                comparison = compare_attribute(
                    source,
                    target,
                    source_column,
                    target_column
                )

                feature_match = row_level_match_rate(
                    source,
                    target,
                    source_column,
                    target_column
                )

                multiplicity = (
                    inspect_shared_value_multiplicity(
                        source,
                        target,
                        source_column,
                        target_column
                    )
                )

                print()
                print(
                    f"Shared unique values: "
                    f"{comparison['shared_unique']:,}"
                )

                print(
                    "Source unique-value overlap: "
                    f"{comparison['source_value_overlap_pct']:.2f}%"
                )

                print(
                    "Target unique-value overlap: "
                    f"{comparison['target_value_overlap_pct']:.2f}%"
                )

                print(
                    "Source feature match rate: "
                    f"{feature_match['source_feature_match_pct']:.2f}%"
                )

                print(
                    "Target feature match rate: "
                    f"{feature_match['target_feature_match_pct']:.2f}%"
                )

                print()
                print(
                    "Shared-value multiplicity:"
                )

                print(
                    f"  one-to-one:   "
                    f"{multiplicity['one_to_one']:,}"
                )

                print(
                    f"  one-to-many:  "
                    f"{multiplicity['one_to_many']:,}"
                )

                print(
                    f"  many-to-one:  "
                    f"{multiplicity['many_to_one']:,}"
                )

                print(
                    f"  many-to-many: "
                    f"{multiplicity['many_to_many']:,}"
                )

                if comparison[
                    "shared_examples"
                ]:

                    print(
                        "Shared examples:"
                    )

                    for value in comparison[
                        "shared_examples"
                    ][:10]:

                        print(
                            f"  {value}"
                        )

                attribute_result[
                    "comparison"
                ] = comparison

                attribute_result[
                    "feature_match"
                ] = feature_match

                attribute_result[
                    "multiplicity"
                ] = multiplicity

                all_comparisons.append({
                    "window":
                        window,

                    "city":
                        city,

                    "attribute":
                        requested,

                    "source_non_null":
                        source_description[
                            "non_null"
                        ],

                    "target_non_null":
                        target_description[
                            "non_null"
                        ],

                    "source_unique":
                        comparison[
                            "source_unique"
                        ],

                    "target_unique":
                        comparison[
                            "target_unique"
                        ],

                    "shared_unique":
                        comparison[
                            "shared_unique"
                        ],

                    "source_value_overlap_pct":
                        comparison[
                            "source_value_overlap_pct"
                        ],

                    "target_value_overlap_pct":
                        comparison[
                            "target_value_overlap_pct"
                        ],

                    "source_feature_match_pct":
                        feature_match[
                            "source_feature_match_pct"
                        ],

                    "target_feature_match_pct":
                        feature_match[
                            "target_feature_match_pct"
                        ],

                    "one_to_one":
                        multiplicity[
                            "one_to_one"
                        ],

                    "one_to_many":
                        multiplicity[
                            "one_to_many"
                        ],

                    "many_to_one":
                        multiplicity[
                            "many_to_one"
                        ],

                    "many_to_many":
                        multiplicity[
                            "many_to_many"
                        ],
                })

            city_results[
                "attributes"
            ][requested] = attribute_result

        detailed_results[
            window
        ] = city_results

    # =================================================================
    # GLOBAL SUMMARY
    # =================================================================

    comparison_df = pd.DataFrame(
        all_comparisons
    )

    print()
    print("=" * 120)
    print("GLOBAL ATTRIBUTE COMPARISON")
    print("=" * 120)

    if len(comparison_df):

        display_columns = [
            "city",
            "attribute",
            "source_unique",
            "target_unique",
            "shared_unique",
            "source_value_overlap_pct",
            "target_value_overlap_pct",
            "source_feature_match_pct",
            "target_feature_match_pct",
        ]

        print(
            comparison_df[
                display_columns
            ].to_string(
                index=False,
                formatters={
                    "source_value_overlap_pct":
                        "{:.2f}".format,

                    "target_value_overlap_pct":
                        "{:.2f}".format,

                    "source_feature_match_pct":
                        "{:.2f}".format,

                    "target_feature_match_pct":
                        "{:.2f}".format,
                }
            )
        )

    # =================================================================
    # DECISION SUPPORT
    # =================================================================

    print()
    print("=" * 120)
    print("IDENTIFIER DECISION SUPPORT")
    print("=" * 120)

    candidate_fields = []

    if len(comparison_df):

        grouped = (
            comparison_df
            .groupby("attribute")
            .agg(
                cities=("city", "nunique"),

                mean_source_feature_match_pct=(
                    "source_feature_match_pct",
                    "mean"
                ),

                mean_target_feature_match_pct=(
                    "target_feature_match_pct",
                    "mean"
                ),

                total_shared_unique=(
                    "shared_unique",
                    "sum"
                ),

                total_one_to_one=(
                    "one_to_one",
                    "sum"
                ),

                total_many_to_one=(
                    "many_to_one",
                    "sum"
                ),

                total_many_to_many=(
                    "many_to_many",
                    "sum"
                ),
            )
            .reset_index()
        )

        grouped[
            "identifier_score"
        ] = (
            0.45
            * grouped[
                "mean_target_feature_match_pct"
            ]
            +
            0.35
            * grouped[
                "mean_source_feature_match_pct"
            ]
            +
            0.20
            * (
                grouped[
                    "total_one_to_one"
                ]
                /
                grouped[
                    "total_shared_unique"
                ]
                .clip(lower=1)
                * 100.0
            )
        )

        grouped = grouped.sort_values(
            "identifier_score",
            ascending=False
        )

        print(
            grouped.to_string(
                index=False,
                formatters={
                    "mean_source_feature_match_pct":
                        "{:.2f}".format,

                    "mean_target_feature_match_pct":
                        "{:.2f}".format,

                    "identifier_score":
                        "{:.2f}".format,
                }
            )
        )

        candidate_fields = (
            grouped[
                grouped[
                    "total_shared_unique"
                ] > 0
            ][
                "attribute"
            ]
            .tolist()
        )

        grouped.to_csv(
            OUTPUT_DIR
            / "road_identifier_ranking.csv",
            index=False
        )

    print()
    print("-" * 120)
    print("INTERPRETATION")
    print("-" * 120)

    if candidate_fields:

        print(
            "Cross-scale shared attribute "
            "values were detected."
        )

        print()
        print(
            "Candidate attributes:"
        )

        for field in candidate_fields:
            print(
                f"  {field}"
            )

        print()
        print(
            "Do NOT yet assume these are "
            "true object identifiers."
        )

        print(
            "Script 40 should combine the "
            "strongest semantic attributes "
            "with spatial/geometric evidence."
        )

    else:

        print(
            "No useful shared road attribute "
            "values were detected."
        )

        print(
            "Script 40 should therefore use "
            "geometry/topology-based "
            "correspondence."
        )

    # =================================================================
    # SAVE
    # =================================================================

    csv_path = (
        OUTPUT_DIR
        / "road_attribute_comparison.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "road_attribute_analysis.json"
    )

    if len(comparison_df):

        comparison_df.to_csv(
            csv_path,
            index=False
        )

    else:

        pd.DataFrame().to_csv(
            csv_path,
            index=False
        )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            detailed_results,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 120)
    print("SCRIPT 39 COMPLETE")
    print("=" * 120)

    print()
    print(
        f"Comparison CSV:\n{csv_path}"
    )

    print()
    print(
        f"Detailed JSON:\n{json_path}"
    )

    ranking_path = (
        OUTPUT_DIR
        / "road_identifier_ranking.csv"
    )

    if ranking_path.exists():

        print()
        print(
            f"Identifier ranking:\n"
            f"{ranking_path}"
        )


if __name__ == "__main__":
    main()