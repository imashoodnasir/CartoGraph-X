from pathlib import Path
import sqlite3
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ZOOMSTACK = (
    PROJECT_ROOT /
    "data" /
    "interim" /
    "os" /
    "OpenZoomstack" /
    "OS_Open_Zoomstack.gpkg"
)

OUTPUT_JSON = (
    PROJECT_ROOT /
    "data" /
    "interim" /
    "os" /
    "zoomstack_scale_hierarchy.json"
)

OUTPUT_CSV = (
    PROJECT_ROOT /
    "data" /
    "interim" /
    "os" /
    "zoomstack_scale_hierarchy.csv"
)


LAYERS = [
    {
        "family": "building",
        "level": "local",
        "layer": "local_buildings",
    },
    {
        "family": "building",
        "level": "district",
        "layer": "district_buildings",
    },
    {
        "family": "road",
        "level": "local",
        "layer": "roads_local",
    },
    {
        "family": "road",
        "level": "regional",
        "layer": "roads_regional",
    },
    {
        "family": "road",
        "level": "national",
        "layer": "roads_national",
    },
]


def quote_identifier(name):

    return (
        '"'
        + name.replace('"', '""')
        + '"'
    )


def count_rows(
    connection,
    table
):

    quoted = quote_identifier(
        table
    )

    return connection.execute(
        f"SELECT COUNT(*) "
        f"FROM {quoted}"
    ).fetchone()[0]


def main():

    print("=" * 100)
    print(
        "CARTOGEN — ANALYZE "
        "ZOOMSTACK SCALE HIERARCHY"
    )
    print("=" * 100)

    if not ZOOMSTACK.exists():

        raise FileNotFoundError(
            ZOOMSTACK
        )

    connection = sqlite3.connect(
        f"file:{ZOOMSTACK}?mode=ro",
        uri=True
    )

    records = []

    try:

        for item in LAYERS:

            count = count_rows(
                connection,
                item["layer"]
            )

            record = {
                **item,
                "feature_count":
                    count,
            }

            records.append(
                record
            )

    finally:

        connection.close()

    df = pd.DataFrame(
        records
    )

    # Calculate retention relative
    # to the most detailed level
    # within each feature family.

    for family in (
        df["family"].unique()
    ):

        mask = (
            df["family"]
            == family
        )

        family_df = df[mask]

        local_rows = (
            family_df[
                family_df["level"]
                == "local"
            ]
        )

        if len(local_rows) == 0:
            continue

        local_count = int(
            local_rows.iloc[0][
                "feature_count"
            ]
        )

        df.loc[
            mask,
            "retention_vs_local_pct"
        ] = (
            df.loc[
                mask,
                "feature_count"
            ]
            / local_count
            * 100
        )

        df.loc[
            mask,
            "reduction_vs_local_pct"
        ] = (
            100
            - df.loc[
                mask,
                "retention_vs_local_pct"
            ]
        )

    print()
    print(
        df.to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    for _, row in df.iterrows():

        print(
            f"{row['family']:<10} "
            f"{row['level']:<10} "
            f"{row['feature_count']:>12,} "
            f"| retention="
            f"{row['retention_vs_local_pct']:.2f}% "
            f"| reduction="
            f"{row['reduction_vs_local_pct']:.2f}%"
        )

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            df.to_dict(
                orient="records"
            ),
            file,
            indent=2
        )

    print()
    print(
        f"CSV saved:\n{OUTPUT_CSV}"
    )

    print(
        f"JSON saved:\n{OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()