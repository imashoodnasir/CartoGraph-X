from pathlib import Path
import sqlite3
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OPENMAP = (
    PROJECT_ROOT /
    "data" /
    "interim" /
    "os" /
    "OpenMapLocal" /
    "opmplc_gb.gpkg"
)

ZOOMSTACK = (
    PROJECT_ROOT /
    "data" /
    "interim" /
    "os" /
    "OpenZoomstack" /
    "OS_Open_Zoomstack.gpkg"
)

OUTPUT = (
    PROJECT_ROOT /
    "data" /
    "interim" /
    "os" /
    "source_layer_comparison.json"
)


COMPARISONS = [
    {
        "name":
            "OpenMap buildings vs "
            "Zoomstack local buildings",

        "left_table":
            "building",

        "right_table":
            "local_buildings",
    },

    {
        "name":
            "OpenMap roads vs "
            "Zoomstack local roads",

        "left_table":
            "road",

        "right_table":
            "roads_local",
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


def get_geometry_metadata(
    connection,
    table
):

    row = connection.execute(
        """
        SELECT
            column_name,
            geometry_type_name,
            srs_id
        FROM gpkg_geometry_columns
        WHERE table_name = ?
        """,
        (table,)
    ).fetchone()

    if row is None:
        return None

    return {
        "column": row[0],
        "geometry_type": row[1],
        "srs_id": row[2],
    }


def get_columns(
    connection,
    table
):

    quoted = quote_identifier(
        table
    )

    rows = connection.execute(
        f"PRAGMA table_info({quoted})"
    ).fetchall()

    return [
        {
            "name": row[1],
            "type": row[2],
        }
        for row in rows
    ]


def main():

    print("=" * 110)
    print(
        "CARTOGEN — COMPARE "
        "OPENMAP AND ZOOMSTACK "
        "SOURCE LAYERS"
    )
    print("=" * 110)

    if not OPENMAP.exists():
        raise FileNotFoundError(
            OPENMAP
        )

    if not ZOOMSTACK.exists():
        raise FileNotFoundError(
            ZOOMSTACK
        )

    left = sqlite3.connect(
        f"file:{OPENMAP}?mode=ro",
        uri=True
    )

    right = sqlite3.connect(
        f"file:{ZOOMSTACK}?mode=ro",
        uri=True
    )

    results = []

    try:

        for comparison in (
            COMPARISONS
        ):

            left_table = (
                comparison[
                    "left_table"
                ]
            )

            right_table = (
                comparison[
                    "right_table"
                ]
            )

            left_count = count_rows(
                left,
                left_table
            )

            right_count = count_rows(
                right,
                right_table
            )

            left_geometry = (
                get_geometry_metadata(
                    left,
                    left_table
                )
            )

            right_geometry = (
                get_geometry_metadata(
                    right,
                    right_table
                )
            )

            left_columns = (
                get_columns(
                    left,
                    left_table
                )
            )

            right_columns = (
                get_columns(
                    right,
                    right_table
                )
            )

            left_names = {
                item["name"]
                for item
                in left_columns
            }

            right_names = {
                item["name"]
                for item
                in right_columns
            }

            common_columns = sorted(
                left_names
                & right_names
            )

            record = {

                "comparison":
                    comparison["name"],

                "left_table":
                    left_table,

                "right_table":
                    right_table,

                "left_count":
                    left_count,

                "right_count":
                    right_count,

                "count_difference":
                    right_count
                    - left_count,

                "count_ratio":
                    (
                        right_count
                        / left_count
                        if left_count
                        else None
                    ),

                "left_geometry":
                    left_geometry,

                "right_geometry":
                    right_geometry,

                "common_columns":
                    common_columns,
            }

            results.append(
                record
            )

            print()
            print("-" * 110)
            print(
                comparison["name"]
            )
            print("-" * 110)

            print(
                f"OpenMap count:   "
                f"{left_count:,}"
            )

            print(
                f"Zoomstack count: "
                f"{right_count:,}"
            )

            print(
                f"Difference:      "
                f"{right_count-left_count:,}"
            )

            print(
                f"Count ratio:     "
                f"{right_count/left_count:.6f}"
            )

            print(
                f"OpenMap geom:    "
                f"{left_geometry}"
            )

            print(
                f"Zoomstack geom:  "
                f"{right_geometry}"
            )

            print(
                "Common columns:"
            )

            for column in (
                common_columns
            ):

                print(
                    f"  {column}"
                )

    finally:

        left.close()
        right.close()

    with open(
        OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2
        )

    print()
    print("=" * 110)
    print("COMPARISON COMPLETE")
    print("=" * 110)

    print(
        f"Saved:\n{OUTPUT}"
    )


if __name__ == "__main__":
    main()