from pathlib import Path
import sqlite3
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERIM_OS = (
    PROJECT_ROOT /
    "data" /
    "interim" /
    "os"
)

OUTPUT_FILE = (
    INTERIM_OS /
    "cartogen_layer_profiles.json"
)


DATASETS = {

    "OpenMapLocal": {

        "path":
            INTERIM_OS /
            "OpenMapLocal" /
            "opmplc_gb.gpkg",

        "layers": [
            "building",
            "important_building",
            "road",
            "road_tunnel",
            "motorway_junction",
        ],
    },

    "OpenZoomstack": {

        "path":
            INTERIM_OS /
            "OpenZoomstack" /
            "OS_Open_Zoomstack.gpkg",

        "layers": [
            "local_buildings",
            "district_buildings",
            "roads_local",
            "roads_regional",
            "roads_national",
        ],
    },
}


def quote_identifier(name):

    return (
        '"'
        + name.replace('"', '""')
        + '"'
    )


def serialize_value(value):

    if isinstance(
        value,
        (bytes, bytearray)
    ):

        return (
            f"<BINARY "
            f"{len(value)} bytes>"
        )

    return value


def get_schema(
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
            "cid": row[0],
            "name": row[1],
            "type": row[2],
            "notnull": row[3],
            "default": row[4],
            "primary_key": row[5],
        }
        for row in rows
    ]


def get_geometry_info(
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
        "geometry_column": row[0],
        "geometry_type": row[1],
        "srs_id": row[2],
    }


def get_sample_rows(
    connection,
    table,
    geometry_column,
    limit=5
):

    schema = get_schema(
        connection,
        table
    )

    columns = [
        item["name"]
        for item in schema
        if item["name"]
        != geometry_column
    ]

    quoted_table = (
        quote_identifier(table)
    )

    quoted_columns = ", ".join(
        quote_identifier(column)
        for column in columns
    )

    query = (
        f"SELECT {quoted_columns} "
        f"FROM {quoted_table} "
        f"LIMIT {limit}"
    )

    rows = connection.execute(
        query
    ).fetchall()

    records = []

    for row in rows:

        record = {}

        for column, value in zip(
            columns,
            row
        ):

            record[column] = (
                serialize_value(value)
            )

        records.append(record)

    return records


def get_distinct_values(
    connection,
    table,
    column,
    limit=20
):

    quoted_table = (
        quote_identifier(table)
    )

    quoted_column = (
        quote_identifier(column)
    )

    query = f"""
        SELECT
            {quoted_column},
            COUNT(*) AS n
        FROM {quoted_table}
        WHERE {quoted_column}
              IS NOT NULL
        GROUP BY {quoted_column}
        ORDER BY n DESC
        LIMIT {limit}
    """

    try:

        rows = connection.execute(
            query
        ).fetchall()

    except sqlite3.Error:

        return []

    return [
        {
            "value":
                serialize_value(row[0]),
            "count":
                row[1],
        }
        for row in rows
    ]


def main():

    print("=" * 120)
    print(
        "CARTOGEN — PROFILE "
        "OS CARTOGRAPHIC LAYERS"
    )
    print("=" * 120)

    output = {}

    for dataset, config in (
        DATASETS.items()
    ):

        path = config["path"]

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        print()
        print("=" * 120)
        print(dataset)
        print("=" * 120)

        connection = sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True
        )

        output[dataset] = {}

        try:

            for table in config["layers"]:

                print()
                print("-" * 120)
                print(f"LAYER: {table}")
                print("-" * 120)

                geometry = (
                    get_geometry_info(
                        connection,
                        table
                    )
                )

                schema = get_schema(
                    connection,
                    table
                )

                print(
                    "Geometry:",
                    geometry
                )

                print()
                print("Columns:")

                for column in schema:

                    print(
                        f"  "
                        f"{column['name']:<35}"
                        f"{column['type']}"
                    )

                geometry_column = (
                    geometry[
                        "geometry_column"
                    ]
                )

                samples = (
                    get_sample_rows(
                        connection,
                        table,
                        geometry_column,
                        limit=5
                    )
                )

                print()
                print("Sample records:")

                for i, record in enumerate(
                    samples,
                    start=1
                ):

                    print(
                        f"\n  SAMPLE {i}"
                    )

                    for key, value in (
                        record.items()
                    ):

                        print(
                            f"    "
                            f"{key:<30}: "
                            f"{value}"
                        )

                categorical = {}

                # Look for columns likely
                # to contain useful semantic
                # classifications.
                for column in schema:

                    name = (
                        column["name"]
                        .lower()
                    )

                    if any(
                        token in name
                        for token in [
                            "class",
                            "type",
                            "function",
                            "name",
                            "category",
                            "classification",
                        ]
                    ):

                        values = (
                            get_distinct_values(
                                connection,
                                table,
                                column["name"],
                                limit=20
                            )
                        )

                        if values:

                            categorical[
                                column["name"]
                            ] = values

                            print()
                            print(
                                "Top values for "
                                f"{column['name']}:"
                            )

                            for item in values:

                                print(
                                    f"    "
                                    f"{item['value']} "
                                    f"({item['count']:,})"
                                )

                output[dataset][table] = {
                    "geometry":
                        geometry,
                    "schema":
                        schema,
                    "samples":
                        samples,
                    "categorical_values":
                        categorical,
                }

        finally:

            connection.close()

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            default=str
        )

    print()
    print("=" * 120)
    print("PROFILE COMPLETE")
    print("=" * 120)

    print(
        f"Saved:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()