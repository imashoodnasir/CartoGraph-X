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

OUTPUT = (
    INTERIM_OS /
    "os_geopackage_layers.json"
)


GEOPACKAGES = {
    "OpenMapLocal":
        INTERIM_OS /
        "OpenMapLocal" /
        "opmplc_gb.gpkg",

    "OpenZoomstack":
        INTERIM_OS /
        "OpenZoomstack" /
        "OS_Open_Zoomstack.gpkg",
}


def human_size(num_bytes):

    gb = num_bytes / (1024 ** 3)

    if gb >= 1:
        return f"{gb:.2f} GB"

    return (
        f"{num_bytes / (1024 ** 2):.2f} MB"
    )


def quote_identifier(name):

    return '"' + name.replace(
        '"',
        '""'
    ) + '"'


def get_feature_tables(connection):

    query = """
        SELECT
            c.table_name,
            c.identifier,
            c.description,
            g.column_name,
            g.geometry_type_name,
            g.srs_id
        FROM gpkg_contents AS c
        LEFT JOIN gpkg_geometry_columns AS g
            ON c.table_name = g.table_name
        WHERE c.data_type = 'features'
        ORDER BY c.table_name
    """

    cursor = connection.execute(query)

    return cursor.fetchall()


def get_columns(
    connection,
    table_name
):

    quoted = quote_identifier(
        table_name
    )

    cursor = connection.execute(
        f"PRAGMA table_info({quoted})"
    )

    columns = []

    for row in cursor.fetchall():

        columns.append({
            "cid": row[0],
            "name": row[1],
            "type": row[2],
            "notnull": row[3],
            "default": row[4],
            "primary_key": row[5],
        })

    return columns


def get_row_count(
    connection,
    table_name
):

    quoted = quote_identifier(
        table_name
    )

    cursor = connection.execute(
        f"SELECT COUNT(*) FROM {quoted}"
    )

    return cursor.fetchone()[0]


def inspect_geopackage(
    name,
    path
):

    if not path.exists():

        raise FileNotFoundError(
            f"GeoPackage missing:\n{path}\n"
            "Run script 10 first."
        )

    print()
    print("=" * 120)
    print(name)
    print("=" * 120)

    print(
        f"File: {path}"
    )

    print(
        f"Size: "
        f"{human_size(path.stat().st_size)}"
    )

    connection = sqlite3.connect(
        f"file:{path}?mode=ro",
        uri=True
    )

    try:

        feature_tables = (
            get_feature_tables(
                connection
            )
        )

        print(
            f"Feature layers: "
            f"{len(feature_tables)}"
        )

        records = []

        for index, row in enumerate(
            feature_tables,
            start=1
        ):

            (
                table_name,
                identifier,
                description,
                geometry_column,
                geometry_type,
                srs_id,
            ) = row

            print()
            print("-" * 120)

            print(
                f"[{index:02d}] "
                f"{table_name}"
            )

            print(
                f"     Identifier: "
                f"{identifier}"
            )

            print(
                f"     Geometry:   "
                f"{geometry_type}"
            )

            print(
                f"     SRS ID:     "
                f"{srs_id}"
            )

            # COUNT(*) can take some time
            # on very large tables but does
            # not load geometries into Python.
            try:

                count = get_row_count(
                    connection,
                    table_name
                )

            except Exception as exc:

                count = None

                print(
                    f"     Count:      "
                    f"ERROR ({exc})"
                )

            else:

                print(
                    f"     Features:   "
                    f"{count:,}"
                )

            columns = get_columns(
                connection,
                table_name
            )

            column_names = [
                column["name"]
                for column in columns
            ]

            print(
                "     Columns:    "
            )

            print(
                "       " +
                ", ".join(
                    column_names
                )
            )

            records.append({
                "table_name":
                    table_name,
                "identifier":
                    identifier,
                "description":
                    description,
                "geometry_column":
                    geometry_column,
                "geometry_type":
                    geometry_type,
                "srs_id":
                    srs_id,
                "feature_count":
                    count,
                "columns":
                    columns,
            })

        return {
            "name": name,
            "file": str(
                path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "size_bytes":
                path.stat().st_size,
            "size":
                human_size(
                    path.stat().st_size
                ),
            "layers":
                records,
        }

    finally:

        connection.close()


def main():

    print("=" * 120)
    print(
        "CARTOGEN — INSPECT OS "
        "GEOPACKAGE LAYERS"
    )
    print("=" * 120)

    inventory = {}

    for name, path in (
        GEOPACKAGES.items()
    ):

        inventory[name] = (
            inspect_geopackage(
                name,
                path
            )
        )

    with open(
        OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            inventory,
            file,
            indent=2
        )

    print()
    print("=" * 120)
    print("GEOPACKAGE INSPECTION COMPLETE")
    print("=" * 120)

    print(
        f"Inventory saved:\n{OUTPUT}"
    )


if __name__ == "__main__":
    main()