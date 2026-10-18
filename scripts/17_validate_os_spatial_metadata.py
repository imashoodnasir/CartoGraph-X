from pathlib import Path
import sqlite3
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GPKG = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "os"
    / "OpenZoomstack"
    / "OS_Open_Zoomstack.gpkg"
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "os"
    / "os_spatial_metadata.json"
)


LAYERS = [
    "local_buildings",
    "district_buildings",
    "roads_local",
    "roads_regional",
    "roads_national",
]


def main():

    print("=" * 110)
    print("CARTOGEN — VALIDATE OS SPATIAL METADATA")
    print("=" * 110)

    if not GPKG.exists():
        raise FileNotFoundError(
            f"GeoPackage not found:\n{GPKG}"
        )

    connection = sqlite3.connect(
        f"file:{GPKG}?mode=ro",
        uri=True
    )

    results = []

    try:

        for layer in LAYERS:

            row = connection.execute(
                """
                SELECT
                    c.table_name,
                    c.min_x,
                    c.min_y,
                    c.max_x,
                    c.max_y,
                    g.column_name,
                    g.geometry_type_name,
                    g.srs_id,
                    s.organization,
                    s.organization_coordsys_id
                FROM gpkg_contents AS c

                JOIN gpkg_geometry_columns AS g
                    ON c.table_name = g.table_name

                JOIN gpkg_spatial_ref_sys AS s
                    ON g.srs_id = s.srs_id

                WHERE c.table_name = ?
                """,
                (layer,)
            ).fetchone()

            if row is None:

                print(
                    f"[ERROR] Metadata not found: "
                    f"{layer}"
                )

                continue

            record = {
                "layer": row[0],
                "min_x": row[1],
                "min_y": row[2],
                "max_x": row[3],
                "max_y": row[4],
                "geometry_column": row[5],
                "geometry_type": row[6],
                "srs_id": row[7],
                "organization": row[8],
                "organization_crs": row[9],
            }

            results.append(record)

            width_km = (
                row[3] - row[1]
            ) / 1000.0

            height_km = (
                row[4] - row[2]
            ) / 1000.0

            print()
            print("-" * 110)
            print(f"LAYER: {layer}")
            print("-" * 110)

            print(
                f"Geometry column: "
                f"{row[5]}"
            )

            print(
                f"Geometry type:   "
                f"{row[6]}"
            )

            print(
                f"SRS ID:          "
                f"{row[7]}"
            )

            print(
                f"CRS:             "
                f"{row[8]}:{row[9]}"
            )

            print(
                f"Min X:           "
                f"{row[1]:,.2f}"
            )

            print(
                f"Min Y:           "
                f"{row[2]:,.2f}"
            )

            print(
                f"Max X:           "
                f"{row[3]:,.2f}"
            )

            print(
                f"Max Y:           "
                f"{row[4]:,.2f}"
            )

            print(
                f"Approx width:    "
                f"{width_km:,.2f} km"
            )

            print(
                f"Approx height:   "
                f"{height_km:,.2f} km"
            )

    finally:

        connection.close()

    if not results:

        raise RuntimeError(
            "No spatial metadata could be read."
        )

    # --------------------------------------------------
    # Verify CRS consistency
    # --------------------------------------------------

    srs_ids = {
        item["srs_id"]
        for item in results
    }

    crs_values = {
        (
            item["organization"],
            item["organization_crs"]
        )
        for item in results
    }

    print()
    print("=" * 110)
    print("CONSISTENCY CHECK")
    print("=" * 110)

    print(
        f"Unique SRS IDs: "
        f"{sorted(srs_ids)}"
    )

    print(
        f"Unique CRS definitions: "
        f"{sorted(crs_values)}"
    )

    same_crs = (
        len(srs_ids) == 1
        and len(crs_values) == 1
    )

    if same_crs:

        print(
            "[PASS] All CartoGen layers "
            "use the same CRS."
        )

    else:

        print(
            "[WARNING] Layers do not all "
            "use the same CRS."
        )

    # --------------------------------------------------
    # Verify bounding-box overlap
    # --------------------------------------------------

    common_min_x = max(
        item["min_x"]
        for item in results
    )

    common_min_y = max(
        item["min_y"]
        for item in results
    )

    common_max_x = min(
        item["max_x"]
        for item in results
    )

    common_max_y = min(
        item["max_y"]
        for item in results
    )

    common_overlap = (
        common_min_x < common_max_x
        and common_min_y < common_max_y
    )

    print()

    if common_overlap:

        print(
            "[PASS] All layers have a "
            "common spatial extent."
        )

        print(
            "Common bounding box:"
        )

        print(
            f"  xmin = "
            f"{common_min_x:,.2f}"
        )

        print(
            f"  ymin = "
            f"{common_min_y:,.2f}"
        )

        print(
            f"  xmax = "
            f"{common_max_x:,.2f}"
        )

        print(
            f"  ymax = "
            f"{common_max_y:,.2f}"
        )

    else:

        print(
            "[WARNING] No common bounding "
            "box exists across all layers."
        )

    output_data = {
        "geopackage":
            str(
                GPKG.relative_to(
                    PROJECT_ROOT
                )
            ),

        "layers":
            results,

        "validation": {
            "same_crs":
                same_crs,

            "srs_ids":
                sorted(srs_ids),

            "common_spatial_overlap":
                common_overlap,

            "common_bbox":
                (
                    {
                        "min_x":
                            common_min_x,
                        "min_y":
                            common_min_y,
                        "max_x":
                            common_max_x,
                        "max_y":
                            common_max_y,
                    }
                    if common_overlap
                    else None
                ),
        },
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output_data,
            file,
            indent=2
        )

    print()
    print("=" * 110)
    print("SCRIPT 17 COMPLETE")
    print("=" * 110)

    print(
        f"Metadata saved:\n{OUTPUT}"
    )


if __name__ == "__main__":
    main()