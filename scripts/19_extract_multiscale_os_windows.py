from pathlib import Path
import json
import sqlite3
import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_GPKG = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "os"
    / "OpenZoomstack"
    / "OS_Open_Zoomstack.gpkg"
)

WINDOW_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "os"
    / "windows"
    / "development_windows.json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "os"
    / "development_windows"
)

MANIFEST_FILE = (
    OUTPUT_ROOT
    / "extraction_manifest.json"
)


LAYERS = [

    "local_buildings",
    "district_buildings",

    "roads_local",
    "roads_regional",
    "roads_national",

]


def load_windows():

    with open(
        WINDOW_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def get_layer_metadata(
    connection,
    layer
):

    row = connection.execute(
        """
        SELECT
            column_name,
            srs_id
        FROM gpkg_geometry_columns
        WHERE table_name = ?
        """,
        (layer,)
    ).fetchone()

    if row is None:

        raise RuntimeError(
            f"No geometry metadata "
            f"for {layer}"
        )

    return {
        "geometry_column": row[0],
        "srs_id": row[1],
    }


def get_primary_key(
    connection,
    layer
):

    rows = connection.execute(
        f'PRAGMA table_info("{layer}")'
    ).fetchall()

    for row in rows:

        if row[5] == 1:
            return row[1]

    # Most GeoPackages use fid.
    return "fid"


def check_rtree(
    connection,
    layer,
    geometry_column
):

    rtree = (
        f"rtree_{layer}_"
        f"{geometry_column}"
    )

    row = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        AND name = ?
        """,
        (rtree,)
    ).fetchone()

    return (
        rtree
        if row
        else None
    )


def get_candidate_fids(
    connection,
    rtree,
    min_x,
    min_y,
    max_x,
    max_y
):

    query = f"""
        SELECT id
        FROM "{rtree}"
        WHERE maxx >= ?
          AND minx <= ?
          AND maxy >= ?
          AND miny <= ?
    """

    rows = connection.execute(
        query,
        (
            min_x,
            max_x,
            min_y,
            max_y,
        )
    ).fetchall()

    return [
        row[0]
        for row in rows
    ]


def extract_layer(
    connection,
    layer,
    bbox,
):

    min_x = bbox["min_x"]
    min_y = bbox["min_y"]
    max_x = bbox["max_x"]
    max_y = bbox["max_y"]

    metadata = get_layer_metadata(
        connection,
        layer
    )

    geometry_column = (
        metadata["geometry_column"]
    )

    primary_key = get_primary_key(
        connection,
        layer
    )

    rtree = check_rtree(
        connection,
        layer,
        geometry_column
    )

    if rtree is None:

        raise RuntimeError(
            f"Spatial RTree missing "
            f"for layer {layer}. "
            f"Do not perform a full "
            f"national table scan."
        )

    candidate_ids = (
        get_candidate_fids(
            connection,
            rtree,
            min_x,
            min_y,
            max_x,
            max_y
        )
    )

    print(
        f"    RTree candidates: "
        f"{len(candidate_ids):,}"
    )

    if not candidate_ids:

        return gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:27700"
        )

    # GeoPandas/Fiona/Pyogrio can read
    # the actual geometry subset using
    # a bounding box efficiently.
    #
    # The RTree check above guarantees
    # that the source has an index.

    gdf = gpd.read_file(
        SOURCE_GPKG,
        layer=layer,
        bbox=(
            min_x,
            min_y,
            max_x,
            max_y
        ),
        engine="pyogrio"
    )

    # Clip features crossing the
    # development-window boundary.
    from shapely.geometry import box

    clipping_box = box(
        min_x,
        min_y,
        max_x,
        max_y
    )

    if not gdf.empty:

        gdf = gdf[
            gdf.geometry.intersects(
                clipping_box
            )
        ].copy()

        gdf["geometry"] = (
            gdf.geometry.intersection(
                clipping_box
            )
        )

        gdf = gdf[
            ~gdf.geometry.is_empty
        ].copy()

    return gdf


def main():

    print("=" * 110)
    print(
        "CARTOGEN — EXTRACT "
        "MULTI-SCALE OS WINDOWS"
    )
    print("=" * 110)

    if not SOURCE_GPKG.exists():

        raise FileNotFoundError(
            SOURCE_GPKG
        )

    if not WINDOW_FILE.exists():

        raise FileNotFoundError(
            "Run script 18 first."
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    windows = load_windows()

    connection = sqlite3.connect(
        f"file:{SOURCE_GPKG}?mode=ro",
        uri=True
    )

    manifest = {}

    try:

        for window in windows:

            window_id = (
                window["window_id"]
            )

            print()
            print("=" * 110)
            print(
                f"WINDOW: {window_id}"
            )
            print("=" * 110)

            output_file = (
                OUTPUT_ROOT
                / f"{window_id}.gpkg"
            )

            if output_file.exists():
                output_file.unlink()

            manifest[window_id] = {
                "city":
                    window["city"],

                "bbox": {
                    "min_x":
                        window["min_x"],
                    "min_y":
                        window["min_y"],
                    "max_x":
                        window["max_x"],
                    "max_y":
                        window["max_y"],
                },

                "layers": {},
            }

            for layer in LAYERS:

                print()
                print(
                    f"  Extracting: "
                    f"{layer}"
                )

                gdf = extract_layer(
                    connection,
                    layer,
                    window
                )

                print(
                    f"    Final features: "
                    f"{len(gdf):,}"
                )

                if not gdf.empty:

                    gdf.to_file(
                        output_file,
                        layer=layer,
                        driver="GPKG",
                        engine="pyogrio"
                    )

                manifest[
                    window_id
                ]["layers"][layer] = {
                    "feature_count":
                        len(gdf),

                    "geometry_types":
                        (
                            sorted(
                                gdf.geometry
                                .geom_type
                                .dropna()
                                .unique()
                                .tolist()
                            )
                            if not gdf.empty
                            else []
                        ),
                }

            print()
            print(
                f"  Saved:\n"
                f"  {output_file}"
            )

    finally:

        connection.close()

    with open(
        MANIFEST_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2
        )

    print()
    print("=" * 110)
    print("EXTRACTION SUMMARY")
    print("=" * 110)

    for window_id, info in (
        manifest.items()
    ):

        print()
        print(
            f"{window_id} "
            f"({info['city']})"
        )

        for layer, layer_info in (
            info["layers"].items()
        ):

            print(
                f"  "
                f"{layer:<25}"
                f"{layer_info['feature_count']:>10,}"
            )

    print()
    print("=" * 110)
    print("SCRIPT 19 COMPLETE")
    print("=" * 110)

    print(
        f"Manifest:\n"
        f"{MANIFEST_FILE}"
    )


if __name__ == "__main__":
    main()