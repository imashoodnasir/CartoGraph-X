from pathlib import Path
import json
import time
import urllib.parse
import urllib.request
import urllib.error

import geopandas as gpd
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WINDOW_JSON = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "lithuania"
    / "windows"
    / "lithuania_development_windows.json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "gdr50lt_windows"
)

SUMMARY_CSV = (
    OUTPUT_ROOT
    / "gdr50lt_window_extraction_summary.csv"
)

MANIFEST_JSON = (
    OUTPUT_ROOT
    / "gdr50lt_window_extraction_manifest.json"
)

SERVICE_ROOT = (
    "https://www.geoportal.lt/"
    "arcgis/rest/services/"
    "NZT/GDR50LT/MapServer"
)

CRS = "EPSG:3346"

TARGET_LAYERS = {
    "target_buildings": {
        "id": 21,
        "name": "Pastatai",
        "geometry": "Polygon",
    },
    "target_roads": {
        "id": 19,
        "name": "Keliai",
        "geometry": "LineString",
    },
}

BATCH_SIZE = 500

MAX_RETRIES = 10

RETRY_SECONDS = 5


# ============================================================
# HTTP
# ============================================================

def request_json(
    url,
    params
):

    encoded = urllib.parse.urlencode(
        params
    )

    request_url = (
        f"{url}?"
        f"{encoded}"
    )

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            request = urllib.request.Request(
                request_url,
                headers={
                    "User-Agent":
                        "Mozilla/5.0 "
                        "CartoGen Research",

                    "Accept":
                        "application/json",
                }
            )

            with urllib.request.urlopen(
                request,
                timeout=120
            ) as response:

                raw = response.read()

            data = json.loads(
                raw.decode("utf-8")
            )

            if "error" in data:

                raise RuntimeError(
                    data["error"]
                )

            return data

        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            RuntimeError,
            json.JSONDecodeError
        ) as exc:

            print(
                f"      Attempt "
                f"{attempt}/"
                f"{MAX_RETRIES} "
                f"failed: {exc}"
            )

            if (
                attempt
                == MAX_RETRIES
            ):
                raise

            time.sleep(
                RETRY_SECONDS
            )


# ============================================================
# GEOMETRY CONVERSION
# ============================================================

def arcgis_polygon_to_shapely(
    geometry
):

    from shapely.geometry import (
        Polygon,
        MultiPolygon
    )

    rings = geometry.get(
        "rings",
        []
    )

    if not rings:
        return None

    polygons = []

    for ring in rings:

        if len(ring) < 4:
            continue

        try:

            polygon = Polygon(
                ring
            )

            if (
                not polygon.is_empty
                and polygon.area > 0
            ):

                if not polygon.is_valid:
                    polygon = (
                        polygon.buffer(0)
                    )

                if (
                    polygon.geom_type
                    == "Polygon"
                ):
                    polygons.append(
                        polygon
                    )

                elif (
                    polygon.geom_type
                    == "MultiPolygon"
                ):
                    polygons.extend(
                        list(
                            polygon.geoms
                        )
                    )

        except Exception:
            continue

    if not polygons:
        return None

    if len(polygons) == 1:
        return polygons[0]

    return MultiPolygon(
        polygons
    )


def arcgis_polyline_to_shapely(
    geometry
):

    from shapely.geometry import (
        LineString,
        MultiLineString
    )

    paths = geometry.get(
        "paths",
        []
    )

    lines = []

    for path in paths:

        if len(path) < 2:
            continue

        try:

            line = LineString(
                path
            )

            if not line.is_empty:
                lines.append(
                    line
                )

        except Exception:
            continue

    if not lines:
        return None

    if len(lines) == 1:
        return lines[0]

    return MultiLineString(
        lines
    )


# ============================================================
# QUERY IDS
# ============================================================

def query_object_ids(
    layer_id,
    bbox
):

    query_url = (
        f"{SERVICE_ROOT}/"
        f"{layer_id}/query"
    )

    xmin, ymin, xmax, ymax = bbox

    params = {
        "f":
            "json",

        "where":
            "1=1",

        "geometry":
            (
                f"{xmin},"
                f"{ymin},"
                f"{xmax},"
                f"{ymax}"
            ),

        "geometryType":
            "esriGeometryEnvelope",

        "inSR":
            "3346",

        "spatialRel":
            "esriSpatialRelIntersects",

        "returnIdsOnly":
            "true",
    }

    data = request_json(
        query_url,
        params
    )

    object_ids = (
        data.get(
            "objectIds"
        )
        or []
    )

    return sorted(
        object_ids
    )


# ============================================================
# QUERY FEATURES
# ============================================================

def query_feature_batch(
    layer_id,
    object_ids
):

    query_url = (
        f"{SERVICE_ROOT}/"
        f"{layer_id}/query"
    )

    params = {
        "f":
            "json",

        "objectIds":
            ",".join(
                str(x)
                for x in object_ids
            ),

        "outFields":
            "*",

        "returnGeometry":
            "true",

        "outSR":
            "3346",

        "returnZ":
            "false",

        "returnM":
            "false",
    }

    return request_json(
        query_url,
        params
    )


# ============================================================
# DOWNLOAD LAYER
# ============================================================

def download_layer(
    output_name,
    config,
    bbox
):

    layer_id = (
        config["id"]
    )

    print(
        "    Querying matching "
        "OBJECTIDs..."
    )

    object_ids = query_object_ids(
        layer_id,
        bbox
    )

    print(
        f"    Matching IDs: "
        f"{len(object_ids):,}"
    )

    if not object_ids:

        return gpd.GeoDataFrame(
            geometry=[],
            crs=CRS
        )

    rows = []

    total_batches = (
        len(object_ids)
        + BATCH_SIZE
        - 1
    ) // BATCH_SIZE

    for batch_number, start in enumerate(
        range(
            0,
            len(object_ids),
            BATCH_SIZE
        ),
        start=1
    ):

        batch_ids = (
            object_ids[
                start:
                start + BATCH_SIZE
            ]
        )

        print(
            f"    Batch "
            f"{batch_number}/"
            f"{total_batches} "
            f"({len(batch_ids)} features)"
        )

        data = query_feature_batch(
            layer_id,
            batch_ids
        )

        features = (
            data.get(
                "features"
            )
            or []
        )

        for feature in features:

            attributes = (
                feature.get(
                    "attributes"
                )
                or {}
            )

            geometry_json = (
                feature.get(
                    "geometry"
                )
                or {}
            )

            if (
                config["geometry"]
                == "Polygon"
            ):

                geometry = (
                    arcgis_polygon_to_shapely(
                        geometry_json
                    )
                )

            else:

                geometry = (
                    arcgis_polyline_to_shapely(
                        geometry_json
                    )
                )

            if geometry is None:
                continue

            row = dict(
                attributes
            )

            row["geometry"] = (
                geometry
            )

            rows.append(
                row
            )

    if not rows:

        return gpd.GeoDataFrame(
            geometry=[],
            crs=CRS
        )

    gdf = gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs=CRS
    )

    # Remove duplicate OBJECTIDs defensively.

    if "OBJECTID" in gdf.columns:

        before = len(gdf)

        gdf = (
            gdf.drop_duplicates(
                subset=[
                    "OBJECTID"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        removed = (
            before
            - len(gdf)
        )

        if removed:

            print(
                f"    Removed "
                f"{removed} duplicate "
                f"OBJECTIDs."
            )

    return gdf


# ============================================================
# STATISTICS
# ============================================================

def geometry_statistics(
    gdf,
    kind
):

    result = {
        "feature_count":
            int(len(gdf)),

        "valid_count":
            int(
                gdf.geometry.is_valid.sum()
            )
            if len(gdf)
            else 0,

        "invalid_count":
            int(
                (~gdf.geometry.is_valid).sum()
            )
            if len(gdf)
            else 0,

        "empty_count":
            int(
                gdf.geometry.is_empty.sum()
            )
            if len(gdf)
            else 0,
    }

    if len(gdf) == 0:
        return result

    if kind == "building":

        values = (
            gdf.geometry.area
        )

        result[
            "total_area_m2"
        ] = float(
            values.sum()
        )

        result[
            "median_area_m2"
        ] = float(
            values.median()
        )

        result[
            "mean_area_m2"
        ] = float(
            values.mean()
        )

    else:

        values = (
            gdf.geometry.length
        )

        result[
            "total_length_m"
        ] = float(
            values.sum()
        )

        result[
            "median_length_m"
        ] = float(
            values.median()
        )

        result[
            "mean_length_m"
        ] = float(
            values.mean()
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 120)
    print(
        "CARTOGEN — EXTRACT "
        "LITHUANIAN GDR50LT TARGET WINDOWS"
    )
    print("=" * 120)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    if not WINDOW_JSON.exists():

        raise FileNotFoundError(
            WINDOW_JSON
        )

    with open(
        WINDOW_JSON,
        "r",
        encoding="utf-8"
    ) as file:

        windows = json.load(
            file
        )

    summary_rows = []

    manifest = {
        "country":
            "Lithuania",

        "dataset":
            "GDR50LT",

        "role":
            "generalized_target",

        "scale":
            "1:50000",

        "crs":
            CRS,

        "service":
            SERVICE_ROOT,

        "windows":
            {},
    }

    for window in windows:

        name = (
            window["window"]
        )

        city = (
            window["city"]
        )

        bbox = (
            float(window["xmin"]),
            float(window["ymin"]),
            float(window["xmax"]),
            float(window["ymax"]),
        )

        print()
        print("=" * 120)
        print(
            f"{name} ({city})"
        )
        print("=" * 120)

        print(
            "BBox: "
            f"{bbox[0]:,.2f}, "
            f"{bbox[1]:,.2f}, "
            f"{bbox[2]:,.2f}, "
            f"{bbox[3]:,.2f}"
        )

        output_gpkg = (
            OUTPUT_ROOT
            / f"{name}_gdr50lt.gpkg"
        )

        if output_gpkg.exists():
            output_gpkg.unlink()

        window_manifest = {
            "city":
                city,

            "bbox":
                list(bbox),

            "output":
                str(output_gpkg),

            "layers":
                {},
        }

        for output_name, config in (
            TARGET_LAYERS.items()
        ):

            print()
            print("-" * 120)
            print(
                f"{output_name} "
                f"<- GDR50LT "
                f"{config['name']} "
                f"(Layer {config['id']})"
            )
            print("-" * 120)

            start_time = (
                time.time()
            )

            gdf = download_layer(
                output_name,
                config,
                bbox
            )

            elapsed = (
                time.time()
                - start_time
            )

            kind = (
                "building"
                if output_name
                == "target_buildings"
                else "road"
            )

            stats = (
                geometry_statistics(
                    gdf,
                    kind
                )
            )

            print()
            print(
                f"    Features:      "
                f"{stats['feature_count']:,}"
            )

            print(
                f"    Valid:         "
                f"{stats['valid_count']:,}"
            )

            print(
                f"    Invalid:       "
                f"{stats['invalid_count']:,}"
            )

            print(
                f"    Empty:         "
                f"{stats['empty_count']:,}"
            )

            if kind == "building":

                print(
                    f"    Total area:    "
                    f"{stats.get('total_area_m2', 0) / 1e6:,.3f} km²"
                )

                print(
                    f"    Median area:   "
                    f"{stats.get('median_area_m2', 0):,.2f} m²"
                )

            else:

                print(
                    f"    Total length:  "
                    f"{stats.get('total_length_m', 0) / 1000:,.2f} km"
                )

                print(
                    f"    Median length: "
                    f"{stats.get('median_length_m', 0):,.2f} m"
                )

            print(
                f"    Query time:    "
                f"{elapsed:.2f}s"
            )

            if len(gdf) > 0:

                gdf.to_file(
                    output_gpkg,
                    layer=output_name,
                    driver="GPKG",
                    engine="pyogrio"
                )

                print(
                    f"    Saved:         "
                    f"{output_name}"
                )

            else:

                print(
                    "    [WARNING] "
                    "No features returned."
                )

            window_manifest[
                "layers"
            ][output_name] = {
                **stats,

                "source_layer_id":
                    config["id"],

                "source_layer_name":
                    config["name"],

                "query_seconds":
                    elapsed,
            }

            summary_rows.append({
                "window":
                    name,

                "city":
                    city,

                "layer":
                    output_name,

                **stats,
            })

        manifest[
            "windows"
        ][name] = (
            window_manifest
        )

        print()
        print(
            f"GeoPackage:\n"
            f"{output_gpkg}"
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False
    )

    with open(
        MANIFEST_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("=" * 120)
    print("GDR50LT EXTRACTION SUMMARY")
    print("=" * 120)

    for window in windows:

        name = (
            window["window"]
        )

        city = (
            window["city"]
        )

        subset = (
            summary_df[
                summary_df[
                    "window"
                ]
                == name
            ]
        )

        print()
        print(
            f"{name} ({city})"
        )

        for _, row in (
            subset.iterrows()
        ):

            print(
                f"  "
                f"{row['layer']:<25}"
                f"{int(row['feature_count']):>10,}"
            )

    print()
    print("=" * 120)
    print("SCRIPT 34 COMPLETE")
    print("=" * 120)

    print()
    print(
        f"Summary CSV:\n"
        f"{SUMMARY_CSV}"
    )

    print()
    print(
        f"Manifest:\n"
        f"{MANIFEST_JSON}"
    )


if __name__ == "__main__":
    main()