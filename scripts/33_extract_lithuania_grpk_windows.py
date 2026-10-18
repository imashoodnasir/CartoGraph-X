from pathlib import Path
import json
import time

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely.geometry import box


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRPK_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "lithuania"
    / "grpk"
    / "extracted"
    / "GRPK_Open_SHP"
)

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
    / "grpk_windows"
)

MANIFEST_JSON = (
    OUTPUT_ROOT
    / "grpk_window_extraction_manifest.json"
)

SUMMARY_CSV = (
    OUTPUT_ROOT
    / "grpk_window_extraction_summary.csv"
)

CRS = "EPSG:3346"


LAYERS = {
    "source_buildings": {
        "file": GRPK_ROOT / "PASTAT.shp",
        "expected_geometry": "Polygon",
    },
    "source_roads": {
        "file": GRPK_ROOT / "KELIAI.shp",
        "expected_geometry": "LineString",
    },
}


# ============================================================
# HELPERS
# ============================================================

def load_windows():

    if not WINDOW_JSON.exists():
        raise FileNotFoundError(
            f"Window definition not found:\n"
            f"{WINDOW_JSON}"
        )

    with open(
        WINDOW_JSON,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def validate_sources():

    print()
    print("=" * 120)
    print("SOURCE VALIDATION")
    print("=" * 120)

    for name, config in LAYERS.items():

        path = config["file"]

        print()
        print(name)
        print("-" * 120)
        print(path)

        if not path.exists():
            raise FileNotFoundError(path)

        info = pyogrio.read_info(path)

        geometry_type = info.get(
            "geometry_type"
        )

        feature_count = int(
            info.get(
                "features",
                0
            )
        )

        crs = info.get("crs")

        print(
            f"National features: "
            f"{feature_count:,}"
        )

        print(
            f"Geometry:          "
            f"{geometry_type}"
        )

        print(
            f"CRS:               "
            f"{crs}"
        )

        if (
            config["expected_geometry"]
            not in str(geometry_type)
        ):
            raise RuntimeError(
                f"Unexpected geometry for "
                f"{name}: {geometry_type}"
            )


def extract_bbox(
    path,
    bbox
):

    start = time.time()

    gdf = gpd.read_file(
        path,
        bbox=bbox,
        engine="pyogrio"
    )

    elapsed = (
        time.time()
        - start
    )

    return gdf, elapsed


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
            ),

        "invalid_count":
            int(
                (~gdf.geometry.is_valid).sum()
            ),

        "empty_count":
            int(
                gdf.geometry.is_empty.sum()
            ),
    }

    if len(gdf) == 0:
        return result

    if kind == "building":

        areas = (
            gdf.geometry.area
        )

        result[
            "total_area_m2"
        ] = float(
            areas.sum()
        )

        result[
            "median_area_m2"
        ] = float(
            areas.median()
        )

        result[
            "mean_area_m2"
        ] = float(
            areas.mean()
        )

    elif kind == "road":

        lengths = (
            gdf.geometry.length
        )

        result[
            "total_length_m"
        ] = float(
            lengths.sum()
        )

        result[
            "median_length_m"
        ] = float(
            lengths.median()
        )

        result[
            "mean_length_m"
        ] = float(
            lengths.mean()
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 120)
    print(
        "CARTOGEN — EXTRACT "
        "LITHUANIAN GRPK CITY WINDOWS"
    )
    print("=" * 120)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    validate_sources()

    windows = load_windows()

    manifest = {
        "country":
            "Lithuania",

        "dataset":
            "GRPK",

        "role":
            "detailed_source",

        "crs":
            CRS,

        "windows":
            {},
    }

    summary_rows = []

    for window in windows:

        window_name = (
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

        window_geom = box(
            *bbox
        )

        print()
        print("=" * 120)
        print(
            f"{window_name} "
            f"({city})"
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
            / f"{window_name}_grpk.gpkg"
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

        for output_layer, config in (
            LAYERS.items()
        ):

            print()
            print("-" * 120)
            print(
                f"Extracting "
                f"{output_layer}"
            )
            print("-" * 120)

            source_path = (
                config["file"]
            )

            gdf, elapsed = (
                extract_bbox(
                    source_path,
                    bbox
                )
            )

            print(
                f"BBox candidates: "
                f"{len(gdf):,}"
            )

            # ------------------------------------------------
            # Defensive CRS check
            # ------------------------------------------------

            if gdf.crs is None:

                raise RuntimeError(
                    f"{output_layer} "
                    f"has no CRS."
                )

            if (
                gdf.crs.to_epsg()
                != 3346
            ):

                raise RuntimeError(
                    f"{output_layer}: "
                    f"expected EPSG:3346, "
                    f"found {gdf.crs}"
                )

            # ------------------------------------------------
            # Exact intersection
            # ------------------------------------------------

            if len(gdf) > 0:

                intersects = (
                    gdf.geometry.intersects(
                        window_geom
                    )
                )

                gdf = (
                    gdf.loc[
                        intersects
                    ]
                    .copy()
                    .reset_index(
                        drop=True
                    )
                )

            print(
                f"Final intersecting: "
                f"{len(gdf):,}"
            )

            print(
                f"Read/filter time:   "
                f"{elapsed:.2f}s"
            )

            # ------------------------------------------------
            # Geometry diagnostics
            # ------------------------------------------------

            kind = (
                "building"
                if output_layer
                == "source_buildings"
                else "road"
            )

            stats = (
                geometry_statistics(
                    gdf,
                    kind
                )
            )

            print(
                f"Valid:              "
                f"{stats['valid_count']:,}"
            )

            print(
                f"Invalid:            "
                f"{stats['invalid_count']:,}"
            )

            print(
                f"Empty:              "
                f"{stats['empty_count']:,}"
            )

            if kind == "building":

                print(
                    "Total area:         "
                    f"{stats.get('total_area_m2', 0) / 1e6:,.3f} km²"
                )

                print(
                    "Median area:        "
                    f"{stats.get('median_area_m2', 0):,.2f} m²"
                )

            else:

                print(
                    "Total length:       "
                    f"{stats.get('total_length_m', 0) / 1000:,.2f} km"
                )

                print(
                    "Median length:      "
                    f"{stats.get('median_length_m', 0):,.2f} m"
                )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            gdf.to_file(
                output_gpkg,
                layer=output_layer,
                driver="GPKG",
                engine="pyogrio"
            )

            print(
                f"Saved layer:        "
                f"{output_layer}"
            )

            window_manifest[
                "layers"
            ][output_layer] = {
                **stats,
                "source_file":
                    str(source_path),

                "read_seconds":
                    elapsed,
            }

            summary_row = {
                "window":
                    window_name,

                "city":
                    city,

                "layer":
                    output_layer,

                **stats,
            }

            summary_rows.append(
                summary_row
            )

        manifest[
            "windows"
        ][window_name] = (
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
    # FINAL DISPLAY
    # ========================================================

    print()
    print("=" * 120)
    print("EXTRACTION SUMMARY")
    print("=" * 120)

    for window in windows:

        name = window["window"]

        subset = (
            summary_df[
                summary_df["window"]
                == name
            ]
        )

        print()
        print(
            f"{name} "
            f"({window['city']})"
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
    print("SCRIPT 33 COMPLETE")
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