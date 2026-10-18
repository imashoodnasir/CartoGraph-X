from pathlib import Path
import json

import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from pyproj import Transformer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "lithuania"
    / "windows"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "lithuania_development_windows.json"
)

OUTPUT_GPKG = (
    OUTPUT_DIR
    / "lithuania_development_windows.gpkg"
)

CRS = "EPSG:3346"

WINDOW_SIZE_M = 6000.0
HALF_SIZE = WINDOW_SIZE_M / 2.0


# City centres in WGS84.
# They will be transformed programmatically to EPSG:3346.

CITIES = [
    {
        "window": "vilnius_central",
        "city": "Vilnius",
        "purpose": "primary_development",
        "latitude": 54.6872,
        "longitude": 25.2797,
    },
    {
        "window": "kaunas_central",
        "city": "Kaunas",
        "purpose": "secondary_development",
        "latitude": 54.8985,
        "longitude": 23.9036,
    },
    {
        "window": "klaipeda_central",
        "city": "Klaipeda",
        "purpose": "secondary_development",
        "latitude": 55.7033,
        "longitude": 21.1443,
    },
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 120)
    print(
        "CARTOGEN — DEFINE LITHUANIAN "
        "DEVELOPMENT WINDOWS"
    )
    print("=" * 120)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        CRS,
        always_xy=True
    )

    records = []
    geometries = []

    for item in CITIES:

        x, y = transformer.transform(
            item["longitude"],
            item["latitude"]
        )

        xmin = x - HALF_SIZE
        ymin = y - HALF_SIZE
        xmax = x + HALF_SIZE
        ymax = y + HALF_SIZE

        geometry = box(
            xmin,
            ymin,
            xmax,
            ymax
        )

        record = {
            "window":
                item["window"],

            "city":
                item["city"],

            "purpose":
                item["purpose"],

            "latitude":
                item["latitude"],

            "longitude":
                item["longitude"],

            "centre_x":
                x,

            "centre_y":
                y,

            "width_m":
                WINDOW_SIZE_M,

            "height_m":
                WINDOW_SIZE_M,

            "area_km2":
                (
                    WINDOW_SIZE_M
                    * WINDOW_SIZE_M
                    / 1_000_000
                ),

            "xmin":
                xmin,

            "ymin":
                ymin,

            "xmax":
                xmax,

            "ymax":
                ymax,

            "crs":
                CRS,
        }

        records.append(record)
        geometries.append(geometry)

        print()
        print("-" * 120)

        print(
            f"Window:   "
            f"{item['window']}"
        )

        print(
            f"City:     "
            f"{item['city']}"
        )

        print(
            f"Purpose:  "
            f"{item['purpose']}"
        )

        print(
            f"WGS84:    "
            f"{item['latitude']:.6f}, "
            f"{item['longitude']:.6f}"
        )

        print(
            f"Centre:   "
            f"({x:,.2f}, {y:,.2f})"
        )

        print(
            f"Size:     "
            f"{WINDOW_SIZE_M / 1000:.1f} × "
            f"{WINDOW_SIZE_M / 1000:.1f} km"
        )

        print(
            f"Area:     "
            f"{record['area_km2']:.2f} km²"
        )

        print(
            f"BBox:     "
            f"{xmin:,.2f}, "
            f"{ymin:,.2f}, "
            f"{xmax:,.2f}, "
            f"{ymax:,.2f}"
        )

    # --------------------------------------------------------
    # GeoDataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(records)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=geometries,
        crs=CRS
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("=" * 120)
    print("VALIDATION")
    print("=" * 120)

    print(
        f"Windows created: "
        f"{len(gdf)}"
    )

    print(
        f"CRS: "
        f"{gdf.crs}"
    )

    valid_count = int(
        gdf.geometry.is_valid.sum()
    )

    print(
        f"Valid geometries: "
        f"{valid_count}/{len(gdf)}"
    )

    expected_area = (
        WINDOW_SIZE_M
        * WINDOW_SIZE_M
    )

    area_ok = True

    for _, row in gdf.iterrows():

        difference = abs(
            row.geometry.area
            - expected_area
        )

        if difference > 1.0:
            area_ok = False

    if valid_count == len(gdf):

        print(
            "[PASS] All window "
            "geometries valid."
        )

    else:

        print(
            "[FAIL] Invalid window "
            "geometry detected."
        )

    if area_ok:

        print(
            "[PASS] All windows "
            "are 36 km²."
        )

    else:

        print(
            "[FAIL] Unexpected "
            "window area."
        )

    # --------------------------------------------------------
    # Save GeoPackage
    # --------------------------------------------------------

    if OUTPUT_GPKG.exists():
        OUTPUT_GPKG.unlink()

    gdf.to_file(
        OUTPUT_GPKG,
        layer="development_windows",
        driver="GPKG"
    )

    # --------------------------------------------------------
    # Save JSON without geometry
    # --------------------------------------------------------

    json_records = []

    for record in records:

        clean = {}

        for key, value in record.items():

            if hasattr(value, "item"):
                value = value.item()

            clean[key] = value

        json_records.append(clean)

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            json_records,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 120)
    print("SCRIPT 32 COMPLETE")
    print("=" * 120)

    print()
    print("JSON:")
    print(OUTPUT_JSON)

    print()
    print("GeoPackage:")
    print(OUTPUT_GPKG)


if __name__ == "__main__":
    main()