from pathlib import Path
import json
import geopandas as gpd
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "os"
    / "windows"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "development_windows.json"
)

OUTPUT_GPKG = (
    OUTPUT_DIR
    / "development_windows.gpkg"
)


# ---------------------------------------------------------
# Development windows
#
# Coordinates are British National Grid (EPSG:27700).
#
# Initial windows are deliberately moderate in size.
# We are NOT yet defining final train/test splits.
# These are development/visual-validation regions.
# ---------------------------------------------------------

WINDOWS = [

    {
        "window_id": "london_central",
        "city": "London",
        "purpose": "primary_development",
        "center_x": 530000,
        "center_y": 180000,
        "width_m": 6000,
        "height_m": 6000,
    },

    {
        "window_id": "manchester_central",
        "city": "Manchester",
        "purpose": "secondary_development",
        "center_x": 384000,
        "center_y": 398000,
        "width_m": 6000,
        "height_m": 6000,
    },

    {
        "window_id": "birmingham_central",
        "city": "Birmingham",
        "purpose": "secondary_development",
        "center_x": 407500,
        "center_y": 287000,
        "width_m": 6000,
        "height_m": 6000,
    },

]


def create_bbox(
    center_x,
    center_y,
    width_m,
    height_m
):

    half_width = width_m / 2.0
    half_height = height_m / 2.0

    return box(
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )


def main():

    print("=" * 110)
    print("CARTOGEN — DEFINE DEVELOPMENT WINDOWS")
    print("=" * 110)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []
    json_records = []

    for item in WINDOWS:

        geometry = create_bbox(
            item["center_x"],
            item["center_y"],
            item["width_m"],
            item["height_m"],
        )

        min_x, min_y, max_x, max_y = (
            geometry.bounds
        )

        area_km2 = (
            geometry.area
            / 1_000_000.0
        )

        record = {
            **item,
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
            "area_km2": area_km2,
            "geometry": geometry,
        }

        records.append(record)

        json_records.append({
            key: value
            for key, value in record.items()
            if key != "geometry"
        })

        print()
        print("-" * 110)

        print(
            f"Window:   "
            f"{item['window_id']}"
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
            f"Centre:   "
            f"({item['center_x']:,.0f}, "
            f"{item['center_y']:,.0f})"
        )

        print(
            f"Size:     "
            f"{item['width_m']/1000:.1f} × "
            f"{item['height_m']/1000:.1f} km"
        )

        print(
            f"Area:     "
            f"{area_km2:.2f} km²"
        )

        print(
            f"BBox:     "
            f"{min_x:,.0f}, "
            f"{min_y:,.0f}, "
            f"{max_x:,.0f}, "
            f"{max_y:,.0f}"
        )

    gdf = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs="EPSG:27700"
    )

    if OUTPUT_GPKG.exists():
        OUTPUT_GPKG.unlink()

    gdf.to_file(
        OUTPUT_GPKG,
        layer="development_windows",
        driver="GPKG"
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            json_records,
            file,
            indent=2
        )

    print()
    print("=" * 110)
    print("VALIDATION")
    print("=" * 110)

    assert len(gdf) == len(WINDOWS)

    assert (
        gdf.crs.to_epsg()
        == 27700
    )

    assert all(
        gdf.geometry.is_valid
    )

    print(
        f"[PASS] Windows created: "
        f"{len(gdf)}"
    )

    print(
        "[PASS] CRS: EPSG:27700"
    )

    print(
        "[PASS] All window "
        "geometries valid."
    )

    print()
    print("=" * 110)
    print("SCRIPT 18 COMPLETE")
    print("=" * 110)

    print(
        f"JSON:\n{OUTPUT_JSON}"
    )

    print(
        f"\nGeoPackage:\n"
        f"{OUTPUT_GPKG}"
    )


if __name__ == "__main__":
    main()