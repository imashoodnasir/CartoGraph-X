from pathlib import Path
import json
import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "os"
    / "development_windows"
)

OUTPUT_JSON = (
    DATA_ROOT
    / "multiscale_validation.json"
)

OUTPUT_CSV = (
    DATA_ROOT
    / "multiscale_validation.csv"
)


WINDOWS = [
    "london_central",
    "manchester_central",
    "birmingham_central",
]


LAYERS = [
    "local_buildings",
    "district_buildings",
    "roads_local",
    "roads_regional",
    "roads_national",
]


def geometry_statistics(gdf):

    if gdf.empty:

        return {
            "count": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "empty_count": 0,
            "null_count": 0,
            "valid_pct": 0.0,
        }

    null_mask = gdf.geometry.isna()
    empty_mask = gdf.geometry.is_empty
    valid_mask = gdf.geometry.is_valid

    count = len(gdf)

    valid_count = int(
        valid_mask.sum()
    )

    invalid_count = int(
        (~valid_mask & ~null_mask).sum()
    )

    empty_count = int(
        empty_mask.sum()
    )

    null_count = int(
        null_mask.sum()
    )

    return {
        "count":
            count,

        "valid_count":
            valid_count,

        "invalid_count":
            invalid_count,

        "empty_count":
            empty_count,

        "null_count":
            null_count,

        "valid_pct":
            (
                100.0
                * valid_count
                / count
            ),
    }


def building_statistics(gdf):

    if gdf.empty:
        return {}

    areas = gdf.geometry.area

    return {
        "total_area_m2":
            float(areas.sum()),

        "mean_area_m2":
            float(areas.mean()),

        "median_area_m2":
            float(areas.median()),

        "min_area_m2":
            float(areas.min()),

        "max_area_m2":
            float(areas.max()),
    }


def road_statistics(gdf):

    if gdf.empty:
        return {}

    lengths = gdf.geometry.length

    return {
        "total_length_km":
            float(
                lengths.sum()
                / 1000.0
            ),

        "mean_length_m":
            float(
                lengths.mean()
            ),

        "median_length_m":
            float(
                lengths.median()
            ),

        "min_length_m":
            float(
                lengths.min()
            ),

        "max_length_m":
            float(
                lengths.max()
            ),
    }


def main():

    print("=" * 120)
    print(
        "CARTOGEN — VALIDATE "
        "MULTI-SCALE GEOMETRY"
    )
    print("=" * 120)

    results = []
    nested = {}

    for window in WINDOWS:

        gpkg = (
            DATA_ROOT
            / f"{window}.gpkg"
        )

        if not gpkg.exists():

            raise FileNotFoundError(
                gpkg
            )

        print()
        print("=" * 120)
        print(window)
        print("=" * 120)

        nested[window] = {}

        for layer in LAYERS:

            print()
            print(
                f"Reading {layer}..."
            )

            gdf = gpd.read_file(
                gpkg,
                layer=layer,
                engine="pyogrio"
            )

            base = (
                geometry_statistics(
                    gdf
                )
            )

            record = {
                "window":
                    window,

                "layer":
                    layer,

                "crs":
                    str(gdf.crs),

                **base,
            }

            if "building" in layer:

                record.update(
                    building_statistics(
                        gdf
                    )
                )

            elif "roads_" in layer:

                record.update(
                    road_statistics(
                        gdf
                    )
                )

            nested[
                window
            ][layer] = record

            results.append(record)

            print(
                f"  Features: "
                f"{len(gdf):,}"
            )

            print(
                f"  CRS: "
                f"{gdf.crs}"
            )

            print(
                f"  Valid: "
                f"{base['valid_pct']:.2f}%"
            )

            print(
                f"  Invalid: "
                f"{base['invalid_count']:,}"
            )

            print(
                f"  Empty: "
                f"{base['empty_count']:,}"
            )

            if "building" in layer:

                stats = (
                    building_statistics(
                        gdf
                    )
                )

                print(
                    f"  Total building area: "
                    f"{stats['total_area_m2']/1e6:.3f} km²"
                )

                print(
                    f"  Median building area: "
                    f"{stats['median_area_m2']:.2f} m²"
                )

            else:

                stats = (
                    road_statistics(
                        gdf
                    )
                )

                print(
                    f"  Total road length: "
                    f"{stats['total_length_km']:.2f} km"
                )

                print(
                    f"  Median segment length: "
                    f"{stats['median_length_m']:.2f} m"
                )

        # ---------------------------------------------
        # Generalization ratios
        # ---------------------------------------------

        lb = nested[
            window
        ]["local_buildings"]["count"]

        db = nested[
            window
        ]["district_buildings"]["count"]

        rl = nested[
            window
        ]["roads_local"]["count"]

        rr = nested[
            window
        ]["roads_regional"]["count"]

        rn = nested[
            window
        ]["roads_national"]["count"]

        print()
        print("-" * 120)
        print("GENERALIZATION RATIOS")
        print("-" * 120)

        building_retention = (
            100.0 * db / lb
            if lb
            else 0
        )

        regional_retention = (
            100.0 * rr / rl
            if rl
            else 0
        )

        national_retention = (
            100.0 * rn / rl
            if rl
            else 0
        )

        print(
            f"Buildings "
            f"local → district: "
            f"{building_retention:.2f}% "
            f"feature retention"
        )

        print(
            f"Roads "
            f"local → regional: "
            f"{regional_retention:.2f}% "
            f"feature retention"
        )

        print(
            f"Roads "
            f"local → national: "
            f"{national_retention:.2f}% "
            f"feature retention"
        )

        nested[
            window
        ]["generalization_ratios"] = {

            "building_retention_pct":
                building_retention,

            "road_regional_retention_pct":
                regional_retention,

            "road_national_retention_pct":
                national_retention,
        }

    df = pd.DataFrame(
        results
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
            nested,
            file,
            indent=2
        )

    print()
    print("=" * 120)
    print("SCRIPT 20 COMPLETE")
    print("=" * 120)

    print(
        f"CSV:\n{OUTPUT_CSV}"
    )

    print(
        f"\nJSON:\n{OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()