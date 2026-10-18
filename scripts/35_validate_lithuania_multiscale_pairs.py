from pathlib import Path
import json

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import make_valid


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRPK_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "grpk_windows"
)

GDR_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "gdr50lt_windows"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "validated_pairs"
)

SUMMARY_CSV = (
    OUTPUT_ROOT
    / "lithuania_multiscale_validation.csv"
)

SUMMARY_JSON = (
    OUTPUT_ROOT
    / "lithuania_multiscale_validation.json"
)

CITIES = [
    ("vilnius_central", "Vilnius"),
    ("kaunas_central", "Kaunas"),
    ("klaipeda_central", "Klaipeda"),
]

CRS_EPSG = 3346


# ============================================================
# HELPERS
# ============================================================

def read_layer(path, layer):

    if not path.exists():
        raise FileNotFoundError(path)

    gdf = gpd.read_file(
        path,
        layer=layer,
        engine="pyogrio"
    )

    return gdf


def repair_geometry(gdf, family):

    gdf = gdf.copy()

    original_count = len(gdf)

    invalid_before = int(
        (~gdf.geometry.is_valid).sum()
    )

    empty_before = int(
        gdf.geometry.is_empty.sum()
    )

    null_before = int(
        gdf.geometry.isna().sum()
    )

    # Repair only invalid, non-null geometries.
    mask = (
        gdf.geometry.notna()
        & ~gdf.geometry.is_valid
    )

    if mask.any():

        gdf.loc[
            mask,
            "geometry"
        ] = gdf.loc[
            mask,
            "geometry"
        ].apply(make_valid)

    # Extract appropriate geometry family from collections.
    def normalize_geometry(geom):

        if geom is None:
            return None

        if geom.is_empty:
            return None

        if family == "building":

            if geom.geom_type in (
                "Polygon",
                "MultiPolygon"
            ):
                return geom

            if geom.geom_type == "GeometryCollection":

                polygons = [
                    x
                    for x in geom.geoms
                    if x.geom_type in (
                        "Polygon",
                        "MultiPolygon"
                    )
                ]

                if not polygons:
                    return None

                from shapely.ops import unary_union

                return unary_union(
                    polygons
                )

        if family == "road":

            if geom.geom_type in (
                "LineString",
                "MultiLineString"
            ):
                return geom

            if geom.geom_type == "GeometryCollection":

                lines = [
                    x
                    for x in geom.geoms
                    if x.geom_type in (
                        "LineString",
                        "MultiLineString"
                    )
                ]

                if not lines:
                    return None

                from shapely.ops import unary_union

                return unary_union(
                    lines
                )

        return None

    gdf["geometry"] = (
        gdf.geometry.apply(
            normalize_geometry
        )
    )

    gdf = (
        gdf[
            gdf.geometry.notna()
            & ~gdf.geometry.is_empty
        ]
        .copy()
        .reset_index(drop=True)
    )

    invalid_after = int(
        (~gdf.geometry.is_valid).sum()
    )

    return gdf, {
        "original_count":
            original_count,

        "invalid_before":
            invalid_before,

        "empty_before":
            empty_before,

        "null_before":
            null_before,

        "final_count":
            int(len(gdf)),

        "invalid_after":
            invalid_after,

        "removed_after_repair":
            int(
                original_count
                - len(gdf)
            ),
    }


def building_stats(gdf):

    areas = gdf.geometry.area

    return {
        "count":
            int(len(gdf)),

        "total_measure":
            float(
                areas.sum()
            ),

        "mean_measure":
            float(
                areas.mean()
            ),

        "median_measure":
            float(
                areas.median()
            ),

        "p25_measure":
            float(
                areas.quantile(0.25)
            ),

        "p75_measure":
            float(
                areas.quantile(0.75)
            ),
    }


def road_stats(gdf):

    lengths = gdf.geometry.length

    return {
        "count":
            int(len(gdf)),

        "total_measure":
            float(
                lengths.sum()
            ),

        "mean_measure":
            float(
                lengths.mean()
            ),

        "median_measure":
            float(
                lengths.median()
            ),

        "p25_measure":
            float(
                lengths.quantile(0.25)
            ),

        "p75_measure":
            float(
                lengths.quantile(0.75)
            ),
    }


def safe_ratio(
    target,
    source
):

    if source == 0:
        return np.nan

    return (
        target
        / source
        * 100.0
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 120)
    print(
        "CARTOGEN — VALIDATE LITHUANIAN "
        "MULTI-SCALE SOURCE/TARGET PAIRS"
    )
    print("=" * 120)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    summary = []
    json_results = {}

    for window, city in CITIES:

        print()
        print("=" * 120)
        print(
            f"{window} ({city})"
        )
        print("=" * 120)

        source_file = (
            GRPK_ROOT
            / f"{window}_grpk.gpkg"
        )

        target_file = (
            GDR_ROOT
            / f"{window}_gdr50lt.gpkg"
        )

        output_file = (
            OUTPUT_ROOT
            / f"{window}_validated.gpkg"
        )

        if output_file.exists():
            output_file.unlink()

        # ====================================================
        # BUILDINGS
        # ====================================================

        print()
        print("-" * 120)
        print("BUILDINGS")
        print("-" * 120)

        source_buildings = read_layer(
            source_file,
            "source_buildings"
        )

        target_buildings = read_layer(
            target_file,
            "target_buildings"
        )

        if (
            source_buildings.crs.to_epsg()
            != CRS_EPSG
            or
            target_buildings.crs.to_epsg()
            != CRS_EPSG
        ):
            raise RuntimeError(
                "Unexpected building CRS."
            )

        source_buildings, sb_repair = (
            repair_geometry(
                source_buildings,
                "building"
            )
        )

        target_buildings, tb_repair = (
            repair_geometry(
                target_buildings,
                "building"
            )
        )

        sb = building_stats(
            source_buildings
        )

        tb = building_stats(
            target_buildings
        )

        building_retention = (
            safe_ratio(
                tb["count"],
                sb["count"]
            )
        )

        building_reduction = (
            100.0
            - building_retention
        )

        building_area_ratio = (
            safe_ratio(
                tb["total_measure"],
                sb["total_measure"]
            )
        )

        median_area_ratio = (
            tb["median_measure"]
            / sb["median_measure"]
            if sb["median_measure"] > 0
            else np.nan
        )

        print(
            f"Source count:             "
            f"{sb['count']:,}"
        )

        print(
            f"Target count:             "
            f"{tb['count']:,}"
        )

        print(
            f"Feature retention:        "
            f"{building_retention:.2f}%"
        )

        print(
            f"Feature reduction:        "
            f"{building_reduction:.2f}%"
        )

        print(
            f"Source total area:        "
            f"{sb['total_measure']/1e6:.3f} km²"
        )

        print(
            f"Target total area:        "
            f"{tb['total_measure']/1e6:.3f} km²"
        )

        print(
            f"Area ratio:               "
            f"{building_area_ratio:.2f}%"
        )

        print(
            f"Source median area:       "
            f"{sb['median_measure']:.2f} m²"
        )

        print(
            f"Target median area:       "
            f"{tb['median_measure']:.2f} m²"
        )

        print(
            f"Median area multiplier:   "
            f"{median_area_ratio:.3f}×"
        )

        print(
            f"Target invalid before:    "
            f"{tb_repair['invalid_before']:,}"
        )

        print(
            f"Target invalid after:     "
            f"{tb_repair['invalid_after']:,}"
        )

        # ====================================================
        # ROADS
        # ====================================================

        print()
        print("-" * 120)
        print("ROADS")
        print("-" * 120)

        source_roads = read_layer(
            source_file,
            "source_roads"
        )

        target_roads = read_layer(
            target_file,
            "target_roads"
        )

        if (
            source_roads.crs.to_epsg()
            != CRS_EPSG
            or
            target_roads.crs.to_epsg()
            != CRS_EPSG
        ):
            raise RuntimeError(
                "Unexpected road CRS."
            )

        source_roads, sr_repair = (
            repair_geometry(
                source_roads,
                "road"
            )
        )

        target_roads, tr_repair = (
            repair_geometry(
                target_roads,
                "road"
            )
        )

        sr = road_stats(
            source_roads
        )

        tr = road_stats(
            target_roads
        )

        road_retention = (
            safe_ratio(
                tr["count"],
                sr["count"]
            )
        )

        road_reduction = (
            100.0
            - road_retention
        )

        road_length_ratio = (
            safe_ratio(
                tr["total_measure"],
                sr["total_measure"]
            )
        )

        median_length_ratio = (
            tr["median_measure"]
            / sr["median_measure"]
            if sr["median_measure"] > 0
            else np.nan
        )

        print(
            f"Source count:             "
            f"{sr['count']:,}"
        )

        print(
            f"Target count:             "
            f"{tr['count']:,}"
        )

        print(
            f"Feature retention:        "
            f"{road_retention:.2f}%"
        )

        print(
            f"Feature reduction:        "
            f"{road_reduction:.2f}%"
        )

        print(
            f"Source total length:      "
            f"{sr['total_measure']/1000:.2f} km"
        )

        print(
            f"Target total length:      "
            f"{tr['total_measure']/1000:.2f} km"
        )

        print(
            f"Length retention:         "
            f"{road_length_ratio:.2f}%"
        )

        print(
            f"Source median segment:    "
            f"{sr['median_measure']:.2f} m"
        )

        print(
            f"Target median segment:    "
            f"{tr['median_measure']:.2f} m"
        )

        print(
            f"Median length multiplier: "
            f"{median_length_ratio:.3f}×"
        )

        # ====================================================
        # SAVE VALIDATED GEOMETRY
        # ====================================================

        source_buildings.to_file(
            output_file,
            layer="source_buildings",
            driver="GPKG",
            engine="pyogrio"
        )

        target_buildings.to_file(
            output_file,
            layer="target_buildings",
            driver="GPKG",
            engine="pyogrio"
        )

        source_roads.to_file(
            output_file,
            layer="source_roads",
            driver="GPKG",
            engine="pyogrio"
        )

        target_roads.to_file(
            output_file,
            layer="target_roads",
            driver="GPKG",
            engine="pyogrio"
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        building_row = {
            "window":
                window,

            "city":
                city,

            "family":
                "building",

            "source_count":
                sb["count"],

            "target_count":
                tb["count"],

            "feature_retention_pct":
                building_retention,

            "feature_reduction_pct":
                building_reduction,

            "source_total_measure":
                sb["total_measure"],

            "target_total_measure":
                tb["total_measure"],

            "measure_retention_pct":
                building_area_ratio,

            "source_median_measure":
                sb["median_measure"],

            "target_median_measure":
                tb["median_measure"],

            "median_measure_multiplier":
                median_area_ratio,

            "target_invalid_before":
                tb_repair["invalid_before"],

            "target_invalid_after":
                tb_repair["invalid_after"],
        }

        road_row = {
            "window":
                window,

            "city":
                city,

            "family":
                "road",

            "source_count":
                sr["count"],

            "target_count":
                tr["count"],

            "feature_retention_pct":
                road_retention,

            "feature_reduction_pct":
                road_reduction,

            "source_total_measure":
                sr["total_measure"],

            "target_total_measure":
                tr["total_measure"],

            "measure_retention_pct":
                road_length_ratio,

            "source_median_measure":
                sr["median_measure"],

            "target_median_measure":
                tr["median_measure"],

            "median_measure_multiplier":
                median_length_ratio,

            "target_invalid_before":
                tr_repair["invalid_before"],

            "target_invalid_after":
                tr_repair["invalid_after"],
        }

        summary.extend([
            building_row,
            road_row
        ])

        json_results[
            window
        ] = {
            "city":
                city,

            "buildings": {
                "source_stats":
                    sb,

                "target_stats":
                    tb,

                "source_repair":
                    sb_repair,

                "target_repair":
                    tb_repair,

                "retention_pct":
                    building_retention,

                "reduction_pct":
                    building_reduction,

                "area_retention_pct":
                    building_area_ratio,
            },

            "roads": {
                "source_stats":
                    sr,

                "target_stats":
                    tr,

                "source_repair":
                    sr_repair,

                "target_repair":
                    tr_repair,

                "retention_pct":
                    road_retention,

                "reduction_pct":
                    road_reduction,

                "length_retention_pct":
                    road_length_ratio,
            },
        }

        print()
        print(
            f"Validated GeoPackage:\n"
            f"{output_file}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    summary_df = pd.DataFrame(
        summary
    )

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False
    )

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            json_results,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    print()
    print("=" * 120)
    print("GLOBAL SOURCE -> TARGET SUMMARY")
    print("=" * 120)

    display_columns = [
        "city",
        "family",
        "source_count",
        "target_count",
        "feature_retention_pct",
        "feature_reduction_pct",
        "measure_retention_pct",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False,
            formatters={
                "feature_retention_pct":
                    lambda x: f"{x:.2f}",

                "feature_reduction_pct":
                    lambda x: f"{x:.2f}",

                "measure_retention_pct":
                    lambda x: f"{x:.2f}",
            }
        )
    )

    print()
    print("=" * 120)
    print("SCRIPT 35 COMPLETE")
    print("=" * 120)

    print()
    print(
        f"CSV:\n{SUMMARY_CSV}"
    )

    print()
    print(
        f"JSON:\n{SUMMARY_JSON}"
    )


if __name__ == "__main__":
    main()