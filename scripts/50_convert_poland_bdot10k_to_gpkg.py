from pathlib import Path
import json
import sys
import time
import traceback

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "poland"
    / "bdot10k_inventory"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "poland"
    / "bdot10k"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


CITY_CONFIG = {
    "warsaw": {
        "prefix": "PL.PZGiK.330.1465",
        "teryt": "1465",
    },
    "krakow": {
        "prefix": "PL.PZGiK.283.1261",
        "teryt": "1261",
    },
    "gdansk": {
        "prefix": "PL.PZGiK.336.2261",
        "teryt": "2261",
    },
}


CLASS_CONFIG = {
    "buildings": {
        "class_name": "OT_BUBD_A",
        "expected_geometry": {"Polygon", "MultiPolygon"},
    },
    "roads": {
        "class_name": "OT_SKDR_L",
        "expected_geometry": {
            "LineString",
            "MultiLineString",
        },
    },
}


def banner(text, char="="):
    print()
    print(char * 120)
    print(text)
    print(char * 120)


def find_xml(city, class_name):

    root = (
        INPUT_ROOT
        / city
        / "outer_extracted"
    )

    candidates = list(
        root.rglob(f"*__{class_name}.xml")
    )

    if not candidates:
        raise FileNotFoundError(
            f"Could not find {class_name} for {city}"
        )

    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple {class_name} files found "
            f"for {city}: {candidates}"
        )

    return candidates[0]


def detect_engine():

    try:
        import pyogrio
        return "pyogrio"
    except ImportError:
        return "fiona"


def read_xml(path, engine):

    print(f"Reading with GeoPandas engine: {engine}")

    try:
        gdf = gpd.read_file(
            path,
            engine=engine,
        )
        return gdf

    except Exception as e:

        print()
        print(
            "[WARNING] Direct read failed."
        )
        print(
            f"{type(e).__name__}: {e}"
        )

        # Some GDAL builds behave better when the
        # GML content has a .gml extension.
        alias = path.with_suffix(".gml")

        print()
        print(
            "Creating temporary .gml alias..."
        )

        # Do not duplicate hundreds of MB unless necessary.
        # Hard link points to the same underlying data.
        if not alias.exists():
            try:
                alias.hardlink_to(path)
            except Exception:
                import shutil
                print(
                    "Hard link unavailable; copying file."
                )
                shutil.copy2(path, alias)

        try:
            gdf = gpd.read_file(
                alias,
                engine=engine,
            )
            return gdf

        finally:
            try:
                alias.unlink()
            except Exception:
                pass


def clean_columns(gdf):

    # Flatten duplicate/awkward names only where needed.
    seen = {}
    new_columns = []

    for col in gdf.columns:

        name = str(col).strip()

        if name == "":
            name = "unnamed"

        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0

        new_columns.append(name)

    gdf.columns = new_columns

    return gdf


def add_metadata(gdf, city, cfg, role, class_name):

    gdf.insert(
        0,
        "cartogen_city",
        city
    )

    gdf.insert(
        1,
        "cartogen_country",
        "Poland"
    )

    gdf.insert(
        2,
        "cartogen_teryt",
        cfg["teryt"]
    )

    gdf.insert(
        3,
        "cartogen_source",
        "BDOT10k"
    )

    gdf.insert(
        4,
        "cartogen_role",
        role
    )

    gdf.insert(
        5,
        "cartogen_class",
        class_name
    )

    return gdf


def validate_layer(gdf, expected_geometry):

    result = {}

    result["features"] = int(len(gdf))

    result["crs"] = (
        str(gdf.crs)
        if gdf.crs is not None
        else None
    )

    result["null_geometry"] = int(
        gdf.geometry.isna().sum()
    )

    result["empty_geometry"] = int(
        gdf.geometry.is_empty.sum()
    )

    geom_counts = (
        gdf.geometry
        .geom_type
        .value_counts(dropna=False)
        .to_dict()
    )

    result["geometry_types"] = {
        str(k): int(v)
        for k, v in geom_counts.items()
    }

    actual = set(
        gdf.geometry
        .dropna()
        .geom_type
        .unique()
    )

    unexpected = sorted(
        actual - expected_geometry
    )

    result["unexpected_geometry_types"] = unexpected

    result["expected_geometry_only"] = (
        len(unexpected) == 0
    )

    if len(gdf) > 0:

        bounds = gdf.total_bounds

        result["bounds"] = {
            "minx": float(bounds[0]),
            "miny": float(bounds[1]),
            "maxx": float(bounds[2]),
            "maxy": float(bounds[3]),
        }

    else:
        result["bounds"] = None

    return result


def write_layer(gdf, output_path, layer_name, engine):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if output_path.exists():
        output_path.unlink()

    print()
    print("Writing GeoPackage...")

    gdf.to_file(
        output_path,
        layer=layer_name,
        driver="GPKG",
        engine=engine,
        index=False,
    )


def verify_roundtrip(path, layer_name, engine):

    print("Verifying written layer...")

    test = gpd.read_file(
        path,
        layer=layer_name,
        engine=engine,
    )

    return {
        "features": int(len(test)),
        "crs": (
            str(test.crs)
            if test.crs is not None
            else None
        ),
        "geometry_types": {
            str(k): int(v)
            for k, v in
            test.geometry.geom_type
            .value_counts(dropna=False)
            .to_dict()
            .items()
        },
    }


def main():

    banner(
        "CARTOGEN — CONVERT POLISH BDOT10k "
        "BUILDINGS AND ROADS TO GEOPACKAGE"
    )

    print(f"GeoPandas: {gpd.__version__}")

    engine = detect_engine()

    print(f"I/O engine: {engine}")

    report = {
        "engine": engine,
        "cities": {},
    }

    summary_rows = []

    for city, cfg in CITY_CONFIG.items():

        banner(city.upper())

        city_report = {}

        for role, class_cfg in CLASS_CONFIG.items():

            class_name = class_cfg["class_name"]

            print()
            print("-" * 120)
            print(
                f"{role.upper()} — {class_name}"
            )
            print("-" * 120)

            xml_path = find_xml(
                city,
                class_name
            )

            print(f"Input: {xml_path}")
            print(
                f"Input size: "
                f"{xml_path.stat().st_size / 1024 / 1024:.2f} MB"
            )

            start = time.time()

            try:

                gdf = read_xml(
                    xml_path,
                    engine
                )

                elapsed_read = time.time() - start

                print()
                print(
                    f"Read features: {len(gdf):,}"
                )
                print(
                    f"Read time: {elapsed_read:.1f} s"
                )
                print(
                    f"CRS: {gdf.crs}"
                )

                if gdf.crs is None:
                    raise RuntimeError(
                        "CRS missing after GML/XML read."
                    )

                # Script 49 established EPSG:2180.
                epsg = gdf.crs.to_epsg()

                if epsg != 2180:
                    raise RuntimeError(
                        f"Unexpected CRS: {gdf.crs}. "
                        "Expected EPSG:2180."
                    )

                gdf = clean_columns(gdf)

                gdf = add_metadata(
                    gdf,
                    city,
                    cfg,
                    role,
                    class_name
                )

                validation = validate_layer(
                    gdf,
                    class_cfg[
                        "expected_geometry"
                    ]
                )

                print()
                print("Geometry types:")

                for geom, count in (
                    validation[
                        "geometry_types"
                    ].items()
                ):
                    print(
                        f"  {geom:<25} "
                        f"{count:>10,}"
                    )

                print(
                    f"Null geometries:  "
                    f"{validation['null_geometry']:,}"
                )

                print(
                    f"Empty geometries: "
                    f"{validation['empty_geometry']:,}"
                )

                if not validation[
                    "expected_geometry_only"
                ]:
                    raise RuntimeError(
                        "Unexpected geometry types: "
                        + ", ".join(
                            validation[
                                "unexpected_geometry_types"
                            ]
                        )
                    )

                output_path = (
                    OUTPUT_ROOT
                    / city
                    / f"{city}_{role}.gpkg"
                )

                layer_name = (
                    f"source_{role}"
                )

                write_start = time.time()

                write_layer(
                    gdf,
                    output_path,
                    layer_name,
                    engine
                )

                elapsed_write = (
                    time.time() - write_start
                )

                verify = verify_roundtrip(
                    output_path,
                    layer_name,
                    engine
                )

                if verify["features"] != len(gdf):
                    raise RuntimeError(
                        "Round-trip feature count mismatch."
                    )

                if (
                    gpd.read_file(
                        output_path,
                        layer=layer_name,
                        rows=1,
                        engine=engine
                    ).crs.to_epsg()
                    != 2180
                ):
                    raise RuntimeError(
                        "Round-trip CRS mismatch."
                    )

                print()
                print("[PASS] Conversion complete")
                print(f"Output: {output_path}")
                print(
                    f"Output size: "
                    f"{output_path.stat().st_size / 1024 / 1024:.2f} MB"
                )
                print(
                    f"Write time: {elapsed_write:.1f} s"
                )

                layer_report = {
                    "status": "PASS",
                    "input": str(xml_path),
                    "output": str(output_path),
                    "layer": layer_name,
                    "read_seconds": elapsed_read,
                    "write_seconds": elapsed_write,
                    "validation": validation,
                    "roundtrip": verify,
                }

                summary_rows.append({
                    "city": city,
                    "role": role,
                    "class_name": class_name,
                    "features": len(gdf),
                    "crs": str(gdf.crs),
                    "null_geometry":
                        validation["null_geometry"],
                    "empty_geometry":
                        validation["empty_geometry"],
                    "geometry_types":
                        "; ".join(
                            f"{k}:{v}"
                            for k, v in
                            validation[
                                "geometry_types"
                            ].items()
                        ),
                    "output": str(output_path),
                    "status": "PASS",
                })

            except Exception as e:

                print()
                print("[FAIL]")
                print(
                    f"{type(e).__name__}: {e}"
                )

                traceback.print_exc()

                layer_report = {
                    "status": "FAIL",
                    "input": str(xml_path),
                    "error":
                        f"{type(e).__name__}: {e}",
                }

                summary_rows.append({
                    "city": city,
                    "role": role,
                    "class_name": class_name,
                    "features": None,
                    "crs": None,
                    "null_geometry": None,
                    "empty_geometry": None,
                    "geometry_types": None,
                    "output": None,
                    "status": "FAIL",
                })

            city_report[role] = layer_report

        report["cities"][city] = city_report

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    banner("CONVERSION SUMMARY")

    summary = pd.DataFrame(summary_rows)

    print(
        summary[
            [
                "city",
                "role",
                "features",
                "crs",
                "status",
            ]
        ].to_string(index=False)
    )

    summary_csv = (
        OUTPUT_ROOT
        / "bdot10k_conversion_summary.csv"
    )

    summary.to_csv(
        summary_csv,
        index=False
    )

    report_json = (
        OUTPUT_ROOT
        / "bdot10k_conversion_report.json"
    )

    with open(
        report_json,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    all_pass = all(
        row["status"] == "PASS"
        for row in summary_rows
    )

    banner("SCRIPT 50 COMPLETE")

    print(f"Summary: {summary_csv}")
    print(f"Report:  {report_json}")

    if all_pass:
        print()
        print(
            "[PASS] All Polish BDOT10k source "
            "layers converted successfully."
        )
        print()
        print(
            "NEXT: Script 51 — acquire and inspect "
            "the corresponding BDOO generalized "
            "building and road target data."
        )
    else:
        print()
        print(
            "[CHECK] One or more conversions failed."
        )
        print(
            "Do not continue to BDOO processing until "
            "the failed layer is resolved."
        )


if __name__ == "__main__":
    main()