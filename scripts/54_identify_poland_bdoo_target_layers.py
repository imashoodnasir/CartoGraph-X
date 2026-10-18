from pathlib import Path
import json

import geopandas as gpd
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BDOO_ROOT = ROOT / "data" / "interim" / "poland" / "bdoo_inventory"
OUT_ROOT = ROOT / "data" / "processed" / "poland" / "bdoo"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


CITIES = {
    "warsaw": {"city": "Warsaw", "code": "14"},
    "krakow": {"city": "Krakow", "code": "12"},
    "gdansk": {"city": "Gdansk", "code": "22"},
}


CANDIDATES = [
    "OT_PTZB_A",
    "OT_ADJA_A",
    "OT_PTUT_A",
    "OT_PTLZ_A",
    "OT_BUUO_L",
    "OT_SKDR_L",
]


def banner(text):
    print()
    print("=" * 120)
    print(text)
    print("=" * 120)


def section(text):
    print()
    print("-" * 120)
    print(text)
    print("-" * 120)


def find_layer(city_key, class_name):

    root = BDOO_ROOT / city_key / "extracted"

    matches = list(root.rglob(f"*__{class_name}.gml"))
    matches += list(root.rglob(f"*__{class_name}.xml"))

    if not matches:
        return None

    return max(matches, key=lambda p: p.stat().st_size)


def safe_value(v):

    if pd.isna(v):
        return None

    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass

    return str(v)


def inspect_layer(path):

    gdf = gpd.read_file(
        path,
        engine="pyogrio"
    )

    geom_counts = (
        gdf.geometry.geom_type
        .value_counts(dropna=False)
        .to_dict()
    )

    attributes = {}

    for col in gdf.columns:

        if col == "geometry":
            continue

        s = gdf[col]

        vals = []

        for value in s.dropna():

            try:
                if hasattr(value, "tolist"):
                    value = value.tolist()

                if isinstance(value, (list, tuple, dict, set)):
                    value = str(value)

                elif not isinstance(
                    value,
                    (str, int, float, bool)
                ):
                    value = str(value)

                if value not in vals:
                    vals.append(value)

                if len(vals) >= 10:
                    break

            except Exception:
                continue

        attributes[col] = {
            "dtype": str(s.dtype),
            "unique_count": None,
            "samples": [safe_value(x) for x in vals],
        }

    info = {
        "features": int(len(gdf)),
        "crs": str(gdf.crs),
        "geometry_types": {
            str(k): int(v)
            for k, v in geom_counts.items()
        },
        "null_geometry": int(gdf.geometry.isna().sum()),
        "empty_geometry": int(gdf.geometry.is_empty.sum()),
        "columns": [str(c) for c in gdf.columns],
        "attributes": attributes,
    }

    if len(gdf):

        bounds = gdf.total_bounds

        info["bounds"] = [
            float(x) for x in bounds
        ]

    return info, gdf


def export_layer(gdf, path, layer_name):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if path.exists():
        path.unlink()

    if gdf is None:
        raise RuntimeError(
            "GeoDataFrame is None."
        )

    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(
            f"Expected GeoDataFrame, got {type(gdf)}"
        )

    gdf.to_file(
        path,
        layer=layer_name,
        driver="GPKG",
        engine="pyogrio",
    )


def main():

    banner(
        "CARTOGEN — IDENTIFY POLISH BDOO "
        "GENERALIZED TARGET LAYERS"
    )

    report = {"cities": {}}
    rows = []

    for city_key, cfg in CITIES.items():

        city = cfg["city"]

        banner(
            f"{city.upper()} — BDOO TARGET INSPECTION"
        )

        city_report = {}

        for class_name in CANDIDATES:

            section(class_name)

            path = find_layer(
                city_key,
                class_name
            )

            if path is None:

                print("[CHECK] File not found.")

                city_report[class_name] = {
                    "status": "NOT_FOUND"
                }

                continue

            print(f"Input: {path}")
            print(
                f"Size: "
                f"{path.stat().st_size / 1024**2:.3f} MB"
            )

            try:

                info, gdf = inspect_layer(path)

            except Exception as exc:

                print(
                    "[CHECK] Read failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                city_report[class_name] = {
                    "status": "READ_FAILED",
                    "error": str(exc),
                }

                continue

            print(f"Features: {info['features']:,}")
            print(f"CRS:      {info['crs']}")

            print("\nGeometry types:")

            for geom, n in info[
                "geometry_types"
            ].items():

                print(
                    f"  {geom:25s} {n:,}"
                )

            print("\nColumns:")

            for col in info["columns"]:
                print(f"  {col}")

            print("\nAttribute samples:")

            for col, attr in info[
                "attributes"
            ].items():

                print(
                    f"  {col}: "
                    f"{attr['samples'][:8]}"
                )

            output = (
                OUT_ROOT
                / city_key
                / "candidate_layers"
                / f"{city_key}_{class_name.lower()}.gpkg"
            )

            try:

                export_layer(
                    gdf,
                    output,
                    class_name
                )

                export_status = "PASS"

                print(
                    "\n[PASS] GeoPackage:"
                )
                print(output)

            except Exception as exc:

                export_status = "FAILED"

                print(
                    "\n[CHECK] GeoPackage export failed:"
                )
                print(
                    f"{type(exc).__name__}: {exc}"
                )

            city_report[class_name] = {
                "status": "PASS",
                "input": str(path),
                "output": str(output),
                "export_status": export_status,
                **info,
            }

            rows.append({
                "city": city,
                "class": class_name,
                "features": info["features"],
                "geometry": ",".join(
                    info["geometry_types"].keys()
                ),
                "crs": info["crs"],
                "size_mb": round(
                    path.stat().st_size / 1024**2,
                    3
                ),
                "export": export_status,
            })

        report["cities"][city_key] = city_report

    banner("GLOBAL BDOO CLASS SUMMARY")

    df = pd.DataFrame(rows)

    if not df.empty:

        print(df.to_string(index=False))

        csv_path = (
            OUT_ROOT
            / "bdoo_candidate_layer_comparison.csv"
        )

        df.to_csv(
            csv_path,
            index=False
        )

        print(f"\nCSV:\n{csv_path}")

    banner("CARTOGEN TARGET ASSESSMENT")

    print(
        "Road target:"
    )
    print(
        "  OT_SKDR_L remains the primary BDOO "
        "generalized road target."
    )

    print()
    print(
        "Building representation:"
    )
    print(
        "  OT_PTZB_A is retained as the primary "
        "candidate for built-up/generalized settlement "
        "representation."
    )

    print(
        "  It is NOT yet being declared equivalent "
        "to individual BDOT10k OT_BUBD_A building "
        "footprints."
    )

    print()
    print(
        "Next validation:"
    )
    print(
        "  Spatially compare OT_BUBD_A with OT_PTZB_A "
        "inside the three city windows to determine "
        "whether PTZB represents building aggregation, "
        "built-up blocks, settlement zones, or another "
        "generalized representation."
    )

    report["current_decision"] = {
        "road_target": "OT_SKDR_L",
        "building_source": "OT_BUBD_A",
        "building_target_candidate": "OT_PTZB_A",
        "building_pairing_confirmed": False,
    }

    json_path = (
        OUT_ROOT
        / "bdoo_target_layer_identification.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    banner("SCRIPT 54 COMPLETE")

    print(f"JSON:\n{json_path}")

    print(
        "\nNEXT: Script 55 — spatial and semantic "
        "validation of BDOT10k OT_BUBD_A -> "
        "BDOO OT_PTZB_A and OT_SKDR_L -> OT_SKDR_L."
    )


if __name__ == "__main__":
    main()