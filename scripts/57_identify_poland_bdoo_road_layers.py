from pathlib import Path
import json
import pandas as pd
import geopandas as gpd

try:
    import pyogrio
except ImportError:
    pyogrio = None


ROOT = Path(r"D:\Research\CartoGen")

BDOO_ROOT = (
    ROOT / "data" / "processed" / "poland" / "bdoo"
)

OUT_ROOT = (
    ROOT / "data" / "interim" / "poland" /
    "bdoo_road_identification"
)

OUT_ROOT.mkdir(parents=True, exist_ok=True)

CITIES = ["warsaw", "krakow", "gdansk"]


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


def list_layers(path):

    if pyogrio is None:
        return []

    try:
        arr = pyogrio.list_layers(path)

        result = []

        for row in arr:
            result.append(
                {
                    "layer": str(row[0]),
                    "geometry_hint": (
                        str(row[1])
                        if len(row) > 1
                        else None
                    ),
                }
            )

        return result

    except Exception as e:

        print(
            f"[CHECK] Could not list layers: "
            f"{type(e).__name__}: {e}"
        )

        return []


def classify_geometry(gdf):

    if len(gdf) == 0:
        return "EMPTY"

    types = (
        gdf.geometry
        .dropna()
        .geom_type
        .value_counts()
        .to_dict()
    )

    if not types:
        return "UNKNOWN"

    line_count = sum(
        n for geom, n in types.items()
        if "Line" in geom
    )

    polygon_count = sum(
        n for geom, n in types.items()
        if "Polygon" in geom
    )

    point_count = sum(
        n for geom, n in types.items()
        if "Point" in geom
    )

    total = sum(types.values())

    if line_count / total >= 0.8:
        return "LINE"

    if polygon_count / total >= 0.8:
        return "POLYGON"

    if point_count / total >= 0.8:
        return "POINT"

    return "MIXED"


def road_name_score(text):

    text = text.lower()

    score = 0
    reasons = []

    strong = {
        "skdr": 100,
        "droga": 40,
        "drogi": 40,
        "road": 40,
        "ulica": 30,
        "street": 30,
        "transport": 15,
        "komunik": 15,
    }

    negative = {
        "adja": -100,
        "admin": -100,
        "granic": -80,
        "boundary": -80,
        "ptzb": -80,
        "zabud": -50,
        "hydro": -50,
        "rzeka": -50,
        "kolej": -20,
        "rail": -20,
    }

    for term, value in strong.items():
        if term in text:
            score += value
            reasons.append(f"+{value}:{term}")

    for term, value in negative.items():
        if term in text:
            score += value
            reasons.append(f"{value}:{term}")

    return score, reasons


def attribute_road_score(columns):

    cols = [str(c).lower() for c in columns]

    score = 0
    hits = []

    terms = [
        "kategoria",
        "kategor",
        "nawierz",
        "jezd",
        "liczba",
        "pas",
        "szerok",
        "numer",
        "klasa",
        "funkcja",
        "rodzajdrogi",
        "typdrogi",
    ]

    for term in terms:

        matches = [
            c for c in cols
            if term in c
        ]

        if matches:
            score += 5
            hits.extend(matches)

    return score, sorted(set(hits))


def inspect_layer(city, path, layer):

    try:

        gdf = gpd.read_file(
            path,
            layer=layer
        )

    except Exception as e:

        return {
            "city": city,
            "file": str(path),
            "layer": layer,
            "status": "READ_ERROR",
            "error": (
                f"{type(e).__name__}: {e}"
            ),
        }

    geometry_class = classify_geometry(gdf)

    geometry_types = (
        gdf.geometry
        .dropna()
        .geom_type
        .value_counts()
        .to_dict()
    )

    crs = str(gdf.crs)

    epsg = None

    try:
        epsg = gdf.crs.to_epsg()
    except Exception:
        pass

    name_score, name_reasons = road_name_score(
        f"{path.name} {layer}"
    )

    attr_score, attr_hits = attribute_road_score(
        gdf.columns
    )

    geometry_score = (
        50 if geometry_class == "LINE"
        else -50
    )

    total_score = (
        name_score
        + attr_score
        + geometry_score
    )

    total_length_km = None
    median_length_m = None

    if geometry_class == "LINE":

        work = gdf

        if epsg != 2180:

            try:
                work = gdf.to_crs(2180)
            except Exception:
                work = None

        if work is not None:

            lengths = (
                work.geometry
                .dropna()
                .length
            )

            total_length_km = float(
                lengths.sum() / 1000
            )

            median_length_m = (
                float(lengths.median())
                if len(lengths)
                else None
            )

    return {
        "city": city,
        "file": str(path),
        "filename": path.name,
        "layer": layer,
        "status": "OK",
        "features": int(len(gdf)),
        "geometry_class": geometry_class,
        "geometry_types": geometry_types,
        "crs": crs,
        "epsg": epsg,
        "columns": [
            str(c)
            for c in gdf.columns
            if c != gdf.geometry.name
        ],
        "name_score": name_score,
        "attribute_score": attr_score,
        "geometry_score": geometry_score,
        "road_candidate_score": total_score,
        "name_reasons": name_reasons,
        "road_attribute_hits": attr_hits,
        "total_length_km": total_length_km,
        "median_length_m": median_length_m,
    }


def main():

    banner(
        "CARTOGEN — IDENTIFY TRUE POLISH BDOO ROAD LAYERS"
    )

    print(f"BDOO root: {BDOO_ROOT}")

    all_records = []

    for city in CITIES:

        banner(city.upper())

        city_dir = BDOO_ROOT / city

        if not city_dir.exists():

            print(
                f"[FAIL] Directory not found: "
                f"{city_dir}"
            )

            continue

        gpkg_files = sorted(
            city_dir.rglob("*.gpkg")
        )

        print(
            f"GeoPackages discovered: "
            f"{len(gpkg_files)}"
        )

        for path in gpkg_files:

            section(
                f"FILE: {path.relative_to(city_dir)}"
            )

            layers = list_layers(path)

            if not layers:

                print(
                    "[CHECK] No readable layers."
                )

                continue

            for info in layers:

                layer = info["layer"]

                print()
                print(
                    f"Inspecting layer: {layer}"
                )

                rec = inspect_layer(
                    city,
                    path,
                    layer
                )

                all_records.append(rec)

                if rec["status"] != "OK":

                    print(
                        f"  [ERROR] "
                        f"{rec.get('error')}"
                    )

                    continue

                print(
                    f"  Features:       "
                    f"{rec['features']:,}"
                )

                print(
                    f"  Geometry:       "
                    f"{rec['geometry_class']}"
                )

                print(
                    f"  Geometry types: "
                    f"{rec['geometry_types']}"
                )

                print(
                    f"  CRS:            "
                    f"{rec['crs']}"
                )

                print(
                    f"  EPSG:           "
                    f"{rec['epsg']}"
                )

                print(
                    f"  Road score:     "
                    f"{rec['road_candidate_score']}"
                )

                if rec["road_attribute_hits"]:

                    print(
                        "  Road attributes: "
                        + ", ".join(
                            rec[
                                "road_attribute_hits"
                            ]
                        )
                    )

                if (
                    rec["total_length_km"]
                    is not None
                ):

                    print(
                        f"  Total length:   "
                        f"{rec['total_length_km']:,.2f} km"
                    )

                    print(
                        f"  Median segment: "
                        f"{rec['median_length_m']:,.2f} m"
                    )

    banner(
        "GLOBAL BDOO LINE-LAYER RANKING"
    )

    valid = [
        r for r in all_records
        if r.get("status") == "OK"
    ]

    if not valid:

        print("[FAIL] No readable layers.")
        return

    df = pd.DataFrame(valid)

    line_df = df[
        df["geometry_class"] == "LINE"
    ].copy()

    if len(line_df) == 0:

        print(
            "[FAIL] No BDOO line layers "
            "were found."
        )

    else:

        line_df = line_df.sort_values(
            [
                "city",
                "road_candidate_score",
                "features",
            ],
            ascending=[
                True,
                False,
                False,
            ]
        )

        display = line_df[
            [
                "city",
                "filename",
                "layer",
                "features",
                "epsg",
                "road_candidate_score",
                "total_length_km",
                "median_length_m",
            ]
        ]

        print(
            display.to_string(
                index=False
            )
        )

    banner(
        "BEST ROAD CANDIDATE PER CITY"
    )

    best_records = []

    for city in CITIES:

        candidates = line_df[
            line_df["city"] == city
        ].copy()

        if len(candidates) == 0:

            print()
            print(
                f"{city}: NO LINE CANDIDATE"
            )

            continue

        candidates = candidates.sort_values(
            [
                "road_candidate_score",
                "features",
            ],
            ascending=[
                False,
                False,
            ]
        )

        best = candidates.iloc[0]

        print()
        print(city.upper())
        print("-" * 80)

        print(
            f"File:     "
            f"{best['file']}"
        )

        print(
            f"Layer:    "
            f"{best['layer']}"
        )

        print(
            f"Features: "
            f"{int(best['features']):,}"
        )

        print(
            f"Score:    "
            f"{best['road_candidate_score']}"
        )

        print(
            f"Length:   "
            f"{best['total_length_km']:,.2f} km"
        )

        best_records.append(
            {
                "city": city,
                "file": best["file"],
                "layer": best["layer"],
                "features": int(
                    best["features"]
                ),
                "road_candidate_score": float(
                    best[
                        "road_candidate_score"
                    ]
                ),
                "total_length_km": float(
                    best[
                        "total_length_km"
                    ]
                ),
            }
        )

    banner(
        "OT_ADJA_A SANITY CHECK"
    )

    adja = df[
        df["layer"]
        .astype(str)
        .str.upper()
        .str.contains("ADJA")
    ]

    if len(adja):

        for _, row in adja.iterrows():

            print(
                f"{row['city']:<10} "
                f"{row['layer']:<20} "
                f"geometry={row['geometry_class']:<10} "
                f"features={row['features']:,} "
                f"score={row['road_candidate_score']}"
            )

        print()
        print(
            "[CONFIRMATION] OT_ADJA_A must NOT be "
            "used as the BDOO road target."
        )

    # ==============================================================================================
    # SAVE
    # ==============================================================================================

    csv_path = (
        OUT_ROOT /
        "bdoo_line_layer_inventory.csv"
    )

    json_path = (
        OUT_ROOT /
        "bdoo_layer_inventory.json"
    )

    best_path = (
        OUT_ROOT /
        "bdoo_best_road_candidates.json"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            valid,
            f,
            indent=2,
            ensure_ascii=False,
            default=str
        )

    with open(
        best_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            best_records,
            f,
            indent=2,
            ensure_ascii=False,
            default=str
        )

    banner("SCRIPT 57 COMPLETE")

    print()
    print("Line-layer inventory:")
    print(csv_path)

    print()
    print("Full inventory:")
    print(json_path)

    print()
    print("Best road candidates:")
    print(best_path)

    print()
    print(
        "NEXT: use the identified true BDOO "
        "road layer to repair Script 56 and "
        "then build road correspondence."
    )


if __name__ == "__main__":
    main()