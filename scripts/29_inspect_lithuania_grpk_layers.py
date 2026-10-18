from pathlib import Path
import json

import geopandas as gpd
import pandas as pd
import pyogrio


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
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "lithuania"
    / "grpk"
)

OUTPUT_JSON = (
    OUTPUT_ROOT
    / "grpk_layer_inventory.json"
)

OUTPUT_CSV = (
    OUTPUT_ROOT
    / "grpk_layer_inventory.csv"
)

SAMPLE_JSON = (
    OUTPUT_ROOT
    / "grpk_layer_samples.json"
)

EXPECTED_EPSG = 3346

SAMPLE_ROWS = 5


# ============================================================
# HELPERS
# ============================================================

def json_safe(value):

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        )
    ):
        return value

    return str(value)


def find_shapefiles():

    shapefiles = sorted(
        GRPK_ROOT.rglob("*.shp")
    )

    if not shapefiles:

        raise RuntimeError(
            "No shapefiles found under:\n"
            f"{GRPK_ROOT}"
        )

    return shapefiles


def get_feature_count(path):

    info = pyogrio.read_info(
        path
    )

    return int(
        info.get(
            "features",
            0
        )
    )


def inspect_layer(path):

    info = pyogrio.read_info(
        path
    )

    feature_count = int(
        info.get(
            "features",
            0
        )
    )

    geometry_type = (
        info.get(
            "geometry_type"
        )
    )

    crs = info.get(
        "crs"
    )

    encoding = info.get(
        "encoding"
    )

    fields = list(
        info.get(
            "fields",
            []
        )
    )

    dtypes = [
        str(x)
        for x in info.get(
            "dtypes",
            []
        )
    ]

    bounds = info.get(
        "total_bounds"
    )

    if bounds is not None:

        bounds = [
            float(x)
            for x in bounds
        ]

    sample = gpd.read_file(
        path,
        rows=SAMPLE_ROWS,
        engine="pyogrio"
    )

    epsg = (
        sample.crs.to_epsg()
        if sample.crs is not None
        else None
    )

    sample_records = []

    attribute_columns = [
        column
        for column in sample.columns
        if column != sample.geometry.name
    ]

    for _, row in sample.iterrows():

        record = {}

        for column in attribute_columns:

            record[column] = json_safe(
                row[column]
            )

        geometry = row.geometry

        record[
            "_geometry_type"
        ] = (
            geometry.geom_type
            if geometry is not None
            else None
        )

        sample_records.append(
            record
        )

    return {
        "file":
            str(path),

        "name":
            path.stem,

        "feature_count":
            feature_count,

        "geometry_type":
            geometry_type,

        "crs":
            str(crs),

        "epsg":
            epsg,

        "encoding":
            encoding,

        "field_count":
            len(fields),

        "fields":
            fields,

        "dtypes":
            dtypes,

        "bounds":
            bounds,

        "sample_records":
            sample_records,
    }


def candidate_score(
    name,
    geometry_type,
    fields
):

    name_upper = (
        name.upper()
    )

    fields_upper = " ".join(
        str(x).upper()
        for x in fields
    )

    building_score = 0
    road_score = 0

    # Lithuanian GRPK layer names

    if "PASTAT" in name_upper:
        building_score += 10

    if "KELIAI" in name_upper:
        road_score += 10

    # Generic semantic indicators

    building_terms = [
        "PASTAT",
        "BUILD",
        "BLDG",
    ]

    road_terms = [
        "KEL",
        "ROAD",
        "STREET",
        "GATV",
    ]

    for term in building_terms:

        if term in fields_upper:
            building_score += 1

    for term in road_terms:

        if term in fields_upper:
            road_score += 1

    # Geometry compatibility

    geometry_upper = (
        str(geometry_type).upper()
    )

    if "POLYGON" in geometry_upper:
        building_score += 2

    if "LINE" in geometry_upper:
        road_score += 2

    return (
        building_score,
        road_score
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 120)
    print(
        "CARTOGEN — INSPECT "
        "LITHUANIAN GRPK LAYERS"
    )
    print("=" * 120)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    shapefiles = find_shapefiles()

    print()
    print(
        f"Shapefiles found: "
        f"{len(shapefiles):,}"
    )

    records = []
    samples = {}

    for index, path in enumerate(
        shapefiles,
        start=1
    ):

        print()
        print("=" * 120)
        print(
            f"[{index:02d}/"
            f"{len(shapefiles):02d}] "
            f"{path.stem}"
        )
        print("=" * 120)

        result = inspect_layer(
            path
        )

        building_score, road_score = (
            candidate_score(
                result["name"],
                result["geometry_type"],
                result["fields"]
            )
        )

        result[
            "building_candidate_score"
        ] = building_score

        result[
            "road_candidate_score"
        ] = road_score

        print(
            f"Features:      "
            f"{result['feature_count']:,}"
        )

        print(
            f"Geometry:      "
            f"{result['geometry_type']}"
        )

        print(
            f"CRS:           "
            f"{result['crs']}"
        )

        print(
            f"EPSG:          "
            f"{result['epsg']}"
        )

        print(
            f"Encoding:      "
            f"{result['encoding']}"
        )

        print(
            f"Fields:        "
            f"{result['field_count']}"
        )

        print(
            f"Building score:"
            f" {building_score}"
        )

        print(
            f"Road score:    "
            f"{road_score}"
        )

        if result["bounds"]:

            xmin, ymin, xmax, ymax = (
                result["bounds"]
            )

            print(
                "Bounds:        "
                f"{xmin:,.2f}, "
                f"{ymin:,.2f}, "
                f"{xmax:,.2f}, "
                f"{ymax:,.2f}"
            )

        print()
        print("Columns:")

        for field, dtype in zip(
            result["fields"],
            result["dtypes"]
        ):

            print(
                f"  {field:<30} "
                f"{dtype}"
            )

        print()
        print("Sample attributes:")

        for sample_index, sample in enumerate(
            result[
                "sample_records"
            ],
            start=1
        ):

            print()
            print(
                f"  Sample "
                f"{sample_index}:"
            )

            for key, value in (
                sample.items()
            ):

                print(
                    f"    "
                    f"{key:<28} "
                    f"{value}"
                )

        records.append({
            "name":
                result["name"],

            "feature_count":
                result["feature_count"],

            "geometry_type":
                result["geometry_type"],

            "epsg":
                result["epsg"],

            "crs":
                result["crs"],

            "encoding":
                result["encoding"],

            "field_count":
                result["field_count"],

            "fields":
                " | ".join(
                    result["fields"]
                ),

            "building_candidate_score":
                building_score,

            "road_candidate_score":
                road_score,

            "file":
                result["file"],
        })

        samples[
            result["name"]
        ] = result[
            "sample_records"
        ]

    # ========================================================
    # SUMMARY
    # ========================================================

    df = pd.DataFrame(
        records
    )

    print()
    print("=" * 120)
    print("LAYER SUMMARY")
    print("=" * 120)

    display_columns = [
        "name",
        "feature_count",
        "geometry_type",
        "epsg",
        "building_candidate_score",
        "road_candidate_score",
    ]

    print(
        df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # CRS VALIDATION
    # ========================================================

    print()
    print("=" * 120)
    print("CRS VALIDATION")
    print("=" * 120)

    epsg_values = sorted(
        {
            int(x)
            for x in df[
                "epsg"
            ].dropna()
        }
    )

    print(
        f"EPSG values found: "
        f"{epsg_values}"
    )

    if (
        len(epsg_values) == 1
        and epsg_values[0]
        == EXPECTED_EPSG
    ):

        print(
            "[PASS] All readable GRPK "
            "layers use EPSG:3346."
        )

    else:

        print(
            "[WARNING] CRS requires "
            "manual inspection."
        )

    # ========================================================
    # CANDIDATE IDENTIFICATION
    # ========================================================

    print()
    print("=" * 120)
    print("CARTOGEN CANDIDATES")
    print("=" * 120)

    buildings = (
        df.sort_values(
            [
                "building_candidate_score",
                "feature_count",
            ],
            ascending=[
                False,
                False,
            ]
        )
        .head(5)
    )

    roads = (
        df.sort_values(
            [
                "road_candidate_score",
                "feature_count",
            ],
            ascending=[
                False,
                False,
            ]
        )
        .head(5)
    )

    print()
    print("BUILDING CANDIDATES")
    print("-" * 120)

    for _, row in (
        buildings.iterrows()
    ):

        print(
            f"{row['name']:<25}"
            f"score="
            f"{row['building_candidate_score']:<3}"
            f" | geometry="
            f"{row['geometry_type']:<20}"
            f" | features="
            f"{row['feature_count']:,}"
        )

    print()
    print("ROAD CANDIDATES")
    print("-" * 120)

    for _, row in (
        roads.iterrows()
    ):

        print(
            f"{row['name']:<25}"
            f"score="
            f"{row['road_candidate_score']:<3}"
            f" | geometry="
            f"{row['geometry_type']:<20}"
            f" | features="
            f"{row['feature_count']:,}"
        )

    # ========================================================
    # SAVE
    # ========================================================

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
            records,
            file,
            indent=2,
            ensure_ascii=False
        )

    with open(
        SAMPLE_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            samples,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 120)
    print("SCRIPT 29 COMPLETE")
    print("=" * 120)

    print(
        f"Inventory CSV:\n"
        f"{OUTPUT_CSV}"
    )

    print()
    print(
        f"Inventory JSON:\n"
        f"{OUTPUT_JSON}"
    )

    print()
    print(
        f"Samples JSON:\n"
        f"{SAMPLE_JSON}"
    )


if __name__ == "__main__":
    main()