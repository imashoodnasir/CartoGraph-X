from pathlib import Path
import json
import urllib.request


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "lithuania"
    / "gdr50lt"
)

OUTPUT_JSON = (
    OUTPUT_ROOT
    / "gdr50lt_service_inventory.json"
)

SERVICE_URL = (
    "https://www.geoportal.lt/"
    "arcgis/rest/services/"
    "NZT/GDR50LT/MapServer"
)

EXPECTED_EPSG = 3346


# ============================================================
# HELPERS
# ============================================================

def fetch_json(url):

    separator = (
        "&"
        if "?" in url
        else "?"
    )

    request_url = (
        f"{url}"
        f"{separator}"
        f"f=pjson"
    )

    request = urllib.request.Request(
        request_url,
        headers={
            "User-Agent":
                "Mozilla/5.0 "
                "CartoGen Research"
        }
    )

    with urllib.request.urlopen(
        request,
        timeout=120
    ) as response:

        data = response.read()

    return json.loads(
        data.decode(
            "utf-8"
        )
    )


def geometry_label(
    geometry_type
):

    mapping = {
        "esriGeometryPoint":
            "Point",

        "esriGeometryMultipoint":
            "MultiPoint",

        "esriGeometryPolyline":
            "LineString",

        "esriGeometryPolygon":
            "Polygon",
    }

    return mapping.get(
        geometry_type,
        geometry_type
    )


def candidate_scores(
    name,
    geometry_type
):

    text = (
        name
        .upper()
        .replace("Ą", "A")
        .replace("Č", "C")
        .replace("Ę", "E")
        .replace("Ė", "E")
        .replace("Į", "I")
        .replace("Š", "S")
        .replace("Ų", "U")
        .replace("Ū", "U")
        .replace("Ž", "Z")
    )

    building_score = 0
    road_score = 0

    building_terms = [
        "PASTAT",
        "STATIN",
        "BUILD",
    ]

    road_terms = [
        "KELIAI",
        "KELIAS",
        "GATV",
        "ROAD",
        "STREET",
    ]

    for term in building_terms:

        if term in text:
            building_score += 10

    for term in road_terms:

        if term in text:
            road_score += 10

    # Geometry should support a semantic match,
    # not create a candidate by itself.

    if (
        building_score > 0
        and geometry_type == "esriGeometryPolygon"
    ):
        building_score += 2

    if (
        road_score > 0
        and geometry_type == "esriGeometryPolyline"
    ):
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
        "LITHUANIAN GDR50LT SERVICE"
    )
    print("=" * 120)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("Official service:")
    print(SERVICE_URL)

    print()
    print(
        "Reading service metadata..."
    )

    service = fetch_json(
        SERVICE_URL
    )

    print()
    print("=" * 120)
    print("SERVICE INFORMATION")
    print("=" * 120)

    print(
        f"Name: "
        f"{service.get('mapName')}"
    )

    print(
        f"Description: "
        f"{service.get('description')}"
    )

    spatial_reference = (
        service
        .get(
            "spatialReference",
            {}
        )
    )

    wkid = (
        spatial_reference.get(
            "latestWkid"
        )
        or
        spatial_reference.get(
            "wkid"
        )
    )

    print(
        f"Service CRS WKID: "
        f"{wkid}"
    )

    if wkid in (
        EXPECTED_EPSG,
        2600
    ):

        print(
            "[PASS] Service uses "
            "Lithuanian LKS-94 / "
            "EPSG:3346."
        )

    else:

        print(
            "[WARNING] Unexpected "
            "service CRS."
        )

    layers = (
        service.get(
            "layers",
            []
        )
    )

    tables = (
        service.get(
            "tables",
            []
        )
    )

    print()
    print(
        f"Top-level layers: "
        f"{len(layers)}"
    )

    print(
        f"Tables: "
        f"{len(tables)}"
    )

    # --------------------------------------------------------
    # Inspect all leaf layers
    # --------------------------------------------------------

    records = []

    print()
    print("=" * 120)
    print("LAYER INVENTORY")
    print("=" * 120)

    for item in layers:

        layer_id = item[
            "id"
        ]

        layer_name = item[
            "name"
        ]

        layer_url = (
            f"{SERVICE_URL}/"
            f"{layer_id}"
        )

        print()
        print("-" * 120)

        print(
            f"Layer {layer_id}: "
            f"{layer_name}"
        )

        try:

            info = fetch_json(
                layer_url
            )

        except Exception as exc:

            print(
                f"  Could not inspect: "
                f"{exc}"
            )

            continue

        geometry_type = (
            info.get(
                "geometryType"
            )
        )

        sublayers = (
            info.get("subLayers")
            or []
        )
        layer_type = (
            info.get(
                "type"
            )
        )

        print(
            f"  Type: "
            f"{layer_type}"
        )

        if sublayers:

            print(
                f"  Group layer with "
                f"{len(sublayers)} "
                f"sublayers"
            )

        if geometry_type:

            print(
                f"  Geometry: "
                f"{geometry_label(geometry_type)}"
            )

        fields = (
            info.get("fields")
            or []
        )

        field_names = [
            field.get("name")
            for field in fields
            if isinstance(field, dict)
        ]

        if field_names:

            print(
                "  Fields:"
            )

            for field in fields:

                print(
                    f"    "
                    f"{field.get('name', ''):<30}"
                    f"{field.get('type', '')}"
                )

        (
            building_score,
            road_score
        ) = candidate_scores(
            layer_name,
            geometry_type
        )

        record = {
            "id":
                layer_id,

            "name":
                layer_name,

            "type":
                layer_type,

            "geometry_type":
                geometry_type,

            "geometry_label":
                geometry_label(
                    geometry_type
                ),

            "building_candidate_score":
                building_score,

            "road_candidate_score":
                road_score,

            "fields":
                field_names,

            "sub_layers":
                sublayers,

            "url":
                layer_url,
        }

        records.append(
            record
        )

        print(
            f"  Building score: "
            f"{building_score}"
        )

        print(
            f"  Road score: "
            f"{road_score}"
        )

    # --------------------------------------------------------
    # Candidate summary
    # --------------------------------------------------------

    print()
    print("=" * 120)
    print("CARTOGEN TARGET CANDIDATES")
    print("=" * 120)

    building_candidates = sorted(
        records,
        key=lambda x: (
            x[
                "building_candidate_score"
            ],
            x["id"]
        ),
        reverse=True
    )

    road_candidates = sorted(
        records,
        key=lambda x: (
            x[
                "road_candidate_score"
            ],
            x["id"]
        ),
        reverse=True
    )

    print()
    print("BUILDING CANDIDATES")
    print("-" * 120)

    found_building = False

    for record in (
        building_candidates
    ):

        if (
            record[
                "building_candidate_score"
            ]
            <= 0
        ):
            continue

        found_building = True

        print(
            f"ID={record['id']:<4} "
            f"{record['name']:<40} "
            f"| geometry="
            f"{record['geometry_label']:<15} "
            f"| score="
            f"{record['building_candidate_score']}"
        )

    if not found_building:

        print(
            "No building candidate "
            "identified automatically."
        )

    print()
    print("ROAD CANDIDATES")
    print("-" * 120)

    found_road = False

    for record in (
        road_candidates
    ):

        if (
            record[
                "road_candidate_score"
            ]
            <= 0
        ):
            continue

        found_road = True

        print(
            f"ID={record['id']:<4} "
            f"{record['name']:<40} "
            f"| geometry="
            f"{record['geometry_label']:<15} "
            f"| score="
            f"{record['road_candidate_score']}"
        )

    if not found_road:

        print(
            "No road candidate "
            "identified automatically."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {
        "service_url":
            SERVICE_URL,

        "service_name":
            service.get(
                "mapName"
            ),

        "wkid":
            wkid,

        "expected_epsg":
            EXPECTED_EPSG,

        "layers":
            records,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 120)
    print("SCRIPT 30 COMPLETE")
    print("=" * 120)

    print(
        f"Inventory:\n"
        f"{OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()