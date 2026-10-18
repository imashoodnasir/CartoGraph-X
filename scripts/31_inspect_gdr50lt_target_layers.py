from pathlib import Path
import json
import time
import urllib.request
import urllib.error


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "lithuania"
    / "gdr50lt"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "gdr50lt_target_layers.json"
)

SERVICE_URL = (
    "https://www.geoportal.lt/"
    "arcgis/rest/services/"
    "NZT/GDR50LT/MapServer"
)

TARGETS = {
    "roads": {
        "id": 19,
        "expected_name": "Keliai",
    },
    "buildings": {
        "id": 21,
        "expected_name": "Pastatai",
    },
}

MAX_RETRIES = 10
RETRY_SECONDS = 8


def fetch_json(url):

    request_url = f"{url}?f=pjson"

    for attempt in range(1, MAX_RETRIES + 1):

        print(
            f"  Request attempt "
            f"{attempt}/{MAX_RETRIES}..."
        )

        request = urllib.request.Request(
            request_url,
            headers={
                "User-Agent":
                    "Mozilla/5.0 CartoGen Research",
                "Accept":
                    "application/json",
            }
        )

        try:

            with urllib.request.urlopen(
                request,
                timeout=120
            ) as response:

                raw = response.read()

            data = json.loads(
                raw.decode("utf-8")
            )

            if "error" in data:

                raise RuntimeError(
                    data["error"]
                )

            return data

        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            RuntimeError
        ) as exc:

            print(
                f"  Failed: {exc}"
            )

            if attempt == MAX_RETRIES:
                raise

            print(
                f"  Waiting "
                f"{RETRY_SECONDS} seconds..."
            )

            time.sleep(
                RETRY_SECONDS
            )


def print_fields(fields):

    if not fields:
        print("  Fields: none")
        return

    print("  Fields:")

    for field in fields:

        print(
            f"    "
            f"{field.get('name', ''):<30}"
            f"{field.get('type', ''):<30}"
            f"{field.get('alias', '')}"
        )


def main():

    print("=" * 120)
    print(
        "CARTOGEN — INSPECT GDR50LT "
        "TARGET BUILDING AND ROAD LAYERS"
    )
    print("=" * 120)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {}

    for role, target in TARGETS.items():

        print()
        print("=" * 120)
        print(
            f"{role.upper()} TARGET"
        )
        print("=" * 120)

        layer_id = target["id"]

        url = (
            f"{SERVICE_URL}/"
            f"{layer_id}"
        )

        print(
            f"Layer ID: {layer_id}"
        )

        print(
            f"Expected: "
            f"{target['expected_name']}"
        )

        print(
            f"URL: {url}"
        )

        try:

            info = fetch_json(url)

        except Exception as exc:

            print()
            print(
                "[FAILED] Could not retrieve "
                "layer after all retries."
            )

            print(exc)

            results[role] = {
                "id": layer_id,
                "status": "failed",
                "error": str(exc),
            }

            continue

        name = info.get("name")

        layer_type = info.get("type")

        geometry_type = info.get(
            "geometryType"
        )

        fields = (
            info.get("fields")
            or []
        )

        capabilities = info.get(
            "capabilities"
        )

        max_record_count = info.get(
            "maxRecordCount"
        )

        object_id_field = (
            info.get("objectIdField")
            or info.get(
                "objectIdFieldName"
            )
        )

        spatial_reference = (
            info.get(
                "extent",
                {}
            ).get(
                "spatialReference",
                {}
            )
        )

        wkid = (
            spatial_reference.get(
                "latestWkid"
            )
            or spatial_reference.get(
                "wkid"
            )
        )

        print()
        print(
            f"Name:             {name}"
        )

        print(
            f"Type:             {layer_type}"
        )

        print(
            f"Geometry:         {geometry_type}"
        )

        print(
            f"CRS WKID:         {wkid}"
        )

        print(
            f"Object ID field:  {object_id_field}"
        )

        print(
            f"Max record count: {max_record_count}"
        )

        print(
            f"Capabilities:     {capabilities}"
        )

        print()

        print_fields(fields)

        supports_query = (
            capabilities is not None
            and
            "query" in capabilities.lower()
        )

        print()
        print(
            "Query supported:  "
            f"{supports_query}"
        )

        expected_geometry = (
            "esriGeometryPolygon"
            if role == "buildings"
            else
            "esriGeometryPolyline"
        )

        geometry_ok = (
            geometry_type
            == expected_geometry
        )

        print(
            "Expected geometry: "
            f"{expected_geometry}"
        )

        print(
            "Geometry check:    "
            f"{'PASS' if geometry_ok else 'FAIL'}"
        )

        results[role] = {
            "status":
                "success",

            "id":
                layer_id,

            "name":
                name,

            "url":
                url,

            "type":
                layer_type,

            "geometry_type":
                geometry_type,

            "expected_geometry":
                expected_geometry,

            "geometry_ok":
                geometry_ok,

            "wkid":
                wkid,

            "object_id_field":
                object_id_field,

            "max_record_count":
                max_record_count,

            "capabilities":
                capabilities,

            "supports_query":
                supports_query,

            "fields": [
                {
                    "name":
                        field.get("name"),

                    "alias":
                        field.get("alias"),

                    "type":
                        field.get("type"),
                }
                for field in fields
            ],
        }

    print()
    print("=" * 120)
    print("VALIDATION")
    print("=" * 120)

    road = results.get(
        "roads",
        {}
    )

    building = results.get(
        "buildings",
        {}
    )

    roads_ok = (
        road.get("status")
        == "success"
        and
        road.get("geometry_ok")
    )

    buildings_ok = (
        building.get("status")
        == "success"
        and
        building.get("geometry_ok")
    )

    if roads_ok:

        print(
            "[PASS] GDR50LT Keliai "
            "is usable as road target."
        )

    else:

        print(
            "[FAIL] GDR50LT Keliai "
            "road target not validated."
        )

    if buildings_ok:

        print(
            "[PASS] GDR50LT Pastatai "
            "is usable as building target."
        )

    else:

        print(
            "[FAIL] GDR50LT Pastatai "
            "building target not validated."
        )

    if roads_ok and buildings_ok:

        print()
        print(
            "[PASS] Lithuanian CartoGen "
            "source-target pairing is viable."
        )

        print()
        print(
            "GRPK PASTAT  -> "
            "GDR50LT Pastatai"
        )

        print(
            "GRPK KELIAI  -> "
            "GDR50LT Keliai"
        )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 120)
    print("SCRIPT 31 COMPLETE")
    print("=" * 120)

    print(
        f"Saved:\n{OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()