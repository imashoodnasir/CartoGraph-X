from pathlib import Path
import json
import requests
import certifi
import xml.etree.ElementTree as ET
from datetime import datetime

import urllib3

urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "poland"
    / "source_validation"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "poland_cartogen_source_validation.json"
)


# ======================================================================
# OFFICIAL POLISH DATA SOURCES
# ======================================================================

SOURCES = {
    "BDOT10k": {
        "country": "Poland",
        "organization": "GUGiK",
        "full_name": "Baza Danych Obiektow Topograficznych",
        "english_name": "Topographic Objects Database",
        "role": "detailed_source",
        "nominal_scale": "1:10000",
        "expected_content": [
            "roads",
            "buildings",
            "transport_network",
            "land_cover",
            "hydrography",
        ],
        "wfs": (
            "https://mapy.geoportal.gov.pl/"
            "wss/service/PZGIK/BDOT/WFS/PobieranieBDOT10k"
        ),
        "download_wms": (
            "https://mapy.geoportal.gov.pl/"
            "wss/service/PZGIK/BDOT/WMS/PobieranieBDOT10k"
        ),
        "expected_formats": [
            "GML",
            "GPKG",
            "GeoParquet",
        ],
    },

    "BDOO": {
        "country": "Poland",
        "organization": "GUGiK",
        "full_name": "Baza Danych Obiektow Ogolnogeograficznych",
        "english_name": "Database of General Geographic Objects",
        "role": "generalized_target",
        "generation": (
            "automatic generalization of BDOT10k"
        ),
        "expected_content": [
            "roads",
            "buildings",
            "transport_network",
            "land_cover",
            "hydrography",
        ],
        "download_wms": (
            "https://mapy.geoportal.gov.pl/"
            "wss/service/PZGIK/BDOO/WMS/PobieranieBDOO"
        ),
        "display_wms": (
            "https://mapy.geoportal.gov.pl/"
            "wss/service/PZGIK/BDOO/WMS/2021"
        ),
        "expected_formats": [
            "GML",
        ],
    },
}


# ======================================================================
# NETWORK HELPERS
# ======================================================================

def request_bytes(url, timeout=60):

    headers = {
        "User-Agent":
            "Mozilla/5.0 CartoGen Research/1.0"
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=timeout,
            verify=certifi.where(),
        )

        response.raise_for_status()

        return {
            "status":
                response.status_code,

            "content_type":
                response.headers.get(
                    "Content-Type"
                ),

            "data":
                response.content,

            "ssl_verified":
                True,
        }

    except requests.exceptions.SSLError:

        print(
            "  Standard certificate verification failed."
        )
        print(
            "  Retrying this official GUGiK endpoint "
            "without certificate verification..."
        )

        response = requests.get(
            url,
            headers=headers,
            timeout=timeout,
            verify=False,
        )

        response.raise_for_status()

        return {
            "status":
                response.status_code,

            "content_type":
                response.headers.get(
                    "Content-Type"
                ),

            "data":
                response.content,

            "ssl_verified":
                False,
        }


def test_wms(base_url):

    separator = (
        "&"
        if "?" in base_url
        else "?"
    )

    url = (
        base_url
        + separator
        + "SERVICE=WMS"
        + "&REQUEST=GetCapabilities"
        + "&VERSION=1.3.0"
    )

    result = {
        "url": url,
        "reachable": False,
        "status": None,
        "content_type": None,
        "service_title": None,
        "layer_count": None,
        "error": None,
    }

    try:

        response = request_bytes(
            url
        )

        result["status"] = (
            response["status"]
        )

        result["content_type"] = (
            response["content_type"]
        )

        root = ET.fromstring(
            response["data"]
        )

        result["reachable"] = True

        # Namespace-independent search
        titles = []

        layers = []

        for element in root.iter():

            tag = (
                element.tag
                .split("}")[-1]
            )

            if tag == "Title":

                if element.text:
                    titles.append(
                        element.text.strip()
                    )

            elif tag == "Layer":

                layers.append(
                    element
                )

        if titles:
            result["service_title"] = (
                titles[0]
            )

        result["layer_count"] = (
            max(
                len(layers) - 1,
                0
            )
        )

    except Exception as exc:

        result["error"] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    return result


def test_wfs(base_url):

    separator = (
        "&"
        if "?" in base_url
        else "?"
    )

    url = (
        base_url
        + separator
        + "SERVICE=WFS"
        + "&REQUEST=GetCapabilities"
        + "&VERSION=2.0.0"
    )

    result = {
        "url": url,
        "reachable": False,
        "status": None,
        "content_type": None,
        "service_title": None,
        "feature_type_count": None,
        "feature_types": [],
        "error": None,
    }

    try:

        response = request_bytes(
            url
        )

        result["status"] = (
            response["status"]
        )

        result["content_type"] = (
            response["content_type"]
        )

        root = ET.fromstring(
            response["data"]
        )

        result["reachable"] = True

        feature_types = []

        for element in root.iter():

            tag = (
                element.tag
                .split("}")[-1]
            )

            if tag == "FeatureType":

                name = None
                title = None

                for child in element:

                    child_tag = (
                        child.tag
                        .split("}")[-1]
                    )

                    if (
                        child_tag == "Name"
                        and child.text
                    ):
                        name = (
                            child.text.strip()
                        )

                    elif (
                        child_tag == "Title"
                        and child.text
                    ):
                        title = (
                            child.text.strip()
                        )

                feature_types.append(
                    {
                        "name": name,
                        "title": title,
                    }
                )

        result[
            "feature_type_count"
        ] = len(feature_types)

        result[
            "feature_types"
        ] = feature_types

        # Try service title
        for element in root.iter():

            tag = (
                element.tag
                .split("}")[-1]
            )

            if (
                tag == "Title"
                and element.text
            ):

                result[
                    "service_title"
                ] = (
                    element.text.strip()
                )

                break

    except Exception as exc:

        result["error"] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    return result


# ======================================================================
# FEATURE-TYPE DISCOVERY
# ======================================================================

def score_feature_type(
    name,
    title
):

    text = (
        f"{name or ''} "
        f"{title or ''}"
    ).lower()

    building_terms = [
        "budyn",
        "building",
        "budow",
    ]

    road_terms = [
        "droga",
        "drog",
        "road",
        "jezd",
        "ulic",
        "komunik",
    ]

    building_score = sum(
        1
        for term
        in building_terms
        if term in text
    )

    road_score = sum(
        1
        for term
        in road_terms
        if term in text
    )

    return (
        building_score,
        road_score
    )


def identify_candidates(
    feature_types
):

    building_candidates = []
    road_candidates = []

    for item in feature_types:

        building_score, road_score = (
            score_feature_type(
                item.get("name"),
                item.get("title"),
            )
        )

        record = {
            **item,
            "building_score":
                building_score,
            "road_score":
                road_score,
        }

        if building_score > 0:

            building_candidates.append(
                record
            )

        if road_score > 0:

            road_candidates.append(
                record
            )

    building_candidates.sort(
        key=lambda x:
            x["building_score"],
        reverse=True,
    )

    road_candidates.sort(
        key=lambda x:
            x["road_score"],
        reverse=True,
    )

    return (
        building_candidates,
        road_candidates
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 120)
    print(
        "CARTOGEN — VALIDATE POLISH "
        "BDOT10k / BDOO DATA SOURCES"
    )
    print("=" * 120)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "generated":
            datetime.now().isoformat(),

        "country":
            "Poland",

        "sources":
            SOURCES,

        "tests":
            {},
    }


    # ==================================================================
    # BDOT10k
    # ==================================================================

    print()
    print("=" * 120)
    print("BDOT10k — DETAILED SOURCE")
    print("=" * 120)

    bdot = SOURCES["BDOT10k"]

    print(
        f"Full name:     "
        f"{bdot['full_name']}"
    )

    print(
        f"Role:          "
        f"{bdot['role']}"
    )

    print(
        f"Nominal scale: "
        f"{bdot['nominal_scale']}"
    )

    print()
    print(
        "Testing official "
        "BDOT10k WFS..."
    )

    bdot_wfs = test_wfs(
        bdot["wfs"]
    )

    print(
        f"Reachable:     "
        f"{bdot_wfs['reachable']}"
    )

    print(
        f"HTTP status:   "
        f"{bdot_wfs['status']}"
    )

    print(
        f"Feature types: "
        f"{bdot_wfs['feature_type_count']}"
    )

    if bdot_wfs["error"]:

        print(
            f"Error: "
            f"{bdot_wfs['error']}"
        )

    building_candidates = []
    road_candidates = []

    if bdot_wfs["reachable"]:

        (
            building_candidates,
            road_candidates
        ) = identify_candidates(
            bdot_wfs[
                "feature_types"
            ]
        )

        print()
        print(
            "Potential building "
            "feature types:"
        )

        for item in (
            building_candidates[:15]
        ):

            print(
                f"  {str(item['name']):<45} "
                f"| {item['title']}"
            )

        print()
        print(
            "Potential road "
            "feature types:"
        )

        for item in (
            road_candidates[:15]
        ):

            print(
                f"  {str(item['name']):<45} "
                f"| {item['title']}"
            )

    print()
    print(
        "Testing BDOT10k "
        "download WMS..."
    )

    bdot_download = test_wms(
        bdot["download_wms"]
    )

    print(
        f"Reachable:     "
        f"{bdot_download['reachable']}"
    )

    print(
        f"HTTP status:   "
        f"{bdot_download['status']}"
    )

    print(
        f"Layers:        "
        f"{bdot_download['layer_count']}"
    )

    if bdot_download["error"]:

        print(
            f"Error: "
            f"{bdot_download['error']}"
        )


    # ==================================================================
    # BDOO
    # ==================================================================

    print()
    print("=" * 120)
    print("BDOO — GENERALIZED TARGET")
    print("=" * 120)

    bdoo = SOURCES["BDOO"]

    print(
        f"Full name:     "
        f"{bdoo['full_name']}"
    )

    print(
        f"Role:          "
        f"{bdoo['role']}"
    )

    print(
        "Generation:    "
        "automatic generalization "
        "of BDOT10k"
    )

    print()
    print(
        "Testing BDOO "
        "download WMS..."
    )

    bdoo_download = test_wms(
        bdoo["download_wms"]
    )

    print(
        f"Reachable:     "
        f"{bdoo_download['reachable']}"
    )

    print(
        f"HTTP status:   "
        f"{bdoo_download['status']}"
    )

    print(
        f"Layers:        "
        f"{bdoo_download['layer_count']}"
    )

    if bdoo_download["error"]:

        print(
            f"Error: "
            f"{bdoo_download['error']}"
        )

    print()
    print(
        "Testing BDOO "
        "display WMS..."
    )

    bdoo_display = test_wms(
        bdoo["display_wms"]
    )

    print(
        f"Reachable:     "
        f"{bdoo_display['reachable']}"
    )

    print(
        f"HTTP status:   "
        f"{bdoo_display['status']}"
    )

    print(
        f"Layers:        "
        f"{bdoo_display['layer_count']}"
    )

    if bdoo_display["error"]:

        print(
            f"Error: "
            f"{bdoo_display['error']}"
        )


    # ==================================================================
    # CARTOGEN DECISION
    # ==================================================================

    print()
    print("=" * 120)
    print("CARTOGEN POLAND DECISION")
    print("=" * 120)

    source_access = (
        bdot_wfs["reachable"]
        or
        bdot_download["reachable"]
    )

    target_access = (
        bdoo_download["reachable"]
        or
        bdoo_display["reachable"]
    )

    pairing_viable = (
        source_access
        and
        target_access
    )

    checks = {
        "BDOT10k_accessible":
            bool(source_access),

        "BDOO_accessible":
            bool(target_access),

        "source_target_pairing_viable":
            bool(pairing_viable),

        "BDOT10k_building_candidates":
            bool(
                len(
                    building_candidates
                ) > 0
            ),

        "BDOT10k_road_candidates":
            bool(
                len(
                    road_candidates
                ) > 0
            ),
    }

    for key, value in (
        checks.items()
    ):

        status = (
            "PASS"
            if value
            else "CHECK"
        )

        print(
            f"{key:<45} "
            f"{status}"
        )

    print()

    if pairing_viable:

        print(
            "[PASS] Polish CartoGen "
            "source-target pairing is viable."
        )

        print()
        print(
            "BDOT10k detailed geometry"
            "  ->  BDOO generalized geometry"
        )

    else:

        print(
            "[CHECK] At least one official "
            "service could not be reached."
        )

        print(
            "This does not necessarily mean "
            "the dataset is unavailable."
        )


    # ==================================================================
    # SAVE REPORT
    # ==================================================================

    report["tests"] = {
        "BDOT10k_WFS":
            bdot_wfs,

        "BDOT10k_download_WMS":
            bdot_download,

        "BDOO_download_WMS":
            bdoo_download,

        "BDOO_display_WMS":
            bdoo_display,

        "building_candidates":
            building_candidates,

        "road_candidates":
            road_candidates,

        "checks":
            checks,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 120)
    print("SCRIPT 45 COMPLETE")
    print("=" * 120)

    print()
    print("Saved:")
    print(OUTPUT_JSON)

    print()

    if pairing_viable:

        print(
            "NEXT: Script 46 — "
            "discover and download the "
            "required Polish BDOT10k "
            "building and road data."
        )


if __name__ == "__main__":
    main()