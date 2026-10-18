from pathlib import Path
import json
import requests
import certifi
import urllib3
import xml.etree.ElementTree as ET
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "poland"
    / "bdot10k_discovery"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "bdot10k_download_service_inventory.json"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


BASE_WFS = (
    "https://mapy.geoportal.gov.pl/"
    "wss/service/PZGIK/BDOT/WFS/PobieranieBDOT10k"
)

BASE_WMS = (
    "https://mapy.geoportal.gov.pl/"
    "wss/service/PZGIK/BDOT/WMS/PobieranieBDOT10k"
)


urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)


# ======================================================================
# HTTP
# ======================================================================

def request(
    url,
    params=None,
    timeout=120
):

    headers = {
        "User-Agent":
            "Mozilla/5.0 CartoGen Research/1.0"
    }

    try:

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
            verify=certifi.where(),
        )

    except requests.exceptions.SSLError:

        print(
            "  SSL verification failed; "
            "retrying official GUGiK service..."
        )

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
            verify=False,
        )

    response.raise_for_status()

    return response


# ======================================================================
# XML HELPERS
# ======================================================================

def local_name(tag):

    return tag.split("}")[-1]


def child_text(
    element,
    name
):

    for child in element:

        if local_name(
            child.tag
        ) == name:

            if child.text:
                return (
                    child.text.strip()
                )

    return None


# ======================================================================
# WFS CAPABILITIES
# ======================================================================

def inspect_wfs():

    print()
    print("=" * 120)
    print("1. BDOT10k WFS CAPABILITIES")
    print("=" * 120)

    params = {
        "SERVICE": "WFS",
        "REQUEST": "GetCapabilities",
        "VERSION": "2.0.0",
    }

    response = request(
        BASE_WFS,
        params=params
    )

    print(
        f"HTTP status: "
        f"{response.status_code}"
    )

    root = ET.fromstring(
        response.content
    )

    feature_types = []

    for element in root.iter():

        if local_name(
            element.tag
        ) != "FeatureType":
            continue

        record = {
            "name":
                child_text(
                    element,
                    "Name"
                ),

            "title":
                child_text(
                    element,
                    "Title"
                ),

            "abstract":
                child_text(
                    element,
                    "Abstract"
                ),

            "default_crs":
                child_text(
                    element,
                    "DefaultCRS"
                ),
        }

        feature_types.append(
            record
        )

    print()
    print(
        f"Feature types: "
        f"{len(feature_types)}"
    )

    for i, item in enumerate(
        feature_types,
        start=1
    ):

        print()
        print("-" * 120)

        print(
            f"Feature type {i}"
        )

        for key, value in (
            item.items()
        ):

            print(
                f"{key:<15}: "
                f"{value}"
            )

    return feature_types


# ======================================================================
# DESCRIBE FEATURE TYPE
# ======================================================================

def describe_feature_type(
    feature_name
):

    print()
    print("=" * 120)
    print("2. DESCRIBE BDOT10k DOWNLOAD FEATURE")
    print("=" * 120)

    params = {
        "SERVICE": "WFS",
        "REQUEST":
            "DescribeFeatureType",
        "VERSION": "2.0.0",
        "TYPENAMES":
            feature_name,
    }

    response = request(
        BASE_WFS,
        params=params
    )

    print(
        f"HTTP status: "
        f"{response.status_code}"
    )

    root = ET.fromstring(
        response.content
    )

    fields = []

    for element in root.iter():

        if local_name(
            element.tag
        ) != "element":
            continue

        name = element.attrib.get(
            "name"
        )

        field_type = (
            element.attrib.get(
                "type"
            )
        )

        if not name:
            continue

        fields.append(
            {
                "name":
                    name,

                "type":
                    field_type,

                "minOccurs":
                    element.attrib.get(
                        "minOccurs"
                    ),

                "maxOccurs":
                    element.attrib.get(
                        "maxOccurs"
                    ),
            }
        )

    print()
    print("FIELDS")
    print("-" * 120)

    for field in fields:

        print(
            f"{str(field['name']):<40} "
            f"{field['type']}"
        )

    return fields


# ======================================================================
# SAMPLE FEATURES
# ======================================================================

def get_sample_features(
    feature_name
):

    print()
    print("=" * 120)
    print("3. SAMPLE BDOT10k DOWNLOAD FEATURES")
    print("=" * 120)

    params = {
        "SERVICE": "WFS",
        "REQUEST": "GetFeature",
        "VERSION": "2.0.0",
        "TYPENAMES":
            feature_name,
        "COUNT": 10,
    }

    response = request(
        BASE_WFS,
        params=params
    )

    print(
        f"HTTP status: "
        f"{response.status_code}"
    )

    print(
        f"Response size: "
        f"{len(response.content):,} bytes"
    )

    root = ET.fromstring(
        response.content
    )

    members = []

    for element in root.iter():

        if local_name(
            element.tag
        ) not in (
            "member",
            "featureMember"
        ):
            continue

        children = list(
            element
        )

        if not children:
            continue

        feature = children[0]

        record = {}

        for child in feature:

            key = local_name(
                child.tag
            )

            text = (
                child.text.strip()
                if child.text
                else None
            )

            if (
                text is None
                and list(child)
            ):

                # Geometry or nested object.
                record[key] = (
                    "<complex>"
                )

            else:

                record[key] = text

        members.append(
            record
        )

    print()
    print(
        f"Parsed sample features: "
        f"{len(members)}"
    )

    for i, record in enumerate(
        members,
        start=1
    ):

        print()
        print("-" * 120)
        print(f"SAMPLE {i}")
        print("-" * 120)

        for key, value in (
            record.items()
        ):

            value_text = str(
                value
            )

            if len(
                value_text
            ) > 250:

                value_text = (
                    value_text[:247]
                    + "..."
                )

            print(
                f"{key:<35}: "
                f"{value_text}"
            )

    return members


# ======================================================================
# WMS CAPABILITIES
# ======================================================================

def inspect_wms():

    print()
    print("=" * 120)
    print("4. BDOT10k DOWNLOAD WMS")
    print("=" * 120)

    params = {
        "SERVICE": "WMS",
        "REQUEST":
            "GetCapabilities",
        "VERSION": "1.3.0",
    }

    response = request(
        BASE_WMS,
        params=params
    )

    print(
        f"HTTP status: "
        f"{response.status_code}"
    )

    root = ET.fromstring(
        response.content
    )

    layers = []

    for element in root.iter():

        if local_name(
            element.tag
        ) != "Layer":
            continue

        name = child_text(
            element,
            "Name"
        )

        title = child_text(
            element,
            "Title"
        )

        if (
            name is None
            and title is None
        ):
            continue

        layers.append(
            {
                "name":
                    name,

                "title":
                    title,
            }
        )

    print()
    print(
        f"Layers found: "
        f"{len(layers)}"
    )

    for i, layer in enumerate(
        layers,
        start=1
    ):

        print(
            f"{i:02d}. "
            f"{str(layer['name']):<40} "
            f"| {layer['title']}"
        )

    return layers


# ======================================================================
# DISCOVER URL-LIKE VALUES
# ======================================================================

def discover_download_fields(
    samples
):

    print()
    print("=" * 120)
    print("5. DOWNLOAD-LINK DISCOVERY")
    print("=" * 120)

    candidates = []

    keywords = (
        "http",
        ".zip",
        ".gml",
        ".gpkg",
        ".parquet",
    )

    for sample_index, record in enumerate(
        samples
    ):

        for key, value in (
            record.items()
        ):

            if value is None:
                continue

            text = str(
                value
            )

            lower = (
                text.lower()
            )

            if any(
                keyword in lower
                for keyword
                in keywords
            ):

                candidates.append(
                    {
                        "sample":
                            sample_index + 1,

                        "field":
                            key,

                        "value":
                            text,
                    }
                )

    if not candidates:

        print(
            "No direct download URLs "
            "detected in sample attributes."
        )

    else:

        print(
            f"Potential download values: "
            f"{len(candidates)}"
        )

        for item in candidates:

            print()
            print(
                f"Sample: {item['sample']}"
            )

            print(
                f"Field:  {item['field']}"
            )

            print(
                f"Value:  {item['value']}"
            )

    return candidates


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 120)
    print(
        "CARTOGEN — DISCOVER POLISH "
        "BDOT10k DOWNLOAD STRUCTURE"
    )
    print("=" * 120)

    feature_types = (
        inspect_wfs()
    )

    if not feature_types:

        raise RuntimeError(
            "No BDOT10k WFS "
            "feature types found."
        )

    feature_name = (
        feature_types[0][
            "name"
        ]
    )

    if not feature_name:

        raise RuntimeError(
            "BDOT10k feature type "
            "has no name."
        )

    print()
    print(
        "Using feature type:"
    )
    print(feature_name)

    fields = (
        describe_feature_type(
            feature_name
        )
    )

    samples = (
        get_sample_features(
            feature_name
        )
    )

    wms_layers = (
        inspect_wms()
    )

    download_candidates = (
        discover_download_fields(
            samples
        )
    )


    # ==================================================================
    # REPORT
    # ==================================================================

    report = {
        "generated":
            datetime.now().isoformat(),

        "dataset":
            "BDOT10k",

        "country":
            "Poland",

        "role":
            "CartoGen detailed source",

        "feature_types":
            feature_types,

        "selected_feature_type":
            feature_name,

        "fields":
            fields,

        "sample_features":
            samples,

        "wms_layers":
            wms_layers,

        "download_candidates":
            download_candidates,
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
    print("SCRIPT 46 COMPLETE")
    print("=" * 120)

    print()
    print("Saved:")
    print(OUTPUT_JSON)

    print()
    print(
        "Do not manually download "
        "anything yet."
    )

    print(
        "The output of this script "
        "will determine the safest "
        "download strategy for "
        "Scripts 47+."
    )


if __name__ == "__main__":
    main()