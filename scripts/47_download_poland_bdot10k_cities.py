from pathlib import Path
import json
import time
import zipfile
import hashlib
import requests
import certifi
import urllib3
import xml.etree.ElementTree as ET
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "poland"
    / "bdot10k"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MANIFEST_JSON = (
    OUTPUT_DIR
    / "bdot10k_city_download_manifest.json"
)


BASE_WFS = (
    "https://mapy.geoportal.gov.pl/"
    "wss/service/PZGIK/BDOT/WFS/PobieranieBDOT10k"
)

FEATURE_TYPE = "ms:BDOT10k_powiaty"


# ----------------------------------------------------------------------
# TARGET CITIES
# ----------------------------------------------------------------------

TARGETS = {
    "warsaw": {
        "display_name": "Warsaw",
        "search_terms": [
            "warszawa",
            "m.st. warszawa",
            "miasto stołeczne warszawa",
        ],
    },

    "krakow": {
        "display_name": "Krakow",
        "search_terms": [
            "kraków",
            "krakow",
        ],
    },

    "gdansk": {
        "display_name": "Gdansk",
        "search_terms": [
            "gdańsk",
            "gdansk",
        ],
    },
}


urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)


# ======================================================================
# HELPERS
# ======================================================================

def local_name(tag):
    return tag.split("}")[-1]


def http_get(
    url,
    params=None,
    timeout=180,
    stream=False
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
            stream=stream,
            verify=certifi.where(),
        )

    except requests.exceptions.SSLError:

        print(
            "  SSL verification failed; "
            "retrying official GUGiK endpoint..."
        )

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
            stream=stream,
            verify=False,
        )

    response.raise_for_status()

    return response


def normalize_text(value):

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
    )


# ======================================================================
# GET COMPLETE POWIAT INDEX
# ======================================================================

def get_all_powiat_records():

    print()
    print("=" * 120)
    print("1. QUERY COMPLETE BDOT10k POWIAT INDEX")
    print("=" * 120)

    params = {
        "SERVICE": "WFS",
        "REQUEST": "GetFeature",
        "VERSION": "2.0.0",
        "TYPENAMES": FEATURE_TYPE,
    }

    response = http_get(
        BASE_WFS,
        params=params,
        timeout=300,
    )

    print(
        f"HTTP status:   "
        f"{response.status_code}"
    )

    print(
        f"Response size: "
        f"{len(response.content) / 1024 / 1024:.2f} MB"
    )

    root = ET.fromstring(
        response.content
    )

    records = []

    for element in root.iter():

        if local_name(
            element.tag
        ) not in (
            "member",
            "featureMember"
        ):
            continue

        children = list(element)

        if not children:
            continue

        feature = children[0]

        record = {}

        for child in feature:

            key = local_name(
                child.tag
            )

            if key in (
                "TERYT",
                "NAZWA_POWIATU",
                "Data_aktualizacji",
                "URL_GML",
            ):

                record[key] = (
                    child.text.strip()
                    if child.text
                    else None
                )

        if record.get(
            "NAZWA_POWIATU"
        ):

            records.append(
                record
            )

    print(
        f"Powiat records: "
        f"{len(records):,}"
    )

    if not records:
        raise RuntimeError(
            "No powiat records were parsed."
        )

    return records


# ======================================================================
# MATCH TARGET CITIES
# ======================================================================

def find_city_records(records):

    print()
    print("=" * 120)
    print("2. IDENTIFY TARGET CITY COUNTIES")
    print("=" * 120)

    selected = {}

    for city_key, config in (
        TARGETS.items()
    ):

        matches = []

        for record in records:

            name = normalize_text(
                record.get(
                    "NAZWA_POWIATU"
                )
            )

            for term in (
                config["search_terms"]
            ):

                term_normalized = (
                    normalize_text(term)
                )

                if (
                    name == term_normalized
                    or
                    term_normalized in name
                ):

                    matches.append(
                        record
                    )

                    break

        # Remove duplicate records
        unique = {}

        for record in matches:

            key = (
                record.get("TERYT"),
                record.get("URL_GML"),
            )

            unique[key] = record

        matches = list(
            unique.values()
        )

        print()
        print("-" * 120)

        print(
            f"{config['display_name']}"
        )

        if not matches:

            print(
                "  [ERROR] No matching "
                "powiat found."
            )

            continue

        for record in matches:

            print(
                f"  TERYT:   "
                f"{record.get('TERYT')}"
            )

            print(
                f"  Name:    "
                f"{record.get('NAZWA_POWIATU')}"
            )

            print(
                f"  Updated: "
                f"{record.get('Data_aktualizacji')}"
            )

            print(
                f"  URL:     "
                f"{record.get('URL_GML')}"
            )

            print()

        # Prefer an exact city-name match.
        exact = []

        for record in matches:

            name = normalize_text(
                record.get(
                    "NAZWA_POWIATU"
                )
            )

            if any(
                name == normalize_text(term)
                or
                name == normalize_text(
                    "powiat " + term
                )
                for term
                in config["search_terms"]
            ):

                exact.append(
                    record
                )

        candidates = (
            exact
            if exact
            else matches
        )

        if len(candidates) != 1:

            print(
                "  [WARNING] Multiple "
                "candidate records."
            )

            print(
                "  Selecting the first "
                "candidate; verify output."
            )

        selected[
            city_key
        ] = candidates[0]

    missing = (
        set(TARGETS)
        - set(selected)
    )

    if missing:

        raise RuntimeError(
            "Could not identify: "
            + ", ".join(
                sorted(missing)
            )
        )

    return selected


# ======================================================================
# DOWNLOAD
# ======================================================================

def download_file(
    url,
    destination
):

    if destination.exists():

        size_mb = (
            destination.stat().st_size
            / 1024
            / 1024
        )

        print(
            f"Already exists: "
            f"{destination.name} "
            f"({size_mb:.2f} MB)"
        )

        return

    response = http_get(
        url,
        timeout=600,
        stream=True,
    )

    total = int(
        response.headers.get(
            "Content-Length",
            0
        )
    )

    downloaded = 0
    last_print = time.time()

    with open(
        destination,
        "wb"
    ) as f:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if not chunk:
                continue

            f.write(chunk)

            downloaded += len(
                chunk
            )

            now = time.time()

            if (
                now - last_print >= 1
            ):

                if total > 0:

                    pct = (
                        downloaded
                        / total
                        * 100
                    )

                    print(
                        f"\r  Downloaded: "
                        f"{downloaded / 1024 / 1024:,.1f} MB "
                        f"({pct:.1f}%)",
                        end="",
                        flush=True,
                    )

                else:

                    print(
                        f"\r  Downloaded: "
                        f"{downloaded / 1024 / 1024:,.1f} MB",
                        end="",
                        flush=True,
                    )

                last_print = now

    print()

    print(
        f"  Final size: "
        f"{destination.stat().st_size / 1024 / 1024:.2f} MB"
    )


# ======================================================================
# ZIP VALIDATION
# ======================================================================

def validate_zip(path):

    result = {
        "valid_zip": False,
        "file_count": 0,
        "gml_count": 0,
        "gml_files": [],
    }

    if not zipfile.is_zipfile(
        path
    ):

        return result

    result[
        "valid_zip"
    ] = True

    with zipfile.ZipFile(
        path,
        "r"
    ) as archive:

        names = (
            archive.namelist()
        )

        result[
            "file_count"
        ] = len(names)

        gml_files = [
            name
            for name in names
            if name.lower().endswith(
                ".gml"
            )
        ]

        result[
            "gml_count"
        ] = len(
            gml_files
        )

        result[
            "gml_files"
        ] = gml_files

    return result


def sha256_file(path):

    digest = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                8 * 1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 120)
    print(
        "CARTOGEN — DOWNLOAD POLISH "
        "BDOT10k CITY DATA"
    )
    print("=" * 120)

    records = (
        get_all_powiat_records()
    )

    selected = (
        find_city_records(
            records
        )
    )

    print()
    print("=" * 120)
    print("3. DOWNLOAD SELECTED BDOT10k PACKAGES")
    print("=" * 120)

    manifest = {
        "generated":
            datetime.now().isoformat(),

        "dataset":
            "BDOT10k",

        "role":
            "detailed_source",

        "crs":
            "EPSG:2180",

        "cities":
            {},
    }

    for city_key, record in (
        selected.items()
    ):

        config = (
            TARGETS[
                city_key
            ]
        )

        print()
        print("=" * 120)
        print(
            config[
                "display_name"
            ].upper()
        )
        print("=" * 120)

        url = (
            record.get(
                "URL_GML"
            )
        )

        if not url:

            raise RuntimeError(
                f"No URL_GML for "
                f"{city_key}"
            )

        city_dir = (
            OUTPUT_DIR
            / city_key
        )

        city_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        filename = (
            url.split("/")[-1]
        )

        destination = (
            city_dir
            / filename
        )

        print(
            f"TERYT:   "
            f"{record.get('TERYT')}"
        )

        print(
            f"County:  "
            f"{record.get('NAZWA_POWIATU')}"
        )

        print(
            f"Updated: "
            f"{record.get('Data_aktualizacji')}"
        )

        print(
            f"URL:     {url}"
        )

        print()
        print(
            f"Downloading to:"
        )
        print(destination)

        download_file(
            url,
            destination
        )

        print()
        print(
            "Validating ZIP..."
        )

        validation = (
            validate_zip(
                destination
            )
        )

        if not validation[
            "valid_zip"
        ]:

            raise RuntimeError(
                f"Invalid ZIP: "
                f"{destination}"
            )

        print(
            "[PASS] Valid ZIP archive"
        )

        print(
            f"Files: "
            f"{validation['file_count']}"
        )

        print(
            f"GML files: "
            f"{validation['gml_count']}"
        )

        print()
        print(
            "GML CONTENTS"
        )
        print("-" * 120)

        for name in (
            validation[
                "gml_files"
            ]
        ):

            print(
                f"  {name}"
            )

        print()
        print(
            "Calculating SHA-256..."
        )

        sha256 = (
            sha256_file(
                destination
            )
        )

        print(
            f"SHA-256: {sha256}"
        )

        manifest[
            "cities"
        ][city_key] = {
            "display_name":
                config[
                    "display_name"
                ],

            "TERYT":
                record.get(
                    "TERYT"
                ),

            "county_name":
                record.get(
                    "NAZWA_POWIATU"
                ),

            "data_updated":
                record.get(
                    "Data_aktualizacji"
                ),

            "url":
                url,

            "archive":
                str(
                    destination
                ),

            "archive_size_bytes":
                int(
                    destination
                    .stat()
                    .st_size
                ),

            "sha256":
                sha256,

            "zip_validation":
                validation,
        }

    with open(
        MANIFEST_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 120)
    print("DOWNLOAD SUMMARY")
    print("=" * 120)

    for city_key, info in (
        manifest["cities"].items()
    ):

        print()

        print(
            f"{info['display_name']}"
        )

        print(
            f"  TERYT: "
            f"{info['TERYT']}"
        )

        print(
            f"  GML files: "
            f"{info['zip_validation']['gml_count']}"
        )

        print(
            f"  Archive: "
            f"{info['archive']}"
        )

    print()
    print("=" * 120)
    print("SCRIPT 47 COMPLETE")
    print("=" * 120)

    print()
    print("Manifest:")
    print(MANIFEST_JSON)

    print()
    print(
        "NEXT: inspect the internal "
        "BDOT10k GML feature classes "
        "and identify building/road layers."
    )


if __name__ == "__main__":
    main()