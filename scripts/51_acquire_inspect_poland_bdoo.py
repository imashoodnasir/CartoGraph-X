from pathlib import Path
import json
import time
import zipfile
import hashlib
import shutil
import re
import xml.etree.ElementTree as ET

import requests


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "poland"
    / "bdoo"
)

INTERIM_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "poland"
    / "bdoo"
)

RAW_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

INTERIM_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# OFFICIAL SERVICES
# =============================================================================

# We already established in Script 45 that the official
# BDOO download WMS is reachable.
#
# The purpose of Script 51 is:
#
# 1. interrogate the official service,
# 2. discover download/index information,
# 3. identify downloadable BDOO resources,
# 4. download them where possible,
# 5. recursively inspect archives,
# 6. identify candidate building and road classes.
#
# Do NOT hard-code target class names yet.

BDOO_CAPABILITIES_URLS = [
    (
        "https://mapy.geoportal.gov.pl/"
        "wss/service/PZGIK/BDOO/WMS/"
        "SkorowidzeArchiwalne"
        "?SERVICE=WMS"
        "&REQUEST=GetCapabilities"
        "&VERSION=1.3.0"
    ),
    (
        "https://mapy.geoportal.gov.pl/"
        "wss/service/PZGIK/BDOO/WMS/"
        "PobieranieBDOO"
        "?SERVICE=WMS"
        "&REQUEST=GetCapabilities"
        "&VERSION=1.3.0"
    ),
]


# =============================================================================
# REQUEST SETTINGS
# =============================================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/154.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

TIMEOUT = 120


# =============================================================================
# UTILITIES
# =============================================================================

def banner(text, char="="):

    print()
    print(char * 120)
    print(text)
    print(char * 120)


def separator(text):

    print()
    print("-" * 120)
    print(text)
    print("-" * 120)


def safe_name(value):

    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value
    )

    return value.strip("_")


def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def request_url(session, url, stream=False, attempts=5):

    last_error = None

    for attempt in range(1, attempts + 1):

        try:
            print(f"  Request attempt {attempt}/{attempts}...")

            response = session.get(
                url,
                headers=HEADERS,
                timeout=TIMEOUT,
                stream=stream,
                verify=True,
            )

            # Do not retry authentication/permission failures.
            if response.status_code in (401, 403):
                raise RuntimeError(
                    f"HTTP {response.status_code}: "
                    f"service requires authorization or blocks anonymous access"
                )

            response.raise_for_status()
            return response

        except RuntimeError:
            raise

        except Exception as e:

            last_error = e
            print(f"  Attempt failed: {type(e).__name__}: {e}")

            if attempt < attempts:
                time.sleep(min(2 ** attempt, 10))

    raise last_error


# =============================================================================
# WMS PARSING
# =============================================================================

def strip_namespace(tag):

    if "}" in tag:
        return tag.split(
            "}",
            1
        )[1]

    return tag


def parse_wms_capabilities(xml_bytes):

    root = ET.fromstring(
        xml_bytes
    )

    layers = []

    formats = []

    online_resources = []

    for element in root.iter():

        tag = strip_namespace(
            element.tag
        )

        if (
            tag == "Format"
            and element.text
        ):
            formats.append(
                element.text.strip()
            )

        if tag == "OnlineResource":

            href = None

            for key, value in (
                element.attrib.items()
            ):

                if (
                    key.endswith("href")
                    or key == "href"
                ):
                    href = value
                    break

            if href:
                online_resources.append(
                    href
                )

    # Extract Layer Name + Title pairs.
    for layer in root.iter():

        if (
            strip_namespace(
                layer.tag
            )
            != "Layer"
        ):
            continue

        name = None
        title = None
        abstract = None

        for child in list(layer):

            tag = strip_namespace(
                child.tag
            )

            if (
                tag == "Name"
                and child.text
            ):
                name = child.text.strip()

            elif (
                tag == "Title"
                and child.text
            ):
                title = child.text.strip()

            elif (
                tag == "Abstract"
                and child.text
            ):
                abstract = (
                    child.text.strip()
                )

        if name or title:

            layers.append({
                "name": name,
                "title": title,
                "abstract": abstract,
            })

    return {
        "layers": layers,
        "formats": sorted(
            set(formats)
        ),
        "online_resources": sorted(
            set(online_resources)
        ),
    }


# =============================================================================
# URL DISCOVERY
# =============================================================================

def collect_urls_from_text(text):

    if not text:
        return []

    pattern = (
        r'https?://'
        r'[^\s<>"\']+'
    )

    urls = re.findall(
        pattern,
        text,
        flags=re.IGNORECASE
    )

    cleaned = []

    for url in urls:

        url = (
            url
            .replace("&amp;", "&")
            .rstrip(".,);]")
        )

        if url not in cleaned:
            cleaned.append(url)

    return cleaned


def score_download_url(url):

    lower = url.lower()

    score = 0

    if "bdoo" in lower:
        score += 10

    if "opendata" in lower:
        score += 8

    if "download" in lower:
        score += 6

    if "pobier" in lower:
        score += 5

    if lower.endswith(".zip"):
        score += 12

    if ".zip?" in lower:
        score += 10

    if "gml" in lower:
        score += 5

    if "xml" in lower:
        score += 2

    return score


# =============================================================================
# DOWNLOAD
# =============================================================================

def download_file(
    session,
    url,
    output_path
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if output_path.exists():

        print(
            f"Already exists: "
            f"{output_path}"
        )

        return output_path

    response = request_url(
        session,
        url,
        stream=True
    )

    total = int(
        response.headers.get(
            "content-length",
            0
        )
        or 0
    )

    downloaded = 0

    temp_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    with open(
        temp_path,
        "wb"
    ) as f:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if not chunk:
                continue

            f.write(chunk)

            downloaded += len(chunk)

            if total:

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
                    flush=True
                )

    print()

    temp_path.replace(
        output_path
    )

    return output_path


# =============================================================================
# ARCHIVE INSPECTION
# =============================================================================

def inspect_zip(path):

    result = {
        "path": str(path),
        "valid_zip": False,
        "file_count": 0,
        "xml_files": [],
        "gml_files": [],
        "zip_files": [],
        "building_candidates": [],
        "road_candidates": [],
    }

    if not zipfile.is_zipfile(path):

        return result

    result["valid_zip"] = True

    with zipfile.ZipFile(
        path,
        "r"
    ) as archive:

        names = archive.namelist()

    result["file_count"] = len(
        names
    )

    for name in names:

        lower = name.lower()

        if lower.endswith(".xml"):
            result[
                "xml_files"
            ].append(name)

        elif lower.endswith(".gml"):
            result[
                "gml_files"
            ].append(name)

        elif lower.endswith(".zip"):
            result[
                "zip_files"
            ].append(name)

        basename = (
            Path(name)
            .name
            .lower()
        )

        # Only candidate discovery.
        # No assumptions yet.

        building_tokens = [
            "bud",
            "budy",
            "budyn",
            "building",
            "bubd",
        ]

        road_tokens = [
            "dro",
            "drog",
            "road",
            "skdr",
            "ulic",
            "komunik",
        ]

        building_score = sum(
            token in basename
            for token in building_tokens
        )

        road_score = sum(
            token in basename
            for token in road_tokens
        )

        if building_score > 0:

            result[
                "building_candidates"
            ].append({
                "name": name,
                "score":
                    int(building_score),
            })

        if road_score > 0:

            result[
                "road_candidates"
            ].append({
                "name": name,
                "score":
                    int(road_score),
            })

    result[
        "building_candidates"
    ].sort(
        key=lambda x:
        (-x["score"], x["name"])
    )

    result[
        "road_candidates"
    ].sort(
        key=lambda x:
        (-x["score"], x["name"])
    )

    return result


def extract_nested_archives(
    archive_path,
    output_root
):

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    extracted_archives = []

    if not zipfile.is_zipfile(
        archive_path
    ):
        return extracted_archives

    primary_dir = (
        output_root
        / safe_name(
            archive_path.stem
        )
    )

    if primary_dir.exists():
        shutil.rmtree(
            primary_dir
        )

    primary_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    with zipfile.ZipFile(
        archive_path,
        "r"
    ) as archive:

        archive.extractall(
            primary_dir
        )

    # Find every nested ZIP.
    nested = list(
        primary_dir.rglob("*.zip")
    )

    for nested_zip in nested:

        nested_output = (
            nested_zip.parent
            / (
                nested_zip.stem
                + "_extracted"
            )
        )

        try:

            if zipfile.is_zipfile(
                nested_zip
            ):

                nested_output.mkdir(
                    parents=True,
                    exist_ok=True
                )

                with zipfile.ZipFile(
                    nested_zip,
                    "r"
                ) as archive:

                    archive.extractall(
                        nested_output
                    )

                extracted_archives.append({
                    "archive":
                        str(nested_zip),
                    "output":
                        str(nested_output),
                })

        except Exception as e:

            extracted_archives.append({
                "archive":
                    str(nested_zip),
                "error":
                    f"{type(e).__name__}: {e}",
            })

    return extracted_archives


# =============================================================================
# EXTRACTED FILE INVENTORY
# =============================================================================

def inventory_extracted_files(root):

    records = []

    if not root.exists():
        return records

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        suffix = (
            path.suffix.lower()
        )

        if suffix not in {
            ".xml",
            ".gml",
            ".shp",
            ".gpkg",
            ".zip",
        }:
            continue

        name = path.name.lower()

        building_tokens = [
            "bud",
            "budy",
            "budyn",
            "building",
            "bubd",
        ]

        road_tokens = [
            "dro",
            "drog",
            "road",
            "skdr",
            "ulic",
            "komunik",
        ]

        building_score = sum(
            token in name
            for token in building_tokens
        )

        road_score = sum(
            token in name
            for token in road_tokens
        )

        records.append({
            "path": str(path),
            "name": path.name,
            "suffix": suffix,
            "size_mb":
                path.stat().st_size
                / 1024
                / 1024,
            "building_score":
                int(building_score),
            "road_score":
                int(road_score),
        })

    return records


# =============================================================================
# MAIN
# =============================================================================

def main():

    banner(
        "CARTOGEN — ACQUIRE AND INSPECT "
        "POLISH BDOO GENERALIZED TARGET DATA"
    )

    session = requests.Session()

    report = {
        "dataset": "BDOO",
        "role": "generalized_target",
        "country": "Poland",
        "services": [],
        "candidate_urls": [],
        "downloads": [],
        "extracted_files": [],
    }

    all_candidate_urls = []

    # -------------------------------------------------------------------------
    # 1. INSPECT OFFICIAL SERVICES
    # -------------------------------------------------------------------------

    banner(
        "1. INSPECT OFFICIAL BDOO SERVICES"
    )

    for index, url in enumerate(
        BDOO_CAPABILITIES_URLS,
        start=1
    ):

        separator(
            f"BDOO service {index}"
        )

        print(url)

        service_record = {
            "url": url,
            "reachable": False,
        }

        try:

            response = request_url(
                session,
                url
            )

            print(
                f"HTTP status: "
                f"{response.status_code}"
            )

            print(
                f"Response size: "
                f"{len(response.content) / 1024:.2f} KB"
            )

            parsed = (
                parse_wms_capabilities(
                    response.content
                )
            )

            service_record.update({
                "reachable": True,
                "http_status":
                    response.status_code,
                "layers":
                    parsed["layers"],
                "formats":
                    parsed["formats"],
                "online_resources":
                    parsed[
                        "online_resources"
                    ],
            })

            print()
            print(
                f"Layers: "
                f"{len(parsed['layers'])}"
            )

            for layer in (
                parsed["layers"]
            ):

                print(
                    f"  {layer['name']} "
                    f"| {layer['title']}"
                )

            text = response.text

            discovered = (
                collect_urls_from_text(
                    text
                )
            )

            for candidate in (
                parsed[
                    "online_resources"
                ]
                + discovered
            ):

                if candidate not in (
                    all_candidate_urls
                ):
                    all_candidate_urls.append(
                        candidate
                    )

        except Exception as e:

            service_record[
                "error"
            ] = (
                f"{type(e).__name__}: {e}"
            )

            print(
                f"[CHECK] "
                f"{service_record['error']}"
            )

        report[
            "services"
        ].append(
            service_record
        )

    # -------------------------------------------------------------------------
    # 2. RANK DOWNLOAD CANDIDATES
    # -------------------------------------------------------------------------

    banner(
        "2. DISCOVER BDOO DOWNLOAD CANDIDATES"
    )

    ranked_urls = []

    for url in all_candidate_urls:

        score = score_download_url(
            url
        )

        if score <= 0:
            continue

        ranked_urls.append({
            "url": url,
            "score": score,
        })

    ranked_urls.sort(
        key=lambda x:
        (-x["score"], x["url"])
    )

    report[
        "candidate_urls"
    ] = ranked_urls

    if ranked_urls:

        for item in ranked_urls[:50]:

            print(
                f"score={item['score']:>2} "
                f"| {item['url']}"
            )

    else:

        print(
            "No direct downloadable ZIP URL "
            "was embedded in GetCapabilities."
        )

        print()
        print(
            "This is not a failure. "
            "The BDOO WMS may expose an index "
            "layer whose features contain the "
            "actual download links."
        )

    # -------------------------------------------------------------------------
    # 3. TRY DIRECT ZIP CANDIDATES
    # -------------------------------------------------------------------------

    banner(
        "3. TEST DIRECT BDOO ARCHIVE CANDIDATES"
    )

    direct_zip_urls = [
        item["url"]
        for item in ranked_urls
        if (
            ".zip" in
            item["url"].lower()
        )
    ]

    if not direct_zip_urls:

        print(
            "No direct ZIP URLs discovered "
            "in capabilities."
        )

    for i, url in enumerate(
        direct_zip_urls,
        start=1
    ):

        separator(
            f"Candidate archive {i}"
        )

        filename = (
            url.split("?")[0]
            .rstrip("/")
            .split("/")[-1]
        )

        if not filename.lower().endswith(
            ".zip"
        ):
            filename = (
                f"bdoo_candidate_{i}.zip"
            )

        filename = safe_name(
            filename
        )

        output_path = (
            RAW_ROOT
            / filename
        )

        record = {
            "url": url,
            "output":
                str(output_path),
        }

        try:

            download_file(
                session,
                url,
                output_path
            )

            record[
                "size_mb"
            ] = (
                output_path.stat().st_size
                / 1024
                / 1024
            )

            record[
                "sha256"
            ] = sha256_file(
                output_path
            )

            inspection = inspect_zip(
                output_path
            )

            record[
                "inspection"
            ] = inspection

            print(
                f"Archive size: "
                f"{record['size_mb']:.2f} MB"
            )

            print(
                f"Valid ZIP: "
                f"{inspection['valid_zip']}"
            )

            print(
                f"Files: "
                f"{inspection['file_count']}"
            )

            print(
                f"XML: "
                f"{len(inspection['xml_files'])}"
            )

            print(
                f"GML: "
                f"{len(inspection['gml_files'])}"
            )

            print(
                f"Nested ZIP: "
                f"{len(inspection['zip_files'])}"
            )

            if inspection[
                "valid_zip"
            ]:

                extract_root = (
                    INTERIM_ROOT
                    / "extracted"
                )

                nested_result = (
                    extract_nested_archives(
                        output_path,
                        extract_root
                    )
                )

                record[
                    "nested_extraction"
                ] = nested_result

        except Exception as e:

            record[
                "error"
            ] = (
                f"{type(e).__name__}: {e}"
            )

            print(
                f"[CHECK] "
                f"{record['error']}"
            )

        report[
            "downloads"
        ].append(
            record
        )

    # -------------------------------------------------------------------------
    # 4. INVENTORY ANY EXTRACTED CONTENT
    # -------------------------------------------------------------------------

    banner(
        "4. INVENTORY EXTRACTED BDOO CONTENT"
    )

    extracted_root = (
        INTERIM_ROOT
        / "extracted"
    )

    inventory = (
        inventory_extracted_files(
            extracted_root
        )
    )

    report[
        "extracted_files"
    ] = inventory

    print(
        f"Relevant extracted files: "
        f"{len(inventory):,}"
    )

    building_candidates = sorted(
        [
            x
            for x in inventory
            if x[
                "building_score"
            ] > 0
        ],
        key=lambda x:
        (
            -x[
                "building_score"
            ],
            x["name"]
        )
    )

    road_candidates = sorted(
        [
            x
            for x in inventory
            if x[
                "road_score"
            ] > 0
        ],
        key=lambda x:
        (
            -x[
                "road_score"
            ],
            x["name"]
        )
    )

    separator(
        "BUILDING CANDIDATES"
    )

    if building_candidates:

        for item in (
            building_candidates[:50]
        ):

            print(
                f"score="
                f"{item['building_score']} "
                f"| {item['name']} "
                f"| {item['size_mb']:.2f} MB"
            )

    else:

        print(
            "No extracted building "
            "candidate identified yet."
        )

    separator(
        "ROAD CANDIDATES"
    )

    if road_candidates:

        for item in (
            road_candidates[:50]
        ):

            print(
                f"score="
                f"{item['road_score']} "
                f"| {item['name']} "
                f"| {item['size_mb']:.2f} MB"
            )

    else:

        print(
            "No extracted road "
            "candidate identified yet."
        )

    # -------------------------------------------------------------------------
    # 5. SAVE REPORT
    # -------------------------------------------------------------------------

    banner(
        "5. SAVE BDOO DISCOVERY REPORT"
    )

    report_path = (
        INTERIM_ROOT
        / "bdoo_acquisition_inspection.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Saved:\n{report_path}"
    )

    # -------------------------------------------------------------------------
    # FINAL DECISION
    # -------------------------------------------------------------------------

    banner(
        "SCRIPT 51 COMPLETE"
    )

    print(
        f"Official services inspected: "
        f"{len(report['services'])}"
    )

    print(
        f"Candidate URLs: "
        f"{len(ranked_urls)}"
    )

    print(
        f"Direct ZIP candidates: "
        f"{len(direct_zip_urls)}"
    )

    print(
        f"Extracted relevant files: "
        f"{len(inventory)}"
    )

    if (
        building_candidates
        and road_candidates
    ):

        print()
        print(
            "[PASS] Candidate BDOO building "
            "and road target files discovered."
        )

        print()
        print(
            "NEXT: Script 52 — inspect the "
            "candidate BDOO feature classes, "
            "geometry, CRS and attributes."
        )

    else:

        print()
        print(
            "[CHECK] Direct BDOO target files "
            "were not fully identified yet."
        )

        print(
            "This likely means the WMS index "
            "must be queried to obtain the "
            "actual BDOO download links."
        )

        print()
        print(
            "NEXT: use this Script 51 output "
            "to build Script 52 around the "
            "specific index layer/service "
            "returned by Geoportal."
        )


if __name__ == "__main__":
    main()