from pathlib import Path
import json
import re
import time
import requests
from urllib.parse import urljoin, unquote
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = ROOT / "data" / "interim" / "poland" / "bdoo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_JSON = OUT_DIR / "bdoo_download_discovery.json"

BASE = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/BDOO/WMS/PobieranieBDOO"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/154.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

TIMEOUT = 60

TARGETS = {
    "warsaw": {
        "city": "Warsaw",
        "voivodeship": "Mazowieckie",
        "lon": 21.0122,
        "lat": 52.2297,
    },
    "krakow": {
        "city": "Krakow",
        "voivodeship": "Malopolskie",
        "lon": 19.9450,
        "lat": 50.0647,
    },
    "gdansk": {
        "city": "Gdansk",
        "voivodeship": "Pomorskie",
        "lon": 18.6466,
        "lat": 54.3520,
    },
}


def banner(text):
    print("\n" + "=" * 120)
    print(text)
    print("=" * 120)


def section(text):
    print("\n" + "-" * 120)
    print(text)
    print("-" * 120)


def request(session, params=None, url=None):
    target = url or BASE

    for attempt in range(1, 6):
        try:
            print(f"  Request attempt {attempt}/5...")

            r = session.get(
                target,
                params=params,
                headers=HEADERS,
                timeout=TIMEOUT,
            )

            if r.status_code in (401, 403):
                raise RuntimeError(
                    f"HTTP {r.status_code}: authorization/access restriction"
                )

            r.raise_for_status()
            return r

        except RuntimeError:
            raise

        except Exception as exc:
            print(f"  Attempt failed: {type(exc).__name__}: {exc}")

            if attempt == 5:
                raise

            time.sleep(min(2 ** attempt, 10))


def extract_urls(text):
    if not text:
        return []

    text = unquote(text)

    patterns = [
        r'https?://[^\s<>"\']+',
        r'https?:\\?/\\?/[^\s<>"\']+',
    ]

    found = []

    for pattern in patterns:
        for value in re.findall(pattern, text, flags=re.I):
            value = value.replace("\\/", "/")
            value = value.rstrip(".,);]")
            found.append(value)

    return sorted(set(found))


def print_response(r):
    print(f"HTTP status:   {r.status_code}")
    print(f"Content-Type:  {r.headers.get('Content-Type')}")
    print(f"Response size: {len(r.content) / 1024:.2f} KB")

    preview = r.text[:4000]

    print("\nResponse preview:")
    print(preview)

    return preview


def get_capabilities(session):
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetCapabilities",
        "VERSION": "1.3.0",
    }

    return request(session, params=params)


def get_layer_bbox(capabilities_text, layer_name):
    """
    Attempt to recover EX_GeographicBoundingBox for a named layer.
    """

    try:
        root = ET.fromstring(capabilities_text)
    except Exception:
        return None

    for layer in root.iter():
        if not layer.tag.endswith("Layer"):
            continue

        name = None

        for child in layer:
            if child.tag.endswith("Name"):
                name = child.text
                break

        if name != layer_name:
            continue

        west = east = south = north = None

        for child in layer.iter():

            tag = child.tag.split("}")[-1]

            if tag == "westBoundLongitude":
                west = float(child.text)

            elif tag == "eastBoundLongitude":
                east = float(child.text)

            elif tag == "southBoundLatitude":
                south = float(child.text)

            elif tag == "northBoundLatitude":
                north = float(child.text)

        if None not in (west, south, east, north):
            return [west, south, east, north]

    return None


def get_feature_info(session, lon, lat, info_format):
    """
    Build a small WMS map around the target point and query its centre.

    WMS 1.3.0 + EPSG:4326 uses latitude/longitude axis order
    in BBOX.
    """

    delta = 0.15

    west = lon - delta
    east = lon + delta
    south = lat - delta
    north = lat + delta

    width = 101
    height = 101

    # EPSG:4326 axis order in WMS 1.3.0:
    # min_lat,min_lon,max_lat,max_lon
    bbox = f"{south},{west},{north},{east}"

    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetFeatureInfo",
        "LAYERS": "Wojewodztwa",
        "QUERY_LAYERS": "Wojewodztwa",
        "STYLES": "",
        "CRS": "EPSG:4326",
        "BBOX": bbox,
        "WIDTH": width,
        "HEIGHT": height,
        "I": width // 2,
        "J": height // 2,
        "INFO_FORMAT": info_format,
        "FEATURE_COUNT": 10,
    }

    return request(session, params=params)


def metadata_request(session, layer):
    params = {
        "request": "GetMetadata",
        "layer": layer,
    }

    return request(session, params=params)


def main():

    banner("CARTOGEN — DISCOVER POLISH BDOO DOWNLOAD LINKS")

    session = requests.Session()

    report = {
        "service": BASE,
        "layer": "Wojewodztwa",
        "targets": {},
        "candidate_download_urls": [],
    }

    # ------------------------------------------------------------------
    # 1. Capabilities
    # ------------------------------------------------------------------

    banner("1. VALIDATE BDOO DOWNLOAD INDEX")

    r = get_capabilities(session)

    print(f"HTTP status:   {r.status_code}")
    print(f"Response size: {len(r.content) / 1024:.2f} KB")

    capabilities_text = r.text

    bbox = get_layer_bbox(capabilities_text, "Wojewodztwa")

    print(f"Wojewodztwa geographic bbox: {bbox}")

    report["wojewodztwa_bbox"] = bbox

    # ------------------------------------------------------------------
    # 2. GetMetadata
    # ------------------------------------------------------------------

    banner("2. INSPECT CUSTOM GETMETADATA RESPONSES")

    for layer in ["Wojewodztwa", "Panstwo"]:

        section(f"GetMetadata — {layer}")

        try:
            r = metadata_request(session, layer)

            preview = print_response(r)

            urls = extract_urls(r.text)

            print("\nURLs detected:")
            if urls:
                for url in urls:
                    print(f"  {url}")
            else:
                print("  None")

            report[f"metadata_{layer.lower()}"] = {
                "status": r.status_code,
                "content_type": r.headers.get("Content-Type"),
                "urls": urls,
                "preview": preview,
            }

        except Exception as exc:
            print(f"[CHECK] {type(exc).__name__}: {exc}")

            report[f"metadata_{layer.lower()}"] = {
                "error": f"{type(exc).__name__}: {exc}"
            }

    # ------------------------------------------------------------------
    # 3. FeatureInfo
    # ------------------------------------------------------------------

    banner("3. QUERY BDOO VOIVODESHIP INDEX")

    formats = [
        "application/json",
        "application/geo+json",
        "text/html",
        "text/plain",
        "text/xml",
        "application/vnd.ogc.gml",
    ]

    all_urls = set()

    for key, target in TARGETS.items():

        banner(
            f"{target['city'].upper()} — "
            f"{target['voivodeship'].upper()}"
        )

        target_report = {
            "city": target["city"],
            "voivodeship": target["voivodeship"],
            "longitude": target["lon"],
            "latitude": target["lat"],
            "responses": [],
        }

        for info_format in formats:

            section(f"INFO_FORMAT = {info_format}")

            try:

                r = get_feature_info(
                    session,
                    target["lon"],
                    target["lat"],
                    info_format,
                )

                preview = print_response(r)

                urls = extract_urls(r.text)

                if urls:
                    print("\nCandidate URLs:")
                    for url in urls:
                        print(f"  {url}")
                        all_urls.add(url)
                else:
                    print("\nCandidate URLs: none")

                target_report["responses"].append(
                    {
                        "info_format": info_format,
                        "status": r.status_code,
                        "content_type": r.headers.get("Content-Type"),
                        "urls": urls,
                        "preview": preview,
                    }
                )

                # Once we get useful feature data, no need to hammer
                # every remaining format.
                lower = r.text.lower()

                useful_tokens = [
                    ".zip",
                    "href",
                    "url",
                    "download",
                    "pobierz",
                    "wojew",
                    "mazowie",
                    "malopol",
                    "małopol",
                    "pomorsk",
                ]

                if any(token in lower for token in useful_tokens):
                    print("\n[PASS] Potentially useful response detected.")
                    break

            except Exception as exc:

                print(f"[CHECK] {type(exc).__name__}: {exc}")

                target_report["responses"].append(
                    {
                        "info_format": info_format,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        report["targets"][key] = target_report

    # ------------------------------------------------------------------
    # 4. Search every collected response for archive-like strings
    # ------------------------------------------------------------------

    banner("4. CONSOLIDATE DOWNLOAD CANDIDATES")

    candidate_urls = sorted(all_urls)

    archive_candidates = []

    for url in candidate_urls:

        score = 0

        low = url.lower()

        if ".zip" in low:
            score += 10

        if "bdoo" in low:
            score += 5

        if "opendata" in low:
            score += 5

        if "download" in low or "pobier" in low:
            score += 3

        archive_candidates.append(
            {
                "url": url,
                "score": score,
            }
        )

    archive_candidates.sort(
        key=lambda x: (-x["score"], x["url"])
    )

    if archive_candidates:

        for item in archive_candidates:
            print(
                f"score={item['score']:2d} | "
                f"{item['url']}"
            )

    else:
        print("No explicit archive URLs recovered.")

    report["candidate_download_urls"] = archive_candidates

    # ------------------------------------------------------------------
    # 5. Final decision
    # ------------------------------------------------------------------

    banner("5. BDOO DISCOVERY DECISION")

    explicit_zip = [
        x for x in archive_candidates
        if ".zip" in x["url"].lower()
    ]

    if explicit_zip:

        print(
            f"[PASS] Explicit BDOO archive URL(s) discovered: "
            f"{len(explicit_zip)}"
        )

        print("\nNEXT:")
        print(
            "Script 53 should download the required BDOO "
            "voivodeship packages."
        )

    elif archive_candidates:

        print(
            "[CHECK] URLs were recovered, but no explicit ZIP "
            "was identified."
        )

        print(
            "Inspect the candidate URLs above before constructing "
            "the downloader."
        )

    else:

        print(
            "[CHECK] GetFeatureInfo did not expose direct archive URLs."
        )

        print(
            "The next step will be to inspect the WMS index response "
            "structure / GetMap hyperlinks rather than guessing URLs."
        )

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    banner("SCRIPT 52 COMPLETE")

    print(f"Saved:\n{REPORT_JSON}")

    if explicit_zip:
        print(
            "\nNEXT: Script 53 — download and inventory BDOO "
            "generalized target packages."
        )
    else:
        print(
            "\nNEXT: send me the Script 52 output. "
            "We will use the actual server response to determine "
            "the BDOO download mechanism."
        )


if __name__ == "__main__":
    main()