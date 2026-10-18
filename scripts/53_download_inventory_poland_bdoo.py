from pathlib import Path
import hashlib
import json
import re
import shutil
import time
import zipfile

import requests


ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = ROOT / "data" / "raw" / "poland" / "bdoo"
INTERIM_ROOT = ROOT / "data" / "interim" / "poland" / "bdoo_inventory"

RAW_ROOT.mkdir(parents=True, exist_ok=True)
INTERIM_ROOT.mkdir(parents=True, exist_ok=True)

YEAR = 2026

TARGETS = {
    "warsaw": {
        "city": "Warsaw",
        "voivodeship": "mazowieckie",
        "code": "14",
        "url": (
            "https://opendata.geoportal.gov.pl/bdoo/2026/"
            "PL.PZGiK.201.14.zip"
        ),
    },
    "krakow": {
        "city": "Krakow",
        "voivodeship": "malopolskie",
        "code": "12",
        "url": (
            "https://opendata.geoportal.gov.pl/bdoo/2026/"
            "PL.PZGiK.201.12.zip"
        ),
    },
    "gdansk": {
        "city": "Gdansk",
        "voivodeship": "pomorskie",
        "code": "22",
        "url": (
            "https://opendata.geoportal.gov.pl/bdoo/2026/"
            "PL.PZGiK.201.22.zip"
        ),
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/154.0 Safari/537.36"
    )
}

TIMEOUT = 120


def banner(text):
    print("\n" + "=" * 120)
    print(text)
    print("=" * 120)


def section(text):
    print("\n" + "-" * 120)
    print(text)
    print("-" * 120)


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def download_file(session, url, destination):
    if destination.exists() and destination.stat().st_size > 0:

        print("Archive already exists:")
        print(destination)
        print(
            f"Existing size: "
            f"{destination.stat().st_size / 1024 / 1024:.2f} MB"
        )

        try:
            with zipfile.ZipFile(destination, "r") as z:
                bad = z.testzip()

            if bad is None:
                print("[PASS] Existing archive is valid; skipping download.")
                return
            else:
                print(
                    f"[CHECK] Existing archive contains bad member: {bad}"
                )

        except Exception as exc:
            print(
                f"[CHECK] Existing file is not a valid ZIP: {exc}"
            )

        print("Removing invalid/incomplete existing archive.")
        destination.unlink()

    print(f"Downloading:\n{url}")
    print(f"Destination:\n{destination}")

    temp = destination.with_suffix(destination.suffix + ".part")

    if temp.exists():
        temp.unlink()

    for attempt in range(1, 6):

        try:
            print(f"Request attempt {attempt}/5...")

            with session.get(
                url,
                headers=HEADERS,
                timeout=TIMEOUT,
                stream=True,
            ) as r:

                r.raise_for_status()

                total = int(r.headers.get("Content-Length", 0))

                downloaded = 0
                last_print = 0

                with open(temp, "wb") as f:

                    for chunk in r.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if not chunk:
                            continue

                        f.write(chunk)
                        downloaded += len(chunk)

                        downloaded_mb = downloaded / 1024 / 1024

                        if downloaded_mb - last_print >= 10:
                            if total > 0:
                                pct = 100 * downloaded / total
                                print(
                                    f"  Downloaded: "
                                    f"{downloaded_mb:.1f} MB "
                                    f"({pct:.1f}%)"
                                )
                            else:
                                print(
                                    f"  Downloaded: "
                                    f"{downloaded_mb:.1f} MB"
                                )

                            last_print = downloaded_mb

            temp.replace(destination)

            print(
                f"Final size: "
                f"{destination.stat().st_size / 1024 / 1024:.2f} MB"
            )

            return

        except Exception as exc:

            print(
                f"Attempt failed: "
                f"{type(exc).__name__}: {exc}"
            )

            if temp.exists():
                temp.unlink()

            if attempt == 5:
                raise

            time.sleep(min(2 ** attempt, 10))


def safe_extract(zf, destination):
    destination = destination.resolve()

    for member in zf.infolist():

        target = (destination / member.filename).resolve()

        if destination not in target.parents and target != destination:
            raise RuntimeError(
                f"Unsafe ZIP member path: {member.filename}"
            )

    zf.extractall(destination)


def extract_zip(zip_path, destination):
    destination.mkdir(parents=True, exist_ok=True)

    marker = destination / ".extraction_complete"

    if marker.exists():
        print(f"Already extracted: {destination}")
        return

    # Clean partial extraction.
    for p in destination.iterdir():
        if p.name == ".extraction_complete":
            continue

        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()

    with zipfile.ZipFile(zip_path, "r") as z:
        safe_extract(z, destination)

    marker.write_text(
        f"Extracted from {zip_path.name}\n",
        encoding="utf-8",
    )


def recursive_extract_nested_zips(root):
    """
    Some Geoportal packages contain ZIPs inside the main ZIP.
    Extract them recursively so that XML/GML layers become visible.
    """

    processed = set()
    extraction_records = []

    while True:

        nested = [
            p for p in root.rglob("*.zip")
            if str(p.resolve()) not in processed
        ]

        if not nested:
            break

        for zip_path in nested:

            processed.add(str(zip_path.resolve()))

            relative = zip_path.relative_to(root)

            nested_dir = (
                zip_path.parent /
                f"{zip_path.stem}_extracted"
            )

            print(f"\nNested ZIP: {relative}")
            print(f"Extracting to: {nested_dir}")

            try:

                extract_zip(zip_path, nested_dir)

                extraction_records.append(
                    {
                        "zip": str(relative),
                        "destination": str(
                            nested_dir.relative_to(root)
                        ),
                        "status": "PASS",
                    }
                )

            except Exception as exc:

                print(
                    f"[CHECK] Failed nested extraction: "
                    f"{type(exc).__name__}: {exc}"
                )

                extraction_records.append(
                    {
                        "zip": str(relative),
                        "destination": str(
                            nested_dir.relative_to(root)
                        ),
                        "status": "CHECK",
                        "error": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
                )

    return extraction_records


def classify_file(path):
    suffix = path.suffix.lower()
    name = path.name.lower()

    if suffix in {".xml", ".gml"}:
        family = "vector_xml"
    elif suffix in {".shp", ".gpkg", ".geojson"}:
        family = "vector_other"
    elif suffix == ".zip":
        family = "archive"
    elif suffix in {".xsd"}:
        family = "schema"
    elif suffix in {".txt", ".csv"}:
        family = "text"
    else:
        family = "other"

    role = []

    building_tokens = [
        "bubd",
        "budyn",
        "building",
        "zabud",
    ]

    road_tokens = [
        "skdr",
        "droga",
        "drogi",
        "road",
        "ulic",
        "komunik",
    ]

    if any(x in name for x in building_tokens):
        role.append("building_candidate")

    if any(x in name for x in road_tokens):
        role.append("road_candidate")

    return family, role


def inspect_xml_root(path):
    result = {
        "readable": False,
        "root_tag": None,
        "sample_tags": [],
    }

    try:
        import xml.etree.ElementTree as ET

        tags = []

        for event, elem in ET.iterparse(
            path,
            events=("start",),
        ):
            tag = elem.tag.split("}")[-1]

            if result["root_tag"] is None:
                result["root_tag"] = tag

            if tag not in tags:
                tags.append(tag)

            if len(tags) >= 25:
                break

        result["readable"] = True
        result["sample_tags"] = tags

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def inventory_tree(root):
    records = []

    for path in sorted(root.rglob("*")):

        if not path.is_file():
            continue

        if path.name == ".extraction_complete":
            continue

        relative = path.relative_to(root)

        family, roles = classify_file(path)

        record = {
            "relative_path": str(relative),
            "filename": path.name,
            "extension": path.suffix.lower(),
            "size_mb": round(
                path.stat().st_size / 1024 / 1024,
                4,
            ),
            "family": family,
            "roles": roles,
        }

        if (
            family == "vector_xml"
            and path.stat().st_size < 500 * 1024 * 1024
        ):
            record["xml"] = inspect_xml_root(path)

        records.append(record)

    return records


def find_candidates(records, role):
    return [
        x for x in records
        if role in x.get("roles", [])
    ]


def print_candidates(title, candidates):
    section(title)

    if not candidates:
        print("None detected automatically.")
        return

    for item in candidates:
        print(
            f"{item['size_mb']:10.3f} MB | "
            f"{item['relative_path']}"
        )


def main():

    banner(
        "CARTOGEN — DOWNLOAD AND INVENTORY "
        "POLISH BDOO 2026 TARGET DATA"
    )

    session = requests.Session()

    report = {
        "year": YEAR,
        "targets": {},
    }

    for key, target in TARGETS.items():

        banner(
            f"{target['city'].upper()} — "
            f"{target['voivodeship'].upper()}"
        )

        city_raw = RAW_ROOT / key
        city_raw.mkdir(parents=True, exist_ok=True)

        archive_name = Path(target["url"]).name
        archive = city_raw / archive_name

        # --------------------------------------------------------------
        # Download
        # --------------------------------------------------------------

        section("1. DOWNLOAD BDOO ARCHIVE")

        download_file(
            session,
            target["url"],
            archive,
        )

        # --------------------------------------------------------------
        # Validate
        # --------------------------------------------------------------

        section("2. VALIDATE ARCHIVE")

        with zipfile.ZipFile(archive, "r") as z:

            bad = z.testzip()

            members = z.infolist()

            print(f"ZIP members: {len(members):,}")

            if bad is None:
                print("[PASS] ZIP integrity test passed.")
            else:
                raise RuntimeError(
                    f"Corrupt ZIP member: {bad}"
                )

            print("\nTop-level archive contents:")

            for member in members[:100]:
                print(
                    f"  {member.file_size / 1024 / 1024:9.3f} MB | "
                    f"{member.filename}"
                )

            if len(members) > 100:
                print(
                    f"  ... {len(members) - 100:,} additional members"
                )

        digest = sha256_file(archive)

        print(f"\nSHA-256: {digest}")

        # --------------------------------------------------------------
        # Extract
        # --------------------------------------------------------------

        section("3. EXTRACT ARCHIVE")

        extract_root = (
            INTERIM_ROOT /
            key /
            "extracted"
        )

        extract_zip(
            archive,
            extract_root,
        )

        print(f"Extracted to:\n{extract_root}")

        # --------------------------------------------------------------
        # Nested archives
        # --------------------------------------------------------------

        section("4. RECURSIVELY EXTRACT NESTED ARCHIVES")

        nested_records = recursive_extract_nested_zips(
            extract_root
        )

        print(
            f"\nNested ZIPs processed: "
            f"{len(nested_records):,}"
        )

        # --------------------------------------------------------------
        # Inventory
        # --------------------------------------------------------------

        section("5. INVENTORY EXTRACTED CONTENT")

        records = inventory_tree(extract_root)

        print(f"Files discovered: {len(records):,}")

        counts = {}

        for item in records:
            counts[item["family"]] = (
                counts.get(item["family"], 0) + 1
            )

        print("\nFile families:")

        for family, count in sorted(counts.items()):
            print(f"  {family:20s} {count:,}")

        building_candidates = find_candidates(
            records,
            "building_candidate",
        )

        road_candidates = find_candidates(
            records,
            "road_candidate",
        )

        print_candidates(
            "BUILDING CANDIDATES",
            building_candidates,
        )

        print_candidates(
            "ROAD CANDIDATES",
            road_candidates,
        )

        # --------------------------------------------------------------
        # Relevant XML/GML listing
        # --------------------------------------------------------------

        section("XML / GML FEATURE FILES")

        vector_xml = [
            x for x in records
            if x["family"] == "vector_xml"
        ]

        if vector_xml:

            for item in vector_xml:

                print(
                    f"{item['size_mb']:10.3f} MB | "
                    f"{item['relative_path']}"
                )

        else:
            print("No XML/GML files detected.")

        report["targets"][key] = {
            "city": target["city"],
            "voivodeship": target["voivodeship"],
            "code": target["code"],
            "url": target["url"],
            "archive": str(archive),
            "archive_size_mb": round(
                archive.stat().st_size / 1024 / 1024,
                3,
            ),
            "sha256": digest,
            "nested_archives": nested_records,
            "file_count": len(records),
            "family_counts": counts,
            "building_candidates": building_candidates,
            "road_candidates": road_candidates,
            "files": records,
        }

        city_report = (
            INTERIM_ROOT /
            key /
            "bdoo_inventory.json"
        )

        city_report.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            city_report,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                report["targets"][key],
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\nCity inventory:\n{city_report}")

    # ------------------------------------------------------------------
    # Global report
    # ------------------------------------------------------------------

    banner("GLOBAL BDOO INVENTORY SUMMARY")

    for key, data in report["targets"].items():

        print(f"\n{data['city']}")

        print(
            f"  Archive size:       "
            f"{data['archive_size_mb']:.2f} MB"
        )

        print(
            f"  Extracted files:    "
            f"{data['file_count']:,}"
        )

        print(
            f"  Building candidates:"
            f" {len(data['building_candidates']):,}"
        )

        print(
            f"  Road candidates:    "
            f"{len(data['road_candidates']):,}"
        )

    global_report = (
        INTERIM_ROOT /
        "bdoo_inventory_report.json"
    )

    with open(
        global_report,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    banner("SCRIPT 53 COMPLETE")

    print(f"Global report:\n{global_report}")

    print(
        "\nNEXT: Script 54 — identify the exact BDOO "
        "building and road feature classes and convert them "
        "to GeoPackage."
    )


if __name__ == "__main__":
    main()