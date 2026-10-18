from pathlib import Path
import json
import hashlib
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "lithuania"
)

GRPK_ROOT = (
    RAW_ROOT
    / "grpk"
)

ARCHIVE_PATH = (
    GRPK_ROOT
    / "GRPK_Open_SHP.zip"
)

EXTRACT_ROOT = (
    GRPK_ROOT
    / "extracted"
)

MANIFEST_PATH = (
    GRPK_ROOT
    / "download_manifest.json"
)

INVENTORY_PATH = (
    GRPK_ROOT
    / "archive_inventory.json"
)


# Official Geoportal open-data download
GRPK_URL = (
    "https://www.geoportal.lt/"
    "download/opendata/GRPK/"
    "GRPK_Open_SHP.zip"
)

DATASET_INFO = {
    "country": "Lithuania",
    "dataset": "GRPK",
    "full_name": (
        "Spatial Data Set of "
        "(Geo) Reference Base Cadastre"
    ),
    "role": "detailed_source",
    "nominal_scale": "1:10000",
    "expected_crs": "EPSG:3346",
    "format": "SHP",
    "license": "CC BY 4.0",
    "expected_content": [
        "roads",
        "streets",
        "buildings",
        "transport_network",
        "land_cover",
        "hydrography",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def human_size(size_bytes):

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]

    size = float(size_bytes)

    for unit in units:

        if size < 1024.0:
            return (
                f"{size:,.2f} {unit}"
            )

        size /= 1024.0

    return f"{size:,.2f} PB"


def sha256_file(path):

    digest = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as file:

        while True:

            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def progress_hook(
    block_number,
    block_size,
    total_size
):

    downloaded = (
        block_number
        * block_size
    )

    if total_size <= 0:
        return

    downloaded = min(
        downloaded,
        total_size
    )

    pct = (
        downloaded
        / total_size
        * 100.0
    )

    print(
        "\r"
        f"Downloaded: "
        f"{human_size(downloaded):>12} "
        f"/ "
        f"{human_size(total_size):>12} "
        f"({pct:6.2f}%)",
        end="",
        flush=True
    )


def download_file(
    url,
    output
):

    if output.exists():

        size = output.stat().st_size

        if size > 0:

            print(
                "Archive already exists:"
            )

            print(output)

            print(
                f"Size: "
                f"{human_size(size)}"
            )

            print(
                "Skipping download."
            )

            return False

    print()
    print("Downloading:")
    print(url)

    print()
    print("Destination:")
    print(output)

    print()

    temporary = output.with_suffix(
        output.suffix + ".part"
    )

    if temporary.exists():
        temporary.unlink()

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent":
                "Mozilla/5.0 "
                "CartoGen Research Downloader"
        }
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=120
        ) as response:

            total_size = int(
                response.headers.get(
                    "Content-Length",
                    0
                )
            )

            downloaded = 0

            with open(
                temporary,
                "wb"
            ) as file:

                while True:

                    block = response.read(
                        1024 * 1024
                    )

                    if not block:
                        break

                    file.write(block)

                    downloaded += len(block)

                    if total_size > 0:

                        pct = (
                            downloaded
                            / total_size
                            * 100
                        )

                        print(
                            "\r"
                            f"Downloaded: "
                            f"{human_size(downloaded):>12} "
                            f"/ "
                            f"{human_size(total_size):>12} "
                            f"({pct:6.2f}%)",
                            end="",
                            flush=True
                        )

                    else:

                        print(
                            "\r"
                            f"Downloaded: "
                            f"{human_size(downloaded):>12}",
                            end="",
                            flush=True
                        )

        print()

        temporary.replace(
            output
        )

    except Exception:

        if temporary.exists():
            temporary.unlink()

        raise

    return True


def validate_zip(path):

    print()
    print(
        "Validating ZIP archive..."
    )

    if not zipfile.is_zipfile(
        path
    ):

        raise RuntimeError(
            "Downloaded file is not "
            "a valid ZIP archive."
        )

    with zipfile.ZipFile(
        path,
        "r"
    ) as archive:

        bad_file = (
            archive.testzip()
        )

        if bad_file is not None:

            raise RuntimeError(
                "Corrupt file inside ZIP: "
                f"{bad_file}"
            )

        members = (
            archive.infolist()
        )

    print("[PASS] Valid ZIP archive")

    print(
        f"Files inside: "
        f"{len(members):,}"
    )

    return members


def inventory_archive(
    path,
    members
):

    records = []

    print()
    print("=" * 120)
    print("ARCHIVE INVENTORY")
    print("=" * 120)

    shapefiles = []

    for index, member in enumerate(
        members,
        start=1
    ):

        if member.is_dir():
            continue

        suffix = (
            Path(member.filename)
            .suffix
            .lower()
        )

        record = {
            "path":
                member.filename,

            "compressed_size":
                int(
                    member.compress_size
                ),

            "uncompressed_size":
                int(
                    member.file_size
                ),

            "extension":
                suffix,
        }

        records.append(
            record
        )

        if suffix == ".shp":

            shapefiles.append(
                member.filename
            )

    print(
        f"Total files: "
        f"{len(records):,}"
    )

    print(
        f"Shapefiles: "
        f"{len(shapefiles):,}"
    )

    print()

    for shp in shapefiles:

        print(
            f"  {shp}"
        )

    inventory = {
        "archive":
            str(path),

        "archive_size_bytes":
            int(
                path.stat().st_size
            ),

        "archive_size_human":
            human_size(
                path.stat().st_size
            ),

        "file_count":
            len(records),

        "shapefile_count":
            len(shapefiles),

        "shapefiles":
            shapefiles,

        "files":
            records,
    }

    return inventory


def extract_archive(
    path
):

    print()
    print("=" * 120)
    print("EXTRACT GRPK")
    print("=" * 120)

    if EXTRACT_ROOT.exists():

        existing_shp = list(
            EXTRACT_ROOT.rglob(
                "*.shp"
            )
        )

        if existing_shp:

            print(
                "Extraction already exists."
            )

            print(
                f"Existing shapefiles: "
                f"{len(existing_shp):,}"
            )

            print(
                "Skipping extraction."
            )

            return

        shutil.rmtree(
            EXTRACT_ROOT
        )

    EXTRACT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "Extracting archive..."
    )

    with zipfile.ZipFile(
        path,
        "r"
    ) as archive:

        archive.extractall(
            EXTRACT_ROOT
        )

    print(
        "[PASS] Extraction complete."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 120)
    print(
        "CARTOGEN — DOWNLOAD "
        "LITHUANIAN GRPK DATA"
    )
    print("=" * 120)

    GRPK_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("DATASET")
    print("-" * 120)

    for key, value in (
        DATASET_INFO.items()
    ):

        print(
            f"{key:<24}: "
            f"{value}"
        )

    print()
    print(
        "Official source:"
    )

    print(
        GRPK_URL
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    downloaded_now = (
        download_file(
            GRPK_URL,
            ARCHIVE_PATH
        )
    )

    if not ARCHIVE_PATH.exists():

        raise FileNotFoundError(
            ARCHIVE_PATH
        )

    print()
    print(
        f"Archive size: "
        f"{human_size(ARCHIVE_PATH.stat().st_size)}"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    members = validate_zip(
        ARCHIVE_PATH
    )

    # --------------------------------------------------------
    # Inventory
    # --------------------------------------------------------

    inventory = inventory_archive(
        ARCHIVE_PATH,
        members
    )

    with open(
        INVENTORY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            inventory,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Hash
    # --------------------------------------------------------

    print()
    print(
        "Calculating SHA-256..."
    )

    sha256 = sha256_file(
        ARCHIVE_PATH
    )

    print(
        f"SHA-256: {sha256}"
    )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    extract_archive(
        ARCHIVE_PATH
    )

    # --------------------------------------------------------
    # Final shapefile discovery
    # --------------------------------------------------------

    shapefiles = sorted(
        EXTRACT_ROOT.rglob(
            "*.shp"
        )
    )

    print()
    print("=" * 120)
    print("EXTRACTED SHAPEFILES")
    print("=" * 120)

    for path in shapefiles:

        relative = (
            path.relative_to(
                EXTRACT_ROOT
            )
        )

        print(
            f"  {relative}"
        )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {
        **DATASET_INFO,

        "source_url":
            GRPK_URL,

        "download_timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "downloaded_this_run":
            downloaded_now,

        "archive_path":
            str(
                ARCHIVE_PATH
            ),

        "archive_size_bytes":
            int(
                ARCHIVE_PATH
                .stat()
                .st_size
            ),

        "sha256":
            sha256,

        "extraction_path":
            str(
                EXTRACT_ROOT
            ),

        "shapefile_count":
            len(
                shapefiles
            ),

        "shapefiles": [
            str(
                path.relative_to(
                    EXTRACT_ROOT
                )
            )
            for path in shapefiles
        ],
    }

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 120)
    print("SCRIPT 28 COMPLETE")
    print("=" * 120)

    print(
        f"Archive:\n"
        f"{ARCHIVE_PATH}"
    )

    print()
    print(
        f"Extracted data:\n"
        f"{EXTRACT_ROOT}"
    )

    print()
    print(
        f"Inventory:\n"
        f"{INVENTORY_PATH}"
    )

    print()
    print(
        f"Manifest:\n"
        f"{MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()