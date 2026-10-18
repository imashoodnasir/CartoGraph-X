from pathlib import Path
import json
import requests
from tqdm import tqdm
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_OS = (
    PROJECT_ROOT /
    "data" /
    "raw" /
    "os"
)


def load_selection():

    path = (
        RAW_OS /
        "cartogen_os_download_selection.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Run 07_select_os_downloads.py first."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_candidate(candidates):

    if not candidates:
        return None

    # Prefer complete GB coverage where available.
    gb = [
        item for item in candidates
        if str(item.get("area", "")).upper() == "GB"
    ]

    if gb:
        return gb[0]

    # Otherwise use the first API-ranked candidate.
    #
    # For development we intentionally avoid downloading
    # every National Grid tile until the pipeline is tested.
    return candidates[0]


def safe_filename(item, product_id):

    file_name = item.get("fileName")

    if file_name:
        return Path(file_name).name

    url = item.get("url", "")

    parsed = urlparse(url)

    name = Path(parsed.path).name

    if name:
        return name

    return f"{product_id}_download.zip"


def download_file(url, destination):

    if destination.exists():

        print(
            f"[EXISTS] {destination.name}"
        )

        return

    print(f"Downloading: {destination.name}")

    with requests.get(
        url,
        stream=True,
        timeout=120
    ) as response:

        response.raise_for_status()

        total = int(
            response.headers.get(
                "content-length",
                0
            )
        )

        with open(
            destination,
            "wb"
        ) as file:

            with tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024
            ) as progress:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if not chunk:
                        continue

                    file.write(chunk)
                    progress.update(len(chunk))


def main():

    print("=" * 100)
    print("CARTOGEN — DOWNLOAD OS DEVELOPMENT DATA")
    print("=" * 100)

    selection = load_selection()

    manifest = {}

    for product_id, product in selection.items():

        candidates = product["candidates"]

        candidate = choose_candidate(
            candidates
        )

        if candidate is None:

            print(
                f"[SKIP] No candidate for "
                f"{product['name']}"
            )

            continue

        url = candidate.get("url")

        if not url:

            print(
                f"[SKIP] No download URL for "
                f"{product['name']}"
            )

            continue

        product_dir = (
            RAW_OS /
            product_id
        )

        product_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        filename = safe_filename(
            candidate,
            product_id
        )

        destination = (
            product_dir /
            filename
        )

        print()
        print("-" * 100)
        print(f"Product: {product['name']}")
        print(f"ID:      {product_id}")
        print(f"Area:    {candidate.get('area')}")
        print(f"Format:  {candidate.get('format')}")
        print(f"File:    {filename}")
        print("-" * 100)

        download_file(
            url,
            destination
        )

        manifest[product_id] = {
            "product_name": product["name"],
            "product_id": product_id,
            "area": candidate.get("area"),
            "format": candidate.get("format"),
            "subformat": candidate.get("subformat"),
            "source_url": url,
            "local_file": str(
                destination.relative_to(
                    PROJECT_ROOT
                )
            ),
        }

    manifest_path = (
        RAW_OS /
        "download_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2
        )

    print()
    print("=" * 100)
    print("OS DEVELOPMENT DOWNLOAD COMPLETE")
    print("=" * 100)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()