from pathlib import Path
import json
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_URL = "https://api.os.uk/downloads/v1/products"


def load_selected_products():

    path = (
        PROJECT_ROOT /
        "data" /
        "raw" /
        "os" /
        "cartogen_os_products.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Run 05_discover_os_products.py first."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    print("=" * 100)
    print("CARTOGEN — INSPECT OS DOWNLOAD OPTIONS")
    print("=" * 100)

    products = load_selected_products()

    output = {}

    for product in products:

        product_id = product["id"]
        name = product.get("name", product_id)

        print()
        print("=" * 100)
        print(name)
        print(f"Product ID: {product_id}")
        print("=" * 100)

        url = f"{BASE_URL}/{product_id}/downloads"

        response = requests.get(
            url,
            timeout=60
        )

        response.raise_for_status()

        downloads = response.json()

        output[product_id] = {
            "name": name,
            "downloads": downloads
        }

        print(f"Available files: {len(downloads)}")
        print()

        for i, item in enumerate(downloads, start=1):

            print(f"[{i}]")

            print(
                f"  File:      "
                f"{item.get('fileName', 'N/A')}"
            )

            print(
                f"  Format:    "
                f"{item.get('format', 'N/A')}"
            )

            print(
                f"  Subformat: "
                f"{item.get('subformat', 'N/A')}"
            )

            print(
                f"  Area:      "
                f"{item.get('area', 'N/A')}"
            )

            print(
                f"  Size:      "
                f"{item.get('size', 'N/A')}"
            )

            print()

    output_path = (
        PROJECT_ROOT /
        "data" /
        "raw" /
        "os" /
        "os_download_inventory.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(output, f, indent=2)

    print("=" * 100)
    print("DOWNLOAD INVENTORY COMPLETE")
    print("=" * 100)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()