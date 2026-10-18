from pathlib import Path
import json
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

API_URL = "https://api.os.uk/downloads/v1/products"

TARGET_KEYWORDS = [
    "OpenMap",
    "VectorMap District",
    "Zoomstack",
]


def main():

    print("=" * 90)
    print("CARTOGEN — DISCOVER ORDNANCE SURVEY PRODUCTS")
    print("=" * 90)

    response = requests.get(
        API_URL,
        params={"expanded": "true"},
        timeout=60,
    )

    response.raise_for_status()

    products = response.json()

    print(f"\nProducts returned by OS API: {len(products)}\n")

    output_dir = PROJECT_ROOT / "data" / "raw" / "os"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save complete response for reproducibility.
    raw_output = output_dir / "os_product_catalog.json"

    with open(raw_output, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2)

    matches = []

    for product in products:

        name = str(product.get("name", ""))
        product_id = str(product.get("id", ""))

        searchable = f"{name} {product_id}".lower()

        if any(
            keyword.lower() in searchable
            for keyword in TARGET_KEYWORDS
        ):
            matches.append(product)

    print("Relevant CartoGen products")
    print("-" * 90)

    for product in matches:

        print(f"Name:    {product.get('name')}")
        print(f"ID:      {product.get('id')}")
        print(f"Version: {product.get('version')}")
        print(f"URL:     {product.get('url')}")
        print("-" * 90)

    selected_output = (
        output_dir /
        "cartogen_os_products.json"
    )

    with open(
        selected_output,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(matches, f, indent=2)

    print()
    print(f"Full catalog saved: {raw_output}")
    print(f"CartoGen products saved: {selected_output}")

    if len(matches) < 3:
        print()
        print(
            "WARNING: Fewer than three expected products were found. "
            "Inspect os_product_catalog.json before continuing."
        )


if __name__ == "__main__":
    main()