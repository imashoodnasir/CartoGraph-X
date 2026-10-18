from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PREFERRED_FORMATS = [
    "GeoPackage",
    "Geopackage",
    "GPKG",
    "ESRI® Shapefile",
    "ESRI Shapefile",
    "Shapefile",
    "GML",
]


def load_inventory():

    path = (
        PROJECT_ROOT /
        "data" /
        "raw" /
        "os" /
        "os_download_inventory.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Run 06_inspect_os_downloads.py first."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_rank(format_name):

    if format_name is None:
        return 999

    value = str(format_name).lower()

    for index, preferred in enumerate(
        PREFERRED_FORMATS
    ):
        if preferred.lower() in value:
            return index

    return 999


def main():

    inventory = load_inventory()

    selections = {}

    print("=" * 100)
    print("CARTOGEN — SELECT OS DOWNLOADS")
    print("=" * 100)

    for product_id, product in inventory.items():

        name = product["name"]
        downloads = product["downloads"]

        if not downloads:
            print(f"[SKIP] No downloads for {name}")
            continue

        # Sort primarily by our preferred vector format.
        ranked = sorted(
            downloads,
            key=lambda x: (
                format_rank(x.get("format")),
                0 if x.get("area") == "GB" else 1,
                str(x.get("fileName", ""))
            )
        )

        best_rank = format_rank(
            ranked[0].get("format")
        )

        candidates = [
            x for x in ranked
            if format_rank(x.get("format")) == best_rank
        ]

        selections[product_id] = {
            "name": name,
            "preferred_format": (
                candidates[0].get("format")
                if candidates
                else None
            ),
            "candidates": candidates,
        }

        print()
        print(name)
        print("-" * 100)

        print(
            "Preferred format:",
            selections[product_id]["preferred_format"]
        )

        print(
            "Candidate files:",
            len(candidates)
        )

        for candidate in candidates[:20]:

            print(
                " ",
                candidate.get("area"),
                "|",
                candidate.get("format"),
                "|",
                candidate.get("subformat"),
                "|",
                candidate.get("fileName"),
            )

        if len(candidates) > 20:
            print(
                f"  ... {len(candidates) - 20} "
                f"additional candidates"
            )

    output = (
        PROJECT_ROOT /
        "data" /
        "raw" /
        "os" /
        "cartogen_os_download_selection.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(selections, f, indent=2)

    print()
    print("=" * 100)
    print("SELECTION INVENTORY SAVED")
    print("=" * 100)
    print(output)


if __name__ == "__main__":
    main()
    