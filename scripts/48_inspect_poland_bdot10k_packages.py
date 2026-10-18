from pathlib import Path
import zipfile
import json
import shutil
from collections import Counter, defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "poland"
    / "bdot10k"
)

OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "poland"
    / "bdot10k_inventory"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

CITY_ARCHIVES = {
    "warsaw": RAW_DIR / "warsaw" / "1465_GML.zip",
    "krakow": RAW_DIR / "krakow" / "1261_GML.zip",
    "gdansk": RAW_DIR / "gdansk" / "2261_GML.zip",
}


def separator(title):
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def classify_filename(name):
    lower = name.lower()

    if lower.endswith(".zip"):
        return "zip"
    elif lower.endswith(".gml"):
        return "gml"
    elif lower.endswith(".xml"):
        return "xml"
    elif lower.endswith(".xsd"):
        return "xsd"
    elif lower.endswith(".shp"):
        return "shp"
    elif lower.endswith(".gpkg"):
        return "gpkg"
    elif "." in Path(name).name:
        return Path(name).suffix.lower().lstrip(".")
    else:
        return "no_extension"


def inspect_zip(zip_path):
    result = {
        "archive": str(zip_path),
        "files": [],
        "extension_counts": {},
        "nested_zip_count": 0,
        "gml_count": 0,
    }

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [
            n for n in zf.namelist()
            if not n.endswith("/")
        ]

        counts = Counter()

        for name in names:
            file_type = classify_filename(name)
            counts[file_type] += 1

            result["files"].append({
                "name": name,
                "type": file_type,
                "size_bytes": zf.getinfo(name).file_size,
            })

        result["extension_counts"] = dict(counts)
        result["nested_zip_count"] = counts.get("zip", 0)
        result["gml_count"] = counts.get("gml", 0)

    return result


def extract_outer_archive(city, archive):
    destination = OUT_DIR / city / "outer_extracted"

    if destination.exists():
        shutil.rmtree(destination)

    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(destination)

    return destination


def recursively_find_files(root):
    results = defaultdict(list)

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        kind = classify_filename(path.name)

        results[kind].append(path)

    return results


def inspect_nested_zip(zip_path):
    result = {
        "archive": str(zip_path),
        "files": [],
        "extension_counts": {},
    }

    with zipfile.ZipFile(zip_path, "r") as zf:
        counts = Counter()

        for info in zf.infolist():
            if info.is_dir():
                continue

            kind = classify_filename(info.filename)
            counts[kind] += 1

            result["files"].append({
                "name": info.filename,
                "type": kind,
                "size_bytes": info.file_size,
            })

        result["extension_counts"] = dict(counts)

    return result


def score_bdot10k_name(name):
    """
    Diagnostic scoring only.

    BDOT10k commonly uses class prefixes such as:
      BUBD - building objects
      SKDR - road-related objects

    We do not yet assume these are definitely the final
    learning layers. Script 49 will inspect schemas/geometries.
    """

    upper = Path(name).name.upper()

    building_score = 0
    road_score = 0

    building_tokens = [
        "BUBD",
        "BUDY",
        "BUD",
    ]

    road_tokens = [
        "SKDR",
        "DROG",
        "JEZD",
        "ULIC",
    ]

    for token in building_tokens:
        if token in upper:
            building_score += 5

    for token in road_tokens:
        if token in upper:
            road_score += 5

    return building_score, road_score


def main():
    separator(
        "CARTOGEN — INSPECT POLISH BDOT10k PACKAGE STRUCTURE"
    )

    global_inventory = {
        "cities": {}
    }

    for city, archive in CITY_ARCHIVES.items():

        separator(city.upper())

        if not archive.exists():
            raise FileNotFoundError(
                f"Archive not found:\n{archive}"
            )

        print(f"Archive: {archive}")
        print(
            f"Size:    "
            f"{archive.stat().st_size / 1024 / 1024:.2f} MB"
        )

        # --------------------------------------------------------------
        # OUTER ZIP
        # --------------------------------------------------------------

        print()
        print("-" * 120)
        print("OUTER ARCHIVE")
        print("-" * 120)

        outer = inspect_zip(archive)

        print(
            f"Files:       "
            f"{len(outer['files'])}"
        )
        print(
            f"Nested ZIPs: "
            f"{outer['nested_zip_count']}"
        )
        print(
            f"Direct GML:  "
            f"{outer['gml_count']}"
        )

        print()
        print("Extension distribution:")

        for ext, count in sorted(
            outer["extension_counts"].items(),
            key=lambda x: (-x[1], x[0])
        ):
            print(
                f"  {ext:<20} {count:>6,}"
            )

        print()
        print("OUTER FILE LIST")
        print("-" * 120)

        for item in outer["files"]:
            print(
                f"{item['type']:<12} "
                f"{item['size_bytes'] / 1024:>12,.1f} KB  "
                f"{item['name']}"
            )

        # --------------------------------------------------------------
        # EXTRACT OUTER PACKAGE
        # --------------------------------------------------------------

        print()
        print("-" * 120)
        print("EXTRACTING OUTER PACKAGE")
        print("-" * 120)

        extracted = extract_outer_archive(
            city,
            archive
        )

        print(f"Extracted to:")
        print(extracted)

        found = recursively_find_files(
            extracted
        )

        print()
        print("Extracted file types:")

        for kind, paths in sorted(
            found.items(),
            key=lambda x: (-len(x[1]), x[0])
        ):
            print(
                f"  {kind:<20} {len(paths):>6,}"
            )

        # --------------------------------------------------------------
        # NESTED ARCHIVES
        # --------------------------------------------------------------

        nested_archives = found.get(
            "zip",
            []
        )

        nested_inventory = []

        if nested_archives:

            print()
            print("-" * 120)
            print(
                f"NESTED ARCHIVES ({len(nested_archives)})"
            )
            print("-" * 120)

            for i, nested in enumerate(
                nested_archives,
                start=1
            ):

                print()
                print(
                    f"[{i}/{len(nested_archives)}] "
                    f"{nested.relative_to(extracted)}"
                )

                try:
                    info = inspect_nested_zip(
                        nested
                    )

                except zipfile.BadZipFile:
                    print(
                        "  [WARNING] Invalid nested ZIP"
                    )
                    continue

                nested_inventory.append(
                    info
                )

                print(
                    f"  Files: "
                    f"{len(info['files'])}"
                )

                print(
                    f"  Types: "
                    f"{info['extension_counts']}"
                )

                for file_info in info["files"][:20]:
                    print(
                        f"    "
                        f"{file_info['type']:<10} "
                        f"{file_info['name']}"
                    )

                if len(info["files"]) > 20:
                    print(
                        f"    ... "
                        f"{len(info['files']) - 20} more"
                    )

        # --------------------------------------------------------------
        # SEARCH ALL NAMES FOR LIKELY CARTOGEN CLASSES
        # --------------------------------------------------------------

        print()
        print("-" * 120)
        print("LIKELY BUILDING / ROAD CLASS CANDIDATES")
        print("-" * 120)

        candidates = []

        # Files directly extracted
        for kind, paths in found.items():
            for path in paths:

                b_score, r_score = (
                    score_bdot10k_name(
                        path.name
                    )
                )

                if b_score > 0 or r_score > 0:
                    candidates.append({
                        "container": "outer",
                        "name": str(
                            path.relative_to(
                                extracted
                            )
                        ),
                        "type": kind,
                        "building_score": b_score,
                        "road_score": r_score,
                    })

        # Files inside nested ZIPs
        for nested_info in nested_inventory:

            for item in nested_info["files"]:

                b_score, r_score = (
                    score_bdot10k_name(
                        item["name"]
                    )
                )

                if b_score > 0 or r_score > 0:

                    candidates.append({
                        "container":
                            nested_info["archive"],

                        "name":
                            item["name"],

                        "type":
                            item["type"],

                        "building_score":
                            b_score,

                        "road_score":
                            r_score,
                    })

        candidates.sort(
            key=lambda x: (
                -max(
                    x["building_score"],
                    x["road_score"]
                ),
                x["name"]
            )
        )

        if candidates:

            for candidate in candidates:

                print()
                print(
                    f"Name: "
                    f"{candidate['name']}"
                )

                print(
                    f"Type: "
                    f"{candidate['type']}"
                )

                print(
                    f"Building score: "
                    f"{candidate['building_score']}"
                )

                print(
                    f"Road score:     "
                    f"{candidate['road_score']}"
                )

        else:

            print(
                "No filename-based candidates detected."
            )

        global_inventory["cities"][city] = {
            "outer_archive": outer,
            "extracted_directory":
                str(extracted),
            "nested_archives":
                nested_inventory,
            "candidates":
                candidates,
        }

        city_json = (
            OUT_DIR
            / city
            / "bdot10k_package_inventory.json"
        )

        city_json.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            city_json,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                global_inventory["cities"][city],
                f,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print(f"City inventory:")
        print(city_json)

    # ------------------------------------------------------------------
    # GLOBAL OUTPUT
    # ------------------------------------------------------------------

    global_json = (
        OUT_DIR
        / "poland_bdot10k_package_inventory.json"
    )

    with open(
        global_json,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            global_inventory,
            f,
            indent=2,
            ensure_ascii=False,
        )

    separator("SCRIPT 48 COMPLETE")

    print("Global inventory:")
    print(global_json)

    print()
    print(
        "NEXT: Script 49 will identify and extract "
        "the exact BDOT10k building and road feature "
        "classes after we see this package structure."
    )


if __name__ == "__main__":
    main()