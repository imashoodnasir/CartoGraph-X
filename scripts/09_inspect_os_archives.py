from pathlib import Path
import zipfile
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_OS = (
    PROJECT_ROOT /
    "data" /
    "raw" /
    "os"
)

OUTPUT_FILE = (
    RAW_OS /
    "os_archive_inventory.json"
)


def human_size(size_bytes):

    gb = size_bytes / (1024 ** 3)

    if gb >= 1:
        return f"{gb:.2f} GB"

    mb = size_bytes / (1024 ** 2)

    return f"{mb:.2f} MB"


def inspect_zip(zip_path):

    print()
    print("=" * 110)
    print(f"ARCHIVE: {zip_path.name}")
    print("=" * 110)

    records = []

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as archive:

        members = archive.infolist()

        total_uncompressed = sum(
            item.file_size
            for item in members
        )

        print(
            f"Files inside:        {len(members)}"
        )

        print(
            f"Compressed archive:  "
            f"{human_size(zip_path.stat().st_size)}"
        )

        print(
            f"Uncompressed total:  "
            f"{human_size(total_uncompressed)}"
        )

        print()
        print("CONTENTS")
        print("-" * 110)

        for index, item in enumerate(
            members,
            start=1
        ):

            record = {
                "name": item.filename,
                "size_bytes": item.file_size,
                "size": human_size(
                    item.file_size
                ),
                "compressed_size_bytes":
                    item.compress_size,
            }

            records.append(record)

            print(
                f"{index:04d} | "
                f"{human_size(item.file_size):>12} | "
                f"{item.filename}"
            )

    return {
        "archive": str(
            zip_path.relative_to(
                PROJECT_ROOT
            )
        ),
        "compressed_size_bytes":
            zip_path.stat().st_size,
        "compressed_size":
            human_size(
                zip_path.stat().st_size
            ),
        "uncompressed_size_bytes":
            total_uncompressed,
        "uncompressed_size":
            human_size(
                total_uncompressed
            ),
        "number_of_files":
            len(records),
        "files":
            records,
    }


def main():

    print("=" * 110)
    print("CARTOGEN — INSPECT OS ARCHIVES")
    print("=" * 110)

    archives = sorted(
        RAW_OS.rglob("*.zip")
    )

    if not archives:

        raise FileNotFoundError(
            "No ZIP archives found under "
            "data/raw/os."
        )

    print(
        f"\nZIP archives found: "
        f"{len(archives)}"
    )

    inventory = []

    for archive in archives:

        result = inspect_zip(
            archive
        )

        inventory.append(
            result
        )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            inventory,
            file,
            indent=2
        )

    print()
    print("=" * 110)
    print("ARCHIVE INSPECTION COMPLETE")
    print("=" * 110)

    print(
        f"Inventory saved:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()