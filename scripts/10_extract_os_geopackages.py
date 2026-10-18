from pathlib import Path
import zipfile
import shutil
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_OS = PROJECT_ROOT / "data" / "raw" / "os"
INTERIM_OS = PROJECT_ROOT / "data" / "interim" / "os"


ARCHIVES = {
    "OpenMapLocal": {
        "archive": RAW_OS / "OpenMapLocal" / "opmplc_gpkg_gb.zip",
        "member": "Data/opmplc_gb.gpkg",
        "output": INTERIM_OS / "OpenMapLocal" / "opmplc_gb.gpkg",
    },

    "OpenZoomstack": {
        "archive": RAW_OS / "OpenZoomstack" / "OS_Open_Zoomstack.zip",
        "member": "OS_Open_Zoomstack.gpkg",
        "output": INTERIM_OS / "OpenZoomstack" / "OS_Open_Zoomstack.gpkg",
    },
}


def human_size(num_bytes):

    gb = num_bytes / (1024 ** 3)

    if gb >= 1:
        return f"{gb:.2f} GB"

    return f"{num_bytes / (1024 ** 2):.2f} MB"


def extract_member(
    archive_path,
    member_name,
    output_path
):

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Archive not found:\n{archive_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if output_path.exists():

        print(
            f"[EXISTS] {output_path}"
        )

        print(
            f"Size: "
            f"{human_size(output_path.stat().st_size)}"
        )

        return

    print()
    print("=" * 100)

    print(
        f"Archive: {archive_path.name}"
    )

    print(
        f"Member:  {member_name}"
    )

    print(
        f"Output:  {output_path}"
    )

    print("=" * 100)

    with zipfile.ZipFile(
        archive_path,
        "r"
    ) as archive:

        info = archive.getinfo(
            member_name
        )

        print(
            f"Uncompressed size: "
            f"{human_size(info.file_size)}"
        )

        temp_path = output_path.with_suffix(
            ".gpkg.part"
        )

        try:

            with archive.open(
                info,
                "r"
            ) as source:

                with open(
                    temp_path,
                    "wb"
                ) as target:

                    with tqdm(
                        total=info.file_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc="Extracting",
                    ) as progress:

                        while True:

                            chunk = source.read(
                                8 * 1024 * 1024
                            )

                            if not chunk:
                                break

                            target.write(chunk)

                            progress.update(
                                len(chunk)
                            )

            temp_path.replace(
                output_path
            )

        except Exception:

            if temp_path.exists():
                temp_path.unlink()

            raise

    actual_size = output_path.stat().st_size

    if actual_size != info.file_size:

        raise RuntimeError(
            "Extracted file size does not "
            "match ZIP metadata."
        )

    print(
        f"[OK] Extracted: "
        f"{human_size(actual_size)}"
    )


def main():

    print("=" * 100)
    print("CARTOGEN — EXTRACT OS GEOPACKAGES")
    print("=" * 100)

    for product_name, config in ARCHIVES.items():

        print()
        print(
            f"Processing: {product_name}"
        )

        extract_member(
            config["archive"],
            config["member"],
            config["output"],
        )

    print()
    print("=" * 100)
    print("GEOPACKAGE EXTRACTION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()