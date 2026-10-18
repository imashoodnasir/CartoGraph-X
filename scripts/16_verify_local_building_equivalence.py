from pathlib import Path
import sqlite3
import random
import json
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OPENMAP = (
    PROJECT_ROOT
    / "data/interim/os/OpenMapLocal/opmplc_gb.gpkg"
)

ZOOMSTACK = (
    PROJECT_ROOT
    / "data/interim/os/OpenZoomstack/OS_Open_Zoomstack.gpkg"
)

OUTPUT = (
    PROJECT_ROOT
    / "data/interim/os/local_building_equivalence.json"
)

SAMPLE_SIZE = 1000
SEED = 42
BATCH_SIZE = 400


def fetch_by_ids(connection, table, geom_column, ids):

    result = {}

    for start in range(0, len(ids), BATCH_SIZE):

        batch = ids[start:start + BATCH_SIZE]

        placeholders = ",".join(
            "?" for _ in batch
        )

        query = f"""
            SELECT id, "{geom_column}"
            FROM "{table}"
            WHERE id IN ({placeholders})
        """

        rows = connection.execute(
            query,
            batch
        ).fetchall()

        for object_id, geometry in rows:
            result[object_id] = geometry

    return result


def main():

    print("=" * 100)
    print("CARTOGEN — FAST LOCAL BUILDING EQUIVALENCE TEST")
    print("=" * 100)

    random.seed(SEED)

    con_om = sqlite3.connect(
        f"file:{OPENMAP}?mode=ro",
        uri=True
    )

    con_zs = sqlite3.connect(
        f"file:{ZOOMSTACK}?mode=ro",
        uri=True
    )

    try:

        print("\nSelecting distributed sample...")

        # We know the table has an integer primary key:
        # fid. Sample directly through the fid range instead
        # of scanning all 15 million records.

        min_fid, max_fid = con_om.execute(
            """
            SELECT MIN(fid), MAX(fid)
            FROM building
            """
        ).fetchone()

        print(
            f"FID range: {min_fid:,} - {max_fid:,}"
        )

        candidate_fids = random.sample(
            range(min_fid, max_fid + 1),
            SAMPLE_SIZE * 2
        )

        selected_ids = []

        for start in range(
            0,
            len(candidate_fids),
            BATCH_SIZE
        ):

            batch = candidate_fids[
                start:start + BATCH_SIZE
            ]

            placeholders = ",".join(
                "?" for _ in batch
            )

            rows = con_om.execute(
                f"""
                SELECT id
                FROM building
                WHERE fid IN ({placeholders})
                """,
                batch
            ).fetchall()

            selected_ids.extend(
                row[0]
                for row in rows
                if row[0] is not None
            )

            if len(selected_ids) >= SAMPLE_SIZE:
                break

        selected_ids = (
            selected_ids[:SAMPLE_SIZE]
        )

        print(
            f"Sample IDs obtained: "
            f"{len(selected_ids):,}"
        )

        print("\nReading OpenMap sample...")

        start_time = time.time()

        openmap = fetch_by_ids(
            con_om,
            "building",
            "geometry",
            selected_ids
        )

        print(
            f"Retrieved: {len(openmap):,} "
            f"({time.time() - start_time:.2f}s)"
        )

        print("\nReading Zoomstack sample...")

        start_time = time.time()

        zoomstack = fetch_by_ids(
            con_zs,
            "local_buildings",
            "geom",
            selected_ids
        )

        print(
            f"Retrieved: {len(zoomstack):,} "
            f"({time.time() - start_time:.2f}s)"
        )

        shared = (
            set(openmap)
            & set(zoomstack)
        )

        exact = sum(
            openmap[object_id]
            == zoomstack[object_id]
            for object_id in shared
        )

        missing = (
            set(openmap)
            - set(zoomstack)
        )

        result = {
            "sample_size":
                len(selected_ids),

            "openmap_found":
                len(openmap),

            "zoomstack_found":
                len(zoomstack),

            "shared_ids":
                len(shared),

            "missing_in_zoomstack":
                len(missing),

            "id_match_rate_pct":
                (
                    100
                    * len(shared)
                    / len(openmap)
                    if openmap
                    else 0
                ),

            "exact_binary_geometry_matches":
                exact,

            "exact_geometry_match_rate_pct":
                (
                    100
                    * exact
                    / len(shared)
                    if shared
                    else 0
                ),
        }

        print()
        print("=" * 100)
        print("RESULT")
        print("=" * 100)

        print(
            f"Sample size:            "
            f"{result['sample_size']:,}"
        )

        print(
            f"Shared IDs:             "
            f"{result['shared_ids']:,}"
        )

        print(
            f"Missing in Zoomstack:   "
            f"{result['missing_in_zoomstack']:,}"
        )

        print(
            f"ID match rate:          "
            f"{result['id_match_rate_pct']:.2f}%"
        )

        print(
            f"Exact geometry matches: "
            f"{result['exact_binary_geometry_matches']:,}"
        )

        print(
            f"Geometry match rate:    "
            f"{result['exact_geometry_match_rate_pct']:.2f}%"
        )

        OUTPUT.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            OUTPUT,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                result,
                file,
                indent=2
            )

        print(f"\nSaved:\n{OUTPUT}")

    finally:

        con_om.close()
        con_zs.close()


if __name__ == "__main__":
    main()