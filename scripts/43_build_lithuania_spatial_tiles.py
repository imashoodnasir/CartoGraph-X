from pathlib import Path
import json
import math
import hashlib

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


# ======================================================================
# CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WINDOWS_JSON = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "lithuania"
    / "windows"
    / "lithuania_development_windows.json"
)

VALIDATED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "validated_pairs"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "spatial_tiles"
)

TILE_GPKG_DIR = OUTPUT_DIR / "gpkg"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TILE_GPKG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CRS = "EPSG:3346"

TILE_SIZE_M = 1000.0
CONTEXT_M = 100.0

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42


CITY_CONFIG = {
    "vilnius_central": {
        "city": "Vilnius",
    },
    "kaunas_central": {
        "city": "Kaunas",
    },
    "klaipeda_central": {
        "city": "Klaipeda",
    },
}


LAYERS = [
    "source_buildings",
    "target_buildings",
    "source_roads",
    "target_roads",
]


# ======================================================================
# HELPERS
# ======================================================================

def print_header(text):
    print()
    print("=" * 120)
    print(text)
    print("=" * 120)


def print_subheader(text):
    print()
    print("-" * 120)
    print(text)
    print("-" * 120)


def stable_hash(value):
    """
    Deterministic numeric hash.
    """

    digest = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()

    return int(
        digest[:16],
        16
    )


def load_windows():
    """
    Read window definitions created by Script 32.
    """

    with open(
        WINDOWS_JSON,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    # Support either:
    # [
    #   {...},
    #   {...}
    # ]
    #
    # or:
    # {
    #   "windows": [...]
    # }

    if isinstance(data, dict):

        if "windows" in data:
            data = data["windows"]

        else:
            possible = []

            for key, value in data.items():

                if isinstance(value, dict):
                    row = dict(value)

                    if "window" not in row:
                        row["window"] = key

                    possible.append(row)

            data = possible

    return data


def get_window_name(record):
    """
    Resolve possible naming conventions.
    """

    for key in [
        "window",
        "window_id",
        "name",
        "id"
    ]:

        if key in record:
            return record[key]

    return None


def get_bbox(record):
    """
    Resolve bounding box from Script 32 JSON.
    """

    # Case 1
    if "bbox" in record:

        bbox_value = record["bbox"]

        if isinstance(
            bbox_value,
            (list, tuple)
        ):

            if len(bbox_value) == 4:

                return tuple(
                    float(v)
                    for v in bbox_value
                )

        if isinstance(
            bbox_value,
            dict
        ):

            possible_keys = [
                (
                    "xmin",
                    "ymin",
                    "xmax",
                    "ymax"
                ),
                (
                    "minx",
                    "miny",
                    "maxx",
                    "maxy"
                ),
            ]

            for keys in possible_keys:

                if all(
                    key in bbox_value
                    for key in keys
                ):

                    return tuple(
                        float(
                            bbox_value[key]
                        )
                        for key in keys
                    )

    # Case 2
    key_sets = [
        (
            "xmin",
            "ymin",
            "xmax",
            "ymax"
        ),
        (
            "minx",
            "miny",
            "maxx",
            "maxy"
        ),
    ]

    for keys in key_sets:

        if all(
            key in record
            for key in keys
        ):

            return tuple(
                float(
                    record[key]
                )
                for key in keys
            )

    raise ValueError(
        f"Could not determine bbox "
        f"from window record:\n{record}"
    )


def read_layer(
    gpkg_path,
    layer_name
):
    """
    Read a validated GeoPackage layer.
    """

    gdf = gpd.read_file(
        gpkg_path,
        layer=layer_name
    )

    if gdf.crs is None:

        raise ValueError(
            f"{layer_name} has no CRS."
        )

    if gdf.crs.to_epsg() != 3346:

        gdf = gdf.to_crs(
            CRS
        )

    return gdf


def create_grid(
    xmin,
    ymin,
    xmax,
    ymax
):
    """
    Create 1 km x 1 km core tiles.

    The city windows are currently 6 km x 6 km,
    therefore normally 36 tiles per city.
    """

    width = xmax - xmin
    height = ymax - ymin

    cols = int(
        math.ceil(
            width / TILE_SIZE_M
        )
    )

    rows = int(
        math.ceil(
            height / TILE_SIZE_M
        )
    )

    records = []

    tile_number = 0

    for row in range(rows):

        for col in range(cols):

            core_xmin = (
                xmin
                + col * TILE_SIZE_M
            )

            core_ymin = (
                ymin
                + row * TILE_SIZE_M
            )

            core_xmax = min(
                core_xmin + TILE_SIZE_M,
                xmax
            )

            core_ymax = min(
                core_ymin + TILE_SIZE_M,
                ymax
            )

            if (
                core_xmax <= core_xmin
                or
                core_ymax <= core_ymin
            ):
                continue

            tile_number += 1

            core_geom = box(
                core_xmin,
                core_ymin,
                core_xmax,
                core_ymax
            )

            context_geom = box(
                core_xmin - CONTEXT_M,
                core_ymin - CONTEXT_M,
                core_xmax + CONTEXT_M,
                core_ymax + CONTEXT_M
            )

            records.append({
                "tile_number": tile_number,
                "row": row,
                "col": col,
                "core_xmin": core_xmin,
                "core_ymin": core_ymin,
                "core_xmax": core_xmax,
                "core_ymax": core_ymax,
                "core_area_m2": (
                    core_geom.area
                ),
                "core_geometry": core_geom,
                "context_geometry": (
                    context_geom
                ),
            })

    return records


def assign_spatial_split(
    tile_records,
    city_key
):
    """
    Assign complete tiles to train,
    validation, or test.

    Assignment is deterministic.

    Tiles are NOT randomly assigned at
    object level.
    """

    n = len(tile_records)

    if n < 3:

        raise ValueError(
            "At least three tiles are "
            "required for train/val/test."
        )

    n_train = int(
        round(
            n * TRAIN_RATIO
        )
    )

    n_val = int(
        round(
            n * VAL_RATIO
        )
    )

    n_test = (
        n
        - n_train
        - n_val
    )

    # Ensure every split exists.

    n_train = max(
        1,
        n_train
    )

    n_val = max(
        1,
        n_val
    )

    n_test = max(
        1,
        n_test
    )

    # Reconcile possible rounding issue.

    while (
        n_train
        + n_val
        + n_test
        > n
    ):

        if n_train > 1:
            n_train -= 1

        elif n_val > 1:
            n_val -= 1

        else:
            n_test -= 1

    while (
        n_train
        + n_val
        + n_test
        < n
    ):

        n_train += 1

    indices = list(
        range(n)
    )

    # Deterministic pseudo-random ordering
    # based on tile identity.

    indices.sort(
        key=lambda idx: stable_hash(
            f"{SEED}_"
            f"{city_key}_"
            f"{tile_records[idx]['row']}_"
            f"{tile_records[idx]['col']}"
        )
    )

    train_indices = set(
        indices[:n_train]
    )

    val_indices = set(
        indices[
            n_train:
            n_train + n_val
        ]
    )

    test_indices = set(
        indices[
            n_train + n_val:
        ]
    )

    for idx, record in enumerate(
        tile_records
    ):

        if idx in train_indices:

            record["split"] = "train"

        elif idx in val_indices:

            record["split"] = "validation"

        elif idx in test_indices:

            record["split"] = "test"

        else:

            raise RuntimeError(
                "Tile was not assigned "
                "to a split."
            )

    return tile_records


def subset_by_context(
    gdf,
    context_geom
):
    """
    Return geometries intersecting
    contextual tile extent.

    Spatial index is used where possible.
    """

    if len(gdf) == 0:
        return gdf.copy()

    try:

        candidate_idx = list(
            gdf.sindex.intersection(
                context_geom.bounds
            )
        )

        subset = gdf.iloc[
            candidate_idx
        ].copy()

    except Exception:

        subset = gdf.copy()

    subset = subset[
        subset.geometry.intersects(
            context_geom
        )
    ].copy()

    return subset


def count_core_objects(
    gdf,
    core_geom
):
    """
    Count objects whose representative
    point lies inside the core tile.

    This prevents objects intersecting
    contextual padding from being counted
    multiple times in benchmark statistics.
    """

    if len(gdf) == 0:
        return 0

    points = (
        gdf.geometry
        .representative_point()
    )

    mask = points.within(
        core_geom
    )

    return int(
        mask.sum()
    )


def add_tile_metadata(
    gdf,
    tile_id,
    city,
    split,
    core_geom
):
    """
    Add tile information and mark whether
    each geometry belongs to the core or
    only to contextual padding.
    """

    result = gdf.copy()

    result.insert(
        0,
        "tile_id",
        tile_id
    )

    result.insert(
        1,
        "tile_city",
        city
    )

    result.insert(
        2,
        "tile_split",
        split
    )

    if len(result) > 0:

        rep = (
            result.geometry
            .representative_point()
        )

        result["in_core"] = (
            rep.within(
                core_geom
            )
        )

    else:

        result["in_core"] = pd.Series(
            dtype=bool
        )

    return result


def save_tile_geopackage(
    output_path,
    tile_record,
    layer_data,
    city,
    city_key
):
    """
    Save one tile as a GeoPackage.
    """

    if output_path.exists():
        output_path.unlink()

    core_gdf = gpd.GeoDataFrame(
        [{
            "tile_id":
                tile_record["tile_id"],

            "city":
                city,

            "city_key":
                city_key,

            "split":
                tile_record["split"],

            "row":
                tile_record["row"],

            "col":
                tile_record["col"],

            "tile_size_m":
                TILE_SIZE_M,

            "context_m":
                CONTEXT_M,
        }],
        geometry=[
            tile_record[
                "core_geometry"
            ]
        ],
        crs=CRS
    )

    context_gdf = gpd.GeoDataFrame(
        [{
            "tile_id":
                tile_record["tile_id"],

            "city":
                city,

            "split":
                tile_record["split"],
        }],
        geometry=[
            tile_record[
                "context_geometry"
            ]
        ],
        crs=CRS
    )

    core_gdf.to_file(
        output_path,
        layer="tile_core",
        driver="GPKG"
    )

    context_gdf.to_file(
        output_path,
        layer="tile_context",
        driver="GPKG"
    )

    for layer_name in LAYERS:

        gdf = layer_data[
            layer_name
        ]

        # GeoPackage cannot reliably create
        # an entirely empty geometry layer
        # through every GDAL configuration.
        # Empty layers are therefore omitted,
        # but counts remain in the manifest.

        if len(gdf) == 0:
            continue

        gdf.to_file(
            output_path,
            layer=layer_name,
            driver="GPKG"
        )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print_header(
        "CARTOGEN — BUILD LITHUANIAN "
        "SPATIAL LEARNING TILES"
    )

    print()
    print(
        f"Tile size:       "
        f"{TILE_SIZE_M / 1000:.1f} × "
        f"{TILE_SIZE_M / 1000:.1f} km"
    )

    print(
        f"Context padding: {CONTEXT_M:.0f} m"
    )

    print(
        f"Split target:    "
        f"{TRAIN_RATIO:.0%} train / "
        f"{VAL_RATIO:.0%} validation / "
        f"{TEST_RATIO:.0%} test"
    )

    print(
        f"CRS:             {CRS}"
    )

    windows = load_windows()

    window_lookup = {}

    for record in windows:

        name = get_window_name(
            record
        )

        if name is not None:
            window_lookup[name] = record

    global_tile_records = []

    city_summaries = []

    for city_key, config in (
        CITY_CONFIG.items()
    ):

        city = config["city"]

        print_header(
            f"{city_key} ({city})"
        )

        if city_key not in window_lookup:

            raise KeyError(
                f"{city_key} not found "
                f"in {WINDOWS_JSON}"
            )

        window_record = (
            window_lookup[
                city_key
            ]
        )

        xmin, ymin, xmax, ymax = (
            get_bbox(
                window_record
            )
        )

        print(
            f"Window bbox: "
            f"{xmin:,.2f}, "
            f"{ymin:,.2f}, "
            f"{xmax:,.2f}, "
            f"{ymax:,.2f}"
        )

        gpkg_path = (
            VALIDATED_DIR
            / f"{city_key}_validated.gpkg"
        )

        if not gpkg_path.exists():

            raise FileNotFoundError(
                gpkg_path
            )

        print()
        print(
            "Reading validated layers..."
        )

        layer_frames = {}

        for layer_name in LAYERS:

            layer_frames[
                layer_name
            ] = read_layer(
                gpkg_path,
                layer_name
            )

            print(
                f"  {layer_name:<20} "
                f"{len(layer_frames[layer_name]):>8,}"
            )

        # --------------------------------------------------------------
        # GRID
        # --------------------------------------------------------------

        print_subheader(
            "CREATE SPATIAL GRID"
        )

        tile_records = create_grid(
            xmin,
            ymin,
            xmax,
            ymax
        )

        tile_records = (
            assign_spatial_split(
                tile_records,
                city_key
            )
        )

        print(
            f"Tiles created: "
            f"{len(tile_records)}"
        )

        split_counts = (
            pd.Series(
                [
                    x["split"]
                    for x in tile_records
                ]
            )
            .value_counts()
            .to_dict()
        )

        print(
            f"Train:      "
            f"{split_counts.get('train', 0)}"
        )

        print(
            f"Validation: "
            f"{split_counts.get('validation', 0)}"
        )

        print(
            f"Test:       "
            f"{split_counts.get('test', 0)}"
        )

        # --------------------------------------------------------------
        # TILE EXTRACTION
        # --------------------------------------------------------------

        print_subheader(
            "EXTRACT TILE CONTENT"
        )

        city_output_dir = (
            TILE_GPKG_DIR
            / city_key
        )

        city_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for i, tile in enumerate(
            tile_records,
            start=1
        ):

            tile_id = (
                f"{city_key}_"
                f"r{tile['row']:02d}_"
                f"c{tile['col']:02d}"
            )

            tile["tile_id"] = tile_id
            tile["city"] = city
            tile["city_key"] = city_key

            core_geom = (
                tile["core_geometry"]
            )

            context_geom = (
                tile["context_geometry"]
            )

            layer_data = {}

            counts = {}

            for layer_name in LAYERS:

                subset = subset_by_context(
                    layer_frames[
                        layer_name
                    ],
                    context_geom
                )

                core_count = (
                    count_core_objects(
                        subset,
                        core_geom
                    )
                )

                subset = (
                    add_tile_metadata(
                        subset,
                        tile_id,
                        city,
                        tile["split"],
                        core_geom
                    )
                )

                layer_data[
                    layer_name
                ] = subset

                counts[
                    f"{layer_name}_context"
                ] = len(subset)

                counts[
                    f"{layer_name}_core"
                ] = core_count

            tile.update(
                counts
            )

            output_gpkg = (
                city_output_dir
                / f"{tile_id}.gpkg"
            )

            save_tile_geopackage(
                output_gpkg,
                tile,
                layer_data,
                city,
                city_key
            )

            tile["gpkg_path"] = str(
                output_gpkg
            )

            print(
                f"[{i:02d}/"
                f"{len(tile_records):02d}] "
                f"{tile_id:<35} "
                f"{tile['split']:<10} | "
                f"B "
                f"{counts['source_buildings_core']:>5,}"
                f"→"
                f"{counts['target_buildings_core']:>5,}"
                f" | R "
                f"{counts['source_roads_core']:>4,}"
                f"→"
                f"{counts['target_roads_core']:>4,}"
            )

            global_tile_records.append(
                tile
            )

        # --------------------------------------------------------------
        # CITY SUMMARY
        # --------------------------------------------------------------

        city_df = pd.DataFrame([
            {
                key: value
                for key, value in record.items()
                if key not in [
                    "core_geometry",
                    "context_geometry"
                ]
            }
            for record in tile_records
        ])

        city_csv = (
            OUTPUT_DIR
            / f"{city_key}_tile_manifest.csv"
        )

        city_df.to_csv(
            city_csv,
            index=False
        )

        city_summary = {
            "city": city,
            "city_key": city_key,
            "tile_count":
                len(tile_records),
            "train_tiles":
                int(
                    (
                        city_df["split"]
                        == "train"
                    ).sum()
                ),
            "validation_tiles":
                int(
                    (
                        city_df["split"]
                        == "validation"
                    ).sum()
                ),
            "test_tiles":
                int(
                    (
                        city_df["split"]
                        == "test"
                    ).sum()
                ),
            "source_buildings_core":
                int(
                    city_df[
                        "source_buildings_core"
                    ].sum()
                ),
            "target_buildings_core":
                int(
                    city_df[
                        "target_buildings_core"
                    ].sum()
                ),
            "source_roads_core":
                int(
                    city_df[
                        "source_roads_core"
                    ].sum()
                ),
            "target_roads_core":
                int(
                    city_df[
                        "target_roads_core"
                    ].sum()
                ),
        }

        city_summaries.append(
            city_summary
        )

        print()
        print(
            f"Manifest:\n{city_csv}"
        )

    # ==================================================================
    # GLOBAL MANIFEST
    # ==================================================================

    print_header(
        "GLOBAL LITHUANIAN TILE DATASET"
    )

    manifest_records = []

    for record in global_tile_records:

        clean_record = {
            key: value
            for key, value in record.items()
            if key not in [
                "core_geometry",
                "context_geometry"
            ]
        }

        manifest_records.append(
            clean_record
        )

    global_df = pd.DataFrame(
        manifest_records
    )

    global_csv = (
        OUTPUT_DIR
        / "lithuania_spatial_tile_manifest.csv"
    )

    global_df.to_csv(
        global_csv,
        index=False
    )

    # ------------------------------------------------------------------
    # SPLIT SUMMARY
    # ------------------------------------------------------------------

    summary_rows = []

    for split in [
        "train",
        "validation",
        "test"
    ]:

        part = global_df[
            global_df["split"]
            == split
        ]

        summary_rows.append({
            "split": split,
            "tiles": len(part),
            "source_buildings":
                int(
                    part[
                        "source_buildings_core"
                    ].sum()
                ),
            "target_buildings":
                int(
                    part[
                        "target_buildings_core"
                    ].sum()
                ),
            "source_roads":
                int(
                    part[
                        "source_roads_core"
                    ].sum()
                ),
            "target_roads":
                int(
                    part[
                        "target_roads_core"
                    ].sum()
                ),
        })

    split_df = pd.DataFrame(
        summary_rows
    )

    split_csv = (
        OUTPUT_DIR
        / "lithuania_spatial_split_summary.csv"
    )

    split_df.to_csv(
        split_csv,
        index=False
    )

    print()
    print(
        split_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # CITY SUMMARY
    # ------------------------------------------------------------------

    city_summary_df = pd.DataFrame(
        city_summaries
    )

    city_summary_csv = (
        OUTPUT_DIR
        / "lithuania_tile_city_summary.csv"
    )

    city_summary_df.to_csv(
        city_summary_csv,
        index=False
    )

    print()
    print("-" * 120)
    print("CITY SUMMARY")
    print("-" * 120)

    print(
        city_summary_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # SPATIAL TILE INDEX
    # ------------------------------------------------------------------

    tile_index_records = []

    for record in global_tile_records:

        tile_index_records.append({
            "tile_id":
                record["tile_id"],
            "city":
                record["city"],
            "city_key":
                record["city_key"],
            "split":
                record["split"],
            "row":
                record["row"],
            "col":
                record["col"],
            "source_buildings_core":
                record[
                    "source_buildings_core"
                ],
            "target_buildings_core":
                record[
                    "target_buildings_core"
                ],
            "source_roads_core":
                record[
                    "source_roads_core"
                ],
            "target_roads_core":
                record[
                    "target_roads_core"
                ],
            "geometry":
                record[
                    "core_geometry"
                ],
        })

    tile_index_gdf = (
        gpd.GeoDataFrame(
            tile_index_records,
            geometry="geometry",
            crs=CRS
        )
    )

    tile_index_path = (
        OUTPUT_DIR
        / "lithuania_spatial_tile_index.gpkg"
    )

    if tile_index_path.exists():
        tile_index_path.unlink()

    tile_index_gdf.to_file(
        tile_index_path,
        layer="tiles",
        driver="GPKG"
    )

    # ------------------------------------------------------------------
    # JSON MANIFEST
    # ------------------------------------------------------------------

    json_manifest = {
        "dataset":
            "CartoGen Lithuania",
        "crs":
            CRS,
        "tile_size_m":
            TILE_SIZE_M,
        "context_m":
            CONTEXT_M,
        "seed":
            SEED,
        "split_target": {
            "train":
                TRAIN_RATIO,
            "validation":
                VAL_RATIO,
            "test":
                TEST_RATIO,
        },
        "cities":
            city_summaries,
        "total_tiles":
            len(global_df),
        "split_summary":
            summary_rows,
    }

    json_path = (
        OUTPUT_DIR
        / "lithuania_spatial_tile_manifest.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            json_manifest,
            f,
            indent=2
        )

    # ==================================================================
    # VALIDATION
    # ==================================================================

    print_header(
        "INITIAL SPLIT VALIDATION"
    )

    duplicate_tiles = (
        global_df[
            "tile_id"
        ].duplicated().sum()
    )

    print(
        f"Duplicate tile IDs: "
        f"{duplicate_tiles}"
    )

    split_values = set(
        global_df["split"]
    )

    print(
        f"Splits found: "
        f"{sorted(split_values)}"
    )

    expected_splits = {
        "train",
        "validation",
        "test"
    }

    if duplicate_tiles == 0:
        print(
            "[PASS] Tile IDs are unique."
        )
    else:
        print(
            "[FAIL] Duplicate tile IDs detected."
        )

    if split_values == expected_splits:
        print(
            "[PASS] All three splits exist."
        )
    else:
        print(
            "[FAIL] Missing split."
        )

    # Core polygons themselves must not overlap.
    overlap_count = 0

    for city_key in CITY_CONFIG:

        city_tiles = (
            tile_index_gdf[
                tile_index_gdf[
                    "city_key"
                ]
                == city_key
            ]
            .reset_index(
                drop=True
            )
        )

        for i in range(
            len(city_tiles)
        ):

            geom_i = (
                city_tiles
                .geometry
                .iloc[i]
            )

            for j in range(
                i + 1,
                len(city_tiles)
            ):

                geom_j = (
                    city_tiles
                    .geometry
                    .iloc[j]
                )

                intersection = (
                    geom_i.intersection(
                        geom_j
                    )
                )

                # Shared boundaries are fine.
                if (
                    not intersection.is_empty
                    and intersection.area > 0
                ):
                    overlap_count += 1

    print(
        f"Positive-area core overlaps: "
        f"{overlap_count}"
    )

    if overlap_count == 0:
        print(
            "[PASS] Core tiles do not overlap."
        )
    else:
        print(
            "[FAIL] Core tile overlap detected."
        )

    print_header(
        "SCRIPT 43 COMPLETE"
    )

    print()
    print(
        "Global tile manifest:"
    )
    print(
        global_csv
    )

    print()
    print(
        "Split summary:"
    )
    print(
        split_csv
    )

    print()
    print(
        "City summary:"
    )
    print(
        city_summary_csv
    )

    print()
    print(
        "Spatial tile index:"
    )
    print(
        tile_index_path
    )

    print()
    print(
        "JSON manifest:"
    )
    print(
        json_path
    )

    print()
    print(
        "Individual tile GeoPackages:"
    )
    print(
        TILE_GPKG_DIR
    )


if __name__ == "__main__":
    main()