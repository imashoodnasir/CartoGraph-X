from pathlib import Path
import json
import math

import pandas as pd
import geopandas as gpd
from shapely.geometry import box


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "os"
    / "development_windows"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cartogen"
    / "tiles"
)

MANIFEST_CSV = (
    OUTPUT_ROOT
    / "tile_manifest.csv"
)

MANIFEST_JSON = (
    OUTPUT_ROOT
    / "tile_manifest.json"
)

WINDOWS = [
    "london_central",
    "manchester_central",
    "birmingham_central",
]

CRS = "EPSG:27700"

CORE_SIZE_M = 1000.0

CONTEXT_M = 100.0


LAYERS = [
    "local_buildings",
    "district_buildings",
    "roads_local",
    "roads_regional",
]


# ============================================================
# HELPERS
# ============================================================

def clip_layer(
    gdf,
    geometry
):

    if len(gdf) == 0:
        return gdf.copy()

    candidate_idx = list(
        gdf.sindex.intersection(
            geometry.bounds
        )
    )

    if not candidate_idx:
        return gdf.iloc[0:0].copy()

    subset = (
        gdf.iloc[
            candidate_idx
        ]
        .copy()
    )

    subset = subset[
        subset.geometry.intersects(
            geometry
        )
    ].copy()

    if len(subset) == 0:
        return subset

    subset[
        "geometry"
    ] = subset.geometry.intersection(
        geometry
    )

    subset = subset[
        ~subset.geometry.is_empty
    ].copy()

    subset = subset[
        subset.geometry.notna()
    ].copy()

    subset.reset_index(
        drop=True,
        inplace=True
    )

    return subset


def count_features(
    gdf
):

    return int(
        len(gdf)
    )


def geometry_measure(
    gdf
):

    if len(gdf) == 0:
        return 0.0

    geom_types = set(
        gdf.geometry.geom_type
    )

    polygon_types = {
        "Polygon",
        "MultiPolygon",
    }

    line_types = {
        "LineString",
        "MultiLineString",
    }

    if geom_types.issubset(
        polygon_types
    ):

        return float(
            gdf.geometry.area.sum()
        )

    if geom_types.issubset(
        line_types
    ):

        return float(
            gdf.geometry.length.sum()
        )

    return 0.0


# ============================================================
# TILE GENERATION
# ============================================================

def create_tiles_for_window(
    window
):

    print()
    print("=" * 120)
    print(window)
    print("=" * 120)

    source_file = (
        SOURCE_ROOT
        / f"{window}.gpkg"
    )

    if not source_file.exists():

        raise FileNotFoundError(
            source_file
        )

    # --------------------------------------------------------
    # Read required layers
    # --------------------------------------------------------

    layers = {}

    for layer in LAYERS:

        print(
            f"Reading {layer}..."
        )

        gdf = gpd.read_file(
            source_file,
            layer=layer,
            engine="pyogrio"
        )

        if gdf.crs is None:

            raise RuntimeError(
                f"{layer} has no CRS."
            )

        if gdf.crs.to_epsg() != 27700:

            raise RuntimeError(
                f"{layer} is not EPSG:27700."
            )

        layers[layer] = gdf

        print(
            f"  {len(gdf):,} features"
        )

    # --------------------------------------------------------
    # Use local buildings/roads to recover common window
    # --------------------------------------------------------

    bounds = (
        layers[
            "local_buildings"
        ].total_bounds
    )

    min_x, min_y, max_x, max_y = bounds

    # Align to 1 km grid inside the extracted window.
    start_x = (
        math.ceil(
            min_x / CORE_SIZE_M
        )
        * CORE_SIZE_M
    )

    start_y = (
        math.ceil(
            min_y / CORE_SIZE_M
        )
        * CORE_SIZE_M
    )

    end_x = (
        math.floor(
            max_x / CORE_SIZE_M
        )
        * CORE_SIZE_M
    )

    end_y = (
        math.floor(
            max_y / CORE_SIZE_M
        )
        * CORE_SIZE_M
    )

    print()
    print(
        f"Aligned grid extent: "
        f"{start_x:.0f}, {start_y:.0f} -> "
        f"{end_x:.0f}, {end_y:.0f}"
    )

    city_output = (
        OUTPUT_ROOT
        / window
    )

    city_output.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    row_id = 0

    y = start_y

    while (
        y + CORE_SIZE_M
        <= end_y
    ):

        col_id = 0

        x = start_x

        while (
            x + CORE_SIZE_M
            <= end_x
        ):

            tile_id = (
                f"{window}_"
                f"r{row_id:02d}_"
                f"c{col_id:02d}"
            )

            core_geom = box(
                x,
                y,
                x + CORE_SIZE_M,
                y + CORE_SIZE_M
            )

            context_geom = box(
                x - CONTEXT_M,
                y - CONTEXT_M,
                x + CORE_SIZE_M + CONTEXT_M,
                y + CORE_SIZE_M + CONTEXT_M
            )

            print()
            print(
                f"Tile: {tile_id}"
            )

            # ------------------------------------------------
            # Extract context features
            # ------------------------------------------------

            context_layers = {}

            for layer_name, gdf in (
                layers.items()
            ):

                clipped = clip_layer(
                    gdf,
                    context_geom
                )

                context_layers[
                    layer_name
                ] = clipped

            # ------------------------------------------------
            # Extract core features independently
            # ------------------------------------------------

            core_layers = {}

            for layer_name, gdf in (
                layers.items()
            ):

                clipped = clip_layer(
                    gdf,
                    core_geom
                )

                core_layers[
                    layer_name
                ] = clipped

            # ------------------------------------------------
            # Skip genuinely empty tiles
            # ------------------------------------------------

            if (
                len(
                    core_layers[
                        "local_buildings"
                    ]
                ) == 0
                and
                len(
                    core_layers[
                        "roads_local"
                    ]
                ) == 0
            ):

                print(
                    "  Skipped: empty core."
                )

                x += CORE_SIZE_M
                col_id += 1
                continue

            # ------------------------------------------------
            # Save tile GeoPackage
            # ------------------------------------------------

            output_file = (
                city_output
                / f"{tile_id}.gpkg"
            )

            if output_file.exists():

                output_file.unlink()

            # Context is what the model receives.

            for layer_name, gdf in (
                context_layers.items()
            ):

                output_layer = (
                    f"context_{layer_name}"
                )

                if len(gdf) > 0:

                    gdf.to_file(
                        output_file,
                        layer=output_layer,
                        driver="GPKG",
                        engine="pyogrio"
                    )

            # Core layers are retained for loss/evaluation.

            for layer_name, gdf in (
                core_layers.items()
            ):

                output_layer = (
                    f"core_{layer_name}"
                )

                if len(gdf) > 0:

                    gdf.to_file(
                        output_file,
                        layer=output_layer,
                        driver="GPKG",
                        engine="pyogrio"
                    )

            # Save core/context boundaries.

            boundary_gdf = (
                gpd.GeoDataFrame(
                    {
                        "tile_id": [
                            tile_id,
                            tile_id,
                        ],

                        "boundary_type": [
                            "core",
                            "context",
                        ],
                    },

                    geometry=[
                        core_geom,
                        context_geom,
                    ],

                    crs=CRS
                )
            )

            boundary_gdf.to_file(
                output_file,
                layer="tile_boundaries",
                driver="GPKG",
                engine="pyogrio"
            )

            # ------------------------------------------------
            # Manifest statistics
            # ------------------------------------------------

            record = {

                "tile_id":
                    tile_id,

                "window":
                    window,

                "row":
                    row_id,

                "column":
                    col_id,

                "core_xmin":
                    x,

                "core_ymin":
                    y,

                "core_xmax":
                    x + CORE_SIZE_M,

                "core_ymax":
                    y + CORE_SIZE_M,

                "context_xmin":
                    x - CONTEXT_M,

                "context_ymin":
                    y - CONTEXT_M,

                "context_xmax":
                    x
                    + CORE_SIZE_M
                    + CONTEXT_M,

                "context_ymax":
                    y
                    + CORE_SIZE_M
                    + CONTEXT_M,

                "core_local_buildings":
                    count_features(
                        core_layers[
                            "local_buildings"
                        ]
                    ),

                "core_district_buildings":
                    count_features(
                        core_layers[
                            "district_buildings"
                        ]
                    ),

                "core_local_roads":
                    count_features(
                        core_layers[
                            "roads_local"
                        ]
                    ),

                "core_regional_roads":
                    count_features(
                        core_layers[
                            "roads_regional"
                        ]
                    ),

                "context_local_buildings":
                    count_features(
                        context_layers[
                            "local_buildings"
                        ]
                    ),

                "context_district_buildings":
                    count_features(
                        context_layers[
                            "district_buildings"
                        ]
                    ),

                "context_local_roads":
                    count_features(
                        context_layers[
                            "roads_local"
                        ]
                    ),

                "context_regional_roads":
                    count_features(
                        context_layers[
                            "roads_regional"
                        ]
                    ),

                "core_local_building_area_m2":
                    geometry_measure(
                        core_layers[
                            "local_buildings"
                        ]
                    ),

                "core_district_building_area_m2":
                    geometry_measure(
                        core_layers[
                            "district_buildings"
                        ]
                    ),

                "core_local_road_length_m":
                    geometry_measure(
                        core_layers[
                            "roads_local"
                        ]
                    ),

                "core_regional_road_length_m":
                    geometry_measure(
                        core_layers[
                            "roads_regional"
                        ]
                    ),

                "file":
                    str(
                        output_file
                    ),
            }

            records.append(
                record
            )

            print(
                "  "
                f"Buildings "
                f"{record['core_local_buildings']:,}"
                f" -> "
                f"{record['core_district_buildings']:,}"
                f" | Roads "
                f"{record['core_local_roads']:,}"
                f" -> "
                f"{record['core_regional_roads']:,}"
            )

            x += CORE_SIZE_M
            col_id += 1

        y += CORE_SIZE_M
        row_id += 1

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 120)
    print(
        "CARTOGEN — CREATE "
        "MODEL TRAINING TILES"
    )
    print("=" * 120)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    all_records = []

    for window in WINDOWS:

        records = (
            create_tiles_for_window(
                window
            )
        )

        all_records.extend(
            records
        )

    manifest = pd.DataFrame(
        all_records
    )

    manifest.to_csv(
        MANIFEST_CSV,
        index=False
    )

    with open(
        MANIFEST_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest.to_dict(
                orient="records"
            ),
            file,
            indent=2
        )

    print()
    print("=" * 120)
    print("DATASET SUMMARY")
    print("=" * 120)

    print(
        f"Total tiles: "
        f"{len(manifest):,}"
    )

    print()

    city_counts = (
        manifest[
            "window"
        ]
        .value_counts()
    )

    for city, count in (
        city_counts.items()
    ):

        print(
            f"{city:<30}"
            f"{count:>5}"
        )

    print()
    print(
        "FEATURE TOTALS"
    )
    print("-" * 120)

    columns = [
        "core_local_buildings",
        "core_district_buildings",
        "core_local_roads",
        "core_regional_roads",
    ]

    for column in columns:

        print(
            f"{column:<40}"
            f"{int(manifest[column].sum()):>12,}"
        )

    print()
    print("=" * 120)
    print("SCRIPT 27 COMPLETE")
    print("=" * 120)

    print(
        f"Manifest CSV:\n"
        f"{MANIFEST_CSV}"
    )

    print(
        f"\nManifest JSON:\n"
        f"{MANIFEST_JSON}"
    )


if __name__ == "__main__":
    main()