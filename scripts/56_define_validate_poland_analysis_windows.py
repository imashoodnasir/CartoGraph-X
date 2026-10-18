from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import box, Point
from shapely.ops import unary_union

warnings.filterwarnings("ignore")


# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

ROOT = Path(r"D:\Research\CartoGen")

BDOT_ROOT = ROOT / "data" / "processed" / "poland" / "bdot10k"
BDOO_ROOT = ROOT / "data" / "processed" / "poland" / "bdoo"

OUT_ROOT = ROOT / "data" / "processed" / "poland" / "analysis_windows"
FIG_ROOT = OUT_ROOT / "figures"

OUT_ROOT.mkdir(parents=True, exist_ok=True)
FIG_ROOT.mkdir(parents=True, exist_ok=True)

CRS = "EPSG:2180"

WINDOW_SIZE_M = 6000.0
HALF_WINDOW = WINDOW_SIZE_M / 2.0

# City centres in WGS84.
# These are transformed to EPSG:2180.
CITY_CENTRES = {
    "warsaw": {
        "city": "Warsaw",
        "lat": 52.2297,
        "lon": 21.0122,
        "purpose": "primary_development",
    },
    "krakow": {
        "city": "Krakow",
        "lat": 50.0647,
        "lon": 19.9450,
        "purpose": "secondary_development",
    },
    "gdansk": {
        "city": "Gdansk",
        "lat": 54.3520,
        "lon": 18.6466,
        "purpose": "secondary_development",
    },
}


# ==================================================================================================
# UTILITIES
# ==================================================================================================

def banner(text):
    print()
    print("=" * 120)
    print(text)
    print("=" * 120)


def section(text):
    print()
    print("-" * 120)
    print(text)
    print("-" * 120)


def find_gpkg(directory, keywords, exclude_keywords=None):
    """
    Robustly find a GeoPackage by filename/layer hints.
    """

    directory = Path(directory)

    if not directory.exists():
        return None

    exclude_keywords = exclude_keywords or []

    candidates = list(directory.rglob("*.gpkg"))

    scored = []

    for path in candidates:
        name = path.name.lower()
        full = str(path).lower()

        if any(x.lower() in full for x in exclude_keywords):
            continue

        score = 0

        for keyword in keywords:
            keyword = keyword.lower()

            if keyword in name:
                score += 5

            if keyword in full:
                score += 1

        if score > 0:
            scored.append((score, path))

    if not scored:
        return None

    scored.sort(key=lambda x: (-x[0], len(str(x[1]))))

    return scored[0][1]


def list_layers(path):
    try:
        import pyogrio
        info = pyogrio.list_layers(path)
        return [x[0] for x in info]
    except Exception:
        return []


def choose_layer(path, role):
    """
    Find likely layer inside a GeoPackage.
    """

    layers = list_layers(path)

    if not layers:
        return None

    role = role.lower()

    rules = {
        "source_buildings": [
            "building",
            "buildings",
            "ot_bubd_a",
            "bubd",
        ],
        "source_roads": [
            "road",
            "roads",
            "ot_skdr_l",
            "skdr",
        ],
        "target_builtup": [
            "ptzb",
            "builtup",
            "built_up",
            "ot_ptzb_a",
        ],
        "target_roads": [
            "road",
            "roads",
            "ot_skdr_l",
            "skdr",
        ],
    }

    wanted = rules.get(role, [])

    scored = []

    for layer in layers:
        lname = layer.lower()

        score = 0

        for keyword in wanted:
            if keyword == lname:
                score += 20
            elif keyword in lname:
                score += 5

        if score > 0:
            scored.append((score, layer))

    if scored:
        scored.sort(reverse=True)
        return scored[0][1]

    if len(layers) == 1:
        return layers[0]

    return layers[0]


def locate_city_inputs(city_key):
    """
    Locate BDOT10k + BDOO files without hard-coding the exact
    filenames produced by earlier scripts.
    """

    source_city_dir = BDOT_ROOT / city_key

    source_buildings = find_gpkg(
        source_city_dir,
        ["building", "buildings", "bubd"]
    )

    source_roads = find_gpkg(
        source_city_dir,
        ["road", "roads", "skdr"]
    )

    # Search all BDOO output recursively.
    target_builtup = find_gpkg(
        BDOO_ROOT,
        [city_key, "ptzb"],
        exclude_keywords=["validation", "analysis_windows"]
    )

    target_roads = find_gpkg(
        BDOO_ROOT,
        [city_key, "road"],
        exclude_keywords=["validation", "analysis_windows"]
    )

    # Some earlier scripts may have stored all BDOO layers
    # for a city in a single GeoPackage.
    if target_builtup is None:
        target_builtup = find_gpkg(
            BDOO_ROOT,
            [city_key],
            exclude_keywords=["validation", "analysis_windows"]
        )

    if target_roads is None:
        target_roads = find_gpkg(
            BDOO_ROOT,
            [city_key],
            exclude_keywords=["validation", "analysis_windows"]
        )

    return {
        "source_buildings": source_buildings,
        "source_roads": source_roads,
        "target_builtup": target_builtup,
        "target_roads": target_roads,
    }


def read_role(path, role):
    if path is None:
        raise FileNotFoundError(
            f"Could not locate input file for role: {role}"
        )

    layer = choose_layer(path, role)

    print(f"  File:  {path}")
    print(f"  Layer: {layer}")

    if layer:
        gdf = gpd.read_file(path, layer=layer)
    else:
        gdf = gpd.read_file(path)

    if gdf.crs is None:
        raise RuntimeError(
            f"{role}: CRS missing in {path}"
        )

    if gdf.crs.to_epsg() != 2180:
        print(
            f"  Reprojecting {gdf.crs} -> EPSG:2180"
        )
        gdf = gdf.to_crs(CRS)

    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    return gdf


def make_valid_safe(gdf):
    if len(gdf) == 0:
        return gdf

    invalid = ~gdf.geometry.is_valid

    if invalid.any():
        try:
            gdf.loc[invalid, "geometry"] = (
                gdf.loc[invalid, "geometry"].make_valid()
            )
        except Exception:
            gdf.loc[invalid, "geometry"] = (
                gdf.loc[invalid, "geometry"].buffer(0)
            )

    return gdf


def clip_to_window(gdf, window_geom):
    if len(gdf) == 0:
        return gdf.copy()

    xmin, ymin, xmax, ymax = window_geom.bounds

    subset = gdf.cx[xmin:xmax, ymin:ymax].copy()

    if len(subset) == 0:
        return subset

    subset = subset[
        subset.geometry.intersects(window_geom)
    ].copy()

    subset["geometry"] = subset.geometry.intersection(window_geom)

    subset = subset[
        subset.geometry.notna()
        & ~subset.geometry.is_empty
    ].copy()

    return subset


def geometry_measure(gdf, family):
    if len(gdf) == 0:
        return 0.0

    if family == "polygon":
        return float(gdf.geometry.area.sum())

    return float(gdf.geometry.length.sum())


def build_city_window(lat, lon):
    p = gpd.GeoSeries(
        [Point(lon, lat)],
        crs="EPSG:4326"
    ).to_crs(CRS).iloc[0]

    x = p.x
    y = p.y

    geom = box(
        x - HALF_WINDOW,
        y - HALF_WINDOW,
        x + HALF_WINDOW,
        y + HALF_WINDOW,
    )

    return x, y, geom


def nearest_distance_stats(source_roads, target_roads):
    """
    Compute nearest target->source geometric distances.

    GeoPandas sjoin_nearest is substantially faster than
    pairwise distance calculation.
    """

    if len(source_roads) == 0 or len(target_roads) == 0:
        return {
            "count": 0,
            "mean_m": np.nan,
            "median_m": np.nan,
            "p95_m": np.nan,
            "max_m": np.nan,
            "within_1m_pct": np.nan,
            "within_5m_pct": np.nan,
            "within_10m_pct": np.nan,
            "within_25m_pct": np.nan,
            "within_50m_pct": np.nan,
        }

    src = source_roads[["geometry"]].copy()
    tgt = target_roads[["geometry"]].copy()

    tgt["_target_id"] = np.arange(len(tgt))

    nearest = gpd.sjoin_nearest(
        tgt,
        src,
        how="left",
        distance_col="_distance_m",
    )

    # A target can tie with multiple source objects.
    # Retain minimum distance per target.
    distances = (
        nearest.groupby("_target_id")["_distance_m"]
        .min()
        .dropna()
        .astype(float)
    )

    if len(distances) == 0:
        return {
            "count": 0,
            "mean_m": np.nan,
            "median_m": np.nan,
            "p95_m": np.nan,
            "max_m": np.nan,
            "within_1m_pct": np.nan,
            "within_5m_pct": np.nan,
            "within_10m_pct": np.nan,
            "within_25m_pct": np.nan,
            "within_50m_pct": np.nan,
        }

    return {
        "count": int(len(distances)),
        "mean_m": float(distances.mean()),
        "median_m": float(distances.median()),
        "p95_m": float(np.percentile(distances, 95)),
        "max_m": float(distances.max()),
        "within_1m_pct": float((distances <= 1).mean() * 100),
        "within_5m_pct": float((distances <= 5).mean() * 100),
        "within_10m_pct": float((distances <= 10).mean() * 100),
        "within_25m_pct": float((distances <= 25).mean() * 100),
        "within_50m_pct": float((distances <= 50).mean() * 100),
    }


def calculate_building_builtup_overlap(buildings, builtup):
    if len(buildings) == 0 or len(builtup) == 0:
        return {
            "source_buildings": len(buildings),
            "target_builtup": len(builtup),
            "buildings_intersecting": 0,
            "buildings_intersecting_pct": 0.0,
            "building_area_km2": 0.0,
            "builtup_area_km2": 0.0,
            "intersection_area_km2": 0.0,
            "building_area_covered_pct": 0.0,
            "builtup_area_occupied_pct": 0.0,
        }

    builtup_union = unary_union(builtup.geometry.values)

    intersects = buildings.geometry.intersects(builtup_union)

    source_area = float(buildings.geometry.area.sum())
    target_area = float(builtup.geometry.area.sum())

    intersection_area = float(
        buildings.geometry.intersection(
            builtup_union
        ).area.sum()
    )

    return {
        "source_buildings": int(len(buildings)),
        "target_builtup": int(len(builtup)),
        "buildings_intersecting": int(intersects.sum()),
        "buildings_intersecting_pct": (
            float(intersects.mean() * 100)
        ),
        "building_area_km2": source_area / 1e6,
        "builtup_area_km2": target_area / 1e6,
        "intersection_area_km2": intersection_area / 1e6,
        "building_area_covered_pct": (
            intersection_area / source_area * 100
            if source_area > 0 else np.nan
        ),
        "builtup_area_occupied_pct": (
            intersection_area / target_area * 100
            if target_area > 0 else np.nan
        ),
    }


def render_validation(
    city_name,
    city_key,
    window_geom,
    source_buildings,
    target_builtup,
    source_roads,
    target_roads,
):
    """
    Publication-oriented diagnostic figure.
    """

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    # ----------------------------------------------------------------------------------------------
    # Buildings
    # ----------------------------------------------------------------------------------------------

    ax = axes[0]

    if len(target_builtup):
        target_builtup.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=1.1,
        )

    if len(source_buildings):
        source_buildings.plot(
            ax=ax,
            facecolor="0.75",
            edgecolor="0.35",
            linewidth=0.15,
        )

    ax.set_title(
        f"{city_name}\nBDOT10k buildings + BDOO built-up"
    )

    # ----------------------------------------------------------------------------------------------
    # Roads
    # ----------------------------------------------------------------------------------------------

    ax = axes[1]

    if len(source_roads):
        source_roads.plot(
            ax=ax,
            linewidth=0.55,
            alpha=0.55,
        )

    if len(target_roads):
        target_roads.plot(
            ax=ax,
            linewidth=1.15,
        )

    ax.set_title(
        f"{city_name}\nBDOT10k → BDOO roads"
    )

    # ----------------------------------------------------------------------------------------------
    # Combined
    # ----------------------------------------------------------------------------------------------

    ax = axes[2]

    if len(target_builtup):
        target_builtup.plot(
            ax=ax,
            facecolor="0.92",
            edgecolor="0.45",
            linewidth=0.5,
        )

    if len(source_buildings):
        source_buildings.plot(
            ax=ax,
            facecolor="0.45",
            edgecolor="none",
            alpha=0.65,
        )

    if len(source_roads):
        source_roads.plot(
            ax=ax,
            linewidth=0.35,
            alpha=0.40,
        )

    if len(target_roads):
        target_roads.plot(
            ax=ax,
            linewidth=1.0,
        )

    ax.set_title(
        f"{city_name}\nCommon 6 × 6 km analysis window"
    )

    xmin, ymin, xmax, ymax = window_geom.bounds

    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_axis_off()

    plt.tight_layout()

    png_path = FIG_ROOT / f"{city_key}_source_target_window.png"
    pdf_path = FIG_ROOT / f"{city_key}_source_target_window.pdf"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")

    return png_path, pdf_path


# ==================================================================================================
# MAIN
# ==================================================================================================

def main():

    banner(
        "CARTOGEN — DEFINE AND VALIDATE POLISH ANALYSIS WINDOWS"
    )

    window_records = []
    summary_records = []
    distance_records = []

    for city_key, cfg in CITY_CENTRES.items():

        banner(
            f"{city_key.upper()} ({cfg['city']})"
        )

        # ==========================================================================================
        # Locate input datasets
        # ==========================================================================================

        section("LOCATING INPUT DATA")

        inputs = locate_city_inputs(city_key)

        for role, path in inputs.items():
            print(f"{role:<25}: {path}")

        missing = [
            role
            for role, path in inputs.items()
            if path is None
        ]

        if missing:
            print()
            print(
                "[FAIL] Missing input datasets: "
                + ", ".join(missing)
            )

            print(
                "Check the BDOO output directory produced by "
                "Scripts 52–55."
            )

            continue

        # ==========================================================================================
        # Read
        # ==========================================================================================

        section("READING SOURCE / TARGET LAYERS")

        print("\nsource_buildings")
        source_buildings = read_role(
            inputs["source_buildings"],
            "source_buildings"
        )

        print("\nsource_roads")
        source_roads = read_role(
            inputs["source_roads"],
            "source_roads"
        )

        print("\ntarget_builtup")
        target_builtup = read_role(
            inputs["target_builtup"],
            "target_builtup"
        )

        print("\ntarget_roads")
        target_roads = read_role(
            inputs["target_roads"],
            "target_roads"
        )

        target_builtup = make_valid_safe(target_builtup)
        source_buildings = make_valid_safe(source_buildings)

        # ==========================================================================================
        # Create controlled window
        # ==========================================================================================

        section("CONTROLLED 6 × 6 KM WINDOW")

        centre_x, centre_y, window_geom = build_city_window(
            cfg["lat"],
            cfg["lon"]
        )

        xmin, ymin, xmax, ymax = window_geom.bounds

        print(
            f"WGS84 centre: ({cfg['lat']:.6f}, "
            f"{cfg['lon']:.6f})"
        )

        print(
            f"EPSG:2180 centre: "
            f"({centre_x:,.2f}, {centre_y:,.2f})"
        )

        print(
            f"BBox: {xmin:,.2f}, {ymin:,.2f}, "
            f"{xmax:,.2f}, {ymax:,.2f}"
        )

        print(
            f"Area: {window_geom.area / 1e6:.2f} km²"
        )

        # ==========================================================================================
        # Clip all layers to identical extent
        # ==========================================================================================

        section("EXTRACTING COMMON ANALYSIS WINDOW")

        sb = clip_to_window(
            source_buildings,
            window_geom
        )

        sr = clip_to_window(
            source_roads,
            window_geom
        )

        tb = clip_to_window(
            target_builtup,
            window_geom
        )

        tr = clip_to_window(
            target_roads,
            window_geom
        )

        print(f"Source buildings: {len(sb):,}")
        print(f"Target built-up:   {len(tb):,}")
        print(f"Source roads:      {len(sr):,}")
        print(f"Target roads:      {len(tr):,}")

        # ==========================================================================================
        # Basic measurements
        # ==========================================================================================

        section("COMMON-WINDOW QUANTITATIVE VALIDATION")

        source_building_area = geometry_measure(
            sb,
            "polygon"
        ) / 1e6

        target_builtup_area = geometry_measure(
            tb,
            "polygon"
        ) / 1e6

        source_road_length = geometry_measure(
            sr,
            "line"
        ) / 1000

        target_road_length = geometry_measure(
            tr,
            "line"
        ) / 1000

        road_feature_retention = (
            len(tr) / len(sr) * 100
            if len(sr) else np.nan
        )

        road_length_retention = (
            target_road_length / source_road_length * 100
            if source_road_length > 0
            else np.nan
        )

        print()
        print("BUILDINGS / BUILT-UP")
        print(
            f"  Source buildings:      {len(sb):,}"
        )
        print(
            f"  Target built-up areas: {len(tb):,}"
        )
        print(
            f"  Building area:         "
            f"{source_building_area:.3f} km²"
        )
        print(
            f"  Built-up area:         "
            f"{target_builtup_area:.3f} km²"
        )

        print()
        print("ROADS")
        print(
            f"  Source roads:          {len(sr):,}"
        )
        print(
            f"  Target roads:          {len(tr):,}"
        )
        print(
            f"  Feature retention:     "
            f"{road_feature_retention:.2f}%"
        )
        print(
            f"  Source length:         "
            f"{source_road_length:.2f} km"
        )
        print(
            f"  Target length:         "
            f"{target_road_length:.2f} km"
        )
        print(
            f"  Length retention:      "
            f"{road_length_retention:.2f}%"
        )

        # ==========================================================================================
        # Built-up relationship
        # ==========================================================================================

        section("BUILDING → BUILT-UP RELATIONSHIP")

        builtup_stats = calculate_building_builtup_overlap(
            sb,
            tb
        )

        print(
            f"Buildings intersecting BDOO built-up: "
            f"{builtup_stats['buildings_intersecting']:,} / "
            f"{builtup_stats['source_buildings']:,}"
        )

        print(
            f"Building intersection rate: "
            f"{builtup_stats['buildings_intersecting_pct']:.2f}%"
        )

        print(
            f"Building area covered by built-up: "
            f"{builtup_stats['building_area_covered_pct']:.2f}%"
        )

        print(
            f"Built-up area occupied by buildings: "
            f"{builtup_stats['builtup_area_occupied_pct']:.2f}%"
        )

        # ==========================================================================================
        # Road proximity
        # ==========================================================================================

        section("ROAD SOURCE → TARGET GEOMETRIC ALIGNMENT")

        distance_stats = nearest_distance_stats(
            sr,
            tr
        )

        print(
            f"Target roads evaluated: "
            f"{distance_stats['count']:,}"
        )

        print(
            f"Mean nearest distance:   "
            f"{distance_stats['mean_m']:.3f} m"
        )

        print(
            f"Median nearest distance: "
            f"{distance_stats['median_m']:.3f} m"
        )

        print(
            f"95th percentile:         "
            f"{distance_stats['p95_m']:.3f} m"
        )

        print(
            f"Maximum distance:        "
            f"{distance_stats['max_m']:.3f} m"
        )

        print()
        print("Threshold agreement:")

        for threshold in [1, 5, 10, 25, 50]:
            key = f"within_{threshold}m_pct"

            print(
                f"  <= {threshold:2d} m: "
                f"{distance_stats[key]:6.2f}%"
            )

        # ==========================================================================================
        # Save city GeoPackage
        # ==========================================================================================

        section("SAVING CONTROLLED WINDOW")

        gpkg_path = (
            OUT_ROOT /
            f"{city_key}_analysis_window.gpkg"
        )

        if gpkg_path.exists():
            gpkg_path.unlink()

        window_gdf = gpd.GeoDataFrame(
            [{
                "city": cfg["city"],
                "city_key": city_key,
                "purpose": cfg["purpose"],
                "centre_x": centre_x,
                "centre_y": centre_y,
                "width_m": WINDOW_SIZE_M,
                "height_m": WINDOW_SIZE_M,
                "area_km2": 36.0,
                "geometry": window_geom,
            }],
            crs=CRS
        )

        window_gdf.to_file(
            gpkg_path,
            layer="analysis_window",
            driver="GPKG"
        )

        sb.to_file(
            gpkg_path,
            layer="source_buildings",
            driver="GPKG"
        )

        sr.to_file(
            gpkg_path,
            layer="source_roads",
            driver="GPKG"
        )

        tb.to_file(
            gpkg_path,
            layer="target_builtup",
            driver="GPKG"
        )

        tr.to_file(
            gpkg_path,
            layer="target_roads",
            driver="GPKG"
        )

        print(f"GeoPackage: {gpkg_path}")

        # ==========================================================================================
        # Render
        # ==========================================================================================

        section("GENERATING VALIDATION FIGURES")

        png_path, pdf_path = render_validation(
            cfg["city"],
            city_key,
            window_geom,
            sb,
            tb,
            sr,
            tr,
        )

        # ==========================================================================================
        # Records
        # ==========================================================================================

        window_records.append({
            "city_key": city_key,
            "city": cfg["city"],
            "purpose": cfg["purpose"],
            "latitude": cfg["lat"],
            "longitude": cfg["lon"],
            "centre_x": centre_x,
            "centre_y": centre_y,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "width_m": WINDOW_SIZE_M,
            "height_m": WINDOW_SIZE_M,
            "area_km2": 36.0,
        })

        summary = {
            "city_key": city_key,
            "city": cfg["city"],

            "source_buildings": int(len(sb)),
            "target_builtup": int(len(tb)),
            "source_roads": int(len(sr)),
            "target_roads": int(len(tr)),

            "source_building_area_km2":
                source_building_area,

            "target_builtup_area_km2":
                target_builtup_area,

            "source_road_length_km":
                source_road_length,

            "target_road_length_km":
                target_road_length,

            "road_feature_retention_pct":
                road_feature_retention,

            "road_length_retention_pct":
                road_length_retention,

            "building_intersection_pct":
                builtup_stats[
                    "buildings_intersecting_pct"
                ],

            "building_area_covered_pct":
                builtup_stats[
                    "building_area_covered_pct"
                ],

            "builtup_area_occupied_pct":
                builtup_stats[
                    "builtup_area_occupied_pct"
                ],

            "road_mean_distance_m":
                distance_stats["mean_m"],

            "road_median_distance_m":
                distance_stats["median_m"],

            "road_p95_distance_m":
                distance_stats["p95_m"],

            "road_within_1m_pct":
                distance_stats["within_1m_pct"],

            "road_within_5m_pct":
                distance_stats["within_5m_pct"],

            "road_within_10m_pct":
                distance_stats["within_10m_pct"],

            "road_within_25m_pct":
                distance_stats["within_25m_pct"],

            "road_within_50m_pct":
                distance_stats["within_50m_pct"],
        }

        summary_records.append(summary)

        distance_records.append({
            "city": cfg["city"],
            **distance_stats,
        })

    # ==================================================================================================
    # GLOBAL OUTPUTS
    # ==================================================================================================

    banner("GLOBAL POLISH ANALYSIS-WINDOW SUMMARY")

    if not summary_records:
        print("[FAIL] No city completed.")
        return

    summary_df = pd.DataFrame(summary_records)

    display_cols = [
        "city",
        "source_buildings",
        "target_builtup",
        "source_roads",
        "target_roads",
        "road_feature_retention_pct",
        "road_length_retention_pct",
        "building_area_covered_pct",
        "road_median_distance_m",
        "road_within_10m_pct",
        "road_within_50m_pct",
    ]

    print(
        summary_df[display_cols].to_string(
            index=False,
            formatters={
                "road_feature_retention_pct":
                    lambda x: f"{x:.2f}",
                "road_length_retention_pct":
                    lambda x: f"{x:.2f}",
                "building_area_covered_pct":
                    lambda x: f"{x:.2f}",
                "road_median_distance_m":
                    lambda x: f"{x:.3f}",
                "road_within_10m_pct":
                    lambda x: f"{x:.2f}",
                "road_within_50m_pct":
                    lambda x: f"{x:.2f}",
            }
        )
    )

    # ==================================================================================================
    # SAVE GLOBAL FILES
    # ==================================================================================================

    summary_csv = (
        OUT_ROOT /
        "poland_analysis_window_summary.csv"
    )

    summary_json = (
        OUT_ROOT /
        "poland_analysis_window_summary.json"
    )

    windows_json = (
        OUT_ROOT /
        "poland_analysis_windows.json"
    )

    windows_gpkg = (
        OUT_ROOT /
        "poland_analysis_windows.gpkg"
    )

    summary_df.to_csv(
        summary_csv,
        index=False
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary_records,
            f,
            indent=2,
            ensure_ascii=False,
            default=lambda x: (
                x.item()
                if hasattr(x, "item")
                else str(x)
            )
        )

    with open(
        windows_json,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            window_records,
            f,
            indent=2,
            ensure_ascii=False,
            default=lambda x: (
                x.item()
                if hasattr(x, "item")
                else str(x)
            )
        )

    # Save global window geometry.
    window_geometries = []

    for rec in window_records:
        geom = box(
            rec["xmin"],
            rec["ymin"],
            rec["xmax"],
            rec["ymax"],
        )

        row = dict(rec)
        row["geometry"] = geom

        window_geometries.append(row)

    windows_gdf = gpd.GeoDataFrame(
        window_geometries,
        crs=CRS
    )

    if windows_gpkg.exists():
        windows_gpkg.unlink()

    windows_gdf.to_file(
        windows_gpkg,
        layer="analysis_windows",
        driver="GPKG"
    )

    # ==================================================================================================
    # FINAL DECISION SUPPORT
    # ==================================================================================================

    banner("POLAND SUPERVISION READINESS")

    for _, row in summary_df.iterrows():

        city = row["city"]

        print()
        print(city)
        print("-" * 60)

        road_good = (
            row["road_within_50m_pct"] >= 70
            if not pd.isna(
                row["road_within_50m_pct"]
            )
            else False
        )

        builtup_good = (
            row["building_area_covered_pct"] >= 70
            if not pd.isna(
                row["building_area_covered_pct"]
            )
            else False
        )

        if road_good:
            print(
                "[PASS] Road source-target geometry "
                "shows usable spatial correspondence."
            )
        else:
            print(
                "[CHECK] Road correspondence needs "
                "further inspection."
            )

        if builtup_good:
            print(
                "[PASS] BDOO built-up areas provide "
                "usable contextual/aggregation supervision."
            )
        else:
            print(
                "[CHECK] Built-up supervision needs "
                "further inspection."
            )

        print(
            "Building interpretation: "
            "BDOT10k individual buildings -> "
            "BDOO generalized built-up context."
        )

        print(
            "Road interpretation: "
            "BDOT10k detailed network -> "
            "BDOO generalized road network."
        )

    banner("SCRIPT 56 COMPLETE")

    print()
    print("Summary CSV:")
    print(summary_csv)

    print()
    print("Summary JSON:")
    print(summary_json)

    print()
    print("Windows JSON:")
    print(windows_json)

    print()
    print("Windows GeoPackage:")
    print(windows_gpkg)

    print()
    print("Figures (PNG + PDF):")
    print(FIG_ROOT)

    print()
    print(
        "NEXT: Script 57 — build Polish road correspondence "
        "inside the validated common 6 × 6 km windows."
    )


if __name__ == "__main__":
    main()