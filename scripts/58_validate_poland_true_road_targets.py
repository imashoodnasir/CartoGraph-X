from pathlib import Path
import json
import math

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from shapely.ops import unary_union


ROOT = Path(r"D:\Research\CartoGen")

WINDOW_ROOT = (
    ROOT / "data" / "processed" / "poland" /
    "analysis_windows"
)

BDOO_ROOT = (
    ROOT / "data" / "processed" / "poland" /
    "bdoo"
)

OUT_ROOT = (
    ROOT / "data" / "processed" / "poland" /
    "road_target_validation"
)

FIG_ROOT = OUT_ROOT / "figures"

OUT_ROOT.mkdir(parents=True, exist_ok=True)
FIG_ROOT.mkdir(parents=True, exist_ok=True)


CITIES = {
    "warsaw": {
        "name": "Warsaw",
        "window_file":
            WINDOW_ROOT / "warsaw_analysis_window.gpkg",
        "target_file":
            BDOO_ROOT / "warsaw" / "candidate_layers" /
            "warsaw_ot_skdr_l.gpkg",
    },

    "krakow": {
        "name": "Krakow",
        "window_file":
            WINDOW_ROOT / "krakow_analysis_window.gpkg",
        "target_file":
            BDOO_ROOT / "krakow" / "candidate_layers" /
            "krakow_ot_skdr_l.gpkg",
    },

    "gdansk": {
        "name": "Gdansk",
        "window_file":
            WINDOW_ROOT / "gdansk_analysis_window.gpkg",
        "target_file":
            BDOO_ROOT / "gdansk" / "candidate_layers" /
            "gdansk_ot_skdr_l.gpkg",
    },
}


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


def list_layers(path):
    import pyogrio
    return [x[0] for x in pyogrio.list_layers(path)]


def find_layer(path, candidates):

    layers = list_layers(path)

    lower = {
        x.lower(): x
        for x in layers
    }

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    return None


def safe_make_valid(gdf):

    result = gdf.copy()

    invalid = ~result.geometry.is_valid

    if invalid.any():

        try:
            result.loc[invalid, "geometry"] = (
                result.loc[invalid, "geometry"]
                .make_valid()
            )

        except Exception:

            result.loc[invalid, "geometry"] = (
                result.loc[invalid, "geometry"]
                .buffer(0)
            )

    result = result[
        result.geometry.notna()
        & ~result.geometry.is_empty
    ].copy()

    return result


def clip_to_window(gdf, window_geom):

    minx, miny, maxx, maxy = window_geom.bounds

    try:
        subset = gdf.cx[
            minx:maxx,
            miny:maxy
        ].copy()
    except Exception:
        subset = gdf[
            gdf.geometry.intersects(window_geom)
        ].copy()

    subset = subset[
        subset.geometry.intersects(window_geom)
    ].copy()

    subset["geometry"] = subset.geometry.intersection(
        window_geom
    )

    subset = subset[
        subset.geometry.notna()
        & ~subset.geometry.is_empty
    ].copy()

    return subset


def nearest_distance_stats(source, target):

    if len(source) == 0 or len(target) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "max": np.nan,
            "within_1": np.nan,
            "within_5": np.nan,
            "within_10": np.nan,
            "within_25": np.nan,
            "within_50": np.nan,
        }

    source_union = unary_union(
        source.geometry.tolist()
    )

    distances = np.array(
        [
            geom.distance(source_union)
            for geom in target.geometry
        ],
        dtype=float
    )

    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "p95": float(np.percentile(distances, 95)),
        "max": float(np.max(distances)),
        "within_1":
            float(np.mean(distances <= 1) * 100),
        "within_5":
            float(np.mean(distances <= 5) * 100),
        "within_10":
            float(np.mean(distances <= 10) * 100),
        "within_25":
            float(np.mean(distances <= 25) * 100),
        "within_50":
            float(np.mean(distances <= 50) * 100),
    }


def source_coverage_by_target(source, target, buffer_m):

    if len(source) == 0 or len(target) == 0:
        return np.nan

    buffered_target = unary_union(
        target.geometry.buffer(buffer_m).tolist()
    )

    source_lengths = source.geometry.length

    total = float(source_lengths.sum())

    if total == 0:
        return np.nan

    covered = 0.0

    for geom in source.geometry:

        try:
            covered += geom.intersection(
                buffered_target
            ).length
        except Exception:
            pass

    return 100.0 * covered / total


def target_coverage_by_source(source, target, buffer_m):

    if len(source) == 0 or len(target) == 0:
        return np.nan

    buffered_source = unary_union(
        source.geometry.buffer(buffer_m).tolist()
    )

    target_lengths = target.geometry.length

    total = float(target_lengths.sum())

    if total == 0:
        return np.nan

    covered = 0.0

    for geom in target.geometry:

        try:
            covered += geom.intersection(
                buffered_source
            ).length
        except Exception:
            pass

    return 100.0 * covered / total


def save_figure(city_name, source, target, window_geom, path_base):

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    source.plot(
        ax=ax,
        linewidth=0.35,
        alpha=0.55,
        label="BDOT10k detailed roads"
    )

    target.plot(
        ax=ax,
        linewidth=1.0,
        alpha=0.90,
        label="BDOO OT_SKDR_L"
    )

    boundary = gpd.GeoSeries(
        [window_geom],
        crs=source.crs
    ).boundary

    boundary.plot(
        ax=ax,
        linewidth=1.2,
        linestyle="--"
    )

    ax.set_title(
        f"{city_name}: BDOT10k → BDOO road target validation"
    )

    ax.set_aspect("equal")

    ax.legend(
        loc="best"
    )

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    plt.tight_layout()

    png = path_base.with_suffix(".png")
    pdf = path_base.with_suffix(".pdf")

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"PNG: {png}")
    print(f"PDF: {pdf}")


def main():

    banner(
        "CARTOGEN — VALIDATE TRUE POLISH BDOO ROAD TARGETS"
    )

    summaries = []

    for city_key, cfg in CITIES.items():

        city_name = cfg["name"]

        banner(city_name.upper())

        window_file = cfg["window_file"]
        target_file = cfg["target_file"]

        if not window_file.exists():
            raise FileNotFoundError(window_file)

        if not target_file.exists():
            raise FileNotFoundError(target_file)

        # ----------------------------------------------------------------------
        # READ SOURCE FROM SCRIPT 56 WINDOW
        # ----------------------------------------------------------------------

        section("READING EXISTING CONTROLLED WINDOW")

        layers = list_layers(window_file)

        print("Layers:")
        for layer in layers:
            print(f"  {layer}")

        source_layer = find_layer(
            window_file,
            [
                "source_roads",
                "roads_source",
                "bdot10k_roads",
            ]
        )

        if source_layer is None:
            raise RuntimeError(
                f"Could not identify source road layer in "
                f"{window_file}. Layers: {layers}"
            )

        source = gpd.read_file(
            window_file,
            layer=source_layer
        )

        source = safe_make_valid(source)

        print()
        print(f"Source layer: {source_layer}")
        print(f"Source roads: {len(source):,}")
        print(f"CRS:          {source.crs}")

        # ----------------------------------------------------------------------
        # RECOVER WINDOW GEOMETRY
        # ----------------------------------------------------------------------

        total_bounds = source.total_bounds

        minx, miny, maxx, maxy = total_bounds

        # Script 56 clipped the source geometry to the
        # exact 6 x 6 km analysis window. Reconstruct
        # the controlled window from the known city
        # centres used in Script 56.

        centres = {
            "warsaw": (637382.20, 486757.21),
            "krakow": (567613.62, 244587.53),
            "gdansk": (477037.60, 720936.52),
        }

        cx, cy = centres[city_key]

        window_geom = box(
            cx - 3000,
            cy - 3000,
            cx + 3000,
            cy + 3000,
        )

        print(
            "Window bbox: "
            f"{window_geom.bounds}"
        )

        # ----------------------------------------------------------------------
        # READ TRUE TARGET
        # ----------------------------------------------------------------------

        section(
            "READING TRUE BDOO ROAD TARGET — OT_SKDR_L"
        )

        target_layers = list_layers(target_file)

        if "OT_SKDR_L" in target_layers:
            target_layer = "OT_SKDR_L"
        else:
            target_layer = target_layers[0]

        target = gpd.read_file(
            target_file,
            layer=target_layer
        )

        target = safe_make_valid(target)

        print(f"Target layer: {target_layer}")
        print(f"National/regional features: {len(target):,}")
        print(f"CRS: {target.crs}")

        if source.crs != target.crs:
            target = target.to_crs(source.crs)

        # ----------------------------------------------------------------------
        # CLIP TARGET
        # ----------------------------------------------------------------------

        section(
            "EXTRACTING TRUE TARGET INSIDE 6 × 6 KM WINDOW"
        )

        target_window = clip_to_window(
            target,
            window_geom
        )

        print(
            f"Target roads in window: "
            f"{len(target_window):,}"
        )

        # Ensure source is exactly clipped as well.
        source_window = clip_to_window(
            source,
            window_geom
        )

        print(
            f"Source roads in window: "
            f"{len(source_window):,}"
        )

        # ----------------------------------------------------------------------
        # BASIC STATISTICS
        # ----------------------------------------------------------------------

        section("SOURCE → TARGET GENERALIZATION STATISTICS")

        source_length = float(
            source_window.geometry.length.sum()
            / 1000
        )

        target_length = float(
            target_window.geometry.length.sum()
            / 1000
        )

        source_median = float(
            source_window.geometry.length.median()
        )

        target_median = float(
            target_window.geometry.length.median()
        )

        feature_retention = (
            100 * len(target_window) /
            len(source_window)
            if len(source_window)
            else np.nan
        )

        length_retention = (
            100 * target_length /
            source_length
            if source_length > 0
            else np.nan
        )

        median_multiplier = (
            target_median / source_median
            if source_median > 0
            else np.nan
        )

        print(
            f"Source count:             "
            f"{len(source_window):,}"
        )

        print(
            f"Target count:             "
            f"{len(target_window):,}"
        )

        print(
            f"Feature retention:        "
            f"{feature_retention:.2f}%"
        )

        print(
            f"Source total length:      "
            f"{source_length:.2f} km"
        )

        print(
            f"Target total length:      "
            f"{target_length:.2f} km"
        )

        print(
            f"Length retention:         "
            f"{length_retention:.2f}%"
        )

        print(
            f"Source median segment:    "
            f"{source_median:.2f} m"
        )

        print(
            f"Target median segment:    "
            f"{target_median:.2f} m"
        )

        print(
            f"Median length multiplier: "
            f"{median_multiplier:.3f}×"
        )

        # ----------------------------------------------------------------------
        # GEOMETRIC ALIGNMENT
        # ----------------------------------------------------------------------

        section("TARGET → SOURCE GEOMETRIC ALIGNMENT")

        distances = nearest_distance_stats(
            source_window,
            target_window
        )

        print(
            f"Mean nearest distance:   "
            f"{distances['mean']:.3f} m"
        )

        print(
            f"Median nearest distance: "
            f"{distances['median']:.3f} m"
        )

        print(
            f"95th percentile:         "
            f"{distances['p95']:.3f} m"
        )

        print(
            f"Maximum distance:        "
            f"{distances['max']:.3f} m"
        )

        print()
        print("Threshold agreement:")

        for threshold in [1, 5, 10, 25, 50]:

            value = distances[
                f"within_{threshold}"
            ]

            print(
                f"  <= {threshold:2d} m: "
                f"{value:.2f}%"
            )

        # ----------------------------------------------------------------------
        # NETWORK COVERAGE
        # ----------------------------------------------------------------------

        section("NETWORK COVERAGE")

        coverage_rows = []

        for threshold in [1, 5, 10, 25, 50]:

            source_cov = source_coverage_by_target(
                source_window,
                target_window,
                threshold
            )

            target_cov = target_coverage_by_source(
                source_window,
                target_window,
                threshold
            )

            coverage_rows.append(
                {
                    "threshold_m": threshold,
                    "source_length_covered_pct":
                        source_cov,
                    "target_length_supported_pct":
                        target_cov,
                }
            )

            print(
                f"{threshold:>2} m | "
                f"source covered: "
                f"{source_cov:6.2f}% | "
                f"target supported: "
                f"{target_cov:6.2f}%"
            )

        coverage_df = pd.DataFrame(
            coverage_rows
        )

        # ----------------------------------------------------------------------
        # SAVE VALIDATED PAIR
        # ----------------------------------------------------------------------

        section("SAVING VALIDATED ROAD PAIR")

        gpkg = (
            OUT_ROOT /
            f"{city_key}_validated_road_pair.gpkg"
        )

        if gpkg.exists():
            gpkg.unlink()

        source_window.to_file(
            gpkg,
            layer="source_roads",
            driver="GPKG"
        )

        target_window.to_file(
            gpkg,
            layer="target_roads",
            driver="GPKG"
        )

        window_gdf = gpd.GeoDataFrame(
            {
                "city": [city_name],
                "geometry": [window_geom],
            },
            crs=source_window.crs
        )

        window_gdf.to_file(
            gpkg,
            layer="analysis_window",
            driver="GPKG"
        )

        coverage_csv = (
            OUT_ROOT /
            f"{city_key}_road_coverage.csv"
        )

        coverage_df.to_csv(
            coverage_csv,
            index=False
        )

        print(f"GeoPackage: {gpkg}")
        print(f"Coverage CSV: {coverage_csv}")

        # ----------------------------------------------------------------------
        # FIGURE
        # ----------------------------------------------------------------------

        section("GENERATING VALIDATION FIGURE")

        save_figure(
            city_name,
            source_window,
            target_window,
            window_geom,
            FIG_ROOT /
            f"{city_key}_true_road_target"
        )

        # ----------------------------------------------------------------------
        # READINESS
        # ----------------------------------------------------------------------

        section("READINESS ASSESSMENT")

        checks = {
            "target_not_empty":
                len(target_window) > 0,

            "target_has_multiple_features":
                len(target_window) >= 20,

            "target_is_line_geometry":
                target_window.geom_type
                .isin(
                    [
                        "LineString",
                        "MultiLineString",
                    ]
                )
                .all(),

            "reasonable_length":
                target_length > 10,

            "geometric_support":
                distances["within_50"] >= 80,
        }

        for key, value in checks.items():

            print(
                f"{key:<35} "
                f"{'PASS' if value else 'CHECK'}"
            )

        ready = all(checks.values())

        if ready:

            print()
            print(
                "[PASS] TRUE BDOO ROAD TARGET "
                "IS USABLE."
            )

        else:

            print()
            print(
                "[CHECK] Review this city before "
                "building correspondence."
            )

        summaries.append(
            {
                "city": city_name,
                "source_roads":
                    int(len(source_window)),
                "target_roads":
                    int(len(target_window)),
                "feature_retention_pct":
                    float(feature_retention),
                "source_length_km":
                    source_length,
                "target_length_km":
                    target_length,
                "length_retention_pct":
                    float(length_retention),
                "source_median_segment_m":
                    source_median,
                "target_median_segment_m":
                    target_median,
                "median_length_multiplier":
                    float(median_multiplier),
                "mean_nearest_distance_m":
                    distances["mean"],
                "median_nearest_distance_m":
                    distances["median"],
                "within_10m_pct":
                    distances["within_10"],
                "within_25m_pct":
                    distances["within_25"],
                "within_50m_pct":
                    distances["within_50"],
                "ready":
                    bool(ready),
            }
        )

    # ==========================================================================
    # GLOBAL SUMMARY
    # ==========================================================================

    banner("GLOBAL TRUE BDOO ROAD TARGET SUMMARY")

    summary_df = pd.DataFrame(
        summaries
    )

    print(
        summary_df.to_string(
            index=False,
            formatters={
                "feature_retention_pct":
                    "{:.2f}".format,
                "source_length_km":
                    "{:.2f}".format,
                "target_length_km":
                    "{:.2f}".format,
                "length_retention_pct":
                    "{:.2f}".format,
                "median_nearest_distance_m":
                    "{:.3f}".format,
                "within_10m_pct":
                    "{:.2f}".format,
                "within_25m_pct":
                    "{:.2f}".format,
                "within_50m_pct":
                    "{:.2f}".format,
            }
        )
    )

    csv_path = (
        OUT_ROOT /
        "poland_true_road_target_validation.csv"
    )

    json_path = (
        OUT_ROOT /
        "poland_true_road_target_validation.json"
    )

    summary_df.to_csv(
        csv_path,
        index=False
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summaries,
            f,
            indent=2,
            ensure_ascii=False,
            default=str
        )

    banner("SCRIPT 58 COMPLETE")

    print()
    print("Summary CSV:")
    print(csv_path)

    print()
    print("Summary JSON:")
    print(json_path)

    print()
    print("Figures (PNG + PDF):")
    print(FIG_ROOT)

    print()

    if all(
        x["ready"]
        for x in summaries
    ):
        print(
            "[PASS] Polish BDOO road targets "
            "validated for all cities."
        )
        print()
        print(
            "NEXT: Script 59 — build BDOT10k → "
            "BDOO road correspondence and infer "
            "selection/generalization operations."
        )
    else:
        print(
            "[CHECK] At least one city requires "
            "review before correspondence."
        )


if __name__ == "__main__":
    main()