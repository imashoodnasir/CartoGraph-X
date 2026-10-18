from pathlib import Path
import json
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

BDOT_ROOT = ROOT / "data" / "processed" / "poland" / "bdot10k"
BDOO_ROOT = ROOT / "data" / "processed" / "poland" / "bdoo"
OUT_ROOT = ROOT / "data" / "processed" / "poland" / "source_target_validation"
FIG_ROOT = OUT_ROOT / "figures"

OUT_ROOT.mkdir(parents=True, exist_ok=True)
FIG_ROOT.mkdir(parents=True, exist_ok=True)

CITIES = ["warsaw", "krakow", "gdansk"]

# We deliberately use the BDOT10k city/county extent as the comparison
# domain. BDOO packages cover an entire voivodeship, so they MUST be
# spatially clipped before feature-count comparisons are meaningful.
FILES = {
    "warsaw": {
        "buildings_source": BDOT_ROOT / "warsaw" / "warsaw_buildings.gpkg",
        "roads_source": BDOT_ROOT / "warsaw" / "warsaw_roads.gpkg",
        "buildings_target": BDOO_ROOT / "warsaw" / "candidate_layers" / "warsaw_ot_ptzb_a.gpkg",
        "roads_target": BDOO_ROOT / "warsaw" / "candidate_layers" / "warsaw_ot_skdr_l.gpkg",
    },
    "krakow": {
        "buildings_source": BDOT_ROOT / "krakow" / "krakow_buildings.gpkg",
        "roads_source": BDOT_ROOT / "krakow" / "krakow_roads.gpkg",
        "buildings_target": BDOO_ROOT / "krakow" / "candidate_layers" / "krakow_ot_ptzb_a.gpkg",
        "roads_target": BDOO_ROOT / "krakow" / "candidate_layers" / "krakow_ot_skdr_l.gpkg",
    },
    "gdansk": {
        "buildings_source": BDOT_ROOT / "gdansk" / "gdansk_buildings.gpkg",
        "roads_source": BDOT_ROOT / "gdansk" / "gdansk_roads.gpkg",
        "buildings_target": BDOO_ROOT / "gdansk" / "candidate_layers" / "gdansk_ot_ptzb_a.gpkg",
        "roads_target": BDOO_ROOT / "gdansk" / "candidate_layers" / "gdansk_ot_skdr_l.gpkg",
    },
}

TARGET_CRS = "EPSG:2180"


def banner(text):
    print("\n" + "=" * 120)
    print(text)
    print("=" * 120)


def section(text):
    print("\n" + "-" * 120)
    print(text)
    print("-" * 120)


def read_vector(path):
    if not path.exists():
        raise FileNotFoundError(path)

    gdf = gpd.read_file(path, engine="pyogrio")

    if gdf.crs is None:
        raise RuntimeError(f"No CRS: {path}")

    if str(gdf.crs).upper() != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    return gdf


def bbox_polygon(gdf):
    minx, miny, maxx, maxy = gdf.total_bounds
    return gpd.GeoSeries.from_bbox((minx, miny, maxx, maxy)).set_crs(gdf.crs)


def clip_to_source_extent(target, source):
    """
    First-stage clipping.

    This removes the large voivodeship-wide BDOO area outside the downloaded
    BDOT10k city/county source package.
    """
    minx, miny, maxx, maxy = source.total_bounds

    clipped = target.cx[minx:maxx, miny:maxy].copy()

    return clipped


def safe_union(gdf):
    if len(gdf) == 0:
        return None

    try:
        return gdf.geometry.union_all()
    except Exception:
        return gdf.geometry.unary_union


def percent(num, den):
    if den is None or den == 0:
        return np.nan
    return 100.0 * num / den


def building_validation(city, source, target):
    section(f"{city.upper()} — BUILDING / BUILT-UP RELATIONSHIP")

    target_clip = clip_to_source_extent(target, source)

    print(f"BDOT10k OT_BUBD_A source features: {len(source):,}")
    print(f"BDOO OT_PTZB_A full features:       {len(target):,}")
    print(f"BDOO OT_PTZB_A clipped features:    {len(target_clip):,}")

    source_area = float(source.geometry.area.sum())
    target_area = float(target_clip.geometry.area.sum())

    print(f"\nBDOT building area: {source_area / 1e6:,.3f} km²")
    print(f"BDOO PTZB area:     {target_area / 1e6:,.3f} km²")

    # ------------------------------------------------------------------
    # Spatial relationship between detailed buildings and PTZB polygons
    # ------------------------------------------------------------------
    src = source[["geometry"]].copy()
    src["source_id"] = np.arange(len(src))

    trg = target_clip[["geometry"]].copy()
    trg["target_id"] = np.arange(len(trg))

    if len(trg) == 0:
        return {
            "source_buildings": len(source),
            "target_ptzb_full": len(target),
            "target_ptzb_clipped": 0,
            "source_area_km2": source_area / 1e6,
            "target_area_km2": 0,
            "buildings_intersecting_ptzb": 0,
            "buildings_intersecting_ptzb_pct": 0,
            "median_buildings_per_ptzb": np.nan,
            "mean_buildings_per_ptzb": np.nan,
            "max_buildings_per_ptzb": np.nan,
            "ptzb_with_multiple_buildings_pct": np.nan,
        }, target_clip

    joined = gpd.sjoin(
        src,
        trg,
        how="left",
        predicate="intersects"
    )

    matched_source = joined.loc[
        joined["target_id"].notna(), "source_id"
    ].nunique()

    buildings_per_target = (
        joined.dropna(subset=["target_id"])
        .groupby("target_id")["source_id"]
        .nunique()
    )

    all_counts = pd.Series(
        0,
        index=np.arange(len(trg)),
        dtype=float
    )

    if len(buildings_per_target):
        all_counts.loc[
            buildings_per_target.index.astype(int)
        ] = buildings_per_target.values

    multi_target = int((all_counts > 1).sum())

    print("\nSpatial correspondence:")
    print(
        f"Buildings intersecting PTZB: "
        f"{matched_source:,} / {len(source):,} "
        f"({percent(matched_source, len(source)):.2f}%)"
    )

    print(
        f"Mean detailed buildings / PTZB polygon:   "
        f"{all_counts.mean():.2f}"
    )
    print(
        f"Median detailed buildings / PTZB polygon: "
        f"{all_counts.median():.2f}"
    )
    print(
        f"Maximum detailed buildings / PTZB polygon:"
        f" {all_counts.max():.0f}"
    )
    print(
        f"PTZB polygons containing/intersecting >1 building: "
        f"{multi_target:,} / {len(trg):,} "
        f"({percent(multi_target, len(trg)):.2f}%)"
    )

    # ------------------------------------------------------------------
    # Union-based coverage
    # ------------------------------------------------------------------
    source_union = safe_union(source)
    target_union = safe_union(target_clip)

    intersection_area = 0.0

    if source_union is not None and target_union is not None:
        try:
            intersection_area = source_union.intersection(target_union).area
        except Exception:
            intersection_area = 0.0

    source_covered = percent(intersection_area, source_area)
    target_covered = percent(intersection_area, target_area)

    print("\nArea overlap:")
    print(f"Intersection area:             {intersection_area / 1e6:,.3f} km²")
    print(f"Detailed building area in PTZB: {source_covered:.2f}%")
    print(f"PTZB area occupied by buildings: {target_covered:.2f}%")

    result = {
        "source_buildings": int(len(source)),
        "target_ptzb_full": int(len(target)),
        "target_ptzb_clipped": int(len(target_clip)),
        "source_area_km2": source_area / 1e6,
        "target_area_km2": target_area / 1e6,
        "intersection_area_km2": intersection_area / 1e6,
        "source_building_area_covered_pct": source_covered,
        "target_ptzb_area_occupied_pct": target_covered,
        "buildings_intersecting_ptzb": int(matched_source),
        "buildings_intersecting_ptzb_pct": percent(
            matched_source, len(source)
        ),
        "mean_buildings_per_ptzb": float(all_counts.mean()),
        "median_buildings_per_ptzb": float(all_counts.median()),
        "max_buildings_per_ptzb": float(all_counts.max()),
        "ptzb_with_multiple_buildings": multi_target,
        "ptzb_with_multiple_buildings_pct": percent(
            multi_target, len(trg)
        ),
    }

    return result, target_clip


def road_validation(city, source, target):
    section(f"{city.upper()} — ROAD GENERALIZATION RELATIONSHIP")

    target_clip = clip_to_source_extent(target, source)

    print(f"BDOT10k OT_SKDR_L source features: {len(source):,}")
    print(f"BDOO OT_SKDR_L full features:       {len(target):,}")
    print(f"BDOO OT_SKDR_L clipped features:    {len(target_clip):,}")

    source_length = float(source.geometry.length.sum())
    target_length = float(target_clip.geometry.length.sum())

    print(f"\nBDOT road length: {source_length / 1000:,.2f} km")
    print(f"BDOO road length: {target_length / 1000:,.2f} km")

    feature_retention = percent(len(target_clip), len(source))
    length_retention = percent(target_length, source_length)

    print(f"Feature-count retention: {feature_retention:.2f}%")
    print(f"Network-length retention: {length_retention:.2f}%")

    # ------------------------------------------------------------------
    # Proximity from generalized roads to detailed source network
    # ------------------------------------------------------------------
    distances = []

    if len(source) and len(target_clip):
        source_union = safe_union(source)

        for geom in target_clip.geometry:
            try:
                distances.append(float(geom.distance(source_union)))
            except Exception:
                pass

    distances = np.asarray(distances, dtype=float)

    if len(distances):
        print("\nTarget-to-source geometric proximity:")
        print(f"Mean distance:   {np.mean(distances):.3f} m")
        print(f"Median distance: {np.median(distances):.3f} m")
        print(f"95th percentile: {np.percentile(distances, 95):.3f} m")
        print(f"Maximum distance:{np.max(distances):.3f} m")
    else:
        print("\n[CHECK] Could not calculate road proximity.")

    # ------------------------------------------------------------------
    # Class distributions
    # ------------------------------------------------------------------
    source_class = {}
    target_class = {}

    if "klasaDrogi" in source.columns:
        source_class = (
            source["klasaDrogi"]
            .fillna("NULL")
            .astype(str)
            .value_counts()
            .to_dict()
        )

    if "klasaDrogi" in target_clip.columns:
        target_class = (
            target_clip["klasaDrogi"]
            .fillna("NULL")
            .astype(str)
            .value_counts()
            .to_dict()
        )

    if source_class or target_class:
        print("\nRoad-class distributions:")

        classes = sorted(
            set(source_class.keys()) | set(target_class.keys())
        )

        for c in classes:
            print(
                f"  {c[:55]:55s} "
                f"BDOT={source_class.get(c, 0):7,d} "
                f"BDOO={target_class.get(c, 0):7,d}"
            )

    result = {
        "source_roads": int(len(source)),
        "target_roads_full": int(len(target)),
        "target_roads_clipped": int(len(target_clip)),
        "source_length_km": source_length / 1000,
        "target_length_km": target_length / 1000,
        "feature_retention_pct": feature_retention,
        "length_retention_pct": length_retention,
        "mean_target_source_distance_m":
            float(np.mean(distances)) if len(distances) else np.nan,
        "median_target_source_distance_m":
            float(np.median(distances)) if len(distances) else np.nan,
        "p95_target_source_distance_m":
            float(np.percentile(distances, 95)) if len(distances) else np.nan,
        "max_target_source_distance_m":
            float(np.max(distances)) if len(distances) else np.nan,
        "source_class_distribution": source_class,
        "target_class_distribution": target_class,
    }

    return result, target_clip


def choose_visual_window(buildings, ptzb, size=4000):
    """
    Find a useful urban window automatically.

    Prefer a PTZB polygon around which there are many detailed buildings.
    """
    if len(ptzb):
        centroids = ptzb.geometry.centroid

        best = None
        best_count = -1

        # Sample at most 150 PTZB objects for speed.
        idxs = np.linspace(
            0,
            len(ptzb) - 1,
            min(150, len(ptzb)),
            dtype=int
        )

        half = size / 2

        for idx in idxs:
            c = centroids.iloc[idx]
            count = len(
                buildings.cx[
                    c.x - half:c.x + half,
                    c.y - half:c.y + half
                ]
            )

            if count > best_count:
                best_count = count
                best = c

        if best is not None:
            return (
                best.x - size / 2,
                best.y - size / 2,
                best.x + size / 2,
                best.y + size / 2,
            )

    # fallback
    minx, miny, maxx, maxy = buildings.total_bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2

    return (
        cx - size / 2,
        cy - size / 2,
        cx + size / 2,
        cy + size / 2,
    )


def make_figure(city, buildings, ptzb, roads, roads_target):
    section(f"{city.upper()} — GENERATING VISUAL VALIDATION")

    xmin, ymin, xmax, ymax = choose_visual_window(
        buildings,
        ptzb,
        size=4000
    )

    b = buildings.cx[xmin:xmax, ymin:ymax]
    p = ptzb.cx[xmin:xmax, ymin:ymax]
    r1 = roads.cx[xmin:xmax, ymin:ymax]
    r2 = roads_target.cx[xmin:xmax, ymin:ymax]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Buildings
    b.plot(ax=axes[0, 0])
    axes[0, 0].set_title(
        f"{city.title()} — BDOT10k detailed buildings\nOT_BUBD_A"
    )

    p.plot(ax=axes[0, 1])
    axes[0, 1].set_title(
        "BDOO generalized built-up representation\nOT_PTZB_A"
    )

    if len(p):
        p.plot(ax=axes[0, 2], alpha=0.45)
    if len(b):
        b.boundary.plot(ax=axes[0, 2], linewidth=0.35)

    axes[0, 2].set_title(
        "Building overlay\nBDOT10k + BDOO"
    )

    # Roads
    r1.plot(ax=axes[1, 0], linewidth=0.6)
    axes[1, 0].set_title(
        "BDOT10k detailed roads\nOT_SKDR_L"
    )

    r2.plot(ax=axes[1, 1], linewidth=0.9)
    axes[1, 1].set_title(
        "BDOO generalized roads\nOT_SKDR_L"
    )

    if len(r1):
        r1.plot(ax=axes[1, 2], linewidth=0.7)
    if len(r2):
        r2.plot(ax=axes[1, 2], linewidth=1.2, alpha=0.7)

    axes[1, 2].set_title(
        "Road overlay\nBDOT10k + BDOO"
    )

    for ax in axes.flat:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.axis("off")

    plt.tight_layout()

    output = FIG_ROOT / f"{city}_bdot10k_bdoo_validation.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[PASS] Figure: {output}")

    return str(output)


def main():
    banner(
        "CARTOGEN — VALIDATE POLISH BDOT10k → BDOO "
        "SOURCE-TARGET RELATIONSHIP"
    )

    all_rows = []
    report = {}

    for city in CITIES:
        banner(city.upper())

        paths = FILES[city]

        for role, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"{city} {role} missing:\n{path}"
                )

        print("Loading BDOT10k buildings...")
        buildings = read_vector(paths["buildings_source"])

        print("Loading BDOT10k roads...")
        roads = read_vector(paths["roads_source"])

        print("Loading BDOO PTZB...")
        ptzb = read_vector(paths["buildings_target"])

        print("Loading BDOO roads...")
        roads_target = read_vector(paths["roads_target"])

        print("\nCRS:")
        print(f"  Buildings source: {buildings.crs}")
        print(f"  Buildings target: {ptzb.crs}")
        print(f"  Roads source:     {roads.crs}")
        print(f"  Roads target:     {roads_target.crs}")

        building_result, ptzb_clip = building_validation(
            city,
            buildings,
            ptzb
        )

        road_result, roads_target_clip = road_validation(
            city,
            roads,
            roads_target
        )

        figure = make_figure(
            city,
            buildings,
            ptzb_clip,
            roads,
            roads_target_clip
        )

        report[city] = {
            "buildings": building_result,
            "roads": road_result,
            "figure": figure,
        }

        row = {
            "city": city,
            **{
                f"building_{k}": v
                for k, v in building_result.items()
                if not isinstance(v, dict)
            },
            **{
                f"road_{k}": v
                for k, v in road_result.items()
                if not isinstance(v, dict)
            },
        }

        all_rows.append(row)

    banner("GLOBAL SOURCE-TARGET VALIDATION SUMMARY")

    df = pd.DataFrame(all_rows)

    print(df.to_string(index=False))

    csv_path = OUT_ROOT / "poland_source_target_validation.csv"
    df.to_csv(csv_path, index=False)

    json_path = OUT_ROOT / "poland_source_target_validation.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
            default=str
        )

    banner("SCRIPT 55 COMPLETE")

    print(f"CSV:     {csv_path}")
    print(f"JSON:    {json_path}")
    print(f"Figures: {FIG_ROOT}")

    print(
        "\nNEXT: inspect the quantitative relationships and "
        "validation figures before declaring the Polish "
        "BDOT10k → BDOO supervision strategy."
    )


if __name__ == "__main__":
    main()