from pathlib import Path
import json
import math
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union, linemerge
import networkx as nx

warnings.filterwarnings("ignore")

# ======================================================================================================================
# CONFIGURATION
# ======================================================================================================================

ROOT = Path(r"D:\Research\CartoGen")

VALIDATED_DIR = ROOT / "data" / "processed" / "poland" / "validated_pairs"
CORR_DIR = ROOT / "data" / "processed" / "poland" / "road_correspondence"
OUTPUT_DIR = ROOT / "data" / "processed" / "poland" / "road_topology"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CITIES = {
    "warsaw": "Warsaw",
    "krakow": "Krakow",
    "gdansk": "Gdansk",
}

CRS = "EPSG:2180"

# Endpoint snapping tolerance for graph construction.
# Small enough to preserve actual network structure but tolerant to
# tiny coordinate differences.
NODE_SNAP_M = 1.0

# Classification thresholds.
SIMPLIFICATION_LENGTH_RATIO_MIN = 0.70
SIMPLIFICATION_LENGTH_RATIO_MAX = 1.30
LOW_DISTANCE_M = 2.0
RESHAPE_DISTANCE_M = 2.0


# ======================================================================================================================
# HELPERS
# ======================================================================================================================

def header(text):
    print()
    print("=" * 120)
    print(text)
    print("=" * 120)


def subheader(text):
    print()
    print("-" * 120)
    print(text)
    print("-" * 120)


def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        value = float(value)
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, pd.Series):
        return json_safe(obj.to_dict())

    if isinstance(obj, pd.DataFrame):
        return json_safe(obj.to_dict(orient="records"))

    if pd.isna(obj):
        return None

    return obj


def find_validated_gpkg(city):
    """
    Locate the validated BDOT10k -> BDOO road-pair GeoPackage
    produced by the Polish road-target validation stage.
    """
    path = (
        ROOT
        / "data"
        / "processed"
        / "poland"
        / "road_target_validation"
        / f"{city}_validated_road_pair.gpkg"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Validated road-pair GeoPackage not found for {city}:\n{path}"
        )

    return path

def identify_road_layers(gpkg_path):
    layers = gpd.list_layers(gpkg_path)
    names = layers["name"].tolist()

    source_candidates = [
        "source_roads",
        "bdot10k_roads",
        "roads_source",
        "source_road",
    ]

    target_candidates = [
        "target_roads",
        "bdoo_roads",
        "roads_target",
        "target_road",
    ]

    source_layer = None
    target_layer = None

    for candidate in source_candidates:
        if candidate in names:
            source_layer = candidate
            break

    for candidate in target_candidates:
        if candidate in names:
            target_layer = candidate
            break

    if source_layer is None:
        for name in names:
            low = name.lower()
            if "road" in low and (
                "source" in low or "bdot" in low
            ):
                source_layer = name
                break

    if target_layer is None:
        for name in names:
            low = name.lower()
            if "road" in low and (
                "target" in low or "bdoo" in low
            ):
                target_layer = name
                break

    if source_layer is None or target_layer is None:
        raise RuntimeError(
            f"Could not identify source/target road layers in {gpkg_path}\n"
            f"Available layers: {names}"
        )

    return source_layer, target_layer


def detect_id_column(df, preferred=None):
    if preferred:
        for col in preferred:
            if col in df.columns:
                return col

    candidates = [
        "source_id",
        "target_id",
        "src_id",
        "tgt_id",
        "OBJECTID",
        "objectid",
        "fid",
        "FID",
        "id",
        "ID",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def normalize_identifier(value):
    if pd.isna(value):
        return None

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value)


def coordinate_key(coord, tolerance=NODE_SNAP_M):
    x, y = coord

    if tolerance <= 0:
        return (float(x), float(y))

    return (
        round(float(x) / tolerance) * tolerance,
        round(float(y) / tolerance) * tolerance,
    )


def iter_lines(geometry):
    if geometry is None or geometry.is_empty:
        return

    if isinstance(geometry, LineString):
        yield geometry

    elif isinstance(geometry, MultiLineString):
        for line in geometry.geoms:
            yield line

    else:
        try:
            if geometry.geom_type == "GeometryCollection":
                for geom in geometry.geoms:
                    yield from iter_lines(geom)
        except Exception:
            return


def count_vertices(geometry):
    total = 0

    for line in iter_lines(geometry):
        try:
            total += len(line.coords)
        except Exception:
            pass

    return total


def graph_from_lines(gdf, snap_tolerance=NODE_SNAP_M):
    """
    Build an undirected graph using every consecutive pair of vertices.
    This preserves internal road vertices and provides a more meaningful
    topology representation than using feature endpoints alone.
    """

    G = nx.Graph()

    total_length = 0.0

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue

        for line in iter_lines(geom):
            coords = list(line.coords)

            if len(coords) < 2:
                continue

            for i in range(len(coords) - 1):
                c1 = coords[i]
                c2 = coords[i + 1]

                n1 = coordinate_key(c1, snap_tolerance)
                n2 = coordinate_key(c2, snap_tolerance)

                if n1 == n2:
                    continue

                seg_length = math.dist(
                    (float(c1[0]), float(c1[1])),
                    (float(c2[0]), float(c2[1])),
                )

                if seg_length <= 0:
                    continue

                total_length += seg_length

                if G.has_edge(n1, n2):
                    G[n1][n2]["weight"] += seg_length
                    G[n1][n2]["multiplicity"] += 1
                else:
                    G.add_edge(
                        n1,
                        n2,
                        weight=seg_length,
                        multiplicity=1,
                    )

    return G, total_length


def summarize_graph(G, total_length):
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()

    if nodes == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "components": 0,
            "largest_component_nodes": 0,
            "largest_component_pct": 0.0,
            "mean_degree": 0.0,
            "dead_ends": 0,
            "junctions": 0,
            "cycle_rank": 0,
            "total_graph_length_m": float(total_length),
        }

    components = list(nx.connected_components(G))
    component_count = len(components)

    largest_component_nodes = max(
        (len(c) for c in components),
        default=0,
    )

    largest_component_pct = (
        100.0 * largest_component_nodes / nodes
        if nodes > 0
        else 0.0
    )

    degrees = np.array(
        [degree for _, degree in G.degree()],
        dtype=float,
    )

    mean_degree = (
        float(degrees.mean())
        if len(degrees)
        else 0.0
    )

    dead_ends = int(np.sum(degrees == 1))
    junctions = int(np.sum(degrees >= 3))

    cycle_rank = max(
        0,
        edges - nodes + component_count,
    )

    return {
        "nodes": int(nodes),
        "edges": int(edges),
        "components": int(component_count),
        "largest_component_nodes": int(largest_component_nodes),
        "largest_component_pct": float(largest_component_pct),
        "mean_degree": float(mean_degree),
        "dead_ends": int(dead_ends),
        "junctions": int(junctions),
        "cycle_rank": int(cycle_rank),
        "total_graph_length_m": float(total_length),
    }


def retention(target_value, source_value):
    if source_value is None or source_value == 0:
        return np.nan

    return 100.0 * float(target_value) / float(source_value)


def get_correspondence_files(city):
    pairs = CORR_DIR / f"{city}_road_correspondence_pairs.csv"
    labels = CORR_DIR / f"{city}_source_road_labels.csv"
    targets = CORR_DIR / f"{city}_target_road_correspondence.csv"

    missing = [
        p for p in [pairs, labels, targets]
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing Script 59 output(s):\n" +
            "\n".join(str(p) for p in missing)
        )

    return pairs, labels, targets


def infer_pair_columns(df):
    source_candidates = [
        "source_id",
        "source_index",
        "source_idx",
        "src_id",
        "src_index",
        "source_fid",
    ]

    target_candidates = [
        "target_id",
        "target_index",
        "target_idx",
        "tgt_id",
        "tgt_index",
        "target_fid",
    ]

    source_col = next(
        (c for c in source_candidates if c in df.columns),
        None,
    )

    target_col = next(
        (c for c in target_candidates if c in df.columns),
        None,
    )

    if source_col is None:
        source_col = next(
            (
                c for c in df.columns
                if "source" in c.lower()
                and (
                    "id" in c.lower()
                    or "index" in c.lower()
                )
            ),
            None,
        )

    if target_col is None:
        target_col = next(
            (
                c for c in df.columns
                if "target" in c.lower()
                and (
                    "id" in c.lower()
                    or "index" in c.lower()
                )
            ),
            None,
        )

    if source_col is None or target_col is None:
        raise RuntimeError(
            "Could not infer source/target identifier columns "
            f"from correspondence columns:\n{list(df.columns)}"
        )

    return source_col, target_col


def build_geometry_lookup(gdf):
    """
    Build multiple lookup possibilities:
    index, sequential position and common ID fields.
    """

    lookup = {}

    for pos, (idx, row) in enumerate(gdf.iterrows()):
        geom = row.geometry

        lookup[normalize_identifier(idx)] = geom
        lookup[normalize_identifier(pos)] = geom

        for col in [
            "OBJECTID",
            "objectid",
            "fid",
            "FID",
            "id",
            "ID",
            "TOP_ID",
            "lokalnyId",
        ]:
            if col in gdf.columns:
                value = row[col]

                if pd.notna(value):
                    lookup[normalize_identifier(value)] = geom

    return lookup


def classify_target_operation(
    source_geoms,
    target_geom,
    correspondence_type=None,
):
    if target_geom is None or target_geom.is_empty:
        return "unresolved"

    valid_sources = [
        g for g in source_geoms
        if g is not None and not g.is_empty
    ]

    if not valid_sources:
        return "unmatched_target"

    source_count = len(valid_sources)

    source_union = unary_union(valid_sources)

    source_length = float(
        sum(g.length for g in valid_sources)
    )

    target_length = float(target_geom.length)

    if source_length <= 0:
        length_ratio = np.nan
    else:
        length_ratio = target_length / source_length

    distance = float(source_union.distance(target_geom))

    source_vertices = count_vertices(source_union)
    target_vertices = count_vertices(target_geom)

    if source_vertices > 0:
        vertex_ratio = target_vertices / source_vertices
    else:
        vertex_ratio = np.nan

    # Multi-source correspondences represent merge / chain behaviour.
    if source_count >= 2:
        return "merge_or_chain"

    # One-source correspondences.
    if distance > RESHAPE_DISTANCE_M:
        return "displacement_or_reshape"

    if (
        np.isfinite(length_ratio)
        and SIMPLIFICATION_LENGTH_RATIO_MIN
        <= length_ratio
        <= SIMPLIFICATION_LENGTH_RATIO_MAX
    ):
        return "retained_or_simplified"

    return "displacement_or_reshape"


# ======================================================================================================================
# MAIN
# ======================================================================================================================

def main():

    header(
        "CARTOGEN — ANALYZE POLISH ROAD GENERALIZATION OPERATIONS AND TOPOLOGY"
    )

    global_summary = []
    global_operations = []

    for city_key, city_name in CITIES.items():

        header(f"{city_key.upper()} ({city_name})")

        # --------------------------------------------------------------------------------------------------------------
        # LOAD VALIDATED NETWORKS
        # --------------------------------------------------------------------------------------------------------------

        subheader("READING VALIDATED BDOT10k / BDOO ROAD NETWORKS")

        gpkg = find_validated_gpkg(city_key)

        source_layer, target_layer = identify_road_layers(gpkg)

        print(f"GeoPackage:   {gpkg}")
        print(f"Source layer: {source_layer}")
        print(f"Target layer: {target_layer}")

        source = gpd.read_file(
            gpkg,
            layer=source_layer,
        )

        target = gpd.read_file(
            gpkg,
            layer=target_layer,
        )

        if source.crs is None:
            source = source.set_crs(CRS)

        if target.crs is None:
            target = target.set_crs(CRS)

        if source.crs.to_epsg() != 2180:
            source = source.to_crs(CRS)

        if target.crs.to_epsg() != 2180:
            target = target.to_crs(CRS)

        source = source[
            source.geometry.notna()
            & ~source.geometry.is_empty
        ].copy()

        target = target[
            target.geometry.notna()
            & ~target.geometry.is_empty
        ].copy()

        print(f"BDOT10k source roads: {len(source):,}")
        print(f"BDOO target roads:    {len(target):,}")
        print(f"CRS:                  {source.crs}")
        print(
            f"Source length:         "
            f"{source.geometry.length.sum()/1000:.2f} km"
        )
        print(
            f"Target length:         "
            f"{target.geometry.length.sum()/1000:.2f} km"
        )

        # --------------------------------------------------------------------------------------------------------------
        # SCRIPT 59 OUTPUTS
        # --------------------------------------------------------------------------------------------------------------

        subheader("READING SCRIPT 59 CORRESPONDENCE")

        pair_file, label_file, target_file = (
            get_correspondence_files(city_key)
        )

        pairs = pd.read_csv(pair_file)
        source_labels = pd.read_csv(label_file)
        target_corr = pd.read_csv(target_file)

        print(f"Selected pairs:       {len(pairs):,}")
        print(f"Source labels:        {len(source_labels):,}")
        print(f"Target records:       {len(target_corr):,}")

        source_col, target_col = infer_pair_columns(pairs)

        print(f"Pair source ID field: {source_col}")
        print(f"Pair target ID field: {target_col}")

        # --------------------------------------------------------------------------------------------------------------
        # BUILD GEOMETRY LOOKUPS
        # --------------------------------------------------------------------------------------------------------------

        subheader("BUILDING GEOMETRY LOOKUPS")

        source_lookup = build_geometry_lookup(source)
        target_lookup = build_geometry_lookup(target)

        print(
            f"Source geometry lookup entries: "
            f"{len(source_lookup):,}"
        )
        print(
            f"Target geometry lookup entries: "
            f"{len(target_lookup):,}"
        )

        # --------------------------------------------------------------------------------------------------------------
        # GROUP SOURCE ROADS BY TARGET
        # --------------------------------------------------------------------------------------------------------------

        grouped = {}

        for _, row in pairs.iterrows():

            source_id = normalize_identifier(
                row[source_col]
            )

            target_id = normalize_identifier(
                row[target_col]
            )

            if target_id not in grouped:
                grouped[target_id] = []

            grouped[target_id].append(source_id)

        # --------------------------------------------------------------------------------------------------------------
        # TARGET OPERATIONS
        # --------------------------------------------------------------------------------------------------------------

        subheader("ANALYZING ROAD GENERALIZATION OPERATIONS")

        operation_rows = []

        target_ids = list(grouped.keys())

        print(
            f"Analyzing {len(target_ids):,} matched "
            f"BDOO target road segments..."
        )

        for i, target_id in enumerate(target_ids, start=1):

            source_ids = grouped[target_id]

            source_geoms = [
                source_lookup.get(sid)
                for sid in source_ids
            ]

            source_geoms = [
                g for g in source_geoms
                if g is not None and not g.is_empty
            ]

            target_geom = target_lookup.get(target_id)

            if target_geom is None:
                continue

            source_count = len(source_geoms)

            if source_count == 0:
                continue

            source_union = unary_union(source_geoms)

            source_length = float(
                sum(g.length for g in source_geoms)
            )

            target_length = float(target_geom.length)

            length_ratio = (
                target_length / source_length
                if source_length > 0
                else np.nan
            )

            geometry_distance = float(
                source_union.distance(target_geom)
            )

            source_vertices = count_vertices(source_union)
            target_vertices = count_vertices(target_geom)

            vertex_ratio = (
                target_vertices / source_vertices
                if source_vertices > 0
                else np.nan
            )

            operation = classify_target_operation(
                source_geoms,
                target_geom,
            )

            operation_rows.append(
                {
                    "city": city_name,
                    "target_id": target_id,
                    "source_count": source_count,
                    "operation": operation,
                    "source_total_length_m": source_length,
                    "target_length_m": target_length,
                    "length_ratio": length_ratio,
                    "geometry_distance_m": geometry_distance,
                    "source_vertices": source_vertices,
                    "target_vertices": target_vertices,
                    "vertex_ratio": vertex_ratio,
                }
            )

            if i % 250 == 0 or i == len(target_ids):
                print(
                    f"  {i:,}/{len(target_ids):,}"
                )

        operations = pd.DataFrame(operation_rows)

        print()

        if len(operations):

            counts = operations[
                "operation"
            ].value_counts()

            print("ROAD OPERATION DISTRIBUTION")
            print("-" * 120)

            for op, count in counts.items():

                pct = (
                    100.0 * count / len(operations)
                )

                print(
                    f"{op:<35}"
                    f"{count:>8,} "
                    f"({pct:>6.2f}%)"
                )

        # --------------------------------------------------------------------------------------------------------------
        # SOURCE SELECTION / ELIMINATION
        # --------------------------------------------------------------------------------------------------------------

        subheader("SOURCE ROAD SELECTION / ELIMINATION")

        label_col = None

        for candidate in [
            "selection_label",
            "label",
            "operation",
        ]:
            if candidate in source_labels.columns:
                label_col = candidate
                break

        if label_col is None:
            raise RuntimeError(
                "Could not find selection label column in:\n"
                f"{list(source_labels.columns)}"
            )

        label_counts = (
            source_labels[label_col]
            .astype(str)
            .value_counts()
        )

        for label, count in label_counts.items():
            pct = (
                100.0 * count / len(source_labels)
            )

            print(
                f"{label:<30}"
                f"{count:>8,} "
                f"({pct:>6.2f}%)"
            )

        selected_count = int(
            label_counts.get("selected", 0)
            + label_counts.get("weak_selected", 0)
        )

        strong_selected_count = int(
            label_counts.get("selected", 0)
        )

        weak_selected_count = int(
            label_counts.get("weak_selected", 0)
        )

        eliminated_count = int(
            label_counts.get("elimination", 0)
        )

        selection_pct = (
            100.0 * selected_count / len(source_labels)
            if len(source_labels)
            else np.nan
        )

        elimination_pct = (
            100.0 * eliminated_count / len(source_labels)
            if len(source_labels)
            else np.nan
        )

        print()
        print(
            f"Selected source roads:   "
            f"{selected_count:,}"
        )
        print(
            f"Strong selected roads:   "
            f"{strong_selected_count:,}"
        )
        print(
            f"Weak selected roads:     "
            f"{weak_selected_count:,}"
        )
        print(
            f"Eliminated source roads: "
            f"{eliminated_count:,}"
        )
        print(
            f"Selection rate:          "
            f"{selection_pct:.2f}%"
        )
        print(
            f"Elimination rate:        "
            f"{elimination_pct:.2f}%"
        )

        # --------------------------------------------------------------------------------------------------------------
        # BUILD NETWORK GRAPHS
        # --------------------------------------------------------------------------------------------------------------

        subheader("BUILDING BDOT10k ROAD GRAPH")

        source_graph, source_graph_length = (
            graph_from_lines(
                source,
                NODE_SNAP_M,
            )
        )

        source_graph_summary = summarize_graph(
            source_graph,
            source_graph_length,
        )

        print(
            f"Nodes:      "
            f"{source_graph_summary['nodes']:,}"
        )
        print(
            f"Edges:      "
            f"{source_graph_summary['edges']:,}"
        )
        print(
            f"Components: "
            f"{source_graph_summary['components']:,}"
        )

        subheader("BUILDING BDOO ROAD GRAPH")

        target_graph, target_graph_length = (
            graph_from_lines(
                target,
                NODE_SNAP_M,
            )
        )

        target_graph_summary = summarize_graph(
            target_graph,
            target_graph_length,
        )

        print(
            f"Nodes:      "
            f"{target_graph_summary['nodes']:,}"
        )
        print(
            f"Edges:      "
            f"{target_graph_summary['edges']:,}"
        )
        print(
            f"Components: "
            f"{target_graph_summary['components']:,}"
        )

        # --------------------------------------------------------------------------------------------------------------
        # TOPOLOGY SUMMARY
        # --------------------------------------------------------------------------------------------------------------

        subheader("TOPOLOGY SUMMARY")

        metrics = [
            "nodes",
            "edges",
            "components",
            "largest_component_pct",
            "mean_degree",
            "dead_ends",
            "junctions",
            "cycle_rank",
            "total_graph_length_m",
        ]

        print(
            f"{'Metric':<35}"
            f"{'BDOT10k':>18}"
            f"{'BDOO':>18}"
        )

        print("-" * 71)

        for metric in metrics:

            s = source_graph_summary[metric]
            t = target_graph_summary[metric]

            if isinstance(s, float) or isinstance(t, float):
                print(
                    f"{metric:<35}"
                    f"{s:>18.3f}"
                    f"{t:>18.3f}"
                )
            else:
                print(
                    f"{metric:<35}"
                    f"{s:>18,}"
                    f"{t:>18,}"
                )

        node_retention = retention(
            target_graph_summary["nodes"],
            source_graph_summary["nodes"],
        )

        edge_retention = retention(
            target_graph_summary["edges"],
            source_graph_summary["edges"],
        )

        junction_retention = retention(
            target_graph_summary["junctions"],
            source_graph_summary["junctions"],
        )

        cycle_retention = retention(
            target_graph_summary["cycle_rank"],
            source_graph_summary["cycle_rank"],
        )

        graph_length_retention = retention(
            target_graph_summary["total_graph_length_m"],
            source_graph_summary["total_graph_length_m"],
        )

        print()
        print(
            f"Node retention:         "
            f"{node_retention:.2f}%"
        )
        print(
            f"Graph-edge retention:   "
            f"{edge_retention:.2f}%"
        )
        print(
            f"Junction retention:     "
            f"{junction_retention:.2f}%"
        )
        print(
            f"Cycle retention:        "
            f"{cycle_retention:.2f}%"
        )
        print(
            f"Network-length retention: "
            f"{graph_length_retention:.2f}%"
        )

        # --------------------------------------------------------------------------------------------------------------
        # OPERATION STATISTICS
        # --------------------------------------------------------------------------------------------------------------

        if len(operations):

            mean_length_ratio = safe_float(
                operations["length_ratio"].mean()
            )

            median_length_ratio = safe_float(
                operations["length_ratio"].median()
            )

            mean_distance = safe_float(
                operations["geometry_distance_m"].mean()
            )

            median_distance = safe_float(
                operations["geometry_distance_m"].median()
            )

            mean_vertex_ratio = safe_float(
                operations["vertex_ratio"].mean()
            )

            median_vertex_ratio = safe_float(
                operations["vertex_ratio"].median()
            )

        else:
            mean_length_ratio = np.nan
            median_length_ratio = np.nan
            mean_distance = np.nan
            median_distance = np.nan
            mean_vertex_ratio = np.nan
            median_vertex_ratio = np.nan

        print()
        print(
            f"Mean target/source length ratio:   "
            f"{mean_length_ratio:.3f}"
        )
        print(
            f"Median target/source length ratio: "
            f"{median_length_ratio:.3f}"
        )
        print(
            f"Mean geometry distance:            "
            f"{mean_distance:.3f} m"
        )
        print(
            f"Median geometry distance:          "
            f"{median_distance:.3f} m"
        )
        print(
            f"Mean vertex ratio:                 "
            f"{mean_vertex_ratio:.3f}"
        )
        print(
            f"Median vertex ratio:               "
            f"{median_vertex_ratio:.3f}"
        )

        # --------------------------------------------------------------------------------------------------------------
        # SAVE CITY OUTPUTS
        # --------------------------------------------------------------------------------------------------------------

        subheader("SAVING CITY OUTPUTS")

        operations_csv = (
            OUTPUT_DIR
            / f"{city_key}_road_operations.csv"
        )

        operations.to_csv(
            operations_csv,
            index=False,
        )

        print(f"Operations CSV:\n{operations_csv}")

        topology_rows = []

        for network_name, summary in [
            ("BDOT10k_source", source_graph_summary),
            ("BDOO_target", target_graph_summary),
        ]:

            row = {
                "city": city_name,
                "network": network_name,
            }

            row.update(summary)

            topology_rows.append(row)

        topology_df = pd.DataFrame(topology_rows)

        topology_csv = (
            OUTPUT_DIR
            / f"{city_key}_road_topology.csv"
        )

        topology_df.to_csv(
            topology_csv,
            index=False,
        )

        print(f"\nTopology CSV:\n{topology_csv}")

        # --------------------------------------------------------------------------------------------------------------
        # GPKG DIAGNOSTIC
        # --------------------------------------------------------------------------------------------------------------

        gpkg_out = (
            OUTPUT_DIR
            / f"{city_key}_road_topology.gpkg"
        )

        if gpkg_out.exists():
            gpkg_out.unlink()

        source.to_file(
            gpkg_out,
            layer="source_roads",
            driver="GPKG",
        )

        target.to_file(
            gpkg_out,
            layer="target_roads",
            driver="GPKG",
        )

        print(f"\nTopology GPKG:\n{gpkg_out}")

        # --------------------------------------------------------------------------------------------------------------
        # GLOBAL SUMMARY
        # --------------------------------------------------------------------------------------------------------------

        operation_counts = (
            operations["operation"]
            .value_counts()
            .to_dict()
            if len(operations)
            else {}
        )

        summary = {
            "city": city_name,

            "source_count": int(len(source)),
            "target_count": int(len(target)),

            "selected_source": selected_count,
            "strong_selected_source": strong_selected_count,
            "weak_selected_source": weak_selected_count,
            "eliminated_source": eliminated_count,

            "source_selection_pct": selection_pct,
            "source_elimination_pct": elimination_pct,

            "matched_targets_analyzed": int(
                len(operations)
            ),

            "retained_or_simplified": int(
                operation_counts.get(
                    "retained_or_simplified",
                    0,
                )
            ),

            "merge_or_chain": int(
                operation_counts.get(
                    "merge_or_chain",
                    0,
                )
            ),

            "displacement_or_reshape": int(
                operation_counts.get(
                    "displacement_or_reshape",
                    0,
                )
            ),

            "mean_length_ratio": mean_length_ratio,
            "median_length_ratio": median_length_ratio,

            "mean_geometry_distance_m": mean_distance,
            "median_geometry_distance_m": median_distance,

            "mean_vertex_ratio": mean_vertex_ratio,
            "median_vertex_ratio": median_vertex_ratio,

            "source_nodes": source_graph_summary["nodes"],
            "target_nodes": target_graph_summary["nodes"],

            "source_edges": source_graph_summary["edges"],
            "target_edges": target_graph_summary["edges"],

            "source_components": source_graph_summary[
                "components"
            ],

            "target_components": target_graph_summary[
                "components"
            ],

            "source_largest_component_pct":
                source_graph_summary[
                    "largest_component_pct"
                ],

            "target_largest_component_pct":
                target_graph_summary[
                    "largest_component_pct"
                ],

            "source_mean_degree":
                source_graph_summary["mean_degree"],

            "target_mean_degree":
                target_graph_summary["mean_degree"],

            "source_dead_ends":
                source_graph_summary["dead_ends"],

            "target_dead_ends":
                target_graph_summary["dead_ends"],

            "source_junctions":
                source_graph_summary["junctions"],

            "target_junctions":
                target_graph_summary["junctions"],

            "source_cycle_rank":
                source_graph_summary["cycle_rank"],

            "target_cycle_rank":
                target_graph_summary["cycle_rank"],

            "source_graph_length_m":
                source_graph_summary[
                    "total_graph_length_m"
                ],

            "target_graph_length_m":
                target_graph_summary[
                    "total_graph_length_m"
                ],

            "node_retention_pct": node_retention,
            "edge_retention_pct": edge_retention,
            "junction_retention_pct": junction_retention,
            "cycle_retention_pct": cycle_retention,
            "network_length_retention_pct":
                graph_length_retention,
        }

        global_summary.append(summary)

        if len(operations):
            global_operations.append(
                operations.copy()
            )

    # ==================================================================================================================
    # GLOBAL OUTPUTS
    # ==================================================================================================================

    header("GLOBAL POLISH ROAD GENERALIZATION SUMMARY")

    summary_df = pd.DataFrame(global_summary)

    display_columns = [
        "city",
        "source_count",
        "target_count",
        "source_selection_pct",
        "source_elimination_pct",
        "retained_or_simplified",
        "merge_or_chain",
        "displacement_or_reshape",
        "junction_retention_pct",
        "cycle_retention_pct",
        "network_length_retention_pct",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(index=False)
    )

    summary_csv = (
        OUTPUT_DIR
        / "poland_road_topology_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    if global_operations:

        operations_global = pd.concat(
            global_operations,
            ignore_index=True,
        )

    else:
        operations_global = pd.DataFrame()

    global_operations_csv = (
        OUTPUT_DIR
        / "poland_road_operations.csv"
    )

    operations_global.to_csv(
        global_operations_csv,
        index=False,
    )

    report = {
        "dataset": "CartoGen Poland",
        "source": "BDOT10k",
        "target": "BDOO",
        "crs": CRS,
        "node_snap_tolerance_m": NODE_SNAP_M,
        "cities": global_summary,
    }

    report_json = (
        OUTPUT_DIR
        / "poland_road_topology_summary.json"
    )

    with open(
        report_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_safe(report),
            f,
            indent=2,
            ensure_ascii=False,
        )

    header("SCRIPT 60 COMPLETE")

    print(f"\nGlobal operations:")
    print(global_operations_csv)

    print(f"\nSummary CSV:")
    print(summary_csv)

    print(f"\nSummary JSON:")
    print(report_json)

    print(
        "\nNEXT: Script 61 — build the unified Polish "
        "building + road learning dataset."
    )


if __name__ == "__main__":
    main()