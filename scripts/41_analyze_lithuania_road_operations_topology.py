from pathlib import Path
import json
import math
import warnings

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "validated_pairs"
)

CORRESPONDENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "road_correspondence"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "road_topology"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CITIES = {
    "vilnius_central": "Vilnius",
    "kaunas_central": "Kaunas",
    "klaipeda_central": "Klaipeda",
}


# Graph snapping tolerance in metres.
# EPSG:3346 is metric.
SNAP_TOLERANCE = 1.0


# =============================================================================
# BASIC HELPERS
# =============================================================================

def safe_float(value, default=0.0):

    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


def find_column(
    df,
    candidates,
    required=True
):

    lower_map = {
        str(col).lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        if candidate in df.columns:
            return candidate

        key = candidate.lower()

        if key in lower_map:
            return lower_map[key]

    if required:

        raise KeyError(
            "Could not find any of these columns: "
            + ", ".join(candidates)
            + "\nAvailable columns:\n"
            + ", ".join(
                map(str, df.columns)
            )
        )

    return None


def clean_geometry(gdf):

    gdf = gdf.copy()

    gdf = gdf[
        gdf.geometry.notna()
    ].copy()

    gdf = gdf[
        ~gdf.geometry.is_empty
    ].copy()

    return gdf.reset_index(
        drop=True
    )


# =============================================================================
# CORRESPONDENCE COLUMN DETECTION
# =============================================================================

def detect_pair_columns(pairs):

    source_candidates = [
        "source_index",
        "source_idx",
        "source_id",
        "source_fid",
        "source_objectid",
        "src_index",
        "src_idx",
        "src_id",
        "local_index",
        "local_idx",
        "local_id",
    ]

    target_candidates = [
        "target_index",
        "target_idx",
        "target_id",
        "target_fid",
        "target_objectid",
        "tgt_index",
        "tgt_idx",
        "tgt_id",
        "regional_index",
        "regional_idx",
        "regional_id",
    ]

    source_col = find_column(
        pairs,
        source_candidates
    )

    target_col = find_column(
        pairs,
        target_candidates
    )

    return source_col, target_col


# =============================================================================
# MAP PAIR IDENTIFIERS TO ROW INDICES
# =============================================================================

def build_identifier_lookup(
    gdf
):

    lookup = {}

    possible_id_columns = [
        "OBJECTID",
        "objectid",
        "fid",
        "FID",
        "TOP_ID",
        "top_id",
        "id",
        "ID",
    ]

    for col in possible_id_columns:

        if col not in gdf.columns:
            continue

        mapping = {}

        for idx, value in gdf[col].items():

            if pd.isna(value):
                continue

            mapping[str(value)] = idx

        lookup[col] = mapping

    return lookup


def resolve_pair_identifier(
    value,
    gdf,
    lookup
):

    # First try direct dataframe index.
    try:

        numeric = int(float(value))

        if numeric in gdf.index:
            return numeric

    except Exception:
        pass

    value_string = str(value)

    # Then try known identifier fields.
    for mapping in lookup.values():

        if value_string in mapping:
            return mapping[value_string]

    return None


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def snap_coordinate(
    x,
    y,
    tolerance
):

    return (
        round(x / tolerance) * tolerance,
        round(y / tolerance) * tolerance
    )


def geometry_lines(geometry):

    if geometry is None:
        return []

    if geometry.is_empty:
        return []

    geom_type = geometry.geom_type

    if geom_type == "LineString":
        return [geometry]

    if geom_type == "MultiLineString":
        return list(geometry.geoms)

    if geom_type == "GeometryCollection":

        lines = []

        for part in geometry.geoms:
            lines.extend(
                geometry_lines(part)
            )

        return lines

    return []


def build_road_graph(
    gdf,
    tolerance=1.0
):

    graph = nx.MultiGraph()

    for row_idx, geometry in enumerate(
        gdf.geometry
    ):

        for line in geometry_lines(
            geometry
        ):

            coords = list(
                line.coords
            )

            if len(coords) < 2:
                continue

            start = snap_coordinate(
                coords[0][0],
                coords[0][1],
                tolerance
            )

            end = snap_coordinate(
                coords[-1][0],
                coords[-1][1],
                tolerance
            )

            length = float(
                line.length
            )

            if length <= 0:
                continue

            graph.add_edge(
                start,
                end,
                length=length,
                feature_index=row_idx
            )

    return graph


# =============================================================================
# GRAPH STATISTICS
# =============================================================================

def graph_statistics(graph):

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()

    if node_count == 0:

        return {
            "nodes": 0,
            "edges": 0,
            "components": 0,
            "largest_component_nodes": 0,
            "largest_component_pct": 0.0,
            "mean_degree": 0.0,
            "dead_ends": 0,
            "junctions": 0,
            "degree_2_nodes": 0,
            "cycle_rank": 0,
            "total_graph_length_m": 0.0,
        }

    components = list(
        nx.connected_components(
            graph
        )
    )

    component_count = len(
        components
    )

    largest_component_nodes = max(
        len(component)
        for component in components
    )

    largest_component_pct = (
        largest_component_nodes
        / node_count
        * 100.0
    )

    degrees = dict(
        graph.degree()
    )

    degree_values = list(
        degrees.values()
    )

    mean_degree = float(
        np.mean(
            degree_values
        )
    )

    dead_ends = sum(
        degree == 1
        for degree in degree_values
    )

    degree_2_nodes = sum(
        degree == 2
        for degree in degree_values
    )

    junctions = sum(
        degree >= 3
        for degree in degree_values
    )

    # Cyclomatic number for an undirected graph.
    cycle_rank = (
        edge_count
        - node_count
        + component_count
    )

    cycle_rank = max(
        cycle_rank,
        0
    )

    total_length = sum(
        safe_float(data.get("length"))
        for _, _, data
        in graph.edges(data=True)
    )

    return {
        "nodes":
            int(node_count),

        "edges":
            int(edge_count),

        "components":
            int(component_count),

        "largest_component_nodes":
            int(largest_component_nodes),

        "largest_component_pct":
            float(largest_component_pct),

        "mean_degree":
            float(mean_degree),

        "dead_ends":
            int(dead_ends),

        "junctions":
            int(junctions),

        "degree_2_nodes":
            int(degree_2_nodes),

        "cycle_rank":
            int(cycle_rank),

        "total_graph_length_m":
            float(total_length),
    }


# =============================================================================
# OPERATION CLASSIFICATION
# =============================================================================

def classify_correspondence_operation(
    source_degree,
    target_degree
):

    # source_degree:
    # number of targets connected to this source
    #
    # target_degree:
    # number of sources connected to this target

    if (
        source_degree == 1
        and target_degree == 1
    ):
        return "retained_or_simplified"

    if (
        source_degree == 1
        and target_degree > 1
    ):
        return "merge_or_chain"

    if (
        source_degree > 1
        and target_degree == 1
    ):
        return "split_or_resegmentation"

    return "complex_network_transformation"


# =============================================================================
# TARGET-LEVEL OPERATION ANALYSIS
# =============================================================================

def analyze_operations(
    source,
    target,
    pairs,
    source_col,
    target_col
):

    source_lookup = build_identifier_lookup(
        source
    )

    target_lookup = build_identifier_lookup(
        target
    )

    resolved = []

    print(
        "Resolving correspondence identifiers..."
    )

    for row in pairs.itertuples(
        index=False
    ):

        source_value = getattr(
            row,
            source_col
        )

        target_value = getattr(
            row,
            target_col
        )

        source_idx = resolve_pair_identifier(
            source_value,
            source,
            source_lookup
        )

        target_idx = resolve_pair_identifier(
            target_value,
            target,
            target_lookup
        )

        if (
            source_idx is None
            or target_idx is None
        ):
            continue

        resolved.append(
            (
                source_idx,
                target_idx
            )
        )

    resolved = list(
        set(resolved)
    )

    print(
        f"Resolved unique pairs: "
        f"{len(resolved):,}"
    )

    source_to_targets = {}

    target_to_sources = {}

    for source_idx, target_idx in resolved:

        source_to_targets.setdefault(
            source_idx,
            set()
        ).add(
            target_idx
        )

        target_to_sources.setdefault(
            target_idx,
            set()
        ).add(
            source_idx
        )

    matched_sources = set(
        source_to_targets.keys()
    )

    matched_targets = set(
        target_to_sources.keys()
    )

    eliminated_sources = (
        set(source.index)
        - matched_sources
    )

    unmatched_targets = (
        set(target.index)
        - matched_targets
    )

    records = []

    total_targets = len(
        target_to_sources
    )

    print(
        f"Analyzing {total_targets:,} "
        f"matched target roads..."
    )

    for counter, (
        target_idx,
        source_indices
    ) in enumerate(
        target_to_sources.items(),
        start=1
    ):

        target_targets_per_source = [
            len(
                source_to_targets[
                    source_idx
                ]
            )
            for source_idx
            in source_indices
        ]

        max_source_degree = max(
            target_targets_per_source
        )

        source_count = len(
            source_indices
        )

        if (
            source_count == 1
            and max_source_degree == 1
        ):

            operation = (
                "retained_or_simplified"
            )

        elif (
            source_count > 1
            and max_source_degree == 1
        ):

            operation = (
                "merge_or_chain"
            )

        elif (
            source_count == 1
            and max_source_degree > 1
        ):

            operation = (
                "split_or_resegmentation"
            )

        else:

            operation = (
                "complex_network_transformation"
            )

        target_geometry = (
            target.loc[
                target_idx
            ].geometry
        )

        source_geometries = [
            source.loc[
                source_idx
            ].geometry
            for source_idx
            in source_indices
        ]

        source_union = unary_union(
            source_geometries
        )

        source_length = sum(
            geom.length
            for geom
            in source_geometries
        )

        target_length = (
            target_geometry.length
        )

        length_ratio = (
            target_length
            / source_length
            if source_length > 0
            else np.nan
        )

        hausdorff = np.nan

        try:

            hausdorff = (
                source_union
                .hausdorff_distance(
                    target_geometry
                )
            )

        except Exception:
            pass

        records.append(
            {
                "target_index":
                    int(target_idx),

                "source_count":
                    int(source_count),

                "max_targets_per_source":
                    int(max_source_degree),

                "operation":
                    operation,

                "source_total_length_m":
                    float(source_length),

                "target_length_m":
                    float(target_length),

                "length_ratio":
                    float(length_ratio)
                    if np.isfinite(
                        length_ratio
                    )
                    else np.nan,

                "hausdorff_distance_m":
                    float(hausdorff)
                    if np.isfinite(
                        hausdorff
                    )
                    else np.nan,

                "geometry":
                    target_geometry,
            }
        )

        if (
            counter % 500 == 0
            or counter == total_targets
        ):

            print(
                f"  {counter:,}/"
                f"{total_targets:,}"
            )

    operations = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=target.crs
    )

    elimination_records = []

    for source_idx in sorted(
        eliminated_sources
    ):

        elimination_records.append(
            {
                "source_index":
                    int(source_idx),

                "operation":
                    "elimination",

                "length_m":
                    float(
                        source.loc[
                            source_idx
                        ].geometry.length
                    ),

                "geometry":
                    source.loc[
                        source_idx
                    ].geometry,
            }
        )

    eliminations = gpd.GeoDataFrame(
        elimination_records,
        geometry="geometry",
        crs=source.crs
    )

    unmatched_target_records = []

    for target_idx in sorted(
        unmatched_targets
    ):

        unmatched_target_records.append(
            {
                "target_index":
                    int(target_idx),

                "operation":
                    "unmatched_target",

                "length_m":
                    float(
                        target.loc[
                            target_idx
                        ].geometry.length
                    ),

                "geometry":
                    target.loc[
                        target_idx
                    ].geometry,
            }
        )

    unmatched_target_gdf = (
        gpd.GeoDataFrame(
            unmatched_target_records,
            geometry="geometry",
            crs=target.crs
        )
    )

    return (
        operations,
        eliminations,
        unmatched_target_gdf,
        source_to_targets,
        target_to_sources,
    )


# =============================================================================
# RETENTION METRICS
# =============================================================================

def retention_pct(
    target_value,
    source_value
):

    if source_value == 0:
        return 0.0

    return (
        target_value
        / source_value
        * 100.0
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "=" * 120
    )

    print(
        "CARTOGEN — ANALYZE LITHUANIAN "
        "ROAD GENERALIZATION AND TOPOLOGY"
    )

    print(
        "=" * 120
    )

    global_summary = []
    topology_rows = []
    operation_rows = []

    for window_name, city in CITIES.items():

        print()
        print(
            "=" * 120
        )
        print(
            f"{window_name} ({city})"
        )
        print(
            "=" * 120
        )

        gpkg = (
            VALIDATED_DIR
            / f"{window_name}_validated.gpkg"
        )

        pair_csv = (
            CORRESPONDENCE_DIR
            / f"{window_name}_road_pairs.csv"
        )

        if not gpkg.exists():

            raise FileNotFoundError(
                gpkg
            )

        if not pair_csv.exists():

            raise FileNotFoundError(
                pair_csv
            )

        print(
            "Reading source roads..."
        )

        source = gpd.read_file(
            gpkg,
            layer="source_roads"
        )

        print(
            "Reading target roads..."
        )

        target = gpd.read_file(
            gpkg,
            layer="target_roads"
        )

        source = clean_geometry(
            source
        )

        target = clean_geometry(
            target
        )

        print(
            "Reading Script 40 "
            "correspondences..."
        )

        pairs = pd.read_csv(
            pair_csv
        )

        print(
            f"Source roads: "
            f"{len(source):,}"
        )

        print(
            f"Target roads: "
            f"{len(target):,}"
        )

        print(
            f"Correspondence rows: "
            f"{len(pairs):,}"
        )

        source_col, target_col = (
            detect_pair_columns(
                pairs
            )
        )

        print()
        print(
            "Detected pair columns:"
        )

        print(
            f"  source = {source_col}"
        )

        print(
            f"  target = {target_col}"
        )

        (
            operations,
            eliminations,
            unmatched_targets,
            source_to_targets,
            target_to_sources,
        ) = analyze_operations(
            source,
            target,
            pairs,
            source_col,
            target_col
        )

        # =============================================================
        # OPERATION DISTRIBUTION
        # =============================================================

        print()
        print(
            "-" * 120
        )

        print(
            "ROAD OPERATION DISTRIBUTION"
        )

        print(
            "-" * 120
        )

        operation_counts = (
            operations[
                "operation"
            ]
            .value_counts()
            .to_dict()
        )

        matched_target_count = len(
            operations
        )

        for operation, count in sorted(
            operation_counts.items()
        ):

            percentage = (
                count
                / matched_target_count
                * 100
                if matched_target_count
                else 0
            )

            print(
                f"{operation:<38}"
                f"{count:>8,} "
                f"({percentage:>6.2f}%)"
            )

            operation_rows.append(
                {
                    "city":
                        city,

                    "operation":
                        operation,

                    "count":
                        int(count),

                    "percentage":
                        float(
                            percentage
                        ),
                }
            )

        print()
        print(
            f"Eliminated source roads: "
            f"{len(eliminations):,}"
        )

        print(
            f"Unmatched target roads: "
            f"{len(unmatched_targets):,}"
        )

        source_elimination_pct = (
            len(eliminations)
            / len(source)
            * 100
            if len(source)
            else 0
        )

        print(
            f"Source elimination rate: "
            f"{source_elimination_pct:.2f}%"
        )

        if len(operations):

            print(
                f"Mean length ratio: "
                f"{operations['length_ratio'].mean():.3f}"
            )

            print(
                f"Median length ratio: "
                f"{operations['length_ratio'].median():.3f}"
            )

            print(
                f"Mean Hausdorff distance: "
                f"{operations['hausdorff_distance_m'].mean():.3f} m"
            )

            print(
                f"Median Hausdorff distance: "
                f"{operations['hausdorff_distance_m'].median():.3f} m"
            )

        # =============================================================
        # BUILD NETWORKS
        # =============================================================

        print()
        print(
            "Building source road graph..."
        )

        source_graph = build_road_graph(
            source,
            SNAP_TOLERANCE
        )

        print(
            "Building target road graph..."
        )

        target_graph = build_road_graph(
            target,
            SNAP_TOLERANCE
        )

        source_stats = graph_statistics(
            source_graph
        )

        target_stats = graph_statistics(
            target_graph
        )

        # =============================================================
        # TOPOLOGY SUMMARY
        # =============================================================

        print()
        print(
            "-" * 120
        )

        print(
            "TOPOLOGY SUMMARY"
        )

        print(
            "-" * 120
        )

        print(
            f"{'Metric':<38}"
            f"{'Source':>16}"
            f"{'Target':>16}"
        )

        print(
            "-" * 70
        )

        display_metrics = [
            "nodes",
            "edges",
            "components",
            "largest_component_pct",
            "mean_degree",
            "dead_ends",
            "junctions",
            "degree_2_nodes",
            "cycle_rank",
            "total_graph_length_m",
        ]

        for metric in display_metrics:

            source_value = (
                source_stats[
                    metric
                ]
            )

            target_value = (
                target_stats[
                    metric
                ]
            )

            if isinstance(
                source_value,
                float
            ):

                print(
                    f"{metric:<38}"
                    f"{source_value:>16.3f}"
                    f"{target_value:>16.3f}"
                )

            else:

                print(
                    f"{metric:<38}"
                    f"{source_value:>16,}"
                    f"{target_value:>16,}"
                )

        junction_retention = (
            retention_pct(
                target_stats[
                    "junctions"
                ],
                source_stats[
                    "junctions"
                ]
            )
        )

        edge_retention = (
            retention_pct(
                target_stats[
                    "edges"
                ],
                source_stats[
                    "edges"
                ]
            )
        )

        cycle_retention = (
            retention_pct(
                target_stats[
                    "cycle_rank"
                ],
                source_stats[
                    "cycle_rank"
                ]
            )
        )

        length_retention = (
            retention_pct(
                target_stats[
                    "total_graph_length_m"
                ],
                source_stats[
                    "total_graph_length_m"
                ]
            )
        )

        largest_component_change = (
            target_stats[
                "largest_component_pct"
            ]
            - source_stats[
                "largest_component_pct"
            ]
        )

        print()
        print(
            f"Junction retention: "
            f"{junction_retention:.2f}%"
        )

        print(
            f"Graph-edge retention: "
            f"{edge_retention:.2f}%"
        )

        print(
            f"Cycle retention: "
            f"{cycle_retention:.2f}%"
        )

        print(
            f"Network-length retention: "
            f"{length_retention:.2f}%"
        )

        print(
            "Largest-component change: "
            f"{largest_component_change:+.2f} pp"
        )

        # =============================================================
        # SAVE GPKG
        # =============================================================

        output_gpkg = (
            OUTPUT_DIR
            / f"{window_name}_road_topology.gpkg"
        )

        if output_gpkg.exists():
            output_gpkg.unlink()

        operations.to_file(
            output_gpkg,
            layer="road_operations",
            driver="GPKG"
        )

        if len(eliminations):

            eliminations.to_file(
                output_gpkg,
                layer="eliminated_source_roads",
                driver="GPKG"
            )

        if len(unmatched_targets):

            unmatched_targets.to_file(
                output_gpkg,
                layer="unmatched_target_roads",
                driver="GPKG"
            )

        # =============================================================
        # SAVE CSV
        # =============================================================

        operation_csv = (
            OUTPUT_DIR
            / f"{window_name}_road_operations.csv"
        )

        operations.drop(
            columns="geometry"
        ).to_csv(
            operation_csv,
            index=False
        )

        elimination_csv = (
            OUTPUT_DIR
            / f"{window_name}_road_eliminations.csv"
        )

        eliminations.drop(
            columns="geometry"
        ).to_csv(
            elimination_csv,
            index=False
        )

        # =============================================================
        # SUMMARY RECORD
        # =============================================================

        summary = {
            "window":
                window_name,

            "city":
                city,

            "source_count":
                int(len(source)),

            "target_count":
                int(len(target)),

            "matched_source":
                int(
                    len(
                        source_to_targets
                    )
                ),

            "matched_target":
                int(
                    len(
                        target_to_sources
                    )
                ),

            "eliminated_source":
                int(
                    len(eliminations)
                ),

            "unmatched_target":
                int(
                    len(
                        unmatched_targets
                    )
                ),

            "source_elimination_pct":
                float(
                    source_elimination_pct
                ),

            "junction_retention_pct":
                float(
                    junction_retention
                ),

            "edge_retention_pct":
                float(
                    edge_retention
                ),

            "cycle_retention_pct":
                float(
                    cycle_retention
                ),

            "network_length_retention_pct":
                float(
                    length_retention
                ),

            "source_largest_component_pct":
                float(
                    source_stats[
                        "largest_component_pct"
                    ]
                ),

            "target_largest_component_pct":
                float(
                    target_stats[
                        "largest_component_pct"
                    ]
                ),

            "largest_component_change_pp":
                float(
                    largest_component_change
                ),

            "mean_length_ratio":
                (
                    float(
                        operations[
                            "length_ratio"
                        ].mean()
                    )
                    if len(operations)
                    else np.nan
                ),

            "mean_hausdorff_distance_m":
                (
                    float(
                        operations[
                            "hausdorff_distance_m"
                        ].mean()
                    )
                    if len(operations)
                    else np.nan
                ),
        }

        for operation in [
            "retained_or_simplified",
            "merge_or_chain",
            "split_or_resegmentation",
            "complex_network_transformation",
        ]:

            summary[
                f"n_{operation}"
            ] = int(
                operation_counts.get(
                    operation,
                    0
                )
            )

        global_summary.append(
            summary
        )

        for scale, stats in [
            (
                "source",
                source_stats
            ),
            (
                "target",
                target_stats
            ),
        ]:

            topology_rows.append(
                {
                    "window":
                        window_name,

                    "city":
                        city,

                    "scale":
                        scale,

                    **stats,
                }
            )

        print()
        print(
            "Operations CSV:"
        )

        print(
            operation_csv
        )

        print()
        print(
            "Topology GPKG:"
        )

        print(
            output_gpkg
        )

    # =================================================================
    # GLOBAL OUTPUT
    # =================================================================

    summary_df = pd.DataFrame(
        global_summary
    )

    topology_df = pd.DataFrame(
        topology_rows
    )

    operation_df = pd.DataFrame(
        operation_rows
    )

    summary_csv = (
        OUTPUT_DIR
        / "road_topology_summary.csv"
    )

    topology_csv = (
        OUTPUT_DIR
        / "road_graph_statistics.csv"
    )

    operation_summary_csv = (
        OUTPUT_DIR
        / "road_operation_distribution.csv"
    )

    summary_json = (
        OUTPUT_DIR
        / "road_topology_summary.json"
    )

    summary_df.to_csv(
        summary_csv,
        index=False
    )

    topology_df.to_csv(
        topology_csv,
        index=False
    )

    operation_df.to_csv(
        operation_summary_csv,
        index=False
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            global_summary,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        "=" * 120
    )

    print(
        "GLOBAL ROAD TOPOLOGY SUMMARY"
    )

    print(
        "=" * 120
    )

    display_columns = [
        "city",
        "source_count",
        "target_count",
        "matched_source",
        "matched_target",
        "eliminated_source",
        "source_elimination_pct",
        "junction_retention_pct",
        "edge_retention_pct",
        "cycle_retention_pct",
        "network_length_retention_pct",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.2f}"
        )
    )

    print()
    print(
        "=" * 120
    )

    print(
        "SCRIPT 41 COMPLETE"
    )

    print(
        "=" * 120
    )

    print()
    print(
        "Summary CSV:"
    )
    print(
        summary_csv
    )

    print()
    print(
        "Graph statistics:"
    )
    print(
        topology_csv
    )

    print()
    print(
        "Operation distribution:"
    )
    print(
        operation_summary_csv
    )

    print()
    print(
        "Summary JSON:"
    )
    print(
        summary_json
    )


if __name__ == "__main__":
    main() 