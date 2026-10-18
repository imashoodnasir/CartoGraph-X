from pathlib import Path
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx

from shapely.ops import unary_union


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAIR_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cartogen"
    / "road_correspondence_refined"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cartogen"
    / "road_topology"
)

SUMMARY_JSON = (
    OUTPUT_ROOT
    / "road_topology_summary.json"
)

SUMMARY_CSV = (
    OUTPUT_ROOT
    / "road_topology_summary.csv"
)

WINDOWS = [
    "london_central",
    "manchester_central",
    "birmingham_central",
]

# Coordinates within this tolerance are considered
# the same graph node.
NODE_SNAP_TOLERANCE_M = 3.0


# ============================================================
# BASIC GEOMETRY
# ============================================================

def get_lines(geometry):

    if geometry is None or geometry.is_empty:
        return []

    if geometry.geom_type == "LineString":
        return [geometry]

    if geometry.geom_type == "MultiLineString":
        return list(geometry.geoms)

    if geometry.geom_type == "GeometryCollection":

        output = []

        for geom in geometry.geoms:
            output.extend(
                get_lines(geom)
            )

        return output

    return []


def snap_coordinate(
    x,
    y,
    tolerance
):

    return (
        round(x / tolerance) * tolerance,
        round(y / tolerance) * tolerance,
    )


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_graph(
    gdf,
    tolerance
):

    graph = nx.Graph()

    for row in gdf.itertuples():

        geometry = row.geometry

        for line in get_lines(
            geometry
        ):

            coords = list(
                line.coords
            )

            if len(coords) < 2:
                continue

            # Use every consecutive coordinate pair.
            # This preserves bends and intermediate
            # network vertices.

            for i in range(
                len(coords) - 1
            ):

                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]

                node_a = snap_coordinate(
                    x1,
                    y1,
                    tolerance
                )

                node_b = snap_coordinate(
                    x2,
                    y2,
                    tolerance
                )

                if node_a == node_b:
                    continue

                segment_length = (
                    (
                        (x2 - x1) ** 2
                        +
                        (y2 - y1) ** 2
                    )
                    ** 0.5
                )

                if graph.has_edge(
                    node_a,
                    node_b
                ):

                    graph[
                        node_a
                    ][
                        node_b
                    ][
                        "length"
                    ] += segment_length

                else:

                    graph.add_edge(
                        node_a,
                        node_b,
                        length=segment_length
                    )

    return graph


# ============================================================
# GRAPH STATISTICS
# ============================================================

def graph_statistics(
    graph
):

    n_nodes = (
        graph.number_of_nodes()
    )

    n_edges = (
        graph.number_of_edges()
    )

    if n_nodes == 0:

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
            "total_graph_length_m": 0.0,
        }

    components = list(
        nx.connected_components(
            graph
        )
    )

    largest = max(
        components,
        key=len
    )

    degrees = np.array(
        [
            degree
            for _, degree
            in graph.degree()
        ],
        dtype=float
    )

    dead_ends = int(
        np.sum(
            degrees == 1
        )
    )

    junctions = int(
        np.sum(
            degrees >= 3
        )
    )

    component_count = (
        len(components)
    )

    # Cyclomatic number:
    # E - V + C
    cycle_rank = int(
        n_edges
        - n_nodes
        + component_count
    )

    total_length = float(
        sum(
            data.get(
                "length",
                0.0
            )
            for _, _, data
            in graph.edges(
                data=True
            )
        )
    )

    return {
        "nodes":
            int(n_nodes),

        "edges":
            int(n_edges),

        "components":
            int(component_count),

        "largest_component_nodes":
            int(len(largest)),

        "largest_component_pct":
            float(
                100.0
                * len(largest)
                / n_nodes
            ),

        "mean_degree":
            float(
                degrees.mean()
            ),

        "dead_ends":
            dead_ends,

        "junctions":
            junctions,

        "cycle_rank":
            cycle_rank,

        "total_graph_length_m":
            total_length,
    }


# ============================================================
# OPERATION ANALYSIS
# ============================================================

def correspondence_statistics(
    local,
    regional,
    pairs
):

    selected = pairs[
        pairs[
            "selected"
        ].astype(str)
        .str.lower()
        .isin(
            ["true", "1"]
        )
    ].copy()

    matched_local_ids = set(
        selected[
            "local_road_idx"
        ]
        .astype(int)
    )

    matched_regional_ids = set(
        selected[
            "regional_road_idx"
        ]
        .astype(int)
    )

    selected_local = (
        local[
            local[
                "local_road_idx"
            ].isin(
                matched_local_ids
            )
        ].copy()
    )

    eliminated_local = (
        local[
            ~local[
                "local_road_idx"
            ].isin(
                matched_local_ids
            )
        ].copy()
    )

    matched_regional = (
        regional[
            regional[
                "regional_road_idx"
            ].isin(
                matched_regional_ids
            )
        ].copy()
    )

    contributor_counts = (
        selected
        .groupby(
            "regional_road_idx"
        )[
            "local_road_idx"
        ]
        .nunique()
    )

    one_to_one = int(
        (
            contributor_counts == 1
        ).sum()
    )

    many_to_one = int(
        (
            contributor_counts > 1
        ).sum()
    )

    return {
        "selected_pairs":
            selected,

        "selected_local":
            selected_local,

        "eliminated_local":
            eliminated_local,

        "matched_regional":
            matched_regional,

        "one_source_targets":
            one_to_one,

        "multi_source_targets":
            many_to_one,
    }


# ============================================================
# GROUP GEOMETRY ANALYSIS
# ============================================================

def analyze_groups(
    local,
    regional,
    selected_pairs
):

    local_geom = {
        int(row.local_road_idx):
            row.geometry
        for row in local.itertuples()
    }

    regional_geom = {
        int(row.regional_road_idx):
            row.geometry
        for row in regional.itertuples()
    }

    records = []

    groups = (
        selected_pairs
        .groupby(
            "regional_road_idx"
        )
    )

    total = groups.ngroups

    print(
        f"  Analyzing "
        f"{total:,} matched "
        f"regional roads..."
    )

    for counter, (
        regional_idx,
        group
    ) in enumerate(
        groups,
        start=1
    ):

        regional_idx = int(
            regional_idx
        )

        target = (
            regional_geom.get(
                regional_idx
            )
        )

        if target is None:
            continue

        source_ids = (
            group[
                "local_road_idx"
            ]
            .astype(int)
            .unique()
            .tolist()
        )

        source_geometries = [
            local_geom[idx]
            for idx in source_ids
            if idx in local_geom
        ]

        if not source_geometries:
            continue

        source_union = unary_union(
            source_geometries
        )

        source_length = float(
            sum(
                geom.length
                for geom
                in source_geometries
            )
        )

        target_length = float(
            target.length
        )

        length_ratio = (
            target_length
            / source_length
            if source_length > 0
            else 0.0
        )

        centroid_shift = float(
            source_union.centroid.distance(
                target.centroid
            )
        )

        hausdorff = float(
            source_union.hausdorff_distance(
                target
            )
        )

        mean_pair_score = float(
            group[
                "match_score"
            ].mean()
        )

        mean_local_coverage = float(
            group[
                "local_coverage"
            ].mean()
        )

        mean_regional_coverage = float(
            group[
                "regional_coverage"
            ].mean()
        )

        source_count = (
            len(
                source_ids
            )
        )

        if source_count > 1:
            operation = "merge_or_chain"
        else:

            if hausdorff <= 15.0:

                if (
                    0.75
                    <= length_ratio
                    <= 1.25
                ):
                    operation = (
                        "retained_or_simplified"
                    )

                else:
                    operation = (
                        "length_transformation"
                    )

            else:
                operation = (
                    "displacement_or_reshape"
                )

        records.append({

            "regional_road_idx":
                regional_idx,

            "source_count":
                source_count,

            "source_ids":
                ";".join(
                    str(x)
                    for x in source_ids
                ),

            "source_total_length_m":
                source_length,

            "target_length_m":
                target_length,

            "length_ratio":
                length_ratio,

            "centroid_shift_m":
                centroid_shift,

            "hausdorff_distance_m":
                hausdorff,

            "mean_pair_score":
                mean_pair_score,

            "mean_local_coverage":
                mean_local_coverage,

            "mean_regional_coverage":
                mean_regional_coverage,

            "operation":
                operation,
        })

        if (
            counter % 500 == 0
            or counter == total
        ):

            print(
                f"    "
                f"{counter:,}/"
                f"{total:,}"
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 120)
    print(
        "CARTOGEN — ANALYZE ROAD "
        "GENERALIZATION AND TOPOLOGY"
    )
    print("=" * 120)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    all_summaries = {}
    csv_rows = []

    for window in WINDOWS:

        print()
        print("=" * 120)
        print(window)
        print("=" * 120)

        diagnostic = (
            PAIR_ROOT
            / (
                f"{window}_"
                f"refined_road_correspondence.gpkg"
            )
        )

        pair_file = (
            PAIR_ROOT
            / (
                f"{window}_"
                f"refined_road_pairs.csv"
            )
        )

        if not diagnostic.exists():
            raise FileNotFoundError(
                diagnostic
            )

        if not pair_file.exists():
            raise FileNotFoundError(
                pair_file
            )

        local = gpd.read_file(
            diagnostic,
            layer="local_roads",
            engine="pyogrio"
        )

        regional = gpd.read_file(
            diagnostic,
            layer="regional_roads",
            engine="pyogrio"
        )

        pairs = pd.read_csv(
            pair_file
        )

        print(
            f"Local roads: "
            f"{len(local):,}"
        )

        print(
            f"Regional roads: "
            f"{len(regional):,}"
        )

        # ====================================================
        # CORRESPONDENCE OPERATIONS
        # ====================================================

        operation_data = (
            correspondence_statistics(
                local,
                regional,
                pairs
            )
        )

        selected_pairs = (
            operation_data[
                "selected_pairs"
            ]
        )

        selected_local = (
            operation_data[
                "selected_local"
            ]
        )

        eliminated_local = (
            operation_data[
                "eliminated_local"
            ]
        )

        matched_regional = (
            operation_data[
                "matched_regional"
            ]
        )

        print()
        print("-" * 120)
        print(
            "SELECTION / ELIMINATION"
        )
        print("-" * 120)

        print(
            f"Selected local roads: "
            f"{len(selected_local):,}"
        )

        print(
            f"Eliminated local roads: "
            f"{len(eliminated_local):,}"
        )

        selection_pct = (
            100.0
            * len(selected_local)
            / len(local)
            if len(local)
            else 0.0
        )

        elimination_pct = (
            100.0
            * len(eliminated_local)
            / len(local)
            if len(local)
            else 0.0
        )

        print(
            f"Selection rate: "
            f"{selection_pct:.2f}%"
        )

        print(
            f"Elimination rate: "
            f"{elimination_pct:.2f}%"
        )

        print(
            f"One-source targets: "
            f"{operation_data['one_source_targets']:,}"
        )

        print(
            f"Multi-source targets: "
            f"{operation_data['multi_source_targets']:,}"
        )

        # ====================================================
        # GEOMETRIC OPERATIONS
        # ====================================================

        group_results = analyze_groups(
            local,
            regional,
            selected_pairs
        )

        group_csv = (
            OUTPUT_ROOT
            / (
                f"{window}_"
                f"road_operations.csv"
            )
        )

        group_results.to_csv(
            group_csv,
            index=False
        )

        print()
        print("-" * 120)
        print(
            "ROAD OPERATION DISTRIBUTION"
        )
        print("-" * 120)

        operation_counts = (
            group_results[
                "operation"
            ]
            .value_counts()
        )

        for operation, count in (
            operation_counts.items()
        ):

            pct = (
                100.0
                * count
                / len(group_results)
            )

            print(
                f"{operation:<35}"
                f"{count:>7,} "
                f"({pct:6.2f}%)"
            )

        # ====================================================
        # TOPOLOGY
        # ====================================================

        print()
        print(
            "Building local road graph..."
        )

        local_graph = build_graph(
            local,
            NODE_SNAP_TOLERANCE_M
        )

        print(
            "Building regional road graph..."
        )

        regional_graph = build_graph(
            regional,
            NODE_SNAP_TOLERANCE_M
        )

        local_stats = (
            graph_statistics(
                local_graph
            )
        )

        regional_stats = (
            graph_statistics(
                regional_graph
            )
        )

        print()
        print("-" * 120)
        print(
            "TOPOLOGY SUMMARY"
        )
        print("-" * 120)

        print(
            f"{'Metric':<32}"
            f"{'Local':>15}"
            f"{'Regional':>15}"
        )

        print("-" * 62)

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

        for metric in metrics:

            local_value = (
                local_stats[
                    metric
                ]
            )

            regional_value = (
                regional_stats[
                    metric
                ]
            )

            if isinstance(
                local_value,
                float
            ):

                print(
                    f"{metric:<32}"
                    f"{local_value:>15.3f}"
                    f"{regional_value:>15.3f}"
                )

            else:

                print(
                    f"{metric:<32}"
                    f"{local_value:>15,}"
                    f"{regional_value:>15,}"
                )

        # ====================================================
        # RETENTION METRICS
        # ====================================================

        junction_retention = (
            100.0
            * regional_stats[
                "junctions"
            ]
            / local_stats[
                "junctions"
            ]
            if local_stats[
                "junctions"
            ] > 0
            else 0.0
        )

        edge_retention = (
            100.0
            * regional_stats[
                "edges"
            ]
            / local_stats[
                "edges"
            ]
            if local_stats[
                "edges"
            ] > 0
            else 0.0
        )

        cycle_retention = (
            100.0
            * regional_stats[
                "cycle_rank"
            ]
            / local_stats[
                "cycle_rank"
            ]
            if local_stats[
                "cycle_rank"
            ] > 0
            else 0.0
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

        # ====================================================
        # SAVE GEOMETRY DIAGNOSTICS
        # ====================================================

        gpkg_file = (
            OUTPUT_ROOT
            / (
                f"{window}_"
                f"road_topology.gpkg"
            )
        )

        if gpkg_file.exists():
            gpkg_file.unlink()

        selected_local.to_file(
            gpkg_file,
            layer="selected_local_roads",
            driver="GPKG",
            engine="pyogrio"
        )

        eliminated_local.to_file(
            gpkg_file,
            layer="eliminated_local_roads",
            driver="GPKG",
            engine="pyogrio"
        )

        matched_regional.to_file(
            gpkg_file,
            layer="matched_regional_roads",
            driver="GPKG",
            engine="pyogrio"
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        summary = {

            "local_road_count":
                int(len(local)),

            "regional_road_count":
                int(len(regional)),

            "selected_local_count":
                int(
                    len(selected_local)
                ),

            "eliminated_local_count":
                int(
                    len(eliminated_local)
                ),

            "selection_pct":
                float(selection_pct),

            "elimination_pct":
                float(elimination_pct),

            "one_source_targets":
                int(
                    operation_data[
                        "one_source_targets"
                    ]
                ),

            "multi_source_targets":
                int(
                    operation_data[
                        "multi_source_targets"
                    ]
                ),

            "mean_length_ratio":
                float(
                    group_results[
                        "length_ratio"
                    ].mean()
                ),

            "median_length_ratio":
                float(
                    group_results[
                        "length_ratio"
                    ].median()
                ),

            "mean_centroid_shift_m":
                float(
                    group_results[
                        "centroid_shift_m"
                    ].mean()
                ),

            "median_centroid_shift_m":
                float(
                    group_results[
                        "centroid_shift_m"
                    ].median()
                ),

            "mean_hausdorff_distance_m":
                float(
                    group_results[
                        "hausdorff_distance_m"
                    ].mean()
                ),

            "local_topology":
                local_stats,

            "regional_topology":
                regional_stats,

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
        }

        all_summaries[
            window
        ] = summary

        csv_rows.append({

            "window":
                window,

            "local_roads":
                len(local),

            "regional_roads":
                len(regional),

            "selection_pct":
                selection_pct,

            "elimination_pct":
                elimination_pct,

            "local_nodes":
                local_stats[
                    "nodes"
                ],

            "regional_nodes":
                regional_stats[
                    "nodes"
                ],

            "local_edges":
                local_stats[
                    "edges"
                ],

            "regional_edges":
                regional_stats[
                    "edges"
                ],

            "local_components":
                local_stats[
                    "components"
                ],

            "regional_components":
                regional_stats[
                    "components"
                ],

            "local_junctions":
                local_stats[
                    "junctions"
                ],

            "regional_junctions":
                regional_stats[
                    "junctions"
                ],

            "junction_retention_pct":
                junction_retention,

            "edge_retention_pct":
                edge_retention,

            "cycle_retention_pct":
                cycle_retention,
        })

        print()
        print(
            f"Operations CSV:\n"
            f"{group_csv}"
        )

        print(
            f"\nTopology GPKG:\n"
            f"{gpkg_file}"
        )

    # ========================================================
    # GLOBAL OUTPUTS
    # ========================================================

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_summaries,
            file,
            indent=2
        )

    pd.DataFrame(
        csv_rows
    ).to_csv(
        SUMMARY_CSV,
        index=False
    )

    print()
    print("=" * 120)
    print("SCRIPT 26 COMPLETE")
    print("=" * 120)

    print(
        f"Summary JSON:\n"
        f"{SUMMARY_JSON}"
    )

    print(
        f"\nSummary CSV:\n"
        f"{SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()