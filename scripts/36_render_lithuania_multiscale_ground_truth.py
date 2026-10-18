from pathlib import Path
import json

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lithuania"
    / "validated_pairs"
)

WINDOW_JSON = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "lithuania"
    / "windows"
    / "lithuania_development_windows.json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "lithuania_multiscale_ground_truth"
)

DPI = 300


# ============================================================
# PLOTTING HELPERS
# ============================================================

def configure_axis(
    ax,
    bbox,
    title
):

    xmin, ymin, xmax, ymax = bbox

    ax.set_xlim(
        xmin,
        xmax
    )

    ax.set_ylim(
        ymin,
        ymax
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_title(
        title,
        fontsize=11
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def add_scale_bar(
    ax,
    bbox,
    length_m=1000
):

    xmin, ymin, xmax, ymax = bbox

    width = xmax - xmin
    height = ymax - ymin

    x0 = (
        xmin
        + 0.06 * width
    )

    y0 = (
        ymin
        + 0.055 * height
    )

    ax.plot(
        [
            x0,
            x0 + length_m
        ],
        [
            y0,
            y0
        ],
        linewidth=2
    )

    ax.plot(
        [x0, x0],
        [
            y0 - 0.008 * height,
            y0 + 0.008 * height
        ],
        linewidth=1.5
    )

    ax.plot(
        [
            x0 + length_m,
            x0 + length_m
        ],
        [
            y0 - 0.008 * height,
            y0 + 0.008 * height
        ],
        linewidth=1.5
    )

    ax.text(
        x0 + length_m / 2,
        y0 + 0.018 * height,
        f"{length_m / 1000:g} km",
        ha="center",
        va="bottom",
        fontsize=8
    )


def add_window_border(
    ax,
    bbox
):

    xmin, ymin, xmax, ymax = bbox

    rectangle = Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=False,
        linewidth=0.8
    )

    ax.add_patch(
        rectangle
    )


# ============================================================
# BUILDING FIGURE
# ============================================================

def render_buildings(
    city,
    source,
    target,
    bbox,
    output_path
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6)
    )

    source.plot(
        ax=axes[0],
        edgecolor="black",
        facecolor="none",
        linewidth=0.25
    )

    target.plot(
        ax=axes[1],
        edgecolor="black",
        facecolor="none",
        linewidth=0.30
    )

    configure_axis(
        axes[0],
        bbox,
        (
            "Detailed source\n"
            "GRPK buildings (1:10,000)"
        )
    )

    configure_axis(
        axes[1],
        bbox,
        (
            "Generalized target\n"
            "GDR50LT buildings (1:50,000)"
        )
    )

    add_scale_bar(
        axes[0],
        bbox
    )

    add_scale_bar(
        axes[1],
        bbox
    )

    add_window_border(
        axes[0],
        bbox
    )

    add_window_border(
        axes[1],
        bbox
    )

    fig.suptitle(
        (
            f"{city}: Building "
            f"Generalization Ground Truth"
        ),
        fontsize=14,
        y=0.98
    )

    fig.text(
        0.5,
        0.025,
        (
            f"Source features: "
            f"{len(source):,}    |    "
            f"Target features: "
            f"{len(target):,}"
        ),
        ha="center",
        fontsize=9
    )

    plt.tight_layout(
        rect=[
            0,
            0.05,
            1,
            0.95
        ]
    )

    # Save high-resolution PNG
    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    # Save vector PDF
    pdf_path = output_path.with_suffix(".pdf")

    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# ROAD FIGURE
# ============================================================

def render_roads(
    city,
    source,
    target,
    bbox,
    output_path
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6)
    )

    source.plot(
        ax=axes[0],
        linewidth=0.35
    )

    target.plot(
        ax=axes[1],
        linewidth=0.55
    )

    configure_axis(
        axes[0],
        bbox,
        (
            "Detailed source\n"
            "GRPK roads (1:10,000)"
        )
    )

    configure_axis(
        axes[1],
        bbox,
        (
            "Generalized target\n"
            "GDR50LT roads (1:50,000)"
        )
    )

    add_scale_bar(
        axes[0],
        bbox
    )

    add_scale_bar(
        axes[1],
        bbox
    )

    add_window_border(
        axes[0],
        bbox
    )

    add_window_border(
        axes[1],
        bbox
    )

    fig.suptitle(
        (
            f"{city}: Road "
            f"Generalization Ground Truth"
        ),
        fontsize=14,
        y=0.98
    )

    fig.text(
        0.5,
        0.025,
        (
            f"Source features: "
            f"{len(source):,}    |    "
            f"Target features: "
            f"{len(target):,}"
        ),
        ha="center",
        fontsize=9
    )

    plt.tight_layout(
        rect=[
            0,
            0.05,
            1,
            0.95
        ]
    )

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 120)
    print(
        "CARTOGEN — RENDER LITHUANIAN "
        "MULTI-SCALE GROUND TRUTH"
    )
    print("=" * 120)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        WINDOW_JSON,
        "r",
        encoding="utf-8"
    ) as file:

        windows = json.load(
            file
        )

    for window in windows:

        name = (
            window["window"]
        )

        city = (
            window["city"]
        )

        bbox = (
            float(window["xmin"]),
            float(window["ymin"]),
            float(window["xmax"]),
            float(window["ymax"])
        )

        print()
        print("=" * 120)
        print(city)
        print("=" * 120)

        input_file = (
            INPUT_ROOT
            / f"{name}_validated.gpkg"
        )

        if not input_file.exists():

            raise FileNotFoundError(
                input_file
            )

        print(
            "Reading validated layers..."
        )

        source_buildings = (
            gpd.read_file(
                input_file,
                layer="source_buildings",
                engine="pyogrio"
            )
        )

        target_buildings = (
            gpd.read_file(
                input_file,
                layer="target_buildings",
                engine="pyogrio"
            )
        )

        source_roads = (
            gpd.read_file(
                input_file,
                layer="source_roads",
                engine="pyogrio"
            )
        )

        target_roads = (
            gpd.read_file(
                input_file,
                layer="target_roads",
                engine="pyogrio"
            )
        )

        print(
            f"  Source buildings: "
            f"{len(source_buildings):,}"
        )

        print(
            f"  Target buildings: "
            f"{len(target_buildings):,}"
        )

        print(
            f"  Source roads:     "
            f"{len(source_roads):,}"
        )

        print(
            f"  Target roads:     "
            f"{len(target_roads):,}"
        )

        building_output = (
            OUTPUT_ROOT
            / (
                f"{name}_"
                f"buildings.png"
            )
        )

        road_output = (
            OUTPUT_ROOT
            / (
                f"{name}_"
                f"roads.png"
            )
        )

        print()
        print(
            "Rendering buildings..."
        )

        render_buildings(
            city,
            source_buildings,
            target_buildings,
            bbox,
            building_output
        )

        print(
            f"  Saved:\n"
            f"  {building_output}"
        )

        print()
        print(
            "Rendering roads..."
        )

        render_roads(
            city,
            source_roads,
            target_roads,
            bbox,
            road_output
        )

        print(
            f"  Saved:\n"
            f"  {road_output}"
        )

    print()
    print("=" * 120)
    print("SCRIPT 36 COMPLETE")
    print("=" * 120)

    print()
    print(
        "Figures saved in:"
    )

    print(
        OUTPUT_ROOT
    )


if __name__ == "__main__":
    main()