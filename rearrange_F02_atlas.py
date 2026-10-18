
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

CITIES = [
    "Vilnius",
    "Kaunas",
    "Klaipeda",
    "Warsaw",
    "Krakow",
    "Gdansk"
]

CITY_LABELS = [
    "Vilnius (LT)",
    "Kaunas (LT)",
    "Klaipėda (LT)",
    "Warsaw (PL)",
    "Kraków (PL)",
    "Gdańsk (PL)"
]

ROW_LABELS = [
    "Detailed source",
    "Generalized target",
    "CartoGraph-X retained network"
]

TITLE = (
    "Six-city cartographic atlas | "
    "Actual source, reference and predicted road geometries"
)


# ============================================================
# FIND ORIGINAL FIGURE
# ============================================================

def find_original(root):

    candidates = list(
        root.rglob(
            "F02_six_city_source_reference_prediction_atlas.png"
        )
    )

    if not candidates:
        raise FileNotFoundError(
            "Original F02 PNG was not found under:\n"
            f"{root}\n\n"
            "Search manually using:\n"
            "Get-ChildItem -Recurse -Filter "
            "F02_six_city_source_reference_prediction_atlas.png"
        )

    candidates.sort(
        key=lambda p: (
            "final_publication_package" not in str(p).lower(),
            len(str(p))
        )
    )

    return candidates[0]


# ============================================================
# EXTRACT ORIGINAL MAP PANELS
# ============================================================

def extract_panels(image):

    """
    Original layout:
        6 city rows
        3 representation columns

    New layout:
        3 representation rows
        6 city columns

    Crop boundaries are proportional to the original
    supplied F02 image (895 x 1670).

    These boundaries preserve map content while excluding
    original row/column labels.
    """

    width, height = image.size

    # Original image reference dimensions
    ref_w = 895
    ref_h = 1670

    # Original map column boundaries
    # Each entry: left, top-independent right
    columns = [
        (28, 263),
        (339, 575),
        (651, 887)
    ]

    # Original map row boundaries
    rows = [
        (75, 310),
        (332, 568),
        (590, 825),
        (847, 1082),
        (1104, 1340),
        (1362, 1597)
    ]

    sx = width / ref_w
    sy = height / ref_h

    panels = []

    for row_index in range(6):

        city_panels = []

        for col_index in range(3):

            x0, x1 = columns[col_index]
            y0, y1 = rows[row_index]

            box = (
                round(x0 * sx),
                round(y0 * sy),
                round(x1 * sx),
                round(y1 * sy)
            )

            panel = image.crop(box)
            city_panels.append(panel)

        panels.append(city_panels)

    return panels


# ============================================================
# GENERATE NEW 6 x 3 ATLAS
# ============================================================

def generate_atlas(panels, output_dir):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "figure.facecolor": "white",
        "savefig.facecolor": "white"
    })

    fig, axes = plt.subplots(
        nrows=3,
        ncols=6,
        figsize=(24, 12),
        constrained_layout=False
    )

    # Transpose original city x representation structure
    for representation_index in range(3):

        for city_index in range(6):

            ax = axes[
                representation_index,
                city_index
            ]

            panel = panels[
                city_index
            ][
                representation_index
            ]

            ax.imshow(
                np.asarray(panel),
                interpolation="nearest"
            )

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

            # Column headings
            if representation_index == 0:

                ax.set_title(
                    CITY_LABELS[city_index],
                    fontsize=13,
                    fontweight="bold",
                    pad=12
                )

    # Row headings on left
    for row_index, label in enumerate(ROW_LABELS):

        axes[row_index, 0].annotate(
            label,
            xy=(-0.15, 0.5),
            xycoords="axes fraction",
            rotation=90,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold"
        )

    fig.suptitle(
        TITLE,
        fontsize=18,
        fontweight="bold",
        y=0.985
    )

    fig.subplots_adjust(
        left=0.055,
        right=0.99,
        bottom=0.035,
        top=0.90,
        wspace=0.045,
        hspace=0.10
    )

    png_path = (
        output_dir /
        "F02_six_city_atlas_6cols_3rows.png"
    )

    pdf_path = (
        output_dir /
        "F02_six_city_atlas_6cols_3rows.pdf"
    )

    fig.savefig(
        png_path,
        dpi=350,
        bbox_inches="tight",
        pad_inches=0.12
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.12
    )

    plt.close(fig)

    print("\nFIGURE GENERATED")
    print("PNG:", png_path)
    print("PDF:", pdf_path)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=r"D:\Research\CartoGen"
    )

    parser.add_argument(
        "--input",
        default=None,
        help="Optional exact path to original F02 PNG"
    )

    parser.add_argument(
        "--output",
        default="CartoGraphX_Publication_Figures/02_Main_Results"
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    if args.input:
        original = Path(args.input).resolve()
    else:
        original = find_original(root)

    if not original.exists():
        raise FileNotFoundError(original)

    output_dir = Path(args.output)

    if not output_dir.is_absolute():
        output_dir = root / output_dir

    print("Original figure:", original)

    image = Image.open(original).convert("RGB")

    print("Original dimensions:", image.size)

    panels = extract_panels(image)

    print("Extracted panels:", sum(map(len, panels)))

    generate_atlas(
        panels,
        output_dir
    )


if __name__ == "__main__":
    main()
