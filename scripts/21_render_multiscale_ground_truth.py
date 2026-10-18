from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "os"
    / "development_windows"
)

FIGURE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "multiscale_ground_truth"
)

WINDOWS = {
    "london_central": {
        "city": "London",
        "center_x": 530000,
        "center_y": 180000,
    },

    "manchester_central": {
        "city": "Manchester",
        "center_x": 384000,
        "center_y": 398000,
    },

    "birmingham_central": {
        "city": "Birmingham",
        "center_x": 407500,
        "center_y": 287000,
    },
}


VISUAL_SIZE_M = 2000


def visualization_bbox(
    center_x,
    center_y
):

    half = (
        VISUAL_SIZE_M / 2
    )

    return box(
        center_x - half,
        center_y - half,
        center_x + half,
        center_y + half,
    )


def clip_layer(
    gpkg,
    layer,
    bbox
):

    bounds = bbox.bounds

    gdf = gpd.read_file(
        gpkg,
        layer=layer,
        bbox=bounds,
        engine="pyogrio"
    )

    if gdf.empty:
        return gdf

    gdf = gdf[
        gdf.geometry.intersects(
            bbox
        )
    ].copy()

    gdf["geometry"] = (
        gdf.geometry.intersection(
            bbox
        )
    )

    gdf = gdf[
        ~gdf.geometry.is_empty
    ].copy()

    return gdf


def configure_axis(
    ax,
    bbox,
    title
):

    min_x, min_y, max_x, max_y = (
        bbox.bounds
    )

    ax.set_xlim(
        min_x,
        max_x
    )

    ax.set_ylim(
        min_y,
        max_y
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_title(
        title,
        fontsize=12
    )

    ax.set_xticks([])
    ax.set_yticks([])


def render_buildings(
    window_id,
    city,
    gpkg,
    bbox
):

    local = clip_layer(
        gpkg,
        "local_buildings",
        bbox
    )

    district = clip_layer(
        gpkg,
        "district_buildings",
        bbox
    )

    print(
        f"  Building visual subset: "
        f"{len(local):,} local, "
        f"{len(district):,} district"
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6)
    )

    if not local.empty:

        local.plot(
            ax=axes[0],
            facecolor="black",
            edgecolor="black",
            linewidth=0.15
        )

    configure_axis(
        axes[0],
        bbox,
        "Detailed / Local Buildings"
    )

    if not district.empty:

        district.plot(
            ax=axes[1],
            facecolor="black",
            edgecolor="black",
            linewidth=0.15
        )

    configure_axis(
        axes[1],
        bbox,
        "District Generalized Buildings"
    )

    fig.suptitle(
        f"{city}: Building Generalization Ground Truth",
        fontsize=14
    )

    fig.tight_layout()

    png = (
        FIGURE_ROOT
        / f"{window_id}_buildings.png"
    )

    pdf = (
        FIGURE_ROOT
        / f"{window_id}_buildings.pdf"
    )

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

    return png


def render_roads(
    window_id,
    city,
    gpkg,
    bbox
):

    local = clip_layer(
        gpkg,
        "roads_local",
        bbox
    )

    regional = clip_layer(
        gpkg,
        "roads_regional",
        bbox
    )

    national = clip_layer(
        gpkg,
        "roads_national",
        bbox
    )

    print(
        f"  Road visual subset: "
        f"{len(local):,} local, "
        f"{len(regional):,} regional, "
        f"{len(national):,} national"
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    road_sets = [
        (
            local,
            "Detailed / Local Roads",
            0.35
        ),
        (
            regional,
            "Regional Generalized Roads",
            0.55
        ),
        (
            national,
            "National Generalized Roads",
            0.75
        ),
    ]

    for ax, (
        gdf,
        title,
        width
    ) in zip(
        axes,
        road_sets
    ):

        if not gdf.empty:

            gdf.plot(
                ax=ax,
                color="black",
                linewidth=width
            )

        configure_axis(
            ax,
            bbox,
            title
        )

    fig.suptitle(
        f"{city}: Road Generalization Ground Truth",
        fontsize=14
    )

    fig.tight_layout()

    png = (
        FIGURE_ROOT
        / f"{window_id}_roads.png"
    )

    pdf = (
        FIGURE_ROOT
        / f"{window_id}_roads.pdf"
    )

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

    return png


def main():

    print("=" * 110)
    print(
        "CARTOGEN — RENDER "
        "MULTI-SCALE GROUND TRUTH"
    )
    print("=" * 110)

    FIGURE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    for window_id, info in (
        WINDOWS.items()
    ):

        city = info["city"]

        gpkg = (
            DATA_ROOT
            / f"{window_id}.gpkg"
        )

        if not gpkg.exists():

            raise FileNotFoundError(
                gpkg
            )

        bbox = visualization_bbox(
            info["center_x"],
            info["center_y"]
        )

        print()
        print("=" * 110)
        print(city)
        print("=" * 110)

        building_figure = (
            render_buildings(
                window_id,
                city,
                gpkg,
                bbox
            )
        )

        road_figure = (
            render_roads(
                window_id,
                city,
                gpkg,
                bbox
            )
        )

        print(
            f"  Buildings:\n"
            f"  {building_figure}"
        )

        print(
            f"  Roads:\n"
            f"  {road_figure}"
        )

    print()
    print("=" * 110)
    print("SCRIPT 21 COMPLETE")
    print("=" * 110)

    print(
        f"Figures saved in:\n"
        f"{FIGURE_ROOT}"
    )


if __name__ == "__main__":
    main()