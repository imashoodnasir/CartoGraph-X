import json
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]


TABLES = {
    "T01": "Dataset Statistics",
    "T02": "Overall Generalization Performance",
    "T03": "Building Generalization",
    "T04": "Road Generalization",
    "T05": "Multi-Scale Generalization",
    "T06": "Topology Preservation",
    "T07": "Cartographic Conflict Analysis",
    "T08": "Generalization Operation Recognition",
    "T09": "Cross-City Generalization",
    "T10": "Cross-Dataset OS-to-OSM Generalization",
    "T11": "Architecture Ablation",
    "T12": "Loss Ablation",
    "T13": "Computational Efficiency",
    "T14": "Robustness Analysis",
}


FIGURES = {
    "F01": "CartoGen Architecture",
    "F02": "Multi-Scale Qualitative Results",
    "F03": "Baseline Visual Comparison",
    "F04": "Topology Preservation Examples",
    "F05": "Scale Response Curves",
    "F06": "Generalization Operation Visualization",
    "F07": "OS-to-OSM External Generalization",
    "F08": "Failure Cases",
}


def main():

    manifest = {
        "project": "CartoGen",
        "created": datetime.now().isoformat(),

        "tables": {
            key: {
                "name": value,
                "status": "pending"
            }
            for key, value in TABLES.items()
        },

        "figures": {
            key: {
                "name": value,
                "status": "pending"
            }
            for key, value in FIGURES.items()
        }
    }

    output = (
        PROJECT_ROOT
        / "results"
        / "results_manifest.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4
        )

    print("=" * 80)
    print("CARTOGEN RESULTS MANIFEST")
    print("=" * 80)

    print(
        f"Tables registered: {len(TABLES)}"
    )

    print(
        f"Figures registered: {len(FIGURES)}"
    )

    print(
        f"Saved: {output}"
    )


if __name__ == "__main__":
    main()