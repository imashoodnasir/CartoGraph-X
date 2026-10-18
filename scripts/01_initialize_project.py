from pathlib import Path
import json
import platform
import sys
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DIRECTORIES = [
    # Configuration
    "configs",

    # Data
    "data/raw/os",
    "data/raw/osm",
    "data/interim/os",
    "data/interim/osm",
    "data/processed/os",
    "data/processed/osm",
    "data/processed/paired",
    "data/processed/graphs",
    "data/splits",

    # Source
    "src",
    "src/data",
    "src/geometry",
    "src/topology",
    "src/models",
    "src/losses",
    "src/baselines",
    "src/evaluation",
    "src/visualization",
    "src/utils",

    # Experiments
    "experiments/main",
    "experiments/multiscale",
    "experiments/cross_city",
    "experiments/cross_dataset",
    "experiments/ablation_architecture",
    "experiments/ablation_loss",
    "experiments/robustness",
    "experiments/efficiency",

    # Models
    "checkpoints",

    # Results
    "results/runs",
    "results/tables",
    "results/figures",
    "results/maps",
    "results/statistics",
    "results/logs",

    # Scripts
    "scripts",

    # Temporary
    "tmp",
]


INIT_PACKAGES = [
    "src",
    "src/data",
    "src/geometry",
    "src/topology",
    "src/models",
    "src/losses",
    "src/baselines",
    "src/evaluation",
    "src/visualization",
    "src/utils",
]


def create_directories():
    print("=" * 80)
    print("CREATING CARTOGEN PROJECT STRUCTURE")
    print("=" * 80)

    for directory in DIRECTORIES:
        path = PROJECT_ROOT / directory
        path.mkdir(parents=True, exist_ok=True)
        print(f"[OK] {path.relative_to(PROJECT_ROOT)}")


def create_init_files():
    for package in INIT_PACKAGES:
        init_file = PROJECT_ROOT / package / "__init__.py"

        if not init_file.exists():
            init_file.touch()

        print(f"[OK] {init_file.relative_to(PROJECT_ROOT)}")


def create_project_metadata():
    metadata = {
        "project": "CartoGen",
        "title": (
            "Topology-Constrained Generative AI for Automated "
            "Multi-Scale Cartographic Generalization of Urban Maps"
        ),
        "created": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "datasets": [
            "Ordnance Survey Great Britain",
            "OpenStreetMap",
        ],
        "objects": [
            "buildings",
            "roads",
        ],
        "target_scales": [
            25000,
            50000,
            100000,
        ],
        "number_of_runs": 10,
        "primary_output": "GeoPackage",
    }

    output = PROJECT_ROOT / "project_metadata.json"

    with open(output, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[OK] {output.relative_to(PROJECT_ROOT)}")


def main():
    create_directories()
    create_init_files()
    create_project_metadata()

    print()
    print("=" * 80)
    print("CARTOGEN INITIALIZATION COMPLETE")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")


if __name__ == "__main__":
    main()