import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

import numpy
import pandas
import geopandas
import shapely
import pyproj
import networkx
import sklearn
import scipy
import torch
import yaml

from src.utils.config import (
    load_main_config,
    load_experiment_config,
)

from src.utils.reproducibility import (
    set_seed,
    print_device_information,
)


def print_package(name, version):
    print(
        f"{name:<20} {version}"
    )


def main():

    print("=" * 80)
    print("CARTOGEN ENVIRONMENT CHECK")
    print("=" * 80)

    print()
    print("Python")
    print("-" * 80)

    print(sys.version)

    print()
    print("Packages")
    print("-" * 80)

    print_package(
        "NumPy",
        numpy.__version__
    )

    print_package(
        "Pandas",
        pandas.__version__
    )

    print_package(
        "GeoPandas",
        geopandas.__version__
    )

    print_package(
        "Shapely",
        shapely.__version__
    )

    print_package(
        "PyProj",
        pyproj.__version__
    )

    print_package(
        "NetworkX",
        networkx.__version__
    )

    print_package(
        "Scikit-learn",
        sklearn.__version__
    )

    print_package(
        "SciPy",
        scipy.__version__
    )

    print_package(
        "PyTorch",
        torch.__version__
    )

    print()
    print_device_information()

    print()
    print("=" * 80)
    print("CONFIGURATION CHECK")
    print("=" * 80)

    config = load_main_config()
    experiments = load_experiment_config()

    print(
        "Project:",
        config["project"]["name"]
    )

    print(
        "Runs:",
        config["experiment"]["runs"]
    )

    print(
        "Seeds:",
        config["experiment"]["seeds"]
    )

    print(
        "Source scale:",
        config["scales"]["source"]
    )

    print(
        "Target scales:",
        config["scales"]["targets"]
    )

    print(
        "Number of tables:",
        len(experiments["tables"])
    )

    print(
        "Number of figures:",
        len(experiments["figures"])
    )

    print()
    print("=" * 80)
    print("REPRODUCIBILITY TEST")
    print("=" * 80)

    set_seed(42)

    a = torch.rand(5)

    set_seed(42)

    b = torch.rand(5)

    if torch.equal(a, b):
        print("[PASS] Reproducibility test.")
    else:
        raise RuntimeError(
            "Reproducibility test failed."
        )

    print()
    print("=" * 80)
    print("PHASE 0 CHECK PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()