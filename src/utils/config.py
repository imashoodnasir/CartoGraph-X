from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path):
    """
    Load YAML configuration.
    """

    path = Path(path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def load_main_config():
    return load_yaml("configs/cartogen.yaml")


def load_experiment_config():
    return load_yaml("configs/experiments.yaml")