from pathlib import Path
import json
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_run_directory(
    experiment_name: str,
    seed: int
) -> Path:

    run_directory = (
        PROJECT_ROOT
        / "results"
        / "runs"
        / experiment_name
        / f"seed_{seed}"
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = {
        "experiment": experiment_name,
        "seed": seed,
        "created": datetime.now().isoformat(),
    }

    metadata_file = run_directory / "run_metadata.json"

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    return run_directory