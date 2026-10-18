import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


from src.utils.config import load_main_config
from src.utils.experiment import create_run_directory
from src.utils.reproducibility import set_seed


def main():

    config = load_main_config()

    seeds = config["experiment"]["seeds"]

    print("=" * 80)
    print("TESTING 10-RUN EXPERIMENT STRUCTURE")
    print("=" * 80)

    for seed in seeds:

        set_seed(seed)

        directory = create_run_directory(
            experiment_name="phase0_test",
            seed=seed,
        )

        print(
            f"[OK] Seed {seed}: "
            f"{directory.relative_to(PROJECT_ROOT)}"
        )

    print()
    print(
        f"Total runs prepared: {len(seeds)}"
    )

    assert len(seeds) == 10

    print()
    print("=" * 80)
    print("10-RUN STRUCTURE PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()