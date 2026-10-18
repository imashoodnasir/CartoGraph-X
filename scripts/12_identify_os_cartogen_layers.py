from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INVENTORY_FILE = (
    PROJECT_ROOT /
    "data" /
    "interim" /
    "os" /
    "os_geopackage_layers.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT /
    "data" /
    "interim" /
    "os" /
    "cartogen_candidate_layers.json"
)


BUILDING_TERMS = [
    "building",
    "buildings",
    "structure",
    "structures",
]


ROAD_TERMS = [
    "road",
    "roads",
    "street",
    "streets",
    "highway",
    "highways",
    "transport",
    "transportation",
    "motorway",
]


def score_text(
    text,
    terms
):

    text = text.lower()

    score = 0

    matched = []

    for term in terms:

        if term in text:

            score += 1
            matched.append(term)

    return score, matched


def main():

    if not INVENTORY_FILE.exists():

        raise FileNotFoundError(
            "Run "
            "11_inspect_os_geopackage_layers.py "
            "first."
        )

    with open(
        INVENTORY_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        inventory = json.load(file)

    result = {}

    print("=" * 110)
    print(
        "CARTOGEN — IDENTIFY "
        "BUILDING AND ROAD LAYERS"
    )
    print("=" * 110)

    for dataset, info in (
        inventory.items()
    ):

        print()
        print("=" * 110)
        print(dataset)
        print("=" * 110)

        building_candidates = []
        road_candidates = []

        for layer in info["layers"]:

            table = (
                layer["table_name"]
                or ""
            )

            identifier = (
                layer["identifier"]
                or ""
            )

            description = (
                layer["description"]
                or ""
            )

            columns = " ".join(
                column["name"]
                for column
                in layer["columns"]
            )

            searchable = " ".join([
                table,
                identifier,
                description,
                columns,
            ])

            b_score, b_terms = (
                score_text(
                    searchable,
                    BUILDING_TERMS
                )
            )

            r_score, r_terms = (
                score_text(
                    searchable,
                    ROAD_TERMS
                )
            )

            if b_score > 0:

                building_candidates.append({
                    "table_name":
                        table,
                    "score":
                        b_score,
                    "matched_terms":
                        b_terms,
                    "geometry_type":
                        layer[
                            "geometry_type"
                        ],
                    "feature_count":
                        layer[
                            "feature_count"
                        ],
                })

            if r_score > 0:

                road_candidates.append({
                    "table_name":
                        table,
                    "score":
                        r_score,
                    "matched_terms":
                        r_terms,
                    "geometry_type":
                        layer[
                            "geometry_type"
                        ],
                    "feature_count":
                        layer[
                            "feature_count"
                        ],
                })

        building_candidates.sort(
            key=lambda x: (
                -x["score"],
                x["table_name"]
            )
        )

        road_candidates.sort(
            key=lambda x: (
                -x["score"],
                x["table_name"]
            )
        )

        result[dataset] = {
            "building_candidates":
                building_candidates,
            "road_candidates":
                road_candidates,
        }

        print()
        print(
            "BUILDING CANDIDATES"
        )

        print("-" * 110)

        if not building_candidates:
            print("None detected.")

        for item in building_candidates:

            print(
                f"{item['table_name']:<45} "
                f"| score={item['score']} "
                f"| geometry="
                f"{item['geometry_type']} "
                f"| features="
                f"{item['feature_count']}"
            )

        print()
        print(
            "ROAD CANDIDATES"
        )

        print("-" * 110)

        if not road_candidates:
            print("None detected.")

        for item in road_candidates:

            print(
                f"{item['table_name']:<45} "
                f"| score={item['score']} "
                f"| geometry="
                f"{item['geometry_type']} "
                f"| features="
                f"{item['feature_count']}"
            )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2
        )

    print()
    print("=" * 110)
    print(
        "CANDIDATE LAYER "
        "IDENTIFICATION COMPLETE"
    )
    print("=" * 110)

    print(
        f"Saved:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()