from pathlib import Path
import json
import re
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "poland"
    / "bdot10k_inventory"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "poland"
    / "bdot10k_class_inspection"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


CITY_CONFIG = {
    "warsaw": {
        "prefix": "PL.PZGiK.330.1465",
    },
    "krakow": {
        "prefix": "PL.PZGiK.283.1261",
    },
    "gdansk": {
        "prefix": "PL.PZGiK.336.2261",
    },
}

TARGET_CLASSES = {
    "buildings": "OT_BUBD_A",
    "roads": "OT_SKDR_L",
}


def sep(title, char="="):
    print()
    print(char * 120)
    print(title)
    print(char * 120)


def local_name(tag):
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def namespace(tag):
    if tag.startswith("{") and "}" in tag:
        return tag[1:].split("}", 1)[0]
    return None


def find_target_file(city, prefix, class_name):

    root = (
        INPUT_ROOT
        / city
        / "outer_extracted"
    )

    if not root.exists():
        raise FileNotFoundError(
            f"Extracted directory does not exist:\n{root}"
        )

    candidates = list(
        root.rglob(f"*__{class_name}.xml")
    )

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No XML found for {city}: {class_name}"
        )

    if len(candidates) > 1:
        print(
            f"[WARNING] Multiple files found for "
            f"{city}/{class_name}:"
        )
        for p in candidates:
            print(f"  {p}")

    return candidates[0]


def inspect_header(path, max_events=100):

    namespaces = {}
    root_tag = None
    root_attrib = {}

    event_count = 0

    for event, elem in ET.iterparse(
        path,
        events=("start", "start-ns")
    ):

        if event == "start-ns":
            prefix, uri = elem
            namespaces[prefix or "default"] = uri

        elif event == "start":

            if root_tag is None:
                root_tag = elem.tag
                root_attrib = dict(elem.attrib)

            event_count += 1

            if event_count >= max_events:
                break

    return {
        "root_tag": root_tag,
        "root_local_name":
            local_name(root_tag)
            if root_tag else None,
        "root_namespace":
            namespace(root_tag)
            if root_tag else None,
        "root_attributes": root_attrib,
        "namespaces": namespaces,
    }


def scan_xml(path, expected_class):

    feature_count = 0

    tag_counts = Counter()
    attribute_counts = Counter()
    srs_counts = Counter()
    geometry_tag_counts = Counter()
    id_attribute_counts = Counter()

    sample_features = []

    geometry_names = {
        "Polygon",
        "MultiPolygon",
        "Surface",
        "MultiSurface",
        "LineString",
        "MultiLineString",
        "Curve",
        "MultiCurve",
        "LinearRing",
        "pos",
        "posList",
        "coordinates",
    }

    expected_upper = expected_class.upper()

    context = ET.iterparse(
        path,
        events=("start", "end")
    )

    _, root = next(context)

    for event, elem in context:

        if event == "start":

            lname = local_name(elem.tag)
            tag_counts[lname] += 1

            for attr_name, attr_value in elem.attrib.items():

                alocal = local_name(attr_name)
                attribute_counts[alocal] += 1

                if "srsName" in alocal:
                    srs_counts[str(attr_value)] += 1

                if (
                    alocal.lower() == "id"
                    or "identifier" in alocal.lower()
                ):
                    id_attribute_counts[
                        f"{alocal}={attr_value}"
                    ] += 1

            if lname in geometry_names:
                geometry_tag_counts[lname] += 1

        elif event == "end":

            lname = local_name(elem.tag)

            # BDOT feature elements generally carry class name,
            # e.g. OT_BUBD_A or OT_SKDR_L.
            if lname.upper() == expected_upper:

                feature_count += 1

                if len(sample_features) < 3:

                    sample = {
                        "tag": lname,
                        "attributes":
                            dict(elem.attrib),
                        "children": [],
                    }

                    for child in list(elem):

                        child_name = local_name(
                            child.tag
                        )

                        text = (
                            child.text.strip()
                            if child.text
                            and child.text.strip()
                            else None
                        )

                        child_record = {
                            "name": child_name,
                            "text": text,
                            "attributes":
                                dict(child.attrib),
                        }

                        # Record grandchildren names,
                        # but avoid dumping coordinates.
                        grandchildren = []

                        for grandchild in list(child):

                            gname = local_name(
                                grandchild.tag
                            )

                            grandchildren.append(
                                gname
                            )

                        if grandchildren:
                            child_record[
                                "child_tags"
                            ] = grandchildren[:30]

                        sample["children"].append(
                            child_record
                        )

                    sample_features.append(sample)

            # Clear elements to keep memory usage low.
            elem.clear()

            if elem is root:
                root.clear()

    return {
        "feature_count": feature_count,
        "top_tags":
            tag_counts.most_common(40),
        "top_attributes":
            attribute_counts.most_common(40),
        "srs_names":
            dict(srs_counts),
        "geometry_tags":
            dict(geometry_tag_counts),
        "id_attribute_examples":
            list(id_attribute_counts.keys())[:20],
        "sample_features":
            sample_features,
    }


def infer_geometry_family(geometry_tags):

    polygon_tokens = {
        "Polygon",
        "MultiPolygon",
        "Surface",
        "MultiSurface",
    }

    line_tokens = {
        "LineString",
        "MultiLineString",
        "Curve",
        "MultiCurve",
    }

    polygon_hits = sum(
        geometry_tags.get(k, 0)
        for k in polygon_tokens
    )

    line_hits = sum(
        geometry_tags.get(k, 0)
        for k in line_tokens
    )

    if polygon_hits > 0 and line_hits == 0:
        return "polygon"

    if line_hits > 0 and polygon_hits == 0:
        return "line"

    if polygon_hits > line_hits:
        return "mostly_polygon"

    if line_hits > polygon_hits:
        return "mostly_line"

    return "unknown"


def print_sample(sample, index):

    print()
    print(f"Sample feature {index}")
    print("-" * 120)

    print(f"Element: {sample['tag']}")

    if sample["attributes"]:
        print("Attributes:")
        for k, v in sample["attributes"].items():
            print(f"  {local_name(k)} = {v}")

    print("Child fields:")

    for child in sample["children"]:

        text = child["text"]

        if text is not None and len(text) > 120:
            text = text[:117] + "..."

        if text is not None:
            print(
                f"  {child['name']:<35} "
                f"{text}"
            )
        else:
            print(
                f"  {child['name']:<35} "
                f"[nested]"
            )

        if "child_tags" in child:
            print(
                " " * 39
                + "children: "
                + ", ".join(
                    child["child_tags"]
                )
            )


def main():

    sep(
        "CARTOGEN — INSPECT POLISH BDOT10k "
        "BUILDING AND ROAD FEATURE CLASSES"
    )

    report = {
        "cities": {},
        "target_classes": TARGET_CLASSES,
    }

    global_class_summary = defaultdict(list)

    for city, cfg in CITY_CONFIG.items():

        sep(city.upper())

        city_result = {}

        for role, class_name in TARGET_CLASSES.items():

            print()
            print("-" * 120)
            print(
                f"{role.upper()} — {class_name}"
            )
            print("-" * 120)

            path = find_target_file(
                city,
                cfg["prefix"],
                class_name,
            )

            print(f"File: {path}")
            print(
                f"Size: "
                f"{path.stat().st_size / 1024 / 1024:.2f} MB"
            )

            print()
            print("Inspecting XML header...")

            header = inspect_header(path)

            print(
                f"Root element: "
                f"{header['root_local_name']}"
            )

            print(
                f"Root namespace: "
                f"{header['root_namespace']}"
            )

            print()
            print("Namespaces:")

            for prefix, uri in sorted(
                header["namespaces"].items()
            ):
                print(
                    f"  {prefix:<15} {uri}"
                )

            print()
            print(
                "Streaming feature/schema inspection..."
            )

            scan = scan_xml(
                path,
                class_name
            )

            geom_family = infer_geometry_family(
                scan["geometry_tags"]
            )

            print()
            print("RESULT")
            print("-" * 120)

            print(
                f"Feature class:     {class_name}"
            )
            print(
                f"Feature count:     "
                f"{scan['feature_count']:,}"
            )
            print(
                f"Geometry family:   "
                f"{geom_family}"
            )

            print()
            print("SRS declarations:")

            if scan["srs_names"]:
                for srs, count in sorted(
                    scan["srs_names"].items(),
                    key=lambda x: -x[1]
                ):
                    print(
                        f"  {srs:<60} "
                        f"{count:>10,}"
                    )
            else:
                print("  None detected")

            print()
            print("Geometry elements:")

            if scan["geometry_tags"]:
                for tag, count in sorted(
                    scan["geometry_tags"].items(),
                    key=lambda x: -x[1]
                ):
                    print(
                        f"  {tag:<30} "
                        f"{count:>10,}"
                    )
            else:
                print("  None detected")

            print()
            print("Most frequent XML tags:")

            for tag, count in scan["top_tags"][:25]:
                print(
                    f"  {tag:<40} "
                    f"{count:>12,}"
                )

            if scan["sample_features"]:

                for i, sample in enumerate(
                    scan["sample_features"],
                    start=1
                ):
                    print_sample(
                        sample,
                        i
                    )

            else:
                print()
                print(
                    "[WARNING] No elements exactly matching "
                    f"{class_name} were found."
                )

            result = {
                "file": str(path),
                "size_bytes":
                    path.stat().st_size,
                "header": header,
                "scan": scan,
                "geometry_family":
                    geom_family,
            }

            city_result[role] = result

            global_class_summary[role].append({
                "city": city,
                "class_name": class_name,
                "feature_count":
                    scan["feature_count"],
                "geometry_family":
                    geom_family,
                "srs_names":
                    scan["srs_names"],
            })

        report["cities"][city] = city_result

    # ------------------------------------------------------------------
    # CROSS-CITY VALIDATION
    # ------------------------------------------------------------------

    sep("CROSS-CITY CLASS VALIDATION")

    decisions = {}

    for role in TARGET_CLASSES:

        rows = global_class_summary[role]

        print()
        print(role.upper())
        print("-" * 120)

        geometries = set()

        for row in rows:

            geometries.add(
                row["geometry_family"]
            )

            print(
                f"{row['city']:<15} "
                f"{row['class_name']:<15} "
                f"{row['feature_count']:>10,} features   "
                f"{row['geometry_family']}"
            )

        if role == "buildings":
            valid_geom = all(
                r["geometry_family"]
                in {
                    "polygon",
                    "mostly_polygon",
                }
                for r in rows
            )

        else:
            valid_geom = all(
                r["geometry_family"]
                in {
                    "line",
                    "mostly_line",
                }
                for r in rows
            )

        nonempty = all(
            r["feature_count"] > 0
            for r in rows
        )

        decisions[role] = {
            "nonempty_all_cities":
                bool(nonempty),
            "expected_geometry_all_cities":
                bool(valid_geom),
            "pass":
                bool(nonempty and valid_geom),
        }

        print()

        if nonempty and valid_geom:
            print(
                f"[PASS] {role} class is structurally "
                f"consistent across all three cities."
            )
        else:
            print(
                f"[CHECK] {role} requires further inspection."
            )

    report["decisions"] = decisions

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    output_json = (
        OUTPUT_ROOT
        / "poland_bdot10k_class_inspection.json"
    )

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    sep("SCRIPT 49 COMPLETE")

    print("Saved:")
    print(output_json)

    print()

    if all(
        d["pass"]
        for d in decisions.values()
    ):
        print(
            "[PASS] Candidate BDOT10k building and road "
            "classes confirmed structurally."
        )
        print()
        print(
            "NEXT: Script 50 — convert the confirmed "
            "BDOT10k XML classes to clean GeoPackage layers."
        )
    else:
        print(
            "[CHECK] Do not convert yet. Review the class "
            "inspection output first."
        )


if __name__ == "__main__":
    main()