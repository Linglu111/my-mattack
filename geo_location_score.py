"""Evaluate image geolocation predictions.

This follows the GeoRanker-style protocol: compute the geodesic distance
between each dataset row's ground-truth GPS coordinate and its predicted GPS
coordinate, then report accuracy under 1, 25, 200, 750, and 2500 km thresholds.

The benchmark denominator is the full ground-truth dataset, not only the rows
that have valid model predictions. Missing, unmatched, duplicate, invalid, or
unparsed predictions are counted as failures for benchmark accuracy.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


EARTH_RADIUS_KM = 6371.0088
THRESHOLDS = [
    ("street_1km", 1.0),
    ("city_25km", 25.0),
    ("region_200km", 200.0),
    ("country_750km", 750.0),
    ("continent_2500km", 2500.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score one-point geolocation predictions.")
    parser.add_argument(
        "--predictions",
        default="LAT/geolocation_predictions/gpt41.jsonl",
        help="JSONL file produced by geo_location_predict.py.",
    )
    parser.add_argument(
        "--gt-csv",
        default="data/images/img2gps3k/im2gps3k_places365.csv",
        help="Ground-truth CSV with image name and coordinates.",
    )
    parser.add_argument("--name-col", default="name", help="Ground-truth image-name column.")
    parser.add_argument("--lat-col", default="LAT", help="Ground-truth latitude column.")
    parser.add_argument("--lon-col", default="LON", help="Ground-truth longitude column.")
    parser.add_argument(
        "--output-dir",
        default="LAT/geolocation_score",
        help="Directory for scored_predictions.jsonl, summary.json, and summary.csv.",
    )
    parser.add_argument(
        "--inclusive-threshold",
        action="store_true",
        help="Use distance <= threshold. Default matches GeoRanker and uses distance < threshold.",
    )
    return parser.parse_args()


def normalize_key(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.lower()


def image_keys(image_name: object, image_path: object = None, image_stem: object = None) -> List[str]:
    candidates = [
        normalize_key(image_name),
        normalize_key(Path(str(image_name)).stem) if image_name else None,
        normalize_key(image_stem),
        normalize_key(Path(str(image_path)).name) if image_path else None,
        normalize_key(Path(str(image_path)).stem) if image_path else None,
    ]

    keys = []
    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            keys.append(candidate)
            seen.add(candidate)
    return keys


def load_ground_truth(
    csv_path: str,
    name_col: str,
    lat_col: str,
    lon_col: str,
) -> Tuple[List[dict], Dict[str, dict]]:
    rows: List[dict] = []
    index: Dict[str, dict] = {}

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {name_col, lat_col, lon_col}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=1):
            image_name = row[name_col].strip()
            if not image_name:
                continue

            item = {
                "gt_row_number": row_number,
                "gt_image_name": image_name,
                "gt_latitude": float(row[lat_col]),
                "gt_longitude": float(row[lon_col]),
            }
            rows.append(item)

            for key in image_keys(image_name):
                if key in index:
                    raise ValueError(
                        f"Duplicate ground-truth image key {key!r} in {csv_path}; "
                        "cannot evaluate unambiguously."
                    )
                index[key] = item

    return rows, index


def load_predictions(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                row["_prediction_line_number"] = line_number
                rows.append(row)
    return rows


def match_ground_truth(prediction: dict, ground_truth_index: Dict[str, dict]) -> Optional[dict]:
    for key in image_keys(
        prediction.get("image_name"),
        prediction.get("image_path"),
        prediction.get("image_stem"),
    ):
        if key in ground_truth_index:
            return ground_truth_index[key]
    return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Use GeoRanker's geopy geodesic distance when available."""
    try:
        from geopy.distance import geodesic
    except ImportError:
        return haversine_km(lat1, lon1, lat2, lon2)
    return geodesic((lat1, lon1), (lat2, lon2)).km


def valid_prediction(row: dict) -> bool:
    lat = row.get("pred_latitude")
    lon = row.get("pred_longitude")
    return (
        isinstance(lat, (int, float))
        and isinstance(lon, (int, float))
        and not isinstance(lat, bool)
        and not isinstance(lon, bool)
        and -90.0 <= lat <= 90.0
        and -180.0 <= lon <= 180.0
    )


def within_threshold(distance: float, threshold: float, inclusive: bool) -> bool:
    return distance <= threshold if inclusive else distance < threshold


def false_thresholds() -> Dict[str, bool]:
    return {name: False for name, _ in THRESHOLDS}


def index_predictions(
    predictions: Iterable[dict],
    ground_truth_index: Dict[str, dict],
) -> Tuple[Dict[str, dict], List[dict], List[dict]]:
    matched_by_gt_name: Dict[str, dict] = {}
    duplicates: List[dict] = []
    unmatched: List[dict] = []

    for prediction in predictions:
        gt = match_ground_truth(prediction, ground_truth_index)
        if gt is None:
            result = dict(prediction)
            result.update(
                {
                    "matched_ground_truth": False,
                    "score_error": "No matching ground-truth row.",
                    "distance_km": None,
                    "within_thresholds": false_thresholds(),
                }
            )
            unmatched.append(result)
            continue

        gt_name = gt["gt_image_name"]
        if gt_name in matched_by_gt_name:
            result = dict(prediction)
            result.update(gt)
            result.update(
                {
                    "matched_ground_truth": True,
                    "score_error": "Duplicate prediction for this ground-truth image; ignored.",
                    "distance_km": None,
                    "within_thresholds": false_thresholds(),
                }
            )
            duplicates.append(result)
            continue

        matched_by_gt_name[gt_name] = prediction

    return matched_by_gt_name, unmatched, duplicates


def score_dataset(
    ground_truth_rows: List[dict],
    predictions: List[dict],
    ground_truth_index: Dict[str, dict],
    inclusive_threshold: bool,
) -> Tuple[List[dict], List[dict], List[dict]]:
    matched_predictions, unmatched_predictions, duplicate_predictions = index_predictions(
        predictions,
        ground_truth_index,
    )

    scored: List[dict] = []
    for gt in ground_truth_rows:
        prediction = matched_predictions.get(gt["gt_image_name"])

        if prediction is None:
            result = dict(gt)
            result.update(
                {
                    "image_name": gt["gt_image_name"],
                    "matched_ground_truth": True,
                    "has_prediction": False,
                    "valid_prediction": False,
                    "distance_km": None,
                    "within_thresholds": false_thresholds(),
                    "score_error": "Missing prediction for this ground-truth image.",
                }
            )
            scored.append(result)
            continue

        result = dict(prediction)
        result.update(gt)
        result["matched_ground_truth"] = True
        result["has_prediction"] = True

        if not valid_prediction(prediction):
            result.update(
                {
                    "valid_prediction": False,
                    "distance_km": None,
                    "within_thresholds": false_thresholds(),
                    "score_error": prediction.get("parse_error")
                    or prediction.get("model_error")
                    or "Invalid predicted coordinate.",
                }
            )
            scored.append(result)
            continue

        distance = distance_km(
            gt["gt_latitude"],
            gt["gt_longitude"],
            prediction["pred_latitude"],
            prediction["pred_longitude"],
        )
        result["valid_prediction"] = True
        result["distance_km"] = distance
        result["within_thresholds"] = {
            name: within_threshold(distance, threshold, inclusive_threshold)
            for name, threshold in THRESHOLDS
        }
        result["score_error"] = None
        scored.append(result)

    return scored, unmatched_predictions, duplicate_predictions


def median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def summarize(
    scored: List[dict],
    unmatched: List[dict],
    duplicates: List[dict],
    inclusive_threshold: bool,
) -> dict:
    total = len(scored)
    predicted = [row for row in scored if row.get("has_prediction")]
    valid = [row for row in scored if row.get("valid_prediction")]
    distances = [row["distance_km"] for row in valid]

    thresholds = {}
    for name, threshold in THRESHOLDS:
        hits = sum(1 for row in scored if row.get("within_thresholds", {}).get(name, False))
        valid_hits = sum(1 for row in valid if row.get("within_thresholds", {}).get(name, False))
        thresholds[name] = {
            "threshold_km": threshold,
            "hits": hits,
            "benchmark_accuracy": hits / total if total else 0.0,
            "benchmark_accuracy_percent": 100.0 * hits / total if total else 0.0,
            "valid_accuracy": valid_hits / len(valid) if valid else 0.0,
            "valid_accuracy_percent": 100.0 * valid_hits / len(valid) if valid else 0.0,
        }

    return {
        "benchmark_total": total,
        "prediction_rows_matched_to_dataset": len(predicted),
        "valid_predictions": len(valid),
        "missing_predictions": total - len(predicted),
        "invalid_or_unparsed_predictions": len(predicted) - len(valid),
        "unmatched_prediction_rows": len(unmatched),
        "duplicate_prediction_rows_ignored": len(duplicates),
        "mean_distance_km": sum(distances) / len(distances) if distances else None,
        "median_distance_km": median(distances),
        "threshold_rule": "distance <= threshold" if inclusive_threshold else "distance < threshold",
        "thresholds": thresholds,
    }


def write_outputs(
    output_dir: Path,
    scored: List[dict],
    unmatched: List[dict],
    duplicates: List[dict],
    summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "scored_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (output_dir / "unmatched_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in unmatched:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (output_dir / "duplicate_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in duplicates:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric",
                "threshold_km",
                "hits",
                "benchmark_accuracy",
                "benchmark_accuracy_percent",
                "valid_accuracy",
                "valid_accuracy_percent",
            ],
        )
        writer.writeheader()
        for name, item in summary["thresholds"].items():
            writer.writerow(
                {
                    "metric": name,
                    "threshold_km": item["threshold_km"],
                    "hits": item["hits"],
                    "benchmark_accuracy": item["benchmark_accuracy"],
                    "benchmark_accuracy_percent": item["benchmark_accuracy_percent"],
                    "valid_accuracy": item["valid_accuracy"],
                    "valid_accuracy_percent": item["valid_accuracy_percent"],
                }
            )


def main() -> None:
    args = parse_args()
    ground_truth_rows, ground_truth_index = load_ground_truth(
        args.gt_csv,
        args.name_col,
        args.lat_col,
        args.lon_col,
    )
    predictions = load_predictions(args.predictions)
    scored, unmatched, duplicates = score_dataset(
        ground_truth_rows,
        predictions,
        ground_truth_index,
        inclusive_threshold=args.inclusive_threshold,
    )
    summary = summarize(
        scored,
        unmatched,
        duplicates,
        inclusive_threshold=args.inclusive_threshold,
    )
    write_outputs(Path(args.output_dir), scored, unmatched, duplicates, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
