import argparse
import csv
from pathlib import Path


def load_predictions(path):
    with open(path, "r", newline="", encoding="utf-8") as fp:
        return [int(float(row["frame"])) for row in csv.DictReader(fp) if row.get("event_type") == "bounce"]


def load_manual(path):
    events = []
    with open(path, "r", newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            start = str(row.get("manual_bounce_start", "")).strip()
            end = str(row.get("manual_bounce_end", "")).strip()
            if start and end:
                events.append((int(float(start)), int(float(end))))
    return events


def distance_to_interval(frame, interval):
    start, end = interval
    if start <= frame <= end:
        return 0
    return min(abs(frame - start), abs(frame - end))


def evaluate(predictions, ground_truth, tolerance):
    pairs = []
    available = set(range(len(ground_truth)))
    for frame in predictions:
        candidates = sorted(
            ((distance_to_interval(frame, ground_truth[idx]), idx) for idx in available),
            key=lambda item: item[0],
        )
        if candidates and candidates[0][0] <= tolerance:
            distance, idx = candidates[0]
            available.remove(idx)
            pairs.append((frame, idx, distance))
    tp = len(pairs)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    mean_error = sum(pair[2] for pair in pairs) / max(tp, 1)
    return precision, recall, f1, mean_error, pairs


def main():
    parser = argparse.ArgumentParser(description="Evaluate bounce-event CSV against manually verified frame ranges.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manual", required=True)
    parser.add_argument("--tolerances", default="3,5,8")
    parser.add_argument("--out-csv", default="")
    args = parser.parse_args()

    predictions = load_predictions(args.predictions)
    ground_truth = load_manual(args.manual)
    rows = []
    for tolerance in [int(value) for value in args.tolerances.split(",") if value.strip()]:
        precision, recall, f1, mean_error, pairs = evaluate(predictions, ground_truth, tolerance)
        rows.append(
            {
                "tolerance": tolerance,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_interval_error": mean_error,
                "matches": len(pairs),
                "predictions": len(predictions),
                "ground_truth": len(ground_truth),
            }
        )
        print(
            f"tol=+/-{tolerance} precision={precision:.4f} recall={recall:.4f} "
            f"f1={f1:.4f} mean_interval_error={mean_error:.2f}"
        )

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
