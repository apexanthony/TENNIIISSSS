import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def parse_frames(text: str) -> List[int]:
    return [int(item) for item in str(text or "").split() if item.strip()]


def load_clip_lengths(labels_root: Path) -> dict:
    lengths = {}
    for label_path in labels_root.glob("game*/Clip*/Label.csv"):
        clip = "/".join(label_path.parts[-3:-1])
        frames = []
        with label_path.open(newline="", encoding="utf-8-sig") as fp:
            for row in csv.DictReader(fp):
                image_path = row.get("file name") or row.get("path1") or ""
                try:
                    frames.append(int(float(Path(image_path).stem)))
                except ValueError:
                    continue
        lengths[clip] = max(frames) + 1 if frames else 0
    return lengths


def match_events(predicted: Sequence[int], true: Sequence[int], tolerance: int) -> Tuple[int, int, int, List[int]]:
    used_true = set()
    errors: List[int] = []
    tp = 0
    for pred in sorted(predicted):
        best_idx = None
        best_error = None
        for idx, frame in enumerate(sorted(true)):
            if idx in used_true:
                continue
            error = pred - frame
            if abs(error) <= tolerance and (best_error is None or abs(error) < abs(best_error)):
                best_idx = idx
                best_error = error
        if best_idx is not None:
            used_true.add(best_idx)
            tp += 1
            errors.append(abs(best_error or 0))
    fp = len(predicted) - tp
    fn = len(true) - tp
    return tp, fp, fn, errors


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def summarize_rows(rows: Iterable[dict], lengths: dict, tolerance: int, boundary_margin: int) -> Tuple[dict, List[dict]]:
    totals = defaultdict(int)
    errors: List[int] = []
    by_game = defaultdict(lambda: defaultdict(int))
    details = []
    for row in rows:
        clip = row["clip"]
        game = clip.split("/")[0]
        length = lengths.get(clip, 0)
        true_raw = parse_frames(row.get("true_frames", ""))
        pred_raw = parse_frames(row.get("pred_frames", ""))
        true = [frame for frame in true_raw if boundary_margin <= frame <= length - 1 - boundary_margin]
        pred = [frame for frame in pred_raw if boundary_margin <= frame <= length - 1 - boundary_margin]
        tp, fp, fn, clip_errors = match_events(pred, true, tolerance)
        boundary_true = len(true_raw) - len(true)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["true"] += len(true)
        totals["pred"] += len(pred)
        totals["boundary_true"] += boundary_true
        errors.extend(clip_errors)
        by_game[game]["clips"] += 1
        by_game[game]["tp"] += tp
        by_game[game]["fp"] += fp
        by_game[game]["fn"] += fn
        by_game[game]["true"] += len(true)
        by_game[game]["pred"] += len(pred)
        by_game[game]["boundary_true"] += boundary_true
        if fp or fn:
            details.append(
                {
                    "clip": clip,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "pred_frames": " ".join(str(frame) for frame in pred),
                    "true_frames": " ".join(str(frame) for frame in true),
                    "boundary_true": boundary_true,
                }
            )

    precision, recall, f1 = prf(totals["tp"], totals["fp"], totals["fn"])
    summary = {
        "clips": sum(game["clips"] for game in by_game.values()),
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "true_events": totals["true"],
        "pred_events": totals["pred"],
        "boundary_true": totals["boundary_true"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_frame_error": sum(errors) / len(errors) if errors else 0.0,
    }
    game_rows = []
    for game, item in sorted(by_game.items()):
        precision, recall, f1 = prf(item["tp"], item["fp"], item["fn"])
        game_rows.append(
            {
                "game": game,
                "clips": item["clips"],
                "tp": item["tp"],
                "fp": item["fp"],
                "fn": item["fn"],
                "true_events": item["true"],
                "pred_events": item["pred"],
                "boundary_true": item["boundary_true"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return summary, game_rows + details


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize V4.3 detail CSV with optional boundary filtering.")
    parser.add_argument("--detail-csv", required=True)
    parser.add_argument("--labels-root", default="./datasets/trackNet/images")
    parser.add_argument("--match-tolerance", type=int, default=3)
    parser.add_argument("--boundary-margin", type=int, default=12)
    parser.add_argument("--out-prefix", default="")
    args = parser.parse_args()

    lengths = load_clip_lengths(Path(args.labels_root))
    with open(args.detail_csv, newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    summary, detail_rows = summarize_rows(rows, lengths, args.match_tolerance, args.boundary_margin)

    print(f"detail          = {args.detail_csv}")
    print(f"boundary_margin = {args.boundary_margin}")
    print(f"clips           = {summary['clips']}")
    print(f"TP              = {summary['tp']}")
    print(f"FP              = {summary['fp']}")
    print(f"FN              = {summary['fn']}")
    print(f"Boundary GT     = {summary['boundary_true']}")
    print(f"Precision       = {summary['precision']:.4f}")
    print(f"Recall          = {summary['recall']:.4f}")
    print(f"F1              = {summary['f1']:.4f}")
    print(f"AvgErr          = {summary['avg_frame_error']:.3f}")

    if args.out_prefix:
        prefix = Path(args.out_prefix)
        write_csv(prefix.with_name(prefix.name + "_summary.csv"), [summary])
        write_csv(prefix.with_name(prefix.name + "_details.csv"), detail_rows)


if __name__ == "__main__":
    main()
