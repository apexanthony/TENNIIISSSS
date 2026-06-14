import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from bounce_rule_detector import (
    BounceRuleParams,
    TrackPoint,
    count_near_events,
    detect_bounces,
    load_label_csv,
    load_playable_regions_for_clip,
    match_events,
    metrics_from_counts,
    point_in_regions,
    true_bounce_frames,
    true_hit_frames,
)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-conf", type=float, default=0.30)
    parser.add_argument("--max-gap", type=int, default=3)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--before-window", type=int, default=3)
    parser.add_argument("--after-window", type=int, default=3)
    parser.add_argument("--min-speed", type=float, default=2.0)
    parser.add_argument("--speed-max", type=float, default=90.0)
    parser.add_argument("--speed-ratio-max", type=float, default=5.0)
    parser.add_argument("--hit-speedup-ratio", type=float, default=1.2)
    parser.add_argument("--hit-speedup-delta", type=float, default=7.0)
    parser.add_argument("--hit-speedup-penalty", type=float, default=2.0)
    parser.add_argument("--angle-min", type=float, default=20.0)
    parser.add_argument("--angle-weak-max", type=float, default=12.0)
    parser.add_argument("--angle-good-max", type=float, default=75.0)
    parser.add_argument("--angle-hard-max", type=float, default=90.0)
    parser.add_argument("--accel-min", type=float, default=3.0)
    parser.add_argument("--accel-weak-max", type=float, default=8.0)
    parser.add_argument("--accel-good-max", type=float, default=24.0)
    parser.add_argument("--accel-hard-max", type=float, default=32.0)
    parser.add_argument("--jump-good-max", type=float, default=50.0)
    parser.add_argument("--jump-hard-max", type=float, default=120.0)
    parser.add_argument("--min-valid-points", type=int, default=6)
    parser.add_argument("--min-score", type=float, default=9.0)
    parser.add_argument("--nms-window", type=int, default=22)
    parser.add_argument("--region-bonus", type=float, default=1.0)
    parser.add_argument("--region-penalty", type=float, default=4.0)
    parser.add_argument("--region-hard-filter", action="store_true")
    parser.add_argument("--enable-v41", action="store_true", help="Enable V4.1 hit-like guard and temporal relocation.")
    parser.add_argument("--hit-guard-window", type=int, default=8)
    parser.add_argument("--hit-guard-speedup-ratio", type=float, default=1.6)
    parser.add_argument("--hit-guard-speedup-delta", type=float, default=10.0)
    parser.add_argument("--hit-guard-min-speed-after", type=float, default=18.0)
    parser.add_argument("--hit-guard-penalty", type=float, default=3.0)
    parser.add_argument("--enable-relocation", action="store_true", help="Experimental V4.1 candidate time relocation.")
    parser.add_argument("--relocate-back", type=int, default=2)
    parser.add_argument("--relocate-forward", type=int, default=12)
    parser.add_argument("--relocate-min-score", type=float, default=7.0)
    parser.add_argument("--relocate-angle-weight", type=float, default=0.035)
    parser.add_argument("--relocate-late-weight", type=float, default=0.18)
    parser.add_argument("--relocate-speed-weight", type=float, default=0.045)
    parser.add_argument("--enable-v42", action="store_true", help="Enable V4.2 event-merge-aware scoring refinements.")
    parser.add_argument("--low-speed-angle-min", type=float, default=80.0)
    parser.add_argument("--low-speed-jump-max", type=float, default=12.0)
    parser.add_argument("--low-speed-before-max", type=float, default=8.0)
    parser.add_argument("--low-speed-after-max", type=float, default=9.0)
    parser.add_argument("--low-speed-bonus", type=float, default=2.0)
    parser.add_argument("--sharp-angle-min", type=float, default=120.0)
    parser.add_argument("--sharp-accel-min", type=float, default=32.0)
    parser.add_argument("--sharp-speed-ratio-max", type=float, default=1.10)
    parser.add_argument("--sharp-bounce-bonus", type=float, default=2.0)
    parser.add_argument("--enable-late-refine", action="store_true", help="Experimental V4.2 forward-only time refinement.")
    parser.add_argument("--late-refine-forward", type=int, default=12)
    parser.add_argument("--late-refine-min-score", type=float, default=7.0)
    parser.add_argument("--late-refine-angle-min", type=float, default=45.0)
    parser.add_argument("--late-refine-max-distance", type=float, default=120.0)
    parser.add_argument("--late-refine-speed-mean-max", type=float, default=22.0)
    parser.add_argument("--late-refine-angle-weight", type=float, default=0.035)
    parser.add_argument("--late-refine-late-weight", type=float, default=0.18)
    parser.add_argument("--late-refine-speed-weight", type=float, default=0.045)
    parser.add_argument("--enable-v43", action="store_true", help="Enable V4.3 adaptive per-clip trajectory thresholds.")
    parser.add_argument("--adaptive-low-speed-percentile", type=float, default=35.0)
    parser.add_argument("--adaptive-jump-percentile", type=float, default=75.0)
    parser.add_argument("--adaptive-hard-jump-percentile", type=float, default=95.0)
    parser.add_argument("--adaptive-accel-high-percentile", type=float, default=85.0)
    parser.add_argument("--adaptive-low-speed-bonus", type=float, default=1.5)
    parser.add_argument("--adaptive-sharp-bonus", type=float, default=1.0)
    parser.add_argument("--adaptive-jump-penalty", type=float, default=2.0)


def params_from_args(args: argparse.Namespace) -> BounceRuleParams:
    return BounceRuleParams(
        min_conf=args.min_conf,
        max_gap=args.max_gap,
        smooth_window=args.smooth_window,
        before_window=args.before_window,
        after_window=args.after_window,
        min_speed=args.min_speed,
        speed_max=args.speed_max,
        speed_ratio_max=args.speed_ratio_max,
        hit_speedup_ratio=args.hit_speedup_ratio,
        hit_speedup_delta=args.hit_speedup_delta,
        hit_speedup_penalty=args.hit_speedup_penalty,
        angle_min=args.angle_min,
        angle_weak_max=args.angle_weak_max,
        angle_good_max=args.angle_good_max,
        angle_hard_max=args.angle_hard_max,
        accel_min=args.accel_min,
        accel_weak_max=args.accel_weak_max,
        accel_good_max=args.accel_good_max,
        accel_hard_max=args.accel_hard_max,
        jump_good_max=args.jump_good_max,
        jump_hard_max=args.jump_hard_max,
        min_valid_points=args.min_valid_points,
        min_score=args.min_score,
        nms_window=args.nms_window,
        region_bonus=args.region_bonus,
        region_penalty=args.region_penalty,
        region_hard_filter=args.region_hard_filter,
        enable_v41=args.enable_v41,
        hit_guard_window=args.hit_guard_window,
        hit_guard_speedup_ratio=args.hit_guard_speedup_ratio,
        hit_guard_speedup_delta=args.hit_guard_speedup_delta,
        hit_guard_min_speed_after=args.hit_guard_min_speed_after,
        hit_guard_penalty=args.hit_guard_penalty,
        enable_relocation=args.enable_relocation,
        relocate_back=args.relocate_back,
        relocate_forward=args.relocate_forward,
        relocate_min_score=args.relocate_min_score,
        relocate_angle_weight=args.relocate_angle_weight,
        relocate_late_weight=args.relocate_late_weight,
        relocate_speed_weight=args.relocate_speed_weight,
        enable_v42=args.enable_v42,
        low_speed_angle_min=args.low_speed_angle_min,
        low_speed_jump_max=args.low_speed_jump_max,
        low_speed_before_max=args.low_speed_before_max,
        low_speed_after_max=args.low_speed_after_max,
        low_speed_bonus=args.low_speed_bonus,
        sharp_angle_min=args.sharp_angle_min,
        sharp_accel_min=args.sharp_accel_min,
        sharp_speed_ratio_max=args.sharp_speed_ratio_max,
        sharp_bounce_bonus=args.sharp_bounce_bonus,
        late_refine_enabled=args.enable_late_refine,
        late_refine_forward=args.late_refine_forward,
        late_refine_min_score=args.late_refine_min_score,
        late_refine_angle_min=args.late_refine_angle_min,
        late_refine_max_distance=args.late_refine_max_distance,
        late_refine_speed_mean_max=args.late_refine_speed_mean_max,
        late_refine_angle_weight=args.late_refine_angle_weight,
        late_refine_late_weight=args.late_refine_late_weight,
        late_refine_speed_weight=args.late_refine_speed_weight,
        enable_v43=args.enable_v43,
        adaptive_low_speed_percentile=args.adaptive_low_speed_percentile,
        adaptive_jump_percentile=args.adaptive_jump_percentile,
        adaptive_hard_jump_percentile=args.adaptive_hard_jump_percentile,
        adaptive_accel_high_percentile=args.adaptive_accel_high_percentile,
        adaptive_low_speed_bonus=args.adaptive_low_speed_bonus,
        adaptive_sharp_bonus=args.adaptive_sharp_bonus,
        adaptive_jump_penalty=args.adaptive_jump_penalty,
    )


def parse_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_float(value, default=-1.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def frame_id_from_path(path: str) -> int:
    return int(Path(path).stem)


def split_key(path: str) -> str:
    parts = Path(path.replace("\\", "/")).parts
    if len(parts) >= 4 and parts[0] == "images":
        return "/".join(parts[:3])
    if len(parts) >= 3:
        return "/".join(parts[-3:-1])
    return str(Path(path).parent)


def load_split_csv(path: str) -> Dict[str, List[TrackPoint]]:
    grouped: Dict[str, List[TrackPoint]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        for idx, row in enumerate(reader):
            image_path = row.get("path1") or row.get("file name") or str(idx)
            key = split_key(image_path)
            frame_id = frame_id_from_path(image_path)
            x = parse_float(row.get("x-coordinate") or row.get("x"))
            y = parse_float(row.get("y-coordinate") or row.get("y"))
            status = parse_int(row.get("status"))
            visibility = parse_int(row.get("visibility"), 1)
            valid = visibility != 0 and x >= 0 and y >= 0
            grouped[key].append(TrackPoint(frame_id, x, y, 1.0, valid, status))
    return {key: sorted(points, key=lambda p: p.frame_id) for key, points in grouped.items()}


def load_eval_tracks(args: argparse.Namespace) -> Dict[str, List[TrackPoint]]:
    if args.split == "all":
        tracks: Dict[str, List[TrackPoint]] = {}
        root = Path(args.labels_root)
        for label_path in sorted(root.glob("game*/Clip*/Label.csv")):
            key = "/".join(label_path.parts[-3:-1])
            tracks[key] = load_label_csv(str(label_path))
        return tracks

    split_path = Path(args.dataset_root) / f"labels_{args.split}.csv"
    return load_split_csv(str(split_path))


def evaluate_tracks(
    tracks: Dict[str, List[TrackPoint]],
    params: BounceRuleParams,
    tolerance: int,
    region_root: str = "",
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    total_tp = total_fp = total_fn = 0
    hit_fp = 0
    frame_errors: List[int] = []
    rows: List[Dict[str, object]] = []

    for name, points in sorted(tracks.items()):
        playable_regions = load_playable_regions_for_clip(name, region_root)
        candidates = detect_bounces(points, params, playable_regions)
        pred_frames = [c.frame_id for c in candidates]
        bounce_frames = true_bounce_frames(points)
        hit_frames = true_hit_frames(points)
        tp, fp, fn, errors = match_events(pred_frames, bounce_frames, tolerance)
        hit_like = count_near_events(pred_frames, hit_frames, tolerance)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        hit_fp += hit_like
        frame_errors.extend(errors)

        for cand in candidates:
            rows.append(
                {
                    "clip": name,
                    "frame_id": cand.frame_id,
                    "x": round(cand.x, 2),
                    "y": round(cand.y, 2),
                    "score": round(cand.score, 3),
                    "angle_change": round(cand.angle_change, 3),
                    "accel_norm": round(cand.accel_norm, 3),
                    "speed_before": round(cand.speed_before, 3),
                    "speed_after": round(cand.speed_after, 3),
                    "jump_distance": round(cand.jump_distance, 3),
                    "in_ground_region": int(not playable_regions or point_in_regions(cand.x, cand.y, playable_regions)),
                    "near_hit": int(any(abs(cand.frame_id - h) <= tolerance for h in hit_frames)),
                    "near_bounce": int(any(abs(cand.frame_id - b) <= tolerance for b in bounce_frames)),
                }
            )

    metrics = metrics_from_counts(total_tp, total_fp, total_fn)
    metrics.update(
        {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "hit_fp": hit_fp,
            "avg_frame_error": sum(frame_errors) / len(frame_errors) if frame_errors else 0.0,
            "num_tracks": len(tracks),
        }
    )
    return metrics, rows


def write_candidates(path: str, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rule-based tennis bounce detection.")
    parser.add_argument("--dataset-root", default="./datasets/trackNet")
    parser.add_argument("--labels-root", default="./datasets/trackNet/images")
    parser.add_argument("--region-root", default="", help="Directory with game-level or clip-level ROI JSON files.")
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--match-tolerance", type=int, default=3)
    parser.add_argument("--out-csv", default="")
    add_common_args(parser)
    args = parser.parse_args()

    tracks = load_eval_tracks(args)
    params = params_from_args(args)
    metrics, rows = evaluate_tracks(tracks, params, args.match_tolerance, args.region_root)

    print(f"tracks    = {metrics['num_tracks']}")
    print(f"TP        = {metrics['tp']}")
    print(f"FP        = {metrics['fp']}")
    print(f"FN        = {metrics['fn']}")
    print(f"Hit FP    = {metrics['hit_fp']}")
    print(f"Precision = {metrics['precision']:.4f}")
    print(f"Recall    = {metrics['recall']:.4f}")
    print(f"F1        = {metrics['f1']:.4f}")
    print(f"AvgErr    = {metrics['avg_frame_error']:.3f} frames")

    if args.out_csv:
        write_candidates(args.out_csv, rows)
        print(f"candidates saved to {args.out_csv}")


if __name__ == "__main__":
    main()
