import argparse
import csv
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from bounce_rule_detector import (
    BounceCandidate,
    BounceRuleParams,
    TrackPoint,
    count_near_events,
    extract_candidate_features,
    load_playable_regions_for_clip,
    match_events,
    metrics_from_counts,
    score_features,
    true_bounce_frames,
    true_hit_frames,
)
from eval_bounce_rule import add_common_args, load_eval_tracks, params_from_args


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def write_results(path: str, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def precompute_features(
    tracks: Dict[str, List[TrackPoint]],
    params: BounceRuleParams,
    region_root: str = "",
) -> Dict[str, Dict[str, Sequence]]:
    cached = {}
    for name, points in tracks.items():
        cached[name] = {
            "features": extract_candidate_features(points, params),
            "bounce_frames": true_bounce_frames(points),
            "hit_frames": true_hit_frames(points),
            "playable_regions": load_playable_regions_for_clip(name, region_root),
        }
    return cached


def evaluate_cached(
    cached: Dict[str, Dict[str, Sequence]],
    params: BounceRuleParams,
    tolerance: int,
) -> dict:
    total_tp = total_fp = total_fn = 0
    hit_fp = 0
    frame_errors: List[int] = []

    for item in cached.values():
        candidates: List[BounceCandidate] = score_features(
            item["features"],
            params,
            item.get("playable_regions"),
        )
        pred_frames = [c.frame_id for c in candidates]
        bounce_frames = item["bounce_frames"]
        hit_frames = item["hit_frames"]
        tp, fp, fn, errors = match_events(pred_frames, bounce_frames, tolerance)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        hit_fp += count_near_events(pred_frames, hit_frames, tolerance)
        frame_errors.extend(errors)

    metrics = metrics_from_counts(total_tp, total_fp, total_fn)
    metrics.update(
        {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "hit_fp": hit_fp,
            "avg_frame_error": sum(frame_errors) / len(frame_errors) if frame_errors else 0.0,
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search V4 bounce rule parameters.")
    parser.add_argument("--dataset-root", default="./datasets/trackNet")
    parser.add_argument("--labels-root", default="./datasets/trackNet/images")
    parser.add_argument("--region-root", default="", help="Directory with game-level or clip-level ROI JSON files.")
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--match-tolerance", type=int, default=3)
    parser.add_argument("--out-csv", default="./exps/v4_1_bounce_rule/bounce_rule_tuning.csv")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-score-list", default="11,12,13,14,15")
    parser.add_argument("--angle-weak-max-list", default="8,12,16")
    parser.add_argument("--angle-good-max-list", default="45,55,65")
    parser.add_argument("--angle-hard-max-list", default="80,90,100")
    parser.add_argument("--accel-weak-max-list", default="3,4,5")
    parser.add_argument("--accel-good-max-list", default="12,16,20")
    parser.add_argument("--accel-hard-max-list", default="20,26,32")
    parser.add_argument("--jump-good-max-list", default="30,40,60")
    parser.add_argument(
        "--jump-hard-max-list",
        default="120",
        help="Keep this fixed when using cached tuning because it also affects trajectory cleaning.",
    )
    parser.add_argument("--nms-window-list", default="8,10,12,14")
    parser.add_argument("--hit-speedup-ratio-list", default="1.8,2.2,2.6")
    parser.add_argument("--hit-speedup-delta-list", default="3,4,6")
    parser.add_argument("--hit-speedup-penalty-list", default="2,3,4")
    parser.add_argument("--hit-guard-window-list", default="", help="V4.1 grid. Empty means use --hit-guard-window.")
    parser.add_argument("--hit-guard-penalty-list", default="", help="V4.1 grid. Empty means use --hit-guard-penalty.")
    add_common_args(parser)
    args = parser.parse_args()

    tracks = load_eval_tracks(args)
    base_params = params_from_args(args)
    cached = precompute_features(tracks, base_params, args.region_root)
    results = []

    grids = product(
        parse_float_list(args.min_score_list),
        parse_float_list(args.angle_weak_max_list),
        parse_float_list(args.angle_good_max_list),
        parse_float_list(args.angle_hard_max_list),
        parse_float_list(args.accel_weak_max_list),
        parse_float_list(args.accel_good_max_list),
        parse_float_list(args.accel_hard_max_list),
        parse_float_list(args.jump_good_max_list),
        parse_float_list(args.jump_hard_max_list),
        parse_float_list(args.nms_window_list),
        parse_float_list(args.hit_speedup_ratio_list),
        parse_float_list(args.hit_speedup_delta_list),
        parse_float_list(args.hit_speedup_penalty_list),
        parse_float_list(args.hit_guard_window_list) if args.hit_guard_window_list else [base_params.hit_guard_window],
        parse_float_list(args.hit_guard_penalty_list) if args.hit_guard_penalty_list else [base_params.hit_guard_penalty],
    )

    for (
        min_score,
        angle_weak_max,
        angle_good_max,
        angle_hard_max,
        accel_weak_max,
        accel_good_max,
        accel_hard_max,
        jump_good_max,
        jump_hard_max,
        nms_window,
        hit_speedup_ratio,
        hit_speedup_delta,
        hit_speedup_penalty,
        hit_guard_window,
        hit_guard_penalty,
    ) in grids:
        if angle_weak_max >= base_params.angle_min:
            continue
        if angle_hard_max <= angle_good_max:
            continue
        if accel_weak_max >= accel_good_max:
            continue
        if accel_hard_max <= accel_good_max:
            continue
        if jump_hard_max <= jump_good_max:
            continue

        params = replace(
            base_params,
            min_score=min_score,
            angle_weak_max=angle_weak_max,
            angle_good_max=angle_good_max,
            angle_hard_max=angle_hard_max,
            accel_weak_max=accel_weak_max,
            accel_good_max=accel_good_max,
            accel_hard_max=accel_hard_max,
            jump_good_max=jump_good_max,
            jump_hard_max=jump_hard_max,
            nms_window=int(nms_window),
            hit_speedup_ratio=hit_speedup_ratio,
            hit_speedup_delta=hit_speedup_delta,
            hit_speedup_penalty=hit_speedup_penalty,
            hit_guard_window=int(hit_guard_window),
            hit_guard_penalty=hit_guard_penalty,
        )
        metrics = evaluate_cached(cached, params, args.match_tolerance)
        results.append(
            {
                "precision": round(metrics["precision"], 6),
                "recall": round(metrics["recall"], 6),
                "f1": round(metrics["f1"], 6),
                "hit_fp": metrics["hit_fp"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "avg_frame_error": round(metrics["avg_frame_error"], 4),
                "min_score": min_score,
                "angle_weak_max": angle_weak_max,
                "angle_good_max": angle_good_max,
                "angle_hard_max": angle_hard_max,
                "accel_weak_max": accel_weak_max,
                "accel_good_max": accel_good_max,
                "accel_hard_max": accel_hard_max,
                "jump_good_max": jump_good_max,
                "jump_hard_max": jump_hard_max,
                "nms_window": int(nms_window),
                "hit_speedup_ratio": hit_speedup_ratio,
                "hit_speedup_delta": hit_speedup_delta,
                "hit_speedup_penalty": hit_speedup_penalty,
                "hit_guard_window": int(hit_guard_window),
                "hit_guard_penalty": hit_guard_penalty,
            }
        )

    results.sort(key=lambda r: (-r["f1"], r["hit_fp"], -r["precision"], -r["recall"]))
    write_results(args.out_csv, results)

    print(f"tested = {len(results)}")
    print(f"csv    = {args.out_csv}")
    for row in results[: args.top_k]:
        print(
            "F1={f1:.4f} P={precision:.4f} R={recall:.4f} "
            "HitFP={hit_fp} min_score={min_score} "
            "angle=({angle_weak_max},{angle_good_max},{angle_hard_max}) "
            "accel=({accel_weak_max},{accel_good_max},{accel_hard_max}) "
            "jump=({jump_good_max},{jump_hard_max}) "
            "nms={nms_window} "
            "speedup=({hit_speedup_ratio},{hit_speedup_delta},{hit_speedup_penalty}) "
            "hit_guard=({hit_guard_window},{hit_guard_penalty})".format(**row)
        )


if __name__ == "__main__":
    main()
