import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
V41_DIR = ROOT / "versions" / "v4" / "v4_1_bounce_rule"
V42_DIR = ROOT / "versions" / "v4" / "v4_2_bounce_classifier"
for path in (V41_DIR, V42_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bounce_rule_detector import (  # noqa: E402
    BounceRuleParams,
    TrackPoint,
    load_label_csv,
    load_playable_regions_for_clip,
    match_events,
    metrics_from_counts,
    true_bounce_frames,
)
from infer_bounce_classifier import (  # noqa: E402
    add_leading_gap_fill,
    build_inference_samples,
    default_rule_params,
    filter_final_rows,
    load_model,
    merge_refined_rows,
    predict_bounces,
    predict_candidate_rows,
    refine_bounce_rows,
)


def parse_games(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def split_clip_name(label_path: Path) -> str:
    return "/".join(label_path.parts[-3:-1])


def clip_game(clip: str) -> str:
    return clip.replace("\\", "/").split("/")[0]


def load_tracks(labels_root: Path, games: Sequence[str]) -> Dict[str, List[TrackPoint]]:
    game_set = set(games)
    tracks: Dict[str, List[TrackPoint]] = {}
    for label_path in sorted(labels_root.glob("game*/Clip*/Label.csv")):
        clip = split_clip_name(label_path)
        if game_set and clip_game(clip) not in game_set:
            continue
        tracks[clip] = load_label_csv(str(label_path))
    return tracks


def rows_to_frames(rows: Sequence[dict]) -> List[int]:
    return [row["feature"].frame_id for row in rows]


def evaluate_tracks(
    tracks: Dict[str, List[TrackPoint]],
    model,
    threshold: float,
    args: argparse.Namespace,
) -> tuple[dict, List[dict]]:
    total_tp = total_fp = total_fn = 0
    errors: List[int] = []
    rows: List[dict] = []

    params: BounceRuleParams = default_rule_params(args.region_penalty)
    params.jump_hard_max = args.jump_hard_max

    for clip, points in sorted(tracks.items()):
        regions = load_playable_regions_for_clip(clip, args.region_root)
        samples = build_inference_samples(points, params, regions, args.candidate_min_score)
        bounce_rows = predict_bounces(
            model,
            samples,
            threshold,
            args.nms_window,
            args.adaptive_nms,
            args.event_merge_gap,
            args.event_max_span,
        )

        if args.enable_refine:
            bounce_rows = refine_bounce_rows(
                bounce_rows,
                points,
                args.refine_back,
                args.refine_forward,
                args.refine_min_conf,
                args.refine_max_step,
                args.strong_source_prob,
                args.strong_source_rule,
                args.strong_source_angle,
                args.strong_source_accel,
                args.strong_source_max_forward,
            )
            bounce_rows = merge_refined_rows(bounce_rows, args.min_event_gap)

        if args.final_min_confidence > 0 or args.final_min_rule_score > -999.0 or args.final_min_refine_score > 0:
            bounce_rows = filter_final_rows(
                bounce_rows,
                args.final_min_confidence,
                args.final_min_rule_score,
                args.final_min_refine_score,
            )

        if args.enable_leading_gap_fill:
            low_rows = predict_candidate_rows(model, samples, args.gap_fill_threshold)
            bounce_rows = add_leading_gap_fill(
                bounce_rows,
                low_rows,
                args.gap_fill_lookback,
                args.gap_fill_end_margin,
                args.gap_fill_min_first_frame,
                args.event_merge_gap,
                args.event_max_span,
                args.gap_fill_mode,
            )

        pred_frames = rows_to_frames(bounce_rows)
        true_frames = true_bounce_frames(points)
        tp, fp, fn, clip_errors = match_events(pred_frames, true_frames, args.match_tolerance)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        errors.extend(clip_errors)

        rows.append(
            {
                "clip": clip,
                "true_events": len(true_frames),
                "pred_events": len(pred_frames),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "pred_frames": " ".join(str(frame) for frame in pred_frames),
                "true_frames": " ".join(str(frame) for frame in true_frames),
                "errors": " ".join(str(error) for error in clip_errors),
            }
        )

    metrics = metrics_from_counts(total_tp, total_fp, total_fn)
    metrics.update(
        {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "predictions": total_tp + total_fp,
            "avg_frame_error": sum(errors) / len(errors) if errors else 0.0,
            "clips": len(tracks),
        }
    )
    return metrics, rows


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V4.2 bounce classifier post-processing on TrackNet labels.")
    parser.add_argument("--labels-root", default="./datasets/trackNet/images")
    parser.add_argument("--region-root", default="./configs/court_regions")
    parser.add_argument("--model-path", default="./exps/v4_2_bounce_classifier/model_random_forest.pkl")
    parser.add_argument("--games", default="game8,game9,game10")
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--candidate-min-score", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--match-tolerance", type=int, default=3)
    parser.add_argument("--nms-window", type=int, default=22)
    parser.add_argument("--adaptive-nms", action="store_true")
    parser.add_argument("--event-merge-gap", type=int, default=12)
    parser.add_argument("--event-max-span", type=int, default=18)
    parser.add_argument("--min-event-gap", type=int, default=12)
    parser.add_argument("--enable-refine", action="store_true")
    parser.add_argument("--refine-back", type=int, default=2)
    parser.add_argument("--refine-forward", type=int, default=10)
    parser.add_argument("--refine-min-conf", type=float, default=0.30)
    parser.add_argument("--refine-max-step", type=float, default=260.0)
    parser.add_argument("--strong-source-prob", type=float, default=0.90)
    parser.add_argument("--strong-source-rule", type=float, default=10.0)
    parser.add_argument("--strong-source-angle", type=float, default=25.0)
    parser.add_argument("--strong-source-accel", type=float, default=18.0)
    parser.add_argument("--strong-source-max-forward", type=int, default=3)
    parser.add_argument("--final-min-confidence", type=float, default=0.0)
    parser.add_argument("--final-min-rule-score", type=float, default=-999.0)
    parser.add_argument("--final-min-refine-score", type=float, default=0.0)
    parser.add_argument("--enable-leading-gap-fill", action="store_true")
    parser.add_argument("--gap-fill-threshold", type=float, default=0.09)
    parser.add_argument("--gap-fill-lookback", type=int, default=70)
    parser.add_argument("--gap-fill-end-margin", type=int, default=20)
    parser.add_argument("--gap-fill-min-first-frame", type=int, default=190)
    parser.add_argument("--gap-fill-mode", choices=["best", "earliest", "latest"], default="earliest")
    parser.add_argument("--region-penalty", type=float, default=4.0)
    parser.add_argument("--jump-hard-max", type=float, default=120.0)
    args = parser.parse_args()

    model, _, saved_threshold, _ = load_model(Path(args.model_path))
    threshold = saved_threshold if args.threshold is None else args.threshold
    tracks = load_tracks(Path(args.labels_root), parse_games(args.games))
    metrics, rows = evaluate_tracks(tracks, model, threshold, args)

    print(f"games       = {args.games}")
    print(f"clips       = {metrics['clips']}")
    print(f"threshold   = {threshold:.4f}")
    print(f"TP          = {metrics['tp']}")
    print(f"FP          = {metrics['fp']}")
    print(f"FN          = {metrics['fn']}")
    print(f"Predictions = {metrics['predictions']}")
    print(f"Precision   = {metrics['precision']:.4f}")
    print(f"Recall      = {metrics['recall']:.4f}")
    print(f"F1          = {metrics['f1']:.4f}")
    print(f"AvgErr      = {metrics['avg_frame_error']:.3f} frames")

    if args.out_csv:
        write_rows(Path(args.out_csv), rows)
        print(f"rows saved  = {args.out_csv}")


if __name__ == "__main__":
    main()
