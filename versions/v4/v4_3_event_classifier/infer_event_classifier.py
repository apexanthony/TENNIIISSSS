import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
V41_DIR = ROOT / "versions" / "v4" / "v4_1_bounce_rule"
V42_DIR = ROOT / "versions" / "v4" / "v4_2_bounce_classifier"
V43_DIR = ROOT / "versions" / "v4" / "v4_3_event_classifier"
for path in (V41_DIR, V42_DIR, V43_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bounce_rule_detector import (  # noqa: E402
    BounceCandidate,
    BounceRuleParams,
    TrackPoint,
    candidate_score_with_region,
    extract_candidate_features,
    load_playable_regions,
    load_track_csv,
    nms_candidates,
    point_in_regions,
    recent_hit_like_penalty,
)
from infer_bounce_classifier import (  # noqa: E402
    draw_inference_video,
    video_fps,
    write_bounce_csv,
)
from train_eval_bounce_classifier import feature_context, make_feature_vector  # noqa: E402
from train_eval_event_classifier import (  # noqa: E402
    FEATURE_NAMES,
    default_rule_params,
    extra_features,
)


def load_event_model(path: Path) -> Tuple[object, object, float, dict]:
    with path.open("rb") as fp:
        payload = pickle.load(fp)
    names = list(payload.get("feature_names") or [])
    if names != FEATURE_NAMES:
        raise RuntimeError("V4.3 model feature names do not match current V4.3 feature schema.")
    return (
        payload["event_model"],
        payload["offset_model"],
        float(payload.get("threshold", 0.5)),
        dict(payload.get("params") or {}),
    )


def corrected_candidate(
    feature: BounceCandidate,
    corrected_frame: int,
    points_by_frame: Dict[int, TrackPoint],
    probability: float,
) -> BounceCandidate:
    point = points_by_frame.get(corrected_frame)
    if point is None or not point.valid:
        point = TrackPoint(corrected_frame, feature.x, feature.y, feature.confidence, True)
    return BounceCandidate(
        frame_id=corrected_frame,
        x=point.x,
        y=point.y,
        score=probability,
        confidence=point.confidence,
        angle_change=feature.angle_change,
        accel_norm=feature.accel_norm,
        speed_before=feature.speed_before,
        speed_after=feature.speed_after,
        jump_distance=feature.jump_distance,
        valid_points=feature.valid_points,
    )


def build_samples(
    points: Sequence[TrackPoint],
    params: BounceRuleParams,
    playable_regions,
    candidate_min_score: float,
    min_conf: float,
) -> List[dict]:
    features = extract_candidate_features(points, params)
    ctx = feature_context(features)
    previous_features: List[BounceCandidate] = []
    samples: List[dict] = []
    for feature in features:
        score = candidate_score_with_region(feature, params, playable_regions)
        if score is None:
            previous_features.append(feature)
            continue

        recent_penalty = recent_hit_like_penalty(feature, previous_features, params)
        adjusted_score = score - recent_penalty
        previous_features.append(feature)
        if adjusted_score < candidate_min_score:
            continue

        in_region = 1.0
        if playable_regions:
            in_region = 1.0 if point_in_regions(feature.x, feature.y, playable_regions) else 0.0

        base = make_feature_vector(
            feature,
            adjusted_score,
            recent_penalty,
            in_region,
            len(points),
            ctx,
        )
        samples.append(
            {
                "frame_id": feature.frame_id,
                "x": feature.x,
                "y": feature.y,
                "feature": feature,
                "rule_score": adjusted_score,
                "recent_hit_penalty": recent_penalty,
                "in_ground_region": in_region,
                "vector": base + extra_features(feature, points, min_conf),
            }
        )
    return samples


def predict_rows(
    event_model,
    offset_model,
    samples: Sequence[dict],
    points: Sequence[TrackPoint],
    threshold: float,
    nms_window: int,
    offset_min: int,
    offset_max: int,
) -> List[dict]:
    if not samples:
        return []

    x = np.asarray([sample["vector"] for sample in samples], dtype=np.float32)
    probabilities = event_model.predict_proba(x)[:, 1]
    offsets = offset_model.predict(x)
    points_by_frame = {point.frame_id: point for point in points}

    candidates: List[BounceCandidate] = []
    by_frame: Dict[int, dict] = {}
    for sample, probability, offset in zip(samples, probabilities, offsets):
        probability = float(probability)
        if probability < threshold:
            continue
        offset = int(max(offset_min, min(offset_max, int(offset))))
        original: BounceCandidate = sample["feature"]
        corrected_frame = max(0, original.frame_id + offset)
        candidate = corrected_candidate(original, corrected_frame, points_by_frame, probability)
        row = dict(sample)
        row["feature"] = candidate
        row["candidate"] = candidate
        row["probability"] = probability
        row["original_frame_id"] = original.frame_id
        row["refined_offset"] = offset
        row["refine_score"] = 0.0
        row["gap_filled"] = 0
        candidates.append(candidate)
        by_frame[candidate.frame_id] = row

    kept = nms_candidates(candidates, nms_window)
    return sorted((by_frame[candidate.frame_id] for candidate in kept), key=lambda item: item["feature"].frame_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer bounce events with the V4.3 event + offset classifier.")
    parser.add_argument("--track-csv", required=True)
    parser.add_argument("--model-path", default="./exps/v4_3_event_classifier_cand2_final/model_event_offset.pkl")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--video-path", default="")
    parser.add_argument("--video-out-path", default="")
    parser.add_argument("--court-region-json", default="")
    parser.add_argument("--candidate-min-score", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--nms-window", type=int, default=22)
    parser.add_argument("--region-penalty", type=float, default=4.0)
    parser.add_argument("--jump-hard-max", type=float, default=120.0)
    parser.add_argument("--min-conf", type=float, default=0.30)
    parser.add_argument("--offset-min", type=int, default=None)
    parser.add_argument("--offset-max", type=int, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--trail", type=int, default=7)
    parser.add_argument("--visual-min-conf", type=float, default=0.70)
    parser.add_argument("--visual-jump-max", type=float, default=180.0)
    parser.add_argument("--visual-neighbor-window", type=int, default=2)
    parser.add_argument("--show-bounce-history-labels", action="store_true")
    args = parser.parse_args()

    event_model, offset_model, saved_threshold, saved_params = load_event_model(Path(args.model_path))
    threshold = saved_threshold if args.threshold is None else args.threshold
    candidate_min_score = (
        float(saved_params.get("candidate_min_score", 2.0))
        if args.candidate_min_score is None
        else args.candidate_min_score
    )
    offset_min = int(saved_params.get("offset_min", -3)) if args.offset_min is None else args.offset_min
    offset_max = int(saved_params.get("offset_max", 10)) if args.offset_max is None else args.offset_max

    points = load_track_csv(args.track_csv)
    playable_regions = load_playable_regions(args.court_region_json)
    params = default_rule_params(args.region_penalty)
    params.jump_hard_max = args.jump_hard_max
    samples = build_samples(points, params, playable_regions, candidate_min_score, args.min_conf)
    rows = predict_rows(
        event_model,
        offset_model,
        samples,
        points,
        threshold,
        args.nms_window,
        offset_min,
        offset_max,
    )

    fps = video_fps(args.video_path, args.fps)
    write_bounce_csv(Path(args.out_csv), rows, fps)

    print(f"track_points      = {len(points)}")
    print(f"candidate_samples = {len(samples)}")
    print(f"threshold         = {threshold:.4f}")
    print(f"bounces           = {len(rows)}")
    print(f"csv               = {args.out_csv}")

    if args.video_path and args.video_out_path:
        draw_inference_video(
            args.video_path,
            Path(args.video_out_path),
            points,
            rows,
            args.codec,
            args.trail,
            fps,
            args.visual_min_conf,
            args.visual_jump_max,
            args.visual_neighbor_window,
            args.show_bounce_history_labels,
        )
        print(f"video             = {args.video_out_path}")


if __name__ == "__main__":
    main()
