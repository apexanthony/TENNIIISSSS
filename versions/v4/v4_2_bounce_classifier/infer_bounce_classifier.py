import argparse
import csv
import pickle
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
V4_DIR = ROOT / "versions" / "v4" / "v4_1_bounce_rule"
if str(V4_DIR) not in sys.path:
    sys.path.insert(0, str(V4_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

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
from train_eval_bounce_classifier import (  # noqa: E402
    FEATURE_NAMES,
    feature_context,
    make_feature_vector,
)


def default_rule_params(region_penalty: float) -> BounceRuleParams:
    return BounceRuleParams(
        region_bonus=0.0,
        region_penalty=region_penalty,
        min_score=9.5,
        enable_v41=True,
        enable_v42=True,
    )


def candidate_from_feature(feature: BounceCandidate, probability: float) -> BounceCandidate:
    return BounceCandidate(
        frame_id=feature.frame_id,
        x=feature.x,
        y=feature.y,
        score=probability,
        confidence=feature.confidence,
        angle_change=feature.angle_change,
        accel_norm=feature.accel_norm,
        speed_before=feature.speed_before,
        speed_after=feature.speed_after,
        jump_distance=feature.jump_distance,
        valid_points=feature.valid_points,
    )


def candidate_from_point(
    point: TrackPoint,
    probability: float,
    source: BounceCandidate,
) -> BounceCandidate:
    return BounceCandidate(
        frame_id=point.frame_id,
        x=point.x,
        y=point.y,
        score=probability,
        confidence=point.confidence,
        angle_change=source.angle_change,
        accel_norm=source.accel_norm,
        speed_before=source.speed_before,
        speed_after=source.speed_after,
        jump_distance=source.jump_distance,
        valid_points=source.valid_points,
    )


def load_model(path: Path) -> Tuple[object, List[str], float, dict]:
    with path.open("rb") as fp:
        payload = pickle.load(fp)
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        names = list(payload.get("feature_names") or FEATURE_NAMES)
        threshold = float(payload.get("threshold", 0.5))
        params = dict(payload.get("params") or {})
        return model, names, threshold, params
    return payload, FEATURE_NAMES, 0.5, {}


def build_inference_samples(
    points: Sequence[TrackPoint],
    params: BounceRuleParams,
    playable_regions,
    candidate_min_score: float,
) -> List[dict]:
    features = extract_candidate_features(points, params)
    ctx = feature_context(features)
    previous_features: List[BounceCandidate] = []
    samples: List[dict] = []
    for feature in features:
        rule_score = candidate_score_with_region(feature, params, playable_regions)
        if rule_score is None:
            previous_features.append(feature)
            continue

        recent_penalty = recent_hit_like_penalty(feature, previous_features, params)
        adjusted_score = rule_score - recent_penalty
        previous_features.append(feature)
        if adjusted_score < candidate_min_score:
            continue

        in_region = 1.0
        if playable_regions:
            in_region = 1.0 if point_in_regions(feature.x, feature.y, playable_regions) else 0.0

        samples.append(
            {
                "frame_id": feature.frame_id,
                "x": feature.x,
                "y": feature.y,
                "feature": feature,
                "rule_score": adjusted_score,
                "recent_hit_penalty": recent_penalty,
                "in_ground_region": in_region,
                "vector": make_feature_vector(
                    feature,
                    adjusted_score,
                    recent_penalty,
                    in_region,
                    len(points),
                    ctx,
                ),
            }
        )
    return samples


def predict_candidate_rows(
    model,
    samples: Sequence[dict],
    threshold: float,
) -> List[dict]:
    if not samples:
        return []

    x = np.asarray([sample["vector"] for sample in samples], dtype=np.float32)
    probabilities = model.predict_proba(x)[:, 1]
    rows: List[dict] = []

    for sample, probability in zip(samples, probabilities):
        if probability < threshold:
            continue
        row = dict(sample)
        row["probability"] = float(probability)
        row["original_frame_id"] = sample["feature"].frame_id
        row["refined_offset"] = 0
        row["refine_score"] = 0.0
        rows.append(row)
    return sorted(rows, key=lambda item: item["frame_id"])


def legacy_nms_rows(rows: Sequence[dict], nms_window: int) -> List[dict]:
    candidates: List[BounceCandidate] = []
    by_frame: Dict[int, dict] = {}
    for row in rows:
        candidate = candidate_from_feature(row["feature"], row["probability"])
        candidates.append(candidate)
        by_frame[candidate.frame_id] = row

    kept = nms_candidates(candidates, nms_window)
    out: List[dict] = []
    for candidate in kept:
        row = by_frame[candidate.frame_id]
        row["candidate"] = candidate
        out.append(row)
    return sorted(out, key=lambda item: item["frame_id"])


def group_candidate_rows(
    rows: Sequence[dict],
    event_merge_gap: int,
    event_max_span: int,
) -> List[List[dict]]:
    groups: List[List[dict]] = []
    for row in sorted(rows, key=lambda item: item["frame_id"]):
        starts_new_group = (
            not groups
            or row["frame_id"] - groups[-1][-1]["frame_id"] > event_merge_gap
            or row["frame_id"] - groups[-1][0]["frame_id"] > event_max_span
        )
        if starts_new_group:
            groups.append([row])
        else:
            groups[-1].append(row)
    return groups


def choose_group_representative(group: Sequence[dict]) -> dict:
    return dict(
        max(
            group,
            key=lambda item: (
                item["probability"],
                item["rule_score"],
                item["feature"].confidence,
            ),
        )
    )


def point_distance_xy(a: TrackPoint, b: TrackPoint) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def valid_window_points(
    points_by_frame: Dict[int, TrackPoint],
    start: int,
    end: int,
    min_conf: float,
    max_step: float,
) -> List[TrackPoint]:
    out: List[TrackPoint] = []
    prev: Optional[TrackPoint] = None
    for frame_id in range(start, end + 1):
        point = points_by_frame.get(frame_id)
        if point is None or not point.valid or point.confidence < min_conf:
            continue
        if prev is not None and point_distance_xy(prev, point) > max_step:
            prev = point
            continue
        out.append(point)
        prev = point
    return out


def velocity_between(a: TrackPoint, b: TrackPoint) -> np.ndarray:
    frame_gap = max(1, b.frame_id - a.frame_id)
    return np.asarray([(b.x - a.x) / frame_gap, (b.y - a.y) / frame_gap], dtype=np.float32)


def angle_score(before: np.ndarray, after: np.ndarray) -> float:
    nb = float(np.linalg.norm(before))
    na = float(np.linalg.norm(after))
    if nb < 1e-6 or na < 1e-6:
        return 0.0
    cosine = float(np.clip(np.dot(before, after) / (nb * na), -1.0, 1.0))
    return (1.0 - cosine) * 0.5


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def refine_single_bounce(
    row: dict,
    points_by_frame: Dict[int, TrackPoint],
    refine_back: int,
    refine_forward: int,
    min_conf: float,
    max_step: float,
    strong_source_prob: float,
    strong_source_rule: float,
    strong_source_angle: float,
    strong_source_accel: float,
    strong_source_max_forward: int,
) -> dict:
    source: BounceCandidate = row["feature"]
    start = source.frame_id - refine_back
    is_strong_source = (
        row["probability"] >= strong_source_prob
        and row["rule_score"] >= strong_source_rule
        and source.angle_change >= strong_source_angle
        and source.accel_norm >= strong_source_accel
    )
    if is_strong_source:
        # A high-confidence geometric bounce candidate is already close to contact time.
        # In this case, chasing the later image-space lowest point over-shifts bounces
        # under perspective, especially between near/far court halves.
        refine_back = 0
        refine_forward = min(refine_forward, strong_source_max_forward)
    start = source.frame_id - refine_back
    end = source.frame_id + refine_forward
    window_points = valid_window_points(points_by_frame, start, end, min_conf, max_step)
    if len(window_points) < 3:
        out = dict(row)
        out["candidate"] = candidate_from_feature(source, row["probability"])
        return out

    y_values = [point.y for point in window_points]
    y_min = min(y_values)
    y_range = max(max(y_values) - y_min, 1e-6)

    accel_values: Dict[int, float] = {}
    reverse_values: Dict[int, float] = {}
    for point in window_points:
        prev = points_by_frame.get(point.frame_id - 1)
        nxt = points_by_frame.get(point.frame_id + 1)
        if prev is None or nxt is None or not prev.valid or not nxt.valid:
            continue
        if prev.confidence < min_conf or nxt.confidence < min_conf:
            continue
        if point_distance_xy(prev, point) > max_step or point_distance_xy(point, nxt) > max_step:
            continue
        before = velocity_between(prev, point)
        after = velocity_between(point, nxt)
        accel_values[point.frame_id] = float(np.linalg.norm(after - before))
        reverse_values[point.frame_id] = angle_score(before, after)

    max_accel = max(accel_values.values(), default=1e-6)
    best_point = window_points[0]
    best_score = float("-inf")
    for point in window_points:
        offset = point.frame_id - source.frame_id
        lowest_score = (point.y - y_min) / y_range
        reverse_score = reverse_values.get(point.frame_id, 0.0)
        accel_score = accel_values.get(point.frame_id, 0.0) / max_accel

        prev_conf = [
            points_by_frame[f].confidence
            for f in range(point.frame_id - 3, point.frame_id)
            if f in points_by_frame and points_by_frame[f].valid
        ]
        next_conf = [
            points_by_frame[f].confidence
            for f in range(point.frame_id, point.frame_id + 4)
            if f in points_by_frame and points_by_frame[f].valid
        ]
        prev_mean = float(np.mean(prev_conf)) if prev_conf else point.confidence
        next_mean = float(np.mean(next_conf)) if next_conf else point.confidence
        confidence_recovery = clamp01((next_mean - prev_mean + 0.2) / 0.4)
        confidence_score = clamp01(0.5 * point.confidence + 0.5 * confidence_recovery)
        forward_score = clamp01((offset + refine_back) / max(1, refine_back + refine_forward))

        score = (
            0.28 * lowest_score
            + 0.28 * reverse_score
            + 0.22 * accel_score
            + 0.14 * confidence_score
            + 0.08 * forward_score
        )
        if score > best_score:
            best_score = score
            best_point = point

    out = dict(row)
    out["original_frame_id"] = source.frame_id
    out["refined_offset"] = best_point.frame_id - source.frame_id
    out["refine_score"] = float(best_score)
    out["feature"] = candidate_from_point(best_point, row["probability"], source)
    out["candidate"] = out["feature"]
    return out


def refine_bounce_rows(
    rows: Sequence[dict],
    points: Sequence[TrackPoint],
    refine_back: int,
    refine_forward: int,
    min_conf: float,
    max_step: float,
    strong_source_prob: float,
    strong_source_rule: float,
    strong_source_angle: float,
    strong_source_accel: float,
    strong_source_max_forward: int,
) -> List[dict]:
    points_by_frame = {point.frame_id: point for point in points}
    return [
        refine_single_bounce(
            row,
            points_by_frame,
            refine_back,
            refine_forward,
            min_conf,
            max_step,
            strong_source_prob,
            strong_source_rule,
            strong_source_angle,
            strong_source_accel,
            strong_source_max_forward,
        )
        for row in rows
    ]


def score_for_merge(row: dict) -> float:
    return float(row["probability"]) + 0.08 * float(row.get("refine_score", 0.0))


def merge_refined_rows(rows: Sequence[dict], min_event_gap: int) -> List[dict]:
    kept: List[dict] = []
    for row in sorted(rows, key=lambda item: (-score_for_merge(item), item["feature"].frame_id)):
        frame_id = row["feature"].frame_id
        if any(abs(frame_id - prev["feature"].frame_id) <= min_event_gap for prev in kept):
            continue
        kept.append(row)
    return sorted(kept, key=lambda item: item["feature"].frame_id)


def filter_final_rows(
    rows: Sequence[dict],
    min_confidence: float,
    min_rule_score: float,
    min_refine_score: float,
) -> List[dict]:
    out: List[dict] = []
    for row in rows:
        feature: BounceCandidate = row["feature"]
        if feature.confidence < min_confidence:
            continue
        if float(row.get("rule_score", 0.0)) < min_rule_score:
            continue
        if float(row.get("refine_score", 0.0)) < min_refine_score:
            continue
        out.append(row)
    return out


def choose_gap_fill_row(group: Sequence[dict], mode: str) -> dict:
    if mode == "earliest":
        return dict(min(group, key=lambda item: item["frame_id"]))
    if mode == "latest":
        return dict(max(group, key=lambda item: item["frame_id"]))
    return choose_group_representative(group)


def add_leading_gap_fill(
    rows: Sequence[dict],
    low_threshold_rows: Sequence[dict],
    lookback: int,
    end_margin: int,
    min_first_frame: int,
    event_merge_gap: int,
    event_max_span: int,
    mode: str,
) -> List[dict]:
    if not rows:
        return list(rows)

    first_frame = rows[0]["feature"].frame_id
    if first_frame < min_first_frame:
        return list(rows)

    start = max(0, first_frame - lookback)
    end = max(start, first_frame - end_margin)
    candidates = [
        row
        for row in low_threshold_rows
        if start <= row["frame_id"] <= end
        and abs(row["frame_id"] - first_frame) > end_margin
    ]
    if not candidates:
        return list(rows)

    groups = group_candidate_rows(candidates, event_merge_gap, event_max_span)
    if not groups:
        return list(rows)

    # For the leading missed bounce, prefer the last low-threshold candidate run before the
    # first high-confidence event; this avoids pulling in serve-preparation noise.
    group = max(groups, key=lambda item: item[-1]["frame_id"])
    row = choose_gap_fill_row(group, mode)
    row["gap_filled"] = 1
    row["candidate"] = candidate_from_feature(row["feature"], row["probability"])
    return merge_refined_rows([*rows, row], end_margin)


def predict_bounces(
    model,
    samples: Sequence[dict],
    threshold: float,
    nms_window: int,
    adaptive_nms: bool,
    event_merge_gap: int,
    event_max_span: int,
) -> List[dict]:
    rows = predict_candidate_rows(model, samples, threshold)
    if adaptive_nms:
        grouped = group_candidate_rows(rows, event_merge_gap, event_max_span)
        out = [choose_group_representative(group) for group in grouped]
        for row in out:
            row["candidate"] = candidate_from_feature(row["feature"], row["probability"])
        return sorted(out, key=lambda item: item["frame_id"])
    return legacy_nms_rows(rows, nms_window)


def video_fps(video_path: str, fallback: float) -> float:
    if not video_path:
        return fallback
    cap = cv2.VideoCapture(video_path)
    fps = fallback
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or fallback
    cap.release()
    return fps


def write_bounce_csv(path: Path, rows: Iterable[dict], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bounce_id",
        "frame_id",
        "original_frame_id",
        "refined_offset",
        "gap_filled",
        "time_sec",
        "x",
        "y",
        "probability",
        "refine_score",
        "rule_score",
        "confidence",
        "angle_change",
        "accel_norm",
        "speed_before",
        "speed_after",
        "jump_distance",
        "valid_points",
        "recent_hit_penalty",
        "in_ground_region",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows, 1):
            feature: BounceCandidate = row["feature"]
            writer.writerow(
                {
                    "bounce_id": idx,
                    "frame_id": feature.frame_id,
                    "original_frame_id": row.get("original_frame_id", feature.frame_id),
                    "refined_offset": row.get("refined_offset", 0),
                    "gap_filled": row.get("gap_filled", 0),
                    "time_sec": round(feature.frame_id / fps, 4) if fps > 0 else "",
                    "x": round(feature.x, 2),
                    "y": round(feature.y, 2),
                    "probability": round(row["probability"], 4),
                    "refine_score": round(row.get("refine_score", 0.0), 4),
                    "rule_score": round(row["rule_score"], 4),
                    "confidence": round(feature.confidence, 4),
                    "angle_change": round(feature.angle_change, 4),
                    "accel_norm": round(feature.accel_norm, 4),
                    "speed_before": round(feature.speed_before, 4),
                    "speed_after": round(feature.speed_after, 4),
                    "jump_distance": round(feature.jump_distance, 4),
                    "valid_points": feature.valid_points,
                    "recent_hit_penalty": round(row["recent_hit_penalty"], 4),
                    "in_ground_region": int(row["in_ground_region"]),
                }
            )


def draw_v3_trace(frame, trace_points) -> np.ndarray:
    for age, point in enumerate(reversed(trace_points)):
        if point is None:
            continue
        x, y = point
        thickness = max(2, 10 - age)
        cv2.circle(frame, (int(x), int(y)), radius=0, color=(0, 0, 255), thickness=thickness)
    return frame


def distance(a: TrackPoint, b: TrackPoint) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def build_visual_track_points(
    points: Sequence[TrackPoint],
    min_conf: float,
    max_jump: float,
    neighbor_window: int,
) -> Dict[int, Tuple[float, float]]:
    candidates = [point for point in points if point.valid and point.confidence >= min_conf]
    by_frame = {point.frame_id: point for point in candidates}
    visual: Dict[int, Tuple[float, float]] = {}
    for point in candidates:
        has_neighbor = False
        for offset in range(1, neighbor_window + 1):
            prev = by_frame.get(point.frame_id - offset)
            if prev is not None and distance(point, prev) <= max_jump:
                has_neighbor = True
                break
            nxt = by_frame.get(point.frame_id + offset)
            if nxt is not None and distance(point, nxt) <= max_jump:
                has_neighbor = True
                break
        if has_neighbor:
            visual[point.frame_id] = (point.x, point.y)
    return visual


def draw_inference_video(
    video_path: str,
    out_path: Path,
    points: Sequence[TrackPoint],
    bounce_rows: Sequence[dict],
    codec: str,
    trail: int,
    fps: float,
    visual_min_conf: float,
    visual_jump_max: float,
    visual_neighbor_window: int,
    show_bounce_history_labels: bool,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"failed to open video writer: {out_path}")

    track_by_frame = build_visual_track_points(
        points,
        visual_min_conf,
        visual_jump_max,
        visual_neighbor_window,
    )
    bounces_by_frame = {row["feature"].frame_id: row for row in bounce_rows}
    bounce_history: List[dict] = []
    trace_points = deque(maxlen=trail)

    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        trace_points.append(track_by_frame.get(frame_id))
        draw_v3_trace(frame, trace_points)

        if frame_id in bounces_by_frame:
            bounce_history.append(bounces_by_frame[frame_id])

        for row in bounce_history:
            feature: BounceCandidate = row["feature"]
            center = (int(round(feature.x)), int(round(feature.y)))
            cv2.circle(frame, center, 14, (0, 255, 255), 2)
            cv2.circle(frame, center, 3, (0, 255, 255), -1)
            if show_bounce_history_labels:
                cv2.putText(
                    frame,
                    f"F{feature.frame_id}",
                    (center[0] + 10, center[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        current = bounces_by_frame.get(frame_id)
        if current is not None:
            feature = current["feature"]
            center = (int(round(feature.x)), int(round(feature.y)))
            cv2.circle(frame, center, 22, (0, 0, 255), 3)
            cv2.putText(
                frame,
                f"Bounce F{feature.frame_id} {feature.frame_id / max(fps, 1e-6):.2f}s p={current['probability']:.2f}",
                (center[0] + 12, max(28, center[1] - 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.rectangle(frame, (18, 18), (430, 72), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"Bounce frame: {feature.frame_id}  time: {feature.frame_id / max(fps, 1e-6):.2f}s",
                (32, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer tennis bounce events with the V4.2 trajectory-feature classifier.")
    parser.add_argument("--track-csv", required=True, help="V3 track CSV with frame,x,y,score columns.")
    parser.add_argument("--model-path", default="./exps/v4_2_bounce_classifier/model_random_forest.pkl")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--video-path", default="")
    parser.add_argument("--video-out-path", default="")
    parser.add_argument("--court-region-json", default="")
    parser.add_argument("--candidate-min-score", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--nms-window", type=int, default=22)
    parser.add_argument("--adaptive-nms", action="store_true", help="Group dense candidate runs before event merging.")
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
    parser.add_argument(
        "--jump-hard-max",
        type=float,
        default=120.0,
        help="Max allowed inter-frame jump during trajectory cleaning. Raise it for videos with shot changes or noisy early tracks.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--trail", type=int, default=7)
    parser.add_argument("--visual-min-conf", type=float, default=0.70)
    parser.add_argument("--visual-jump-max", type=float, default=180.0)
    parser.add_argument("--visual-neighbor-window", type=int, default=2)
    parser.add_argument("--show-bounce-history-labels", action="store_true")
    args = parser.parse_args()

    model, names, saved_threshold, saved_params = load_model(Path(args.model_path))
    if names != FEATURE_NAMES:
        raise RuntimeError("Model feature names do not match the current V4.2 feature schema.")

    threshold = saved_threshold if args.threshold is None else args.threshold
    candidate_min_score = (
        float(saved_params.get("candidate_min_score", 3.0))
        if args.candidate_min_score is None
        else args.candidate_min_score
    )

    points = load_track_csv(args.track_csv)
    playable_regions = load_playable_regions(args.court_region_json)
    params = default_rule_params(args.region_penalty)
    params.jump_hard_max = args.jump_hard_max
    samples = build_inference_samples(points, params, playable_regions, candidate_min_score)
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

    fps = video_fps(args.video_path, args.fps)
    write_bounce_csv(Path(args.out_csv), bounce_rows, fps)

    print(f"track_points      = {len(points)}")
    print(f"candidate_samples = {len(samples)}")
    print(f"threshold         = {threshold:.4f}")
    print(f"adaptive_nms      = {int(args.adaptive_nms)}")
    print(f"refine            = {int(args.enable_refine)}")
    print(f"bounces           = {len(bounce_rows)}")
    print(f"csv               = {args.out_csv}")

    if args.video_path and args.video_out_path:
        draw_inference_video(
            args.video_path,
            Path(args.video_out_path),
            points,
            bounce_rows,
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
