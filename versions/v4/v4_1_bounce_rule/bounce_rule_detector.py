import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class TrackPoint:
    frame_id: int
    x: float
    y: float
    confidence: float = 1.0
    valid: bool = True
    status: Optional[int] = None


@dataclass
class BounceCandidate:
    frame_id: int
    x: float
    y: float
    score: float
    confidence: float
    angle_change: float
    accel_norm: float
    speed_before: float
    speed_after: float
    jump_distance: float
    valid_points: int


@dataclass
class BounceRuleParams:
    min_conf: float = 0.30
    max_gap: int = 3
    smooth_window: int = 5
    before_window: int = 3
    after_window: int = 3
    min_speed: float = 2.0
    speed_max: float = 90.0
    speed_ratio_max: float = 5.0
    hit_speedup_ratio: float = 1.2
    hit_speedup_delta: float = 7.0
    hit_speedup_penalty: float = 2.0
    angle_min: float = 20.0
    angle_weak_max: float = 12.0
    angle_good_max: float = 75.0
    angle_hard_max: float = 90.0
    accel_min: float = 3.0
    accel_weak_max: float = 8.0
    accel_good_max: float = 24.0
    accel_hard_max: float = 32.0
    jump_good_max: float = 50.0
    jump_hard_max: float = 120.0
    min_valid_points: int = 6
    min_score: float = 9.0
    nms_window: int = 22
    region_bonus: float = 1.0
    region_penalty: float = 4.0
    region_hard_filter: bool = False
    enable_v41: bool = False
    hit_guard_window: int = 8
    hit_guard_speedup_ratio: float = 1.6
    hit_guard_speedup_delta: float = 10.0
    hit_guard_min_speed_after: float = 18.0
    hit_guard_penalty: float = 3.0
    enable_relocation: bool = False
    relocate_back: int = 2
    relocate_forward: int = 12
    relocate_min_score: float = 7.0
    relocate_angle_weight: float = 0.035
    relocate_late_weight: float = 0.18
    relocate_speed_weight: float = 0.045
    enable_v42: bool = False
    low_speed_angle_min: float = 80.0
    low_speed_jump_max: float = 12.0
    low_speed_before_max: float = 8.0
    low_speed_after_max: float = 9.0
    low_speed_bonus: float = 2.0
    sharp_angle_min: float = 120.0
    sharp_accel_min: float = 32.0
    sharp_speed_ratio_max: float = 1.10
    sharp_bounce_bonus: float = 2.0
    late_refine_enabled: bool = False
    late_refine_forward: int = 12
    late_refine_min_score: float = 7.0
    late_refine_angle_min: float = 45.0
    late_refine_max_distance: float = 120.0
    late_refine_speed_mean_max: float = 22.0
    late_refine_angle_weight: float = 0.035
    late_refine_late_weight: float = 0.18
    late_refine_speed_weight: float = 0.045
    enable_v43: bool = False
    adaptive_low_speed_percentile: float = 35.0
    adaptive_jump_percentile: float = 75.0
    adaptive_hard_jump_percentile: float = 95.0
    adaptive_accel_high_percentile: float = 85.0
    adaptive_low_speed_bonus: float = 1.5
    adaptive_sharp_bonus: float = 1.0
    adaptive_jump_penalty: float = 2.0


@dataclass
class AdaptiveTrackStats:
    low_speed_mean: float = 8.0
    normal_jump: float = 12.0
    hard_jump: float = 50.0
    high_accel: float = 24.0


def parse_int(value, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def frame_id_from_name(name: str, fallback: int) -> int:
    stem = Path(str(name)).stem
    return parse_int(stem, fallback) or fallback


def load_track_csv(path: str) -> List[TrackPoint]:
    """Load either V3 inference CSV or TrackNet Label.csv as a trajectory."""
    points: List[TrackPoint] = []
    with open(path, newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        for idx, row in enumerate(reader):
            frame = (
                parse_int(row.get("frame_id"))
                if "frame_id" in row
                else parse_int(row.get("frame"))
            )
            if frame is None:
                frame = parse_int(row.get("frame_index"))
            if frame is None:
                frame = frame_id_from_name(
                    row.get("file name") or row.get("path1") or row.get("image") or idx,
                    idx,
                )

            x = parse_float(
                row.get("x")
                if "x" in row
                else row.get("x-coordinate")
                if "x-coordinate" in row
                else row.get("pred_x"),
                -1.0,
            )
            y = parse_float(
                row.get("y")
                if "y" in row
                else row.get("y-coordinate")
                if "y-coordinate" in row
                else row.get("pred_y"),
                -1.0,
            )
            conf = parse_float(
                row.get("confidence")
                if "confidence" in row
                else row.get("conf")
                if "conf" in row
                else row.get("score")
                if "score" in row
                else row.get("prob"),
                1.0,
            )
            status = parse_int(row.get("status"))
            visibility = parse_int(row.get("visibility") or row.get("Visibility Class"), 1)
            valid = visibility != 0 and x >= 0 and y >= 0 and conf > 0
            points.append(TrackPoint(frame, x, y, conf, valid, status))
    return sorted(points, key=lambda p: p.frame_id)


def load_label_csv(path: str) -> List[TrackPoint]:
    return load_track_csv(path)


def normalize_polygons(raw_polygons) -> List[List[Tuple[float, float]]]:
    polygons: List[List[Tuple[float, float]]] = []
    for polygon in raw_polygons or []:
        points = []
        for point in polygon:
            if len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
        if len(points) >= 3:
            polygons.append(points)
    return polygons


def load_playable_regions(path: str) -> List[List[Tuple[float, float]]]:
    if not path:
        return []
    region_path = Path(path)
    if not region_path.exists():
        return []
    with region_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return normalize_polygons(data.get("playable_ground") or data.get("regions", {}).get("playable_ground"))


def region_path_candidates(clip_name: str, region_root: str) -> List[Path]:
    if not region_root:
        return []
    parts = Path(clip_name.replace("\\", "/")).parts
    game = parts[0] if parts else ""
    clip = parts[1] if len(parts) > 1 else ""
    root = Path(region_root)
    candidates = []
    if game and clip:
        candidates.extend([root / f"{game}_{clip}.json", root / game / f"{clip}.json"])
    if game:
        candidates.append(root / f"{game}.json")
    return candidates


def load_playable_regions_for_clip(clip_name: str, region_root: str) -> List[List[Tuple[float, float]]]:
    for path in region_path_candidates(clip_name, region_root):
        regions = load_playable_regions(str(path))
        if regions:
            return regions
    return []


def point_in_polygon(x: float, y: float, polygon: Sequence[Tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) if yj != yi else 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def point_in_regions(
    x: float,
    y: float,
    regions: Optional[Sequence[Sequence[Tuple[float, float]]]],
) -> bool:
    return any(point_in_polygon(x, y, polygon) for polygon in regions or [])


def clean_points(points: Sequence[TrackPoint], params: BounceRuleParams) -> List[TrackPoint]:
    cleaned: List[TrackPoint] = []
    prev_valid: Optional[TrackPoint] = None
    for p in points:
        valid = p.valid and p.confidence >= params.min_conf
        if valid and prev_valid is not None:
            jump = math.hypot(p.x - prev_valid.x, p.y - prev_valid.y)
            if jump > params.jump_hard_max:
                valid = False
        cp = TrackPoint(p.frame_id, p.x, p.y, p.confidence, valid, p.status)
        cleaned.append(cp)
        if cp.valid:
            prev_valid = cp
    return cleaned


def interpolate_short_gaps(points: Sequence[TrackPoint], max_gap: int) -> List[TrackPoint]:
    out = [TrackPoint(p.frame_id, p.x, p.y, p.confidence, p.valid, p.status) for p in points]
    valid_idx = [i for i, p in enumerate(out) if p.valid]
    for left, right in zip(valid_idx, valid_idx[1:]):
        gap = right - left - 1
        if gap <= 0 or gap > max_gap:
            continue
        p0, p1 = out[left], out[right]
        frame_span = max(1, p1.frame_id - p0.frame_id)
        for i in range(left + 1, right):
            alpha = (out[i].frame_id - p0.frame_id) / frame_span
            out[i].x = p0.x * (1 - alpha) + p1.x * alpha
            out[i].y = p0.y * (1 - alpha) + p1.y * alpha
            out[i].confidence = min(p0.confidence, p1.confidence) * 0.75
            out[i].valid = True
    return out


def smooth_points(points: Sequence[TrackPoint], window: int) -> List[TrackPoint]:
    if window <= 1:
        return list(points)
    if window % 2 == 0:
        window += 1
    radius = window // 2
    out: List[TrackPoint] = []
    for i, p in enumerate(points):
        if not p.valid:
            out.append(TrackPoint(p.frame_id, p.x, p.y, p.confidence, p.valid, p.status))
            continue
        xs, ys, ws = [], [], []
        for j in range(max(0, i - radius), min(len(points), i + radius + 1)):
            q = points[j]
            if q.valid:
                xs.append(q.x)
                ys.append(q.y)
                ws.append(max(q.confidence, 1e-3))
        if not xs:
            out.append(TrackPoint(p.frame_id, p.x, p.y, p.confidence, p.valid, p.status))
            continue
        weights = np.asarray(ws, dtype=np.float64)
        out.append(
            TrackPoint(
                p.frame_id,
                float(np.average(np.asarray(xs), weights=weights)),
                float(np.average(np.asarray(ys), weights=weights)),
                p.confidence,
                p.valid,
                p.status,
            )
        )
    return out


def mean_velocity(points: Sequence[TrackPoint], start: int, end: int) -> Optional[np.ndarray]:
    vectors: List[np.ndarray] = []
    for i in range(start + 1, end + 1):
        if i <= 0 or i >= len(points):
            continue
        p0, p1 = points[i - 1], points[i]
        if p0.valid and p1.valid:
            frame_delta = max(1, p1.frame_id - p0.frame_id)
            vectors.append(np.array([(p1.x - p0.x) / frame_delta, (p1.y - p0.y) / frame_delta]))
    if not vectors:
        return None
    return np.mean(np.stack(vectors, axis=0), axis=0)


def vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_value = float(np.dot(v1, v2) / (n1 * n2))
    cos_value = max(-1.0, min(1.0, cos_value))
    return math.degrees(math.acos(cos_value))


def max_jump(points: Sequence[TrackPoint], start: int, end: int) -> float:
    jumps: List[float] = []
    for i in range(start + 1, end + 1):
        if i <= 0 or i >= len(points):
            continue
        p0, p1 = points[i - 1], points[i]
        if p0.valid and p1.valid:
            jumps.append(math.hypot(p1.x - p0.x, p1.y - p0.y))
    return max(jumps) if jumps else float("inf")


def valid_count(points: Sequence[TrackPoint], start: int, end: int) -> int:
    return sum(1 for i in range(max(0, start), min(len(points), end + 1)) if points[i].valid)


def score_candidate(
    point: TrackPoint,
    angle: float,
    accel_norm: float,
    speed_before: float,
    speed_after: float,
    jump_distance: float,
    valid_points: int,
    params: BounceRuleParams,
) -> float:
    score = 0.0
    hard_bounce_like = (
        params.enable_v42
        and jump_distance <= params.jump_good_max
        and valid_points >= params.min_valid_points
        and speed_after <= speed_before * params.sharp_speed_ratio_max
    )

    if angle < params.angle_weak_max:
        score -= 2.0
    elif params.angle_min <= angle <= params.angle_good_max:
        score += 3.0
    elif params.angle_good_max < angle <= params.angle_hard_max:
        score += 1.0
    elif angle > params.angle_hard_max:
        score += 0.5 if hard_bounce_like else -2.0

    if accel_norm < params.accel_weak_max:
        score -= 2.0
    elif params.accel_min <= accel_norm <= params.accel_good_max:
        score += 3.0
    elif params.accel_good_max < accel_norm <= params.accel_hard_max:
        score += 1.0
    elif accel_norm > params.accel_hard_max:
        score += 0.5 if hard_bounce_like else -2.0

    if params.min_speed <= speed_before <= params.speed_max:
        score += 1.0
    elif speed_before > params.speed_max:
        score -= 1.0

    if params.min_speed <= speed_after <= params.speed_max:
        score += 1.0
    elif speed_after > params.speed_max:
        score -= 1.0

    speed_min = max(min(speed_before, speed_after), 1e-6)
    speed_ratio = max(speed_before, speed_after) / speed_min
    if speed_ratio > params.speed_ratio_max:
        score -= 2.0

    after_before_ratio = speed_after / max(speed_before, 1e-6)
    speed_delta = speed_after - speed_before
    if after_before_ratio > params.hit_speedup_ratio and speed_delta > params.hit_speedup_delta:
        score -= params.hit_speedup_penalty

    if jump_distance <= params.jump_good_max:
        score += 2.0
    elif jump_distance > params.jump_hard_max:
        score -= 3.0

    if valid_points >= params.min_valid_points:
        score += 2.0
    else:
        score -= 2.0

    if point.confidence >= 0.50:
        score += 1.0

    if (
        params.enable_v42
        and angle >= params.low_speed_angle_min
        and jump_distance <= params.low_speed_jump_max
        and speed_before <= params.low_speed_before_max
        and speed_after <= params.low_speed_after_max
    ):
        score += params.low_speed_bonus

    if (
        params.enable_v42
        and hard_bounce_like
        and angle >= params.sharp_angle_min
        and accel_norm >= params.sharp_accel_min
    ):
        score += params.sharp_bounce_bonus

    return score


def percentile(values: Sequence[float], q: float, default: float) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return default
    return float(np.percentile(np.asarray(clean, dtype=np.float64), q))


def compute_adaptive_track_stats(
    features: Sequence[BounceCandidate],
    params: BounceRuleParams,
) -> AdaptiveTrackStats:
    speed_means = [0.5 * (feature.speed_before + feature.speed_after) for feature in features]
    jumps = [feature.jump_distance for feature in features]
    accels = [feature.accel_norm for feature in features]
    return AdaptiveTrackStats(
        low_speed_mean=percentile(speed_means, params.adaptive_low_speed_percentile, 8.0),
        normal_jump=percentile(jumps, params.adaptive_jump_percentile, 12.0),
        hard_jump=percentile(jumps, params.adaptive_hard_jump_percentile, params.jump_hard_max),
        high_accel=percentile(accels, params.adaptive_accel_high_percentile, params.accel_good_max),
    )


def adaptive_score_adjustment(
    feature: BounceCandidate,
    params: BounceRuleParams,
    stats: Optional[AdaptiveTrackStats],
) -> float:
    if not params.enable_v43 or stats is None:
        return 0.0

    score = 0.0
    speed_mean = 0.5 * (feature.speed_before + feature.speed_after)
    normal_jump = max(stats.normal_jump, 1e-6)
    hard_jump = max(stats.hard_jump, normal_jump)

    if (
        speed_mean <= stats.low_speed_mean
        and feature.angle_change >= params.low_speed_angle_min
        and feature.jump_distance <= normal_jump
    ):
        score += params.adaptive_low_speed_bonus

    if (
        feature.angle_change >= params.sharp_angle_min
        and feature.accel_norm >= stats.high_accel
        and feature.jump_distance <= normal_jump
        and feature.speed_after <= feature.speed_before * params.sharp_speed_ratio_max
    ):
        score += params.adaptive_sharp_bonus

    if feature.jump_distance > hard_jump:
        score -= params.adaptive_jump_penalty

    return score


def candidate_score_with_region(
    feature: BounceCandidate,
    params: BounceRuleParams,
    playable_regions: Optional[Sequence[Sequence[Tuple[float, float]]]] = None,
) -> Optional[float]:
    point = TrackPoint(feature.frame_id, feature.x, feature.y, feature.confidence, True)
    score = score_candidate(
        point,
        feature.angle_change,
        feature.accel_norm,
        feature.speed_before,
        feature.speed_after,
        feature.jump_distance,
        feature.valid_points,
        params,
    )
    if playable_regions:
        in_region = point_in_regions(feature.x, feature.y, playable_regions)
        if in_region:
            score += params.region_bonus
        elif params.region_hard_filter:
            return None
        else:
            score -= params.region_penalty
    return score


def recent_hit_like_penalty(
    feature: BounceCandidate,
    previous_features: Sequence[BounceCandidate],
    params: BounceRuleParams,
) -> float:
    if not params.enable_v41 or params.hit_guard_penalty <= 0:
        return 0.0

    for prev in reversed(previous_features):
        frame_gap = feature.frame_id - prev.frame_id
        if frame_gap <= 0:
            continue
        if frame_gap > params.hit_guard_window:
            break
        speed_ratio = prev.speed_after / max(prev.speed_before, 1e-6)
        speed_delta = prev.speed_after - prev.speed_before
        if (
            prev.speed_after >= params.hit_guard_min_speed_after
            and speed_ratio >= params.hit_guard_speedup_ratio
            and speed_delta >= params.hit_guard_speedup_delta
        ):
            return params.hit_guard_penalty
    return 0.0


def feature_to_candidate(feature: BounceCandidate, score: float) -> BounceCandidate:
    return BounceCandidate(
        feature.frame_id,
        feature.x,
        feature.y,
        score,
        feature.confidence,
        feature.angle_change,
        feature.accel_norm,
        feature.speed_before,
        feature.speed_after,
        feature.jump_distance,
        feature.valid_points,
    )


def relocation_score(
    feature: BounceCandidate,
    adjusted_score: float,
    original_frame: int,
    params: BounceRuleParams,
) -> float:
    late_offset = max(0, feature.frame_id - original_frame)
    speed_mean = 0.5 * (feature.speed_before + feature.speed_after)
    speed_bonus = max(0.0, params.speed_max - speed_mean) * params.relocate_speed_weight
    return (
        adjusted_score
        + feature.angle_change * params.relocate_angle_weight
        + late_offset * params.relocate_late_weight
        + speed_bonus
    )


def relocate_candidates(
    kept: Sequence[BounceCandidate],
    scored_features: Sequence[Tuple[BounceCandidate, float]],
    params: BounceRuleParams,
) -> List[BounceCandidate]:
    if not params.enable_v41 or not params.enable_relocation:
        return list(kept)

    relocated: List[BounceCandidate] = []
    for cand in kept:
        left = cand.frame_id - params.relocate_back
        right = cand.frame_id + params.relocate_forward
        best_feature: Optional[BounceCandidate] = None
        best_score = float("-inf")
        best_adjusted_score = cand.score

        for feature, adjusted_score in scored_features:
            if feature.frame_id < left or feature.frame_id > right:
                continue
            if adjusted_score < params.relocate_min_score:
                continue
            score = relocation_score(feature, adjusted_score, cand.frame_id, params)
            if score > best_score:
                best_score = score
                best_feature = feature
                best_adjusted_score = adjusted_score

        if best_feature is None:
            relocated.append(cand)
        else:
            relocated.append(feature_to_candidate(best_feature, best_adjusted_score))

    return nms_candidates(relocated, params.nms_window)


def late_refine_score(
    feature: BounceCandidate,
    adjusted_score: float,
    original: BounceCandidate,
    params: BounceRuleParams,
) -> float:
    late_offset = max(0, feature.frame_id - original.frame_id)
    speed_mean = 0.5 * (feature.speed_before + feature.speed_after)
    speed_bonus = max(0.0, params.late_refine_speed_mean_max - speed_mean) * params.late_refine_speed_weight
    return (
        adjusted_score
        + feature.angle_change * params.late_refine_angle_weight
        + late_offset * params.late_refine_late_weight
        + speed_bonus
    )


def conservative_late_refine_candidates(
    kept: Sequence[BounceCandidate],
    scored_features: Sequence[Tuple[BounceCandidate, float]],
    params: BounceRuleParams,
) -> List[BounceCandidate]:
    if not params.enable_v42 or not params.late_refine_enabled:
        return list(kept)

    refined: List[BounceCandidate] = []
    for cand in kept:
        best_feature: Optional[BounceCandidate] = None
        best_adjusted_score = cand.score
        best_score = late_refine_score(cand, cand.score, cand, params)

        for feature, adjusted_score in scored_features:
            frame_delta = feature.frame_id - cand.frame_id
            if frame_delta <= 0 or frame_delta > params.late_refine_forward:
                continue
            if adjusted_score < params.late_refine_min_score:
                continue
            if feature.angle_change < params.late_refine_angle_min:
                continue
            speed_mean = 0.5 * (feature.speed_before + feature.speed_after)
            if speed_mean > params.late_refine_speed_mean_max:
                continue
            distance = math.hypot(feature.x - cand.x, feature.y - cand.y)
            if distance > params.late_refine_max_distance:
                continue

            score = late_refine_score(feature, adjusted_score, cand, params)
            if score > best_score:
                best_score = score
                best_feature = feature
                best_adjusted_score = adjusted_score

        if best_feature is None:
            refined.append(cand)
        else:
            refined.append(feature_to_candidate(best_feature, best_adjusted_score))

    return nms_candidates(refined, params.nms_window)


def prepare_points(points: Sequence[TrackPoint], params: BounceRuleParams) -> List[TrackPoint]:
    prepared = clean_points(points, params)
    prepared = interpolate_short_gaps(prepared, params.max_gap)
    return smooth_points(prepared, params.smooth_window)


def extract_candidate_features(
    points: Sequence[TrackPoint],
    params: BounceRuleParams,
) -> List[BounceCandidate]:
    if len(points) < params.before_window + params.after_window + 2:
        return []

    prepared = prepare_points(points, params)

    features: List[BounceCandidate] = []
    left_margin = params.before_window + 1
    right_margin = params.after_window + 1

    for i in range(left_margin, len(prepared) - right_margin):
        p = prepared[i]
        if not p.valid:
            continue

        before = mean_velocity(prepared, i - params.before_window, i - 1)
        after = mean_velocity(prepared, i + 1, i + params.after_window)
        if before is None or after is None:
            continue

        speed_before = float(np.linalg.norm(before))
        speed_after = float(np.linalg.norm(after))
        if speed_before < params.min_speed and speed_after < params.min_speed:
            continue

        angle = vector_angle(before, after)
        accel_norm = float(np.linalg.norm(after - before))
        jump_distance = max_jump(prepared, i - params.before_window, i + params.after_window)
        valid_points = valid_count(prepared, i - params.before_window, i + params.after_window)
        features.append(
            BounceCandidate(
                p.frame_id,
                p.x,
                p.y,
                0.0,
                p.confidence,
                angle,
                accel_norm,
                speed_before,
                speed_after,
                jump_distance,
                valid_points,
            )
        )

    return features


def score_features(
    features: Sequence[BounceCandidate],
    params: BounceRuleParams,
    playable_regions: Optional[Sequence[Sequence[Tuple[float, float]]]] = None,
) -> List[BounceCandidate]:
    adaptive_stats = compute_adaptive_track_stats(features, params) if params.enable_v43 else None
    scored_features: List[Tuple[BounceCandidate, float]] = []
    previous_features: List[BounceCandidate] = []
    for feature in features:
        score = candidate_score_with_region(feature, params, playable_regions)
        if score is None:
            previous_features.append(feature)
            continue
        score += adaptive_score_adjustment(feature, params, adaptive_stats)
        score -= recent_hit_like_penalty(feature, previous_features, params)
        scored_features.append((feature, score))
        previous_features.append(feature)

    candidates: List[BounceCandidate] = []
    for feature, score in scored_features:
        if score >= params.min_score:
            candidates.append(feature_to_candidate(feature, score))

    kept = nms_candidates(candidates, params.nms_window)
    kept = relocate_candidates(kept, scored_features, params)
    return conservative_late_refine_candidates(kept, scored_features, params)


def detect_bounces(
    points: Sequence[TrackPoint],
    params: BounceRuleParams,
    playable_regions: Optional[Sequence[Sequence[Tuple[float, float]]]] = None,
) -> List[BounceCandidate]:
    features = extract_candidate_features(points, params)
    return score_features(features, params, playable_regions)



def nms_candidates(candidates: Sequence[BounceCandidate], window: int) -> List[BounceCandidate]:
    kept: List[BounceCandidate] = []
    for cand in sorted(candidates, key=lambda c: (-c.score, c.frame_id)):
        if any(abs(cand.frame_id - prev.frame_id) <= window for prev in kept):
            continue
        kept.append(cand)
    return sorted(kept, key=lambda c: c.frame_id)


def true_bounce_frames(points: Iterable[TrackPoint]) -> List[int]:
    return merge_event_frames(p.frame_id for p in points if p.status == 2)


def merge_event_frames(frames: Iterable[int], max_gap: int = 3) -> List[int]:
    groups: List[List[int]] = []
    current: List[int] = []
    for frame in sorted(frames):
        if not current or frame - current[-1] <= max_gap:
            current.append(frame)
        else:
            groups.append(current)
            current = [frame]
    if current:
        groups.append(current)
    return [int(round(sum(group) / len(group))) for group in groups]


def true_hit_frames(points: Iterable[TrackPoint]) -> List[int]:
    return sorted(p.frame_id for p in points if p.status == 1)


def match_events(
    predicted_frames: Sequence[int],
    true_frames: Sequence[int],
    tolerance: int,
) -> Tuple[int, int, int, List[int]]:
    used_true = set()
    errors: List[int] = []
    tp = 0
    for pred in sorted(predicted_frames):
        best_idx = None
        best_err = tolerance + 1
        for idx, true in enumerate(true_frames):
            if idx in used_true:
                continue
            err = abs(pred - true)
            if err <= tolerance and err < best_err:
                best_idx = idx
                best_err = err
        if best_idx is not None:
            used_true.add(best_idx)
            tp += 1
            errors.append(best_err)
    fp = len(predicted_frames) - tp
    fn = len(true_frames) - tp
    return tp, fp, fn, errors


def count_near_events(predicted_frames: Sequence[int], event_frames: Sequence[int], tolerance: int) -> int:
    return sum(any(abs(pred - event) <= tolerance for event in event_frames) for pred in predicted_frames)


def metrics_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
