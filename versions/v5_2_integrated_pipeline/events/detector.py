import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from court import point_in_polygons
from player import PlayerFrameContext, empty_player_context, nearest_player_distance, player_region_scores


Point = Tuple[float, float]


@dataclass
class Event:
    frame: int
    event_type: str
    x: float
    y: float
    score: float
    near_player: float
    accel: float
    angle_change: float
    source: str
    track_support: float = 1.0
    observed_count: int = 5
    court_evidence: float = 0.0
    court_agreement: float = 0.0
    frame_start: int = -1
    frame_end: int = -1
    timing_confidence: float = 0.0
    original_frame: int = -1
    frame_offset: int = 0
    timing_method: str = "legacy_refinement"


@dataclass
class FrameFeatures:
    frame: int
    point: Point
    score: float
    near_player: float
    player_dist: float
    upper_player: float
    lower_player: float
    body_inside: float
    accel: float
    angle: float
    speed_change: float
    step_before: float
    step_after: float
    y_velocity_before: float
    y_velocity_after: float
    y_accel: float
    y_turn: bool
    court_accel: float
    court_angle: float
    court_y_velocity_before: float
    court_y_velocity_after: float
    court_y_accel: float
    court_y_turn: bool
    court_step_before: float
    court_step_after: float
    court_speed_change: float
    in_court: bool
    player_local_min: bool
    track_support: float
    observed_count: int
    interpolated_count: int
    center_observed: bool
    robust_accel: float
    robust_angle: float
    robust_y_accel: float
    robust_y_turn: bool
    extended_player: float
    court_available: bool
    court_inside: bool
    court_far_outside: bool
    court_quality: float
    player_height: float
    player_width: float
    serve_player_alignment: float
    player_quality: float


def seconds_to_frames(seconds: float, fps: float, minimum: int = 1) -> int:
    """Convert a time window to frames while keeping a useful lower bound."""
    return max(int(minimum), int(round(max(0.0, float(seconds)) * max(float(fps), 1.0))))


def _source_reliability(source: str, accepted: int) -> float:
    source = (source or "").lower()
    if accepted <= 0 or any(token in source for token in ("missing", "removed", "rejected", "expired", "pending")):
        return 0.0
    if accepted == 2 or "interpol" in source:
        return 0.30
    if "weak_motion_recovered" in source:
        return 0.72
    if "seed_confirmed" in source:
        return 0.90
    if "high_" in source or "observed" in source or source in ("accepted", "track_csv"):
        return 1.0
    return 0.85


def _dist(a: Point, b: Point) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _serve_player_alignment(point: Point, ctx: PlayerFrameContext) -> float:
    """Score whether a ball is in the vertical toss column above a player."""
    best = 0.0
    x, y = point
    for x1, y1, x2, y2 in ctx.boxes:
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        center_x = 0.5 * (x1 + x2)
        horizontal = abs(x - center_x) / max(width, 1.0)
        above = max(0.0, y1 - y) / height
        if horizontal <= 1.0 and y <= y2 and above <= 3.0:
            best = max(best, (1.0 - horizontal) * (1.0 - above / 3.0))
    return best


def _angle_change(p0: Point, p1: Point, p2: Point) -> float:
    v1 = (p1[0] - p0[0], p1[1] - p0[1])
    v2 = (p2[0] - p1[0], p2[1] - p1[1])
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 <= 1e-6 or n2 <= 1e-6:
        return 0.0
    cos_v = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return float(math.degrees(math.acos(cos_v)))


def _valid_neighbors(tracks: Sequence[Optional[Point]], idx: int, sample_step: int = 1):
    sample_step = max(1, int(sample_step))
    if idx - 2 * sample_step < 0 or idx + 2 * sample_step >= len(tracks):
        return None
    points = [
        tracks[idx - 2 * sample_step],
        tracks[idx - sample_step],
        tracks[idx],
        tracks[idx + sample_step],
        tracks[idx + 2 * sample_step],
    ]
    if any(p is None for p in points):
        return None
    return points  # type: ignore[return-value]


def _merge_events(events: List[Event], min_gap: int) -> List[Event]:
    kept: List[Event] = []
    for event in sorted(events, key=lambda item: item.score, reverse=True):
        if all(abs(event.frame - prev.frame) > min_gap or event.event_type != prev.event_type for prev in kept):
            kept.append(event)
    return sorted(kept, key=lambda item: item.frame)


def _sequence_events(events: List[Event], min_gap: int) -> List[Event]:
    merged = _merge_events(events, min_gap)
    sequenced: List[Event] = []
    last_type = None
    last_frame = -10**9
    for event in sorted(merged, key=lambda item: item.frame):
        if sequenced and event.event_type == last_type and event.frame - last_frame < min_gap * 2:
            if event.score > sequenced[-1].score:
                sequenced[-1] = event
                last_frame = event.frame
            continue
        sequenced.append(event)
        last_type = event.event_type
        last_frame = event.frame
    return sequenced


def _suppress_nearby_opposite_events(events: List[Event], min_gap: int) -> List[Event]:
    kept: List[Event] = []
    for event in sorted(events, key=lambda item: item.score, reverse=True):
        too_close = False
        for prev in kept:
            if abs(event.frame - prev.frame) <= min_gap:
                too_close = True
                break
        if not too_close:
            kept.append(event)
    return sorted(kept, key=lambda item: item.frame)


def build_frame_features(
    tracks: Sequence[Optional[Point]],
    scores: Sequence[Optional[float]],
    player_contexts: Dict[int, PlayerFrameContext],
    court_polygons,
    fps: float,
    court_tracks: Optional[Sequence[Optional[Point]]] = None,
    track_sources: Optional[Sequence[str]] = None,
    accepted_flags: Optional[Sequence[int]] = None,
    feature_sample_seconds: float = 1.0 / 30.0,
    court_qualities: Optional[Sequence[float]] = None,
) -> List[FrameFeatures]:
    features: List[FrameFeatures] = []
    sample_step = seconds_to_frames(feature_sample_seconds, fps)
    fps_scale = max(float(fps), 1.0) / (30.0 * sample_step)
    court_tracks = court_tracks or [None] * len(tracks)
    track_sources = track_sources or ["track_csv"] * len(tracks)
    accepted_flags = accepted_flags or [1 if point is not None else 0 for point in tracks]
    court_qualities = court_qualities or [1.0 if point is not None else 0.0 for point in court_tracks]

    for idx in range(2 * sample_step, len(tracks) - 2 * sample_step):
        neighbors = _valid_neighbors(tracks, idx, sample_step)
        if neighbors is None:
            continue
        p0, p_prev, p, p_next, p4 = neighbors
        window_indices = [
            idx - 2 * sample_step,
            idx - sample_step,
            idx,
            idx + sample_step,
            idx + 2 * sample_step,
        ]
        reliabilities = [
            _source_reliability(track_sources[window_idx], accepted_flags[window_idx])
            for window_idx in window_indices
        ]
        observed_count = sum(value >= 0.70 for value in reliabilities)
        interpolated_count = sum(
            accepted_flags[window_idx] == 2 or "interpol" in (track_sources[window_idx] or "").lower()
            for window_idx in window_indices
        )
        center_reliability = reliabilities[2]
        center_observed = center_reliability >= 0.70 and accepted_flags[idx] == 1
        track_support = sum(reliabilities) / len(reliabilities)
        conf = scores[idx] if scores[idx] is not None else 0.0
        if conf < 0.10:
            continue

        speed_before = _dist(p0, p_prev)
        speed_after = _dist(p_next, p4)
        speed_mid_1 = _dist(p_prev, p)
        speed_mid_2 = _dist(p, p_next)
        accel = abs(speed_mid_2 - speed_mid_1) * fps_scale
        angle = _angle_change(p_prev, p, p_next)
        speed_change = abs(speed_after - speed_before)
        y_velocity_before = (p[1] - p_prev[1]) * fps_scale
        y_velocity_after = (p_next[1] - p[1]) * fps_scale
        y_accel = abs(y_velocity_after - y_velocity_before)
        robust_step_before = _dist(p0, p) / 2.0
        robust_step_after = _dist(p, p4) / 2.0
        robust_accel = abs(robust_step_after - robust_step_before) * fps_scale
        robust_angle = _angle_change(p0, p, p4)
        robust_y_velocity_before = ((p[1] - p0[1]) / 2.0) * fps_scale
        robust_y_velocity_after = ((p4[1] - p[1]) / 2.0) * fps_scale
        robust_y_accel = abs(robust_y_velocity_after - robust_y_velocity_before)
        robust_y_turn = robust_y_velocity_before * robust_y_velocity_after <= 0

        ctx = player_contexts.get(idx, empty_player_context())
        player_dist, near_player = nearest_player_distance(p, ctx)
        extended_radius = max(ctx.hit_radius * 1.8, 1.0)
        extended_player = max(0.0, 1.0 - player_dist / extended_radius)
        upper_player, lower_player, body_inside = player_region_scores(p, ctx)
        player_heights = [max(1.0, box[3] - box[1]) for box in ctx.boxes]
        player_widths = [max(1.0, box[2] - box[0]) for box in ctx.boxes]
        player_height = float(sorted(player_heights)[len(player_heights) // 2]) if player_heights else 0.0
        player_width = float(sorted(player_widths)[len(player_widths) // 2]) if player_widths else 0.0
        serve_player_alignment = _serve_player_alignment(p, ctx)

        player_dist_prev = nearest_player_distance(p_prev, player_contexts.get(idx - sample_step, ctx))[0]
        player_dist_curr = nearest_player_distance(p, ctx)[0]
        player_dist_next = nearest_player_distance(p_next, player_contexts.get(idx + sample_step, ctx))[0]
        player_local_min = player_dist_curr <= player_dist_prev and player_dist_curr <= player_dist_next

        cp_neighbors = _valid_neighbors(court_tracks, idx, sample_step)
        court_point = court_tracks[idx]
        court_available = court_point is not None
        if court_point is None:
            court_inside = True
            court_far_outside = False
        else:
            court_x, court_y = court_point
            court_inside = -200.0 <= court_x <= 1200.0 and -200.0 <= court_y <= 2368.0
            court_far_outside = (
                court_x < -260.0 or court_x > 1260.0 or court_y < -260.0 or court_y > 2428.0
            )
        if cp_neighbors is None:
            court_accel = 0.0
            court_angle = 0.0
            court_y_velocity_before = 0.0
            court_y_velocity_after = 0.0
            court_y_accel = 0.0
            court_y_turn = False
            court_step_before = 0.0
            court_step_after = 0.0
            court_speed_change = 0.0
        else:
            c0, c_prev, c, c_next, c4 = cp_neighbors
            court_step_before = _dist(c_prev, c)
            court_step_after = _dist(c, c_next)
            court_accel = abs(court_step_after - court_step_before) * fps_scale
            court_angle = _angle_change(c_prev, c, c_next)
            court_y_velocity_before = (c[1] - c_prev[1]) * fps_scale
            court_y_velocity_after = (c_next[1] - c[1]) * fps_scale
            court_y_accel = abs(court_y_velocity_after - court_y_velocity_before)
            court_y_turn = (c[1] - c_prev[1]) * (c_next[1] - c[1]) <= 0
            court_speed_before = _dist(c0, c_prev)
            court_speed_after = _dist(c_next, c4)
            court_speed_change = abs(court_speed_after - court_speed_before) * fps_scale

        features.append(
            FrameFeatures(
                frame=idx,
                point=p,
                score=conf,
                near_player=near_player,
                player_dist=player_dist,
                upper_player=upper_player,
                lower_player=lower_player,
                body_inside=body_inside,
                accel=accel,
                angle=angle,
                speed_change=speed_change,
                step_before=speed_mid_1,
                step_after=speed_mid_2,
                y_velocity_before=y_velocity_before,
                y_velocity_after=y_velocity_after,
                y_accel=y_accel,
                y_turn=(p[1] - p_prev[1]) * (p_next[1] - p[1]) <= 0,
                court_accel=court_accel,
                court_angle=court_angle,
                court_y_velocity_before=court_y_velocity_before,
                court_y_velocity_after=court_y_velocity_after,
                court_y_accel=court_y_accel,
                court_y_turn=court_y_turn,
                court_step_before=court_step_before,
                court_step_after=court_step_after,
                court_speed_change=court_speed_change,
                in_court=point_in_polygons(p, court_polygons),
                player_local_min=player_local_min,
                track_support=track_support,
                observed_count=observed_count,
                interpolated_count=interpolated_count,
                center_observed=center_observed,
                robust_accel=robust_accel,
                robust_angle=robust_angle,
                robust_y_accel=robust_y_accel,
                robust_y_turn=robust_y_turn,
                extended_player=extended_player,
                court_available=court_available,
                court_inside=court_inside,
                court_far_outside=court_far_outside,
                court_quality=max(0.0, min(float(court_qualities[idx]), 1.0)),
                player_height=player_height,
                player_width=player_width,
                serve_player_alignment=serve_player_alignment,
                player_quality=max(0.0, min(ctx.quality, 1.0)),
            )
        )
    return features


def detect_hit_events_state_machine(
    features: Sequence[FrameFeatures],
    hit_threshold: float,
    min_event_gap: int,
) -> List[Event]:
    events: List[Event] = []
    active = False
    best_item: Optional[FrameFeatures] = None
    best_score = -1.0
    last_event_frame = -10**9

    for item in features:
        event_accel = max(item.accel, min(item.court_accel, 180.0))
        event_angle = max(item.angle, min(item.court_angle, 150.0))
        proximity_score = max(item.near_player, item.upper_player)
        in_hit_zone = proximity_score > 0.18
        candidate_score = (
            0.35 * item.near_player
            + 0.28 * item.upper_player
            + 0.16 * min(event_angle / 120.0, 1.0)
            + 0.14 * min(event_accel / 55.0, 1.0)
            + 0.07 * (1.0 if item.player_local_min else 0.0)
            - 0.12 * item.lower_player
        )

        if in_hit_zone and not active:
            active = True
            best_item = item
            best_score = candidate_score
            continue

        if active and in_hit_zone:
            if item.player_dist < (best_item.player_dist if best_item is not None else float("inf")) or candidate_score > best_score:
                best_item = item
                best_score = candidate_score
            continue

        if active and not in_hit_zone:
            if best_item is not None and best_score >= hit_threshold and best_item.frame - last_event_frame >= min_event_gap:
                event_accel = max(best_item.accel, best_item.court_accel)
                event_angle = max(best_item.angle, best_item.court_angle)
                events.append(
                    Event(
                        frame=best_item.frame,
                        event_type="hit",
                        x=best_item.point[0],
                        y=best_item.point[1],
                        score=best_score,
                        near_player=best_item.near_player,
                        accel=event_accel,
                        angle_change=event_angle,
                        source="hit_state_machine",
                    )
                )
                last_event_frame = best_item.frame
            active = False
            best_item = None
            best_score = -1.0

    return events


def _is_in_frame(item: FrameFeatures, frame_width: Optional[int], frame_height: Optional[int], frame_margin: float) -> bool:
    if frame_width is None or frame_height is None:
        return True
    x, y = item.point
    return not (x < -frame_margin or y < -frame_margin or x > frame_width + frame_margin or y > frame_height + frame_margin)


def _event_motion(item: FrameFeatures) -> Tuple[float, float, bool, float]:
    """Return accel, angle, y-turn, y-accel with mini-court used only when sane."""
    image_accel = max(item.accel, item.y_accel, item.robust_accel, item.robust_y_accel)
    image_angle = max(item.angle, item.robust_angle)
    image_y_turn = item.robust_y_turn or (item.y_turn and item.track_support >= 0.90)
    image_y_accel = max(item.y_accel, item.robust_y_accel)

    court_is_sane = 0.0 < item.court_accel < 500.0 and 0.0 <= item.court_angle <= 170.0
    if not court_is_sane:
        return image_accel, image_angle, image_y_turn, image_y_accel

    accel = max(image_accel, min(max(item.court_accel, item.court_y_accel), 180.0))
    angle = max(image_angle, min(item.court_angle, 150.0))
    y_turn = image_y_turn or item.court_y_turn
    y_accel = max(image_y_accel, min(item.court_y_accel, 180.0))
    return accel, angle, y_turn, y_accel


def _court_bounce_evidence(item: FrameFeatures) -> Tuple[float, float, float]:
    """Return mini-court evidence, image/court agreement and mapping-jitter risk."""
    if not item.court_available or item.court_far_outside or item.court_quality < 0.15:
        return 0.0, 0.0, 0.0

    court_motion_sane = (
        0.0 <= item.court_angle <= 175.0
        and item.court_accel < 260.0
        and item.court_y_accel < 320.0
        and max(item.court_step_before, item.court_step_after) < 420.0
    )
    if not court_motion_sane:
        return 0.0, 0.0, 1.0

    court_turn = max(
        1.0 if item.court_y_turn else 0.0,
        min(item.court_angle / 120.0, 1.0),
    )
    court_accel = min(max(item.court_accel, item.court_y_accel) / 100.0, 1.0)
    court_speed_change = min(item.court_speed_change / 100.0, 1.0)
    raw_court_evidence = 0.48 * court_turn + 0.34 * court_accel + 0.18 * court_speed_change
    court_evidence = raw_court_evidence * item.court_quality

    image_turn = max(
        1.0 if (item.y_turn or item.robust_y_turn) else 0.0,
        min(max(item.angle, item.robust_angle) / 120.0, 1.0),
    )
    image_accel = min(max(item.accel, item.y_accel, item.robust_accel, item.robust_y_accel) / 55.0, 1.0)
    image_evidence = 0.58 * image_turn + 0.42 * image_accel
    agreement = min(court_evidence, image_evidence)

    court_only_spike = max(0.0, court_evidence - image_evidence - 0.25)
    boundary_like_jump = (
        item.court_accel >= 110.0
        and item.court_angle < 18.0
        and image_evidence < 0.35
    )
    jitter_risk = min(court_only_spike + (0.35 if boundary_like_jump else 0.0), 1.0)
    return court_evidence, agreement, jitter_risk


def _reference_bounce_score(item: FrameFeatures, flight_frames: int, min_flight_frames: int) -> Tuple[float, float, float]:
    event_accel, event_angle, y_turn, y_accel = _event_motion(item)
    near_contact = max(item.near_player, item.upper_player)
    flight_score = min(flight_frames / max(1.0, float(min_flight_frames)), 1.0)
    court_reliability = item.court_quality if item.court_available else 0.0
    player_reliability = max(0.20, item.player_quality)
    track_reliability = max(0.25, item.track_support)
    court_bonus = (0.10 if item.in_court else -0.35) * max(0.35, court_reliability)
    y_turn_score = 1.0 if y_turn else 0.0
    support_bonus = 0.14 * track_reliability
    interpolation_penalty = 0.08 * item.interpolated_count * (1.25 - 0.25 * track_reliability)
    court_evidence, court_agreement, court_jitter = _court_bounce_evidence(item)
    score = (
        0.34 * min(y_accel / 55.0, 1.0)
        + 0.22 * y_turn_score
        + 0.18 * min(event_angle / 120.0, 1.0)
        + 0.12 * min(item.speed_change / 45.0, 1.0)
        + 0.10 * flight_score
        + support_bonus
        + court_bonus
        - interpolation_penalty
        + (0.08 + 0.10 * court_reliability) * court_evidence
        + (0.04 + 0.08 * court_reliability) * court_agreement
        - 0.16 * court_jitter
        - 0.36 * near_contact * player_reliability
        - 0.18 * item.lower_player * player_reliability
        - 0.12 * item.body_inside * player_reliability
    )
    return score, event_accel, event_angle


def detect_reference_bounce_events(
    features: Sequence[FrameFeatures],
    bounce_threshold: float,
    min_event_gap: int,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
    frame_margin: float = 4.0,
    contact_threshold: float = 0.18,
    min_flight_frames: int = 6,
    max_bounce_step: float = 260.0,
    min_bounce_y_accel: float = 8.0,
) -> List[Event]:
    """Reference-style bounce logic.

    Ball contact/near-hand regions only reset the flying state. Bounce is allowed
    after the ball has been away from players for several valid frames.
    """
    events: List[Event] = []
    flight_frames = 0
    last_event_frame = -10**9
    best_candidate: Optional[Event] = None

    for item in sorted(features, key=lambda row: row.frame):
        if not _is_in_frame(item, frame_width, frame_height, frame_margin):
            flight_frames = 0
            best_candidate = None
            continue

        near_contact = max(item.near_player, item.upper_player)
        if near_contact >= contact_threshold:
            flight_frames = 0
            best_candidate = None
            continue

        flight_frames += 1
        if item.frame - last_event_frame <= min_event_gap:
            continue
        if not item.center_observed or item.observed_count < 3:
            continue
        if flight_frames < min_flight_frames:
            continue
        if max(item.step_before, item.step_after) > max_bounce_step:
            continue

        score, event_accel, event_angle = _reference_bounce_score(item, flight_frames, min_flight_frames)
        _, _, y_turn, y_accel = _event_motion(item)
        if y_accel < min_bounce_y_accel and not y_turn:
            continue
        if score < bounce_threshold:
            continue

        candidate = Event(
            frame=item.frame,
            event_type="bounce",
            x=item.point[0],
            y=item.point[1],
            score=score,
            near_player=item.near_player,
            accel=event_accel,
            angle_change=event_angle,
            source="reference_flying_y_accel",
            track_support=item.track_support,
            observed_count=item.observed_count,
        )

        if best_candidate is None:
            best_candidate = candidate
            continue
        if candidate.frame - best_candidate.frame <= max(2, min_event_gap // 2):
            if candidate.score > best_candidate.score:
                best_candidate = candidate
            continue

        events.append(best_candidate)
        last_event_frame = best_candidate.frame
        best_candidate = candidate

    if best_candidate is not None and best_candidate.frame - last_event_frame > min_event_gap:
        events.append(best_candidate)

    return _merge_events(events, min_event_gap)


def _v37_contact_score(item: FrameFeatures) -> float:
    # A mediocre pose estimate may reduce scoring penalties, but a visible ball
    # near the detected hand must still terminate the flight segment.
    reliability = max(0.60, item.player_quality)
    score = max(item.near_player, item.upper_player, item.extended_player) * reliability
    return score


def _serve_toss_frames(
    features: Sequence[FrameFeatures],
    fps: float,
    contact_threshold: float,
    max_seconds: float,
    min_rise_seconds: float,
    min_rise_player_ratio: float,
    max_lateral_player_ratio: float,
) -> set:
    """Find compact upward releases near a player that look like a serve toss.

    A toss starts around a hand/upper-body contact, rises mostly vertically and
    stays close to the serving player. Frames in that upward segment are not
    valid court-bounce candidates. Once the ball travels laterally, normal
    flight-event detection resumes.
    """
    ordered = sorted(features, key=lambda row: row.frame)
    if not ordered:
        return set()

    max_frames = seconds_to_frames(max_seconds, fps)
    min_rise_frames = seconds_to_frames(min_rise_seconds, fps)
    start_grace_frames = seconds_to_frames(0.35, fps)
    track_reset_frames = seconds_to_frames(0.50, fps)
    first_feature_frame = ordered[0].frame
    release_threshold = max(0.08, contact_threshold * 0.60)
    suppressed = set()

    for release_idx, release in enumerate(ordered):
        release_contact = max(
            release.near_player,
            release.upper_player,
            release.body_inside,
            release.extended_player,
        )
        starts_mid_toss = (
            release.frame - first_feature_frame <= start_grace_frames
            and release.serve_player_alignment >= 0.18
        )
        previous_frame = ordered[release_idx - 1].frame if release_idx > 0 else None
        starts_after_track_reset = (
            previous_frame is None or release.frame - previous_frame > track_reset_frames
        )
        if not starts_mid_toss and not starts_after_track_reset:
            continue
        if release_contact < release_threshold and not starts_mid_toss:
            continue

        player_height = max(release.player_height, 1.0)
        player_width = max(release.player_width, player_height * 0.35, 1.0)
        min_rise = max(12.0, min_rise_player_ratio * player_height)
        max_lateral = max(18.0, max_lateral_player_ratio * player_width)
        previous = release
        upward_steps = 0
        valid_steps = 0
        toss_end_frame = None

        for current in ordered[release_idx + 1 :]:
            elapsed_frames = current.frame - release.frame
            if elapsed_frames > max_frames:
                break
            if current.frame - previous.frame > max(2, seconds_to_frames(0.10, fps)):
                break

            dy = current.point[1] - previous.point[1]
            if abs(dy) >= 0.5:
                valid_steps += 1
                if dy < 0.0:
                    upward_steps += 1

            rise = release.point[1] - current.point[1]
            lateral = abs(current.point[0] - release.point[0])
            upward_fraction = upward_steps / max(valid_steps, 1)
            if (
                elapsed_frames >= min_rise_frames
                and rise >= min_rise
                and lateral <= max(max_lateral, 0.85 * rise)
                and upward_fraction >= 0.60
            ):
                toss_end_frame = current.frame
            previous = current

        if toss_end_frame is not None:
            extended_end = toss_end_frame
            for current in ordered[release_idx + 1 :]:
                elapsed_frames = current.frame - release.frame
                if elapsed_frames > max_frames:
                    break
                rise = release.point[1] - current.point[1]
                lateral = abs(current.point[0] - release.point[0])
                if rise < 0.5 * min_rise:
                    break
                if lateral > max(1.5 * max_lateral, 1.20 * rise):
                    break
                extended_end = current.frame
            # A serve toss ends at the next hand/upper-body contact. Extending
            # through that contact prevents the racket strike from becoming a
            # bounce while allowing the post-serve flight immediately after it.
            for current in ordered[release_idx + 1 :]:
                if current.frame - release.frame > max_frames:
                    break
                current_contact = max(
                    current.near_player,
                    current.upper_player,
                    current.body_inside,
                    current.extended_player,
                )
                if current.frame > toss_end_frame and current_contact >= release_threshold:
                    extended_end = max(extended_end, current.frame)
                    break
            suppressed.update(range(release.frame, extended_end + 1))

    return suppressed


def _v37_bounce_candidate(
    item: FrameFeatures,
    flight_frames: int,
    min_flight_frames: int,
    require_court: bool,
):
    if not item.center_observed or item.observed_count < 3:
        return None
    if require_court and not item.court_available:
        return None
    if item.court_available and not item.court_inside:
        return None

    score, event_accel, event_angle = _reference_bounce_score(item, flight_frames, min_flight_frames)
    _, _, y_turn, y_accel = _event_motion(item)
    decel_score = min(item.speed_change / 35.0, 1.0)
    slow_contact_score = min(12.0 / max(min(item.step_before, item.step_after), 1.0), 1.0)
    score += 0.10 * decel_score + 0.08 * slow_contact_score

    # Reject a one-frame stall/kink when the wider temporal window keeps the
    # same direction. These spikes commonly come from a brief localization
    # wobble after a racket hit and otherwise look like a high-angle bounce.
    step_ratio = min(item.step_before, item.step_after) / max(
        item.step_before, item.step_after, 1.0
    )
    single_frame_kink = (
        item.angle >= 110.0
        and item.robust_angle < 25.0
        and not item.robust_y_turn
        and step_ratio < 0.30
    )
    if single_frame_kink:
        return None

    if not y_turn and y_accel < 6.0 and event_angle < 25.0:
        score -= 0.12
    return score, event_accel, event_angle


def _select_v37_segment_bounce(
    candidates: Sequence[Tuple[FrameFeatures, float, float, float]],
    threshold: float,
) -> Optional[Event]:
    eligible = [candidate for candidate in candidates if candidate[1] >= threshold]
    if not eligible:
        return None
    item, score, event_accel, event_angle = max(eligible, key=lambda candidate: candidate[1])
    court_evidence, court_agreement, _ = _court_bounce_evidence(item)
    source = (
        "v37_dual_space_flight_segment"
        if court_evidence >= 0.45 and court_agreement >= 0.25
        else "v37_flight_segment"
    )
    return Event(
        frame=item.frame,
        event_type="bounce",
        x=item.point[0],
        y=item.point[1],
        score=score,
        near_player=item.near_player,
        accel=event_accel,
        angle_change=event_angle,
        source=source,
        track_support=item.track_support,
        observed_count=item.observed_count,
        court_evidence=court_evidence,
        court_agreement=court_agreement,
        frame_start=item.frame,
        frame_end=item.frame,
    )


def _select_v37_segment_bounces(
    candidates: Sequence[Tuple[FrameFeatures, float, float, float]],
    threshold: float,
    min_event_gap: int,
) -> List[Event]:
    """Select local event peaks before applying rally-order constraints."""
    eligible = sorted(
        (candidate for candidate in candidates if candidate[1] >= threshold),
        key=lambda candidate: candidate[0].frame,
    )
    if not eligible:
        return []

    peak_radius = max(2, min_event_gap // 3)
    peaks = []
    for candidate in eligible:
        frame = candidate[0].frame
        score = candidate[1]
        local_scores = [
            other[1]
            for other in eligible
            if abs(other[0].frame - frame) <= peak_radius
        ]
        if score >= max(local_scores):
            peaks.append(candidate)

    # Collapse equal-score plateaus while preserving chronological order.
    selected_peaks = []
    for candidate in peaks:
        if selected_peaks and candidate[0].frame - selected_peaks[-1][0].frame <= peak_radius:
            if candidate[1] > selected_peaks[-1][1]:
                selected_peaks[-1] = candidate
            continue
        selected_peaks.append(candidate)

    events: List[Event] = []
    for peak in selected_peaks:
        event = _select_v37_segment_bounce([peak], threshold)
        if event is None:
            continue
        if events and event.frame - events[-1].frame < min_event_gap:
            # Peaks inside the physical minimum gap describe the same contact
            # cluster. Keep the strongest peak instead of locking onto an
            # earlier smooth trajectory apex merely because it is in court.
            if event.score > events[-1].score:
                events[-1] = event
            continue
        events.append(event)
    return events


def detect_v37_bounce_events(
    features: Sequence[FrameFeatures],
    bounce_threshold: float,
    min_event_gap: int,
    contact_threshold: float,
    min_flight_frames: int,
    fps: float = 30.0,
    feature_gap_frames: int = 2,
    suppress_serve_toss: bool = True,
    serve_toss_max_seconds: float = 1.20,
    serve_toss_min_rise_seconds: float = 0.10,
    serve_toss_min_rise_player_ratio: float = 0.22,
    serve_toss_max_lateral_player_ratio: float = 0.85,
    refine_backward_seconds: float = 0.12,
    refine_forward_seconds: float = 6.0 / 30.0,
    refine_fast_forward_seconds: float = 0.50,
    refine_merge_seconds: float = 2.0 / 30.0,
) -> List[Event]:
    """Detect at most one bounce in each flight segment between player contacts.

    The contact boundary combines MediaPipe proximity and mini-court coordinates.
    This prevents the much stronger direction change at a racket hit from replacing
    the preceding, often subtler, court bounce.
    """
    events: List[Event] = []
    candidates: List[Tuple[FrameFeatures, float, float, float]] = []
    flight_frames = 0
    state = "waiting"
    last_frame: Optional[int] = None
    require_court = any(item.court_available for item in features)
    toss_frames = (
        _serve_toss_frames(
            features,
            fps,
            contact_threshold,
            serve_toss_max_seconds,
            serve_toss_min_rise_seconds,
            serve_toss_min_rise_player_ratio,
            serve_toss_max_lateral_player_ratio,
        )
        if suppress_serve_toss
        else set()
    )

    def finish_segment():
        nonlocal candidates
        for event in _select_v37_segment_bounces(candidates, bounce_threshold, min_event_gap):
            if not events or event.frame - events[-1].frame >= min_event_gap:
                events.append(event)
            elif (
                events[-1].court_evidence < 0.25
                or events[-1].track_support < 0.70
            ) and event.score > events[-1].score:
                events[-1] = event
        candidates = []

    for item in sorted(features, key=lambda row: row.frame):
        if last_frame is not None and item.frame - last_frame > feature_gap_frames:
            finish_segment()
            flight_frames = 0
            state = "waiting"
        last_frame = item.frame

        if item.frame in toss_frames:
            if state == "flying":
                finish_segment()
            state = "serve_toss"
            flight_frames = 0
            continue

        contact = _v37_contact_score(item) >= contact_threshold
        if contact:
            if state == "flying":
                finish_segment()
            state = "serve_contact" if state == "serve_toss" else "approaching_player"
            flight_frames = 0
            continue

        if state in ("waiting", "serve_toss", "serve_contact", "approaching_player"):
            state = "flying"
            flight_frames = 0

        flight_frames += 1
        if flight_frames < min_flight_frames:
            continue
        candidate = _v37_bounce_candidate(item, flight_frames, min_flight_frames, require_court)
        if candidate is not None:
            candidates.append((item, *candidate))

    finish_segment()
    return refine_v37_event_frames(
        _merge_events(events, min_event_gap),
        features,
        require_court,
        fps=fps,
        backward_seconds=refine_backward_seconds,
        forward_seconds=refine_forward_seconds,
        fast_forward_seconds=refine_fast_forward_seconds,
        merge_seconds=refine_merge_seconds,
    )


def _v37_timing_score(item: FrameFeatures, source_frame: int, fps: float) -> float:
    vertical_speed = abs(item.y_velocity_before) + abs(item.y_velocity_after)
    _, event_angle, y_turn, y_accel = _event_motion(item)
    court_evidence, court_agreement, court_jitter = _court_bounce_evidence(item)
    return (
        0.50 * (1.0 - min(vertical_speed / 40.0, 1.0))
        + 0.18 * (1.0 if y_turn else 0.0)
        + 0.14 * min(event_angle / 120.0, 1.0)
        + 0.12 * item.track_support
        + 0.06 * min(y_accel / 30.0, 1.0)
        + 0.12 * court_evidence
        + 0.08 * court_agreement
        - 0.14 * court_jitter
        - 0.12 * abs(item.frame - source_frame) / max(float(fps), 1.0)
    )


def refine_v37_event_frames(
    events: Sequence[Event],
    features: Sequence[FrameFeatures],
    require_court: bool,
    fps: float = 30.0,
    backward_seconds: float = 0.12,
    forward_seconds: float = 6.0 / 30.0,
    fast_forward_seconds: float = 0.50,
    merge_seconds: float = 2.0 / 30.0,
) -> List[Event]:
    by_frame = {item.frame: item for item in features}
    refined: List[Event] = []
    for event in events:
        source_item = by_frame.get(event.frame)
        if source_item is None:
            refined.append(event)
            continue
        source_vertical_speed = abs(source_item.y_velocity_before) + abs(source_item.y_velocity_after)
        backward = seconds_to_frames(backward_seconds, fps)
        forward = seconds_to_frames(
            fast_forward_seconds if source_vertical_speed > 25.0 else forward_seconds,
            fps,
        )
        candidates: List[FrameFeatures] = []
        for frame in range(event.frame - backward, event.frame + forward + 1):
            item = by_frame.get(frame)
            if item is None or not item.center_observed or item.observed_count < 3:
                continue
            if require_court and not item.court_available:
                continue
            if item.court_available and not item.court_inside:
                continue
            candidates.append(item)
        if not candidates:
            refined.append(event)
            continue
        timing_scores = {item.frame: _v37_timing_score(item, event.frame, fps) for item in candidates}
        best = max(candidates, key=lambda item: timing_scores[item.frame])
        best_timing_score = timing_scores[best.frame]
        contact_onsets = []
        for item in candidates:
            _, _, y_turn, _ = _event_motion(item)
            court_evidence, court_agreement, court_jitter = _court_bounce_evidence(item)
            vertical_speed = abs(item.y_velocity_before) + abs(item.y_velocity_after)
            if (
                timing_scores[item.frame] >= best_timing_score - 0.14
                and y_turn
                and vertical_speed <= 9.0
                and court_evidence >= 0.30
                and court_agreement >= 0.20
                and court_jitter < 0.50
            ):
                contact_onsets.append(item)
        if contact_onsets:
            best = min(contact_onsets, key=lambda item: item.frame)
            best_timing_score = timing_scores[best.frame]
        support_frames = sorted(
            frame for frame, score in timing_scores.items()
            if score >= best_timing_score - 0.10
        )
        interval = [best.frame]
        for frame in reversed([value for value in support_frames if value < best.frame]):
            if interval[0] - frame <= max(1, seconds_to_frames(0.05, fps)):
                interval.insert(0, frame)
            else:
                break
        for frame in [value for value in support_frames if value > best.frame]:
            if frame - interval[-1] <= max(1, seconds_to_frames(0.05, fps)):
                interval.append(frame)
            else:
                break
        runner_up = max(
            (score for frame, score in timing_scores.items() if frame != best.frame),
            default=best_timing_score - 0.25,
        )
        timing_confidence = max(0.0, min(1.0, 0.55 + 1.8 * (best_timing_score - runner_up)))
        event_accel, event_angle, _, _ = _event_motion(best)
        court_evidence, court_agreement, _ = _court_bounce_evidence(best)
        refined_source = (
            "v37_dual_space_refined"
            if court_evidence >= 0.45 and court_agreement >= 0.25
            else "v37_flight_segment_refined"
        )
        refined.append(
            Event(
                frame=best.frame,
                event_type=event.event_type,
                x=best.point[0],
                y=best.point[1],
                score=event.score,
                near_player=best.near_player,
                accel=event_accel,
                angle_change=event_angle,
                source=refined_source,
                track_support=best.track_support,
                observed_count=best.observed_count,
                court_evidence=court_evidence,
                court_agreement=court_agreement,
                frame_start=min(interval),
                frame_end=max(interval),
                timing_confidence=timing_confidence,
            )
        )
    return _merge_events(refined, seconds_to_frames(merge_seconds, fps))


def detect_events(
    tracks: Sequence[Optional[Point]],
    scores: Sequence[Optional[float]],
    player_contexts: Dict[int, PlayerFrameContext],
    court_polygons,
    fps: float,
    court_tracks: Optional[Sequence[Optional[Point]]] = None,
    hit_threshold: float = 0.55,
    bounce_threshold: float = 0.60,
    min_event_gap: Optional[int] = None,
    event_mode: str = "all",
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
    frame_margin: float = 4.0,
    contact_threshold: float = 0.18,
    min_flight_frames: Optional[int] = None,
    max_bounce_step: float = 260.0,
    min_bounce_y_accel: float = 8.0,
    track_sources: Optional[Sequence[str]] = None,
    accepted_flags: Optional[Sequence[int]] = None,
    court_qualities: Optional[Sequence[float]] = None,
    min_event_gap_seconds: float = 0.60,
    min_flight_seconds: float = 0.15,
    feature_sample_seconds: float = 1.0 / 30.0,
    feature_gap_seconds: float = 2.0 / 30.0,
    refine_backward_seconds: float = 0.12,
    refine_forward_seconds: float = 6.0 / 30.0,
    refine_fast_forward_seconds: float = 0.50,
    refine_merge_seconds: float = 2.0 / 30.0,
    suppress_serve_toss: bool = True,
    serve_toss_max_seconds: float = 1.20,
    serve_toss_min_rise_seconds: float = 0.10,
    serve_toss_min_rise_player_ratio: float = 0.22,
    serve_toss_max_lateral_player_ratio: float = 0.85,
) -> List[Event]:
    events: List[Event] = []
    resolved_event_gap = (
        max(1, int(min_event_gap))
        if min_event_gap is not None
        else seconds_to_frames(min_event_gap_seconds, fps)
    )
    resolved_flight_frames = (
        max(1, int(min_flight_frames))
        if min_flight_frames is not None
        else seconds_to_frames(min_flight_seconds, fps)
    )
    resolved_feature_gap = seconds_to_frames(feature_gap_seconds, fps)
    features = build_frame_features(
        tracks,
        scores,
        player_contexts,
        court_polygons,
        fps,
        court_tracks,
        track_sources=track_sources,
        accepted_flags=accepted_flags,
        feature_sample_seconds=feature_sample_seconds,
        court_qualities=court_qualities,
    )
    if event_mode == "v37_bounce":
        return detect_v37_bounce_events(
            features,
            bounce_threshold,
            resolved_event_gap,
            contact_threshold,
            resolved_flight_frames,
            fps=fps,
            feature_gap_frames=resolved_feature_gap,
            suppress_serve_toss=suppress_serve_toss,
            serve_toss_max_seconds=serve_toss_max_seconds,
            serve_toss_min_rise_seconds=serve_toss_min_rise_seconds,
            serve_toss_min_rise_player_ratio=serve_toss_min_rise_player_ratio,
            serve_toss_max_lateral_player_ratio=serve_toss_max_lateral_player_ratio,
            refine_backward_seconds=refine_backward_seconds,
            refine_forward_seconds=refine_forward_seconds,
            refine_fast_forward_seconds=refine_fast_forward_seconds,
            refine_merge_seconds=refine_merge_seconds,
        )
    if event_mode == "reference_bounce":
        return detect_reference_bounce_events(
            features,
            bounce_threshold,
            resolved_event_gap,
            frame_width=frame_width,
            frame_height=frame_height,
            frame_margin=frame_margin,
            contact_threshold=contact_threshold,
            min_flight_frames=resolved_flight_frames,
            max_bounce_step=max_bounce_step,
            min_bounce_y_accel=min_bounce_y_accel,
        )

    hit_events = detect_hit_events_state_machine(features, hit_threshold, resolved_event_gap)
    hit_frames = [event.frame for event in hit_events]
    if event_mode != "bounce_only":
        events.extend(hit_events)

    for item in features:
        if not item.center_observed or item.observed_count < 3:
            continue
        if not _is_in_frame(item, frame_width, frame_height, frame_margin):
            continue
        if any(abs(item.frame - hit_frame) <= resolved_event_gap for hit_frame in hit_frames):
            continue
        court_bonus = 0.10 if item.in_court else -0.20
        event_accel, event_angle, y_turn, _ = _event_motion(item)
        y_turn_score = 1.0 if y_turn else 0.0
        bounce_score = (
            0.32 * min(event_accel / 65.0, 1.0)
            + 0.28 * min(event_angle / 130.0, 1.0)
            + 0.14 * min(item.speed_change / 45.0, 1.0)
            + 0.12 * y_turn_score
            + court_bonus
            - 0.30 * item.near_player
            - 0.18 * item.upper_player
        )
        if bounce_score >= bounce_threshold:
            events.append(
                Event(
                    frame=item.frame,
                    event_type="bounce",
                    x=item.point[0],
                    y=item.point[1],
                    score=bounce_score,
                    near_player=item.near_player,
                    accel=event_accel,
                    angle_change=event_angle,
                    source="court_trajectory_geometry" if 0 < item.court_accel < 500.0 else "trajectory_geometry",
                    track_support=item.track_support,
                    observed_count=item.observed_count,
                )
            )

    sequenced = _sequence_events(
        _suppress_nearby_opposite_events(events, resolved_event_gap),
        resolved_event_gap,
    )
    if event_mode == "bounce_only":
        return [event for event in sequenced if event.event_type == "bounce"]
    return sequenced


def write_event_csv(path: str, events: Sequence[Event]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "frame", "event_type", "x", "y", "score", "near_player",
                "accel", "angle_change", "track_support", "observed_count",
                "court_evidence", "court_agreement", "source",
                "frame_start", "frame_end", "timing_confidence",
                "original_frame", "frame_offset", "timing_method",
            ]
        )
        for event in events:
            writer.writerow(
                [
                    event.frame,
                    event.event_type,
                    f"{event.x:.2f}",
                    f"{event.y:.2f}",
                    f"{event.score:.6f}",
                    f"{event.near_player:.6f}",
                    f"{event.accel:.6f}",
                    f"{event.angle_change:.6f}",
                    f"{event.track_support:.6f}",
                    event.observed_count,
                    f"{event.court_evidence:.6f}",
                    f"{event.court_agreement:.6f}",
                    event.source,
                    event.frame if event.frame_start < 0 else event.frame_start,
                    event.frame if event.frame_end < 0 else event.frame_end,
                    f"{event.timing_confidence:.6f}",
                    event.frame if event.original_frame < 0 else event.original_frame,
                    event.frame_offset,
                    event.timing_method,
                ]
            )
