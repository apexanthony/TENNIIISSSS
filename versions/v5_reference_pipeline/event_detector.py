import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from court_context import point_in_polygons
from player_context import PlayerFrameContext, empty_player_context, nearest_player_distance, player_region_scores


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
    in_court: bool
    player_local_min: bool


def _dist(a: Point, b: Point) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _angle_change(p0: Point, p1: Point, p2: Point) -> float:
    v1 = (p1[0] - p0[0], p1[1] - p0[1])
    v2 = (p2[0] - p1[0], p2[1] - p1[1])
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 <= 1e-6 or n2 <= 1e-6:
        return 0.0
    cos_v = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return float(math.degrees(math.acos(cos_v)))


def _valid_neighbors(tracks: Sequence[Optional[Point]], idx: int):
    if idx - 2 < 0 or idx + 2 >= len(tracks):
        return None
    points = [tracks[idx - 2], tracks[idx - 1], tracks[idx], tracks[idx + 1], tracks[idx + 2]]
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
) -> List[FrameFeatures]:
    features: List[FrameFeatures] = []
    fps_scale = max(float(fps), 1.0) / 30.0
    court_tracks = court_tracks or [None] * len(tracks)

    for idx in range(2, len(tracks) - 2):
        neighbors = _valid_neighbors(tracks, idx)
        if neighbors is None:
            continue
        p0, p_prev, p, p_next, p4 = neighbors
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

        ctx = player_contexts.get(idx, empty_player_context())
        player_dist, near_player = nearest_player_distance(p, ctx)
        upper_player, lower_player, body_inside = player_region_scores(p, ctx)

        player_dist_prev = nearest_player_distance(p_prev, ctx)[0]
        player_dist_curr = nearest_player_distance(p, ctx)[0]
        player_dist_next = nearest_player_distance(p_next, ctx)[0]
        player_local_min = player_dist_curr <= player_dist_prev and player_dist_curr <= player_dist_next

        cp_neighbors = _valid_neighbors(court_tracks, idx)
        if cp_neighbors is None:
            court_accel = 0.0
            court_angle = 0.0
            court_y_velocity_before = 0.0
            court_y_velocity_after = 0.0
            court_y_accel = 0.0
            court_y_turn = False
        else:
            c0, c_prev, c, c_next, c4 = cp_neighbors
            court_accel = abs(_dist(c, c_next) - _dist(c_prev, c)) * fps_scale
            court_angle = _angle_change(c_prev, c, c_next)
            court_y_velocity_before = (c[1] - c_prev[1]) * fps_scale
            court_y_velocity_after = (c_next[1] - c[1]) * fps_scale
            court_y_accel = abs(court_y_velocity_after - court_y_velocity_before)
            court_y_turn = (c[1] - c_prev[1]) * (c_next[1] - c[1]) <= 0

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
                in_court=point_in_polygons(p, court_polygons),
                player_local_min=player_local_min,
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
    image_accel = max(item.accel, item.y_accel)
    image_angle = item.angle
    image_y_turn = item.y_turn
    image_y_accel = item.y_accel

    court_is_sane = 0.0 < item.court_accel < 500.0 and 0.0 <= item.court_angle <= 170.0
    if not court_is_sane:
        return image_accel, image_angle, image_y_turn, image_y_accel

    accel = max(image_accel, min(max(item.court_accel, item.court_y_accel), 180.0))
    angle = max(image_angle, min(item.court_angle, 150.0))
    y_turn = image_y_turn or item.court_y_turn
    y_accel = max(image_y_accel, min(item.court_y_accel, 180.0))
    return accel, angle, y_turn, y_accel


def _reference_bounce_score(item: FrameFeatures, flight_frames: int, min_flight_frames: int) -> Tuple[float, float, float]:
    event_accel, event_angle, y_turn, y_accel = _event_motion(item)
    near_contact = max(item.near_player, item.upper_player)
    flight_score = min(flight_frames / max(1.0, float(min_flight_frames)), 1.0)
    court_bonus = 0.10 if item.in_court else -0.35
    y_turn_score = 1.0 if y_turn else 0.0
    score = (
        0.34 * min(y_accel / 55.0, 1.0)
        + 0.22 * y_turn_score
        + 0.18 * min(event_angle / 120.0, 1.0)
        + 0.12 * min(item.speed_change / 45.0, 1.0)
        + 0.10 * flight_score
        + court_bonus
        - 0.36 * near_contact
        - 0.18 * item.lower_player
        - 0.12 * item.body_inside
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


def detect_events(
    tracks: Sequence[Optional[Point]],
    scores: Sequence[Optional[float]],
    player_contexts: Dict[int, PlayerFrameContext],
    court_polygons,
    fps: float,
    court_tracks: Optional[Sequence[Optional[Point]]] = None,
    hit_threshold: float = 0.55,
    bounce_threshold: float = 0.60,
    min_event_gap: int = 10,
    event_mode: str = "all",
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
    frame_margin: float = 4.0,
    contact_threshold: float = 0.18,
    min_flight_frames: int = 6,
    max_bounce_step: float = 260.0,
    min_bounce_y_accel: float = 8.0,
) -> List[Event]:
    events: List[Event] = []
    features = build_frame_features(tracks, scores, player_contexts, court_polygons, fps, court_tracks)
    if event_mode == "reference_bounce":
        return detect_reference_bounce_events(
            features,
            bounce_threshold,
            min_event_gap,
            frame_width=frame_width,
            frame_height=frame_height,
            frame_margin=frame_margin,
            contact_threshold=contact_threshold,
            min_flight_frames=min_flight_frames,
            max_bounce_step=max_bounce_step,
            min_bounce_y_accel=min_bounce_y_accel,
        )

    hit_events = detect_hit_events_state_machine(features, hit_threshold, min_event_gap)
    hit_frames = [event.frame for event in hit_events]
    if event_mode != "bounce_only":
        events.extend(hit_events)

    for item in features:
        if not _is_in_frame(item, frame_width, frame_height, frame_margin):
            continue
        if any(abs(item.frame - hit_frame) <= min_event_gap for hit_frame in hit_frames):
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
                )
            )

    sequenced = _sequence_events(_suppress_nearby_opposite_events(events, min_event_gap), min_event_gap)
    if event_mode == "bounce_only":
        return [event for event in sequenced if event.event_type == "bounce"]
    return sequenced


def write_event_csv(path: str, events: Sequence[Event]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["frame", "event_type", "x", "y", "score", "near_player", "accel", "angle_change", "source"])
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
                    event.source,
                ]
            )
