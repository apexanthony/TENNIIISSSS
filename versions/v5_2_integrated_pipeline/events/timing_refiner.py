import argparse
import csv
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence, Tuple

V52_DIR = Path(__file__).resolve().parents[1]
for module_dir in (V52_DIR / "context", V52_DIR / "events"):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from detector import Event, write_event_csv


Point = Tuple[float, float]


def _point(row, x_name="x", y_name="y") -> Optional[Point]:
    try:
        x = str(row.get(x_name, "")).strip()
        y = str(row.get(y_name, "")).strip()
        return (float(x), float(y)) if x and y else None
    except (TypeError, ValueError):
        return None


def read_timing_tracks(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as fp:
        rows = list(csv.DictReader(fp))
    count = max((int(float(row["frame"])) for row in rows), default=-1) + 1
    tracks = [None] * count
    court_tracks = [None] * count
    accepted = [0] * count
    sources = ["missing"] * count
    for row in rows:
        frame = int(float(row["frame"]))
        tracks[frame] = _point(row)
        court_tracks[frame] = _point(row, "court_x", "court_y")
        accepted[frame] = int(float(row.get("accepted") or (1 if tracks[frame] else 0)))
        sources[frame] = row.get("source") or "track_csv"
    return tracks, court_tracks, accepted, sources


def read_events(path):
    events = []
    with open(path, "r", newline="", encoding="utf-8-sig") as fp:
        for row in csv.DictReader(fp):
            events.append(
                Event(
                    frame=int(float(row["frame"])),
                    event_type=row["event_type"],
                    x=float(row["x"]),
                    y=float(row["y"]),
                    score=float(row["score"]),
                    near_player=float(row.get("near_player") or 0.0),
                    accel=float(row.get("accel") or 0.0),
                    angle_change=float(row.get("angle_change") or 0.0),
                    source=row.get("source") or "event_csv",
                    track_support=float(row.get("track_support") or 1.0),
                    observed_count=int(float(row.get("observed_count") or 0)),
                    court_evidence=float(row.get("court_evidence") or 0.0),
                    court_agreement=float(row.get("court_agreement") or 0.0),
                    frame_start=int(float(row.get("frame_start") or row["frame"])),
                    frame_end=int(float(row.get("frame_end") or row["frame"])),
                    timing_confidence=float(row.get("timing_confidence") or 0.0),
                    original_frame=int(float(row.get("original_frame") or row["frame"])),
                    frame_offset=int(float(row.get("frame_offset") or 0)),
                    timing_method=row.get("timing_method") or "legacy_refinement",
                )
            )
    return events


def _source_quality(source, accepted):
    source = (source or "").lower()
    if accepted <= 0 or any(token in source for token in ("missing", "removed", "rejected", "expired")):
        return 0.0
    if accepted == 2 or "interpol" in source:
        return 0.35
    if "weak" in source:
        return 0.70
    return 1.0


def _fit_line(samples):
    if len(samples) < 3:
        return None
    mean_t = sum(item[0] for item in samples) / len(samples)
    denom = sum((item[0] - mean_t) ** 2 for item in samples)
    if denom <= 1e-9:
        return None

    def fit_axis(axis):
        mean_v = sum(item[1][axis] for item in samples) / len(samples)
        slope = sum((item[0] - mean_t) * (item[1][axis] - mean_v) for item in samples) / denom
        intercept = mean_v - slope * mean_t
        return slope, intercept

    vx, bx = fit_axis(0)
    vy, by = fit_axis(1)
    residual = math.sqrt(
        sum(
            (point[0] - (vx * frame + bx)) ** 2 + (point[1] - (vy * frame + by)) ** 2
            for frame, point in samples
        )
        / len(samples)
    )
    return (vx, vy), residual


def _angle(a, b):
    na = math.hypot(*a)
    nb = math.hypot(*b)
    if na <= 1e-6 or nb <= 1e-6:
        return 0.0
    cosine = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (na * nb)))
    return math.degrees(math.acos(cosine))


def _candidate_score(frame, original, tracks, accepted, sources, court_tracks, radius):
    if frame < 0 or frame >= len(tracks) or tracks[frame] is None:
        return None
    quality = _source_quality(sources[frame], accepted[frame])
    if quality < 0.65:
        return None

    def samples(sequence, start, end):
        output = []
        for idx in range(max(0, start), min(len(sequence), end + 1)):
            point = sequence[idx]
            if point is not None and _source_quality(sources[idx], accepted[idx]) >= 0.65:
                output.append((idx - frame, point))
        return output

    before = samples(tracks, frame - radius, frame)
    after = samples(tracks, frame, frame + radius)
    fit_before = _fit_line(before)
    fit_after = _fit_line(after)
    if fit_before is None or fit_after is None:
        return None
    velocity_before, residual_before = fit_before
    velocity_after, residual_after = fit_after
    local_steps = []
    local_points = [item[1] for item in before[:-1] + after]
    for first, second in zip(local_points, local_points[1:]):
        local_steps.append(math.dist(first, second))
    scale = max(sorted(local_steps)[len(local_steps) // 2] if local_steps else 1.0, 1.0)
    fit_quality = 1.0 / (1.0 + (residual_before + residual_after) / (2.0 * scale))
    velocity_change = math.dist(velocity_before, velocity_after)
    change_quality = min(velocity_change / max(scale * 1.8, 1.0), 1.0)
    angle_quality = min(_angle(velocity_before, velocity_after) / 120.0, 1.0)

    court_quality = 0.0
    if court_tracks and frame < len(court_tracks) and court_tracks[frame] is not None:
        court_before = [(idx - frame, court_tracks[idx]) for idx in range(max(0, frame - radius), frame + 1) if court_tracks[idx] is not None]
        court_after = [(idx - frame, court_tracks[idx]) for idx in range(frame, min(len(court_tracks), frame + radius + 1)) if court_tracks[idx] is not None]
        cb = _fit_line(court_before)
        ca = _fit_line(court_after)
        if cb is not None and ca is not None:
            court_angle = min(_angle(cb[0], ca[0]) / 120.0, 1.0)
            court_change = min(math.dist(cb[0], ca[0]) / 80.0, 1.0)
            court_quality = 0.55 * court_angle + 0.45 * court_change

    distance_penalty = 0.08 * abs(frame - original) / max(radius, 1)
    score = (
        0.34 * fit_quality
        + 0.28 * change_quality
        + 0.18 * angle_quality
        + 0.12 * quality
        + 0.08 * court_quality
        - distance_penalty
    )
    return score


def refine_contact_timing(
    events: Sequence[Event],
    tracks,
    accepted,
    sources,
    court_tracks=None,
    fps=30.0,
    search_seconds=0.10,
    fit_seconds=0.10,
    min_gain=0.060,
    max_confidence_for_shift=0.60,
    min_interval_width=2,
):
    search = max(1, int(round(search_seconds * max(fps, 1.0))))
    radius = max(3, int(round(fit_seconds * max(fps, 1.0))))
    refined = []
    for event in events:
        if event.event_type != "bounce":
            refined.append(event)
            continue
        original = event.frame
        start = max(0, min(original, event.frame_start if event.frame_start >= 0 else original))
        end = min(len(tracks) - 1, max(original, event.frame_end if event.frame_end >= 0 else original))
        start = max(start, original - search)
        end = min(end, original + search)
        allow_shift = (
            event.timing_confidence < max_confidence_for_shift
            and end - start >= min_interval_width
        )
        scores = {}
        for frame in range(start, end + 1):
            value = _candidate_score(frame, original, tracks, accepted, sources, court_tracks, radius)
            if value is not None:
                scores[frame] = value
        if not scores:
            refined.append(replace(event, original_frame=original, frame_offset=0, timing_method="legacy_refinement"))
            continue
        best_frame = max(scores, key=scores.get)
        original_score = scores.get(original, max(scores.values()) - min_gain)
        if not allow_shift or (best_frame != original and scores[best_frame] < original_score + min_gain):
            best_frame = original
        point = tracks[best_frame] if tracks[best_frame] is not None else (event.x, event.y)
        top = max(scores.values())
        interval_start = event.frame_start if event.frame_start >= 0 else original
        interval_end = event.frame_end if event.frame_end >= 0 else original
        runner_up = max((score for frame, score in scores.items() if frame != best_frame), default=top - 0.20)
        confidence = event.timing_confidence
        if best_frame != original:
            confidence = max(confidence, min(1.0, 0.55 + 2.0 * (top - runner_up)))
        refined.append(
            replace(
                event,
                frame=best_frame,
                x=point[0],
                y=point[1],
                source=event.source + ("+piecewise_timing" if best_frame != original else ""),
                frame_start=interval_start,
                frame_end=interval_end,
                timing_confidence=confidence,
                original_frame=original,
                frame_offset=best_frame - original,
                timing_method="piecewise_motion_break" if best_frame != original else "piecewise_checked_no_change",
            )
        )
    return refined


def main():
    parser = argparse.ArgumentParser(description="Refine bounce frames without rerunning TrackNet or event detection.")
    parser.add_argument("--events-in", required=True)
    parser.add_argument("--track-csv", required=True)
    parser.add_argument("--events-out", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--search-seconds", type=float, default=0.10)
    parser.add_argument("--fit-seconds", type=float, default=0.10)
    parser.add_argument("--min-gain", type=float, default=0.060)
    parser.add_argument("--max-confidence-for-shift", type=float, default=0.60)
    parser.add_argument("--min-interval-width", type=int, default=2)
    args = parser.parse_args()
    tracks, court_tracks, accepted, sources = read_timing_tracks(args.track_csv)
    events = read_events(args.events_in)
    events = refine_contact_timing(
        events, tracks, accepted, sources, court_tracks, args.fps,
        args.search_seconds, args.fit_seconds, args.min_gain,
        args.max_confidence_for_shift, args.min_interval_width,
    )
    Path(args.events_out).parent.mkdir(parents=True, exist_ok=True)
    write_event_csv(args.events_out, events)
    print("refined_offsets=", [event.frame_offset for event in events if event.event_type == "bounce"])


if __name__ == "__main__":
    main()
