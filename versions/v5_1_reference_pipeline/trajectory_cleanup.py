import math


def _distance(a, b):
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def clean_bidirectional_outliers(stable, fps, residual_seconds=0.08):
    """Remove isolated observations rejected by agreeing forward/backward motion."""
    tracks = list(stable["tracks"])
    accepted = list(stable["accepted"])
    sources = list(stable["source"])
    scores = stable.get("scores", [None] * len(tracks))
    fps_scale = 30.0 / max(float(fps), 1.0)
    agreement_limit = max(16.0, 42.0 * fps_scale)
    residual_limit = max(28.0, 72.0 * fps_scale)
    neighbor_limit = max(22.0, 55.0 * fps_scale)
    removed = 0

    for idx in range(2, len(tracks) - 2):
        p2, p1, current, n1, n2 = tracks[idx - 2], tracks[idx - 1], tracks[idx], tracks[idx + 1], tracks[idx + 2]
        if any(point is None for point in (p2, p1, current, n1, n2)):
            continue

        forward = (2.0 * p1[0] - p2[0], 2.0 * p1[1] - p2[1])
        backward = (2.0 * n1[0] - n2[0], 2.0 * n1[1] - n2[1])
        agreement = _distance(forward, backward)
        residual = 0.5 * (_distance(current, forward) + _distance(current, backward))
        neighbor_jump = min(_distance(current, p1), _distance(current, n1))
        bridge = _distance(p1, n1)
        confidence = scores[idx] if idx < len(scores) and scores[idx] is not None else 0.0

        stable_motion = agreement <= agreement_limit and bridge <= 2.5 * neighbor_limit
        isolated = residual >= residual_limit and neighbor_jump >= neighbor_limit
        exceptionally_strong = confidence >= 0.9995 and residual < 1.6 * residual_limit
        if stable_motion and isolated and not exceptionally_strong:
            tracks[idx] = None
            accepted[idx] = 0
            sources[idx] = "bidirectional_outlier_removed"
            removed += 1

    result = dict(stable)
    result["tracks"] = tracks
    result["accepted"] = accepted
    result["source"] = sources
    result["bidirectional_removed"] = removed
    return result
