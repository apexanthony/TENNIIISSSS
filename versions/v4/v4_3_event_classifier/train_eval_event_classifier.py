import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
V41_DIR = ROOT / "versions" / "v4" / "v4_1_bounce_rule"
V42_DIR = ROOT / "versions" / "v4" / "v4_2_bounce_classifier"
for path in (V41_DIR, V42_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bounce_rule_detector import (  # noqa: E402
    BounceCandidate,
    BounceRuleParams,
    TrackPoint,
    candidate_score_with_region,
    extract_candidate_features,
    load_label_csv,
    load_playable_regions_for_clip,
    match_events,
    metrics_from_counts,
    nms_candidates,
    point_in_regions,
    recent_hit_like_penalty,
    true_bounce_frames,
)
from train_eval_bounce_classifier import (  # noqa: E402
    FEATURE_NAMES as V42_FEATURE_NAMES,
    feature_context,
    make_feature_vector,
)


EXTRA_FEATURE_NAMES = [
    "local_valid_count",
    "local_conf_mean",
    "local_conf_min",
    "pre_conf_mean",
    "post_conf_mean",
    "conf_delta",
    "local_x_range",
    "local_y_range",
    "y_pos_in_local_range",
    "future_y_max_offset",
    "future_y_gain",
    "future_y_drop",
    "dy_before_3",
    "dy_after_3",
    "dy_delta_3",
    "speed_before_3",
    "speed_after_3",
    "speed_delta_3",
    "dy_before_5",
    "dy_after_5",
    "dy_delta_5",
    "speed_before_5",
    "speed_after_5",
    "speed_delta_5",
    "dy_before_8",
    "dy_after_8",
    "dy_delta_8",
    "speed_before_8",
    "speed_after_8",
    "speed_delta_8",
]
FEATURE_NAMES = V42_FEATURE_NAMES + EXTRA_FEATURE_NAMES


def parse_games(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def split_clip_name(label_path: Path) -> str:
    return "/".join(label_path.parts[-3:-1])


def clip_game(clip: str) -> str:
    return clip.replace("\\", "/").split("/")[0]


def load_tracks(labels_root: Path) -> Dict[str, List[TrackPoint]]:
    tracks: Dict[str, List[TrackPoint]] = {}
    for label_path in sorted(labels_root.glob("game*/Clip*/Label.csv")):
        tracks[split_clip_name(label_path)] = load_label_csv(str(label_path))
    return tracks


def default_rule_params(region_penalty: float) -> BounceRuleParams:
    return BounceRuleParams(
        region_bonus=0.0,
        region_penalty=region_penalty,
        min_score=9.5,
        enable_v41=True,
        enable_v42=True,
    )


def finite_mean(values: Sequence[float], default: float = 0.0) -> float:
    clean = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(clean)) if clean else default


def point_by_frame(points: Sequence[TrackPoint]) -> Dict[int, TrackPoint]:
    return {point.frame_id: point for point in points}


def valid_points_in_window(
    by_frame: Dict[int, TrackPoint],
    start: int,
    end: int,
    min_conf: float,
) -> List[TrackPoint]:
    return [
        point
        for frame_id in range(start, end + 1)
        if (point := by_frame.get(frame_id)) is not None
        and point.valid
        and point.confidence >= min_conf
    ]


def velocity_features(
    by_frame: Dict[int, TrackPoint],
    frame_id: int,
    k: int,
    min_conf: float,
) -> Tuple[float, float, float, float, float, float]:
    center = by_frame.get(frame_id)
    prev = by_frame.get(frame_id - k)
    nxt = by_frame.get(frame_id + k)
    if center is None or not center.valid or center.confidence < min_conf:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    if prev is not None and prev.valid and prev.confidence >= min_conf:
        dx_before = (center.x - prev.x) / max(1, k)
        dy_before = (center.y - prev.y) / max(1, k)
        speed_before = float(np.hypot(dx_before, dy_before))
    else:
        dy_before = 0.0
        speed_before = 0.0

    if nxt is not None and nxt.valid and nxt.confidence >= min_conf:
        dx_after = (nxt.x - center.x) / max(1, k)
        dy_after = (nxt.y - center.y) / max(1, k)
        speed_after = float(np.hypot(dx_after, dy_after))
    else:
        dy_after = 0.0
        speed_after = 0.0

    return (
        dy_before,
        dy_after,
        dy_after - dy_before,
        speed_before,
        speed_after,
        speed_after - speed_before,
    )


def extra_features(
    feature: BounceCandidate,
    points: Sequence[TrackPoint],
    min_conf: float,
) -> List[float]:
    by_frame = point_by_frame(points)
    local = valid_points_in_window(by_frame, feature.frame_id - 8, feature.frame_id + 12, min_conf)
    pre = valid_points_in_window(by_frame, feature.frame_id - 8, feature.frame_id - 1, min_conf)
    post = valid_points_in_window(by_frame, feature.frame_id, feature.frame_id + 12, min_conf)

    confs = [point.confidence for point in local]
    pre_conf = [point.confidence for point in pre]
    post_conf = [point.confidence for point in post]
    xs = [point.x for point in local]
    ys = [point.y for point in local]
    x_range = max(xs) - min(xs) if xs else 0.0
    y_range = max(ys) - min(ys) if ys else 0.0
    y_pos = (feature.y - min(ys)) / max(y_range, 1e-6) if ys else 0.0

    future = valid_points_in_window(by_frame, feature.frame_id, feature.frame_id + 12, min_conf)
    if future:
        future_y = [point.y for point in future]
        max_index = int(np.argmax(np.asarray(future_y, dtype=np.float32)))
        future_y_max_offset = future[max_index].frame_id - feature.frame_id
        future_y_gain = max(future_y) - feature.y
        future_y_drop = feature.y - min(future_y)
    else:
        future_y_max_offset = 0
        future_y_gain = 0.0
        future_y_drop = 0.0

    values = [
        float(len(local)),
        finite_mean(confs),
        min(confs) if confs else 0.0,
        finite_mean(pre_conf, feature.confidence),
        finite_mean(post_conf, feature.confidence),
        finite_mean(post_conf, feature.confidence) - finite_mean(pre_conf, feature.confidence),
        x_range,
        y_range,
        y_pos,
        float(future_y_max_offset),
        future_y_gain,
        future_y_drop,
    ]
    for k in (3, 5, 8):
        values.extend(velocity_features(by_frame, feature.frame_id, k, min_conf))
    return values


def nearest_offset(frame_id: int, true_frames: Sequence[int]) -> Optional[int]:
    if not true_frames:
        return None
    nearest = min(true_frames, key=lambda item: abs(item - frame_id))
    return int(nearest - frame_id)


def candidate_to_row(
    clip: str,
    points: Sequence[TrackPoint],
    feature: BounceCandidate,
    adjusted_score: float,
    recent_penalty: float,
    in_region: float,
    ctx: dict,
    true_frames: Sequence[int],
    offset_min: int,
    offset_max: int,
    negative_gap: int,
    min_conf: float,
) -> Optional[dict]:
    offset = nearest_offset(feature.frame_id, true_frames)
    if offset is not None and offset_min <= offset <= offset_max:
        label = 1
    elif offset is None or abs(offset) > negative_gap:
        label = 0
        offset = 0
    else:
        return None

    base = make_feature_vector(feature, adjusted_score, recent_penalty, in_region, len(points), ctx)
    vector = base + extra_features(feature, points, min_conf)
    return {
        "clip": clip,
        "frame_id": feature.frame_id,
        "x": feature.x,
        "y": feature.y,
        "label": label,
        "offset": int(offset),
        "rule_score": adjusted_score,
        "features": vector,
    }


def build_clip_samples(
    clip: str,
    points: Sequence[TrackPoint],
    params: BounceRuleParams,
    region_root: str,
    candidate_min_score: float,
    offset_min: int,
    offset_max: int,
    negative_gap: int,
    min_conf: float,
) -> Tuple[List[dict], List[int]]:
    regions = load_playable_regions_for_clip(clip, region_root)
    features = extract_candidate_features(points, params)
    ctx = feature_context(features)
    true_frames = true_bounce_frames(points)

    previous_features: List[BounceCandidate] = []
    samples: List[dict] = []
    for feature in features:
        score = candidate_score_with_region(feature, params, regions)
        if score is None:
            previous_features.append(feature)
            continue
        recent_penalty = recent_hit_like_penalty(feature, previous_features, params)
        adjusted_score = score - recent_penalty
        previous_features.append(feature)
        if adjusted_score < candidate_min_score:
            continue

        in_region = 1.0
        if regions:
            in_region = 1.0 if point_in_regions(feature.x, feature.y, regions) else 0.0

        row = candidate_to_row(
            clip,
            points,
            feature,
            adjusted_score,
            recent_penalty,
            in_region,
            ctx,
            true_frames,
            offset_min,
            offset_max,
            negative_gap,
            min_conf,
        )
        if row is not None:
            samples.append(row)
    return samples, true_frames


def select_xy(samples: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray([sample["features"] for sample in samples], dtype=np.float32)
    y = np.asarray([sample["label"] for sample in samples], dtype=np.int64)
    return x, y


def train_event_classifier(
    samples: Sequence[dict],
    random_state: int,
    hard_negative_threshold: float,
    hard_negative_weight: float,
    event_trees: int,
    event_max_depth: int,
    event_min_leaf: int,
) -> RandomForestClassifier:
    x, y = select_xy(samples)
    base = RandomForestClassifier(
        n_estimators=max(50, event_trees // 2),
        max_depth=event_max_depth,
        min_samples_leaf=event_min_leaf,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    base.fit(x, y)
    base_probs = base.predict_proba(x)[:, 1]
    weights = np.ones(len(samples), dtype=np.float32)
    hard_negatives = (y == 0) & (base_probs >= hard_negative_threshold)
    weights[hard_negatives] = hard_negative_weight

    clf = RandomForestClassifier(
        n_estimators=event_trees,
        max_depth=event_max_depth,
        min_samples_leaf=event_min_leaf,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state + 1,
    )
    clf.fit(x, y, sample_weight=weights)
    return clf


def train_offset_classifier(
    samples: Sequence[dict],
    random_state: int,
    offset_trees: int,
    offset_max_depth: int,
    offset_min_leaf: int,
) -> RandomForestClassifier:
    positives = [sample for sample in samples if sample["label"] == 1]
    x = np.asarray([sample["features"] for sample in positives], dtype=np.float32)
    y = np.asarray([sample["offset"] for sample in positives], dtype=np.int64)
    clf = RandomForestClassifier(
        n_estimators=offset_trees,
        max_depth=offset_max_depth,
        min_samples_leaf=offset_min_leaf,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state + 2,
    )
    clf.fit(x, y)
    return clf


def group_by_clip(samples: Sequence[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for sample in samples:
        out.setdefault(sample["clip"], []).append(sample)
    return out


def predict_clip_events(
    event_clf: RandomForestClassifier,
    offset_clf: RandomForestClassifier,
    samples: Sequence[dict],
    threshold: float,
    nms_window: int,
    use_offset: bool,
) -> List[int]:
    if not samples:
        return []
    x = np.asarray([sample["features"] for sample in samples], dtype=np.float32)
    probs = event_clf.predict_proba(x)[:, 1]
    offsets = offset_clf.predict(x) if use_offset else np.zeros(len(samples), dtype=np.int64)

    candidates: List[BounceCandidate] = []
    for sample, prob, offset in zip(samples, probs, offsets):
        if prob < threshold:
            continue
        frame_id = int(sample["frame_id"] + int(offset))
        candidates.append(
            BounceCandidate(
                frame_id=frame_id,
                x=float(sample["x"]),
                y=float(sample["y"]),
                score=float(prob),
                confidence=1.0,
                angle_change=0.0,
                accel_norm=0.0,
                speed_before=0.0,
                speed_after=0.0,
                jump_distance=0.0,
                valid_points=0,
            )
        )
    return [candidate.frame_id for candidate in nms_candidates(candidates, nms_window)]


def evaluate_events(
    event_clf: RandomForestClassifier,
    offset_clf: RandomForestClassifier,
    samples_by_clip: Dict[str, List[dict]],
    true_by_clip: Dict[str, List[int]],
    threshold: float,
    nms_window: int,
    tolerance: int,
    use_offset: bool,
) -> dict:
    total_tp = total_fp = total_fn = 0
    errors: List[int] = []
    predictions = 0
    for clip, samples in samples_by_clip.items():
        pred_frames = predict_clip_events(event_clf, offset_clf, samples, threshold, nms_window, use_offset)
        true_frames = true_by_clip.get(clip, [])
        tp, fp, fn, clip_errors = match_events(pred_frames, true_frames, tolerance)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        errors.extend(clip_errors)
        predictions += len(pred_frames)
    metrics = metrics_from_counts(total_tp, total_fp, total_fn)
    metrics.update(
        {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "predictions": predictions,
            "avg_frame_error": sum(errors) / len(errors) if errors else 0.0,
        }
    )
    return metrics


def evaluate_events_detail(
    event_clf: RandomForestClassifier,
    offset_clf: RandomForestClassifier,
    samples_by_clip: Dict[str, List[dict]],
    true_by_clip: Dict[str, List[int]],
    threshold: float,
    nms_window: int,
    tolerance: int,
    use_offset: bool,
) -> Tuple[dict, List[dict]]:
    total_tp = total_fp = total_fn = 0
    errors: List[int] = []
    predictions = 0
    rows: List[dict] = []
    for clip, samples in sorted(samples_by_clip.items()):
        pred_frames = predict_clip_events(event_clf, offset_clf, samples, threshold, nms_window, use_offset)
        true_frames = true_by_clip.get(clip, [])
        tp, fp, fn, clip_errors = match_events(pred_frames, true_frames, tolerance)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        errors.extend(clip_errors)
        predictions += len(pred_frames)
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
            "predictions": predictions,
            "avg_frame_error": sum(errors) / len(errors) if errors else 0.0,
        }
    )
    return metrics, rows


def tune_threshold(
    event_clf: RandomForestClassifier,
    offset_clf: RandomForestClassifier,
    samples_by_clip: Dict[str, List[dict]],
    true_by_clip: Dict[str, List[int]],
    nms_window: int,
    tolerance: int,
    use_offset: bool,
) -> Tuple[float, dict]:
    best_threshold = 0.5
    best_metrics: Optional[dict] = None
    for threshold in np.linspace(0.10, 0.90, 81):
        metrics = evaluate_events(
            event_clf,
            offset_clf,
            samples_by_clip,
            true_by_clip,
            float(threshold),
            nms_window,
            tolerance,
            use_offset,
        )
        key = (metrics["f1"], metrics["precision"], metrics["recall"])
        best_key = (
            best_metrics["f1"],
            best_metrics["precision"],
            best_metrics["recall"],
        ) if best_metrics else (-1.0, -1.0, -1.0)
        if key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics or {}


def safe_auc(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    if len(set(y_true.tolist())) < 2:
        return 0.0, 0.0
    return float(roc_auc_score(y_true, probs)), float(average_precision_score(y_true, probs))


def write_metrics(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_rows(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_feature_importance(path: Path, clf: RandomForestClassifier) -> None:
    rows = sorted(
        (
            {"feature": name, "importance": float(value)}
            for name, value in zip(FEATURE_NAMES, clf.feature_importances_)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    write_rows(path, rows)


def print_metrics(name: str, metrics: dict) -> None:
    print(
        f"{name}: P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
        f"F1={metrics['f1']:.4f} TP={metrics['tp']} FP={metrics['fp']} "
        f"FN={metrics['fn']} AvgErr={metrics['avg_frame_error']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate V4.3 event-level bounce classifier with offset prediction.")
    parser.add_argument("--labels-root", default="./datasets/trackNet/images")
    parser.add_argument("--region-root", default="./configs/court_regions")
    parser.add_argument("--out-dir", default="./exps/v4_3_event_classifier")
    parser.add_argument("--train-games", default="game1,game2,game3,game4,game5,game6,game7")
    parser.add_argument("--val-games", default="game8,game9,game10")
    parser.add_argument("--candidate-min-score", type=float, default=3.0)
    parser.add_argument("--offset-min", type=int, default=-3)
    parser.add_argument("--offset-max", type=int, default=10)
    parser.add_argument("--negative-gap", type=int, default=12)
    parser.add_argument("--match-tolerance", type=int, default=3)
    parser.add_argument("--nms-window", type=int, default=22)
    parser.add_argument("--region-penalty", type=float, default=4.0)
    parser.add_argument("--min-conf", type=float, default=0.30)
    parser.add_argument("--hard-negative-threshold", type=float, default=0.35)
    parser.add_argument("--hard-negative-weight", type=float, default=3.0)
    parser.add_argument("--event-trees", type=int, default=500)
    parser.add_argument("--event-max-depth", type=int, default=9)
    parser.add_argument("--event-min-leaf", type=int, default=4)
    parser.add_argument("--offset-trees", type=int, default=400)
    parser.add_argument("--offset-max-depth", type=int, default=8)
    parser.add_argument("--offset-min-leaf", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    labels_root = Path(args.labels_root)
    out_dir = Path(args.out_dir)
    train_games = set(parse_games(args.train_games))
    val_games = set(parse_games(args.val_games))
    params = default_rule_params(args.region_penalty)

    train_samples: List[dict] = []
    val_samples: List[dict] = []
    all_samples: List[dict] = []
    true_by_clip: Dict[str, List[int]] = {}
    for clip, points in load_tracks(labels_root).items():
        samples, true_frames = build_clip_samples(
            clip,
            points,
            params,
            args.region_root,
            args.candidate_min_score,
            args.offset_min,
            args.offset_max,
            args.negative_gap,
            args.min_conf,
        )
        true_by_clip[clip] = true_frames
        all_samples.extend(samples)
        game = clip_game(clip)
        if game in train_games:
            train_samples.extend(samples)
        elif game in val_games:
            val_samples.extend(samples)

    if not train_samples or not val_samples:
        raise RuntimeError("No train/val samples generated. Check dataset paths and game split.")

    event_clf = train_event_classifier(
        train_samples,
        args.random_state,
        args.hard_negative_threshold,
        args.hard_negative_weight,
        args.event_trees,
        args.event_max_depth,
        args.event_min_leaf,
    )
    offset_clf = train_offset_classifier(
        train_samples,
        args.random_state,
        args.offset_trees,
        args.offset_max_depth,
        args.offset_min_leaf,
    )

    train_by_clip = group_by_clip(train_samples)
    val_by_clip = group_by_clip(val_samples)
    all_by_clip = group_by_clip(all_samples)

    threshold, train_metrics = tune_threshold(
        event_clf,
        offset_clf,
        train_by_clip,
        true_by_clip,
        args.nms_window,
        args.match_tolerance,
        True,
    )
    val_metrics, val_detail = evaluate_events_detail(
        event_clf,
        offset_clf,
        val_by_clip,
        true_by_clip,
        threshold,
        args.nms_window,
        args.match_tolerance,
        True,
    )
    all_metrics, all_detail = evaluate_events_detail(
        event_clf,
        offset_clf,
        all_by_clip,
        true_by_clip,
        threshold,
        args.nms_window,
        args.match_tolerance,
        True,
    )
    val_no_offset, val_no_offset_detail = evaluate_events_detail(
        event_clf,
        offset_clf,
        val_by_clip,
        true_by_clip,
        threshold,
        args.nms_window,
        args.match_tolerance,
        False,
    )

    train_x, train_y = select_xy(train_samples)
    val_x, val_y = select_xy(val_samples)
    train_probs = event_clf.predict_proba(train_x)[:, 1]
    val_probs = event_clf.predict_proba(val_x)[:, 1]
    train_auc, train_ap = safe_auc(train_y, train_probs)
    val_auc, val_ap = safe_auc(val_y, val_probs)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "model_event_offset.pkl").open("wb") as fp:
        pickle.dump(
            {
                "event_model": event_clf,
                "offset_model": offset_clf,
                "feature_names": FEATURE_NAMES,
                "threshold": threshold,
                "params": vars(args),
            },
            fp,
        )

    rows = [
        {"split": "train", "threshold": round(threshold, 4), "roc_auc": round(train_auc, 6), "avg_precision": round(train_ap, 6), **train_metrics},
        {"split": "val", "threshold": round(threshold, 4), "roc_auc": round(val_auc, 6), "avg_precision": round(val_ap, 6), **val_metrics},
        {"split": "val_no_offset", "threshold": round(threshold, 4), "roc_auc": round(val_auc, 6), "avg_precision": round(val_ap, 6), **val_no_offset},
        {"split": "all", "threshold": round(threshold, 4), "roc_auc": 0.0, "avg_precision": 0.0, **all_metrics},
    ]
    write_metrics(out_dir / "metrics.csv", rows)
    write_rows(out_dir / "detail_val.csv", val_detail)
    write_rows(out_dir / "detail_val_no_offset.csv", val_no_offset_detail)
    write_rows(out_dir / "detail_all.csv", all_detail)
    write_feature_importance(out_dir / "feature_importance_event.csv", event_clf)
    write_feature_importance(out_dir / "feature_importance_offset.csv", offset_clf)

    print(f"train samples = {len(train_samples)} positives = {sum(s['label'] for s in train_samples)}")
    print(f"val samples   = {len(val_samples)} positives = {sum(s['label'] for s in val_samples)}")
    print(f"threshold     = {threshold:.3f}")
    print(f"sample AUC    = train {train_auc:.4f}, val {val_auc:.4f}")
    print(f"sample AP     = train {train_ap:.4f}, val {val_ap:.4f}")
    print_metrics("train events", train_metrics)
    print_metrics("val events", val_metrics)
    print_metrics("val no offset", val_no_offset)
    print_metrics("all events", all_metrics)
    print(f"out           = {out_dir}")


if __name__ == "__main__":
    main()
