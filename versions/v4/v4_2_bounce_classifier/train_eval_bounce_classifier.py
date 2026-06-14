import argparse
import csv
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
V4_DIR = ROOT / "versions" / "v4" / "v4_1_bounce_rule"
if str(V4_DIR) not in sys.path:
    sys.path.insert(0, str(V4_DIR))

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
    recent_hit_like_penalty,
    true_bounce_frames,
)


FEATURE_NAMES = [
    "rule_score",
    "angle_change",
    "accel_norm",
    "speed_before",
    "speed_after",
    "speed_mean",
    "speed_min",
    "speed_max",
    "speed_ratio",
    "speed_delta",
    "abs_speed_delta",
    "jump_distance",
    "valid_points",
    "confidence",
    "frame_pos",
    "in_ground_region",
    "recent_hit_penalty",
    "angle_over_speed",
    "accel_over_speed",
    "jump_over_speed",
    "speed_mean_rel",
    "accel_rel",
    "jump_rel",
]


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


def finite_percentile(values: Sequence[float], q: float, default: float) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return default
    return float(np.percentile(np.asarray(clean, dtype=np.float64), q))


def feature_context(features: Sequence[BounceCandidate]) -> dict:
    speed_means = [0.5 * (f.speed_before + f.speed_after) for f in features]
    accels = [f.accel_norm for f in features]
    jumps = [f.jump_distance for f in features]
    return {
        "speed_med": max(finite_percentile(speed_means, 50, 1.0), 1e-6),
        "accel_med": max(finite_percentile(accels, 50, 1.0), 1e-6),
        "jump_med": max(finite_percentile(jumps, 50, 1.0), 1e-6),
    }


def candidate_from_feature(feature: BounceCandidate, score: float) -> BounceCandidate:
    return BounceCandidate(
        frame_id=feature.frame_id,
        x=feature.x,
        y=feature.y,
        score=score,
        confidence=feature.confidence,
        angle_change=feature.angle_change,
        accel_norm=feature.accel_norm,
        speed_before=feature.speed_before,
        speed_after=feature.speed_after,
        jump_distance=feature.jump_distance,
        valid_points=feature.valid_points,
    )


def nearest_event_distance(frame: int, events: Sequence[int]) -> int:
    if not events:
        return 10**9
    return min(abs(frame - event) for event in events)


def make_feature_vector(
    feature: BounceCandidate,
    rule_score: float,
    recent_penalty: float,
    in_region: float,
    num_frames: int,
    ctx: dict,
) -> List[float]:
    speed_mean = 0.5 * (feature.speed_before + feature.speed_after)
    speed_min = max(min(feature.speed_before, feature.speed_after), 1e-6)
    speed_max = max(feature.speed_before, feature.speed_after)
    speed_ratio = speed_max / speed_min
    speed_delta = feature.speed_after - feature.speed_before
    frame_pos = feature.frame_id / max(1, num_frames - 1)
    return [
        rule_score,
        feature.angle_change,
        feature.accel_norm,
        feature.speed_before,
        feature.speed_after,
        speed_mean,
        speed_min,
        speed_max,
        speed_ratio,
        speed_delta,
        abs(speed_delta),
        feature.jump_distance,
        feature.valid_points,
        feature.confidence,
        frame_pos,
        in_region,
        recent_penalty,
        feature.angle_change / max(speed_mean, 1e-6),
        feature.accel_norm / max(speed_mean, 1e-6),
        feature.jump_distance / max(speed_mean, 1e-6),
        speed_mean / ctx["speed_med"],
        feature.accel_norm / ctx["accel_med"],
        feature.jump_distance / ctx["jump_med"],
    ]


def build_clip_samples(
    clip: str,
    points: Sequence[TrackPoint],
    params: BounceRuleParams,
    region_root: str,
    candidate_min_score: float,
    positive_tolerance: int,
    negative_gap: int,
) -> Tuple[List[dict], List[BounceCandidate], List[int]]:
    regions = load_playable_regions_for_clip(clip, region_root)
    features = extract_candidate_features(points, params)
    ctx = feature_context(features)
    true_frames = true_bounce_frames(points)

    previous_features: List[BounceCandidate] = []
    samples: List[dict] = []
    feature_candidates: List[BounceCandidate] = []
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
            # Region has already affected score; this binary feature gives the classifier context.
            from bounce_rule_detector import point_in_regions

            in_region = 1.0 if point_in_regions(feature.x, feature.y, regions) else 0.0

        dist = nearest_event_distance(feature.frame_id, true_frames)
        if dist <= positive_tolerance:
            label = 1
        elif dist > negative_gap:
            label = 0
        else:
            continue

        vector = make_feature_vector(feature, adjusted_score, recent_penalty, in_region, len(points), ctx)
        samples.append(
            {
                "clip": clip,
                "frame_id": feature.frame_id,
                "x": feature.x,
                "y": feature.y,
                "label": label,
                "event_distance": dist,
                "features": vector,
                "rule_score": adjusted_score,
            }
        )
        feature_candidates.append(candidate_from_feature(feature, adjusted_score))

    return samples, feature_candidates, true_frames


def select_samples(samples: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray([sample["features"] for sample in samples], dtype=np.float32)
    y = np.asarray([sample["label"] for sample in samples], dtype=np.int64)
    return x, y


def train_classifier(samples: Sequence[dict], random_state: int) -> RandomForestClassifier:
    x, y = select_samples(samples)
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(x, y)
    return clf


def predict_probabilities(clf: RandomForestClassifier, samples: Sequence[dict]) -> np.ndarray:
    if not samples:
        return np.asarray([], dtype=np.float32)
    x, _ = select_samples(samples)
    return clf.predict_proba(x)[:, 1]


def evaluate_events(
    samples_by_clip: Dict[str, List[dict]],
    true_by_clip: Dict[str, List[int]],
    probabilities_by_clip: Dict[str, np.ndarray],
    threshold: float,
    nms_window: int,
    tolerance: int,
) -> dict:
    total_tp = total_fp = total_fn = 0
    errors: List[int] = []
    num_predictions = 0

    for clip, samples in samples_by_clip.items():
        probs = probabilities_by_clip.get(clip, np.asarray([], dtype=np.float32))
        candidates = []
        for sample, prob in zip(samples, probs):
            if prob < threshold:
                continue
            candidates.append(
                BounceCandidate(
                    frame_id=sample["frame_id"],
                    x=sample["x"],
                    y=sample["y"],
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
        kept = nms_candidates(candidates, nms_window)
        pred_frames = [cand.frame_id for cand in kept]
        tp, fp, fn, clip_errors = match_events(pred_frames, true_by_clip.get(clip, []), tolerance)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        errors.extend(clip_errors)
        num_predictions += len(pred_frames)

    metrics = metrics_from_counts(total_tp, total_fp, total_fn)
    metrics.update(
        {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "predictions": num_predictions,
            "avg_frame_error": sum(errors) / len(errors) if errors else 0.0,
        }
    )
    return metrics


def tune_threshold(
    samples_by_clip: Dict[str, List[dict]],
    true_by_clip: Dict[str, List[int]],
    probabilities_by_clip: Dict[str, np.ndarray],
    nms_window: int,
    tolerance: int,
) -> Tuple[float, dict]:
    best_threshold = 0.5
    best_metrics: Optional[dict] = None
    for threshold in np.linspace(0.10, 0.90, 81):
        metrics = evaluate_events(samples_by_clip, true_by_clip, probabilities_by_clip, float(threshold), nms_window, tolerance)
        if best_metrics is None or (
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
        ) > (
            best_metrics["f1"],
            best_metrics["precision"],
            best_metrics["recall"],
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics or {}


def group_samples_by_clip(samples: Sequence[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for sample in samples:
        out.setdefault(sample["clip"], []).append(sample)
    return out


def probabilities_by_clip(clf: RandomForestClassifier, samples_by_clip: Dict[str, List[dict]]) -> Dict[str, np.ndarray]:
    return {clip: predict_probabilities(clf, samples) for clip, samples in samples_by_clip.items()}


def write_samples_csv(path: Path, samples: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["clip", "frame_id", "x", "y", "label", "event_distance", "rule_score"] + FEATURE_NAMES
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            row = {
                "clip": sample["clip"],
                "frame_id": sample["frame_id"],
                "x": round(sample["x"], 3),
                "y": round(sample["y"], 3),
                "label": sample["label"],
                "event_distance": sample["event_distance"],
                "rule_score": round(sample["rule_score"], 6),
            }
            row.update({name: round(value, 6) for name, value in zip(FEATURE_NAMES, sample["features"])})
            writer.writerow(row)


def write_metrics_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_auc(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    if len(set(y_true.tolist())) < 2:
        return 0.0, 0.0
    return float(roc_auc_score(y_true, probs)), float(average_precision_score(y_true, probs))


def print_event_metrics(name: str, metrics: dict) -> None:
    print(
        f"{name}: P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
        f"F1={metrics['f1']:.4f} TP={metrics['tp']} FP={metrics['fp']} "
        f"FN={metrics['fn']} AvgErr={metrics['avg_frame_error']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate V4.2 trajectory-feature bounce classifier.")
    parser.add_argument("--labels-root", default="./datasets/trackNet/images")
    parser.add_argument("--region-root", default="./configs/court_regions")
    parser.add_argument("--out-dir", default="./exps/v4_2_bounce_classifier")
    parser.add_argument("--train-games", default="game1,game2,game3,game4,game5,game6,game7")
    parser.add_argument("--val-games", default="game8,game9,game10")
    parser.add_argument("--candidate-min-score", type=float, default=3.0)
    parser.add_argument("--positive-tolerance", type=int, default=3)
    parser.add_argument("--negative-gap", type=int, default=6)
    parser.add_argument("--nms-window", type=int, default=22)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    labels_root = Path(args.labels_root)
    out_dir = Path(args.out_dir)
    train_games = set(parse_games(args.train_games))
    val_games = set(parse_games(args.val_games))

    params = BounceRuleParams(region_bonus=0, region_penalty=4, min_score=9.5, enable_v41=True, enable_v42=True)
    tracks = load_tracks(labels_root)

    train_samples: List[dict] = []
    val_samples: List[dict] = []
    all_samples: List[dict] = []
    true_by_clip: Dict[str, List[int]] = {}

    for clip, points in tracks.items():
        samples, _, true_frames = build_clip_samples(
            clip,
            points,
            params,
            args.region_root,
            args.candidate_min_score,
            args.positive_tolerance,
            args.negative_gap,
        )
        true_by_clip[clip] = true_frames
        all_samples.extend(samples)
        game = clip_game(clip)
        if game in train_games:
            train_samples.extend(samples)
        elif game in val_games:
            val_samples.extend(samples)

    if not train_samples or not val_samples:
        raise RuntimeError("No train/val samples generated. Check game split and dataset paths.")

    clf = train_classifier(train_samples, args.random_state)

    train_by_clip = group_samples_by_clip(train_samples)
    val_by_clip = group_samples_by_clip(val_samples)
    all_by_clip = group_samples_by_clip(all_samples)
    train_probs_by_clip = probabilities_by_clip(clf, train_by_clip)
    val_probs_by_clip = probabilities_by_clip(clf, val_by_clip)
    all_probs_by_clip = probabilities_by_clip(clf, all_by_clip)

    threshold, train_event_metrics = tune_threshold(
        train_by_clip,
        true_by_clip,
        train_probs_by_clip,
        args.nms_window,
        args.positive_tolerance,
    )
    val_event_metrics = evaluate_events(val_by_clip, true_by_clip, val_probs_by_clip, threshold, args.nms_window, args.positive_tolerance)
    all_event_metrics = evaluate_events(all_by_clip, true_by_clip, all_probs_by_clip, threshold, args.nms_window, args.positive_tolerance)

    train_x, train_y = select_samples(train_samples)
    val_x, val_y = select_samples(val_samples)
    train_probs = clf.predict_proba(train_x)[:, 1]
    val_probs = clf.predict_proba(val_x)[:, 1]
    train_auc, train_ap = safe_auc(train_y, train_probs)
    val_auc, val_ap = safe_auc(val_y, val_probs)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_samples_csv(out_dir / "train_samples.csv", train_samples)
    write_samples_csv(out_dir / "val_samples.csv", val_samples)
    with (out_dir / "model_random_forest.pkl").open("wb") as fp:
        pickle.dump({"model": clf, "feature_names": FEATURE_NAMES, "threshold": threshold, "params": vars(args)}, fp)

    rows = [
        {"split": "train", "threshold": round(threshold, 4), "roc_auc": round(train_auc, 6), "avg_precision": round(train_ap, 6), **train_event_metrics},
        {"split": "val", "threshold": round(threshold, 4), "roc_auc": round(val_auc, 6), "avg_precision": round(val_ap, 6), **val_event_metrics},
        {"split": "all", "threshold": round(threshold, 4), "roc_auc": 0.0, "avg_precision": 0.0, **all_event_metrics},
    ]
    write_metrics_csv(out_dir / "metrics.csv", rows)

    print(f"train samples = {len(train_samples)} positives = {sum(s['label'] for s in train_samples)}")
    print(f"val samples   = {len(val_samples)} positives = {sum(s['label'] for s in val_samples)}")
    print(f"threshold     = {threshold:.3f}")
    print(f"sample AUC    = train {train_auc:.4f}, val {val_auc:.4f}")
    print(f"sample AP     = train {train_ap:.4f}, val {val_ap:.4f}")
    print_event_metrics("train events", train_event_metrics)
    print_event_metrics("val events", val_event_metrics)
    print_event_metrics("all events", all_event_metrics)
    print(f"out           = {out_dir}")


if __name__ == "__main__":
    main()
