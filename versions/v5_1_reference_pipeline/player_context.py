import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


@dataclass
class PlayerFrameContext:
    hands: List[Point]
    boxes: List[BBox]
    centers: List[Point]
    hit_radius: float
    crop_boxes: List[BBox]
    quality: float = 0.0


def empty_player_context() -> PlayerFrameContext:
    return PlayerFrameContext(hands=[], boxes=[], centers=[], hit_radius=0.0, crop_boxes=[], quality=0.0)


def parse_player_boxes(value: str) -> List[BBox]:
    boxes: List[BBox] = []
    if value is None:
        return boxes
    value = str(value).strip()
    if not value or value.lower() == "nan":
        return boxes
    for item in value.split("|"):
        parts = [p.strip() for p in item.split(",") if p.strip()]
        if len(parts) != 4:
            continue
        x1, y1, x2, y2 = [float(p) for p in parts]
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def context_from_boxes(boxes: Sequence[BBox]) -> PlayerFrameContext:
    centers = [((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5) for b in boxes]
    hands: List[Point] = []
    for x1, y1, x2, y2 in boxes:
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        hands.append((x1 + 0.25 * width, y1 + 0.30 * height))
        hands.append((x1 + 0.75 * width, y1 + 0.30 * height))
    radius = float(np.median([max(b[2] - b[0], b[3] - b[1]) for b in boxes])) * 0.45 if boxes else 0.0
    return PlayerFrameContext(hands=hands, boxes=list(boxes), centers=centers, hit_radius=radius, crop_boxes=[], quality=0.65)


def load_player_box_csv(path: str) -> Dict[int, PlayerFrameContext]:
    """Load optional player boxes: frame,player_id,x1,y1,x2,y2."""
    if not path:
        return {}

    by_frame: Dict[int, List[BBox]] = {}
    with open(path, "r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            frame = int(row.get("frame") or row.get("frame_id") or 0)
            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])
            by_frame.setdefault(frame, []).append((x1, y1, x2, y2))

    return {frame: context_from_boxes(boxes) for frame, boxes in by_frame.items()}


def load_annotation_player_contexts(path: str, game: str = "", clip: str = "") -> Dict[int, PlayerFrameContext]:
    """Load player boxes from mapped Roboflow annotations."""
    if not path:
        return {}

    out: Dict[int, PlayerFrameContext] = {}
    with open(path, "r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if game and row.get("game") != game:
                continue
            if clip and row.get("clip") != clip:
                continue
            frame = int(float(row.get("frame_id") or row.get("frame") or 0))
            boxes = parse_player_boxes(row.get("player_boxes", ""))
            if boxes:
                out[frame] = context_from_boxes(boxes)
    return out


def player_candidate_penalty(point: Optional[Point], ctx: PlayerFrameContext) -> float:
    """Penalty for candidates likely caused by player body/shoes."""
    if point is None or not ctx.boxes:
        return 0.0
    x, y = point
    best = 0.0
    for x1, y1, x2, y2 in ctx.boxes:
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        margin = max(12.0, 0.18 * max(width, height))
        inside = x1 <= x <= x2 and y1 <= y <= y2
        near = (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)
        if inside:
            rel_y = (y - y1) / height
            penalty = 0.28
            if rel_y > 0.55:
                penalty += 0.22
            best = max(best, penalty)
        elif near:
            dx = max(x1 - x, 0.0, x - x2)
            dy = max(y1 - y, 0.0, y - y2)
            dist = float(np.hypot(dx, dy))
            best = max(best, 0.18 * max(0.0, 1.0 - dist / margin))
    return min(best, 0.70)


def player_region_scores(point: Optional[Point], ctx: PlayerFrameContext) -> Tuple[float, float, float]:
    """Return upper-body proximity, lower-body/shoe proximity, and inside-body score."""
    if point is None or not ctx.boxes:
        return 0.0, 0.0, 0.0
    x, y = point
    upper = 0.0
    lower = 0.0
    inside_body = 0.0
    for x1, y1, x2, y2 in ctx.boxes:
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        margin = max(12.0, 0.20 * max(width, height))
        if not ((x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)):
            continue
        dx = max(x1 - x, 0.0, x - x2)
        dy = max(y1 - y, 0.0, y - y2)
        dist = float(np.hypot(dx, dy))
        proximity = max(0.0, 1.0 - dist / margin)
        rel_y = (min(max(y, y1), y2) - y1) / height
        if x1 <= x <= x2 and y1 <= y <= y2:
            inside_body = max(inside_body, 1.0)
        if rel_y <= 0.55:
            upper = max(upper, proximity)
        else:
            lower = max(lower, proximity)
    return upper, lower, inside_body


def load_crop_config(path: str, width: int, height: int):
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    crops = []
    for item in data.get("crops", []):
        if all(key in item for key in ("x1", "y1", "x2", "y2")):
            x1 = float(item["x1"])
            y1 = float(item["y1"])
            x2 = float(item["x2"])
            y2 = float(item["y2"])
            if max(x1, y1, x2, y2) <= 1.5:
                crop = (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height))
            else:
                crop = (int(x1), int(y1), int(x2), int(y2))
            crops.append(crop)
    return crops or None


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(1.0, ax2 - ax1) * max(1.0, ay2 - ay1)
    area_b = max(1.0, bx2 - bx1) * max(1.0, by2 - by1)
    return float(inter / max(area_a + area_b - inter, 1.0))


def _bbox_center(box: BBox) -> Point:
    return (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))


def _dedupe_pose_contexts(contexts: Sequence[PlayerFrameContext]) -> List[PlayerFrameContext]:
    kept: List[PlayerFrameContext] = []
    for ctx in contexts:
        if not ctx.boxes:
            continue
        box = ctx.boxes[0]
        center = _bbox_center(box)
        duplicate = False
        for prev in kept:
            prev_box = prev.boxes[0]
            prev_center = _bbox_center(prev_box)
            center_dist = float(np.hypot(center[0] - prev_center[0], center[1] - prev_center[1]))
            prev_scale = max(prev_box[2] - prev_box[0], prev_box[3] - prev_box[1], 1.0)
            if _bbox_iou(box, prev_box) > 0.18 or center_dist < prev_scale * 0.35:
                duplicate = True
                break
        if not duplicate:
            kept.append(ctx)
    return kept


class MediaPipePoseProvider:
    def __init__(self, width: int, height: int, model_complexity: int = 1, crop_config: str = ""):
        try:
            from mediapipe import solutions  # type: ignore
        except ImportError as exc:
            raise RuntimeError("MediaPipe is not installed. Run: python -m pip install mediapipe") from exc

        self.solutions = solutions
        # Reuse one pose graph for both court crops. Two resident graphs nearly
        # double TensorFlow Lite memory and can exhaust constrained systems.
        self.pose = solutions.pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        self.width = width
        self.height = height
        self.crops = load_crop_config(crop_config, width, height)
        if self.crops is None:
            # Broadcast tennis videos often include spectators/ads above the top
            # player and scoreboards near the lower corners. Keep the default
            # search windows mostly inside the court instead of full half frames.
            self.crops = [
                (int(width * 0.12), int(height * 0.10), int(width * 0.88), int(height * 0.56)),
                (int(width * 0.10), int(height * 0.38), int(width * 0.90), int(height * 0.98)),
            ]

    def close(self) -> None:
        self.pose.close()

    def _process_crop(self, frame, crop, pose) -> PlayerFrameContext:
        x1, y1, x2, y2 = crop
        crop_img = frame[y1:y2, x1:x2]
        if crop_img.size == 0:
            return empty_player_context()

        rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if result.pose_landmarks is None:
            return empty_player_context()

        mp_pose = self.solutions.pose
        landmarks = result.pose_landmarks.landmark

        def point(name) -> Point:
            lm = landmarks[name]
            return (float(lm.x * (x2 - x1) + x1), float(lm.y * (y2 - y1) + y1))

        def visible(name) -> float:
            return float(getattr(landmarks[name], "visibility", 1.0))

        left_hand_id = mp_pose.PoseLandmark.LEFT_INDEX
        right_hand_id = mp_pose.PoseLandmark.RIGHT_INDEX
        left_foot_id = mp_pose.PoseLandmark.LEFT_FOOT_INDEX
        right_foot_id = mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
        nose_id = mp_pose.PoseLandmark.NOSE

        key_visibility = [
            visible(left_hand_id),
            visible(right_hand_id),
            visible(left_foot_id),
            visible(right_foot_id),
            visible(nose_id),
        ]
        if np.mean(key_visibility) < 0.28 or max(key_visibility) < 0.45:
            return empty_player_context()

        hands = [
            point(left_hand_id),
            point(right_hand_id),
        ]
        feet = [
            point(left_foot_id),
            point(right_foot_id),
        ]
        nose = point(nose_id)
        center = ((feet[0][0] + feet[1][0]) * 0.5, max(feet[0][1], feet[1][1]))
        radius = max(20.0, 0.60 * float(np.hypot(center[0] - nose[0], center[1] - nose[1])))

        all_points = hands + feet + [nose]
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        box = (min(xs), min(ys), max(xs), max(ys))
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        crop_width = max(1.0, x2 - x1)
        crop_height = max(1.0, y2 - y1)
        if box_width < crop_width * 0.025 or box_height < crop_height * 0.08:
            return empty_player_context()
        quality = max(0.0, min(float(np.mean(key_visibility)), 1.0))
        return PlayerFrameContext(
            hands=hands,
            boxes=[box],
            centers=[center],
            hit_radius=radius,
            crop_boxes=[crop],
            quality=quality,
        )

    def process(self, frame) -> PlayerFrameContext:
        contexts = [self._process_crop(frame, crop, self.pose) for crop in self.crops]
        contexts = _dedupe_pose_contexts(contexts)
        hands: List[Point] = []
        boxes: List[BBox] = []
        centers: List[Point] = []
        crop_boxes: List[BBox] = list(self.crops)
        radii: List[float] = []
        qualities: List[float] = []
        for ctx in contexts:
            hands.extend(ctx.hands)
            boxes.extend(ctx.boxes)
            centers.extend(ctx.centers)
            if ctx.hit_radius > 0:
                radii.append(ctx.hit_radius)
            if ctx.quality > 0:
                qualities.append(ctx.quality)
        radius = float(np.median(radii)) if radii else 0.0
        quality = float(np.mean(qualities)) if qualities else 0.0
        return PlayerFrameContext(
            hands=hands,
            boxes=boxes,
            centers=centers,
            hit_radius=radius,
            crop_boxes=crop_boxes,
            quality=quality,
        )


def stabilize_player_contexts(
    contexts: Dict[int, PlayerFrameContext],
    width: int,
    height: int,
    max_gap: int = 3,
    alpha: float = 0.55,
) -> Dict[int, PlayerFrameContext]:
    """Keep at most two court players and smooth their boxes/hands over time."""
    if not contexts:
        return {}

    output: Dict[int, PlayerFrameContext] = {}
    previous = [None, None]
    missing = [max_gap + 1, max_gap + 1]
    max_frame = max(contexts)

    def blend_point(old, new):
        if old is None:
            return new
        return ((1.0 - alpha) * old[0] + alpha * new[0], (1.0 - alpha) * old[1] + alpha * new[1])

    def blend_box(old, new):
        if old is None:
            return new
        return tuple((1.0 - alpha) * old[idx] + alpha * new[idx] for idx in range(4))

    for frame in range(max_frame + 1):
        ctx = contexts.get(frame, empty_player_context())
        players = []
        for idx, box in enumerate(ctx.boxes):
            center = ctx.centers[idx] if idx < len(ctx.centers) else _bbox_center(box)
            box_height = box[3] - box[1]
            if not (0.07 * height <= center[1] <= 0.99 * height):
                continue
            if box_height < 0.025 * height or box_height > 0.70 * height:
                continue
            hands = ctx.hands[2 * idx : 2 * idx + 2]
            if len(hands) < 2:
                hands = [center, center]
            players.append((center[1], box, center, hands))

        players.sort(key=lambda item: item[0])
        if len(players) > 2:
            players = [players[0], players[-1]]

        slots = [None, None]
        if len(players) == 1:
            slot = 0 if players[0][2][1] < 0.52 * height else 1
            slots[slot] = players[0]
        elif len(players) == 2:
            slots = players

        smoothed_players = []
        for slot, player in enumerate(slots):
            if player is not None:
                _, box, center, hands = player
                old = previous[slot]
                smooth_box = blend_box(None if old is None else old[0], box)
                smooth_center = blend_point(None if old is None else old[1], center)
                smooth_hands = [
                    blend_point(None if old is None else old[2][idx], hands[idx])
                    for idx in range(2)
                ]
                previous[slot] = (smooth_box, smooth_center, smooth_hands)
                missing[slot] = 0
                smoothed_players.append((smooth_box, smooth_center, smooth_hands, ctx.quality))
            elif previous[slot] is not None and missing[slot] < max_gap:
                missing[slot] += 1
                box, center, hands = previous[slot]
                smoothed_players.append((box, center, hands, max(0.15, ctx.quality * 0.7)))
            else:
                previous[slot] = None
                missing[slot] = max_gap + 1

        boxes = [player[0] for player in smoothed_players]
        centers = [player[1] for player in smoothed_players]
        hands = [hand for player in smoothed_players for hand in player[2]]
        radii = [max(box[2] - box[0], box[3] - box[1]) * 0.45 for box in boxes]
        quality = float(np.mean([player[3] for player in smoothed_players])) if smoothed_players else 0.0
        output[frame] = PlayerFrameContext(
            hands=hands,
            boxes=boxes,
            centers=centers,
            hit_radius=float(np.median(radii)) if radii else 0.0,
            crop_boxes=ctx.crop_boxes,
            quality=quality,
        )
    return output


def nearest_player_distance(point: Optional[Point], ctx: PlayerFrameContext) -> Tuple[float, float]:
    if point is None or not ctx.hands:
        return float("inf"), 0.0
    dist = min(float(np.hypot(point[0] - hand[0], point[1] - hand[1])) for hand in ctx.hands)
    radius = max(ctx.hit_radius, 1.0)
    near_score = max(0.0, 1.0 - dist / radius)
    return dist, near_score


def draw_player_context(frame, ctx: PlayerFrameContext):
    for x1, y1, x2, y2 in ctx.crop_boxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (130, 130, 130), 1)
    for x1, y1, x2, y2 in ctx.boxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (80, 180, 255), 1)
    for hand in ctx.hands:
        cv2.circle(frame, (int(hand[0]), int(hand[1])), 5, (0, 165, 255), 2)
        if ctx.hit_radius > 0:
            cv2.circle(frame, (int(hand[0]), int(hand[1])), int(ctx.hit_radius), (0, 165, 255), 1)
    return frame
