import argparse
import subprocess
import sys
from pathlib import Path


V52_DIR = Path(__file__).resolve().parent
ROOT = V52_DIR.parents[1]
TRACK_SCRIPT = V52_DIR / "tracking" / "track_video.py"
EVENT_SCRIPT = V52_DIR / "pipeline" / "recognize_events.py"
DEFAULT_MODEL = (
    ROOT
    / "exps"
    / "lite_heatmap_v37_clean_hardneg_batch8_360x640"
    / "snapshots"
    / "best_before_error_audit_20260613_191025"
    / "model_best_f1.pt"
)
DEFAULT_PLAYER_CROPS = ROOT / "configs" / "player_crops" / "broadcast_court_only.json"


def run(command):
    print("running:", " ".join(str(item) for item in command), flush=True)
    subprocess.run([str(item) for item in command], cwd=ROOT, check=True)


def resolve_from_root(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(
        description="V5.2 self-contained tennis tracking and bounce-recognition pipeline."
    )
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--output-dir", default="exps/v5_2")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--input-height", type=int, default=360)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--bounce-threshold", type=float, default=0.55)
    parser.add_argument("--contact-threshold", type=float, default=0.18)
    parser.add_argument("--min-event-gap-seconds", type=float, default=0.60)
    parser.add_argument("--min-flight-seconds", type=float, default=0.15)
    parser.add_argument("--player-crop-json", default=str(DEFAULT_PLAYER_CROPS))
    parser.add_argument("--court-detect-stride", type=int, default=15)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--disable-mediapipe", action="store_true")
    parser.add_argument("--disable-auto-court", action="store_true")
    parser.add_argument("--disable-mini-court", action="store_true")
    parser.add_argument("--disable-serve-toss-suppression", action="store_true")
    parser.add_argument("--disable-fps-adaptive", action="store_true")
    parser.add_argument("--disable-contact-timing-refinement", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = resolve_from_root(args.video_path).resolve()
    model_path = resolve_from_root(args.model_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    prefix = args.prefix or video_path.stem
    output_root = resolve_from_root(args.output_dir) / prefix
    tracking_dir = output_root / "tracking"
    event_dir = output_root / "events"
    cache_dir = output_root / "cache"
    for path in (tracking_dir, event_dir, cache_dir):
        path.mkdir(parents=True, exist_ok=True)

    track_video = tracking_dir / "trajectory.mp4"
    track_csv = tracking_dir / "trajectory.csv"
    context_csv = event_dir / "context_trajectory.csv"
    event_csv = event_dir / "bounce_events.csv"
    result_video = output_root / "result.mp4"

    track_command = [
        sys.executable,
        TRACK_SCRIPT,
        "--model_path",
        model_path,
        "--video_path",
        video_path,
        "--video_out_path",
        track_video,
        "--csv_out_path",
        track_csv,
        "--device",
        args.device,
        "--batch_size",
        args.batch_size,
        "--input_height",
        args.input_height,
        "--input_width",
        args.input_width,
        "--codec",
        args.codec,
    ]
    if not args.disable_fps_adaptive:
        track_command.append("--fps_adaptive")
    run(track_command)

    event_command = [
        sys.executable,
        EVENT_SCRIPT,
        "--event-mode",
        "v37_bounce",
        "--video-path",
        video_path,
        "--track-csv-in",
        track_csv,
        "--track-csv-out",
        context_csv,
        "--event-csv-out",
        event_csv,
        "--video-out-path",
        result_video,
        "--bounce-threshold",
        args.bounce_threshold,
        "--contact-threshold",
        args.contact_threshold,
        "--min-event-gap-seconds",
        args.min_event_gap_seconds,
        "--min-flight-seconds",
        args.min_flight_seconds,
        "--codec",
        args.codec,
        "--player-context-cache",
        cache_dir / "player_context.pkl",
    ]
    if not args.disable_mediapipe:
        event_command.extend(
            [
                "--use-mediapipe",
                "--player-crop-json",
                resolve_from_root(args.player_crop_json).resolve(),
            ]
        )
    if not args.disable_auto_court:
        event_command.extend(
            ["--auto-court-lines", "--court-detect-stride", args.court_detect_stride]
        )
    if not args.disable_mini_court:
        event_command.append("--draw-mini-court")
    if args.disable_serve_toss_suppression:
        event_command.append("--disable-serve-toss-suppression")
    if args.disable_contact_timing_refinement:
        event_command.append("--disable-contact-timing-refinement")
    run(event_command)

    print(f"result_video={result_video}", flush=True)
    print(f"trajectory_video={track_video}", flush=True)
    print(f"trajectory_csv={track_csv}", flush=True)
    print(f"context_trajectory_csv={context_csv}", flush=True)
    print(f"bounce_events_csv={event_csv}", flush=True)


if __name__ == "__main__":
    main()
