import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
V37_SCRIPT = ROOT / "versions" / "v3_lightweight" / "v3_7_tracknet_ball_hardneg" / "infer_video_v37_adaptive.py"
V51_SCRIPT = ROOT / "versions" / "v5_1_reference_pipeline" / "apply_reference_assist.py"
DEFAULT_MODEL = (
    ROOT
    / "exps"
    / "lite_heatmap_v37_clean_hardneg_batch8_360x640"
    / "snapshots"
    / "best_before_error_audit_20260613_191025"
    / "model_best_f1.pt"
)


def run(command):
    print("running:", " ".join(str(item) for item in command), flush=True)
    subprocess.run([str(item) for item in command], cwd=ROOT, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end V3.7 tracking and V5.1 bounce-event pipeline.")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--output-dir", default="exps/v5_1_v37")
    parser.add_argument("--prefix", default="v37_v51")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--player-crop-json",
        default="configs/player_crops/broadcast_court_only.json",
    )
    parser.add_argument("--disable-mediapipe", action="store_true")
    parser.add_argument("--disable-auto-court", action="store_true")
    parser.add_argument("--bounce-threshold", type=float, default=0.55)
    parser.add_argument("--contact-threshold", type=float, default=0.18)
    parser.add_argument("--min-event-gap-seconds", type=float, default=0.60)
    parser.add_argument("--min-flight-seconds", type=float, default=0.15)
    parser.add_argument("--min-event-gap", type=int, default=None, help="Deprecated fixed-frame override.")
    parser.add_argument("--min-flight-frames", type=int, default=None, help="Deprecated fixed-frame override.")
    parser.add_argument("--disable-serve-toss-suppression", action="store_true")
    parser.add_argument("--codec", default="mp4v")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    v37_video = output_dir / f"{args.prefix}_v37_track.mp4"
    v37_csv = output_dir / f"{args.prefix}_v37_track.csv"
    final_video = output_dir / f"{args.prefix}_bounce.mp4"
    final_track = output_dir / f"{args.prefix}_context_track.csv"
    final_events = output_dir / f"{args.prefix}_events.csv"

    run(
        [
            sys.executable,
            V37_SCRIPT,
            "--model_path",
            args.model_path,
            "--video_path",
            args.video_path,
            "--video_out_path",
            v37_video,
            "--csv_out_path",
            v37_csv,
            "--device",
            args.device,
        ]
    )

    event_command = [
        sys.executable,
        V51_SCRIPT,
        "--event-mode",
        "v37_bounce",
        "--video-path",
        args.video_path,
        "--track-csv-in",
        v37_csv,
        "--track-csv-out",
        final_track,
        "--event-csv-out",
        final_events,
        "--video-out-path",
        final_video,
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
        "--draw-mini-court",
    ]
    if args.min_event_gap is not None:
        event_command.extend(["--min-event-gap", args.min_event_gap])
    if args.min_flight_frames is not None:
        event_command.extend(["--min-flight-frames", args.min_flight_frames])
    if args.disable_serve_toss_suppression:
        event_command.append("--disable-serve-toss-suppression")
    if not args.disable_mediapipe:
        event_command.extend(["--use-mediapipe", "--player-crop-json", args.player_crop_json])
    if not args.disable_auto_court:
        event_command.extend(["--auto-court-lines", "--court-detect-stride", "15"])
    run(event_command)

    print(f"v37_track_video={v37_video}", flush=True)
    print(f"v37_track_csv={v37_csv}", flush=True)
    print(f"final_video={final_video}", flush=True)
    print(f"final_track_csv={final_track}", flush=True)
    print(f"event_csv={final_events}", flush=True)


if __name__ == "__main__":
    main()
