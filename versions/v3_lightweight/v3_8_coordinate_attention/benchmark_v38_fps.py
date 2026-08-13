import argparse
import json
import platform
import statistics
from pathlib import Path

import torch

from model_v38 import BallTrackerNetV38


def benchmark(model, sample, warmup, iterations, repeats):
    with torch.inference_mode():
        for _ in range(warmup):
            torch.sigmoid(model(sample))
    torch.cuda.synchronize()
    runs = []
    with torch.inference_mode():
        for index in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                torch.sigmoid(model(sample))
            end.record()
            end.synchronize()
            milliseconds = float(start.elapsed_time(end)) / iterations
            runs.append({"run": index + 1, "ms_per_frame": milliseconds, "fps": 1000.0 / milliseconds})
    return {
        "runs": runs,
        "mean_ms": statistics.mean(item["ms_per_frame"] for item in runs),
        "std_ms": statistics.stdev(item["ms_per_frame"] for item in runs) if len(runs) > 1 else 0.0,
        "mean_fps": statistics.mean(item["fps"] for item in runs),
        "std_fps": statistics.stdev(item["fps"] for item in runs) if len(runs) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Paired local V3.7/V3.8 model-only CUDA FPS benchmark.")
    parser.add_argument("--output", default="exps/v38_essay_ablation/fps_model_only.json")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the paired paper FPS benchmark")
    torch.backends.cudnn.benchmark = True
    report = {
        "protocol": {
            "batch_size": 1,
            "input": [1, 9, args.height, args.width],
            "warmup": args.warmup,
            "iterations_per_repeat": args.iterations,
            "repeats": args.repeats,
            "includes": "network forward and sigmoid",
            "excludes": "decode, resize, H2D copy, D2H copy, peak extraction, video encoding",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    for dtype_name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        for graph_name, use_ca in (("v37_backbone", False), ("v38_ca", True)):
            torch.manual_seed(37)
            model = BallTrackerNetV38(use_ca=use_ca).cuda().eval().to(dtype=dtype)
            sample = torch.randn(1, 9, args.height, args.width, device="cuda", dtype=dtype)
            key = f"{graph_name}_{dtype_name}"
            report[key] = benchmark(model, sample, args.warmup, args.iterations, args.repeats)
            print(key, f"{report[key]['mean_ms']:.4f} ms", f"{report[key]['mean_fps']:.2f} FPS", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"output={output.resolve()}")


if __name__ == "__main__":
    main()
