import argparse
import csv
import importlib
import json
import platform
from pathlib import Path

import torch


EXPECTED_SPLIT_ROWS = {"train": 11970, "valid": 2060, "test": 5615}


def csv_row_count(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def main():
    parser = argparse.ArgumentParser(description="Validate the V3.8 NVIDIA training environment and data layout.")
    parser.add_argument("--tracknet-root", default="datasets/trackNet")
    parser.add_argument("--split-root", default="datasets/tracknet_v38_match_split")
    parser.add_argument(
        "--mapped-csv",
        default="datasets/tennis_all_v4i_mapped/annotations_hardneg_cleaned_strict.csv",
    )
    parser.add_argument("--skip-data", action="store_true")
    args = parser.parse_args()

    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "dependencies": {},
        "data": {},
        "errors": [],
    }
    for name in ("numpy", "pandas", "cv2", "scipy"):
        try:
            module = importlib.import_module(name)
            report["dependencies"][name] = getattr(module, "__version__", "unknown")
        except Exception as error:  # pragma: no cover - diagnostic utility
            report["errors"].append(f"cannot import {name}: {error!r}")

    if torch.cuda.is_available():
        report["cuda_device_count"] = torch.cuda.device_count()
        report["cuda_device_0"] = torch.cuda.get_device_name(0)
        report["cudnn"] = torch.backends.cudnn.version()
        try:
            sample = torch.randn(2, 3, 32, 32, device="cuda", dtype=torch.float16)
            layer = torch.nn.Conv2d(3, 8, 3, padding=1, device="cuda", dtype=torch.float16)
            report["cuda_fp16_smoke"] = list(layer(sample).shape)
        except Exception as error:  # pragma: no cover - diagnostic utility
            report["errors"].append(f"CUDA FP16 convolution failed: {error!r}")
    else:
        report["errors"].append("torch.cuda.is_available() is False")

    if not args.skip_data:
        tracknet_root = Path(args.tracknet_root)
        split_root = Path(args.split_root)
        mapped_csv = Path(args.mapped_csv)
        jpg_count = len(list(tracknet_root.glob("game*/Clip*/*.jpg"))) if tracknet_root.exists() else 0
        report["data"]["tracknet_jpg_count"] = jpg_count
        if jpg_count != 19835:
            report["errors"].append(f"expected 19835 TrackNet JPGs, found {jpg_count}")
        for split, expected in EXPECTED_SPLIT_ROWS.items():
            path = split_root / f"{split}.csv"
            actual = csv_row_count(path) if path.exists() else -1
            report["data"][f"{split}_rows"] = actual
            if actual != expected:
                report["errors"].append(f"expected {expected} rows in {path}, found {actual}")
        report["data"]["mapped_csv_exists"] = mapped_csv.exists()
        if not mapped_csv.exists():
            report["errors"].append(f"missing mapped annotations: {mapped_csv}")

    print(json.dumps(report, indent=2))
    if report["errors"]:
        raise SystemExit(1)
    print("V3.8 server environment check passed.")


if __name__ == "__main__":
    main()
