import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from model_v38 import BallTrackerNetV38


def conv_macs(model, sample):
    total = 0
    rows = []
    hooks = []

    def hook(name):
        def count(module, inputs, output):
            nonlocal total
            batch, out_channels, out_height, out_width = output.shape
            kernel_height, kernel_width = module.kernel_size
            operations = (
                batch
                * out_channels
                * out_height
                * out_width
                * (module.in_channels // module.groups)
                * kernel_height
                * kernel_width
            )
            total += int(operations)
            rows.append({"layer": name, "macs": int(operations), "output": list(output.shape)})
        return count

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(hook(name)))
    with torch.inference_mode():
        model(sample)
    for item in hooks:
        item.remove()
    return total, rows


def ca_nonconv_operations(height=90, width=160, channels=96):
    # Approximate scalar operations outside CA's 1x1 convolutions. They are
    # reported separately and are not folded into the conventional Conv MAC.
    reduce_adds = channels * (height * (width - 1) + width * (height - 1))
    reduce_scales = channels * (height + width)
    sigmoid_elements = channels * (height + width)
    gate_multiplies = 2 * channels * height * width
    return {
        "directional_reduce_adds": reduce_adds,
        "directional_mean_scales": reduce_scales,
        "sigmoid_elements": sigmoid_elements,
        "gate_multiplies": gate_multiplies,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure V3.7/V3.8 parameters and convolution MACs.")
    parser.add_argument("--output", default="exps/v38_essay_ablation/complexity.json")
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()
    report = {"input": [1, 9, args.height, args.width], "mac_definition": "multiply-accumulate per Conv2d output"}
    for name, use_ca in (("v37_backbone", False), ("v38_ca", True)):
        model = BallTrackerNetV38(use_ca=use_ca).eval()
        sample = torch.zeros(1, 9, args.height, args.width)
        macs, rows = conv_macs(model, sample)
        report[name] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "conv_macs": macs,
            "conv_gmac": macs / 1e9,
            "layers": rows,
        }
        if use_ca:
            report[name]["ca_nonconv_operations"] = ca_nonconv_operations(args.height // 4, args.width // 4, 96)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("v37_backbone", "v38_ca")}, indent=2))
    for name in ("v37_backbone", "v38_ca"):
        print(name, "parameters", report[name]["parameters"], "conv_gmac", f"{report[name]['conv_gmac']:.6f}")
    print(f"output={output.resolve()}")


if __name__ == "__main__":
    main()
