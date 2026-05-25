import argparse
import os

import torch
import torch.nn as nn

from model_v3 import BallTrackerNetV3


class SigmoidExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.sigmoid(self.model(x))


def export():
    parser = argparse.ArgumentParser(description="Export TrackNet V3 heatmap model to ONNX.")
    parser.add_argument("--model-path", default="./exps/lite_heatmap_v3_180x320/model_best.pt")
    parser.add_argument("--onnx-path", default="./exps/lite_heatmap_v3_180x320/model_v3_b1.onnx")
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-height", type=int, default=180)
    parser.add_argument("--input-width", type=int, default=320)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dynamic-batch", action="store_true", help="export dynamic batch axis")
    parser.add_argument("--no-sigmoid", action="store_true", help="export raw logits instead of 0-1 heatmap")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print("error: weight file not found: {}".format(args.model_path))
        return

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BallTrackerNetV3(base_channels=args.base_channels).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    export_model = model if args.no_sigmoid else SigmoidExportWrapper(model)
    export_model.eval()

    out_dir = os.path.dirname(args.onnx_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dummy_input = torch.randn(args.batch_size, 9, args.input_height, args.input_width).to(device)
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"input": {0: "batch_size"}, "heatmap": {0: "batch_size"}}

    torch.onnx.export(
        export_model,
        dummy_input,
        args.onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["heatmap"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    print("ONNX export success: {}".format(args.onnx_path))
    print(
        "input shape: [{}, 9, {}, {}]".format(args.batch_size, args.input_height, args.input_width)
    )
    print(
        "output shape: [{}, 1, {}, {}]{}".format(
            args.batch_size,
            args.input_height,
            args.input_width,
            " logits" if args.no_sigmoid else " heatmap",
        )
    )


if __name__ == "__main__":
    export()
