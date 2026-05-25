import argparse
import os

import torch
import torch.nn as nn

from model_v2 import BallTrackerNetLite


class SigmoidExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.sigmoid(self.model(x))


def export():
    parser = argparse.ArgumentParser(description="Export TrackNet Lite V2 heatmap model to ONNX.")
    parser.add_argument("--model-path", default="./exps/lite_heatmap_v2/model_best.pt")
    parser.add_argument("--onnx-path", default="./exps/lite_heatmap_v2/model_best_v2.onnx")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-sigmoid", action="store_true", help="export raw logits instead of 0-1 heatmap")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print("error: weight file not found: {}".format(args.model_path))
        return

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BallTrackerNetLite(base_channels=args.base_channels).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    export_model = model if args.no_sigmoid else SigmoidExportWrapper(model)
    export_model.eval()

    out_dir = os.path.dirname(args.onnx_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dummy_input = torch.randn(1, 9, 360, 640).to(device)
    torch.onnx.export(
        export_model,
        dummy_input,
        args.onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["heatmap"],
        dynamic_axes={"input": {0: "batch_size"}, "heatmap": {0: "batch_size"}},
        dynamo=False,
    )
    print("ONNX export success: {}".format(args.onnx_path))
    print("output shape: [batch, 1, 360, 640]{}".format(" logits" if args.no_sigmoid else " heatmap"))


if __name__ == "__main__":
    export()
