import torch
from model import BallTrackerNet
import os

def export():
    # 1. 确保权重文件路径正确
    pt_path = '/workspace/model_best.pt'
    if not os.path.exists(pt_path):
        print(f"❌ 错误: 找不到权重文件 {pt_path}")
        return

    # 2. 初始化模型并加载权重
    model = BallTrackerNet().cuda()
    device = torch.device('cuda')
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()

    # 3. 准备 dummy 输入 (TrackNet 是 9 通道输入: 3帧 * 3通道)
    dummy_input = torch.randn(1, 9, 360, 640).cuda()

    # 4. 导出 (将 opset 改为 11)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            "/workspace/model_best.onnx",
            export_params=True,
            opset_version=11,  # 修改此处：从 17 改为 11
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("✅ ONNX 导出成功！文件位置: /workspace/model_best.onnx")
    except Exception as e:
        print(f"❌ 导出失败: {e}")

if __name__ == "__main__":
    export()