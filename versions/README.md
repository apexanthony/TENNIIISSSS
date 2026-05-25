# 版本代码目录

项目代码已按模型版本整理：

```text
v1_original/
  原始 TrackNet 基线代码和旧版预训练权重。

v2_heatmap/
  单通道 heatmap 版本。保留 360x640 输入分辨率，将原始 256 类逐像素输出
  改为 [B, 1, H, W] heatmap 输出。

v3_lightweight/
  轻量化部署版本。使用深度可分离卷积、低分辨率输入、ONNX 友好 heatmap
  输出，并提供视频 batch 推理工具。
```

请从项目根目录运行脚本，例如：

```powershell
python versions/v3_lightweight/eval_thresholds_v3.py --model-path .\exps\lite_heatmap_v3_270x480_from240\model_best_thr070_pw15.pt --input-height 270 --input-width 480
```

保持工作目录在项目根目录，可以确保 `datasets/trackNet` 和 `exps/` 等路径正确解析。
