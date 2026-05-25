# 基于 TrackNet 的网球轨迹跟踪与轻量化部署

本项目面向网球比赛视频分析与边缘端智能感知场景，目标是在视频中完成网球球心定位、轨迹生成、结果可视化和嵌入式平台轻量化部署验证。项目以 TrackNet 的时序热图跟踪思路为基础，逐步形成 V1、V2、V3 三个版本，为后续商业化轨迹分析、自动判罚和训练辅助系统提供技术底座。

## 项目路线

```text
V1 原始 TrackNet:
  9 通道连续帧输入，256 类逐像素分类输出。
  精度较好，但输出头和后处理对 RK3588 不友好。

V2 单通道 heatmap:
  保留 9 通道连续帧输入，将输出改为 [B, 1, 360, 640]。
  去掉 256 类输出和 ArgMax，提升部署友好性。

V3 轻量化 heatmap:
  使用 Depthwise Separable Conv，并降低输入分辨率。
  当前推荐版本为 270x480，兼顾精度与嵌入式推理速度。
```

## 目录结构

```text
versions/
  v1_original/      原始 TrackNet 基线代码
  v2_heatmap/       V2 单通道 heatmap 版本
  v3_lightweight/   V3 轻量化部署版本

tracknet_webui/     本地 Web 控制台
datasets/           数据集和标签
exps/               训练权重、ONNX、推理输出和日志
```

当前主推模型：

```text
PT:
exps/lite_heatmap_v3_270x480_from240/model_best_thr070_pw15.pt

ONNX:
exps/lite_heatmap_v3_270x480_from240/model_v3_270x480_b1_sigmoid.onnx
```

当前 V3 270x480 验证集记录：

```text
Precision = 0.9252
Recall    = 0.8930
F1        = 0.9088
```

模型规模：

```text
参数量: 43,132
输入: [B, 9, 270, 480]
输出: [B, 1, 270, 480]
计算量: 约 1.057 GMAC / 帧
```

## 环境安装

建议使用 Python 3.11 和 CUDA 版 PyTorch。当前开发环境使用 RTX 3060 Laptop GPU，PyTorch 为 CUDA 版。

```powershell
python -m pip install -r requirements.txt
```

如果 PyTorch 安装失败，可先单独安装 CUDA 版 PyTorch：

```powershell
python -m pip install --no-cache-dir torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```

CPU 环境也可以运行推理和部分测试，但训练速度会很慢。

## 数据集

V2/V3 使用如下数据结构：

```text
datasets/trackNet/
  images/
    game1/
      Clip1/
        0000.jpg
        Label.csv
  labels_train.csv
  labels_val.csv
```

当前项目中已生成：

```text
labels_train.csv: 13,751 条
labels_val.csv:   5,894 条
```

如需重新从压缩包准备数据：

```powershell
python versions/v2_heatmap/prepare_dataset_v2.py
```

## 训练

### V2 训练

```powershell
python versions/v2_heatmap/main_v2.py --num_epochs 300 --batch_size 2 --steps_per_epoch 200 --augment --amp --device cuda
```

V2 的优点是精度较高，缺点是 360x640 分辨率下计算量仍然较大，约 37.8 GMAC / 帧，不适合作为 RK3588 的最终实时部署版本。

### V3 训练

推荐从当前最优配置继续：

```powershell
python versions/v3_lightweight/main_v3.py ^
  --exp_id lite_heatmap_v3_270x480_from240 ^
  --num_epochs 100 ^
  --batch_size 4 ^
  --steps_per_epoch 200 ^
  --input_height 270 ^
  --input_width 480 ^
  --heatmap_radius 6 ^
  --heatmap_sigma 2.25 ^
  --threshold 0.70 ^
  --peak_window 15 ^
  --pos_weight 120 ^
  --amp ^
  --device cuda
```

V3 会保存：

```text
model_last.pt
model_best.pt
training_state.pt
model_epoch_XXX.pt
```

中断后可以用 `training_state.pt` 恢复训练：

```powershell
python versions/v3_lightweight/main_v3.py --resume .\exps\lite_heatmap_v3_270x480_from240\training_state.pt --amp --device cuda
```

## 阈值测试

```powershell
python versions/v3_lightweight/eval_thresholds_v3.py ^
  --model-path .\exps\lite_heatmap_v3_270x480_from240\model_best_thr070_pw15.pt ^
  --input-height 270 ^
  --input-width 480 ^
  --heatmap-radius 6 ^
  --heatmap-sigma 2.25 ^
  --peak-window 15 ^
  --thresholds 0.60,0.65,0.68,0.70,0.72,0.75,0.80,0.85,0.90 ^
  --batch-size 4 ^
  --device cuda
```

## 视频推理

PT 推理：

```powershell
python versions/v3_lightweight/infer_on_video_v3_batch.py ^
  --model_path .\exps\lite_heatmap_v3_270x480_from240\model_best_thr070_pw15.pt ^
  --video_path .\示例视频1.mp4 ^
  --video_out_path .\exps\lite_heatmap_v3_270x480_from240\demo_v3_pt.mp4 ^
  --csv_out_path .\exps\lite_heatmap_v3_270x480_from240\demo_v3_pt.csv ^
  --input_height 270 ^
  --input_width 480 ^
  --threshold 0.70 ^
  --peak_window 15 ^
  --batch_size 1 ^
  --codec mp4v ^
  --device cuda
```

ONNX 推理：

```powershell
python versions/v3_lightweight/infer_on_video_v3_onnx.py ^
  --onnx_path .\exps\lite_heatmap_v3_270x480_from240\model_v3_270x480_b1_sigmoid.onnx ^
  --video_path .\示例视频1.mp4 ^
  --video_out_path .\exps\lite_heatmap_v3_270x480_from240\demo_v3_onnx.mp4 ^
  --csv_out_path .\exps\lite_heatmap_v3_270x480_from240\demo_v3_onnx.csv ^
  --input_height 270 ^
  --input_width 480 ^
  --threshold 0.70 ^
  --peak_window 15 ^
  --codec mp4v ^
  --target cpu
```

## ONNX 导出

导出 batch1 ONNX：

```powershell
python versions/v3_lightweight/export_onnx_v3.py ^
  --model-path .\exps\lite_heatmap_v3_270x480_from240\model_best_thr070_pw15.pt ^
  --onnx-path .\exps\lite_heatmap_v3_270x480_from240\model_v3_270x480_b1_sigmoid.onnx ^
  --input-height 270 ^
  --input-width 480 ^
  --batch-size 1 ^
  --device cuda
```

如需固定 batch4：

```powershell
python versions/v3_lightweight/export_onnx_v3.py ^
  --model-path .\exps\lite_heatmap_v3_270x480_from240\model_best_thr070_pw15.pt ^
  --onnx-path .\exps\lite_heatmap_v3_270x480_from240\model_v3_270x480_b4_sigmoid.onnx ^
  --input-height 270 ^
  --input-width 480 ^
  --batch-size 4 ^
  --device cuda
```

## WebUI

启动本地控制台：

```powershell
python .\tracknet_webui\server.py
```

访问：

```text
http://127.0.0.1:8765
```

WebUI 可用于上传视频、调用 PT/ONNX 推理、运行阈值测试和继续训练。

## 方案说明

本项目不是通用网球类别识别器，而是时序球心定位与轨迹跟踪系统。模型关注的是“球最可能在哪里”，而不是“画面中某个目标是否属于网球类别”。这种设计更适合高速、小尺寸、运动模糊的网球视频，也更符合 RK3588 等边缘设备的轻量化部署需求。

更完整的技术路线见：

```text
TECHNICAL_DOCUMENTATION.md
```
