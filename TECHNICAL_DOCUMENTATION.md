# 项目技术文档：基于 TrackNet 的网球轨迹跟踪与轻量化部署

## 1. 项目目标

本项目面向网球比赛视频中的高速小目标跟踪任务，目标是在视频中定位网球球心、生成运动轨迹，并将模型轻量化后部署到 RK3588 等边缘计算平台。

项目目标可以概括为：

```text
1. 基于机器视觉实现网球球心定位和轨迹跟踪。
2. 分析原始 TrackNet 在嵌入式平台上的性能瓶颈。
3. 对模型输出头、网络结构和输入分辨率进行轻量化改进。
4. 导出 ONNX / RKNN 友好的模型，为 RK3588 部署提供基础。
5. 在精度和速度之间寻找适合边缘设备的折中方案。
```

本项目不以通用目标检测为核心，不额外引入 YOLO 等网球类别识别模块。系统关注的是球心位置和轨迹连续性，而不是对画面中目标进行通用语义分类。

当前代码按版本组织：

```text
versions/v1_original/      原始 TrackNet 基线
versions/v2_heatmap/       单通道 heatmap 改进版本
versions/v3_lightweight/   轻量化部署版本
tracknet_webui/            本地 Web 控制台
datasets/                  数据集与标签
exps/                      实验权重、ONNX 和推理输出
```

## 2. TrackNet 方法基础

TrackNet 类方法通常使用连续多帧图像作为输入，通过卷积神经网络输出球心热图或逐像素分类结果。其核心思想是利用：

```text
1. 网球在连续帧之间的运动连续性。
2. 小目标在局部区域内的亮度、形状和模糊特征。
3. 比赛场景中网球运动轨迹的统计规律。
```

与 YOLO 等目标检测器不同，TrackNet 不直接输出类别和检测框，而是输出球心概率分布。后处理时取热图峰值作为球心坐标。

因此，TrackNet 更像：

```text
专用场景下的高速小目标轨迹定位器
```

而不是：

```text
通用网球类别识别器
```

这种设计适合网球、羽毛球、乒乓球等高速小目标场景。

## 3. 为什么不加入独立网球识别模块

项目没有引入额外网球识别模型，主要原因如下。

第一，识别功能与主任务相关性有限。系统最终需要的是球心坐标和轨迹，而不是判断画面中某个目标是否属于“网球”类别。独立识别模块对最终轨迹输出提升有限。

第二，额外识别模型会增加计算量、显存占用和部署复杂度。RK3588 这类边缘设备算力有限，额外加入 YOLO 或分类模型会降低实时性。

第三，网球高速运动时通常存在尺寸小、边界模糊、运动拖影明显等问题。YOLO 等检测器依赖目标外观、边界框和纹理特征，在高速模糊场景下未必比热图跟踪更稳定。

第四，目标检测方案通常需要额外标注 bbox，而当前 TrackNet 方案只需要球心坐标。额外标注会增加数据准备成本，并可能引入小目标框标注误差。

因此，本项目采用：

```text
连续帧输入 + 球心热图输出 + 轨迹后处理
```

而不是：

```text
目标检测模型 + 类别识别 + bbox 后处理
```

## 4. 版本演进路线

### 4.1 V1：原始 TrackNet 基线

V1 对应原始 TrackNet 思路，主要文件包括：

```text
versions/v1_original/model.py
versions/v1_original/main.py
versions/v1_original/infer_on_video.py
versions/v1_original/general.py
versions/v1_original/datasets.py
```

原始模型使用连续 3 帧 RGB 图像作为输入：

```text
3 帧 RGB = 9 通道输入
输入形状: [B, 9, 360, 640]
```

其输出本质上接近：

```text
[B, 256, 360 * 640]
```

然后再通过 ArgMax 还原热图或类别索引。

V1 的问题：

```text
1. 输出头过重，256 类逐像素分类对 RKNN/NPU 不友好。
2. 后处理需要 ArgMax，增加 CPU 侧开销。
3. 模型结构较重，嵌入式推理速度较低。
4. 原始训练流程依赖 gt 图像，数据准备成本较高。
```

V1 的意义是作为工程基线模型，用于验证 TrackNet 的基本原理，并为后续轻量化改造提供对照。

### 4.2 V2：单通道热图输出版本

V2 的核心改进是重构输出头，将原始 256 类逐像素分类改为单通道热图回归。

主要文件：

```text
versions/v2_heatmap/model_v2.py
versions/v2_heatmap/main_v2.py
versions/v2_heatmap/general_v2.py
versions/v2_heatmap/datasets_v2.py
versions/v2_heatmap/infer_on_video_v2.py
versions/v2_heatmap/infer_on_video_v2_stream.py
versions/v2_heatmap/export_onnx_v2.py
versions/v2_heatmap/eval_thresholds_v2.py
```

V2 输入：

```text
[B, 9, 360, 640]
```

V2 输出：

```text
[B, 1, 360, 640]
```

训练标签由球心坐标在线生成高斯热图，不再强依赖预生成 gt 图片。

损失函数：

```text
BCEWithLogitsLoss + MSELoss
```

后处理：

```text
sigmoid -> heatmap peak -> 阈值判断 -> 球心坐标映射
```

V2 的主要收益：

```text
1. 去掉 256 类输出头。
2. 去掉 ArgMax 类后处理。
3. ONNX 输出直接为 [B, 1, H, W] heatmap。
4. 更适合 RKNN 转换和 C/C++ 部署。
5. 精度保持较好。
```

V2 当前较优结果：

```text
epoch 331, threshold = 0.95
Precision = 0.9416
Recall    = 0.9095
F1        = 0.9253
```

V2 的问题：

```text
输入分辨率仍为 360x640。
普通卷积仍然较多。
计算量约 37.8 GMAC / 帧。
在 RK3588 上仍然偏重。
```

因此，V2 适合作为“输出头轻量化”阶段成果，但不是最终嵌入式实时版本。

### 4.3 V3：轻量化部署版本

V3 的目标是显著降低计算量，使模型更接近 RK3588 部署需求。

主要文件：

```text
versions/v3_lightweight/model_v3.py
versions/v3_lightweight/main_v3.py
versions/v3_lightweight/general_v3.py
versions/v3_lightweight/infer_on_video_v3_batch.py
versions/v3_lightweight/infer_on_video_v3_onnx.py
versions/v3_lightweight/export_onnx_v3.py
versions/v3_lightweight/eval_thresholds_v3.py
```

V3 的核心改动：

```text
1. 输入分辨率从 360x640 降低到 180x320 / 240x426 / 270x480。
2. 普通卷积替换为 Depthwise Separable Conv。
3. 保留单通道 heatmap 输出。
4. 保留 3 帧 RGB 时序输入，即 9 通道输入。
5. 支持固定 batch ONNX 导出。
6. 支持训练断点恢复和 training_state.pt。
```

当前主推 V3 配置：

```text
输入: [B, 9, 270, 480]
输出: [B, 1, 270, 480]
base_channels: 24
heatmap_radius: 6
heatmap_sigma: 2.25
threshold: 0.70 / 0.85 可按场景调整
peak_window: 15
```

V3 270x480 模型规模：

```text
参数量: 43,132
模型大小: 约 0.165 MB fp32
计算量: 约 1.057 GMAC / 帧
```

与 V2 对比：

```text
V2 360x640: 约 37.8 GMAC / 帧
V3 270x480: 约 1.057 GMAC / 帧
计算量下降约 35 倍
```

当前 V3 270x480 验证集记录：

```text
threshold = 0.85
Precision = 0.9252
Recall    = 0.8930
F1        = 0.9088
```

V3 的意义：

```text
V3 是当前最适合部署的版本。
它牺牲少量精度，显著降低计算量。
```

## 5. 数据处理流程

V2/V3 数据集读取由 `versions/v2_heatmap/datasets_v2.py` 和 `versions/v3_lightweight/datasets_v2.py` 完成。两个版本各保留一份数据集读取代码，便于从项目根目录直接运行对应版本脚本。

每个样本读取连续 3 帧：

```text
path1: 当前帧
path2: 前一帧
path3: 前两帧
```

然后进行：

```text
1. resize 到模型输入尺寸。
2. 三帧 RGB 在通道维拼接。
3. 归一化到 [0, 1]。
4. 转为 [9, H, W]。
```

标签处理：

```text
1. 读取球心坐标 x, y 和 visibility。
2. 按原图尺寸映射到模型输入尺寸。
3. 在线生成高斯热图。
4. 输出 [1, H, W]。
```

训练增强包括：

```text
亮度/对比度扰动
运动模糊
JPEG 压缩噪声
高斯噪声
```

这些增强用于模拟比赛视频中的压缩、光照变化和高速运动模糊。

## 6. 训练与评估指标

项目使用 Precision、Recall、F1 评价球心定位效果。

判断逻辑：

```text
1. 模型输出 heatmap。
2. 后处理得到预测球心。
3. 若预测球心与真实球心距离小于 min_dist，则记为 TP。
4. 有球但未检出，记为 FN。
5. 无球但检出，或球心偏差过大，记为 FP。
```

指标含义：

```text
Precision: 检出的点有多少是正确球心。
Recall: 真实有球帧中有多少被检测到。
F1: Precision 和 Recall 的综合平衡。
```

在轨迹任务中，Recall 过低会导致轨迹断裂；Precision 过低会导致误检跳点。因此需要结合阈值、轨迹连续性和后处理共同调整。

## 7. 推理流程

V3 推理流程如下：

```text
1. 从视频中读取连续帧。
2. 组成滑动窗口:
   frame t-2, frame t-1, frame t
3. resize 并拼接成 9 通道输入。
4. 模型输出 heatmap。
5. sigmoid 得到概率图。
6. 取峰值点作为球心。
7. 将低分辨率坐标映射回原视频分辨率。
8. 绘制轨迹并输出 CSV。
```

batch 推理逻辑：

```text
样本1: frame 0,1,2
样本2: frame 1,2,3
样本3: frame 2,3,4
样本4: frame 3,4,5
```

这样可以在离线视频处理时提高吞吐。但 batch 不会降低单帧计算量，只是减少模型调用开销并提高硬件利用率。

## 8. ONNX 与 RK3588 部署思路

V3 ONNX 默认导出 sigmoid 后的 heatmap：

```text
input:  [1, 9, 270, 480]
output: [1, 1, 270, 480]
```

这种输出形式适合 RKNN：

```text
1. 输出通道少。
2. 后处理简单。
3. 不需要 256 类 ArgMax。
4. C/C++ 侧只需找热图最大值和坐标映射。
```

RK3588 部署建议：

```text
1. 优先转换 V3 270x480 ONNX。
2. 分别测试 batch1、batch2、batch4。
3. 实时场景优先 batch1 或 batch2。
4. 离线处理视频可尝试 batch4。
5. 预处理尽量在 C/C++ 中完成，避免 Python 和 OpenCV 写视频开销影响判断。
```

需要注意：RK3588 可以处理 9 通道输入，但 RKNN 的内置图像预处理通常更适合 3 通道图像。因此部署时建议手动构造 9 通道输入 buffer，再送入 `rknn_inputs_set`。

## 9. 当前推荐配置

项目演示和部署验证优先使用：

```text
模型:
exps/lite_heatmap_v3_270x480_from240/model_best_thr070_pw15.pt

ONNX:
exps/lite_heatmap_v3_270x480_from240/model_v3_270x480_b1_sigmoid.onnx

输入:
270x480

阈值:
0.70 用于更连续轨迹
0.85 用于更高 Precision 和 F1 评估

peak_window:
15
```

视频推理建议输出：

```text
带轨迹视频 mp4
逐帧球心坐标 csv
```

## 10. 已知问题与后续优化

当前项目已经完成的工程收口包括：

```text
1. README 和 requirements 已更新到当前 V2/V3 项目环境。
2. WebUI 默认视频路径已指向当前示例视频。
3. WebUI 和推理脚本默认 codec 已改为 mp4v。
4. steps_per_epoch 已修正为实际训练 step 数。
5. V3 validate/eval/infer 的 peak_window 已统一为 15。
6. WebUI 已加入 CUDA 任务并发保护，避免多个 GPU 任务同时运行导致 OOM。
```

后续仍可继续完善的方向：

```text
1. 若要进一步提升 Recall，可尝试 base_channels=32 或增加轻量注意力/浅层细节分支。
2. 针对 RK3588 实测结果继续优化 batch1/batch2/batch4 的吞吐与延迟。
3. 增加轨迹连续性过滤、卡尔曼滤波和 ROI 跟踪。
```

后续模型优化方向：

```text
1. V3 base_channels 24 -> 32，提高小球细节表达。
2. ROI 跟踪：全局低频检测 + 局部高频跟踪。
3. 加入卡尔曼滤波或轨迹连续性约束，减少跳点和断点。
4. 尝试 3 帧灰度输入，降低输入通道和第一层计算量。
5. RKNN int8 量化，实测精度与速度变化。
```

## 11. 项目方案表述

可这样概括技术路线：

```text
项目首先复现 TrackNet 时序球心定位方法，分析其在嵌入式平台部署中存在输出维度大、后处理复杂和计算量高等问题。随后提出单通道热图输出的 V2 改进模型，将原逐像素 256 类分类任务转换为球心热图回归任务，降低输出维度并简化后处理。在此基础上，进一步设计 V3 轻量化模型，通过降低输入分辨率和引入深度可分离卷积显著减少计算量。实验结果表明，V3 在保持较好轨迹跟踪精度的同时，大幅降低模型参数量和计算量，更适合 RK3588 等边缘平台部署。
```

关于不引入 YOLO 的说明：

```text
系统没有引入额外网球类别识别模块，而是采用基于连续帧热图的球心定位方法。原因在于核心任务是轨迹跟踪，最终关注球心位置及其运动轨迹；额外识别模型会增加计算量和部署复杂度；同时网球高速运动时常出现尺寸小、边界模糊和运动拖影，基于外观和检测框的目标检测模型未必能稳定处理该类场景。
```

## 12. 关键文件索引

```text
原始 TrackNet:
  versions/v1_original/model.py
  versions/v1_original/main.py
  versions/v1_original/infer_on_video.py

V2:
  versions/v2_heatmap/model_v2.py
  versions/v2_heatmap/main_v2.py
  versions/v2_heatmap/general_v2.py
  versions/v2_heatmap/infer_on_video_v2_stream.py
  versions/v2_heatmap/export_onnx_v2.py

V3:
  versions/v3_lightweight/model_v3.py
  versions/v3_lightweight/main_v3.py
  versions/v3_lightweight/general_v3.py
  versions/v3_lightweight/infer_on_video_v3_batch.py
  versions/v3_lightweight/infer_on_video_v3_onnx.py
  versions/v3_lightweight/export_onnx_v3.py
  versions/v3_lightweight/eval_thresholds_v3.py

数据:
  versions/v2_heatmap/prepare_dataset_v2.py
  versions/v2_heatmap/datasets_v2.py
  versions/v3_lightweight/datasets_v2.py
  datasets/trackNet/labels_train.csv
  datasets/trackNet/labels_val.csv

WebUI:
  tracknet_webui/server.py
  tracknet_webui/static/index.html
  tracknet_webui/static/app.js
  tracknet_webui/static/styles.css
```
