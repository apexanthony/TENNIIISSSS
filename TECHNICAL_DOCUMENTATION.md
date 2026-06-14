## 当前版本命名与推荐路线

本项目已将落地识别后处理版本整合为连续的 V4 系列：

```text
V4.1 = 规则落地检测，代码位于 versions/v4/v4_1_bounce_rule
V4.2 = 轨迹候选分类器，代码位于 versions/v4/v4_2_bounce_classifier
V4.3 = 事件级分类 + offset 帧修正，代码位于 versions/v4/v4_3_event_classifier
```

旧命名对应关系：

```text
old V4 -> V4.1
old V5 -> V4.2
old V4.3 -> V4.3
```

当前推荐主线是 V3 轻量轨迹模型 + V4.3 事件级落地检测。V4.1 作为可解释规则基线，V4.2 作为轻量二阶段候选分类器，V4.3 作为完整数据集验证后效果最好的泛化版本。
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
versions/v4/
  v4_1_bounce_rule/        V4.1 规则落地识别模块
  v4_2_bounce_classifier/  V4.2 轨迹候选分类器
  v4_3_event_classifier/   V4.3 事件级分类 + offset 修正
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
versions/v3_lightweight/v3_0_base/model_v3.py
versions/v3_lightweight/v3_0_base/main_v3.py
versions/v3_lightweight/v3_0_base/general_v3.py
versions/v3_lightweight/v3_0_base/infer_on_video_v3_batch.py
versions/v3_lightweight/v3_0_base/infer_on_video_v3_onnx.py
versions/v3_lightweight/v3_0_base/export_onnx_v3.py
versions/v3_lightweight/v3_0_base/eval_thresholds_v3.py
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

V2/V3 数据集读取由 `versions/v2_heatmap/datasets_v2.py` 和 `versions/v3_lightweight/v3_0_base/datasets_v2.py` 完成。两个版本各保留一份数据集读取代码，便于从项目根目录直接运行对应版本脚本。

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
  versions/v3_lightweight/v3_0_base/model_v3.py
  versions/v3_lightweight/v3_0_base/main_v3.py
  versions/v3_lightweight/v3_0_base/general_v3.py
  versions/v3_lightweight/v3_0_base/infer_on_video_v3_batch.py
  versions/v3_lightweight/v3_0_base/infer_on_video_v3_onnx.py
  versions/v3_lightweight/v3_0_base/export_onnx_v3.py
  versions/v3_lightweight/v3_0_base/eval_thresholds_v3.py

数据:
  versions/v2_heatmap/prepare_dataset_v2.py
  versions/v2_heatmap/datasets_v2.py
  versions/v3_lightweight/v3_0_base/datasets_v2.py
  datasets/trackNet/labels_train.csv
  datasets/trackNet/labels_val.csv

WebUI:
  tracknet_webui/server.py
  tracknet_webui/static/index.html
  tracknet_webui/static/app.js
  tracknet_webui/static/styles.css

## V4 规则落地识别模块

V4 的目标是在不新增模型、不修改 V3 主模型的前提下，基于 V3 输出的网球球心轨迹完成落地/反弹事件检测。该模块服务于后续自动判罚链路，但当前阶段只判断是否发生落地，不判断界内/出界。

### 设计原则

现有数据集是全场对打视频，画面上下分别对应两个半场。网球可能在上半场或下半场落地，击球方向也会导致图像坐标中的 `y` 变化方向不同。因此 V4 不使用单一的 `vy_before > 0 and vy_after < 0` 规则，而采用方向无关的局部轨迹几何突变。

核心判断不再是“向下后向上”，而是：

```text
连续球心轨迹上出现合理范围内的速度方向变化、加速度突变和轨迹折点。
```

同时，击球、遮挡恢复和误检跳点也会造成轨迹突变，因此 V4 使用软评分机制：

```text
适中的突变加分
过小的变化扣分
过大的突变扣分
轨迹连续加分
轨迹断裂或跳点过大扣分
击球式速度提升扣分
```

### 算法流程

```text
V3 输出球心轨迹
  -> 低置信度点过滤
  -> 异常跳点过滤
  -> 短缺失线性插值
  -> 加权移动平均平滑
  -> 计算前后窗口平均速度向量
  -> 计算夹角、加速度突变、速度范围、跳点和连续性
  -> 软评分得到落地候选
  -> 可落地区域 ROI 加分、扣分或硬过滤
  -> NMS 合并连续候选
  -> 输出落地帧、落地点和评分
```

### 默认参数

```text
min_conf          = 0.30
max_gap           = 3
smooth_window     = 5
before_window     = 3
after_window      = 3
min_speed         = 2.0
speed_max         = 90.0
speed_ratio_max   = 5.0
hit_speedup_ratio = 1.2
hit_speedup_delta = 7.0
hit_speedup_penalty = 2.0
angle_min         = 20.0
angle_weak_max    = 12.0
angle_good_max    = 75.0
angle_hard_max    = 90.0
accel_min         = 3.0
accel_weak_max    = 8.0
accel_good_max    = 24.0
accel_hard_max    = 32.0
jump_good_max     = 50.0
jump_hard_max     = 120.0
min_valid_points  = 6
min_score         = 9.0
nms_window        = 22
region_bonus      = 1.0
region_penalty    = 4.0
region_hard_filter = false
```

### 可落地区域 ROI

观众席、广告牌、球员身体附近、画面边缘和非球场地面区域都可能产生轨迹突变，从而被 V4 误判为落地。为减少这类误判，V4 增加可落地区域 ROI 约束。

这个 ROI 不是逐帧标注，而是场景级标注：对于固定机位或同一 `game`，只需要在一张参考帧上手动圈出网球可能落地的球场地面区域。由于相机视角固定，该多边形可以复用于该 `game` 下的所有 clip。

标注命令：

```powershell
python tools/annotate_ground_region.py --game game1
```

按 clip 单独标注：

```powershell
python tools/annotate_ground_region.py --game game1 --clip Clip1
```

默认输出：

```text
configs/court_regions/game1.json
configs/court_regions/game1_Clip1.json
```

JSON 示例：

```json
{
  "game": "game1",
  "source_frame": "datasets/trackNet/images/game1/Clip1/0000.jpg",
  "frame_width": 1280,
  "frame_height": 720,
  "playable_ground": [
    [[120, 140], [1160, 140], [1210, 690], [80, 690]]
  ]
}
```

ROI 默认作为软约束参与评分：

```text
候选点在可落地区域内: score += region_bonus
候选点在可落地区域外: score -= region_penalty
```

如果启用 `--region-hard-filter`，区域外候选会被直接过滤。实际项目中建议先使用软约束，因为边缘出界落地仍然可能发生在球场外延地面区域；后续做界内/出界判罚时，再单独引入球场线几何模型。

### V4.1 击球保护

错误可视化显示，部分 FP 来自击球附近的轨迹突变。击球后常出现短时间内速度明显增大，随后候选点在 6 到 10 帧内仍保留较高夹角和加速度分数。V4.1 增加 `hit_guard` 二阶段扣分：

```text
若候选点之前 hit_guard_window 帧内出现疑似击球式加速:
  speed_after / speed_before >= hit_guard_speedup_ratio
  speed_after - speed_before >= hit_guard_speedup_delta
  speed_after >= hit_guard_min_speed_after
则对当前候选 score -= hit_guard_penalty
```

当前推荐：

```text
enable_v41                 = true
hit_guard_window           = 8
hit_guard_speedup_ratio    = 1.6
hit_guard_speedup_delta    = 10.0
hit_guard_min_speed_after  = 18.0
hit_guard_penalty          = 3.0
```

V4.1 中还保留了候选时间重定位实验开关 `--enable-relocation`。当前数据上，重定位虽然能降低平均帧误差，但会把部分候选移到击球附近，导致总体 F1 下降，因此默认关闭。

### V4.2 事件合并与弱反弹补分

错误分析发现，部分 `status=2` 标签会连续出现，例如 `[251, 252]`。如果评估时把每一帧都当成独立落地事件，而检测端又使用 NMS 合并候选，就会人为增加 FN。因此 V4.2 将 3 帧以内的连续 `status=2` 合并为同一次落地事件。

V4.2 还增加两类补偿规则：

```text
低速弱反弹补分:
  angle_change 较大
  jump_distance 较小
  speed_before / speed_after 都较低

尖锐真实落地条件放宽:
  angle_change 和 accel_norm 很大
  但轨迹连续、跳点不大、速度不是击球式突增
```

实验性的 `--enable-late-refine` 只允许候选向后重定位，但当前全量评估会增加击球附近 FP，因此默认关闭。

### V4.3 自适应轨迹统计

V4.3 的目标不是继续针对当前数据集手调阈值，而是探索跨视频泛化。它会对每个 clip 的候选轨迹特征计算分位数统计：

```text
low_speed_mean: speed_mean 的低分位数
normal_jump: jump_distance 的常规上界
hard_jump: jump_distance 的高分位异常上界
high_accel: accel_norm 的高分位数
```

然后基于这些相对阈值进行低速弱反弹补分、尖锐落地补分和异常跳点惩罚。当前数据集上，自适应加分会增加击球附近 FP，因此 V4.3 暂时不作为默认推荐；保守配置只保留自适应跳点惩罚，用于后续新机位数据验证。

### 评估方式

数据集 `status` 字段含义：

```text
0 = flying
1 = hit
2 = bouncing
```

使用现有标签中的 `status=2` 作为真实落地帧：

```powershell
python versions/v4/v4_1_bounce_rule/eval_bounce_rule.py ^
  --split all ^
  --match-tolerance 3 ^
  --out-csv .\exps\v4_1_bounce_rule\bounce_rule_all_candidates.csv
```

带 ROI 评估：

```powershell
python versions/v4/v4_1_bounce_rule/eval_bounce_rule.py ^
  --split all ^
  --match-tolerance 3 ^
  --region-root .\configs\court_regions ^
  --out-csv .\exps\v4_1_bounce_rule\bounce_rule_roi_candidates.csv
```

匹配规则：

```text
预测帧与真实 status=2 帧相差 <= 3 帧，记为检测正确。
```

当前推荐参数在完整 95 个 Clip 上的评估结果：

```text
TP        = 327
FP        = 103
FN        = 196
Hit FP    = 39
Precision = 0.7605
Recall    = 0.6252
F1        = 0.6863
AvgErr    = 1.165 frames
```

ROI + V4.1 击球保护推荐命令：

```powershell
python versions/v4/v4_1_bounce_rule/eval_bounce_rule.py ^
  --split all ^
  --match-tolerance 3 ^
  --region-root .\configs\court_regions ^
  --region-bonus 0 ^
  --region-penalty 4 ^
  --min-score 9.5 ^
  --enable-v41 ^
  --out-csv .\exps\v4_1_bounce_rule\bounce_rule_v41_hitguard_candidates.csv
```

当前 V4.1 结果：

```text
TP        = 337
FP        = 75
FN        = 177
Hit FP    = 24
Precision = 0.8180
Recall    = 0.6556
F1        = 0.7279
AvgErr    = 1.092 frames
```

V4.2 推荐命令：

```powershell
python versions/v4/v4_1_bounce_rule/eval_bounce_rule.py ^
  --split all ^
  --match-tolerance 3 ^
  --region-root .\configs\court_regions ^
  --region-bonus 0 ^
  --region-penalty 4 ^
  --min-score 9.5 ^
  --enable-v41 ^
  --enable-v42 ^
  --out-csv .\exps\v4_1_bounce_rule\bounce_rule_v42_nolate_candidates.csv
```

当前 V4.2 结果：

```text
TP        = 364
FP        = 90
FN        = 150
Hit FP    = 34
Precision = 0.8018
Recall    = 0.7082
F1        = 0.7521
AvgErr    = 1.085 frames
```

V4.3 保守配置命令：

```powershell
python versions/v4/v4_1_bounce_rule/eval_bounce_rule.py ^
  --split all ^
  --match-tolerance 3 ^
  --region-root .\configs\court_regions ^
  --region-bonus 0 ^
  --region-penalty 4 ^
  --min-score 10 ^
  --enable-v41 ^
  --enable-v42 ^
  --enable-v43 ^
  --adaptive-low-speed-bonus 0 ^
  --adaptive-sharp-bonus 0 ^
  --adaptive-jump-penalty 2 ^
  --out-csv .\exps\v4_1_bounce_rule\bounce_rule_v43_conservative_candidates.csv
```

当前 V4.3 保守配置结果：

```text
TP        = 363
FP        = 91
FN        = 151
Hit FP    = 35
Precision = 0.7996
Recall    = 0.7062
F1        = 0.7500
AvgErr    = 1.069 frames
```

因此当前项目默认推荐 V4.2，V4.3 作为跨数据泛化实验开关保留。

### ACE 外部验证

为避免只围绕当前 TrackNet clip 调参，项目引入 ACE-Trajectories_noTosses 作为轨迹级外部验证集。该数据集不包含图片，但提供：

```text
r_img.npy: [T, 3]，包含 u, v, visibility
times.npy: 每帧时间戳
hits.npy: 网球落地时间戳
info.json: 原始 TrackNet game、clip、start、end 信息
```

转换脚本：

```text
tools/convert_ace_to_v4_eval.py
```

脚本会将 `r_img.npy` 转为 V4 可读的 `frame_id,x,y,confidence,status` 轨迹 CSV，并将 `hits.npy` 映射到最近的 `times.npy` 帧作为 `status=2` 落地事件，然后分别评估 V4.2 和 V4.3。

运行命令：

```powershell
python tools/convert_ace_to_v4_eval.py ^
  --ace-root .\datasets\external\ACE-Trajectories_noTosses ^
  --out-dir .\exps\v4_1_bounce_rule\ace_no_tosses ^
  --region-root .\configs\court_regions ^
  --match-tolerance 3
```

当前本机已下载部分 ACE 数据，其中 356 条完整轨迹包含落地事件。评估结果：

```text
ACE V4.2:
TP        = 218
FP        = 102
FN        = 146
Precision = 0.6813
Recall    = 0.5989
F1        = 0.6374
AvgErr    = 1.216 frames

ACE V4.3 conservative:
TP        = 217
FP        = 102
FN        = 147
Precision = 0.6803
Recall    = 0.5962
F1        = 0.6354
AvgErr    = 1.217 frames
```

该结果说明，V4.2 在当前 TrackNet clip 评估中已经可用，但跨到 ACE 轨迹级切分后性能明显下降；V4.3 当前的自适应统计并没有解决泛化问题。后续优化应优先转向：

```text
1. 球场几何归一化 / homography。
2. 基于轨迹状态机的落地事件检测。
3. 使用轻量轨迹特征分类器替代更多手写阈值。
4. 构建跨机位、小规模人工落地标注验证集。
```

## V4.2 轨迹候选分类器

V4 系列证明纯规则可以作为落地识别 baseline，但在原始数据集上仍存在 Recall 不足、击球附近误判和跨数据泛化下降的问题。V4.2 将流程改为两阶段：

```text
第一阶段: V4.2 规则生成候选
第二阶段: 轻量轨迹特征分类器判断候选是否为真实落地
```

V4.2 不读取图像，不增加视觉检测模型，只使用候选点附近的轨迹几何特征，因此计算量很小，部署成本远低于新增 YOLO 或图像分类器。

### 样本构造

对每个候选点，根据它与真实落地事件的距离自动生成标签：

```text
positive: 距离最近 status=2 落地事件 <= 3 帧
negative: 距离最近 status=2 落地事件 > 6 帧
ignore:   4 到 6 帧之间的边界样本
```

默认按 game 划分：

```text
train: game1-game7
val:   game8-game10
```

这样比 all split 更接近真实泛化测试。

### 特征

当前 V4.2 使用 23 个轨迹特征，包括：

```text
rule_score
angle_change
accel_norm
speed_before
speed_after
speed_mean
speed_ratio
speed_delta
jump_distance
valid_points
confidence
frame_pos
in_ground_region
recent_hit_penalty
angle_over_speed
accel_over_speed
jump_over_speed
speed_mean_rel
accel_rel
jump_rel
```

当前随机森林的重要特征排序显示，模型主要依赖：

```text
rule_score
in_ground_region
speed_delta
angle_change
accel_rel
accel_norm
speed_ratio
angle_over_speed
accel_over_speed
```

这些特征符合落地事件的轨迹几何特征，不依赖图像外观。

### 训练命令

```powershell
python versions/v4/v4_2_bounce_classifier/train_eval_bounce_classifier.py ^
  --labels-root .\datasets\trackNet\images ^
  --region-root .\configs\court_regions ^
  --out-dir .\exps\v4_2_bounce_classifier ^
  --candidate-min-score 3.0 ^
  --positive-tolerance 3 ^
  --negative-gap 6
```

输出：

```text
exps/v4_2_bounce_classifier/model_random_forest.pkl
exps/v4_2_bounce_classifier/metrics.csv
exps/v4_2_bounce_classifier/train_samples.csv
exps/v4_2_bounce_classifier/val_samples.csv
```

### 当前结果

同一验证集 `game8-game10` 上，V4.2 baseline：

```text
Precision = 0.8559
Recall    = 0.6601
F1        = 0.7454
TP        = 101
FP        = 17
FN        = 52
```

V4.2 RandomForest：

```text
train events:
Precision = 0.9217
Recall    = 0.8809
F1        = 0.9008

val events:
Precision = 0.9318
Recall    = 0.8039
F1        = 0.8632

all events:
Precision = 0.9245
Recall    = 0.8580
F1        = 0.8900
```

该结果说明，当前阶段继续堆手写规则的收益有限，而“规则候选 + 轨迹分类器”能显著提升落地识别效果。

如果应用场景更重视极高 Precision，可使用：

```powershell
python versions/v4/v4_1_bounce_rule/eval_bounce_rule.py ^
  --split all ^
  --match-tolerance 3 ^
  --accel-weak-max 10 ^
  --hit-speedup-delta 8 ^
  --nms-window 24 ^
  --out-csv .\exps\v4_1_bounce_rule\bounce_rule_all_precision080_candidates.csv
```

极高 Precision 参数结果：

```text
TP        = 210
FP        = 52
FN        = 313
Hit FP    = 21
Precision = 0.8015
Recall    = 0.4015
F1        = 0.5350
AvgErr    = 1.214 frames
```

参数网格搜索：

```powershell
python versions/v4/v4_1_bounce_rule/tune_bounce_rule.py ^
  --split all ^
  --match-tolerance 3 ^
  --out-csv .\exps\v4_1_bounce_rule\bounce_rule_tuning.csv
```

带 ROI 参数网格搜索：

```powershell
python versions/v4/v4_1_bounce_rule/tune_bounce_rule.py ^
  --split all ^
  --match-tolerance 3 ^
  --region-root .\configs\court_regions ^
  --out-csv .\exps\v4_1_bounce_rule\bounce_rule_roi_tuning.csv
```

V4.1 击球保护参数小网格搜索：

```powershell
python versions/v4/v4_1_bounce_rule/tune_bounce_rule.py ^
  --split all ^
  --match-tolerance 3 ^
  --region-root .\configs\court_regions ^
  --region-bonus 0 ^
  --region-penalty 4 ^
  --enable-v41 ^
  --min-score-list 9,9.5,10 ^
  --hit-guard-window-list 6,8,10 ^
  --hit-guard-penalty-list 2,3,4 ^
  --out-csv .\exps\v4_1_bounce_rule\bounce_rule_v41_hitguard_tiny_tuning.csv
```

### 错误可视化

完成评估后，可以将 FP/FN 最集中的片段导出为图片序列和 MP4 小视频，用于判断错误来源是击球误判、落地帧偏移、轨迹跳点、遮挡恢复还是标签边界问题。

```powershell
python tools/visualize_bounce_errors.py ^
  --candidates-csv .\exps\v4_1_bounce_rule\bounce_rule_v42_nolate_candidates.csv ^
  --labels-root .\datasets\trackNet\images ^
  --region-root .\configs\court_regions ^
  --out-dir .\exps\v4_1_bounce_rule\error_visuals_v42 ^
  --match-tolerance 3 ^
  --context 10 ^
  --top-clips 8 ^
  --max-events 40
```

### 视频推理

V4 可以直接读取 V3 推理输出的轨迹 CSV：

```powershell
python versions/v4/v4_1_bounce_rule/infer_video_bounce_rule.py ^
  --track-csv .\exps\lite_heatmap_v3_270x480_from240\demo_v3_pt.csv ^
  --out-csv .\exps\lite_heatmap_v3_270x480_from240\demo_bounces.csv ^
  --video-path .\示例视频1.mp4 ^
  --video-out-path .\exps\lite_heatmap_v3_270x480_from240\demo_bounces.mp4 ^
  --court-region-json .\configs\court_regions\game1.json ^
  --region-bonus 0 ^
  --region-penalty 4 ^
  --min-score 9.5 ^
  --enable-v41 ^
  --enable-v42 ^
  --codec mp4v
```

### 新增文件

```text
versions/v4/v4_1_bounce_rule/bounce_rule_detector.py
versions/v4/v4_1_bounce_rule/eval_bounce_rule.py
versions/v4/v4_1_bounce_rule/tune_bounce_rule.py
versions/v4/v4_1_bounce_rule/infer_video_bounce_rule.py
versions/v4/v4_1_bounce_rule/README.md
tools/annotate_ground_region.py
tools/visualize_bounce_errors.py
configs/court_regions/.gitkeep
```
```

## V4 系列统一路线

当前落地识别后处理统一为三个连续版本：

```text
V4.1 规则基线：versions/v4/v4_1_bounce_rule
V4.2 轨迹候选分类器：versions/v4/v4_2_bounce_classifier
V4.3 事件级分类 + offset 修正：versions/v4/v4_3_event_classifier
```

推荐主线：

```text
V3 轻量热图轨迹模型 -> V4.3 event classifier -> 落地帧与标注视频输出
```

V4.3 当前推荐模型：

```text
exps/v4_3_event_classifier_cand2_final/model_event_offset.pkl
```

V4.3 示例视频推理：

```powershell
python versions\v4\v4_3_event_classifier\infer_event_classifier.py ^
  --track-csv .\exps\demo_best_v5\demo_v3_track_thr030.csv ^
  --model-path .\exps\v4_3_event_classifier_cand2_final\model_event_offset.pkl ^
  --out-csv .\exps\demo_best_v5\demo_bounces_v43.csv ^
  --video-path .\示例视频1.mp4 ^
  --video-out-path .\exps\demo_best_v5\demo_bounces_v43.mp4 ^
  --court-region-json .\configs\court_regions\game1.json ^
  --jump-hard-max 240 ^
  --codec mp4v
```

## V5.1 FPS-Adaptive Event Processing

The V3.7 + V5.1 product pipeline now represents event windows in seconds and
converts them to frames with the source-video FPS. This keeps event semantics
consistent across 25/30/60 FPS inputs. Default timing is a 0.60-second minimum
event gap, a 0.15-second minimum flight, and a -0.067 to +0.20-second local
refinement window (extended to +0.40 seconds for fast vertical motion).

The `v37_bounce` state machine also suppresses serve tosses. Suppression is
allowed only when a new trajectory/point begins. A ball rising mostly
vertically from the serving player's hand or upper-body column is marked as a
toss until the next player contact, and those frames cannot generate bounce
events. Once the rally is established, ordinary upward ball motion does not
trigger toss suppression.

V5.1 additionally uses the right-side mini-court trajectory as a second motion
space. Direction reversal, mapped acceleration and mapped speed change provide
court-space bounce evidence. This evidence is rewarded when it agrees with the
image trajectory. A strong mapped-coordinate discontinuity without matching
image motion is treated as likely homography jitter and penalized. Therefore
automatic court-line errors cannot independently produce a bounce event.

The subsequent robustness upgrade validates and smooths homography corners,
assigns a per-frame `court_quality`, stabilizes at most two court-player poses,
and uses an explicit waiting/toss/contact/flying/approaching-player state
machine. Landing-time refinement searches a FPS-scaled interval and prefers
the earliest reliable low-speed reversal instead of the later maximum rebound
change. Event output now includes `frame_start`, `frame_end` and
`timing_confidence`. A conservative bidirectional trajectory check removes an
observation only when forward and backward predictions agree with each other
but both reject the current point.

On the manually reviewed sample video 4 regression, the optimized pipeline
predicted frames `68, 151, 233, 326, 388`. With a +/-3-frame tolerance,
Precision, Recall and F1 were all 1.0000 and the mean distance to the annotated
event intervals was 0.20 frames. This is a single-video regression result, not
a cross-dataset accuracy claim.

完整数据集优化路线以 V4.3 为主：先跑 raw metrics，再跑 observable metrics，最后按 game/clip 级 FP/FN 明细决定是否调整候选阈值、特征或 offset 模型。单视频 repair 规则不再作为默认泛化路径。
## V5.2 Integrated Runtime Package

V5.2 does not introduce a new model or event algorithm. It freezes the current
stable V3.7 + V5.1 behavior into one self-contained runtime directory:

```text
versions/v5_2_integrated_pipeline/
  run_pipeline.py
  tracking/     model, heatmap candidates, trajectory stabilization
  context/      MediaPipe players and court homography
  events/       cleanup and bounce recognition
  pipeline/     context assembly, recognition and rendering
  evaluation/   event Precision/Recall/F1 utility
```

The single entry point produces a rendered result video, a trajectory video,
raw/context trajectory CSV files, a bounce-event CSV, and reusable player-pose
cache. Detailed module flow is maintained in
`versions/v5_2_integrated_pipeline/README.md`.

V5.2 also separates event existence from contact-frame timing. A conservative
piecewise-motion refinement is applied only to low-confidence events with a
multi-frame uncertainty interval. Corrections remain inside the existing
interval and are recorded through `original_frame`, `frame_offset`, and
`timing_method`, so timing changes remain auditable.
