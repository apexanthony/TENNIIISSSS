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

V4 规则落地识别:
  不新增模型，基于 V3 输出的球心轨迹检测落地/反弹事件。
  使用轨迹夹角、速度突变、跳点惩罚和连续性评分，减少击球误判。

V4.2 轨迹候选分类器:
  以 V4.2 规则生成候选，再用轻量轨迹特征分类器筛选。
  不读取图像，不增加视觉检测模型，提升落地事件 Precision / Recall。
```

## 目录结构

```text
versions/
  v1_original/      原始 TrackNet 基线代码
  v2_heatmap/       V2 单通道 heatmap 版本
  v3_lightweight/   V3 轻量化部署版本，内部按 v3_0/v3_1/v3_2/v3_3/v3_4 拆分
  v4/
    v4_1_bounce_rule/        V4.1 规则落地识别模块
    v4_2_bounce_classifier/  V4.2 轨迹候选分类器
    v4_3_event_classifier/   V4.3 事件级分类 + offset 修正

tracknet_webui/     本地 Web 控制台
tools/              标注和辅助工具
configs/            场景级配置，如 V4 可落地区域 ROI
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

下一步优化路线见：

```text
docs/NEXT_STEP_REFERENCE_PIPELINE.md
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
python versions/v3_lightweight/v3_0_base/main_v3.py ^
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
python versions/v3_lightweight/v3_0_base/main_v3.py --resume .\exps\lite_heatmap_v3_270x480_from240\training_state.pt --amp --device cuda
```

## 阈值测试

```powershell
python versions/v3_lightweight/v3_0_base/eval_thresholds_v3.py ^
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
python versions/v3_lightweight/v3_0_base/infer_on_video_v3_batch.py ^
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
python versions/v3_lightweight/v3_0_base/infer_on_video_v3_onnx.py ^
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

## V4 落地识别

V4 不修改 V3 主模型，也不新增神经网络。它读取球心轨迹 `x, y, confidence`，通过轨迹清洗、短缺失插值、平滑、速度向量夹角、加速度突变、跳点惩罚和连续性评分来检测网球落地/反弹帧。

由于全场对打视频中网球可能在上下两个半场落地，V4 不使用单一的 `vy` 正负反转规则，而采用方向无关的局部轨迹几何突变。

为了减少观众席、广告牌、球员身体附近、画面边缘和非球场地面区域造成的误判，V4 支持手动圈定可落地区域 ROI。这个步骤不是逐帧标注，而是每个固定机位或每个 `game` 圈一次多边形。

```powershell
python tools/annotate_ground_region.py --game game1
```

默认保存到：

```text
configs/court_regions/game1.json
```

使用现有数据集标签评估：

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

数据集中 `status` 字段含义：

```text
0 = flying
1 = hit
2 = bouncing
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

ROI + V4.1 击球保护推荐参数：

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

V4.2 在 V4.1 基础上增加连续落地标签合并、低速弱反弹补分和尖锐真实落地条件放宽，默认关闭实验性的 late-refine：

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

V4.3 增加每个 clip 的自适应轨迹统计阈值，用于探索跨机位、跨分辨率泛化。当前数据集上，V4.3 保守配置未超过 V4.2，因此暂作为实验开关保留：

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
Precision = 0.7996
Recall    = 0.7062
F1        = 0.7500
```

因此当前推荐仍为 V4.2；V4.3 更适合在新视频、新机位数据加入后继续验证。

### ACE 外部轨迹验证

ACE-Trajectories_noTosses 是基于 TrackNet Tennis 数据切分出的连续轨迹数据集，不包含图片，但提供 `r_img.npy` 球心轨迹和 `hits.npy` 落地时间戳。它适合验证 V4 规则在“轨迹级外部数据”上的泛化能力。

下载：

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='XSpaceCoderX/ACE-Trajectories_noTosses', repo_type='dataset', local_dir='./datasets/external/ACE-Trajectories_noTosses')"
```

转换并评估 V4.2 / V4.3：

```powershell
python tools/convert_ace_to_v4_eval.py ^
  --ace-root .\datasets\external\ACE-Trajectories_noTosses ^
  --out-dir .\exps\v4_1_bounce_rule\ace_no_tosses ^
  --region-root .\configs\court_regions ^
  --match-tolerance 3
```

当前本机下载到 356 条带落地事件的完整轨迹，外部验证结果：

```text
ACE V4.2:
Precision = 0.6813
Recall    = 0.5989
F1        = 0.6374

ACE V4.3 conservative:
Precision = 0.6803
Recall    = 0.5962
F1        = 0.6354
```

该结果说明：V4.2 在当前 TrackNet clip 评估中表现较好，但在 ACE 轨迹级数据上仍有明显泛化差距。下一步应优先做球场归一化、状态机和轻量轨迹后验分类器，而不是继续只调固定阈值。

## V4.2 轨迹候选分类器

V4.2 使用 V4.2 规则生成高质量候选，再提取候选点的局部轨迹特征训练 RandomForest 分类器。它不读取图像，不增加 YOLO 等视觉模型，计算量很小，适合作为落地识别的二阶段筛选器。

默认按 game 划分，避免 all split 过拟合：

```text
train: game1-game7
val:   game8-game10
```

运行：

```powershell
python versions/v4/v4_2_bounce_classifier/train_eval_bounce_classifier.py ^
  --labels-root .\datasets\trackNet\images ^
  --region-root .\configs\court_regions ^
  --out-dir .\exps\v4_2_bounce_classifier ^
  --candidate-min-score 3.0 ^
  --positive-tolerance 3 ^
  --negative-gap 6
```

当前结果：

```text
V4.2 baseline on game8-game10:
Precision = 0.8559
Recall    = 0.6601
F1        = 0.7454

V4.2 val on game8-game10:
Precision = 0.9318
Recall    = 0.8039
F1        = 0.8632

V4.2 all:
Precision = 0.9245
Recall    = 0.8580
F1        = 0.8900
```

输出：

```text
exps/v4_2_bounce_classifier/model_random_forest.pkl
exps/v4_2_bounce_classifier/metrics.csv
exps/v4_2_bounce_classifier/train_samples.csv
exps/v4_2_bounce_classifier/val_samples.csv
```

如果更重视极高 Precision，可设置 `--accel-weak-max 10 --hit-speedup-delta 8 --nms-window 24`：

```text
Precision = 0.8015
Recall    = 0.4015
F1        = 0.5350
Hit FP    = 21
```

参数网格搜索：

```powershell
python versions/v4/v4_1_bounce_rule/tune_bounce_rule.py ^
  --split all ^
  --match-tolerance 3 ^
  --out-csv .\exps\v4_1_bounce_rule\bounce_rule_tuning.csv
```

带 ROI 调参：

```powershell
python versions/v4/v4_1_bounce_rule/tune_bounce_rule.py ^
  --split all ^
  --match-tolerance 3 ^
  --region-root .\configs\court_regions ^
  --out-csv .\exps\v4_1_bounce_rule\bounce_rule_roi_tuning.csv
```

导出 FP/FN 错误可视化片段：

```powershell
python tools/visualize_bounce_errors.py ^
  --candidates-csv .\exps\v4_1_bounce_rule\bounce_rule_v42_nolate_candidates.csv ^
  --out-dir .\exps\v4_1_bounce_rule\error_visuals_v42
```

接 V3 推理轨迹 CSV：

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

## ONNX 导出

导出 batch1 ONNX：

```powershell
python versions/v3_lightweight/v3_0_base/export_onnx_v3.py ^
  --model-path .\exps\lite_heatmap_v3_270x480_from240\model_best_thr070_pw15.pt ^
  --onnx-path .\exps\lite_heatmap_v3_270x480_from240\model_v3_270x480_b1_sigmoid.onnx ^
  --input-height 270 ^
  --input-width 480 ^
  --batch-size 1 ^
  --device cuda
```

如需固定 batch4：

```powershell
python versions/v3_lightweight/v3_0_base/export_onnx_v3.py ^
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

完整数据集优化路线以 V4.3 为主：先跑 raw metrics，再跑 observable metrics，最后按 game/clip 级 FP/FN 明细决定是否调整候选阈值、特征或 offset 模型。单视频 repair 规则不再作为默认泛化路径。
## V5.2 Unified Inference Entry

The current stable V3.7 tracking and V5.1 bounce-recognition chain is packaged
under `versions/v5_2_integrated_pipeline/`. New end-to-end inference should use:

```powershell
python versions\v5_2_integrated_pipeline\run_pipeline.py `
  --video-path ".\input.mp4" `
  --output-dir ".\exps\v5_2" `
  --device cuda
```

The complete architecture and output layout are documented in
`versions/v5_2_integrated_pipeline/README.md`.
