# TrackNet WebUI

本文件夹包含一个本地 Web 控制台，用来调用 `versions/v3_lightweight/` 中的 V3 训练、阈值测试、视频上传、PT 推理和 ONNX 推理脚本。

## 启动

在项目根目录运行：

```powershell
python .\tracknet_webui\server.py
```

然后打开：

```text
http://127.0.0.1:8765
```

## 目录

```text
tracknet_webui/
  server.py          本地 HTTP 服务和任务调度
  static/            前端页面、样式、脚本
  uploads/           上传视频保存位置
  outputs/           推理视频和 CSV 输出位置
  runs/              训练/测试/推理日志
```

## 默认 V3 配置

```text
input: 270x480
threshold: 0.70
peak_window: 15
PT: exps/lite_heatmap_v3_270x480_from240/model_best_thr070_pw15.pt
ONNX: exps/lite_heatmap_v3_270x480_from240/model_v3_270x480_b1_sigmoid.onnx
```
