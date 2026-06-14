# V3.0 Base

Lightweight TrackNet heatmap tracker for deployment.

Main files:

```text
main_v3.py                  training
infer_on_video_v3_batch.py  PyTorch video inference
infer_on_video_v3_onnx.py   ONNX video inference
export_onnx_v3.py           ONNX export
eval_thresholds_v3.py       threshold evaluation
```

Recommended checkpoint:

```text
exps/lite_heatmap_v3_270x480_from240/model_best_thr070_pw15.pt
```
