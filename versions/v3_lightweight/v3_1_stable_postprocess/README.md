# V3.1 Stable Postprocess

V3.1 keeps the V3.0 model unchanged and improves inference-time trajectory stability.

Main file:

```text
infer_on_video_v31_stable.py
```

Core idea:

```text
heatmap -> top-k candidates -> motion gating -> jump rejection -> stable trajectory CSV
```

Detailed note:

```text
docs/V3_1_TRACK_STABILIZATION.md
```
