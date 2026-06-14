# V3.4 Multitask Context

V3.4 trains a context-aware multi-task model with ball, player, court, and status supervision.

Main files:

```text
main_v34_multitask.py       training
datasets_v34_multitask.py   clip-level split dataset
model_v34.py                multi-task model
```

Current observation:

```text
Ball F1 improved only slightly during fine-tuning.
Status macro-F1 stayed near the majority-class baseline.
```

The next step should focus on V3.5/V4.4 reference-style pipeline improvements rather than simply training V3.4 longer.
