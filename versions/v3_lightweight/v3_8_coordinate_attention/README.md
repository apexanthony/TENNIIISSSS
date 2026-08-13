# V3.8 Coordinate-Attention Paper Experiments

This directory defines the controlled paper ablation around the V3.7
lightweight U-Net backbone.

```text
baseline = V3.7 backbone + heatmap loss
ca       = baseline + bottleneck Coordinate Attention
hardneg  = baseline + training-only hard-negative loss
aux      = baseline + training-only auxiliary-positive loss
full     = baseline + Coordinate Attention + hard-negative loss
```

All variants use three RGB frames, a 360x640 input/output heatmap, seed 37,
and the same match-level split. A localization is correct when its distance to
the label is at most 4 pixels in the resized 360x640 coordinate space.

## Split

```powershell
python versions\v3_lightweight\v3_8_coordinate_attention\prepare_match_split.py
```

Games 1-7 form the development set. The last clip of each development game is
validation-only. Every clip from games 8-10 is test-only.

## Smoke test

```powershell
python versions\v3_lightweight\v3_8_coordinate_attention\train_v38_ablation.py `
  --variant full --epochs 1 --steps-per-epoch 2 --max-val-batches 2 `
  --batch-size 2 --val-batch-size 2 --num-workers 0 `
  --run-name full_smoke
```

## Five one-seed experiments

```powershell
powershell -ExecutionPolicy Bypass -File `
  versions\v3_lightweight\v3_8_coordinate_attention\run_five_ablation.ps1
```

Validation selects both the checkpoint and confidence threshold. The frozen
checkpoint/threshold pair is then evaluated once on games 8-10.

## Linux NVIDIA server (recommended: A10)

The implementation is PyTorch, not PaddlePaddle. Use an NVIDIA CUDA PyTorch
environment. Install the CUDA build of PyTorch that matches the server driver,
then install the remaining packages:

```bash
python -m venv .venv-v38
source .venv-v38/bin/activate
python -m pip install --upgrade pip
# Choose the CUDA wheel command from https://pytorch.org/get-started/locally/
python -m pip install -r versions/v3_lightweight/v3_8_coordinate_attention/requirements-v38.txt
```

The split manifests and strict mapped annotations are versioned with the code.
The original TrackNet images are intentionally not stored in Git. Copy them to
the following layout (19,835 JPG files in total):

```text
datasets/trackNet/
  game1/Clip1/0000.jpg
  ...
  game10/ClipN/NNNN.jpg
```

Validate the GPU, Python packages, manifests, and image count before training:

```bash
python versions/v3_lightweight/v3_8_coordinate_attention/check_server_env.py
```

On the A10 instance (20 CPU cores, 116 GB RAM, 24 GB VRAM), launch all five
groups in a persistent terminal. Training batch size remains fixed at 8; the
larger validation batch does not affect optimization:

```bash
chmod +x versions/v3_lightweight/v3_8_coordinate_attention/run_five_ablation.sh
NUM_WORKERS=8 EVAL_WORKERS=8 BATCH_SIZE=8 VAL_BATCH_SIZE=16 \
  bash versions/v3_lightweight/v3_8_coordinate_attention/run_five_ablation.sh
```

The launcher writes to `exps/v38_essay_ablation`, automatically resumes a run
when `training_state.pt` exists, evaluates the validation-selected checkpoint
once on the held-out test games, and finally creates the ablation summary.
Use `tmux`, `screen`, or the cloud platform's persistent-job facility so an SSH
disconnect does not terminate training.
