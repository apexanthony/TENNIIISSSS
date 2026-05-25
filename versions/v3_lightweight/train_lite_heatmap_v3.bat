@echo off
cd /d "%~dp0\..\.."
python -u versions\v3_lightweight\main_v3.py --num_epochs 100 --batch_size 16 --steps_per_epoch 200 --val_intervals 5 --base_channels 24 --input_height 180 --input_width 320 --heatmap_radius 4 --heatmap_sigma 1.5 --threshold 0.50 --peak_window 15 --min_dist 8 --augment --amp --device cuda
