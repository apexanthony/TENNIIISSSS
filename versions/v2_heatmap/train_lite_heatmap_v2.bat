@echo off
cd /d "%~dp0\..\.."
if not exist "exps\lite_heatmap_v2" mkdir "exps\lite_heatmap_v2"
echo ===== train started %DATE% %TIME% =====>>"exps\lite_heatmap_v2\train_live.log"
python -u versions\v2_heatmap\main_v2.py --exp_id lite_heatmap_v2 --num_epochs 300 --steps_per_epoch 200 --val_intervals 5 --batch_size 2 --device cuda --base_channels 32 --num_workers 0 >>"exps\lite_heatmap_v2\train_live.log" 2>>"exps\lite_heatmap_v2\train_live.err.log"
echo ===== train exited code %ERRORLEVEL% %DATE% %TIME% =====>>"exps\lite_heatmap_v2\train_live.log"
