# TrackNet

1. Run `pip3 install -r requirements.txt` to install packages required. 
2. Run `python gt_gen.py <args>` to create ground truth images and train/test labels.
3. Prepare dataset to the following format:
```
datasets/trackNet
    /images
        /game1
            /Clip1
                /0000.jpg
                ...
                /0206.jpg
            ...
            /Clip13
                /0000.jpg
                ...
                /0252.jpg
        ...
        /game10
    /gts
        /game1
            /Clip1
                /0000.jpg
                ...
                /0206.jpg
            ...
            /Clip13
                /0000.jpg
                ...
                /0252.jpg
        ...
        /game10
    /labels_train.csv
    /labels_val.csv
```
4. Run `python main.py` to start training
5. Run `python export_onnx.py` to export ONNX
6. Run `trtexec --onnx=/workspace/model_best.onnx --saveEngine=/workspace/weights/model_best.engine --fp16` to generate TensorRT Engine
7.Run `python test.py --model_path /workspace/model_best.pt --batch_size 1` to test PyTorch (.pt)
Run `python test.py --model_path /workspace/weights/model_best.engine --batch_size 1` to test TensorRT (.engine)