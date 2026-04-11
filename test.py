import torch
import tensorrt as trt
import numpy as np
import argparse
import os
import json
import datetime
from model import BallTrackerNet
from datasets import trackNetDataset
from general import validate

# --- TensorRT 高级接口类 ---
class TRTEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, 'rb') as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        
        self.tensors = {}
        self.names = []
        self.addresses = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.names.append(name)
            shape = self.engine.get_tensor_shape(name)
            if -1 in shape:
                shape = self.context.get_tensor_shape(name)
            
            # 预分配显存
            data_ptr = torch.empty(tuple(shape), dtype=torch.float32, device='cuda')
            self.tensors[name] = data_ptr
            self.addresses.append(data_ptr.data_ptr())
            
            if hasattr(self.context, 'set_tensor_address'):
                self.context.set_tensor_address(name, data_ptr.data_ptr())

    def __call__(self, input_tensor):
        input_name = self.names[0]
        output_name = self.names[-1]
        batch_size = input_tensor.size(0)

        # 1. 输入数据同步到显存预留位
        self.tensors[input_name][:batch_size].copy_(input_tensor, non_blocking=True)
        
        # 2. 异步执行推理
        if hasattr(self.context, 'execute_async_v3'):
            self.context.execute_async_v3(self.stream.cuda_stream)
        else:
            self.context.execute_v2(self.addresses)
            
        # 3. 获取输出
        output = self.tensors[output_name]
        
        # 🔥 关键修复：强制重塑形状，确保 Channel=256 在第二维
        # 这是为了适配 general.py 里的热力图解析逻辑
        return output.view(batch_size, 256, -1)

    def eval(self): return self
    def to(self, device): return self

# --- 硬件级测速函数 ---
def benchmark_speed(model, device, is_trt=False, num_iters=100):
    dummy_input = torch.randn(1, 9, 360, 640).to(device).contiguous()
    for _ in range(50): _ = model(dummy_input)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    timings = []

    with torch.no_grad():
        for _ in range(num_iters):
            start_event.record()
            _ = model(dummy_input)
            end_event.record()
            if is_trt:
                model.stream.synchronize()
            else:
                torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))
            
    return np.mean(timings)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to .engine')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    device = 'cuda'
    
    print(f"\n 正在评估 TensorRT 引擎: {os.path.basename(args.model_path)}")
    print("="*60)

    # 1. 加载模型
    model = TRTEngine(args.model_path)

    # 2. 加载数据集
    val_dataset = trackNetDataset('val')
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    # 3. 精度评估 (遍历 5894 个样本)
    print("\n" + "-"*20 + " 正在评估精度 " + "-"*20)
    val_loss, precision, recall, f1 = validate(model, val_loader, device, -1)

    # 4. 速度基准测试
    print("\n" + "-"*20 + " 正在评估速度 " + "-"*20)
    avg_ms = benchmark_speed(model, device, is_trt=True)

    # 5. 打印汇总报告
    print(f"模型文件: {os.path.basename(args.model_path)}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"平均延迟: {avg_ms:.2f} ms")
    print(f"吞吐量 (FPS): {1000/avg_ms:.2f}")