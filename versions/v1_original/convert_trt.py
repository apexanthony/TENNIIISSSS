import tensorrt as trt
import os

def build_engine(onnx_file, engine_file):
    # 创建日志记录器 (TensorRT 10 推荐模式)
    logger = trt.Logger(trt.Logger.VERBOSE)
    
    # 创建 Builder
    builder = trt.Builder(logger)
    
    # 创建网络定义，显式指定显式 Batch 标志
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    
    # 创建 ONNX 解析器
    parser = trt.OnnxParser(network, logger)
    
    # 读取并解析 ONNX 模型
    if not os.path.exists(onnx_file):
        print(f"找不到 ONNX 模型: {onnx_file}")
        return
        
    with open(onnx_file, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: 无法解析 ONNX 模型!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # 配置转换参数
    config = builder.create_builder_config()
    
    # 设置显存池限制 (代替旧版本的 workspace)
    # 给 L40 显卡设置 4GB 显存池
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)
    
    # 开启 FP16 模式
    if builder.platform_has_tf32:
        config.set_flag(trt.BuilderFlag.TF32)
    config.set_flag(trt.BuilderFlag.FP16)

    print(f"正在构建 Engine (这可能需要几分钟)...")
    
    # 构建并序列化网络
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: 构建 Engine 失败!")
        return

    # 保存 Engine 文件
    with open(engine_file, "wb") as f:
        f.write(serialized_engine)
        
    print(f"成功! Engine 已保存至: {engine_file}")

if __name__ == "__main__":
    ONNX_PATH = "weights/model_best.onnx"
    ENGINE_PATH = "weights/model_best.engine"
    build_engine(ONNX_PATH, ENGINE_PATH)