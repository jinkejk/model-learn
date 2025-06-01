import torch
import time


def test_gpu_tensorcore_cuda_fp16_compute():
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("CUDA 不可用，请检查你的环境配置。")
        return

    # 获取 CUDA 设备
    device = torch.device("cuda")

    # 定义矩阵大小，可根据显卡性能调整
    matrix_size = 4096

    # 创建 FP16 随机矩阵并移到 CUDA 设备上
    matrix_a = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)
    matrix_b = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)

    # 预热操作，让 CUDA 设备和 Tensor Core 达到稳定状态
    for _ in range(10):
        _ = torch.matmul(matrix_a, matrix_b)

    # 同步 CUDA 设备，确保预热操作完成
    torch.cuda.synchronize()

    # 记录开始时间
    start_time = time.time()

    # 执行多次矩阵乘法以获得更准确的计时
    num_iterations = 100
    for _ in range(num_iterations):
        _ = torch.matmul(matrix_a, matrix_b)

    # 同步 CUDA 设备，确保所有计算完成
    torch.cuda.synchronize()

    # 记录结束时间
    end_time = time.time()

    # 计算总耗时
    elapsed_time = end_time - start_time

    # 计算每次矩阵乘法的浮点运算次数（FLOPs）
    flops_per_iteration = 2 * matrix_size ** 3

    # 计算总浮点运算次数
    total_flops = flops_per_iteration * num_iterations

    # 计算算力（TFLOPS）
    tflops = total_flops / (elapsed_time * 1e12)

    print(f"显卡测试的混合 FP16 算力: {tflops} TFLOPS")

    return tflops


def verify_gpu_fp32_tflops(matrix_size=4096, num_runs=100):
    # 创建两个随机矩阵，数据类型为 torch.float32
    a = torch.randn(matrix_size, matrix_size, dtype=torch.float32).cuda()
    b = torch.randn(matrix_size, matrix_size, dtype=torch.float32).cuda()

    # 进行热身运算，让 GPU 达到稳定状态
    for _ in range(5):
        _ = torch.matmul(a, b)

    # 记录开始时间
    start_time = time.time()

    # 进行多次矩阵乘法运算
    for _ in range(num_runs):
        c = torch.matmul(a, b)

    # 等待所有运算完成
    torch.cuda.synchronize()

    # 记录结束时间
    end_time = time.time()

    # 计算总时间
    total_time = end_time - start_time

    # 计算每次矩阵乘法的浮点运算次数（FLOPs）
    flops_per_matmul = 2 * matrix_size ** 3

    # 计算总的浮点运算次数
    total_flops = flops_per_matmul * num_runs

    # 计算 TFLOPS
    tflops = total_flops / (total_time * 1e12)

    return tflops


if __name__ == "__main__":
    # test_gpu_tensorcore_cuda_fp16_compute()
    tflops = verify_gpu_fp32_tflops()
    print(f"显卡测试的 FP32 算力: {tflops} TFLOPS")
