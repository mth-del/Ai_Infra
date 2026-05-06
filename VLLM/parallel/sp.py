'''
使用多线程模拟多设备，并且使用简单的全连接层。对比
* 不同序列：整个序列数据通过一个完整的模型（多个层）进行计算
* 切序列（序列并行）：将序列分成多个部分，每个部分通过一个设备上的子模型计算，然后将结果合并

步骤：
    1.定义模型
    2.生成输入数据
    3.运行不切序列版本
    4.运行切序列版本（序列并行）
比较结果和时间
'''

import torch 
import torch.nn as nn
import torch.nn.functional as F
import time
import threading
import numpy as np
import copy

class SimpleFeedForwardBlock(nn.Module):
    def __init__(self, d_model: int = 512, hidden_dim: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm(x)

        return x


class SequenceParallelSimulator:
    def __init__(self, seq_len: int = 1024, d_model: int = 512):
        """
        初始化序列并行模拟器

        Args:
            seq_len: 序列长度
            d_model: 模型维度
        """
        self.seq_len = seq_len
        self.d_model = d_model

        # 固定随机种子
        torch.manual_seed(42)
        np.random.seed(42)

        # 创建主模型
        self.model = SimpleFeedForwardBlock(d_model)

        # 生成测试数据
        self.input_data = None
        self.generate_data()

        # 时间记录
        self.serial_time = None
        self.parallel_time = None


    def generate_data(self):
        """生成测试数据"""
        # 固定随机种子生成输入序列
        self.input_data = torch.randn(1, self.seq_len, self.d_model)
        print(f"生成测试数据: batch_size=1, seq_len={self.seq_len}, d_model={self.d_model}")
        print(f"输入数据范围: [{self.input_data.min():.4f}, {self.input_data.max():.4f}]")


    def serial_processing(self) -> torch.Tensor:
        """串行处理：整个序列一次性处理"""
        print(f"\n串行处理 - 完整序列 ({self.seq_len} tokens)")

        # 确保模型在评估模式
        self.model.eval()


         # 预热
        with torch.no_grad():
            _ = self.model(self.input_data[:, :10, :])

        # 正式计时
        start_time = time.time()

        # 完整序列一次性处理
        with torch.no_grad():
            output = self.model(self.input_data)

        end_time = time.time()
        self.serial_time = end_time - start_time

        print(f"串行处理完成，耗时: {self.serial_time:.4f}秒")
        return output   

    def parallel_worker(self,
                    model: nn.Module,
                    input_chunk: torch.Tensor,
                    output_chunk: list,
                    worker_id: int):
        """并行工作线程：处理序列的一个片段"""
        start_time = time.time()

        # 处理序列片段
        with torch.no_grad():
            chunk_output = model(input_chunk)

        end_time = time.time()

        # 存储结果
        output_chunk[worker_id] = chunk_output

        print(f"  工作线程{worker_id+1}: 处理{input_chunk.shape[1]}个token，耗时: {end_time - start_time:.4f}秒")


    def sequence_parallel_processing(self, num_chunks: int = 2) -> torch.Tensor:
        """序列并行处理：将序列切分成多个片段并行处理"""
        print(f"\n序列并行处理 - 将序列分成{num_chunks}个片段")

        # 计算每个片段的大小
        chunk_size = self.seq_len // num_chunks

        print(f"每个片段大小: {chunk_size} tokens")

        # 创建多个模型副本，确保使用相同的权重
        models = []
        for i in range(num_chunks):
            # 深拷贝模型，确保权重相同
            model_copy = copy.deepcopy(self.model)
            model_copy.eval()  # 设置为评估模式
            models.append(model_copy)

            # 验证权重是否相同
            if i > 0:
                params1 = list(models[0].parameters())
                params2 = list(model_copy.parameters())
                for p1, p2 in zip(params1, params2):
                    if not torch.allclose(p1, p2):
                        print(f"警告: 模型{i}的权重与模型0不同！")

        # 切分输入序列
        input_chunks = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else self.seq_len
            chunk = self.input_data[:, start_idx:end_idx, :]
            input_chunks.append(chunk)
            print(f"  片段{i+1}数据范围: [{chunk.min():.4f}, {chunk.max():.4f}]")

        # 准备存储结果的列表
        output_chunks = [None] * num_chunks

        # 创建并启动线程
        threads = []
        start_time = time.time()

        for i in range(num_chunks):
            thread = threading.Thread(
                target=self.parallel_worker,
                args=(models[i], input_chunks[i], output_chunks, i)
            )
            threads.append(thread)

            print(f"  分配任务给工作线程{i+1}: 处理token范围 [{i*chunk_size}:{(i+1)*chunk_size if i < num_chunks-1 else self.seq_len}]")

        print("\n开始并行处理...")
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 合并结果
        print("合并结果...")
        parallel_output = torch.cat(output_chunks, dim=1)

        end_time = time.time()
        self.parallel_time = end_time - start_time

        print(f"序列并行处理完成，总耗时: {self.parallel_time:.4f}秒")
        return parallel_output

    def validate_result(self, serial_output: torch.Tensor, parallel_output: torch.Tensor) -> bool:
        """验证两种方法的结果是否一致"""
        print("\n结果验证:")

        # 打印输出范围以供参考
        print(f"  串行输出范围: [{serial_output.min():.4f}, {serial_output.max():.4f}]")
        print(f"  并行输出范围: [{parallel_output.min():.4f}, {parallel_output.max():.4f}]")

        # 检查形状是否相同
        if serial_output.shape != parallel_output.shape:
            print(f"  形状不匹配: 串行{serial_output.shape} vs 并行{parallel_output.shape}")
            return False

        # 计算差异
        abs_diff = torch.max(torch.abs(serial_output - parallel_output)).item()
        rel_diff = torch.max(torch.abs(serial_output - parallel_output) / (torch.abs(serial_output) + 1e-10)).item()

        print(f"  最大绝对差异: {abs_diff:.6e}")
        print(f"  最大相对差异: {rel_diff:.6e}")

        # 检查是否在可接受的误差范围内（浮点数计算误差）
        tolerance = 1e-6
        is_valid = abs_diff < tolerance

        if is_valid:
            print("序列并行计算结果与串行计算结果一致！")
        else:
            print("计算结果存在显著差异")

        return is_valid

    def compare_performance(self):
        """比较性能并打印结果"""
        print("\n" + "="*60)
        print("性能对比")
        print("="*60)
        print(f"串行处理耗时: {self.serial_time:.4f}秒")
        print(f"序列并行(2线程)处理耗时: {self.parallel_time:.4f}秒")

        if self.serial_time and self.parallel_time:
            if self.serial_time > self.parallel_time:
                speedup = self.serial_time / self.parallel_time
                improvement = (self.serial_time - self.parallel_time) / self.serial_time * 100
                print(f"加速比: {speedup:.2f}倍")
                print(f"性能提升: {improvement:.1f}%")
            else:
                print("注意: 由于线程开销和小矩阵运算，多线程可能不会加速")

def main():
    """主函数"""
    print("="*60)
    print("序列并行(SP)策略演示")
    print("模拟逐位置操作的序列并行（如前馈网络）")
    print("="*60)

    # 创建模拟器
    seq_len = 2048  # 使用较长的序列
    d_model = 512
    simulator = SequenceParallelSimulator(seq_len=seq_len, d_model=d_model)

    # 1. 串行处理
    print("\n1. 串行处理 (基准):")
    serial_output = simulator.serial_processing()

    # 2. 序列并行处理
    print("\n2. 序列并行处理:")
    parallel_output = simulator.sequence_parallel_processing(num_chunks=2)

    # 3. 验证结果
    print("\n3. 验证计算结果:")
    is_valid = simulator.validate_result(serial_output, parallel_output)

    # 4. 性能对比
    simulator.compare_performance()


if __name__ == "__main__":
    main()   

                    