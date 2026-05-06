'''
场景1:一个模型副本，我们用一个线程来运行这个模型，然后有4个数据任务，我们用一个线程池（4个线程）来同时发送数据给这个模型，但是模型处理是串行的，所以我们可以在模型内部加锁，使得同时只能有一个线程（即一个数据）被处理。
场景2:四个模型副本，每个模型副本在一个线程中，然后有4个数据，我们同样用4个线程来发送数据，但是每个数据发送给不同的模型副本，这样就能并行处理。
'''


import threading
import time
from queue import Queue
import concurrent.futures
from turtle import mode
from typing import List
import random
from unittest import result

class  FakeModel:
    def __init__(self, model_id:int):
        self.model_id = model_id
        self.lock = threading.Lock()
    
    def process(self, data:str) ->str:
        """模拟模型处理过程"""
        processing_time = random.uniform(0.1,0.5)
        time.sleep(processing_time)


        # 打印处理信息
        with self.lock:
            result = f"模型副本{self.model_id} 接收数据: '{data}', 已处理 (耗时: {processing_time:.3f}s)"
            print(result)
        
        return result


def single_model_scenario(data_list: List[str]):
    """场景1: 单个模型副本处理所有数据"""
    print("\n" + "="*60)
    print("场景1: 单个模型副本处理4条数据")
    print("="*60)

    model = FakeModel(1)
    start_time = time.time()

    # 串行处理
    results = []
    for data in data_list:
        results.append(model.process(data))
    
    end_time = time.time()
    print(f"\n总耗时: {end_time - start_time:.3f}秒")
    return results, end_time - start_time


def multi_model_scenario(data_list: List[str], num_models: int = 4):
    """场景2: 多个模型副本并行处理数据"""
    print("\n" + "="*60)
    print(f"场景2: {num_models}个模型副本并行处理4条数据")
    print("="*60)

    # 创建多个副本
    models = [FakeModel(i+1) for i in range(num_models)]
    start_time = time.time()

    # 使用线程池并行提交任务；线程数等于模型副本数，模拟每个副本同时处理一份数据
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_models) as executor:
        # 保存 Future 和原始数据的映射，后面收集结果时可以知道每个异步任务对应哪条输入
        future_to_data = {}
        for i, data in enumerate(data_list):
            # 轮询分配：第 i 条数据交给第 i % num_models 个模型副本
            # 这样可以把数据均匀分散到多个副本上，模拟数据并行中的 batch 拆分
            model_idx = i % num_models

            # submit 不会阻塞等待模型处理完成，而是立刻返回一个 Future
            # 真正的模型计算会在线程池中的某个 worker 线程里执行
            future = executor.submit(models[model_idx].process, data)
            future_to_data[future] = data
    
    # 收集结果
    for future in concurrent.futures.as_completed(future_to_data):
        results.append(future.result())

    end_time = time.time()
    print(f"\n总耗时: {end_time - start_time:.3f}秒")
    return results, end_time - start_time


def data_parallel_simulation():
    """主模拟函数"""
    print("数据并行(DP)策略模拟演示")
    print("-" * 60)


    # 模拟4条数据
    data_list = [
        "数据1: 图像分类任务",
        "数据2: 自然语言处理",
        "数据3: 语音识别样本",
        "数据4: 视频分析帧"
    ]   

    print("待处理数据:")    
    for i, data in enumerate(data_list, 1):
        print(f"  数据{i}: {data}")   

     # 场景1: 单个模型副本
    print("\n" + "="*60)
    print("开始模拟: 单个模型副本 vs 多个模型副本")
    print("="*60)

    # 重置随机种子以确保公平比较
    random.seed(42)
    results1, time1 = single_model_scenario(data_list.copy()) 

    # 场景2: 多个模型副本
    random.seed(42)  # 重置随机种子
    results2, time2 = multi_model_scenario(data_list.copy(), num_models=4)

       # 性能对比
    print("\n" + "="*60)
    print("性能对比总结")
    print("="*60)
    print(f"单个模型副本总耗时: {time1:.3f}秒")
    print(f"4个模型副本总耗时:  {time2:.3f}秒")
    print(f"加速比: {time1/time2:.2f}x")

    if time1 > time2:
        print(f"性能提升: {(time1 - time2)/time1*100:.1f}%")
    else:
        print("注意: 由于线程开销，加速效果可能不明显")


if __name__ == "__main__":
    data_parallel_simulation()