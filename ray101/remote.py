import ray
import time

# 1. 初始化 Ray (如果尚未初始化)
if not ray.is_initialized():
    ray.init()

# 2. 使用 @ray.remote 装饰器定义一个远程函数
@ray.remote
def slow_square(x):
    """一个模拟耗时计算的函数"""
    time.sleep(1) # 模拟耗时操作
    print(f"Calculating square of {x}...")
    return x * x

# 3. 调用远程函数
#    - 使用 .remote() 进行调用，这会立即返回一个 ObjectRef (对象引用)
#    - 任务会被调度到 Ray 集群中的某个 worker 进程执行
object_ref1 = slow_square.remote(5)
object_ref2 = slow_square.remote(10)

print(f"ObjectRef 1: {object_ref1}")
print(f"ObjectRef 2: {object_ref2}")
print("Remote functions submitted, waiting for results...")

# 4. 使用 ray.get() 获取结果
#    - ray.get() 会阻塞，直到对应的远程任务完成并返回结果
result1 = ray.get(object_ref1)
result2 = ray.get(object_ref2)

print(f"Result 1: {result1}") # 输出: Result 1: 25
print(f"Result 2: {result2}") # 输出: Result 2: 100

# 关闭 Ray (可选，通常在脚本末尾)
# ray.shutdown()