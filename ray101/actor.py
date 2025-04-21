import ray
import time

# 1. 初始化 Ray
if not ray.is_initialized():
    ray.init()

# 2. 使用 @ray.remote 装饰器定义一个 Actor 类
@ray.remote
class Counter:
    def __init__(self):
        """Actor 的构造函数"""
        self.value = 0
        breakpoint()
        print("Counter Actor initialized.")

    def increment(self):
        """增加计数器值的方法"""
        self.value += 1
        print(f"Counter incremented to {self.value}")
        return self.value

    def get_value(self):
        """获取当前计数器值的方法"""
        print(f"Getting current value: {self.value}")
        return self.value

# 3. 实例化 Actor
#    - 使用 .remote() 实例化 Actor，这会在集群中启动一个进程来运行这个 Actor 实例
counter_actor = Counter.remote()
print(f"Counter Actor instance created: {counter_actor}")

# 4. 调用 Actor 的方法
#    - 同样使用 .remote() 来调用 Actor 的方法
#    - 这些调用也是异步的，返回 ObjectRef
obj_ref1 = counter_actor.increment.remote()
obj_ref2 = counter_actor.increment.remote()

obj_ref3 = counter_actor.get_value.remote() # 调用 get_value

# 等待方法调用完成并获取结果
result1 = ray.get(obj_ref1)
result2 = ray.get(obj_ref2)
result3 = ray.get(obj_ref3) # 获取 get_value 的结果

print(f"Result from first increment: {result1}")  # 可能输出 1
print(f"Result from second increment: {result2}") # 可能输出 2
print(f"Result from get_value: {result3}")       # 可能输出 2 (取决于执行顺序)

# 再调用一次 get_value 确认状态
final_value_ref = counter_actor.get_value.remote()
final_value = ray.get(final_value_ref)
print(f"Final counter value: {final_value}")      # 应该输出 2

# ray.shutdown()