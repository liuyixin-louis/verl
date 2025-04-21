# %% [markdown]
# # VeRL Ray API Tutorial

# %% [markdown]
# ## Chapter 1: Ray Basics

# %%
import os

# %%
import ray
import torch
import warnings
warnings.filterwarnings('ignore')

# %%
# Build a local ray cluster. The head node and worker node are on this machine
ray.init()

# %% [markdown]
# Implement an Accumulator class.

# %%
@ray.remote
class Accumulator:
    def __init__(self):
        self.value = 0
        
    def add(self, x):
        self.value += x
        
    def get_value(self):
        return self.value

# %%
# Instantiate an accumulator. Accumulator can be viewed as a process, acting as an RPC service.
accumulator = Accumulator.remote()

# %%
value_ref = accumulator.get_value.remote() # Check the current value. Note that this function returns immediately and does not actually wait for the remote execution to complete.
# Get the value
value = ray.get(value_ref)
print(value)

# %%
# Accumulate, then check the result.
accumulator.add.remote(10)  # Similarly, the 'add' here will return immediately.
new_value = ray.get(accumulator.get_value.remote())
print(new_value)

# %% [markdown]
# ## Chapter 2: Resource Pool and RayWorkerGroup
# In the previous example, it was a simple single-process worker. 
# In this example, we implement a worker with a GPU and form a RayWorkerGroup. Within this RayWorkerGroup, we implement a simple operation of an accumulator.

# %%
from verl.single_controller.ray.base import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup, merge_resource_pool
from verl.single_controller.base import Worker

# %%
resource_pool = RayResourcePool([4], use_gpu=True)

# %%
@ray.remote
class GPUAccumulator(Worker):

    def __init__(self) -> None:
        super().__init__()
        # The initial value of each rank is the same as the rank
        self.value = torch.zeros(size=(1,), device='cuda') + self.rank

    def add(self, x):
        self.value += x
        print(f'rank {self.rank}, value: {self.value}')
        return self.value.cpu()


# %%
# Each worker's initial value is its rank, and then each rank's value is incremented by 1, so the values obtained on each rank are [1, 2, 3, 4]
class_with_args = RayClassWithInitArgs(cls=GPUAccumulator)
worker_group = RayWorkerGroup(resource_pool, class_with_args)
print(worker_group.execute_all_sync('add', x=[1,1,1,1]))

# %% [markdown]
# The principle of parameter passing: The input parameter is a list of length world_size, where each element in the list is dispatched respectively to each worker in the RayWorkerGroup. 
# The return parameter is also a list, corresponding to the return value of each worker.

# %% [markdown]
# ### GPU Resource Sharing

# %% [markdown]
# RayWorkerGroups mapped to the same resource pool share the GPU. In this example, we implement three resource pools: the first occupies 4 GPUs, the second also occupies 4 GPUs, and the last occupies all 8 GPUs. Among them, the first resource pool reuses the resource pool mentioned above.

# %%
# Create a new resource pool and then merge the newly created resource pool with the previous one.
resource_pool_1 = RayResourcePool([4], use_gpu=True, name_prefix='a')
resource_pool_merge = merge_resource_pool(resource_pool, resource_pool_1)

# %%
# Establish a RayWorkerGroup on the newly created resource pool.
worker_group_1 = RayWorkerGroup(resource_pool_1, class_with_args)
worker_group_merge = RayWorkerGroup(resource_pool_merge, class_with_args)

# %%
# Run 'add' on the second set of 4 GPUs; the result should be [2, 3, 4, 5].
output_1 = worker_group_1.execute_all_sync('add', x=[2,2,2,2])
print(output_1)

# %%
# Run 'add' on the merged set of 8 GPUs; the result should be [3, 4, 5, 6, 7, 8, 9, 10].
output_merge = worker_group_merge.execute_all_sync('add', x=[3,3,3,3,3,3,3,3])
print(output_merge)

# %%
print(worker_group.world_size, worker_group_1.world_size, worker_group_merge.world_size)

# %% [markdown]
# ## Chapter 3: Data Dispatch, Execution and Collection

# %% [markdown]
# In the above example, we used the `execute_all_sync` function in the RayWorkerGroup to dispatch data from the driver to each worker. This is very inconvenient for coding. 
# In this chapter, we use the form of function decorators to allow RayWorkerGroup to directly call functions written in the Worker, and to greatly simplify parameter passing.

# %%
from verl.single_controller.base.decorator import register, Dispatch, Execute

# %%
@ray.remote
class GPUAccumulatorDecorator(Worker):

    def __init__(self) -> None:
        super().__init__()
        # The initial value of each rank is the same as the rank
        self.value = torch.zeros(size=(1,), device='cuda') + self.rank
    
    # map from a single input to all the worker
    @register(Dispatch.ONE_TO_ALL)
    def add(self, x):
        print(x)
        self.value = self.value + x
        print(f'rank {self.rank}, value: {self.value}')
        return self.value.cpu()

# %%
class_with_args = RayClassWithInitArgs(cls=GPUAccumulatorDecorator)
gpu_accumulator_decorator = RayWorkerGroup(resource_pool_merge, class_with_args)

# %%
# As we can see, 10 is automatically dispatched to each Worker in this RayWorkerGroup.
print(gpu_accumulator_decorator.add(x=10))

# %% [markdown]
# ### Custom Dispatch, Collection
# Users can customize `dispatch` and `collection` function. You only need to write the `dispatch_fn` and `collect_fn` functions yourself. We also support executing RPC only on rank_zero, with specific examples provided below.

# %%
from verl.single_controller.base.decorator import register, Dispatch, collect_all_to_all, Execute

# %%
def two_to_all_dispatch_fn(worker_group, *args, **kwargs):
    """
    Assume the input is a list of 2. Duplicate the input interleaved and pass to each worker.
    """
    for arg in args:
        assert len(arg) == 2
        for i in range(worker_group.world_size - 2):
            arg.append(arg[i % 2])
    for k, v in kwargs.items():
        assert len(v) == 2
        for i in range(worker_group.world_size - 2):
            v.append(v[i % 2])
    return args, kwargs


@ray.remote
class TestActor(Worker):
    # TODO: pass *args and **kwargs is bug prone and not very convincing
    def __init__(self, x) -> None:
        super().__init__()
        self._x = x

    def foo(self, y):
        return self._x + y

    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def foo_rank_zero(self, x, y):
        return self._x + y + x

    @register(dispatch_mode={'dispatch_fn': two_to_all_dispatch_fn, 'collect_fn': collect_all_to_all})
    def foo_custom(self, x, y):
        return self._x + y + x

# %%
class_with_args = RayClassWithInitArgs(cls=TestActor, x=2)
worker_group = RayWorkerGroup(resource_pool, class_with_args)

# %%
output_ref = worker_group.foo_custom(x=[1, 2], y=[5, 6])
assert output_ref == [8, 10, 8, 10]

output_ref = worker_group.foo_rank_zero(x=1, y=2)
assert output_ref == 5

# %%
print(gpu_accumulator_decorator.world_size)

# %%
# Shutdown ray cluster
ray.shutdown()

# %% [markdown]
# ## Chapter 4: NVMegatronRayWorkerGroup

# %% [markdown]
# Due to the Ray issue, we can only support max_colocate_count=1 in RayResourcePool for now. 
# This means that each GPU can only have one process.
# We can support max_colocate > 1 when applying this pull request: https://github.com/ray-project/ray/pull/44385

# %% [markdown]
# Therefore, we need to restart the ray and initialize a new resource_pool to demonstrate the **NVMegatronRayWorkerGroup**

# %%
# Build a local ray cluster. The head node and worker node are on this machine
ray.init()

# %% [markdown]
# Finally, we implement a `NVMegatronRayWorkerGroup`, within which we create a Megatron and then run a tensor parallel (tp) split Llama mlp layer. Here, we use a complex dispatch mode, `Megatron_COMPUTE`. This dispatch mode assumes that user passes the data partitioned by DP dimension. The data is dispatched to all tp/pp ranks within the same dp group, and ultimately only collects output data from tp=0 and the last pp. In this way, for users that only write code on the driver, the Megatron behind the RPC becomes transparent.

# %%
import os
import sys
import site


current_pythonpath = os.environ.get('PYTHONPATH', '')

new_path = '/opt/tiger/Megatron-LM'

if current_pythonpath:
    new_pythonpath = f'{new_path}:{current_pythonpath}'
else:
    new_pythonpath = new_path

os.environ['PYTHONPATH'] = new_pythonpath

print(new_path)
sys.path.append(new_path)

import megatron
print(megatron.__file__)

# %%
from verl.single_controller.base.decorator import register, Dispatch, Execute
from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
from verl.single_controller.base.megatron.worker import MegatronWorker
from verl.single_controller.ray.base import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup
from omegaconf import OmegaConf
from megatron.core import parallel_state as mpu

# %%
resource_pool = RayResourcePool([4], use_gpu=True, max_colocate_count=1)

# %%
@ray.remote
class MLPLayerWorker(MegatronWorker):
    def __init__(self):
        super().__init__()
        rank = int(os.environ['LOCAL_RANK'])
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(rank)

        mpu.initialize_model_parallel(
                tensor_model_parallel_size=4,
                pipeline_model_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=1,
                expert_model_parallel_size=1,
                nccl_communicator_config_path=None,
            )
        from megatron.core import tensor_parallel
        tensor_parallel.model_parallel_cuda_manual_seed(10)


    @register(Dispatch.ONE_TO_ALL)
    def init_model(self, config):
        from omegaconf import OmegaConf
        from verl.utils.megatron_utils import init_model_parallel_config
        from verl.models.llama.megatron.layers import ParallelLlamaMLP
        megatron_config = OmegaConf.create({
            'sequence_parallel': False,
            'param_dtype': 'fp32',
            'tensor_model_parallel_size': mpu.get_tensor_model_parallel_world_size(),
            'pipeline_model_parallel_rank': mpu.get_pipeline_model_parallel_rank(),
            'pipeline_model_parallel_size': mpu.get_pipeline_model_parallel_world_size(),
            'virtual_pipeline_model_parallel_rank': mpu.get_virtual_pipeline_model_parallel_rank(),
            'virtual_pipeline_model_parallel_size': mpu.get_virtual_pipeline_model_parallel_world_size()
        })

        megatron_config = init_model_parallel_config(megatron_config)
        self.parallel_layer = ParallelLlamaMLP(config=config, megatron_config=megatron_config)
    
    @register(Dispatch.ONE_TO_ALL)
    def get_weights(self):
        output = {}
        for key, val in self.parallel_layer.named_parameters():
            output[key] = val
        return output
    
    @register(Dispatch.MEGATRON_COMPUTE)
    def run_layer(self, x):
        x = x.to('cuda')
        y = self.parallel_layer(x)
        return y

# %%
layer_cls = RayClassWithInitArgs(cls=MLPLayerWorker)
layer_worker_group = NVMegatronRayWorkerGroup(resource_pool=resource_pool,
                                            ray_cls_with_init=layer_cls,
                                            )


# %%
print(layer_worker_group.world_size, layer_worker_group.tp_size, layer_worker_group.pp_size, layer_worker_group.dp_size)

# %%
ffn_hidden_size = 11008
batch_size = 16
seq_len = 2048
hidden_size = 4096

config = OmegaConf.create({
    'hidden_size': hidden_size,
    'intermediate_size': ffn_hidden_size,
    'hidden_act': 'silu',
    'pretraining_tp': 1,
    'tp': layer_worker_group.tp_size,
})

# %%
x = torch.rand(size=(seq_len, batch_size, hidden_size), dtype=torch.float32)

# %%
layer_worker_group.init_model(config)

# %%
output = layer_worker_group.run_layer([x])  # This must be a list of size 1, ensuring that the input equals the data parallel (dp).
print(output[0].shape)

# %%
# Shutdown ray cluster
ray.shutdown()


