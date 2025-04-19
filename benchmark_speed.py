import time
import functools

import torch
import quant_cuda
 
 
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute.")
        return result
    return wrapper


batch_size = 16
vec = torch.randn(batch_size, 4096).cuda()
qweight = torch.randint(1, 5, (1530, 13824), dtype=torch.uint8).cuda()
cherry_weight = torch.randn(16, 13824).cuda()
cherry_indices = torch.randperm(16, dtype=torch.int32).sort().values.cuda()
scaling_factors = torch.ones(32, 13824).cuda()
mul = torch.zeros(batch_size, 13824).cuda()

vec_half = vec.half()
weight_half = torch.randn(4096, 13824).cuda().half()

# warm up
quant_cuda.vecquant3matmul_cherryq(vec, qweight, cherry_weight, cherry_indices, scaling_factors, mul, 128, False)


@timer
def pytorch_fp16_matmul():
    for _ in range(100):
        torch.matmul(vec_half, weight_half)
        
@timer
def quant_cuda_matmul():
    for _ in range(100):
        quant_cuda.vecquant3matmul_cherryq(vec, qweight, cherry_weight, cherry_indices, scaling_factors, mul, 128, False)


pytorch_fp16_matmul()
quant_cuda_matmul()
