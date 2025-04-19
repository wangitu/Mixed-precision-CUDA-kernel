from setuptools import Extension, setup
from torch.utils import cpp_extension

setup(
    name="quant_cuda",
    ext_modules=[
        cpp_extension.CUDAExtension(
            "quant_cuda", ["csrc/quant_cuda.cpp", "csrc/quant_cuda_kernel.cu"]
        )
    ],
    extra_compile_args={
        'nvcc': [
            '-Xptxas', '-O3',        # 最大优化级别
            # '--use_fast_math'        # 快速数学
        ]
    },
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
