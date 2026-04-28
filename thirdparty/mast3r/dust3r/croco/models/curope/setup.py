# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Explicit gencode list — `torch.cuda.get_gencode_flags()` returns every arch
# PyTorch was compiled for, including sm_100 (Blackwell) which nvcc 12.4
# cannot target.
all_cuda_archs = [
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_75,code=sm_75',
    '-gencode', 'arch=compute_80,code=sm_80',
    '-gencode', 'arch=compute_86,code=sm_86',
    '-gencode', 'arch=compute_89,code=sm_89',
    '-gencode', 'arch=compute_90,code=sm_90',
]

setup(
    name = 'curope',
    ext_modules = [
        CUDAExtension(
                name='curope',
                sources=[
                    "curope.cpp",
                    "kernels.cu",
                ],
                extra_compile_args = dict(
                    nvcc=['-O3','--ptxas-options=-v',"--use_fast_math",
                          '-D_GLIBCXX_USE_CXX11_ABI=0']+all_cuda_archs,
                    cxx=['-O3','-D_GLIBCXX_USE_CXX11_ABI=0'])
                )
    ],
    cmdclass = {
        'build_ext': BuildExtension
    })
