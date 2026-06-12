#!/usr/bin/env bash
set -euo pipefail

source /cluster/project2/gcreduce_data/anaconda/etc/profile.d/conda.sh
conda activate llm

export PATH=/share/apps/cuda-11.8/bin:/share/apps/cmake-3.24/bin:/usr/sbin:/share/apps/gcc-9.2.0/bin:${PATH}
export LD_LIBRARY_PATH=/share/apps/cuda-11.8/lib64:/cluster/project2/gcreduce_data/cudnn-linux-x86_64-8.9.5.29_cuda11-archive/lib:/share/apps/gcc-9.2.0/lib:/cluster/project2/gcreduce_data/anaconda/lib:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/share/apps/cuda-11.8

cd /cluster/project2/gcreduce_data/dynamiq_artifact/cuda_kernels
python tests/benchmark_scaling_quantization.py
