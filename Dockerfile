FROM rapidsai/ci-conda:cuda13.0.2-rockylinux8-py3.13

RUN rpm --import https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    dnf install -y 'dnf-command(config-manager)' && \
    dnf config-manager --add-repo "https://developer.download.nvidia.com/devtools/repos/rhel$(source /etc/os-release; echo ${VERSION_ID%%.*})/$(rpm --eval '%{_arch}' | sed s/aarch/arm/)/" && \
    dnf install -y nsight-systems

RUN conda install -c conda-forge \
    cupy \
    cuda-runtime \
    cuda-compiler \
    cuda-libraries \
    cuda-libraries-dev \
    ipython \
    numcodecs \
    numpy \
    nvcomp \
    nvtx \
    rmm \
    zarr \
    libnvcomp-dev \
    Cython \
    setuptools \
    wheel \
    pip

ENV CUDA_HOME=/opt/conda/
ENV CUPY_CUDA_ARRAY_INTERFACE_SYNC=0
ENV CUDA_ARRAY_INTERFACE_SYNC=0
WORKDIR /workspace

COPY . .

# Run with
# docker run -it --rm --privileged -v $(pwd):/workspace --gpus 1 toaugspurger/cuda-streams-sample
# then

# cd nvcomp_minimal && python setup.py build_ext --inplace && python -m pip install --no-deps --no-build-isolation . && cd ..
# CUDA_VISIBLE_DEVICES=0 CUPY_CUDA_ARRAY_INTERFACE_SYNC=0 CUDA_ARRAY_INTERFACE_SYNC=0 nsys profile --cudabacktrace=all --python-sampling=true --python-backtrace=cuda -o /workspace/zarr-shards -f true --trace cuda,nvtx --gpu-metrics-devices=cuda-visible python zarr_shards.py --benchmark-cpu --benchmark-zarr-gpu --benchmark-gpu

