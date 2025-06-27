"""
A simple example that shows pipelining (overlapping) memory copies and kernel execution.

Run with:

    nsys profile -o stream --trace cuda,nvtx --force-overwrite=true python numba-simple.py

Observe that the memcpy and kernel comptue overlaps.
"""

# first, the basics. Get something that overlaps memory copy and kernel launch.

import cupy as cp
import numba.cuda
import numpy as np


@numba.cuda.jit
def kernel(a, b, c):
    tid = numba.cuda.grid(1)
    size = len(c)
    if tid < size:
        # just repeat this till it's ~ equal to how long the memcpy takes
        for i in range(300):
            c[tid] = a[tid] + b[tid]


def main():
    N_STREAMS = 4
    N = 1_000_000_000 // 4  # 1 GB
    streams = [numba.cuda.stream() for _ in range(N_STREAMS)]
    outputs = [numba.cuda.pinned_array(N, dtype="uint32") for _ in range(N_STREAMS)]  # type: ignore[arg-type]
    host_arrays = [
        numba.cuda.pinned_array(N, dtype="uint32")  # type: ignore[arg-type]
        for _ in range(N_STREAMS)
    ]
    x = np.arange(N, dtype="uint32")
    for host_array in host_arrays:
        host_array[:] = x

    nthreads = 256
    # Enough blocks to cover the entire vector depending on its length
    nblocks = (N // nthreads) + 1

    # warmup
    kernel[256, 1](  # type: ignore[call-overload]
        cp.arange(1, dtype="uint32"),
        cp.arange(1, dtype="uint32"),
        cp.empty(1, dtype="uint32"),
    )

    for stream, host_array, output in zip(streams, host_arrays, outputs):
        a_dev = numba.cuda.to_device(host_array, stream=stream)
        o_dev = numba.cuda.to_device(output, stream=stream)
        kernel[nblocks, nthreads, stream](a_dev, a_dev, o_dev)  # type: ignore[call-overload]

    for stream in streams:
        stream.synchronize()


if __name__ == "__main__":
    main()
