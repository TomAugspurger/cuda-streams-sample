
# /// script
# requires-python = ">=3.12.0,<3.13"
# dependencies = [
#   "numba-cuda[cu12]",
#   "cupy-cuda12x",
#   "numpy",
#   "nvtx",
# ]
# ///
"""
An example that uses asyncio to manage a stream of operaitons.

1. The producer (Host)
2. Host to Device Copier
3. Device Consumer
"""
import asyncio
import math

import cupy
import numba.cuda
import numpy as np
import nvtx



@numba.cuda.jit
def incr(a: numba.cuda.devicearray.DeviceNDArray):
    pos = numba.cuda.grid(1)
    if pos < a.size:
        # Adjust the compute intensity so that the kernel takes
        #  longer than the memcpy.
        for i in range(50000): 
            a[pos] += a[pos] + pos + i


async def producer_chunk(shape=(1048576,), dtype="float32", *, pool: cupy.cuda.pinned_memory.PinnedMemoryPool = None, strides=None, order="C"):
    pool = pool or cupy.get_default_pinned_memory_pool()
    # TODO: Throttle
    # ~15ms for 4MB. Compare with
    # ~1.2us for numpy.empty...
    # ~2.1us for cupy's pinned pool malloc...
    dtype = np.dtype(dtype)
    buffer = pool.malloc(math.prod(shape) * dtype.itemsize)
    return np.ndarray(shape=shape, strides=strides, order=order, dtype=dtype, buffer=buffer)


async def copier(host_array: np.ndarray, stream: numba.cuda.cudadrv.driver.Stream) -> numba.cuda.devicearray.DeviceNDArray:
    # ~ 850us
    return numba.cuda.to_device(host_array, stream=stream)


async def consumer(device_array: numba.cuda.devicearray.DeviceNDArray, stream: numba.cuda.cudadrv.driver.Stream) -> None:
    incr[256, 1, stream](device_array)
    return device_array


async def pipeline(stream: numba.cuda.cudadrv.driver.Stream):
    # stream = numba.cuda.stream()
    rng = nvtx.start_range(message="chunk")
    host_array = await producer_chunk()
    device_array = await copier(host_array, stream)
    result = await consumer(device_array, stream)
    return result, stream, rng



async def main():

    with nvtx.annotate("warmup"):
        stream = numba.cuda.stream()
        _, _, rng = await pipeline(stream)
        stream.synchronize()
        nvtx.end_range(rng)
    
    
    # Total number of chunks to process.
    n_chunks = 20

    # Number of concurrent streams to use.
    n_streams = 20

    n_chunks_per_stream = n_chunks // n_streams
    streams = [numba.cuda.stream() for _ in range(n_streams)]
    coros = []

    for stream in streams:
        for _ in range(n_chunks_per_stream):
            coros.append(pipeline(stream))
        # the remainder from n_chunks not being divisible by n_streams
        for _ in range(n_chunks % n_streams):
            coros.append(pipeline(stream))

    results = await asyncio.gather(*coros)
    for result, stream, rng in results:
        stream.synchronize()
        nvtx.end_range(rng)

    print("done")


if __name__ == "__main__":
    asyncio.run(main())
