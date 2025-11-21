"""
This example uses cupy to transfer a set of chunks from pinned host memory to device memory.

It shows good performance.
"""
import nvtx
import cupy
import cupyx
import numpy as np
import time


def main():
    TOTAL_SIZE = 256 * 1024 * 1024  # 256 MB
    NUM_CHUNKS = 8
    CHUNK_SIZE = TOTAL_SIZE // NUM_CHUNKS
    NUM_FLOATS_PER_CHUNK = CHUNK_SIZE // np.dtype(np.float32).itemsize
    streams = [cupy.cuda.stream.Stream() for _ in range(NUM_CHUNKS)]
    host_arrays = []

    with nvtx.annotate("Initialization"):
        with nvtx.annotate("Allocate Pinned Host Memory"):
            host_arrays = [
                cupyx.empty_pinned(NUM_FLOATS_PER_CHUNK, dtype=np.float32)
                for _ in range(NUM_CHUNKS)
            ]

        with nvtx.annotate("Initialize Host Data"):
            for host_array in host_arrays:
                host_array[:] = np.arange(NUM_FLOATS_PER_CHUNK, dtype=np.float32).view(host_array.dtype)

        with nvtx.annotate("Allocate Device Memory"):
            device_buffers = []
            for stream in streams:
                with stream:
                    device_buffers.append(cupy.empty(CHUNK_SIZE, dtype=np.dtype(np.byte)))

    transfer_start = time.perf_counter()
    with nvtx.annotate("Transfer"):
        for i, (host_array, device_buffer, stream) in enumerate(zip(host_arrays, device_buffers, streams, strict=True)):
            with nvtx.annotate(f"Transfer Chunk {i}"), stream:
                device_buffer.set(host_array.view(device_buffer.dtype), stream=stream)

    with nvtx.annotate("Synchronize"):
        for stream in streams:
            stream.synchronize()
    transfer_end = time.perf_counter()
    
    with nvtx.annotate("Cleanup"):
        del device_buffers
        del host_arrays
        for stream in streams:
            stream.synchronize()
        del streams

    print("Transfer throughput: {:.2f} GB/s".format(TOTAL_SIZE / (transfer_end - transfer_start) / 1024 / 1024 / 1024))


if __name__ == "__main__":
    main()

