"""
This example uses numba-cuda to transfer a set of chunks from pinned host memory to device memory.

It shows good performance.
"""
import nvtx
import numba.cuda
import numpy as np
import time


def main():
    TOTAL_SIZE = 256 * 1024 * 1024  # 256 MB
    NUM_CHUNKS = 8
    CHUNK_SIZE = TOTAL_SIZE // NUM_CHUNKS
    NUM_FLOATS_PER_CHUNK = CHUNK_SIZE // np.dtype(np.float32).itemsize
    streams = [numba.cuda.stream() for _ in range(NUM_CHUNKS)]

    with nvtx.annotate("Initialization"):
        with nvtx.annotate("Allocate Pinned Host Memory"):
            host_buffers = [
                numba.cuda.pinned_array(CHUNK_SIZE, dtype=np.dtype(np.ubyte)) for _ in range(NUM_CHUNKS)
            ]

        with nvtx.annotate("Initialize Host Data"):
            for i, host_buffer in enumerate(host_buffers):
                host_buffer[:] = np.arange(NUM_FLOATS_PER_CHUNK, dtype=np.float32).view(np.dtype(np.ubyte))

        with nvtx.annotate("Allocate Device Memory"):
            device_buffers = [
                numba.cuda.device_array(CHUNK_SIZE, dtype=np.dtype(np.ubyte), stream=stream) for stream in streams
            ]

    transfer_start = time.perf_counter()
    with nvtx.annotate("Transfer"):
        for i, (host_buffer, device_buffer, stream) in enumerate(zip(host_buffers, device_buffers, streams, strict=True)):
            with nvtx.annotate(f"Transfer Chunk {i}"):
                device_buffer.copy_to_device(host_buffer, stream=stream)

    with nvtx.annotate("Synchronize"):
        for stream in streams:
            stream.synchronize()
    transfer_end = time.perf_counter()
    
    with nvtx.annotate("Cleanup"):
        del host_buffers
        del device_buffers
        for stream in streams:
            stream.synchronize()
        del streams

    print("Transfer throughput: {:.2f} GB/s".format(TOTAL_SIZE / (transfer_end - transfer_start) / 1024 / 1024 / 1024))


if __name__ == "__main__":
    main()

