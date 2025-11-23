"""
CUDA Pinned Memory Transfer Demo - Python Version with cuda.core
Demonstrates non-blocking host-to-device transfers using cuda.core's Pythonic API
"""

import numpy as np
import time
import ctypes
from typing import List

import nvtx
from cuda.bindings import runtime
from cuda.core.experimental import Device


class PinnedMemoryBuffer:
    """RAII wrapper for pinned host memory"""

    def __init__(self, size_bytes: int):
        self.size = size_bytes
        err, self.ptr = runtime.cudaMallocHost(size_bytes)
        if err != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(
                f"Failed to allocate pinned memory: {runtime.cudaGetErrorString(err)}"
            )

    def __del__(self):
        if hasattr(self, "ptr") and self.ptr:
            runtime.cudaFreeHost(self.ptr)

    def as_numpy_array(self, dtype, shape):
        """Return a numpy array view of the pinned memory"""
        # Convert the CUDA pointer (int) to a ctypes pointer
        itemsize = np.dtype(dtype).itemsize
        total_items = np.prod(shape)
        total_bytes = itemsize * total_items

        # Create a ctypes pointer from the address
        ptr = ctypes.cast(self.ptr, ctypes.POINTER(ctypes.c_ubyte))

        # Create numpy array from the ctypes pointer
        arr = np.ctypeslib.as_array(ptr, shape=(total_bytes,))

        # View as the correct dtype and reshape
        return arr.view(dtype).reshape(shape)

    @property
    def device_ptr(self):
        """Get the device pointer (for cudaMemcpyAsync)"""
        return self.ptr


class DeviceMemoryBuffer:
    """RAII wrapper for device memory"""

    def __init__(self, size_bytes: int):
        self.size = size_bytes
        err, self.ptr = runtime.cudaMalloc(size_bytes)
        if err != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(
                f"Failed to allocate device memory: {runtime.cudaGetErrorString(err)}"
            )

    def __del__(self):
        if hasattr(self, "ptr") and self.ptr:
            runtime.cudaFree(self.ptr)

    @property
    def device_ptr(self):
        return self.ptr


def measure_time(func):
    """Decorator to measure execution time"""

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, (end - start)

    return wrapper


def main():
    print("=== CUDA Pinned Memory Transfer Demo (Python with cuda.core) ===")
    print()

    # Mark initialization phase
    nvtx.push_range("Initialization")

    # Configuration
    TOTAL_SIZE = 256 * 1024 * 1024  # 256 MB
    NUM_CHUNKS = 8
    CHUNK_SIZE = TOTAL_SIZE // NUM_CHUNKS
    NUM_FLOATS = TOTAL_SIZE // np.dtype(np.float32).itemsize

    print(f"Configuration:")
    print(f"  Total buffer size: {TOTAL_SIZE // (1024 * 1024)} MB")
    print(f"  Number of chunks: {NUM_CHUNKS}")
    print(f"  Chunk size: {CHUNK_SIZE // (1024 * 1024)} MB")
    print()

    # Get default device using cuda.core
    device = Device()
    device.set_current()

    # Allocate pinned host memory
    nvtx.push_range("Allocate Pinned Host Memory")
    host_buffer = PinnedMemoryBuffer(TOTAL_SIZE)
    print(f"✓ Allocated {TOTAL_SIZE // (1024 * 1024)} MB of pinned host memory")
    nvtx.pop_range()

    # Allocate device memory
    nvtx.push_range("Allocate Device Memory")
    device_buffer = DeviceMemoryBuffer(TOTAL_SIZE)
    print(f"✓ Allocated {TOTAL_SIZE // (1024 * 1024)} MB of device memory")
    nvtx.pop_range()

    # Initialize host memory with numpy
    nvtx.push_range("Initialize Host Data")
    host_array = host_buffer.as_numpy_array(np.float32, (NUM_FLOATS,))
    host_array[:] = np.arange(NUM_FLOATS, dtype=np.float32)
    print(f"✓ Initialized host data with {NUM_FLOATS:,} float32 values")
    nvtx.pop_range()

    # Create CUDA streams using cuda.core
    nvtx.push_range("Create CUDA Streams")
    # streams: List[Stream] = [Stream() for _ in range(NUM_CHUNKS)]
    streams = [device.create_stream() for _ in range(NUM_CHUNKS)]
    print(f"✓ Created {NUM_CHUNKS} CUDA streams")
    nvtx.pop_range()

    nvtx.pop_range()  # End Initialization
    print()

    # Perform chunked async transfers
    nvtx.push_range("Async H2D Transfers")
    print("--- Starting Async Transfers ---")

    transfer_start = time.perf_counter()

    for i in range(NUM_CHUNKS):
        nvtx.push_range(f"Transfer Chunk {i}")

        offset = i * CHUNK_SIZE

        print(
            f"  Enqueuing chunk {i}/{NUM_CHUNKS - 1} "
            f"(offset: {offset // (1024 * 1024)} MB, "
            f"size: {CHUNK_SIZE // (1024 * 1024)} MB)"
        )

        # Perform async memory copy
        err = runtime.cudaMemcpyAsync(
            device_buffer.device_ptr + offset,
            host_buffer.device_ptr + offset,
            CHUNK_SIZE,
            runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
            streams[i].handle,
        )

        if err != (runtime.cudaError_t.cudaSuccess,):
            error_str = runtime.cudaGetErrorString(err[0])
            raise RuntimeError(f"cudaMemcpyAsync failed: {error_str}")

        nvtx.pop_range()

    enqueue_end = time.perf_counter()
    enqueue_time_us = (enqueue_end - transfer_start) * 1_000_000

    nvtx.pop_range()  # End Async H2D Transfers

    # Synchronize all streams
    nvtx.push_range("Synchronize Streams")
    print("--- Synchronizing Streams ---")

    for i, stream in enumerate(streams):
        stream.sync()

    sync_end = time.perf_counter()
    total_time_ms = (sync_end - transfer_start) * 1000

    # Calculate bandwidth
    bandwidth_gbps = (TOTAL_SIZE / (1024**3)) / (total_time_ms / 1000)

    nvtx.pop_range()

    # Verify data (copy back small amount to check)
    nvtx.push_range("Verification")

    verify_size = 100
    verify_data = np.zeros(verify_size, dtype=np.float32)

    err = runtime.cudaMemcpy(
        verify_data.ctypes.data,
        device_buffer.device_ptr,
        verify_data.nbytes,
        runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    )

    if err != (runtime.cudaError_t.cudaSuccess,):
        error_str = runtime.cudaGetErrorString(err[0])
        raise RuntimeError(f"Verification copy failed: {error_str}")

    expected = np.arange(verify_size, dtype=np.float32)
    correct = np.allclose(verify_data, expected)

    if not correct:
        print(f"✗ Data verification FAILED")
        print(f"  Expected: {expected[:5]}")
        print(f"  Got: {verify_data[:5]}")

    nvtx.pop_range()
    print()

    # Cleanup happens automatically via __del__
    with nvtx.annotate("Cleanup"):
        del expected, verify_data, host_array, device_buffer, host_buffer, streams

    print("Summary:")
    print(f"  • Transferred {TOTAL_SIZE // (1024 * 1024)} MB in {NUM_CHUNKS} chunks")
    print(f"  • Enqueue time: {enqueue_time_us:.1f} μs (non-blocking!)")
    print(f"  • Actual transfer time: {total_time_ms:.2f} ms")
    print(f"  • Bandwidth: {bandwidth_gbps:.2f} GB/s")
    print(f"  • Data integrity: {'✓ PASSED' if correct else '✗ FAILED'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
