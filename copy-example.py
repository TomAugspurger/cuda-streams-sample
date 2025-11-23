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
Example showing the fastest ways to copy NumPy buffers from host to GPU device(s).

Key optimizations for maximum throughput:
1. Use pinned (page-locked) memory for host buffers
2. Use async memory copies with CUDA streams
3. Distribute across multiple GPUs if available
4. Use multiple streams per GPU to hide latency
5. Read data from disk efficiently (overlapping I/O with GPU transfers)
"""

import concurrent.futures
import os
import pathlib
import tempfile
import time

import cupy as cp
import numba.cuda
import numpy as np
import nvtx


def get_available_devices():
    """Get list of available CUDA devices based on CUDA_VISIBLE_DEVICES."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    n_devices = len(numba.cuda.gpus.lst)

    if cuda_visible:
        print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    print(f"Available devices: {n_devices}")

    return list(range(n_devices))


def create_test_data_files(data_dir, n_files, file_size_mb):
    """
    Create test data files on disk for benchmarking.

    Args:
        data_dir: Directory to store test files
        n_files: Number of files to create
        file_size_mb: Size of each file in MB

    Returns:
        List of file paths
    """
    data_dir = pathlib.Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nCreating {n_files} test files ({file_size_mb} MB each)...")

    file_paths = []
    buffer_shape = (file_size_mb * 1024 * 1024 // 4,)  # float32 = 4 bytes

    start = time.perf_counter()
    for i in range(n_files):
        file_path = data_dir / f"test_data_{i:04d}.npy"

        # Create random data and save to disk
        data = np.random.randn(*buffer_shape).astype(np.float32)
        np.save(file_path, data)

        file_paths.append(file_path)

    elapsed = time.perf_counter() - start
    total_mb = n_files * file_size_mb

    print(f"Created {n_files} files ({total_mb} MB total) in {elapsed:.2f} s")
    print(f"Write bandwidth: {total_mb / elapsed:.2f} MB/s")

    return file_paths


def read_file_to_pinned_memory(file_path, pinned_pool):
    """
    Read a NumPy file directly into pinned memory.

    This is critical for performance: reading into pinned memory avoids
    an extra copy from pageable to pinned memory.

    Args:
        file_path: Path to .npy file
        pinned_pool: CuPy pinned memory pool

    Returns:
        NumPy array backed by pinned memory
    """
    # First, load to get shape and dtype (reads header only)
    with open(file_path, "rb") as f:
        version = np.lib.format.read_magic(f)
        shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)

        # Allocate pinned memory
        nbytes = np.prod(shape) * dtype.itemsize
        pinned_buffer = pinned_pool.malloc(nbytes)

        # Read data directly into pinned buffer
        array = np.ndarray(shape=shape, dtype=dtype, buffer=pinned_buffer)
        np.copyto(array, np.load(file_path))

    return array


def read_files_sequential(file_paths, pinned_pool):
    """
    Read files sequentially into pinned memory.
    Simple but doesn't overlap I/O with GPU transfers.
    """
    print("\nReading files sequentially into pinned memory...")

    start = time.perf_counter()
    host_arrays = []

    for file_path in file_paths:
        array = read_file_to_pinned_memory(file_path, pinned_pool)
        host_arrays.append(array)

    elapsed = time.perf_counter() - start
    total_mb = sum(arr.nbytes for arr in host_arrays) / (1024 * 1024)

    print(f"Read {len(file_paths)} files ({total_mb:.0f} MB) in {elapsed:.2f} s")
    print(f"Read bandwidth: {total_mb / elapsed:.2f} MB/s")

    return host_arrays


def read_files_threaded(file_paths, pinned_pool, max_workers=4):
    """
    Read files concurrently using a thread pool.
    Can improve I/O throughput, especially with fast storage (NVMe, RAID).
    """
    print(f"\nReading files with thread pool (workers={max_workers})...")

    def read_worker(file_path):
        return read_file_to_pinned_memory(file_path, pinned_pool)

    start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        host_arrays = list(executor.map(read_worker, file_paths))

    elapsed = time.perf_counter() - start
    total_mb = sum(arr.nbytes for arr in host_arrays) / (1024 * 1024)

    print(f"Read {len(file_paths)} files ({total_mb:.0f} MB) in {elapsed:.2f} s")
    print(f"Read bandwidth: {total_mb / elapsed:.2f} MB/s")

    return host_arrays


def benchmark_sync_copy(host_data, device_index=0, n_iterations=10):
    """Baseline: synchronous copy (slowest)."""
    print("\n" + "=" * 60)
    print("Benchmark 1: Synchronous Copy (Baseline)")
    print("=" * 60)

    with numba.cuda.gpus[device_index]:
        # Warmup
        _ = numba.cuda.to_device(host_data)
        numba.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(n_iterations):
            device_array = numba.cuda.to_device(host_data)
            numba.cuda.synchronize()
        elapsed = time.perf_counter() - start

        bytes_transferred = host_data.nbytes * n_iterations
        bandwidth_gbps = (bytes_transferred / elapsed) / 1e9

        print(f"Time per copy: {elapsed / n_iterations * 1000:.2f} ms")
        print(f"Bandwidth: {bandwidth_gbps:.2f} GB/s")
        print(f"Total time: {elapsed:.3f} s")

        return device_array


def benchmark_async_copy_single_stream(host_data, device_index=0, n_iterations=10):
    """Async copy with a single stream."""
    print("\n" + "=" * 60)
    print("Benchmark 2: Async Copy (Single Stream)")
    print("=" * 60)

    with numba.cuda.gpus[device_index]:
        stream = numba.cuda.stream()

        # Warmup
        _ = numba.cuda.to_device(host_data, stream=stream)
        stream.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(n_iterations):
            device_array = numba.cuda.to_device(host_data, stream=stream)
        stream.synchronize()
        elapsed = time.perf_counter() - start

        bytes_transferred = host_data.nbytes * n_iterations
        bandwidth_gbps = (bytes_transferred / elapsed) / 1e9

        print(f"Time per copy: {elapsed / n_iterations * 1000:.2f} ms")
        print(f"Bandwidth: {bandwidth_gbps:.2f} GB/s")
        print(f"Total time: {elapsed:.3f} s")

        return device_array


def benchmark_async_copy_multi_stream(host_data_list, device_index=0, n_streams=4):
    """Async copy with multiple streams (overlapping transfers)."""
    print("\n" + "=" * 60)
    print(f"Benchmark 3: Async Copy (Multiple Streams, n={n_streams})")
    print("=" * 60)

    n_buffers = len(host_data_list)

    with numba.cuda.gpus[device_index]:
        # Create multiple streams
        streams = [numba.cuda.stream() for _ in range(n_streams)]

        # Warmup
        for i, host_data in enumerate(host_data_list[:n_streams]):
            stream = streams[i % n_streams]
            _ = numba.cuda.to_device(host_data, stream=stream)
        for stream in streams:
            stream.synchronize()

        # Benchmark
        start = time.perf_counter()
        device_arrays = []
        for i, host_data in enumerate(host_data_list):
            stream = streams[i % n_streams]
            device_array = numba.cuda.to_device(host_data, stream=stream)
            device_arrays.append((device_array, stream))

        # Synchronize all streams
        for stream in streams:
            stream.synchronize()
        elapsed = time.perf_counter() - start

        total_bytes = sum(h.nbytes for h in host_data_list)
        bandwidth_gbps = (total_bytes / elapsed) / 1e9

        print(f"Buffers copied: {n_buffers}")
        print(f"Time per buffer: {elapsed / n_buffers * 1000:.2f} ms")
        print(f"Bandwidth: {bandwidth_gbps:.2f} GB/s")
        print(f"Total time: {elapsed:.3f} s")

        return device_arrays


def benchmark_multi_device_copy(host_data_list, devices, n_streams_per_device=4):
    """
    Fastest approach: distribute copies across multiple GPUs with multiple streams.
    This maximizes PCIe bandwidth utilization.
    """
    print("\n" + "=" * 60)
    print("Benchmark 4: Multi-Device Async Copy")
    print(f"Devices: {len(devices)}, Streams per device: {n_streams_per_device}")
    print("=" * 60)

    n_devices = len(devices)
    n_buffers = len(host_data_list)

    # Create streams for each device
    stream_pools = []
    for device_idx in devices:
        with numba.cuda.gpus[device_idx]:
            stream_pools.append(
                [numba.cuda.stream() for _ in range(n_streams_per_device)]
            )

    # Warmup
    for i, host_data in enumerate(host_data_list[: n_devices * n_streams_per_device]):
        device_idx = devices[i % n_devices]
        stream_idx = (i // n_devices) % n_streams_per_device
        stream = stream_pools[i % n_devices][stream_idx]
        with numba.cuda.gpus[device_idx]:
            _ = numba.cuda.to_device(host_data, stream=stream)

    for streams in stream_pools:
        for stream in streams:
            stream.synchronize()

    # Benchmark
    start = time.perf_counter()
    device_arrays = []

    for i, host_data in enumerate(host_data_list):
        device_idx = devices[i % n_devices]
        stream_idx = (i // n_devices) % n_streams_per_device
        stream = stream_pools[i % n_devices][stream_idx]

        with numba.cuda.gpus[device_idx]:
            device_array = numba.cuda.to_device(host_data, stream=stream)
            device_arrays.append((device_array, stream, device_idx))

    # Synchronize all streams on all devices
    for streams in stream_pools:
        for stream in streams:
            stream.synchronize()

    elapsed = time.perf_counter() - start

    total_bytes = sum(h.nbytes for h in host_data_list)
    bandwidth_gbps = (total_bytes / elapsed) / 1e9

    print(f"Buffers copied: {n_buffers}")
    print(f"Time per buffer: {elapsed / n_buffers * 1000:.2f} ms")
    print(f"Aggregate bandwidth: {bandwidth_gbps:.2f} GB/s")
    print(f"Total time: {elapsed:.3f} s")

    return device_arrays


def benchmark_multi_device_threaded(
    host_data_list, devices, n_streams_per_device=4, max_workers=16
):
    """
    Alternative: use thread pool for concurrent submissions across devices.
    Can sometimes be faster due to better CPU parallelism.
    """
    print("\n" + "=" * 60)
    print("Benchmark 5: Multi-Device with Thread Pool")
    print(f"Devices: {len(devices)}, Streams per device: {n_streams_per_device}")
    print(f"Thread pool workers: {max_workers}")
    print("=" * 60)

    n_devices = len(devices)
    n_buffers = len(host_data_list)

    # Create streams for each device
    stream_pools = []
    for device_idx in devices:
        with numba.cuda.gpus[device_idx]:
            stream_pools.append(
                [numba.cuda.stream() for _ in range(n_streams_per_device)]
            )

    def copy_to_device(host_data, device_idx, stream):
        """Worker function for thread pool."""
        with numba.cuda.gpus[device_idx]:
            return numba.cuda.to_device(host_data, stream=stream), stream, device_idx

    # Warmup
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, host_data in enumerate(
            host_data_list[: min(n_buffers, n_devices * n_streams_per_device)]
        ):
            device_idx = devices[i % n_devices]
            stream_idx = (i // n_devices) % n_streams_per_device
            stream = stream_pools[i % n_devices][stream_idx]
            futures.append(
                executor.submit(copy_to_device, host_data, device_idx, stream)
            )

        for future in concurrent.futures.as_completed(futures):
            _, stream, _ = future.result()
            stream.synchronize()

    # Benchmark
    start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, host_data in enumerate(host_data_list):
            device_idx = devices[i % n_devices]
            stream_idx = (i // n_devices) % n_streams_per_device
            stream = stream_pools[i % n_devices][stream_idx]
            futures.append(
                executor.submit(copy_to_device, host_data, device_idx, stream)
            )

        results = []
        for future in concurrent.futures.as_completed(futures):
            device_array, stream, device_idx = future.result()
            stream.synchronize()
            results.append((device_array, stream, device_idx))

    elapsed = time.perf_counter() - start

    total_bytes = sum(h.nbytes for h in host_data_list)
    bandwidth_gbps = (total_bytes / elapsed) / 1e9

    print(f"Buffers copied: {n_buffers}")
    print(f"Time per buffer: {elapsed / n_buffers * 1000:.2f} ms")
    print(f"Aggregate bandwidth: {bandwidth_gbps:.2f} GB/s")
    print(f"Total time: {elapsed:.3f} s")

    return results


def benchmark_disk_to_gpu_pipelined(
    file_paths,
    devices,
    pinned_pool,
    n_streams_per_device=4,
    io_workers=4,
    gpu_workers=16,
):
    """
    FASTEST END-TO-END: Pipeline that overlaps disk I/O with GPU transfers.

    This is the most realistic and efficient approach:
    1. Read files from disk into pinned memory (I/O thread pool)
    2. Transfer to GPU devices with async streams (GPU thread pool)
    3. Overlap I/O and GPU transfers for maximum throughput

    Args:
        file_paths: List of file paths to read
        devices: List of GPU device indices
        pinned_pool: CuPy pinned memory pool
        n_streams_per_device: Number of CUDA streams per GPU
        io_workers: Number of threads for disk I/O
        gpu_workers: Number of threads for GPU transfers
    """
    print("\n" + "=" * 60)
    print("Benchmark 6: Disk-to-GPU Pipeline (OVERLAPPED I/O)")
    print(f"Devices: {len(devices)}, Streams per device: {n_streams_per_device}")
    print(f"I/O workers: {io_workers}, GPU workers: {gpu_workers}")
    print("=" * 60)

    n_devices = len(devices)
    n_files = len(file_paths)

    # Create streams for each device
    stream_pools = []
    for device_idx in devices:
        with numba.cuda.gpus[device_idx]:
            stream_pools.append(
                [numba.cuda.stream() for _ in range(n_streams_per_device)]
            )

    def read_and_copy_to_gpu(file_path, file_idx):
        """Read file from disk and immediately copy to GPU."""
        # Read into pinned memory
        with nvtx.annotate(f"read-file-{file_idx}"):
            host_array = read_file_to_pinned_memory(file_path, pinned_pool)

        # Determine which device and stream to use
        device_idx = devices[file_idx % n_devices]
        stream_idx = (file_idx // n_devices) % n_streams_per_device
        stream = stream_pools[file_idx % n_devices][stream_idx]

        # Copy to GPU
        with nvtx.annotate(f"copy-to-gpu-{file_idx}"):
            with numba.cuda.gpus[device_idx]:
                device_array = numba.cuda.to_device(host_array, stream=stream)

        return device_array, stream, device_idx, host_array.nbytes

    # Warmup
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=io_workers + gpu_workers
    ) as executor:
        warmup_files = file_paths[: min(n_files, 2)]
        futures = [
            executor.submit(read_and_copy_to_gpu, fp, i)
            for i, fp in enumerate(warmup_files)
        ]
        for future in concurrent.futures.as_completed(futures):
            _, stream, _, _ = future.result()
            stream.synchronize()

    # Benchmark: overlapped I/O and GPU transfers
    start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=io_workers + gpu_workers
    ) as executor:
        # Submit all jobs
        futures = [
            executor.submit(read_and_copy_to_gpu, fp, i)
            for i, fp in enumerate(file_paths)
        ]

        # Collect results as they complete
        results = []
        total_bytes = 0
        for future in concurrent.futures.as_completed(futures):
            device_array, stream, device_idx, nbytes = future.result()
            stream.synchronize()
            results.append((device_array, stream, device_idx))
            total_bytes += nbytes

    elapsed = time.perf_counter() - start

    bandwidth_gbps = (total_bytes / elapsed) / 1e9

    print(f"Files processed: {n_files}")
    print(f"Time per file: {elapsed / n_files * 1000:.2f} ms")
    print(f"Aggregate bandwidth (disk + GPU): {bandwidth_gbps:.2f} GB/s")
    print(f"Total time: {elapsed:.3f} s")

    return results


def main():
    """
    Demonstrate fastest disk-to-GPU copy strategies.
    """
    print("\n" + "=" * 60)
    print("DISK TO GPU COPY BENCHMARK")
    print("=" * 60)

    # Configuration
    BUFFER_SIZE_MB = 64  # Size of each buffer/file in MB
    N_FILES = 40  # Number of files to process
    USE_DISK = True  # Set to False to use in-memory data (faster testing)

    devices = get_available_devices()
    n_devices = len(devices)

    print("\nConfiguration:")
    print(f"  File size: {BUFFER_SIZE_MB} MB")
    print(f"  Number of files: {N_FILES}")
    print(f"  Total data: {BUFFER_SIZE_MB * N_FILES} MB")
    print(f"  Devices: {n_devices}")
    print(f"  Using disk I/O: {USE_DISK}")

    # Setup pinned memory pool
    pinned_pool = cp.cuda.pinned_memory.PinnedMemoryPool()

    if USE_DISK:
        # Create test data files
        print("\n" + "=" * 60)
        print("SETUP: Creating test data files")
        print("=" * 60)

        data_dir = tempfile.mkdtemp(prefix="gpu_copy_test_")
        print(f"Data directory: {data_dir}")

        with nvtx.annotate("create-test-files"):
            file_paths = create_test_data_files(data_dir, N_FILES, BUFFER_SIZE_MB)

        # Benchmark different disk reading strategies
        print("\n" + "=" * 60)
        print("DISK I/O BENCHMARKS")
        print("=" * 60)

        # Read files sequentially
        with nvtx.annotate("read-sequential"):
            host_data_list = read_files_sequential(file_paths, pinned_pool)

        # Read files with threading (for comparison)
        with nvtx.annotate("read-threaded"):
            host_data_list_threaded = read_files_threaded(
                file_paths, pinned_pool, max_workers=8
            )

        # Use the threaded version for GPU benchmarks (typically faster)
        host_data_list = host_data_list_threaded
    else:
        # In-memory data for faster testing
        print("\n" + "=" * 60)
        print("SETUP: Creating in-memory pinned buffers")
        print("=" * 60)

        buffer_size_bytes = BUFFER_SIZE_MB * 1024 * 1024
        buffer_shape = (buffer_size_bytes // 4,)  # float32 = 4 bytes
        total_bytes = buffer_size_bytes * N_FILES

        with nvtx.annotate("allocate-pinned-memory"):
            start = time.perf_counter()
            pinned_buffer = pinned_pool.malloc(total_bytes)
            elapsed = time.perf_counter() - start
            print(
                f"Allocated {total_bytes / 1e9:.2f} GB of pinned memory in {elapsed:.3f} s"
            )

            # Create views into the pinned buffer
            host_data_list = []
            for i in range(N_FILES):
                offset = i * buffer_size_bytes
                buffer_view = np.ndarray(
                    shape=buffer_shape,
                    dtype=np.float32,
                    buffer=pinned_buffer,
                    offset=offset,
                )
                # Initialize with some data
                buffer_view[:] = np.random.randn(*buffer_shape).astype(np.float32)
                host_data_list.append(buffer_view)

        print(f"Created {N_FILES} pinned buffers")
        file_paths = None  # No files in memory mode

    # Run GPU copy benchmarks
    print("\n" + "=" * 60)
    print("GPU COPY BENCHMARKS")
    print("=" * 60)

    # Benchmark 1: Sync copy (baseline)
    with nvtx.annotate("sync-copy"):
        benchmark_sync_copy(host_data_list[0], device_index=0, n_iterations=10)

    # Benchmark 2: Async copy with single stream
    with nvtx.annotate("async-single-stream"):
        benchmark_async_copy_single_stream(
            host_data_list[0], device_index=0, n_iterations=10
        )

    # Benchmark 3: Async copy with multiple streams (single device)
    with nvtx.annotate("async-multi-stream"):
        benchmark_async_copy_multi_stream(
            host_data_list[:20], device_index=0, n_streams=8
        )

    # Benchmark 4: Multi-device copy (if available)
    if n_devices > 1:
        with nvtx.annotate("multi-device"):
            benchmark_multi_device_copy(host_data_list, devices, n_streams_per_device=8)

        # Benchmark 5: Multi-device with thread pool
        with nvtx.annotate("multi-device-threaded"):
            benchmark_multi_device_threaded(
                host_data_list, devices, n_streams_per_device=8, max_workers=32
            )
    else:
        print("\n" + "=" * 60)
        print("Note: Only 1 GPU available. Skipping multi-device benchmarks.")
        print("Set CUDA_VISIBLE_DEVICES to use multiple GPUs.")
        print("=" * 60)

    # Benchmark 6: Disk-to-GPU pipeline (if using disk)
    if USE_DISK and file_paths:
        print("\n" + "=" * 60)
        print("END-TO-END DISK-TO-GPU BENCHMARK")
        print("=" * 60)

        with nvtx.annotate("disk-to-gpu-pipeline"):
            benchmark_disk_to_gpu_pipelined(
                file_paths,
                devices,
                pinned_pool,
                n_streams_per_device=8,
                io_workers=8,
                gpu_workers=32,
            )

    # Cleanup
    if USE_DISK and file_paths:
        print("\n" + "=" * 60)
        print("CLEANUP")
        print("=" * 60)
        print(f"Removing test data directory: {data_dir}")
        import shutil

        shutil.rmtree(data_dir)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("For maximum disk-to-GPU throughput:")
    print("1. ✓ Use pinned (page-locked) memory on host")
    print("2. ✓ Read files directly into pinned memory")
    print("3. ✓ Use async copies with CUDA streams")
    print("4. ✓ Use multiple streams per device (8-16 is typical)")
    print("5. ✓ Distribute across multiple GPUs if available")
    print("6. ✓ Overlap disk I/O with GPU transfers (pipelining)")
    print("7. ✓ Use thread pools for concurrent I/O and GPU ops")
    print("\nExpected speedup:")
    print("  - Pinned memory: 2-3x vs pageable memory")
    print("  - Async streams: 1.5-2x vs synchronous")
    print("  - Multi-GPU: ~Nx speedup (N = number of GPUs)")
    print("  - Overlapped I/O: 1.5-3x vs sequential read-then-copy")
    print("=" * 60)


if __name__ == "__main__":
    main()
