# /// script
# requires-python = ">=3.12.0,<3.13"
# dependencies = [
#   "numba-cuda[cu12]",
#   "cupy-cuda12x",
#   "nvtx",
#   "zarr",
# ]
# ///
"""
This example demonstrates how to use Zarr shards to efficiently read
and decompress chunks of data.

Shards, by providing another layer of hierarchy, allow us to feed
a batch of data to be decompressed in parallel, *without* exploding
the number of files. This suites GPUs well. File I/O has some per-file
overhead that consumes CPU time before we can get data to the device.

The example will

1. Read a shard from disk to (pinned) host memory
2. Copy the host buffer to device memory (asynchronously, using CUDA streams)
3. Decompress *each chunk* in the shard (asynchronously, using CUDA streams)
4. Concatenate the decompressed chunks into a single array

This example is deliberately written to be friendly to GPU acceleration. In particular

- Chunk / shard sizes have been chosen to work well with the GPU I was testing on
  (an NVIDIA V100)
- Chunks are contiguous in shards, and shards are contiguous in the array

Several parts of the example deal only with 1-D arrays. I don't think that
this design fundamentally precludes generalizing to n-d arrays. But we *would* need
to maintain the contiguity invariants.

We currently use a custom wrapper around nvcomp, `nvcomp_minimal`, to

1. Avoid some (unnecessary) stream synchronization on property access,
   which prevents pipelining I/O and computation (decompression and subsequent
   array computation).
2. Allows us to pass in an output array (whose size is known ahead of time, because
   we know the shape of the shard and the itemsize from the dtype). This avoids
   a malloc inside the decompression kernel.

These are supported by nvcomp's C / C++ APIs, but not the Python wrapper.
"""

import argparse
import os
import math
import zarr
import zarr.storage
import numpy as np
import numpy.typing as npt
from nvidia import nvcomp
import concurrent.futures
import nvtx
import cupyx
from rmm.allocators.cupy import rmm_cupy_allocator
import rmm
import cupy


def slices_from_chunks(chunks: tuple[int, ...], shape: tuple[int, ...]) -> list[slice]:
    stride = math.prod(chunks)
    n_chunks = math.prod(shape) // stride
    return [slice(i * stride, (i + 1) * stride) for i in range(n_chunks)]


def ensure_chunk(array: zarr.Array, target_slice: slice, dtype: np.dtype) -> None:
    array[target_slice] = np.arange(
        0, target_slice.stop - target_slice.start, dtype=dtype
    )


def ensure_array(
    store: zarr.storage.LocalStore,
    path: str,
    shape: tuple[int, ...],
    shards: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: np.dtype,
    compressors="auto",
    filters="auto",
    pool: concurrent.futures.ThreadPoolExecutor | None = None,
) -> zarr.Array:
    if compressors == "auto":
        compressors = (zarr.codecs.ZstdCodec(),)
    if filters == "auto":
        filters = ()
    try:
        array = zarr.open_array(store, mode="a", path=path)
    except TypeError:
        pass
    else:
        # TODO: handle "auto" comparison
        if (
            array.shape == shape
            and array.shards == shards
            and array.chunks == chunks
            and array.dtype == dtype
            and array.compressors == compressors
            and array.filters == filters
        ):
            return array

    array = zarr.create_array(
        store,
        shape=shape,
        shards=shards,
        chunks=chunks,
        dtype=dtype,
        compressors=compressors,
        filters=filters,
        overwrite=True,
        name=path,
    )

    # write the shards in parallel
    stride = math.prod(shards)
    n_shards = math.prod(shape) // stride
    slices = [slice(i * stride, (i + 1) * stride) for i in range(n_shards)]

    executor = pool or concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())
    futures = [
        executor.submit(ensure_chunk, array, slice_, dtype)
        for i, slice_ in enumerate(slices)
    ]
    concurrent.futures.wait(futures)

    return array


def compute_index_offset(array: zarr.Array) -> int:
    chunks_per_shard = math.prod(
        math.ceil(s / c) for s, c in zip(array.shards, array.chunks)
    )

    return 16 * chunks_per_shard + 4


def offsets_sizes_array(
    array: zarr.Array, buf: np.ndarray[np.uint8]
) -> tuple[int, npt.NDArray[np.uint64]]:
    sc = array.metadata.codecs[0]
    assert isinstance(sc, zarr.codecs.ShardingCodec)
    index_offset = compute_index_offset(array)

    # offsets, sizes
    return index_offset, buf[-index_offset:-4].view(np.uint64).reshape(-1, 2)


@nvtx.annotate()
def read_shard(
    array: zarr.Array,
    key: str,
    host_buffer: np.ndarray[np.uint8],
    device_buffer: cupy.ndarray,
    stream: cupy.cuda.stream.Stream,
    zstd_codec: nvcomp.Codec | None,
    out: cupy.ndarray | None = None,
) -> cupy.ndarray:
    path = array.store_path.store.root / array.store_path.path / key

    # 1. Disk -> (pinned) host memory.
    with open(path, "rb") as f, nvtx.annotate("read::disk"):
        f.readinto(host_buffer)

    index_offset, index = offsets_sizes_array(array, host_buffer)
    index = index.tolist()
    stride = math.prod(array.chunks)
    out_chunks = [out[slice(i * stride, (i + 1) * stride)] for i in range(len(index))]

    with stream:
        # 2. (pinned) host memory -> device memory.
        with nvtx.annotate("read::transfer"), stream:
            device_buffer.set(
                host_buffer[:-index_offset].view(device_buffer.dtype), stream=stream
            )

        # 3. (optionally) decode the chunks.
        if zstd_codec is not None:
            device_arrays = [
                device_buffer[offset : offset + size] for offset, size in index
            ]
            with nvtx.annotate("read::decode"):
                zstd_codec.decode_batch(device_arrays, out=out_chunks)
        else:
            out = device_buffer

        return out.view(array.dtype).reshape(array.shards)


@nvtx.annotate()
def compute_shard(
    shard: cupy.ndarray, stream: cupy.cuda.stream.Stream
) -> tuple[cupy.ndarray, cupy.ndarray]:
    with stream:
        for i in range(10):
            shard @ shard, shard.cumsum(), shard.sum(), shard.mean()
        return shard @ shard, shard.cumsum(), shard.sum(), shard.mean()


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark zarr shards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--benchmark-gpu", action=argparse.BooleanOptionalAction)
    parser.add_argument("--benchmark-cpu", action=argparse.BooleanOptionalAction)
    parser.add_argument("--benchmark-zarr-gpu", action=argparse.BooleanOptionalAction)
    return parser.parse_args(args)


def main():
    parsed = parse_args()

    CHUNKS = 200_000
    CHUNKS_PER_SHARD = 400
    SHARDS_PER_ARRAY = 4
    DTYPE = np.dtype("float32")
    cupy.cuda.set_allocator(rmm_cupy_allocator)

    mr = rmm.mr.CudaAsyncMemoryResource()
    rmm.mr.set_current_device_resource(mr)

    if os.environ.get("CUPY_CUDA_ARRAY_INTERFACE_SYNC", "") != "0":
        raise RuntimeError(
            "CUPY_CUDA_ARRAY_INTERFACE_SYNC is set, which is not supported"
        )

    COMPRESS = True

    chunks = (CHUNKS,)
    shards = (CHUNKS_PER_SHARD * CHUNKS,)
    shape = (SHARDS_PER_ARRAY * CHUNKS_PER_SHARD * CHUNKS,)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())

    print("Creating array...")
    store = zarr.storage.LocalStore("/tmp/data.zarr")
    if COMPRESS:
        array = ensure_array(
            store, "compressed", shape, shards, chunks, DTYPE, "auto", "auto", pool
        )
    else:
        array = ensure_array(
            store, "uncompressed", shape, shards, chunks, DTYPE, (), (), pool
        )

    print("Array created")
    print(f"Array size: {array.nbytes:>12,} bytes")
    print(
        f"Shard size: {array.shards[0] * array.dtype.itemsize:>12,} bytes"
    )  # TODO: nd
    print(
        f"Chunk size: {array.chunks[0] * array.dtype.itemsize:>12,} bytes"
    )  # TODO: nd

    assert isinstance(array.store, zarr.storage.LocalStore)
    shard_keys = list(array._async_array._iter_shard_keys())
    sizes = [
        (array.store_path.store.root / array.store_path.path / key).stat().st_size
        for key in shard_keys
    ]
    streams = [cupy.cuda.stream.Stream() for _ in shard_keys]
    host_buffers = [cupyx.empty_pinned(size, dtype=np.uint8) for size in sizes]

    out_shards = [cupy.empty(array.shards, dtype=array.dtype) for _ in shard_keys]
    shard_buffers = []
    index_offset = compute_index_offset(array)
    for stream, size in zip(streams, sizes, strict=True):
        with stream:
            shard_buffers.append(cupy.empty(size - index_offset, dtype=np.uint8))

    if COMPRESS:
        config = array.metadata.codecs[0].codecs[1].to_dict()
        assert config["name"] == "zstd", config
        assert len(array.metadata.codecs[0].codecs) == 2, array.metadata.codecs

        import nvcomp_minimal

        codecs = [nvcomp_minimal.ZstdCodec(stream.ptr) for stream in streams]

    else:
        codecs = [None] * len(streams)

    with nvtx.annotate("warmup"):
        stream = streams[0]
        shard = read_shard(
            array,
            shard_keys[0],
            host_buffers[0],
            shard_buffers[0],
            stream,
            codecs[0],
            out_shards[0],
        )
        compute_shard(shard, stream)
        stream.synchronize()

    # results = []
    shards = []
    results = []

    with nvtx.annotate("benchmark"):  # ~300ms
        for shard_key, stream, codec, host_buffer, device_buffer, out_shard in zip(
            shard_keys,
            streams,
            codecs,
            host_buffers,
            shard_buffers,
            out_shards,
            strict=True,
        ):
            shard = read_shard(
                array, shard_key, host_buffer, device_buffer, stream, codec, out_shard
            )
            shards.append(shard)
            results.append(compute_shard(shard, stream))

        with nvtx.annotate("synchronize"):
            for stream in streams:
                stream.synchronize()

    if parsed.benchmark_gpu:
        with nvtx.annotate("parallel-benchmark"):
            futures = {
                pool.submit(
                    read_shard,
                    array,
                    shard_key,
                    host_buffer,
                    device_buffer,
                    stream,
                    codec,
                    out_shard,
                ): stream
                for shard_key, stream, codec, host_buffer, device_buffer, out_shard in zip(
                    shard_keys,
                    streams,
                    codecs,
                    host_buffers,
                    shard_buffers,
                    out_shards,
                    strict=True,
                )
            }
            for future in concurrent.futures.as_completed(futures):
                stream = futures[future]
                shard = future.result()
                shards.append(shard)
                results.append(compute_shard(shard, stream))

            with nvtx.annotate("synchronize"):
                for stream in streams:
                    stream.synchronize()

    slices = slices_from_chunks(array.shards, array.shape)

    if parsed.benchmark_cpu:
        with nvtx.annotate("zarr-python"):  # ~20s
            for slice_ in slices:
                with nvtx.annotate("read"):
                    x = array[slice_]
                with nvtx.annotate("compute"):
                    result = compute_shard(x, stream)
                results.append(result)

    if parsed.benchmark_zarr_gpu:
        with nvtx.annotate("zarr-python-gpu"), zarr.config.enable_gpu():  # ~3s
            stream = cupy.cuda.stream.Stream()
            with stream:
                for slice_ in slices:
                    with nvtx.annotate("read"):
                        x = array[slice_]
                    with nvtx.annotate("compute"):
                        result = compute_shard(x, stream)
                    results.append(result)
                stream.synchronize()


if __name__ == "__main__":
    main()
