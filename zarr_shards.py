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
a batch of data to be decompressed in parallel.

The example will

1. Read a shard from disk to (pinned) host memory
2. Copy the host buffer to device memory (asynchronously, using CUDA streams)
3. Decompress *each chunk* in the shard (asynchronously, using CUDA streams)
4. Concatenate the decompressed chunks into a single array
    (ideally, we could avoid this final concat but it's not supported by nvcomp yet)
"""

# A note on the implementation. For simplicity, we're reimplementing much
# of zarr-python here.
# Our array computation will (probably) be done using cupy.
import os
import math
import zarr
import zarr.storage
import numpy as np
import os
import numpy.typing as npt
from nvidia import nvcomp
import concurrent.futures
import nvtx
import cupy
import cupyx
from rmm.allocators.cupy import rmm_cupy_allocator
import rmm
import cupy


def ensure_chunk(array: zarr.Array, target_slice: slice, dtype: np.dtype) -> None:
    array[target_slice] = np.arange(0, target_slice.stop - target_slice.start, dtype=dtype)

def ensure_array(
    store: zarr.storage.LocalStore,
    path: str,
    shape: tuple[int, ...],
    shards: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: np.dtype,
    compressors = "auto",
    filters = "auto",
    pool: concurrent.futures.ThreadPoolExecutor | None = None,
) -> zarr.Array:
    # store = zarr.storage.LocalStore("/tmp/data.zarr")
    # TODO: avoid re-creating the array if it already exists and has the right configuration.
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
        if array.shape == shape and array.shards == shards and array.chunks == chunks and array.dtype == dtype and array.compressors == compressors and array.filters == filters:
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
    slices = [
        slice(i * stride, (i + 1) * stride) for i in range(n_shards)
    ]

    executor = pool or concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())
    futures = [executor.submit(ensure_chunk, array, slice_, dtype) for i, slice_ in enumerate(slices)]
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
) -> tuple[cupy.ndarray, np.ndarray[np.uint64]]:
    path = array.store_path.store.root / array.store_path.path / key

    # 1. Disk -> (pinned) host memory.
    with nvtx.annotate("read::disk"):
        with open(path, "rb") as f, nvtx.annotate("read"):
            f.readinto(host_buffer)

    index_offset, index = offsets_sizes_array(array, host_buffer)
    index = index.tolist()
    junk = []
    stride = math.prod(array.chunks)
    out_chunks = [
        out[slice(i * stride, (i + 1) * stride)] for i in range(len(index))
    ]

    # 2. (pinned) host memory -> device memory.
    with nvtx.annotate("read::transfer"), stream:
        device_buffer.set(host_buffer[:-index_offset].view(device_buffer.dtype), stream=stream)

    # 3. (optionally) decode the chunks.
    if zstd_codec is not None:
        with stream:
            device_arrays = [
                device_buffer[offset:offset + size] for offset, size in index
            ]
            with nvtx.annotate("read::decode"):
                zstd_codec.decode_batch(device_arrays, out=out_chunks)
    else:
        out = device_buffer

    return out.view(array.dtype).reshape(array.shards), junk, index

    # # 3. GPU decompression.
    # if zstd_codec is not None:
    #     decoded_arrays = zstd_codec.decode(chunks_raw_arrays)
    # else:
    #     decoded_arrays = chunks_raw_arrays

    # # 4. Concatenate and reshape.

    # with cupy.cuda.ExternalStream(int(stream)):
    #     out = cupy.empty(array.shards, dtype=array.dtype)
    #     for i, decoded_array in enumerate(decoded_arrays):
    #         stride = math.prod(array.chunks)
    #         out[i * stride : (i + 1) * stride] = cupy.array(decoded_array).view(array.dtype).reshape(array.chunks)

    # return out, (buf, buf, chunks_raw_d, decoded_arrays, chunks_raw_arrays)


# @nvtx.annotate()
# def compute_shard(shard: cupy.ndarray, stream: numba.cuda.cudadrv.driver.Stream) -> tuple[cupy.ndarray, cupy.ndarray]:
#     with cupy.cuda.ExternalStream(int(stream)):
#         return shard.sum(), cupy.matmul(shard, shard)


# @numba.cuda.jit
# def compute_shard(x, out):
#     tid = numba.cuda.grid(1)
#     stride = numba.cuda.gridsize(1)
#     for i in range(tid, x.shape[0], stride):
#         out[0] += x[i]
#     # size = len(x)
#     # if tid < size:
#     #     # for i in range(1000):
#     #     x[tid] += x[tid] + tid + i
#     # out[tid] = x[tid]

@nvtx.annotate()
def compute_shard(shard: cupy.ndarray, stream: cupy.cuda.stream.Stream) -> tuple[cupy.ndarray, cupy.ndarray]:
    with stream:
        for i in range(10):
            cupy.matmul(shard, shard), shard.cumsum(), shard.sum(), shard.mean()
        return cupy.matmul(shard, shard), shard.cumsum(), shard.sum(), shard.mean()


def main():
    CHUNKS = 200_000
    CHUNKS_PER_SHARD = 400
    SHARDS_PER_ARRAY = 4
    DTYPE = np.dtype("float32")
    cupy.cuda.set_allocator(rmm_cupy_allocator)

    mr = rmm.mr.CudaAsyncMemoryResource()
    rmm.mr.set_current_device_resource(mr)

    if os.environ.get("CUPY_CUDA_ARRAY_INTERFACE_SYNC", "") != "0":
        raise RuntimeError("CUPY_CUDA_ARRAY_INTERFACE_SYNC is set, which is not supported")

    # Currently working through the issue with accessing attributes on the
    # decoded object 2j
    COMPRESS = True

    chunks = (CHUNKS,)
    shards = (CHUNKS_PER_SHARD * CHUNKS,)
    shape = (SHARDS_PER_ARRAY * CHUNKS_PER_SHARD * CHUNKS,)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())

    print("Creating array...")
    store = zarr.storage.LocalStore("/tmp/data.zarr")
    if COMPRESS:
        array = ensure_array(store, "compressed", shape, shards, chunks, DTYPE, "auto", "auto", pool)
    else:
        array = ensure_array(store, "uncompressed", shape, shards, chunks, DTYPE, (), (), pool)

    print("Array created")
    # Summarize the sizes
    print(f"Array size: {array.nbytes:>12,} bytes")
    print(f"Shard size: {array.shards[0] * array.dtype.itemsize:>12,} bytes")  # TODO: nd
    print(f"Chunk size: {array.chunks[0] * array.dtype.itemsize:>12,} bytes")  # TODO: nd

    assert isinstance(array.store, zarr.storage.LocalStore)
    shard_keys = list(array._async_array._iter_shard_keys())
    sizes = [
        (array.store_path.store.root / array.store_path.path / key).stat().st_size for key in shard_keys
    ]
    streams = [cupy.cuda.stream.Stream() for _ in shard_keys]
    host_buffers = [
        cupyx.empty_pinned(size, dtype=np.uint8) for size in sizes
    ]

    out_shards = [
        cupy.empty(array.shards, dtype=array.dtype)
        for _ in shard_keys
    ]
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

        codecs = [
            nvcomp_minimal.ZstdCodec(stream.ptr)
            for stream in streams
        ]

    else:
        codecs = [None] * len(streams)

    # keep these alive, to avoid deallocation triggering (blocking) cudaFreeAsync

    # threadsperblock = 256
    # blockspergrid = (array.shards[0] + threadsperblock - 1) // threadsperblock

    junk = []

    with nvtx.annotate("warmup"):
        stream = streams[0]
        # out = numba.cuda.device_array(1, dtype=np.float32, stream=stream)
        shard, decoded_arrays, index = read_shard(array, shard_keys[0], host_buffers[0], shard_buffers[0], stream, codecs[0], out_shards[0])
        junk.append((shard_buffers, decoded_arrays, index))
        compute_shard(shard, stream)
        stream.synchronize()

    # results = []
    shards = []
    results = []
    # junks = []
    read_futures = {}
    decoded_arrays = []

    with nvtx.annotate("benchmark"):
        for shard_key, stream, codec, host_buffer, device_buffer, out_shard in zip(shard_keys, streams, codecs, host_buffers, shard_buffers, out_shards, strict=True):
            shard, decoded_arrays, index = read_shard(array, shard_key, host_buffer, device_buffer, stream, codec, out_shard)
            junk.append((decoded_arrays, index))
            shards.append(shard)
            results.append(compute_shard(shard, stream))

        # # for future in concurrent.futures.as_completed(read_futures):
        #     # stream = read_futures[future]
        #     shard, decoded_arrays, index = future.result()

        with nvtx.annotate("synchronize"):
            for stream in streams:
                stream.synchronize()

    with nvtx.annotate("parallel-benchmark"):
        futures = {
            pool.submit(read_shard, array, shard_key, host_buffer, device_buffer, stream, codec, out_shard): stream
            for shard_key, stream, codec, host_buffer, device_buffer, out_shard in zip(shard_keys, streams, codecs, host_buffers, shard_buffers, out_shards, strict=True)
        }
        for future in concurrent.futures.as_completed(futures):
            stream = futures[future]
            shard, decoded_arrays, index = future.result()
            shards.append(shard)
            results.append(compute_shard(shard, stream))

        with nvtx.annotate("synchronize"):
            for stream in streams:
                stream.synchronize()

    # cleanup

    del junk
    del shards
    del results
    del read_futures
    del decoded_arrays
    del host_buffers
    del shard_buffers
    del codecs
    del streams

if __name__ == "__main__":
    main()
