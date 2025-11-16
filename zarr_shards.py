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

1. Read data from disk to (pinned) host memory
2. Copy the data from host to device memory (asynchronously, using CUDA streams)
3. Decompress the data on the device (asynchronously, using CUDA streams)
4. Perform some array computation on the device (asynchronously, using CUDA streams)
"""

# A note on the implementation. For simplicity, we're reimplementing much
# of zarr-python here.
# Our array computation will (probably) be done using cupy.
import math
import zarr
import numpy as np
import numpy.typing as npt
from nvidia import nvcomp
import nvtx
import numba.cuda
import cupy

def ensure_array(
    shape: tuple[int, ...],
    shards: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: np.dtype,
) -> zarr.Array:
    store = zarr.storage.LocalStore("data.zarr")
    array = zarr.create_array(
        store,
        shape=shape,
        shards=shards,
        chunks=chunks,
        dtype=dtype,
        overwrite=True,
    )

    array[:] = np.arange(shape[0], dtype=dtype)

    return array


def offsets_sizes_array(
    array: zarr.Array, buf: np.ndarray[np.uint8]
) -> tuple[int, npt.NDArray[np.uint64]]:
    sc = array.metadata.codecs[0]
    assert isinstance(sc, zarr.codecs.ShardingCodec)

    chunks_per_shard = math.prod(
        math.ceil(s / c) for s, c in zip(array.shards, array.chunks)
    )
    index_offset = 16 * chunks_per_shard + 4

    # offsets, sizes
    return index_offset, buf[-index_offset:-4].view(np.uint64).reshape(-1, 2)


@nvtx.annotate()
def read_shard(array: zarr.Array, key: str, stream: numba.cuda.cudadrv.driver.Stream) -> np.ndarray:
    # TODO: I/O in a thread.
    # TODO: Pinned memory pool.
    # TODO: Decide how to use CUDA streams (probably one per shard)
    # TODO: See if we can eliminate the final concat by decoding into an output buffer

    assert isinstance(array.store, zarr.storage.LocalStore)
    config = array.metadata.codecs[0].codecs[1].to_dict()
    assert config["name"] == "zstd", config
    assert len(array.metadata.codecs[0].codecs) == 2, array.metadata.codecs

    zstd_codec = nvcomp.Codec(
        algorithm="Zstd",
        bitstream_kind=nvcomp.BitstreamKind.RAW,
        compression_level=config["configuration"]["level"],  # TODO: confirm this
        cuda_stream=int(stream),
    )

    path = array.store_path.store.root / key

    # 2. Disk -> (pinned) host memory.
    buf = numba.cuda.pinned_array(path.stat().st_size, dtype=np.uint8)
    with open(path, "rb") as f:
        f.readinto(buf)

    index_offset, index = offsets_sizes_array(array, buf)

    # Single device array for all the raw chunks.
    chunks_raw_d = numba.cuda.device_array(len(buf) - index_offset, dtype=np.uint8, stream=stream)

    chunks_raw_arrays = []

    # 2. Host -> device memory.
    for i, (offset, size) in enumerate(index):
        slice_ = slice(offset, offset + size)
        chunk_raw = chunks_raw_d[slice_]
        chunk_raw.copy_to_device(buf[slice_], stream=stream)
        chunks_raw_arrays.append(chunk_raw)

    # 3. GPU decompression.
    decoded_arrays = zstd_codec.decode(chunks_raw_arrays)

    # 4. Concatenate and reshape.

    with cupy.cuda.ExternalStream(int(stream)):
        out = cupy.empty(array.shards, dtype=array.dtype)
        for i, decoded_array in enumerate(decoded_arrays):
            stride = math.prod(array.chunks)
            out[i * stride : (i + 1) * stride] = cupy.array(decoded_array).view(array.dtype).reshape(array.chunks)

    return out


def main():
    SHAPE = (1_200_000,)
    SHARDS = (400_000,)
    CHUNKS = (1000,)
    DTYPE = np.dtype("float32")

    array = ensure_array(SHAPE, SHARDS, CHUNKS, DTYPE)

    shard_keys = list(array._async_array._iter_shard_keys())
    streams = [numba.cuda.stream() for _ in shard_keys]

    with nvtx.annotate("warmup"):
        shard = read_shard(array, shard_keys[0], streams[0])
        with nvtx.annotate("compute"):
            with cupy.cuda.ExternalStream(int(streams[0])):
                (shard.sum(), len(shard), cupy.matmul(shard, shard))

    results = []

    with nvtx.annotate("benchmark"):
        for shard_key, stream in zip(shard_keys, streams):
            shard = read_shard(array, shard_key, stream)

            with nvtx.annotate("compute"):
                with cupy.cuda.ExternalStream(int(stream)):
                    results.append((shard.sum(), len(shard), cupy.matmul(shard, shard)))

    for stream in streams:
        stream.synchronize()

if __name__ == "__main__":
    main()
