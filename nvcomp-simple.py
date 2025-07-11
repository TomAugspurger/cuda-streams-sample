# /// script
# requires-python = ">=3.12.0,<3.13"
# dependencies = [
#   "numba-cuda[cu12]",
#   "nvidia-nvcomp-cu12",
#   "nvtx",
# ]
# ///

import concurrent.futures

import numba.cuda
import numpy as np
import nvtx
from nvidia import nvcomp

LEVEL = 21
BITSTREAM_KIND = nvcomp.BitstreamKind.RAW


def launch_decode(
    codec: nvcomp.Codec, arrays: list[numba.cuda.cudadrv.devicearray.DeviceNDArray]
) -> None:
    codec.decode(arrays)


@nvtx.annotate("direct")
def direct(
    streams: list[numba.cuda.cudadrv.driver.Stream],
    pool: concurrent.futures.ThreadPoolExecutor,
    host_buffer: np.ndarray,
) -> None:
    do(streams, pool, host_buffer, use_thread=False)


@nvtx.annotate("threaded")
def threaded(
    streams: list[numba.cuda.cudadrv.driver.Stream],
    pool: concurrent.futures.ThreadPoolExecutor,
    host_buffer: np.ndarray,
) -> None:
    do(streams, pool, host_buffer, use_thread=True)


def do(
    streams: list[numba.cuda.cudadrv.driver.Stream],
    pool: concurrent.futures.ThreadPoolExecutor,
    host_buffer: np.ndarray,
    use_thread: bool,
):
    futures = []
    for stream in streams:
        # host to device transfer
        # https://docs.nvidia.com/cuda/nvcomp/samples/nvcomp.html#Zero-copy-import-host-array
        device_arrays = nvcomp.as_arrays(
            [numba.cuda.to_device(host_buffer, stream=stream) for _ in range(5)]
        )
        # nvtx.mark("Device arrays")
        codec = nvcomp.Codec(
            algorithm="Zstd",
            level=LEVEL,
            bitstream_kind=BITSTREAM_KIND,
            cuda_stream=int(stream),
        )
        # is this blocking!?
        if use_thread:
            with nvtx.annotate("launch-decode"):
                futures.append(pool.submit(launch_decode, codec, device_arrays))
            # for future in concurrent.futures.as_completed(futures):
            #     future.result()
        else:
            with nvtx.annotate("launch-decode"):
                launch_decode(codec, device_arrays)

    for stream in streams:
        stream.synchronize()


def main():
    codec = nvcomp.Codec(
        algorithm="Zstd",
        level=LEVEL,
        bitstream_kind=BITSTREAM_KIND,
    )

    x = np.arange(2048**2, dtype="uint32")
    encoded = codec.encode(x.view("b"))
    numba.cuda.default_stream().synchronize()

    # Host operations
    host_bytes = memoryview(bytes(encoded.cpu()))
    host_buffer = numba.cuda.pinned_array(len(host_bytes), dtype="b")
    host_buffer[:] = host_bytes

    streams = [numba.cuda.stream() for _ in range(4)]
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    for _ in range(2):
        with nvtx.annotate("iteration"):
            direct(streams, pool, host_buffer)
            threaded(streams, pool, host_buffer)

        print("done")


if __name__ == "__main__":
    main()
