"""
CUDA Pinned Memory Transfer Demo - Python Version with cuda.core
Demonstrates non-blocking host-to-device transfers using cuda.core's Pythonic API
"""

import numpy as np
import time
import dataclasses

import nvtx
from cuda.core.experimental import Device, Stream
import rmm.mr
import rmm.pylibrmm.stream


@dataclasses.dataclass
class PinnedMemoryBuffer:
    mr: rmm.mr.PinnedHostMemoryResource
    ptr: int
    nbytes: int
    stream: Stream

    @property
    def __array_interface__(self):
        return {
            "shape": (self.nbytes,),
            "typestr": "|u1",
            "data": (self.ptr, False),
        }

    def __del__(self):
        # instances must live longer than any views on the buffer.
        self.mr.deallocate(
            self.ptr, self.nbytes, rmm.pylibrmm.stream.Stream(self.stream)
        )


@dataclasses.dataclass
class DeviceMemoryBuffer:
    mr: rmm.mr.DeviceMemoryResource
    ptr: int
    nbytes: int
    stream: Stream

    def __del__(self):
        self.mr.deallocate(
            self.ptr, self.nbytes, rmm.pylibrmm.stream.Stream(self.stream)
        )


def main():
    # Mark initialization phase
    # Configuration
    TOTAL_SIZE = 256 * 1024 * 1024  # 256 MB
    NUM_CHUNKS = 8
    CHUNK_SIZE = TOTAL_SIZE // NUM_CHUNKS
    TOTAL_SIZE // np.dtype(np.float32).itemsize
    NUM_FLOATS_PER_CHUNK = CHUNK_SIZE // np.dtype(np.float32).itemsize

    # Get default device using cuda.core
    device = Device()
    device.set_current()

    host_mr = rmm.mr.PinnedHostMemoryResource()
    device_mr = rmm.mr.CudaAsyncMemoryResource()
    rmm.mr.set_current_device_resource(device_mr)

    host_buffers = []
    host_arrays = []
    device_buffers = []
    streams = [device.create_stream() for _ in range(NUM_CHUNKS)]

    with nvtx.annotate("Initialization"):
        with nvtx.annotate("Allocate host"):
            for i, stream in enumerate(streams):
                host_ptr = host_mr.allocate(CHUNK_SIZE)
                host_buffer = PinnedMemoryBuffer(
                    host_mr, host_ptr, CHUNK_SIZE, rmm.pylibrmm.stream.Stream(stream)
                )
                host_array = np.array(host_buffer).view(np.float32)
                host_array[:] = np.arange(NUM_FLOATS_PER_CHUNK, dtype=np.float32)
                host_buffers.append(host_buffer)
                host_arrays.append(host_array)

        with nvtx.annotate("Allocate device"):
            for i, stream in enumerate(streams):
                device_buffer = rmm.DeviceBuffer(
                    size=CHUNK_SIZE, stream=rmm.pylibrmm.stream.Stream(stream)
                )
                device_buffers.append(device_buffer)

    transfer_start = time.perf_counter()
    with nvtx.annotate("Transfer"):
        for host_array, device_buffer, stream in zip(
            host_arrays, device_buffers, streams, strict=True
        ):
            # XXX: this apparently blocks, so this isn't usable.
            device_buffer.copy_from_host(
                host_array.view(np.dtype(np.ubyte)),
                stream=rmm.pylibrmm.stream.Stream(stream),
            )

    with nvtx.annotate("Synchronize"):
        for stream in streams:
            stream.sync()

    transfer_end = time.perf_counter()
    print(
        "Transfer throughput: {:.2f} GB/s".format(
            TOTAL_SIZE / (transfer_end - transfer_start) / 1024 / 1024 / 1024
        )
    )

    with nvtx.annotate("Cleanup"):
        del host_buffers
        del host_arrays
        del device_buffers
        del streams


if __name__ == "__main__":
    main()
