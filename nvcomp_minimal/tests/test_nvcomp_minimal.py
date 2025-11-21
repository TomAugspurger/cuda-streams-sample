"""Simple test for nvcomp_minimal wrapper."""

import numpy as np
import cupy
import numcodecs
from nvcomp_minimal import ZstdCodec
import pytest


@pytest.mark.parametrize("n", [0, 1, 5])
def test_simple_decompress(n: int) -> None:
    """Test basic compression with numcodecs and decompression with nvcomp_minimal."""
    dtype = np.uint64
    original_data = [i * np.arange(1024, dtype=dtype) for i in range(n)]
    codec = numcodecs.Zstd(level=0)
    compressed_bytes = [codec.encode(o) for o in original_data]

    # 2. Copy compressed data to GPU as a CuPy array
    compressed_gpu = [
        cupy.asarray(np.frombuffer(o, dtype=np.uint8)) for o in compressed_bytes
    ]

    # 3. Decompress with nvcomp_minimal
    stream = cupy.cuda.Stream()
    with stream:
        zstd_codec = ZstdCodec(stream.ptr)
        decompressed_arrays = zstd_codec.decode_batch(compressed_gpu)
        # Convert back to CuPy array
        result_gpu = [cupy.asarray(a).view(dtype) for a in decompressed_arrays]
        for o, r in zip(original_data, result_gpu, strict=True):
            cupy.testing.assert_array_equal(o, r)
        stream.synchronize()
