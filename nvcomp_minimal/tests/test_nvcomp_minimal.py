"""Simple test for nvcomp_minimal wrapper."""

import numpy as np
import cupy
import numcodecs
from nvcomp_minimal import ZstdCodec
import pytest


@pytest.mark.parametrize("n", [0, 1, 5])
@pytest.mark.parametrize("use_out", [False, True])
def test_simple_decompress(n: int, *, use_out: bool) -> None:
    """Test basic compression with numcodecs and decompression with nvcomp_minimal."""
    dtype = np.uint64
    original_data = [i * np.arange(1024, dtype=dtype) for i in range(1, n + 1)]
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

        # Optionally pre-allocate output buffers
        if use_out:
            output_buffers = [
                cupy.empty(o.nbytes, dtype=np.uint8) for o in original_data
            ]
            decompressed_arrays = zstd_codec.decode_batch(
                compressed_gpu, out=output_buffers
            )

            # Verify the output buffers were used (same memory)
            for out_buf, dec_arr in zip(output_buffers, decompressed_arrays):
                assert out_buf.data.ptr == dec_arr.__cuda_array_interface__["data"][0]
        else:
            decompressed_arrays = zstd_codec.decode_batch(compressed_gpu)

        # Convert back to CuPy array
        result_gpu = [cupy.asarray(a).view(dtype) for a in decompressed_arrays]
        for o, r in zip(original_data, result_gpu, strict=True):
            cupy.testing.assert_array_equal(o, r)
        stream.synchronize()


def test_decompress_with_wrong_output_count() -> None:
    """Test that providing wrong number of output buffers raises an error."""
    dtype = np.uint64
    original_data = [np.arange(1024, dtype=dtype), 2 * np.arange(1024, dtype=dtype)]
    codec = numcodecs.Zstd(level=0)
    compressed_bytes = [codec.encode(o) for o in original_data]

    compressed_gpu = [
        cupy.asarray(np.frombuffer(o, dtype=np.uint8)) for o in compressed_bytes
    ]

    # Provide wrong number of output buffers (1 instead of 2)
    output_buffers = [cupy.empty(original_data[0].nbytes, dtype=np.uint8)]

    stream = cupy.cuda.Stream()
    with stream:
        zstd_codec = ZstdCodec(stream.ptr)
        with pytest.raises(ValueError, match="out must have 2 elements, got 1"):
            zstd_codec.decode_batch(compressed_gpu, out=output_buffers)
