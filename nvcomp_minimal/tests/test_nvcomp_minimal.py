"""Simple test for nvcomp_minimal wrapper."""

import numpy as np
import cupy
import numcodecs
from nvcomp_minimal import ZstdCodec


def test_simple_decompress():
    """Test basic compression with numcodecs and decompression with nvcomp_minimal."""
    
    # 1. Create and compress a numpy array with numcodecs
    original_data = np.arange(1024, dtype=np.uint64)
    codec = numcodecs.Zstd(level=0)
    compressed_bytes = codec.encode(original_data)
    
    print(f"Original size: {original_data.nbytes} bytes")
    print(f"Compressed size: {len(compressed_bytes)} bytes")
    print(f"Compression ratio: {original_data.nbytes / len(compressed_bytes):.2f}x")
    
    # 2. Copy compressed data to GPU as a CuPy array
    compressed_gpu = cupy.asarray(np.frombuffer(compressed_bytes, dtype=np.uint8))
    
    # 3. Decompress with nvcomp_minimal
    stream = cupy.cuda.Stream()
    with stream:
        zstd_codec = ZstdCodec(stream.ptr)
        decompressed_arrays = zstd_codec.decode([compressed_gpu])
        
    #     # Convert back to CuPy array
    #     result_gpu = cupy.asarray(decompressed_arrays[0])
        
    #     # View as original dtype
    #     result_gpu = cupy.frombuffer(result_gpu, dtype=np.uint64)
    
    # # Synchronize and verify
    # stream.synchronize()
    
    # # Check results match
    # result_cpu = result_gpu.get()
    # assert np.array_equal(original_data, result_cpu), "Decompressed data doesn't match original!"
    
    # print("✓ Test passed! Decompressed data matches original.")


# def test_batched_decompress():
#     """Test batched decompression of multiple arrays."""
    
#     # 1. Create and compress multiple numpy arrays
#     n_arrays = 5
#     original_arrays = [i * np.arange(512, dtype=np.uint64) for i in range(n_arrays)]
    
#     codec = numcodecs.Zstd(level=0)
#     compressed_arrays = [codec.encode(arr) for arr in original_arrays]
    
#     print(f"Compressing {n_arrays} arrays...")
#     total_original = sum(arr.nbytes for arr in original_arrays)
#     total_compressed = sum(len(c) for c in compressed_arrays)
#     print(f"Total original size: {total_original} bytes")
#     print(f"Total compressed size: {total_compressed} bytes")
#     print(f"Overall compression ratio: {total_original / total_compressed:.2f}x")
    
#     # 2. Copy all compressed arrays to GPU
#     compressed_gpu_arrays = [
#         cupy.asarray(np.frombuffer(c, dtype=np.uint8))
#         for c in compressed_arrays
#     ]
    
#     # 3. Batch decompress with nvcomp_minimal
#     stream = cupy.cuda.Stream()
#     with stream:
#         zstd_codec = ZstdCodec(stream.ptr)
#         decompressed_arrays = zstd_codec.decode_batch(compressed_gpu_arrays)
        
#         # Convert to CuPy arrays with proper dtype
#         results = [
#             cupy.frombuffer(cupy.asarray(arr), dtype=np.uint64)
#             for arr in decompressed_arrays
#         ]
    
#     # Synchronize and verify
#     stream.synchronize()
    
#     # Check all results
#     for i, (original, result_gpu) in enumerate(zip(original_arrays, results)):
#         result_cpu = result_gpu.get()
#         assert np.array_equal(original, result_cpu), f"Array {i} doesn't match!"
    
#     print(f"✓ Test passed! All {n_arrays} decompressed arrays match originals.")


# def test_stream_ordering():
#     """Test that operations are properly stream-ordered."""
    
#     # Create data and compress
#     original_data = np.arange(2048, dtype=np.uint64)
#     codec = numcodecs.Zstd(level=0)
#     compressed_bytes = codec.encode(original_data)
#     compressed_gpu = cupy.asarray(np.frombuffer(compressed_bytes, dtype=np.uint8))
    
#     # Use a non-default stream
#     stream = cupy.cuda.Stream(non_blocking=True)
    
#     with stream:
#         # Create codec on specific stream
#         zstd_codec = ZstdCodec(stream.ptr)
        
#         # Decompress
#         decompressed = zstd_codec.decode_batch([compressed_gpu])
#         result = cupy.frombuffer(cupy.asarray(decompressed[0]), dtype=np.uint64)
        
#         # Schedule more work on the same stream
#         result_squared = result * result
    
#     # Synchronize only this stream
#     stream.synchronize()
    
#     # Verify
#     expected = original_data * original_data
#     assert np.array_equal(expected, result_squared.get()), "Stream-ordered computation failed!"
    
#     print("✓ Stream ordering test passed!")


# if __name__ == "__main__":
#     print("=" * 70)
#     print("Testing nvcomp_minimal wrapper")
#     print("=" * 70)
#     print()
    
#     print("Test 1: Simple decompression")
#     print("-" * 70)
#     test_simple_decompress()
#     print()
    
#     print("Test 2: Batched decompression")
#     print("-" * 70)
#     test_batched_decompress()
#     print()
    
#     print("Test 3: Stream ordering")
#     print("-" * 70)
#     test_stream_ordering()
#     print()
    
#     print("=" * 70)
#     print("All tests passed! ✓")
#     print("=" * 70)

