# Cython declarations for nvcomp C API and CUDA runtime

cdef extern from "cuda_runtime.h":
    ctypedef void* cudaStream_t
    ctypedef int cudaError_t
    
    cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t stream)
    cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t stream)
    cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, int kind, cudaStream_t stream)
    
    # cudaMemcpyKind enum values
    cdef int cudaMemcpyDeviceToDevice
    
    # Error codes
    cdef int cudaSuccess

cdef extern from "nvcomp.h":
    ctypedef int nvcompStatus_t
    
    # Status codes
    cdef int nvcompSuccess
    
cdef extern from "nvcomp/zstd.h":
    # Decompression options structure
    ctypedef struct nvcompBatchedZstdDecompressOpts_t:
        pass
    
    # Default decompression options
    cdef nvcompBatchedZstdDecompressOpts_t nvcompBatchedZstdDecompressDefaultOpts
    
    # Get decompressed sizes
    nvcompStatus_t nvcompBatchedZstdGetDecompressSizeAsync(
        const void* const* compressed_ptrs,
        const size_t* compressed_bytes,
        size_t* decompressed_bytes,
        size_t batch_size,
        cudaStream_t stream
    )
    
    # Get temp buffer size needed for decompression
    nvcompStatus_t nvcompBatchedZstdDecompressGetTempSizeAsync(
        size_t num_chunks,
        size_t max_uncompressed_chunk_bytes,
        nvcompBatchedZstdDecompressOpts_t decompress_opts,
        size_t* temp_bytes,
        size_t max_total_uncompressed_bytes
    )
    
    # Perform batched decompression
    nvcompStatus_t nvcompBatchedZstdDecompressAsync(
        const void* const* device_compressed_chunk_ptrs,
        const size_t* device_compressed_chunk_bytes,
        const size_t* device_uncompressed_buffer_bytes,
        size_t* device_uncompressed_chunk_bytes,
        size_t num_chunks,
        void* device_temp_ptr,
        size_t temp_bytes,
        void* const* device_uncompressed_chunk_ptrs,
        nvcompBatchedZstdDecompressOpts_t decompress_opts,
        nvcompStatus_t* device_statuses,
        cudaStream_t stream
    )

