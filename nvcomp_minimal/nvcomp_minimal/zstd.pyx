# cython: language_level=3
# distutils: language=c++

"""Minimal Cython wrapper for nvcomp batched zstd decompression."""

from libc.stdlib cimport malloc, free
from libc.stdint cimport uint8_t, uintptr_t
from cpython.ref cimport PyObject
import numpy as np
import math
import cupy

cdef class DecompressedArray:
    """Wrapper for decompressed GPU array with __cuda_array_interface__."""
    
    cdef void* ptr
    cdef size_t nbytes
    cdef uintptr_t stream_ptr
    cdef object shape
    cdef object typestr
    cdef object mem_obj  # Keep reference to CuPy memory object to prevent deallocation
    
    def __cinit__(self, uintptr_t ptr, size_t nbytes, uintptr_t stream_ptr, mem_obj):
        self.ptr = <void*>ptr
        self.nbytes = nbytes
        self.stream_ptr = stream_ptr
        self.shape = (nbytes,)
        self.typestr = '|u1'  # uint8
        self.mem_obj = mem_obj  # Keep memory alive
    
    @property
    def __cuda_array_interface__(self):
        """Expose CUDA array interface for zero-copy interop with CuPy."""
        return {
            'version': 3,
            'shape': self.shape,
            'typestr': self.typestr,
            'data': (<uintptr_t>self.ptr, False),
            'stream': self.stream_ptr,
        }


cdef class ZstdCodec:
    """Batched Zstd decompression codec."""
    
    cdef uintptr_t stream_ptr
    cdef cudaStream_t stream
    
    def __cinit__(self, uintptr_t stream_ptr):
        """Initialize codec with CUDA stream pointer.
        
        Args:
            stream_ptr: CUDA stream pointer as integer (e.g., cupy.cuda.Stream().ptr)
        """
        self.stream_ptr = stream_ptr
        self.stream = <cudaStream_t>stream_ptr
     
    def decode_batch(self, compressed_buffers, out=None):
        """Decompress a batch of zstd-compressed buffers.
        
        Args:
            compressed_buffers: List of objects with __cuda_array_interface__
                               containing compressed data on GPU
            out: Optional list of pre-allocated CuPy arrays for output.
                 When provided, skips size query and uses these buffers.
                 Must match the number of compressed_buffers and have
                 sufficient size for decompressed data.
        
        Returns:
            List of DecompressedArray objects with __cuda_array_interface__
        """
        cdef size_t batch_size = len(compressed_buffers)
        cdef void** compressed_ptrs
        cdef size_t* compressed_bytes
        cdef size_t* decompressed_bytes
        cdef size_t* actual_decompressed_bytes
        cdef void** decompressed_ptrs
        cdef nvcompStatus_t status
        cdef nvcompStatus_t* device_statuses
        cdef size_t i
        cdef size_t max_decompressed
        cdef size_t total_decompressed
        cdef size_t temp_bytes
        cdef void* temp_ptr

        if batch_size == 0:
            return []

        # Validate out parameter if provided
        if out is not None:
            if <size_t>len(out) != batch_size:
                raise ValueError(f"out must have {batch_size} elements, got {len(out)}")

        # Extract pointers and sizes from input buffers into Python lists first
        import numpy as np
        compressed_ptrs_list = []
        compressed_bytes_list = []
        for i in range(batch_size):
            cai = compressed_buffers[i].__cuda_array_interface__
            compressed_ptrs_list.append(cai["data"][0])
            nbytes = np.dtype(cai["typestr"]).itemsize * math.prod(cai["shape"])
            compressed_bytes_list.append(nbytes)

        # Create CuPy arrays for device-side pointer/size arrays
        compressed_ptrs_np = np.array(compressed_ptrs_list, dtype=np.uint64)
        compressed_bytes_np = np.array(compressed_bytes_list, dtype=np.uint64)
        
        compressed_ptrs_gpu = cupy.asarray(compressed_ptrs_np)
        compressed_bytes_gpu = cupy.asarray(compressed_bytes_np)
        actual_decompressed_bytes_gpu = cupy.empty(batch_size, dtype=cupy.uint64)
        
        # Get device pointers
        compressed_ptrs = <void**><uintptr_t>compressed_ptrs_gpu.data.ptr
        compressed_bytes = <size_t*><uintptr_t>compressed_bytes_gpu.data.ptr
        actual_decompressed_bytes = <size_t*><uintptr_t>actual_decompressed_bytes_gpu.data.ptr
        
        # Handle output buffers - either query sizes or use provided buffers
        if out is not None:
            # User provided output buffers - extract sizes from them
            output_mems = out
            decompressed_bytes_list = []
            output_ptrs_list = []
            for i in range(batch_size):
                cai = out[i].__cuda_array_interface__
                nbytes = np.dtype(cai["typestr"]).itemsize * math.prod(cai["shape"])
                decompressed_bytes_list.append(nbytes)
                output_ptrs_list.append(cai["data"][0])
            
            decompressed_bytes_np = np.array(decompressed_bytes_list, dtype=np.uint64)
            decompressed_bytes_gpu = cupy.asarray(decompressed_bytes_np)
            decompressed_bytes_host = decompressed_bytes_np
            
            max_decompressed = decompressed_bytes_host.max()
            total_decompressed = decompressed_bytes_host.sum()
        else:
            # Query decompressed sizes and allocate buffers
            decompressed_bytes_gpu = cupy.empty(batch_size, dtype=cupy.uint64)
            decompressed_bytes = <size_t*><uintptr_t>decompressed_bytes_gpu.data.ptr
            
            status = nvcompBatchedZstdGetDecompressSizeAsync(
                <const void* const*>compressed_ptrs,
                compressed_bytes,
                decompressed_bytes,
                batch_size,
                self.stream
            )
            if status != nvcompSuccess:
                raise RuntimeError(f"nvcompBatchedZstdGetDecompressSizeAsync failed: {status}")
            
            # Copy decompressed sizes back to host to calculate max and total
            decompressed_bytes_host = decompressed_bytes_gpu.get()
            
            # Find max and total decompressed sizes
            max_decompressed = decompressed_bytes_host.max()
            total_decompressed = decompressed_bytes_host.sum()
            
            # Allocate output buffers using CuPy and collect their pointers
            output_mems = []
            output_ptrs_list = []
            for i in range(batch_size):
                mem = cupy.empty(int(decompressed_bytes_host[i]), dtype=cupy.uint8)
                output_mems.append(mem)
                output_ptrs_list.append(mem.data.ptr)
        
        # Get final device pointer for decompressed_bytes
        decompressed_bytes = <size_t*><uintptr_t>decompressed_bytes_gpu.data.ptr
        
        # Get temp buffer size
        temp_bytes = 0
        status = nvcompBatchedZstdDecompressGetTempSizeAsync(
            batch_size,
            max_decompressed,
            nvcompBatchedZstdDecompressDefaultOpts,
            &temp_bytes,
            total_decompressed
        )
        if status != nvcompSuccess:
            raise RuntimeError(f"nvcompBatchedZstdDecompressGetTempSizeAsync failed: {status}")
        
        # Allocate temp buffer using CuPy
        temp_mem = None
        temp_ptr = NULL
        if temp_bytes > 0:
            temp_mem = cupy.empty(temp_bytes, dtype=cupy.uint8)
            temp_ptr = <void*><uintptr_t>temp_mem.data.ptr

        # Allocate device statuses buffer using CuPy
        device_statuses_mem = cupy.empty(batch_size, dtype=cupy.int32)
        device_statuses = <nvcompStatus_t*><uintptr_t>device_statuses_mem.data.ptr
        
        # Create device array of output pointers
        output_ptrs_np = np.array(output_ptrs_list, dtype=np.uint64)
        decompressed_ptrs_gpu = cupy.asarray(output_ptrs_np)
        decompressed_ptrs = <void**><uintptr_t>decompressed_ptrs_gpu.data.ptr
        
        # Perform decompression
        status = nvcompBatchedZstdDecompressAsync(
            <const void* const*>compressed_ptrs,
            compressed_bytes,
            decompressed_bytes,
            actual_decompressed_bytes,
            batch_size,
            temp_ptr,
            temp_bytes,
            decompressed_ptrs,
            nvcompBatchedZstdDecompressDefaultOpts,
            device_statuses,
            self.stream
        )
        
        # temp_mem and device_statuses_mem will be automatically freed when they go out of scope
        
        if status != nvcompSuccess:
            raise RuntimeError(f"nvcompBatchedZstdDecompressAsync failed: {status}")
        
        # Create DecompressedArray wrappers with memory object references
        result = []
        for i in range(batch_size):
            arr = DecompressedArray(
                output_ptrs_list[i],  # Use the original pointer value
                int(decompressed_bytes_host[i]),
                self.stream_ptr,
                output_mems[i]  # Keep memory alive
            )
            result.append(arr)

        return result
