#include <cuda_runtime.h>
#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <vector>
#include <chrono>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Custom pinned memory resource using cudaMallocHost/cudaFreeHost
class pinned_memory_resource {
public:
    void* allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) {
        void* ptr = nullptr;
        CUDA_CHECK(cudaMallocHost(&ptr, bytes));
        return ptr;
    }

    void deallocate(void* ptr, std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) {
        CUDA_CHECK(cudaFreeHost(ptr));
    }

    bool operator==(const pinned_memory_resource& other) const noexcept {
        return true;
    }

    bool operator!=(const pinned_memory_resource& other) const noexcept {
        return false;
    }
};

// Simple device memory resource
class device_memory_resource {
public:
    void* allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) {
        void* ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        return ptr;
    }

    void deallocate(void* ptr, std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) {
        CUDA_CHECK(cudaFree(ptr));
    }

    bool operator==(const device_memory_resource& other) const noexcept {
        return true;
    }

    bool operator!=(const device_memory_resource& other) const noexcept {
        return false;
    }
};

// RAII wrapper for allocated memory
template<typename MemoryResource>
class memory_buffer {
    MemoryResource mr_;
    void* ptr_;
    std::size_t size_;

public:
    memory_buffer(MemoryResource mr, std::size_t size) 
        : mr_(mr), ptr_(nullptr), size_(size) {
        ptr_ = mr_.allocate(size);
    }

    ~memory_buffer() {
        if (ptr_) {
            mr_.deallocate(ptr_, size_);
        }
    }

    // Non-copyable, movable
    memory_buffer(const memory_buffer&) = delete;
    memory_buffer& operator=(const memory_buffer&) = delete;
    
    memory_buffer(memory_buffer&& other) noexcept
        : mr_(other.mr_), ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    void* get() { return ptr_; }
    const void* get() const { return ptr_; }
    std::size_t size() const { return size_; }
};

int main() {
    std::cout << "=== CUDA Pinned Memory Transfer Demo ===" << std::endl;
    
    nvtxRangePush("Initialization");
    
    // Configuration
    const std::size_t TOTAL_SIZE = 256 * 1024 * 1024;  // 256 MB
    const std::size_t NUM_CHUNKS = 8;
    const std::size_t CHUNK_SIZE = TOTAL_SIZE / NUM_CHUNKS;
    
    std::cout << "Total buffer size: " << TOTAL_SIZE / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Number of chunks: " << NUM_CHUNKS << std::endl;
    std::cout << "Chunk size: " << CHUNK_SIZE / (1024 * 1024) << " MB" << std::endl;
    
    // Create memory resources
    pinned_memory_resource pinned_mr;
    device_memory_resource device_mr;
    
    // Allocate pinned host memory
    nvtxRangePush("Allocate Pinned Host Memory");
    memory_buffer<pinned_memory_resource> host_buffer(pinned_mr, TOTAL_SIZE);
    std::cout << "Allocated " << TOTAL_SIZE / (1024 * 1024) 
              << " MB of pinned host memory" << std::endl;
    nvtxRangePop();
    
    // Allocate device memory
    nvtxRangePush("Allocate Device Memory");
    memory_buffer<device_memory_resource> device_buffer(device_mr, TOTAL_SIZE);
    std::cout << "Allocated " << TOTAL_SIZE / (1024 * 1024) 
              << " MB of device memory" << std::endl;
    nvtxRangePop();
    
    // Initialize host memory with some data
    nvtxRangePush("Initialize Host Data");
    float* host_data = static_cast<float*>(host_buffer.get());
    for (std::size_t i = 0; i < TOTAL_SIZE / sizeof(float); ++i) {
        host_data[i] = static_cast<float>(i);
    }
    std::cout << "Initialized host data" << std::endl;
    nvtxRangePop();
    
    // Create CUDA streams for async transfers
    nvtxRangePush("Create CUDA Streams");
    std::vector<cudaStream_t> streams(NUM_CHUNKS);
    for (auto& stream : streams) {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    std::cout << "Created " << NUM_CHUNKS << " CUDA streams" << std::endl;
    nvtxRangePop();
    
    nvtxRangePop(); // End Initialization
    
    // Perform chunked async transfers
    nvtxRangePush("Async H2D Transfers");
    std::cout << "\n--- Starting async transfers ---" << std::endl;
    
    auto transfer_start = std::chrono::high_resolution_clock::now();
    
    char* host_ptr = static_cast<char*>(host_buffer.get());
    char* device_ptr = static_cast<char*>(device_buffer.get());
    
    for (std::size_t i = 0; i < NUM_CHUNKS; ++i) {
        nvtxRangePush(("Transfer Chunk " + std::to_string(i)).c_str());
        
        std::size_t offset = i * CHUNK_SIZE;
        
        std::cout << "Enqueuing chunk " << i << " (offset: " 
                  << offset / (1024 * 1024) << " MB)" << std::endl;
        
        CUDA_CHECK(cudaMemcpyAsync(
            device_ptr + offset,
            host_ptr + offset,
            CHUNK_SIZE,
            cudaMemcpyHostToDevice,
            streams[i]
        ));
        
        nvtxRangePop();
    }
    
    auto enqueue_end = std::chrono::high_resolution_clock::now();
    auto enqueue_time = std::chrono::duration_cast<std::chrono::microseconds>(
        enqueue_end - transfer_start).count();
    
    std::cout << "\nAll transfers enqueued in " << enqueue_time 
              << " microseconds" << std::endl;
    std::cout << "This demonstrates non-blocking behavior!" << std::endl;
    
    nvtxRangePop(); // End Async H2D Transfers
    
    // Synchronize all streams to ensure transfers complete
    nvtxRangePush("Synchronize Streams");
    std::cout << "\n--- Synchronizing streams ---" << std::endl;
    
    for (std::size_t i = 0; i < NUM_CHUNKS; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    
    auto sync_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        sync_end - transfer_start).count();
    
    std::cout << "All transfers completed in " << total_time << " milliseconds" << std::endl;
    
    double bandwidth = (TOTAL_SIZE / (1024.0 * 1024.0 * 1024.0)) / (total_time / 1000.0);
    std::cout << "Effective bandwidth: " << bandwidth << " GB/s" << std::endl;
    
    nvtxRangePop();
    
    // Verify a few values (copy back small amount to check)
    nvtxRangePush("Verification");
    std::vector<float> verify_data(10);
    CUDA_CHECK(cudaMemcpy(verify_data.data(), device_buffer.get(), 
                          verify_data.size() * sizeof(float), 
                          cudaMemcpyDeviceToHost));
    
    bool correct = true;
    for (std::size_t i = 0; i < verify_data.size(); ++i) {
        if (verify_data[i] != static_cast<float>(i)) {
            correct = false;
            break;
        }
    }
    
    std::cout << "\nData verification: " << (correct ? "PASSED" : "FAILED") << std::endl;
    nvtxRangePop();
    
    // Cleanup streams
    nvtxRangePush("Cleanup");
    for (auto& stream : streams) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    std::cout << "\nCleanup complete" << std::endl;
    nvtxRangePop();
    
    std::cout << "\n=== Demo Complete ===" << std::endl;
    
    return 0;
}

