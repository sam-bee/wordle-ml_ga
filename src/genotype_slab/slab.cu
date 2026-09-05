#include "genotype_slab/slab.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace wordle_ga::genotype_slab {
namespace {

__global__ void InitializeMetadata(Slot *free_slots, std::uint32_t *references, std::uint32_t *free_count,
                                   Slot capacity) {
    const Slot index = static_cast<Slot>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < capacity) {
        free_slots[index] = capacity - 1 - index;
        references[index] = 0;
    }
    if (index == 0) {
        *free_count = capacity;
    }
}

void ClearView(DeviceView &view) { view = {}; }

void FreeAllocations(DeviceView &view) {
    if (view.free_count != nullptr) {
        cudaFree(view.free_count);
    }
    if (view.references != nullptr) {
        cudaFree(view.references);
    }
    if (view.free_slots != nullptr) {
        cudaFree(view.free_slots);
    }
    if (view.genotypes != nullptr) {
        cudaFree(view.genotypes);
    }
    ClearView(view);
}

} // namespace

Slab::~Slab() { (void)Destroy(); }

cudaError_t Slab::Create(Slot capacity, cudaStream_t stream) {
    if (view_.capacity != 0 || view_.genotypes != nullptr || view_.free_slots != nullptr ||
        view_.references != nullptr || view_.free_count != nullptr || capacity == 0 || capacity == kInvalidSlot) {
        return cudaErrorInvalidValue;
    }

    if (static_cast<std::size_t>(capacity) > std::numeric_limits<std::size_t>::max() / kSlotBytes ||
        static_cast<std::size_t>(capacity) > std::numeric_limits<std::size_t>::max() / sizeof(Slot) ||
        static_cast<std::size_t>(capacity) > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) {
        return cudaErrorInvalidValue;
    }

    DeviceView pending{};
    pending.capacity = capacity;
    cudaError_t error =
        cudaMalloc(reinterpret_cast<void **>(&pending.genotypes), static_cast<std::size_t>(capacity) * kSlotBytes);
    if (error != cudaSuccess) {
        FreeAllocations(pending);
        return error;
    }
    error =
        cudaMalloc(reinterpret_cast<void **>(&pending.free_slots), static_cast<std::size_t>(capacity) * sizeof(Slot));
    if (error != cudaSuccess) {
        FreeAllocations(pending);
        return error;
    }
    error = cudaMalloc(reinterpret_cast<void **>(&pending.references),
                       static_cast<std::size_t>(capacity) * sizeof(std::uint32_t));
    if (error != cudaSuccess) {
        FreeAllocations(pending);
        return error;
    }
    error = cudaMalloc(reinterpret_cast<void **>(&pending.free_count), sizeof(std::uint32_t));
    if (error != cudaSuccess) {
        FreeAllocations(pending);
        return error;
    }

    constexpr unsigned int kThreads = 256;
    const unsigned int blocks = (static_cast<unsigned int>(capacity) - 1) / kThreads + 1;
    InitializeMetadata<<<blocks, kThreads, 0, stream>>>(pending.free_slots, pending.references, pending.free_count,
                                                        capacity);
    error = cudaGetLastError();
    if (error == cudaSuccess) {
        error = cudaStreamSynchronize(stream);
    }
    if (error != cudaSuccess) {
        FreeAllocations(pending);
        return error;
    }

    view_ = pending;
    return cudaSuccess;
}

cudaError_t Slab::Destroy() {
    if (view_.genotypes == nullptr && view_.free_slots == nullptr && view_.references == nullptr &&
        view_.free_count == nullptr && view_.capacity == 0) {
        return cudaSuccess;
    }

    cudaError_t first_error = cudaSuccess;
    auto free_one = [&first_error](void *pointer) {
        if (pointer == nullptr) {
            return;
        }
        const cudaError_t error = cudaFree(pointer);
        if (first_error == cudaSuccess && error != cudaSuccess) {
            first_error = error;
        }
    };
    free_one(view_.free_count);
    free_one(view_.references);
    free_one(view_.free_slots);
    free_one(view_.genotypes);
    ClearView(view_);
    return first_error;
}

} // namespace wordle_ga::genotype_slab
