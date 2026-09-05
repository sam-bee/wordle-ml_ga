#pragma once

#include "model/policy.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace wordle_ga::genotype_slab {

using Slot = std::uint32_t;
inline constexpr Slot kInvalidSlot = std::numeric_limits<Slot>::max();
inline constexpr std::size_t kSlotAlignment = 256;
inline constexpr std::size_t kGenotypeBytes = model::weights::kCount * sizeof(float);
inline constexpr std::size_t kSlotBytes = ((kGenotypeBytes + kSlotAlignment - 1) / kSlotAlignment) * kSlotAlignment;
inline constexpr std::size_t kSlotFloats = kSlotBytes / sizeof(float);
inline constexpr Slot kDefaultSlotCount = 1792; // 32 * 32 organisms, with 1.75 generations of storage.
static_assert(kSlotBytes == 4186624);

// Non-owning handle passed by value to CUDA kernels. All pointers address device
// memory. Indices remain valid until the corresponding reference count reaches zero.
struct DeviceView {
    float *genotypes = nullptr;
    Slot *free_slots = nullptr;
    std::uint32_t *references = nullptr;
    std::uint32_t *free_count = nullptr;
    Slot capacity = 0;
};

// Owns one fixed allocation for genotype bytes plus small device metadata arrays.
// The caller selects the CUDA device and ensures all users finish before Destroy.
// There is no resizing, compaction, host spill, or per-generation cudaMalloc.
class Slab {
  public:
    Slab() = default;
    ~Slab();
    Slab(const Slab &) = delete;
    Slab &operator=(const Slab &) = delete;
    Slab(Slab &&) = delete;
    Slab &operator=(Slab &&) = delete;

    // Creation initializes metadata on stream and waits for that initialization.
    // Capacity is immutable until destruction. Zero/kInvalidSlot capacities and
    // creating an already-created slab return cudaErrorInvalidValue.
    // A failed creation releases partial allocations and leaves this object empty.
    cudaError_t Create(Slot capacity = kDefaultSlotCount, cudaStream_t stream = nullptr);
    cudaError_t Destroy(); // Explicit teardown reports errors; destroying an empty slab succeeds.
    DeviceView view() const { return view_; }

  private:
    DeviceView view_{};
};

} // namespace wordle_ga::genotype_slab
