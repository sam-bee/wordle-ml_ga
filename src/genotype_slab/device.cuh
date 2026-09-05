#pragma once

#include "genotype_slab/slab.hpp"

#include <cstdint>

namespace wordle_ga::genotype_slab {

// Device-side ownership contract:
// - Allocation, retain, and release phases must be separated by kernel/stream
//   completion (or an equivalent CUDA event boundary). In particular, an
//   allocation phase must never overlap a release phase.
// - Allocate returns one owning reference. Retain and Release return false for
//   invalid, non-live, or overflowing reference-count operations.
// - Genotype payloads are uninitialized; callers must write them before use.
//   A slot handle is stale and forbidden after its last reference is released
//   (the slot may already have been reused).
// - Parameters returns a pointer usable only while the caller owns a reference
//   to that slot.

__device__ inline Slot Allocate(DeviceView slab) {
    if (slab.free_count == nullptr || slab.free_slots == nullptr || slab.references == nullptr || slab.capacity == 0) {
        return kInvalidSlot;
    }

    std::uint32_t count = atomicAdd(slab.free_count, 0u);
    while (count != 0) {
        const std::uint32_t previous = atomicCAS(slab.free_count, count, count - 1);
        if (previous == count) {
            const Slot slot = slab.free_slots[count - 1];
            // Allocation and release are separate phases, so the stack entry
            // is fully published before it can be consumed here.
            atomicExch(slab.references + slot, 1u);
            return slot;
        }
        count = previous;
    }
    return kInvalidSlot;
}

__device__ inline bool Retain(DeviceView slab, Slot slot) {
    if (slab.references == nullptr || slot >= slab.capacity) {
        return false;
    }

    std::uint32_t references = atomicAdd(slab.references + slot, 0u);
    while (references != 0 && references != 0xffffffffu) {
        const std::uint32_t previous = atomicCAS(slab.references + slot, references, references + 1);
        if (previous == references) {
            return true;
        }
        references = previous;
    }
    return false;
}

__device__ inline bool Release(DeviceView slab, Slot slot) {
    if (slab.references == nullptr || slab.free_count == nullptr || slab.free_slots == nullptr ||
        slot >= slab.capacity) {
        return false;
    }

    std::uint32_t references = atomicAdd(slab.references + slot, 0u);
    while (references != 0) {
        const std::uint32_t previous = atomicCAS(slab.references + slot, references, references - 1);
        if (previous != references) {
            references = previous;
            continue;
        }
        if (references != 1) {
            return true;
        }

        // The successful 1 -> 0 transition makes this thread the sole owner
        // of the stack push for this slot. Under the phase contract, the
        // free-count invariant is 0 <= count < capacity before this push.
        const std::uint32_t push_index = atomicAdd(slab.free_count, 1u);
        slab.free_slots[push_index] = slot;
        return true;
    }
    return false;
}

__device__ inline float *Parameters(DeviceView slab, Slot slot) {
    if (slab.genotypes == nullptr || slot >= slab.capacity) {
        return nullptr;
    }
    return slab.genotypes + static_cast<std::size_t>(slot) * kSlotFloats;
}

} // namespace wordle_ga::genotype_slab
