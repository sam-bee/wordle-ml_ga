#include "genotype_slab/device.cuh"
#include "genotype_slab/slab.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace slab = wordle_ga::genotype_slab;

namespace {

void Require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(EXIT_FAILURE);
    }
}

template <typename T> T CopyOne(const T *device, cudaStream_t stream) {
    T result{};
    CheckCuda(cudaMemcpyAsync(&result, device, sizeof(result), cudaMemcpyDeviceToHost, stream), "copy result");
    CheckCuda(cudaStreamSynchronize(stream), "synchronize result");
    return result;
}

std::vector<slab::Slot> CopySlots(const slab::Slot *device, std::size_t count, cudaStream_t stream) {
    std::vector<slab::Slot> result(count);
    CheckCuda(cudaMemcpyAsync(result.data(), device, count * sizeof(slab::Slot), cudaMemcpyDeviceToHost, stream),
              "copy slots");
    CheckCuda(cudaStreamSynchronize(stream), "synchronize slots");
    return result;
}

__global__ void AllocateKernel(slab::DeviceView view, slab::Slot *slots, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        slots[i] = slab::Allocate(view);
}

__global__ void RetainKernel(slab::DeviceView view, slab::Slot slot, std::uint32_t *successes) {
    if (slab::Retain(view, slot))
        atomicAdd(successes, 1u);
}

__global__ void ReleaseKernel(slab::DeviceView view, slab::Slot slot, std::uint32_t *successes) {
    if (slab::Release(view, slot))
        atomicAdd(successes, 1u);
}

__global__ void InvalidReleaseKernel(slab::DeviceView view, slab::Slot slot, std::uint32_t *status) {
    if (!slab::Release(view, slab::kInvalidSlot))
        atomicAdd(status, 1u);
    if (!slab::Release(view, slot))
        atomicAdd(status, 1u);
    if (!slab::Retain(view, slot))
        atomicAdd(status, 1u);
    if (!slab::Retain(view, slab::kInvalidSlot))
        atomicAdd(status, 1u);
}

// Multiple blocks compete to drop the last reference to each remaining live slot.
__global__ void ReleaseAllKernel(slab::DeviceView view, std::uint32_t *successes) {
    const auto slot = static_cast<slab::Slot>((blockIdx.x * blockDim.x + threadIdx.x) % view.capacity);
    if (slab::Release(view, slot))
        atomicAdd(successes, 1u);
}

__global__ void InitializeBoundariesKernel(slab::DeviceView view, slab::Slot first, slab::Slot second,
                                           float *addresses) {
    float *a = slab::Parameters(view, first);
    float *b = slab::Parameters(view, second);
    if (a == nullptr || b == nullptr)
        return;
    a[0] = 3.0f;
    a[wordle_ga::model::weights::kCount - 1] = 4.0f;
    a[slab::kSlotFloats - 1] = 5.0f;
    b[0] = 7.0f;
    b[wordle_ga::model::weights::kCount - 1] = 9.0f;
    b[slab::kSlotFloats - 1] = 11.0f;
    addresses[0] = static_cast<float>(reinterpret_cast<std::uintptr_t>(a) % slab::kSlotAlignment);
    addresses[1] = static_cast<float>(reinterpret_cast<std::uintptr_t>(b) % slab::kSlotAlignment);
    addresses[2] = static_cast<float>(reinterpret_cast<std::uintptr_t>(b) - reinterpret_cast<std::uintptr_t>(a));
    addresses[3] = 1.0f;
}

__global__ void ReadWriteLifetimeKernel(slab::DeviceView view, slab::Slot parent, slab::Slot child, float *status) {
    float *p = slab::Parameters(view, parent);
    float *c = slab::Parameters(view, child);
    if (p == nullptr || c == nullptr)
        return;
    c[0] = p[0] + 1.0f;
    c[slab::kSlotFloats - 1] = p[slab::kSlotFloats - 1] + 1.0f;
    status[0] = p[0];
    status[1] = p[slab::kSlotFloats - 1];
}

__global__ void WriteBoundariesKernel(slab::DeviceView view, slab::Slot slot, float *observed) {
    float *parameters = slab::Parameters(view, slot);
    if (parameters == nullptr)
        return;
    parameters[0] = 11.5f;
    parameters[wordle_ga::model::weights::kCount - 1] = 42.0f;
    parameters[slab::kSlotFloats - 1] = 97.25f;
    observed[0] = parameters[0];
    observed[1] = parameters[slab::kSlotFloats - 1];
    observed[2] = slab::Parameters(view, slab::kInvalidSlot) == nullptr ? 1.0f : 0.0f;
    observed[3] = slab::Parameters(view, view.capacity) == nullptr ? 1.0f : 0.0f;
}

__global__ void ReadBoundariesKernel(slab::DeviceView view, slab::Slot slot, float *observed) {
    float *parameters = slab::Parameters(view, slot);
    observed[0] = parameters == nullptr ? -1.0f : parameters[0];
    observed[1] = parameters == nullptr ? -1.0f : parameters[slab::kSlotFloats - 1];
    observed[2] = parameters == nullptr ? -1.0f : parameters[wordle_ga::model::weights::kCount - 1];
}

void LaunchAllocate(slab::DeviceView view, slab::Slot *slots, int count, cudaStream_t stream) {
    AllocateKernel<<<(count + 127) / 128, 128, 0, stream>>>(view, slots, count);
    CheckCuda(cudaGetLastError(), "allocate kernel");
}

void CheckFreeCount(slab::DeviceView view, std::uint32_t expected, cudaStream_t stream, const char *message) {
    const auto count = CopyOne(view.free_count, stream);
    Require(count == expected, message);
}

void TestPlannedLifetime(cudaStream_t stream) {
    slab::Slab object;
    Require(object.Create(4, stream) == cudaSuccess, "lifetime Create failed");
    const auto view = object.view();
    slab::Slot *slots = nullptr;
    std::uint32_t *status = nullptr;
    float *values = nullptr;
    CheckCuda(cudaMalloc(&slots, 2 * sizeof(slab::Slot)), "lifetime slots");
    CheckCuda(cudaMalloc(&status, sizeof(std::uint32_t)), "lifetime status");
    CheckCuda(cudaMalloc(&values, 4 * sizeof(float)), "lifetime values");
    LaunchAllocate(view, slots, 2, stream);
    const auto parents = CopySlots(slots, 2, stream);
    InitializeBoundariesKernel<<<1, 1, 0, stream>>>(view, parents[0], parents[1], values);
    CheckCuda(cudaGetLastError(), "initialize lifetime parents");
    CheckCuda(cudaStreamSynchronize(stream), "sync lifetime parents");
    CheckCuda(cudaMemsetAsync(status, 0, sizeof(*status), stream), "clear lifetime status");
    RetainKernel<<<1, 2, 0, stream>>>(view, parents[0], status);
    CheckCuda(cudaGetLastError(), "lifetime retains");
    ReleaseKernel<<<1, 1, 0, stream>>>(view, parents[0], status);
    ReleaseKernel<<<1, 1, 0, stream>>>(view, parents[1], status);
    CheckCuda(cudaGetLastError(), "lifetime owner releases");
    CheckFreeCount(view, 3, stream, "lifetime owner accounting is wrong");

    LaunchAllocate(view, slots, 1, stream);
    const auto child1 = CopyOne(slots, stream);
    ReadWriteLifetimeKernel<<<1, 1, 0, stream>>>(view, parents[0], child1, values);
    CheckCuda(cudaGetLastError(), "lifetime child one");
    std::array<float, 2> checked{};
    CheckCuda(cudaMemcpyAsync(checked.data(), values, sizeof(checked), cudaMemcpyDeviceToHost, stream),
              "copy child one");
    CheckCuda(cudaStreamSynchronize(stream), "sync child one");
    Require(checked[0] == 3.0f && checked[1] == 5.0f, "lifetime parent payload was not initialized as expected");
    ReleaseKernel<<<1, 1, 0, stream>>>(view, parents[0], status);
    CheckCuda(cudaGetLastError(), "lifetime first use release");
    CheckFreeCount(view, 2, stream, "parent freed before final future use");

    LaunchAllocate(view, slots, 1, stream);
    const auto child2 = CopyOne(slots, stream);
    ReadWriteLifetimeKernel<<<1, 1, 0, stream>>>(view, parents[0], child2, values);
    CheckCuda(cudaGetLastError(), "lifetime child two");
    ReleaseKernel<<<1, 1, 0, stream>>>(view, parents[0], status);
    CheckCuda(cudaGetLastError(), "lifetime final use release");
    CheckFreeCount(view, 2, stream, "parent was not reclaimed after final use");
    ReadWriteLifetimeKernel<<<1, 1, 0, stream>>>(view, child1, child2, values);
    CheckCuda(cudaGetLastError(), "lifetime child stability");
    CheckCuda(cudaMemcpyAsync(checked.data(), values, sizeof(checked), cudaMemcpyDeviceToHost, stream),
              "copy child stability");
    CheckCuda(cudaStreamSynchronize(stream), "sync child stability");
    Require(checked[0] == 4.0f && checked[1] == 6.0f, "child genotype data was not stable");
    ReleaseKernel<<<1, 1, 0, stream>>>(view, child1, status);
    ReleaseKernel<<<1, 1, 0, stream>>>(view, child2, status);
    CheckCuda(cudaGetLastError(), "lifetime child releases");
    CheckFreeCount(view, 4, stream, "child ownership was not released");
    CheckCuda(cudaFree(values), "free lifetime values");
    CheckCuda(cudaFree(status), "free lifetime status");
    CheckCuda(cudaFree(slots), "free lifetime slots");
    Require(object.Destroy() == cudaSuccess, "lifetime Destroy failed");
}

void TestSmallSlab(cudaStream_t stream) {
    slab::Slab slab_object;
    Require(slab_object.Create(0, stream) == cudaErrorInvalidValue, "Create(0) did not fail");
    Require(slab_object.Create(slab::kInvalidSlot, stream) == cudaErrorInvalidValue, "Create(invalid) did not fail");
    Require(slab_object.Destroy() == cudaSuccess, "Destroy after failed Create failed");
    Require(slab_object.Create(4, stream) == cudaSuccess, "small Create failed");
    const auto view = slab_object.view();
    Require(slab_object.Create(4, stream) == cudaErrorInvalidValue, "Create on existing slab did not fail");
    CheckFreeCount(view, 4, stream, "initial free count is wrong");

    // Exhaustion and parallel oversubscription: every successful result is unique.
    constexpr int kRequests = 512;
    slab::Slot *device_slots = nullptr;
    CheckCuda(cudaMalloc(&device_slots, kRequests * sizeof(slab::Slot)), "allocate slot results");
    LaunchAllocate(view, device_slots, kRequests, stream);
    const auto allocated = CopySlots(device_slots, kRequests, stream);
    std::vector<slab::Slot> live;
    for (const auto slot : allocated) {
        if (slot != slab::kInvalidSlot)
            live.push_back(slot);
    }
    std::sort(live.begin(), live.end());
    Require(live.size() == 4 && std::adjacent_find(live.begin(), live.end()) == live.end(),
            "parallel allocation did not return four unique slots");
    CheckFreeCount(view, 0, stream, "free count after exhaustion is wrong");

    // Slot alignment/stride and boundary writes must not touch another genotype.
    float *observed = nullptr;
    CheckCuda(cudaMalloc(&observed, 4 * sizeof(float)), "allocate boundary results");
    InitializeBoundariesKernel<<<1, 1, 0, stream>>>(view, live[0], live[1], observed);
    CheckCuda(cudaGetLastError(), "boundary write kernel");
    std::array<float, 4> values{};
    CheckCuda(cudaMemcpyAsync(values.data(), observed, sizeof(values), cudaMemcpyDeviceToHost, stream),
              "copy boundaries");
    CheckCuda(cudaStreamSynchronize(stream), "synchronize boundaries");
    Require(values[0] == 0.0f && values[1] == 0.0f && values[2] == static_cast<float>(slab::kSlotBytes) &&
                values[3] == 1.0f,
            "slot addresses are not aligned/adjacent");
    WriteBoundariesKernel<<<1, 1, 0, stream>>>(view, live[0], observed);
    CheckCuda(cudaGetLastError(), "boundary overwrite kernel");
    CheckCuda(cudaMemcpyAsync(values.data(), observed, sizeof(values), cudaMemcpyDeviceToHost, stream),
              "copy boundaries");
    CheckCuda(cudaStreamSynchronize(stream), "synchronize boundary overwrite");
    Require(values[0] == 11.5f && values[1] == 97.25f && values[2] == 1.0f && values[3] == 1.0f,
            "boundary parameter access is wrong");
    float *other = nullptr;
    CheckCuda(cudaMalloc(&other, 3 * sizeof(float)), "allocate isolation results");
    ReadBoundariesKernel<<<1, 1, 0, stream>>>(view, live[1], other);
    CheckCuda(cudaGetLastError(), "boundary isolation kernel");
    std::array<float, 3> other_values{};
    CheckCuda(cudaMemcpyAsync(other_values.data(), other, sizeof(other_values), cudaMemcpyDeviceToHost, stream),
              "copy isolation results");
    CheckCuda(cudaStreamSynchronize(stream), "synchronize isolation");
    Require(other_values[0] == 7.0f && other_values[1] == 11.0f && other_values[2] == 9.0f,
            "boundary write leaked into another genotype");
    CheckCuda(cudaFree(other), "free isolation results");
    CheckCuda(cudaFree(observed), "free boundary results");

    // Release one slot, reject invalid/double release, and ensure exactly-once reuse.
    std::uint32_t *device_status = nullptr;
    CheckCuda(cudaMalloc(&device_status, sizeof(std::uint32_t)), "allocate status");
    CheckCuda(cudaMemsetAsync(device_status, 0, sizeof(std::uint32_t), stream), "clear status");
    ReleaseKernel<<<1, 1, 0, stream>>>(view, live[0], device_status);
    CheckCuda(cudaGetLastError(), "first release kernel");
    Require(CopyOne(device_status, stream) == 1, "first release failed");
    CheckCuda(cudaMemsetAsync(device_status, 0, sizeof(std::uint32_t), stream), "clear invalid status");
    InvalidReleaseKernel<<<1, 1, 0, stream>>>(view, live[0], device_status);
    CheckCuda(cudaGetLastError(), "invalid release kernel");
    Require(CopyOne(device_status, stream) == 4, "invalid retain/release or freed-slot access was accepted");
    CheckFreeCount(view, 1, stream, "double release changed free count");
    LaunchAllocate(view, device_slots, 1, stream);
    Require(CopyOne(device_slots, stream) == live[0], "reclaimed slot was not reused");
    CheckCuda(cudaFree(device_status), "free status");

    // Concurrent retains/releases are each a separate phase and reclaim once.
    std::uint32_t *device_successes = nullptr;
    CheckCuda(cudaMalloc(&device_successes, sizeof(std::uint32_t)), "allocate retain results");
    CheckCuda(cudaMemsetAsync(device_successes, 0, sizeof(std::uint32_t), stream), "clear retain results");
    RetainKernel<<<4, 128, 0, stream>>>(view, live[1], device_successes);
    CheckCuda(cudaGetLastError(), "retain kernel");
    Require(CopyOne(device_successes, stream) == 512, "concurrent retains were lost");
    CheckCuda(cudaMemsetAsync(device_successes, 0, sizeof(std::uint32_t), stream), "clear release results");
    ReleaseKernel<<<4, 128, 0, stream>>>(view, live[1], device_successes);
    CheckCuda(cudaGetLastError(), "release kernel");
    Require(CopyOne(device_successes, stream) == 512, "concurrent releases were lost");
    CheckFreeCount(view, 0, stream, "retained slot was prematurely reclaimed");
    ReleaseKernel<<<1, 1, 0, stream>>>(view, live[1], device_successes);
    CheckCuda(cudaGetLastError(), "final release kernel");
    CheckFreeCount(view, 1, stream, "final release did not reclaim slot");
    LaunchAllocate(view, device_slots, 1, stream);
    Require(CopyOne(device_slots, stream) == live[1], "final-released slot was not reclaimed");
    CheckCuda(cudaMemsetAsync(device_successes, 0, sizeof(std::uint32_t), stream), "clear extra release results");
    ReleaseKernel<<<4, 128, 0, stream>>>(view, live[1], device_successes);
    CheckCuda(cudaGetLastError(), "extra release kernel");
    Require(CopyOne(device_successes, stream) == 1, "extra concurrent releases were accepted");
    CheckFreeCount(view, 1, stream, "extra releases pushed more than once");

    CheckCuda(cudaMemsetAsync(device_successes, 0, sizeof(std::uint32_t), stream), "clear bulk release results");
    ReleaseAllKernel<<<4, 128, 0, stream>>>(view, device_successes);
    CheckCuda(cudaGetLastError(), "bulk release kernel");
    Require(CopyOne(device_successes, stream) == 3, "remaining slots were not reclaimed exactly once");
    CheckFreeCount(view, 4, stream, "bulk release lost free slots");
    LaunchAllocate(view, device_slots, 4, stream);
    auto recycled = CopySlots(device_slots, 4, stream);
    std::sort(recycled.begin(), recycled.end());
    Require(recycled == live, "recycling changed the set of available slots");
    CheckFreeCount(view, 0, stream, "bulk reallocation count is wrong");
    CheckCuda(cudaFree(device_successes), "free retain results");

    CheckCuda(cudaFree(device_slots), "free slot results");
    Require(slab_object.Destroy() == cudaSuccess, "small Destroy failed");
    Require(slab_object.Destroy() == cudaSuccess, "repeated Destroy failed");
    Require(slab_object.Create(4, stream) == cudaSuccess, "repeated Create failed");
    Require(slab_object.Destroy() == cudaSuccess, "second slab Destroy failed");
    TestPlannedLifetime(stream);
}

void FullSizeSmoke(cudaStream_t stream) {
    slab::Slab object;
    Require(object.Create(slab::kDefaultSlotCount, stream) == cudaSuccess, "full-size Create failed");
    const auto view = object.view();
    CheckFreeCount(view, slab::kDefaultSlotCount, stream, "full-size free count is wrong");
    std::vector<std::uint32_t> references(slab::kDefaultSlotCount);
    std::vector<slab::Slot> free_slots(slab::kDefaultSlotCount);
    CheckCuda(cudaMemcpyAsync(references.data(), view.references, references.size() * sizeof(references[0]),
                              cudaMemcpyDeviceToHost, stream),
              "copy full-size references");
    CheckCuda(cudaMemcpyAsync(free_slots.data(), view.free_slots, free_slots.size() * sizeof(free_slots[0]),
                              cudaMemcpyDeviceToHost, stream),
              "copy full-size free stack");
    CheckCuda(cudaStreamSynchronize(stream), "synchronize full-size metadata");
    for (std::size_t i = 0; i < references.size(); ++i) {
        Require(references[i] == 0 && free_slots[i] == slab::kDefaultSlotCount - 1 - i,
                "full-size metadata initialization is wrong");
    }
    slab::Slot *slot = nullptr;
    float *values = nullptr;
    CheckCuda(cudaMalloc(&slot, slab::kDefaultSlotCount * sizeof(*slot)), "full-size slot results");
    CheckCuda(cudaMalloc(&values, 4 * sizeof(float)), "full-size values");
    LaunchAllocate(view, slot, slab::kDefaultSlotCount, stream);
    auto allocated = CopySlots(slot, slab::kDefaultSlotCount, stream);
    std::sort(allocated.begin(), allocated.end());
    for (std::size_t i = 0; i < allocated.size(); ++i) {
        Require(allocated[i] == i, "full-size allocation lost or duplicated a slot");
    }
    CheckFreeCount(view, 0, stream, "full-size exhaustion count is wrong");
    WriteBoundariesKernel<<<1, 1, 0, stream>>>(view, allocated.front(), values);
    CheckCuda(cudaGetLastError(), "full-size boundary kernel");
    std::array<float, 2> host_values{};
    CheckCuda(cudaMemcpyAsync(host_values.data(), values, sizeof(host_values), cudaMemcpyDeviceToHost, stream),
              "copy full-size values");
    CheckCuda(cudaStreamSynchronize(stream), "synchronize full-size values");
    Require(host_values[0] == 11.5f && host_values[1] == 97.25f, "full-size boundary touch failed");
    WriteBoundariesKernel<<<1, 1, 0, stream>>>(view, allocated.back(), values);
    CheckCuda(cudaGetLastError(), "full-size last boundary kernel");
    CheckCuda(cudaMemcpyAsync(host_values.data(), values, sizeof(host_values), cudaMemcpyDeviceToHost, stream),
              "copy full-size last values");
    CheckCuda(cudaStreamSynchronize(stream), "synchronize full-size last boundary");
    Require(host_values[0] == 11.5f && host_values[1] == 97.25f, "full-size last boundary touch failed");
    CheckCuda(cudaFree(values), "free full-size values");
    CheckCuda(cudaFree(slot), "free full-size slot");
    Require(object.Destroy() == cudaSuccess, "full-size Destroy failed");
}

} // namespace

int main(int argc, char **argv) {
    const auto gpu = SelectTestGpu();
    cudaStream_t stream = nullptr;
    CheckCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create test stream");
    if (argc == 2 && std::strcmp(argv[1], "--full-size-smoke") == 0) {
        FullSizeSmoke(stream);
        std::printf("CUDA genotype slab full-size smoke passed: gpu=%s\n", gpu.name);
    } else {
        Require(argc == 1, "usage: genotype_slab_test [--full-size-smoke]");
        TestSmallSlab(stream);
        std::printf("CUDA genotype slab tests passed: gpu=%s\n", gpu.name);
    }
    CheckCuda(cudaStreamDestroy(stream), "destroy test stream");
    return EXIT_SUCCESS;
}
