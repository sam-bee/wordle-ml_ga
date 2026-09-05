#include "model/policy.hpp"

#include <cuda_runtime.h>

namespace wordle_ga::model {
namespace {

constexpr int kThreads = 128;
constexpr int kWarpSize = 32;
constexpr int kLogitRowsPerBlock = 64;
constexpr int kLogitTiles = (kNumActions + kLogitRowsPerBlock - 1) / kLogitRowsPerBlock;

__device__ float WarpSum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

// All 128 threads participate. Only thread zero receives the complete sum.
__device__ float BlockSum(float value) {
    __shared__ float warp_sums[kThreads / kWarpSize];
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    value = WarpSum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        value = WarpSum(lane < kThreads / kWarpSize ? warp_sums[lane] : 0.0f);
    }
    return value;
}

template <bool Batch>
__device__ const float *CaseParameters(const float *parameters, const float *const *parameter_array, int case_index) {
    if constexpr (Batch) {
        return parameter_array[case_index];
    } else {
        return parameters;
    }
}

template <bool Batch> __device__ bool CaseIsActive(const int *active, int case_index) {
    return !Batch || active == nullptr || active[case_index] != 0;
}

template <bool Batch>
__global__ void PrepareInput(const Input *inputs, const float *parameters, const float *const *parameter_array,
                             Workspace *workspaces, const int *active) {
    const int case_index = Batch ? blockIdx.x : 0;
    if (!CaseIsActive<Batch>(active, case_index)) {
        return;
    }
    const Input *input = inputs + (Batch ? case_index : 0);
    Workspace *workspace = workspaces + (Batch ? case_index : 0);
    const float *case_parameters = CaseParameters<Batch>(parameters, parameter_array, case_index);
    __shared__ float reciprocal;
    float count = 0.0f;
    for (int i = threadIdx.x; i < kNumSolutions; i += kThreads) {
        count += input->candidate_mask[i];
    }
    count = BlockSum(count);
    if (threadIdx.x == 0) {
        reciprocal = 1.0f / fmaxf(count, 1.0f);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kNumSolutions; i += kThreads) {
        workspace->normalized_candidates[i] = input->candidate_mask[i] * reciprocal;
    }
    if (threadIdx.x < kTurnWidth) {
        workspace->hidden[kCandidateWidth + kStatsWidth + threadIdx.x] =
            case_parameters[weights::kTurnEmbedding + input->turn * kTurnWidth + threadIdx.x];
    }
}

// One warp per output row; adjacent lanes read adjacent FP32 weights.
template <int InputWidth, int OutputWidth, bool Relu, bool AddSkip = false, bool Batch = false,
          bool WorkspaceInput = false>
__global__ void Dense(const Input *inputs, const float *parameters, const float *const *parameter_array,
                      Workspace *workspaces, const int *active, std::size_t input_offset, std::size_t output_offset,
                      std::size_t matrix_offset, std::size_t bias_offset, std::size_t skip_offset = 0) {
    constexpr int kWarpsPerBlock = kThreads / kWarpSize;
    constexpr int kBlocksPerCase = (OutputWidth + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const int tile = Batch ? blockIdx.x % kBlocksPerCase : blockIdx.x;
    const int case_index = Batch ? blockIdx.x / kBlocksPerCase : 0;
    if (!CaseIsActive<Batch>(active, case_index)) {
        return;
    }
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int row = tile * kWarpsPerBlock + warp;
    if (row >= OutputWidth) {
        return;
    }
    const float *case_parameters = CaseParameters<Batch>(parameters, parameter_array, case_index);
    const Workspace *workspace = workspaces + (Batch ? case_index : 0);
    const float *input = WorkspaceInput
                             ? reinterpret_cast<const float *>(workspace) + input_offset
                             : reinterpret_cast<const float *>(inputs + (Batch ? case_index : 0)) + input_offset;
    float *output = reinterpret_cast<float *>(workspaces + (Batch ? case_index : 0)) + output_offset;
    const float *skip = skip_offset == 0 ? nullptr : reinterpret_cast<const float *>(workspace) + skip_offset;
    float value = 0.0f;
    for (int i = lane; i < InputWidth; i += kWarpSize) {
        value = fmaf(case_parameters[matrix_offset + row * InputWidth + i], input[i], value);
    }
    value = WarpSum(value);
    if (lane == 0) {
        value += case_parameters[bias_offset + row];
        if constexpr (AddSkip) {
            value += skip[row];
        }
        output[row] = Relu ? fmaxf(value, 0.0f) : value;
    }
}

// Reuse one hidden vector across 64 rows. Each warp reduces its own dot products.
template <bool Batch>
__global__ void PolicyLogits(const float *parameters, const float *const *parameter_array, const Input *inputs,
                             const Workspace *workspaces, float *logits, const int *active) {
    const int case_index = Batch ? blockIdx.x / kLogitTiles : 0;
    const int tile = Batch ? blockIdx.x % kLogitTiles : blockIdx.x;
    if (!CaseIsActive<Batch>(active, case_index)) {
        return;
    }
    const float *case_parameters = CaseParameters<Batch>(parameters, parameter_array, case_index);
    const Input *input = inputs + (Batch ? case_index : 0);
    const Workspace *workspace = workspaces + (Batch ? case_index : 0);
    float *case_logits = logits + (Batch ? case_index * kNumActions : 0);
    __shared__ float hidden[kTrunkWidth];
    for (int i = threadIdx.x; i < kTrunkWidth; i += kThreads) {
        hidden[i] = workspace->hidden[i];
    }
    __syncthreads();

    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    for (int row = warp; row < kLogitRowsPerBlock; row += kThreads / kWarpSize) {
        const int action = tile * kLogitRowsPerBlock + row;
        if (action >= kNumActions) {
            continue;
        }
        float value = 0.0f;
        for (int i = lane; i < kTrunkWidth; i += kWarpSize) {
            value = fmaf(case_parameters[weights::kLogits + action * kTrunkWidth + i], hidden[i], value);
        }
        value = WarpSum(value);
        if (lane == 0) {
            // The candidate bonus is not a legality mask: probe words retain their logits.
            case_logits[action] = value + case_parameters[weights::kLogitsBias + action] +
                                  workspace->bonus * input->remaining_action_mask[action];
        }
    }
}

template <bool Batch>
cudaError_t LaunchForward(const float *parameters, const float *const *parameter_array, const Input *inputs,
                          Workspace *workspaces, float *logits, int count, const int *active, cudaStream_t stream) {
    if (Batch) {
        PrepareInput<true><<<count, kThreads, 0, stream>>>(inputs, parameters, parameter_array, workspaces, active);
    } else {
        PrepareInput<false><<<1, kThreads, 0, stream>>>(inputs, parameters, parameter_array, workspaces, active);
    }
    if (const auto error = cudaGetLastError(); error != cudaSuccess)
        return error;

    constexpr int kDenseWarpsPerBlock = kThreads / kWarpSize;
    constexpr int kCandidateBlocksPerCase = (kCandidateWidth + kDenseWarpsPerBlock - 1) / kDenseWarpsPerBlock;
    constexpr int kStatsBlocksPerCase = (kStatsWidth + kDenseWarpsPerBlock - 1) / kDenseWarpsPerBlock;
    constexpr int kTrunkBlocksPerCase = (kTrunkWidth + kDenseWarpsPerBlock - 1) / kDenseWarpsPerBlock;
    const int candidate_blocks = Batch ? count * kCandidateBlocksPerCase : kCandidateBlocksPerCase;
    const int stats_blocks = Batch ? count * kStatsBlocksPerCase : kStatsBlocksPerCase;
    const int trunk_blocks = Batch ? count * kTrunkBlocksPerCase : kTrunkBlocksPerCase;
    const int logits_blocks = Batch ? count * kLogitTiles : kLogitTiles;
    Dense<kNumSolutions, kCandidateWidth, true, false, Batch, true><<<candidate_blocks, kThreads, 0, stream>>>(
        inputs, parameters, parameter_array, workspaces, active,
        offsetof(Workspace, normalized_candidates) / sizeof(float), offsetof(Workspace, hidden) / sizeof(float),
        weights::kCandidate, weights::kCandidateBias);
    if (const auto error = cudaGetLastError(); error != cudaSuccess)
        return error;
    Dense<kCandidateStatsSize, kStatsWidth, true, false, Batch, false><<<stats_blocks, kThreads, 0, stream>>>(
        inputs, parameters, parameter_array, workspaces, active, offsetof(Input, candidate_stats) / sizeof(float),
        offsetof(Workspace, hidden) / sizeof(float) + kCandidateWidth, weights::kStats, weights::kStatsBias);
    if (const auto error = cudaGetLastError(); error != cudaSuccess)
        return error;
    Dense<kTrunkWidth, kTrunkWidth, true, false, Batch, true><<<trunk_blocks, kThreads, 0, stream>>>(
        inputs, parameters, parameter_array, workspaces, active, offsetof(Workspace, hidden) / sizeof(float),
        offsetof(Workspace, residual) / sizeof(float), weights::kResidualIn, weights::kResidualInBias);
    if (const auto error = cudaGetLastError(); error != cudaSuccess)
        return error;
    Dense<kTrunkWidth, kTrunkWidth, true, true, Batch, true><<<trunk_blocks, kThreads, 0, stream>>>(
        inputs, parameters, parameter_array, workspaces, active, offsetof(Workspace, residual) / sizeof(float),
        offsetof(Workspace, hidden) / sizeof(float), weights::kResidualOut, weights::kResidualOutBias,
        offsetof(Workspace, hidden) / sizeof(float));
    if (const auto error = cudaGetLastError(); error != cudaSuccess)
        return error;
    Dense<kTrunkWidth, 1, false, false, Batch, true><<<Batch ? count : 1, kThreads, 0, stream>>>(
        inputs, parameters, parameter_array, workspaces, active, offsetof(Workspace, hidden) / sizeof(float),
        offsetof(Workspace, bonus) / sizeof(float), weights::kBonus, weights::kBonusBias);
    if (const auto error = cudaGetLastError(); error != cudaSuccess)
        return error;
    PolicyLogits<Batch>
        <<<logits_blocks, kThreads, 0, stream>>>(parameters, parameter_array, inputs, workspaces, logits, active);
    return cudaGetLastError();
}

} // namespace

cudaError_t Forward(const float *parameters, const Input *input, Workspace *workspace, float *logits,
                    cudaStream_t stream) {
    if (parameters == nullptr || input == nullptr || workspace == nullptr || logits == nullptr) {
        return cudaErrorInvalidValue;
    }
    return LaunchForward<false>(parameters, nullptr, input, workspace, logits, 1, nullptr, stream);
}

cudaError_t ForwardBatch(const float *const *parameters, const Input *inputs, Workspace *workspaces, float *logits,
                         int count, const int *active, cudaStream_t stream) {
    if (parameters == nullptr || inputs == nullptr || workspaces == nullptr || logits == nullptr || count < 1 ||
        count > 65535) {
        return cudaErrorInvalidValue;
    }
    return LaunchForward<true>(nullptr, parameters, inputs, workspaces, logits, count, active, stream);
}

} // namespace wordle_ga::model
