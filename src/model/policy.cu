#include "model/policy.hpp"

#include <cuda_runtime.h>

namespace wordle_ga::model {
namespace {

constexpr int kThreads = 128;
constexpr int kWarpSize = 32;

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

__global__ void PrepareInput(const Input *input, const float *parameters, Workspace *workspace) {
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
            parameters[weights::kTurnEmbedding + input->turn * kTurnWidth + threadIdx.x];
    }
}

// One block per output row; adjacent threads read adjacent FP32 weights.
template <int InputWidth, bool Relu, bool AddSkip = false>
__global__ void Dense(const float *input, const float *matrix, const float *bias, float *output,
                      const float *skip = nullptr) {
    const int row = blockIdx.x;
    float value = 0.0f;
    for (int i = threadIdx.x; i < InputWidth; i += kThreads) {
        value = fmaf(matrix[row * InputWidth + i], input[i], value);
    }
    value = BlockSum(value);
    if (threadIdx.x == 0) {
        value += bias[row];
        if constexpr (AddSkip) {
            value += skip[row];
        }
        output[row] = Relu ? fmaxf(value, 0.0f) : value;
    }
}

__global__ void PolicyLogits(const float *parameters, const Input *input, const Workspace *workspace, float *logits) {
    const int action = blockIdx.x;
    float value = 0.0f;
    for (int i = threadIdx.x; i < kTrunkWidth; i += kThreads) {
        value = fmaf(parameters[weights::kLogits + action * kTrunkWidth + i], workspace->hidden[i], value);
    }
    value = BlockSum(value);
    if (threadIdx.x == 0) {
        // The candidate bonus is not a legality mask: probe words retain their logits.
        logits[action] =
            value + parameters[weights::kLogitsBias + action] + workspace->bonus * input->remaining_action_mask[action];
    }
}

} // namespace

cudaError_t Forward(const float *parameters, const Input *input, Workspace *workspace, float *logits,
                    cudaStream_t stream) {
    if (parameters == nullptr || input == nullptr || workspace == nullptr || logits == nullptr) {
        return cudaErrorInvalidValue;
    }

    PrepareInput<<<1, kThreads, 0, stream>>>(input, parameters, workspace);
    if (const auto error = cudaGetLastError(); error != cudaSuccess) {
        return error;
    }
    Dense<kNumSolutions, true>
        <<<kCandidateWidth, kThreads, 0, stream>>>(workspace->normalized_candidates, parameters + weights::kCandidate,
                                                   parameters + weights::kCandidateBias, workspace->hidden);
    if (const auto error = cudaGetLastError(); error != cudaSuccess) {
        return error;
    }
    Dense<kCandidateStatsSize, true>
        <<<kStatsWidth, kThreads, 0, stream>>>(input->candidate_stats, parameters + weights::kStats,
                                               parameters + weights::kStatsBias, workspace->hidden + kCandidateWidth);
    if (const auto error = cudaGetLastError(); error != cudaSuccess) {
        return error;
    }
    Dense<kTrunkWidth, true><<<kTrunkWidth, kThreads, 0, stream>>>(workspace->hidden, parameters + weights::kResidualIn,
                                                                   parameters + weights::kResidualInBias,
                                                                   workspace->residual);
    if (const auto error = cudaGetLastError(); error != cudaSuccess) {
        return error;
    }
    // Each output reads only its own skip value, so hidden can be updated in place.
    Dense<kTrunkWidth, true, true><<<kTrunkWidth, kThreads, 0, stream>>>(
        workspace->residual, parameters + weights::kResidualOut, parameters + weights::kResidualOutBias,
        workspace->hidden, workspace->hidden);
    if (const auto error = cudaGetLastError(); error != cudaSuccess) {
        return error;
    }
    Dense<kTrunkWidth, false><<<1, kThreads, 0, stream>>>(workspace->hidden, parameters + weights::kBonus,
                                                          parameters + weights::kBonusBias, &workspace->bonus);
    if (const auto error = cudaGetLastError(); error != cudaSuccess) {
        return error;
    }
    PolicyLogits<<<kNumActions, kThreads, 0, stream>>>(parameters, input, workspace, logits);
    return cudaGetLastError();
}

} // namespace wordle_ga::model
