#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace wordle_ga::model {

inline constexpr int kNumSolutions = 2309;
inline constexpr int kNumActions = 4739;
inline constexpr int kCandidateStatsSize = 209;
inline constexpr int kNumTurns = 6;
inline constexpr int kCandidateWidth = 96;
inline constexpr int kStatsWidth = 48;
inline constexpr int kTurnWidth = 16;
inline constexpr int kTrunkWidth = kCandidateWidth + kStatsWidth + kTurnWidth;

// Flat FP32 layout from wordle-cuda-f32-v1. Dense matrices are [output, input].
namespace weights {
inline constexpr std::size_t kCandidate = 0;
inline constexpr std::size_t kCandidateBias = kCandidate + kCandidateWidth * kNumSolutions;
inline constexpr std::size_t kStats = kCandidateBias + kCandidateWidth;
inline constexpr std::size_t kStatsBias = kStats + kStatsWidth * kCandidateStatsSize;
inline constexpr std::size_t kTurnEmbedding = kStatsBias + kStatsWidth;
inline constexpr std::size_t kResidualIn = kTurnEmbedding + kNumTurns * kTurnWidth;
inline constexpr std::size_t kResidualInBias = kResidualIn + kTrunkWidth * kTrunkWidth;
inline constexpr std::size_t kResidualOut = kResidualInBias + kTrunkWidth;
inline constexpr std::size_t kResidualOutBias = kResidualOut + kTrunkWidth * kTrunkWidth;
inline constexpr std::size_t kLogits = kResidualOutBias + kTrunkWidth;
inline constexpr std::size_t kLogitsBias = kLogits + kNumActions * kTrunkWidth;
inline constexpr std::size_t kBonus = kLogitsBias + kNumActions;
inline constexpr std::size_t kBonusBias = kBonus + kTrunkWidth;
inline constexpr std::size_t kCount = kBonusBias + 1;
static_assert(kCount == 1046596);
} // namespace weights

struct Input {
    float candidate_mask[kNumSolutions];
    float candidate_stats[kCandidateStatsSize];
    float remaining_action_mask[kNumActions];
    int turn;
};

// Scratch belongs to the caller. Every field is overwritten during a forward pass.
struct Workspace {
    float normalized_candidates[kNumSolutions];
    float hidden[kTrunkWidth];
    float residual[kTrunkWidth];
    float bonus;
};

// All four pointers address device memory; logits has kNumActions FP32 entries.
// Input masks contain 0/1, statistics and weights are finite, and turn is in [0, 5].
// Empty candidate masks use a denominator of one, matching the source graph.
// Allocations must not overlap. They must remain alive until the stream completes.
// The caller selects the device and owns allocations, transfers, and synchronization.
// Order writes to inputs/weights on this stream, or wait for them with CUDA events.
// A workspace can be reused in stream order; concurrent calls need separate workspaces
// and outputs. Returns launch errors only; execution errors surface at synchronization.
cudaError_t Forward(const float *device_weights, const Input *device_input, Workspace *device_workspace,
                    float *device_logits, cudaStream_t stream = nullptr);

} // namespace wordle_ga::model
