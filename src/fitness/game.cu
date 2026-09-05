#include "fitness/game.cuh"

#include "genotype_slab/device.cuh"

#include <cmath>

namespace wordle_ga::fitness {
namespace {

constexpr int kStats = 208;

__global__ void BuildFeedbackKernel(const Word *actions, const Word *solutions, std::uint8_t *out) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= model::kNumActions * model::kNumSolutions)
        return;
    const int action = index / model::kNumSolutions;
    const int solution = index % model::kNumSolutions;
    out[index] = Feedback(actions[action], solutions[solution]);
}

__global__ void StartKernel(Tables t, genotype_slab::DeviceView slab, const genotype_slab::Slot *slots,
                            std::uint64_t first, int count, Game *games, int *active, const float **parameters) {
    const int i = blockIdx.x;
    if (i >= count)
        return;
    const std::uint64_t pair = first + static_cast<std::uint64_t>(i);
    Game &g = games[i];
    if (threadIdx.x == 0) {
        parameters[i] = nullptr;
        g.organism = static_cast<int>(pair / kTrainingSolutions);
        g.guesses = 0;
        g.won = 0;
        g.invalid = 0;
        g.target = 0;
    }
    for (int c = threadIdx.x; c < model::kNumSolutions; c += blockDim.x)
        g.candidates[c] = 1;
    for (int h = threadIdx.x; h < model::kNumTurns; h += blockDim.x)
        g.history[h] = 0;
    if (threadIdx.x == 0) {
        g.target = t.training_solutions[pair % kTrainingSolutions];
        const genotype_slab::Slot slot = slots[g.organism];
        bool ok = g.target < model::kNumSolutions && slot < slab.capacity && slab.references != nullptr &&
                  slab.genotypes != nullptr;
        if (ok)
            ok = atomicAdd(slab.references + slot, 0u) != 0;
        if (ok)
            parameters[i] = genotype_slab::Parameters(slab, slot);
        if (!ok) {
            g.invalid = 1;
            active[i] = 0;
            parameters[i] = nullptr;
        } else
            active[i] = 1;
    }
}

__global__ void EncodeKernel(Tables t, const Game *games, const int *active, int count, model::Input *inputs) {
    const int game = blockIdx.x;
    if (game >= count || (active && active[game] == 0))
        return;
    __shared__ int hist[kStats];
    for (int i = threadIdx.x; i < kStats; i += blockDim.x)
        hist[i] = 0;
    __syncthreads();
    const Game &g = games[game];
    for (int s = threadIdx.x; s < model::kNumSolutions; s += blockDim.x) {
        if (!g.candidates[s])
            continue;
        const Word w = t.solutions[s];
        int copies[26]{};
        for (int p = 0; p < 5; ++p) {
            atomicAdd(&hist[p * 26 + w.letters[p]], 1);
            ++copies[w.letters[p]];
        }
        for (int l = 0; l < 26; ++l)
            for (int threshold = 1; threshold <= 3; ++threshold)
                if (copies[l] >= threshold)
                    atomicAdd(&hist[130 + l * 3 + threshold - 1], 1);
    }
    __syncthreads();
    model::Input &in = inputs[game];
    for (int s = threadIdx.x; s < model::kNumSolutions; s += blockDim.x) {
        in.candidate_mask[s] = static_cast<float>(g.candidates[s] != 0);
    }
    // The first position's histogram counts every remaining word exactly once.
    __shared__ int candidate_count;
    if (threadIdx.x == 0) {
        candidate_count = 0;
        for (int letter = 0; letter < 26; ++letter)
            candidate_count += hist[letter];
    }
    for (int a = threadIdx.x; a < model::kNumActions; a += blockDim.x)
        in.remaining_action_mask[a] = 0.0f;
    __syncthreads();
    for (int s = threadIdx.x; s < model::kNumSolutions; s += blockDim.x) {
        if (g.candidates[s]) {
            in.remaining_action_mask[t.solution_actions[s]] = 1.0f;
        }
    }
    __syncthreads();
    for (int j = threadIdx.x; j < kStats; j += blockDim.x) {
        const float denom = candidate_count > 0 ? static_cast<float>(candidate_count) : 1.0f;
        in.candidate_stats[j] = static_cast<float>(hist[j]) / denom;
    }
    if (threadIdx.x == 0) {
        in.candidate_stats[kStats] = candidate_count > 0
                                         ? static_cast<float>(log(static_cast<double>(candidate_count)) /
                                                              log(static_cast<double>(model::kNumSolutions)))
                                         : 0.0f;
        in.turn = g.guesses;
    }
}

__global__ void AdvanceKernel(Tables t, Game *games, int *active, int count, const float *logits) {
    const int i = blockIdx.x;
    if (i >= count || active[i] == 0)
        return;
    Game &g = games[i];
    const float *row = logits + static_cast<std::size_t>(i) * model::kNumActions;
    __shared__ int best_id[128];
    __shared__ float best_value[128];
    __shared__ int bad;
    if (threadIdx.x == 0)
        bad = 0;
    __syncthreads();
    int local_id = -1;
    float local_value = -INFINITY;
    for (int a = threadIdx.x; a < model::kNumActions; a += blockDim.x) {
        const float value = row[a];
        if (!isfinite(value)) {
            atomicExch(&bad, 1);
            continue;
        }
        bool used = false;
        for (int h = 0; h < g.guesses; ++h)
            if (g.history[h] == a)
                used = true;
        if (!used && (local_id < 0 || value > local_value || (value == local_value && a < local_id))) {
            local_id = a;
            local_value = value;
        }
    }
    best_id[threadIdx.x] = local_id;
    best_value[threadIdx.x] = local_value;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const int other = best_id[threadIdx.x + stride];
            if (other >= 0 &&
                (best_id[threadIdx.x] < 0 || best_value[threadIdx.x + stride] > best_value[threadIdx.x] ||
                 (best_value[threadIdx.x + stride] == best_value[threadIdx.x] && other < best_id[threadIdx.x]))) {
                best_id[threadIdx.x] = other;
                best_value[threadIdx.x] = best_value[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (bad || best_id[0] < 0 || g.guesses >= model::kNumTurns) {
            g.invalid = 1;
            active[i] = 0;
        } else
            g.history[g.guesses] = static_cast<std::uint16_t>(best_id[0]);
    }
    __syncthreads();
    if (active[i] == 0)
        return;
    const int best = g.history[g.guesses];
    const std::uint8_t mark = t.feedback[static_cast<std::size_t>(best) * model::kNumSolutions + g.target];
    if (mark != 242)
        for (int s = threadIdx.x; s < model::kNumSolutions; s += blockDim.x)
            if (g.candidates[s])
                g.candidates[s] = t.feedback[static_cast<std::size_t>(best) * model::kNumSolutions + s] == mark;
    __syncthreads();
    if (threadIdx.x == 0) {
        ++g.guesses;
        if (mark == 242) {
            g.won = 1;
            active[i] = 0;
        } else if (g.guesses >= model::kNumTurns)
            active[i] = 0;
    }
}

__global__ void AccumulateKernel(const Game *games, int count, Result *results) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const Game &g = games[i];
    Result &r = results[g.organism];
    atomicAdd(&r.games, 1u);
    atomicAdd(&r.wins, static_cast<unsigned>(g.won));
    atomicAdd(&r.guess_sum, static_cast<unsigned>(g.won ? g.guesses : 6));
    atomicAdd(&r.invalid_games, static_cast<unsigned>(g.invalid));
    if (g.won && g.guesses >= 1 && g.guesses <= 6)
        atomicAdd(&r.solved_in[g.guesses - 1], 1u);
}

__global__ void FinishKernel(Result *results, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    Result &r = results[i];
    r.score = r.invalid_games ? 0u : r.wins * (5u * r.games + 1u) + 6u * r.games - r.guess_sum;
}
} // namespace

cudaError_t BuildFeedback(const Word *a, const Word *s, std::uint8_t *f, cudaStream_t stream) {
    if (!a || !s || !f)
        return cudaErrorInvalidValue;
    BuildFeedbackKernel<<<(model::kNumActions * model::kNumSolutions + 127) / 128, 128, 0, stream>>>(a, s, f);
    return cudaGetLastError();
}
cudaError_t StartGames(Tables t, genotype_slab::DeviceView slab, const genotype_slab::Slot *slots, std::uint64_t first,
                       int count, Game *g, int *a, const float **p, cudaStream_t stream) {
    if (count < 0 || count > 65535 || first > kMaxPopulation * kTrainingSolutions ||
        static_cast<std::uint64_t>(count) > kMaxPopulation * kTrainingSolutions - first || !t.training_solutions ||
        !slots || !g || !a || !p)
        return cudaErrorInvalidValue;
    if (!count)
        return cudaSuccess;
    StartKernel<<<count, 128, 0, stream>>>(t, slab, slots, first, count, g, a, p);
    return cudaGetLastError();
}
cudaError_t EncodeGames(Tables t, const Game *g, const int *a, int count, model::Input *in, cudaStream_t stream) {
    if (count < 0 || count > 65535 || !t.solutions || !t.solution_actions || !g || !in)
        return cudaErrorInvalidValue;
    if (!count)
        return cudaSuccess;
    EncodeKernel<<<count, 128, 0, stream>>>(t, g, a, count, in);
    return cudaGetLastError();
}
cudaError_t AdvanceGames(Tables t, Game *g, int *a, int count, const float *l, cudaStream_t stream) {
    if (count < 0 || count > 65535 || !g || !a || !l || !t.feedback)
        return cudaErrorInvalidValue;
    if (!count)
        return cudaSuccess;
    AdvanceKernel<<<count, 128, 0, stream>>>(t, g, a, count, l);
    return cudaGetLastError();
}
cudaError_t AccumulateGames(const Game *g, int count, Result *r, cudaStream_t stream) {
    if (count < 0 || count > 65535 || !g || !r)
        return cudaErrorInvalidValue;
    if (!count)
        return cudaSuccess;
    AccumulateKernel<<<(count + 127) / 128, 128, 0, stream>>>(g, count, r);
    return cudaGetLastError();
}
cudaError_t FinishResults(Result *r, int n, cudaStream_t stream) {
    if (n < 0 || n > kMaxPopulation || !r)
        return cudaErrorInvalidValue;
    if (!n)
        return cudaSuccess;
    FinishKernel<<<(n + 127) / 128, 128, 0, stream>>>(r, n);
    return cudaGetLastError();
}
} // namespace wordle_ga::fitness
