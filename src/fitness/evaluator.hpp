#pragma once

#include "fitness/vocabulary.hpp"
#include "genotype_slab/slab.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace wordle_ga::fitness {

inline constexpr int kDefaultBatchCapacity = 256;
inline constexpr int kMaxPopulation = 1024;

struct Result {
    std::uint32_t games;
    std::uint32_t wins;
    std::uint32_t guess_sum; // Losses and invalid games are charged all six guesses.
    std::uint32_t invalid_games;
    std::uint32_t solved_in[6];
    std::uint32_t score; // wins*(5*games+1) + (6*games-guess_sum); zero if invalid_games>0.
};

// Persistent, bounded GPU scratch plus immutable word/feedback tables. No gameplay
// or policy computation runs on the host. Caller selects device, manages population
// slot ownership, and waits for all uses before Destroy. One evaluation at a time
// per instance; same-stream reuse or explicit event ordering is required.
class Evaluator {
  public:
    Evaluator() = default;
    ~Evaluator();
    Evaluator(const Evaluator &) = delete;
    Evaluator &operator=(const Evaluator &) = delete;
    Evaluator(Evaluator &&) = delete;
    Evaluator &operator=(Evaluator &&) = delete;

    // Use the unchanged result of LoadVocabulary. Capacity is fixed until Destroy
    // and must be 1..65535; creation waits for startup work and cleans up on failure.
    cudaError_t Create(const Vocabulary &vocabulary, int batch_capacity = kDefaultBatchCapacity,
                       cudaStream_t stream = nullptr);
    cudaError_t Destroy();

    // Plays every training answer once for each population entry. Slots and results
    // are device arrays of population_count entries (1..kMaxPopulation). Duplicate
    // slots are allowed.
    // Live genotypes must remain immutable until completion. Output is overwritten.
    // This call enqueues work only; synchronize stream to observe execution errors
    // and read results. Invalid slot IDs/non-finite logits produce invalid games.
    cudaError_t Evaluate(genotype_slab::DeviceView slab, const genotype_slab::Slot *population_slots,
                         int population_count, Result *results, cudaStream_t stream = nullptr);

    std::size_t allocated_bytes() const;

  private:
    struct Storage;
    Storage *storage_ = nullptr;
};

} // namespace wordle_ga::fitness
