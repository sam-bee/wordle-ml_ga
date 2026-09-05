#include "fitness/evaluator.hpp"

#include "fitness/game.cuh"
#include "model/policy.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>

namespace wordle_ga::fitness {
struct Evaluator::Storage {
    Word *actions = nullptr;
    Word *solutions = nullptr;
    std::uint16_t *solution_actions = nullptr;
    std::uint16_t *training_solutions = nullptr;
    std::uint8_t *feedback = nullptr;
    Game *games = nullptr;
    int *active = nullptr;
    const float **parameters = nullptr;
    model::Input *inputs = nullptr;
    model::Workspace *workspaces = nullptr;
    float *logits = nullptr;
    int *opening_actions = nullptr;
    int batch_capacity = 0;
    std::size_t bytes = 0;
};

namespace {

template <typename T> cudaError_t Allocate(T **pointer, std::size_t count) {
    return cudaMalloc(reinterpret_cast<void **>(pointer), count * sizeof(T));
}

} // namespace

Evaluator::~Evaluator() { (void)Destroy(); }

cudaError_t Evaluator::Create(const Vocabulary &vocabulary, int batch_capacity, cudaStream_t stream) {
    if (storage_ != nullptr || batch_capacity < 1 || batch_capacity > 65535)
        return cudaErrorInvalidValue;
    for (const Word &word : vocabulary.actions) {
        for (std::uint8_t letter : word.letters) {
            if (letter >= 26)
                return cudaErrorInvalidValue;
        }
    }
    for (const Word &word : vocabulary.solutions) {
        for (std::uint8_t letter : word.letters) {
            if (letter >= 26)
                return cudaErrorInvalidValue;
        }
    }
    for (std::uint16_t action : vocabulary.solution_actions) {
        if (action >= model::kNumActions)
            return cudaErrorInvalidValue;
    }
    for (std::uint16_t solution : vocabulary.training_solutions) {
        if (solution >= model::kNumSolutions)
            return cudaErrorInvalidValue;
    }
    Storage *pending = new (std::nothrow) Storage;
    if (pending == nullptr)
        return cudaErrorMemoryAllocation;
    const auto cleanup = [&pending]() {
        cudaFree(pending->logits);
        cudaFree(pending->opening_actions);
        cudaFree(pending->workspaces);
        cudaFree(pending->inputs);
        cudaFree(const_cast<float **>(pending->parameters));
        cudaFree(pending->active);
        cudaFree(pending->games);
        cudaFree(pending->feedback);
        cudaFree(pending->training_solutions);
        cudaFree(pending->solution_actions);
        cudaFree(pending->solutions);
        cudaFree(pending->actions);
        delete pending;
    };
    pending->batch_capacity = batch_capacity;

    cudaError_t error = Allocate(&pending->actions, model::kNumActions);
    if (error == cudaSuccess)
        error = Allocate(&pending->solutions, model::kNumSolutions);
    if (error == cudaSuccess)
        error = Allocate(&pending->solution_actions, model::kNumSolutions);
    if (error == cudaSuccess)
        error = Allocate(&pending->training_solutions, kTrainingSolutions);
    if (error == cudaSuccess)
        error = Allocate(&pending->feedback, static_cast<std::size_t>(model::kNumActions) * model::kNumSolutions);
    if (error == cudaSuccess)
        error = Allocate(&pending->games, batch_capacity);
    if (error == cudaSuccess)
        error = Allocate(&pending->active, batch_capacity);
    if (error == cudaSuccess)
        error = cudaMalloc(reinterpret_cast<void **>(&pending->parameters),
                           static_cast<std::size_t>(batch_capacity) * sizeof(float *));
    if (error == cudaSuccess)
        error = Allocate(&pending->inputs, batch_capacity);
    if (error == cudaSuccess)
        error = Allocate(&pending->workspaces, batch_capacity);
    if (error == cudaSuccess)
        error = Allocate(&pending->logits, static_cast<std::size_t>(batch_capacity) * model::kNumActions);
    if (error == cudaSuccess)
        error = Allocate(&pending->opening_actions, kMaxPopulation);
    if (error != cudaSuccess) {
        cleanup();
        return error;
    }

    error = cudaMemcpyAsync(pending->actions, vocabulary.actions.data(), sizeof(vocabulary.actions),
                            cudaMemcpyHostToDevice, stream);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(pending->solutions, vocabulary.solutions.data(), sizeof(vocabulary.solutions),
                                cudaMemcpyHostToDevice, stream);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(pending->solution_actions, vocabulary.solution_actions.data(),
                                sizeof(vocabulary.solution_actions), cudaMemcpyHostToDevice, stream);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(pending->training_solutions, vocabulary.training_solutions.data(),
                                sizeof(vocabulary.training_solutions), cudaMemcpyHostToDevice, stream);
    if (error == cudaSuccess) {
        error = BuildFeedback(pending->actions, pending->solutions, pending->feedback, stream);
    }
    if (error == cudaSuccess)
        error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) {
        cleanup();
        return error;
    }

    pending->bytes = static_cast<std::size_t>(model::kNumActions) * sizeof(Word) +
                     static_cast<std::size_t>(model::kNumSolutions) * sizeof(Word) +
                     static_cast<std::size_t>(model::kNumSolutions + kTrainingSolutions) * sizeof(std::uint16_t) +
                     static_cast<std::size_t>(model::kNumActions) * model::kNumSolutions * sizeof(std::uint8_t) +
                     static_cast<std::size_t>(batch_capacity) *
                         (sizeof(Game) + sizeof(int) + sizeof(float *) + sizeof(model::Input) +
                          sizeof(model::Workspace) + static_cast<std::size_t>(model::kNumActions) * sizeof(float));
    pending->bytes += static_cast<std::size_t>(kMaxPopulation) * sizeof(int);
    storage_ = pending;
    return cudaSuccess;
}

cudaError_t Evaluator::Destroy() {
    if (storage_ == nullptr)
        return cudaSuccess;
    Storage *old = storage_;
    cudaError_t first = cudaSuccess;
    auto free_one = [&first](void *pointer) {
        if (pointer == nullptr)
            return;
        const cudaError_t error = cudaFree(pointer);
        if (first == cudaSuccess && error != cudaSuccess)
            first = error;
    };
    free_one(old->logits);
    free_one(old->opening_actions);
    free_one(old->workspaces);
    free_one(old->inputs);
    free_one(const_cast<float **>(old->parameters));
    free_one(old->active);
    free_one(old->games);
    free_one(old->feedback);
    free_one(old->training_solutions);
    free_one(old->solution_actions);
    free_one(old->solutions);
    free_one(old->actions);
    delete old;
    storage_ = nullptr;
    return first;
}

cudaError_t Evaluator::Evaluate(genotype_slab::DeviceView slab, const genotype_slab::Slot *population_slots,
                                int population_count, Result *results, cudaStream_t stream) {
    if (storage_ == nullptr || population_slots == nullptr || results == nullptr || population_count < 1 ||
        population_count > kMaxPopulation || slab.genotypes == nullptr || slab.capacity == 0 ||
        slab.references == nullptr) {
        return cudaErrorInvalidValue;
    }
    Storage &s = *storage_;
    cudaError_t error =
        cudaMemsetAsync(results, 0, static_cast<std::size_t>(population_count) * sizeof(Result), stream);
    if (error != cudaSuccess)
        return error;
    const Tables tables{s.actions, s.solutions, s.solution_actions, s.training_solutions, s.feedback};
    const std::uint64_t total = static_cast<std::uint64_t>(population_count) * kTrainingSolutions;
    for (int first = 0; first < population_count; first += s.batch_capacity) {
        const int count = std::min(s.batch_capacity, population_count - first);
        error =
            StartOpeningGames(tables, slab, population_slots, first, count, s.games, s.active, s.parameters, stream);
        if (error != cudaSuccess)
            return error;
        error = EncodeGames(tables, s.games, s.active, count, s.inputs, stream);
        if (error != cudaSuccess)
            return error;
        error = model::ForwardBatch(s.parameters, s.inputs, s.workspaces, s.logits, count, s.active, stream);
        if (error != cudaSuccess)
            return error;
        error = SelectOpeningActions(s.games, s.active, count, s.logits, s.opening_actions, stream);
        if (error != cudaSuccess)
            return error;
    }
    for (std::uint64_t first = 0; first < total; first += s.batch_capacity) {
        const int count = static_cast<int>(std::min<std::uint64_t>(s.batch_capacity, total - first));
        error = StartGames(tables, slab, population_slots, first, count, s.games, s.active, s.parameters, stream);
        if (error != cudaSuccess)
            return error;
        error = ApplyOpeningActions(tables, s.games, s.active, count, s.opening_actions, stream);
        if (error != cudaSuccess)
            return error;
        for (int turn = 1; turn < model::kNumTurns; ++turn) {
            error = EncodeGames(tables, s.games, s.active, count, s.inputs, stream);
            if (error != cudaSuccess)
                return error;
            error = model::ForwardBatch(s.parameters, s.inputs, s.workspaces, s.logits, count, s.active, stream);
            if (error != cudaSuccess)
                return error;
            error = AdvanceGames(tables, s.games, s.active, count, s.logits, stream);
            if (error != cudaSuccess)
                return error;
        }
        error = AccumulateGames(s.games, count, results, stream);
        if (error != cudaSuccess)
            return error;
    }
    return FinishResults(results, population_count, stream);
}

std::size_t Evaluator::allocated_bytes() const { return storage_ == nullptr ? 0 : storage_->bytes; }

} // namespace wordle_ga::fitness
