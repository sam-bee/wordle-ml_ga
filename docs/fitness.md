# CUDA fitness evaluation

[`fitness::Evaluator`](../src/fitness/evaluator.hpp) plays one complete game for every training answer and every
population entry. At 1,024 organisms this is 2,159,616 games, each taking at most six guesses. All inference,
feedback, candidate filtering, input statistics, action selection, and fitness accumulation run on the GPU.

## Games and score

Every game begins with all 2,309 possible answers as candidates. Only the 2,109 frozen training answers are used as
targets. The evaluator never loads validation or final-test files. All 4,739 actions remain eligible guesses except
those already played in that game; probe words are allowed even when they are outside the candidate set.

The largest logit wins, with ties resolved by the lowest canonical action ID. Green matches consume letters before
yellow matches, so duplicate letters follow Wordle rules. Correct guesses end the game immediately; the sixth
incorrect guess ends it as a loss. Completed games skip subsequent encoding, inference, and advancement.

Each device `Result` contains games, wins, total charged guesses, wins on each of turns 1–6, invalid games, and a
higher-is-better integer score:

```text
score = wins * (5 * games + 1) + (6 * games - guess_sum)
```

Losses cost six guesses. For equal game counts, one additional win outweighs every possible guess-count tiebreaker;
among equally successful organisms, fewer guesses win. Scores are exact integers, avoiding FP32 rounding of close
rankings. A non-finite logit or invalid/non-live genotype slot makes that game invalid and gives the organism a
score of zero. `invalid_games` makes this distinguishable from an ordinary zero-win organism.

## Bounded memory and ownership

`Create` uploads words and ID mappings, computes an immutable byte feedback table on the GPU, and allocates scratch
for a fixed number of concurrent games. Default batch capacity is 256; persistent device memory totals 26,566,587
bytes (25.34 MiB), including the 10,942,351-byte feedback table. Genotypes stay in the caller's slab and results use
44 bytes per organism. Scratch does not grow with population size and is reused between batches and evaluations.

`Evaluate` accepts 1–1,024 population entries as a device array of slab slot IDs. Population positions and slab
slots are independent; several entries may reference the same genotype. It overwrites one result per population
entry and handles incomplete final batches. No per-turn or per-generation CUDA memory allocation is performed.

The caller keeps population IDs and live genotype weights immutable until evaluation completes, holding the required
slab references throughout. The evaluator does not retain/release slots itself. Do not breed into or reclaim weights
while they are being evaluated.

An evaluator has one set of scratch buffers: reuse it in stream order, or use events to order calls across streams.
Concurrent evaluations need separate evaluator instances and result buffers. `Create` waits for startup transfers
and feedback generation. `Evaluate` enqueues work without explicit synchronization or host result transfers; the
caller synchronizes its stream before reading results or destroying resources.

```cpp
auto vocabulary = fitness::LoadVocabulary(data_directory);
fitness::Evaluator evaluator;
// Check each returned CUDA error in the caller.
evaluator.Create(vocabulary, fitness::kDefaultBatchCapacity, stream);
evaluator.Evaluate(slab.view(), device_population_slots, population_count, device_results, stream);
cudaStreamSynchronize(stream);
// Device results can feed GPU selection directly, or be copied for reporting.
evaluator.Destroy();
```

Vocabulary loading is host file I/O and ID mapping. Its three CSVs must exactly match the frozen build inputs;
changed ordering or training membership is rejected. Pass the returned vocabulary unchanged to `Create`.

## Verification and limits

`make fitness` evaluates the existing trained regression fixture on all training answers and reports CUDA event
time excluding setup and weight upload. It is a smoke benchmark, not a decision to initialize evolution from those
weights. A first RTX 5070 Ti run won 2,101 of 2,109 games, charged 7,696 guesses, and reported zero invalid games in
53.7 ms. This is one model's training-set result, not a measurement of a full population or held-out performance.

`make test` includes golden single/batched inference, saved-input encoding checks, repeated-letter feedback,
repeat suppression, probe selection, completed-game handling, and population evaluation with exact expected scores.
The population tests cover different weights, repeated/permuted slot IDs, batch boundaries, batch-size invariance,
and scratch reuse. Temporary data copies verify that only the three training vocabulary files are required and
that changed files are rejected. `make test-gpu-sanitized` adds memory checks.

The initial clean container rebuild passed all seven GPU tests on the RTX 5070 Ti. Compute Sanitizer memcheck
reported zero errors for the full suite; separate racecheck runs on the gameplay and batched-policy tests reported
zero hazards, and initcheck on the population evaluator test reported zero errors.

This is a straightforward baseline using ordinary kernels and fixed six-turn launch loops. It does not compact
active games, use persistent kernels, cache fitness across generations, or implement breeding/selection.
