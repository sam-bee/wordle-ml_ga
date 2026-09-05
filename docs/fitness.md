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

The initial board is identical for every target. Each evaluation selects the opening action once per population
entry, then applies that action's feedback separately to every training answer. The cache holds only 1,024 action
IDs (4 KiB), indexed by population position, and is recomputed on every evaluation so changed weights or population
order cannot reuse stale actions. Opening selection and ordinary turns share the same CUDA argmax and feedback code.

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
for a fixed number of concurrent games. Default batch capacity is 256; persistent device memory totals 26,570,683
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
weights. The saved model wins 2,101 of 2,109 games, charges 7,696 guesses, and reports zero invalid games. Its score
is 22,162,104 and its wins on turns 1–6 are `[1, 59, 862, 988, 155, 36]`. The test requires every field to match exactly.

`make fitness-benchmark` uses 16 distinct genotype slots containing that same fixture, excludes one warmup, and
reports minimum, median, and maximum CUDA event time over five repetitions. Set `FITNESS_POPULATION=1024` and
`FITNESS_REPETITIONS=3` to measure a full population. Setup, weight upload, result copies, and checks are outside
timing. This measures fitness for that model's game lengths; evolving populations may take longer when they lose
more games. See [profiling results](profiling.md) for measured timings and before/after comparisons.

`make test` includes golden single/batched inference, saved-input encoding checks, repeated-letter feedback,
repeat suppression, probe selection, completed-game handling, and population evaluation with exact expected scores.
The population tests cover different weights, repeated/permuted slot IDs, batch boundaries, batch-size invariance,
and scratch reuse, including recomputing openings after population permutation or live weight changes between
evaluations. Temporary data copies verify that only the three training vocabulary files are required and
that changed files are rejected. `make test-gpu-sanitized` adds memory checks.

The optimized implementation passed a clean container rebuild and all eight GPU tests on the RTX 5070 Ti.
Compute Sanitizer memcheck reported zero errors across the suite, including full-training evaluation. Targeted
racecheck runs on batched inference and gameplay reported zero hazards; initcheck on population evaluation
reported zero errors.

This implementation uses ordinary kernels and a fixed five-turn launch loop after applying the opening. It does not compact
active games, use persistent kernels, cache fitness across generations, or implement breeding/selection.
See [Nsight profiling](profiling.md) for captured timeline and hardware-counter measurements.
