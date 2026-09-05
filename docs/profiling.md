# Nsight fitness profiling

The current evaluator was measured on an RTX 5070 Ti (CUDA 13.1, `sm_120`, Release build), with FP32 weights,
batch capacity 256, one saved model copied into distinct genotype-slab slots, and all 2,109 training answers. The
timed benchmark excludes one warmup. Every timed result for this saved model matched its fixed golden baseline:
2,109 games, 2,101 wins, 7,696 guesses, score 22,162,104, zero invalid games, and the same `solved_in` histogram
(`1 59 862 988 155 36`).
Validation and final-test answers were not included.

| Population | Baseline median (ms) | Current min / median / max (ms) |
| ---: | ---: | ---: |
| 1 (9 timed repetitions) | 53.461 | 12.132 / 12.172 / 12.182 |
| 16 (5 timed repetitions) | 852.970 | 191.645 / 192.114 / 192.133 |
| 1,024 (1 timed repetition for baseline; 3 for current) | 54,593.840 | 12,288.061 / 12,288.123 / 12,386.154 |

The benchmark command is `make fitness-benchmark FITNESS_POPULATION=<population> FITNESS_REPETITIONS=<repetitions>`;
the defaults are population 16 and 5 repetitions. The 1,024-organism baseline is the historical single timed
repetition after one warmup. The current full-population result uses three timed repetitions after one warmup.
The source records are `profiling/baseline-be88d7b/benchmark.txt` and `profiling/benchmark-optimized.txt`; these
generated profiling artifacts are ignored by Git.

Opening reuse alone reduced the single-model median to 42.334 ms. Tiling the logit layer reduced it to 14.827 ms;
applying warp reductions to the other dense layers produced the final 12.172 ms result.

The evaluator now computes one opening action per organism instead of one per answer, refreshing a 4 KiB cache
on every evaluation. The packed logit kernel handles 64 output rows per block and launches
19,200 blocks for a 256-game batch, versus 1,213,184 blocks in the old one-game/action-pair layout. It uses
warp reductions; the other dense layers now handle four rows per block. These are ordinary CUDA kernels using
FP32, with no active-game compaction. Reduction order changed, so arbitrary weights need not produce bitwise
identical logits; the saved reference logits remain within tolerance and the reference game results match exactly.
Less successful genomes can take longer because game lengths and early exits vary.

## Nsight Systems interval accounting

The final Systems capture was summed over the exact device interval from the first `StartOpeningKernel` through the
end of the final `FinishKernel`. The baseline interval starts at the first `StartKernel` and ends at its final
`FinishKernel`. Startup, allocation, model and table copies, and other work outside those boundaries are excluded;
there are no memcpy operations inside either interval. GPU event durations include gaps between launches on the
stream, so the interval is the closest trace counterpart to the benchmark's elapsed evaluation.

| Capture | Interval | CUDA kernel calls | Summed kernel time | `PolicyLogits` calls / summed time | Logit share of summed kernels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline (`StartKernel` → `FinishKernel`) | 53.402999 ms | 505 | 52.828622 ms | 54 / 43.391675 ms | 82.14% |
| Current (`StartOpeningKernel` → `FinishKernel`) | 12.625396 ms | 443 | 12.009038 ms | 46 / 7.388962 ms | 61.53% |

The third current `PolicyLogits` launch lasted 299.395 µs in the Systems trace. The final interval's remaining time
is distributed across the 46 opening/gameplay passes and dense layers; the large reduction comes from doing one
opening inference per organism and from the packed logits launch.

The old capture also showed two feedback-table builds outside the evaluation interval and about 51 ms in
`cudaEventSynchronize`; the latter is host waiting for queued GPU work, not additional CPU fitness computation.

## Nsight Compute

The Compute capture uses `--set full --kernel-name regex:PolicyLogits --launch-skip 2 --launch-count 1` and therefore
profiles the third matching logit launch. The current packed launch has 19,200 blocks of 128 threads. Nsight Compute
used 40 replay passes to collect the full metric set, so its instrumented duration is not a normal evaluation timing.
The final summary reports:

| Metric | Current packed launch |
| --- | ---: |
| Duration | 405.70 µs |
| Registers / thread | 28 |
| Static shared memory / block | 640 bytes |
| Theoretical occupancy | 100% |
| Achieved occupancy | 96.63% |
| L2 hit rate | 99.02% |
| DRAM throughput | 2.16% |
| No eligible scheduler cycles | 60.03% |

For comparison, the old one-game/action-pair launch had 1,213,184 blocks, 20 registers per thread, 16 bytes of
static shared memory per block, 60.11% achieved occupancy, 98.93% L2 hit rate, 0.55% DRAM throughput, and 66.24%
no-eligible cycles. Its Compute duration was 1.58 ms. Replay, cache, and early-exit conditions differ, so these are
kernel diagnostics rather than a standalone speedup experiment. The current report still identifies L1TEX scoreboard
dependencies as the principal stall source; high cache hit rate and low DRAM use point to scheduling and dependency
latency rather than external memory bandwidth.

## Tool access

Run either capture from the repository root:

```sh
make profile-fitness-systems
make profile-fitness-compute
```

Both profile `fitness_test --full-training` with one saved model on all 2,109 training answers. Nsight Systems uses
host Nsight Systems 2026.1.3 (CUDA 13.1) through the `NSYS_HOST_DIR` mount; Nsight Compute 2025.4.0 is installed in
the development image and requires the temporary root/`CAP_SYS_ADMIN` profiling container for hardware counters.
The host driver and normal development service are unchanged. Open `profiling/nsight-systems/fitness.nsys-rep` in
the Systems GUI and `profiling/nsight-compute/policy-logits.ncu-rep` in the Compute GUI. CSV/text exports beside
those reports, plus the historical baseline under `profiling/baseline-be88d7b/`, are local ignored artifacts.
