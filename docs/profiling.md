# Nsight fitness profiling

Both Nsight Systems and Nsight Compute successfully profiled the CUDA fitness evaluator on the RTX 5070 Ti.
Run either capture from the repository root:

```sh
make profile-fitness-systems
make profile-fitness-compute
```

Reports and CSV/text exports are kept locally under `profiling/nsight-systems/` and `profiling/nsight-compute/`.
These generated files are ignored by Git. Open `fitness.nsys-rep` in the Systems GUI and `policy-logits.ncu-rep`
in the Compute GUI. Both commands run `fitness_test --full-training`: one saved model on all 2,109 training answers.

## Tool access

The verified setup uses host Nsight Systems **2026.1.3** and container Nsight Compute **2025.4.0** with CUDA 13.1.
The host also has Compute 2026.2.1; that version was not needed for these captures.

Systems is absent from the development image. Its target locates the host installation through `nsys`, mounts
the complete version directory read-only, and creates a temporary executable symlink inside the container. Set
`NSYS_HOST_DIR` explicitly if auto-discovery is unsuitable. CUDA/NVTX/OS runtime tracing works as the normal container
user with CPU sampling and context-switch tracing disabled.

Compute is already installed in the image. Running it as the normal container user returned `ERR_NVGPUCTRPERM`.
The profiling target grants counter access in a temporary container using root and `CAP_SYS_ADMIN`, as in the talk
project. Reports are returned to the invoking user's ownership, and summary export runs as the normal container
user. The host driver configuration and normal development service are unchanged.

## Initial findings

The initial capture profiled implementation commit `d421663`. The model again won **2,101/2,109** games, charged
7,696 guesses, and produced no invalid games. The Systems trace measured:

| Measurement | Result |
| --- | ---: |
| First game initialization through final score kernel | 53.403 ms |
| Summed GPU kernel time within that evaluation | 52.829 ms |
| Evaluation kernel launches | 505 |
| Logit output kernels, 54 launches | 43.392 ms |
| Logit share of evaluation kernel time | 82.14% |

The whole-process summary gives logits 81.2%; its denominator also includes startup/test kernels. Two feedback-table
builds outside the evaluation account for 0.573 ms: one is a primitive test and the other is evaluator startup.
The evaluation interval contains no host/device transfers. The results-buffer memset occurs before its first game
kernel. The trace therefore points to GPU kernel execution, especially the output layer, as the main cost.

The CUDA API summary lists about 51 ms in `cudaEventSynchronize`. This is the host waiting for queued GPU work,
not CPU calculation or an additional 51 ms to add to the GPU duration. Kernel submission totals about 2.1 ms over
the whole process and overlaps device execution.

Compute captures the **third matching logit launch**: turn three of the first 256-game batch. Its grid contains
1,213,184 blocks of 128 threads, one block per game/action pair. The report shows 20 registers per thread,
100% theoretical occupancy, approximately 60% achieved occupancy, 98.93% L2 hit rate, and 0.55% DRAM throughput.
Schedulers have no eligible warp about 66% of cycles; the tool highlights memory-access dependency stalls.
This makes the distribution of logit dot products across threads a useful next investigation. It does not establish
a speedup for any proposed implementation.

Compute required 40 replay passes to collect the full metric set. Its reported kernel duration (1.58 ms) and the
instrumented application's elapsed time (3.48 seconds) are not normal evaluation timings; the corresponding Systems
launch took 1.028 ms. Cache and replay conditions differ. Some games may already be complete at this turn, and their
blocks return early, so this is not an entirely active synthetic matrix benchmark.

These measurements describe one trained model, batch capacity 256, and this GPU. A full 1,024-genome generation,
different game lengths, or a different batch size requires its own measurement. Profiling did not change inference,
gameplay, fitness scoring, or the chosen FP32 precision.
