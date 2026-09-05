# CUDA policy inference

`wordle_policy` is a small CUDA static library. Its entry point is `wordle_ga::model::Forward` in
[`src/model/policy.hpp`](../src/model/policy.hpp). It evaluates one model on one already-encoded state and produces
4,739 raw FP32 action logits. `ForwardBatch` evaluates multiple model/state pairs with the same kernels. All network
calculations run on the GPU.

## Architecture

This implements the policy from `wordle-ml_machine-learning`, using the `wordle-cuda-f32-v1` parameter layout:

```text
candidate_mask[2309] / max(sum, 1) -> Linear(2309, 96) -> ReLU --+
candidate_stats[209]              -> Linear(209, 48)  -> ReLU --+-> hidden[160]
turn (0..5)                      -> Embedding(6, 16) ---------+

residual = Linear(160, 160)(ReLU(Linear(160, 160)(hidden)))
hidden   = ReLU(hidden + residual)
logits   = Linear(160, 4739)(hidden)
         + Linear(160, 1)(hidden) * remaining_action_mask[4739]
```

Every linear layer has a bias. There are 1,046,596 FP32 parameters (4,186,384 bytes); matrices are stored by output
row. The header defines the offsets of all 13 tensors. Vocabulary IDs must retain the ordering identified by the
source artifact's solution and action hashes.

`candidate_mask` marks compatible solutions. The 209 supplied statistics are 130 positional letter frequencies,
78 letter-multiplicity frequencies, and one normalized logarithmic candidate count. `remaining_action_mask` marks
candidate solutions in action-vocabulary order. It controls a signed learned bonus; non-candidate probe words retain
their base logits. An all-zero candidate mask uses a denominator of one, as in the source graph.

## Device-memory interface

```cpp
cudaError_t Forward(const float *device_weights, const Input *device_input,
                    Workspace *device_workspace, float *device_logits,
                    cudaStream_t stream = nullptr);
```

All pointers refer to caller-owned allocations on the selected CUDA device. The weight buffer has `weights::kCount`
floats and the output has `kNumActions` floats. Masks contain zero or one, weights/statistics are finite, and the turn
must be in `[0, 5]`. Input construction and validity are the caller's responsibility. The function rejects null
pointers but does not copy inputs to the CPU for validation.

The function enqueues seven kernels on the supplied stream. It performs no allocation, transfer, or synchronization.
The caller keeps allocations alive until completion and observes execution errors by synchronizing the stream or
waiting on an event. The return value reports argument and kernel-launch errors.

Enqueue weight and input updates on the same stream, or establish dependencies with CUDA events before inference.
In particular, a non-blocking stream must not rely on implicit ordering with work on the default stream.

Reuse weights and scratch across successive calls on the same stream. Concurrent calls may share read-only weights
and inputs but require separate workspaces and outputs. Scratch needs no initialization. The future fitness
evaluator can produce inputs and consume logits in device memory without a host round trip.

`ForwardBatch` takes a device array of weight pointers, contiguous input/workspace arrays, and a contiguous
`count * 4739` output array. Count must be 1 through 65,535. An optional device `int` activity array skips cases whose
entry is zero; those cases leave scratch/output untouched and may have null weight pointers. Active cases must
have valid inputs and live, immutable weights. The same stream/lifetime contract applies as for `Forward`.

The implementation uses one 128-thread block per dense output row and FP32 arithmetic. The
[fitness evaluator](fitness.md) supplies candidate filtering, statistics, action selection, and game scoring.
No CPU inference implementation or general model-file loader is included.

## Verification

`make inference` runs the CUDA forward pass against all 32 saved GoMLX reference cases, comparing all 151,648 logits
with absolute tolerance `1e-3`. Cases cover turns zero through five, full vocabularies, and small candidate sets.
The same test checks empty-candidate finiteness, positive and negative candidate bonuses, scratch reuse, and null
pointer rejection. The test uses a non-default CUDA stream and persistent device allocations.

`make test` includes this test and the GPU smoke test. `make test-gpu-sanitized` runs both under Compute Sanitizer.
Fixture checksums are verified during CMake configuration. See [fixture provenance](../tests/fixtures/policy/README.md).

The initial implementation passed on the RTX 5070 Ti (compute capability 12.0): all 151,648 reference logits were
within tolerance, with maximum absolute error `7.62939453e-06`. The clean rebuild and both CTest tests passed;
Compute Sanitizer memcheck reported zero errors. A separate racecheck run of `policy_test` reported zero hazards,
errors, or warnings.

Host-side test code loads saved arrays, manages CUDA resources, and compares results. It does not recalculate the
network. The trained checkpoint is test evidence, not a choice of initial population for the GA.
