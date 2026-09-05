# Saved policy regression fixtures

These four files are unmodified copies from:

```text
../../talks/wordle/wordle-ml_machine-learning/
  runs/seed-replication-20260809-132505Z/exports/cuda-f32-v1/best/
```

The path above is relative to the repository root. The selected checkpoint is `best`, update 2,600, trained at commit
`2718164bb80460757592b90aa86b96eb6d596018`.

- `manifest.json`: original parameter shapes, offsets, vocabulary hashes, and weight hash.
- `weights.f32le`: 1,046,596 little-endian FP32 weights, in output-major matrix order.
- `golden-vectors.json`: original metadata for 32 reference inputs and GoMLX raw-logit outputs.
- `golden-vectors.f32le`: the arrays addressed by that metadata, including availability masks that inference does not use.

Weight SHA-256: `b78dc980505998d9dd40551ef4d24788b8378be63e4d09fb90aa0a8be83c870d`.

Golden-array SHA-256: `3c01bb0509a4894592d9250f32de3497518636bf229489548006de4511d3c826`.

The source project's exporter produced the expected logits through its GoMLX graph. The cases come from its validation
population and contain no final-test play. They cover all six turns and candidate counts from one through 2,309.
Original IDs, provenance, and action metadata remain in the JSON.

CMake verifies the binary hashes and generates a small C++ index from the JSON offsets and turn values. This avoids
introducing a runtime JSON dependency. Tests run on the project's Linux/x86-64 CUDA environment, which reads the
little-endian FP32 arrays directly. There is no fixture-generation model or CPU forward pass in this repository.

Keep these files fixed when changing inference. A failing comparison is not a reason to regenerate expected outputs.
The weights serve as regression data; whether they should seed an evolutionary population remains undecided.
