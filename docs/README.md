# Project documentation

See the [main README](../README.md) for setup, GPU selection,
and build/test commands. Repository-wide development and version-control guidance is in [AGENTS.md](../AGENTS.md).

- [CUDA inference](inference.md): model architecture, device-memory API, and verification.
- [Genotype slab](genotype-slab.md): fixed-width allocator, reference ownership, and breeding integration.

Reference projects relative to the repository root:

- `../../../../../ai/neuroevolution-wordle/codebase/`: original CUDA/C++ scaffolding and neuroevolution experiment.
- `../../talks/wordle/wordle-ml_machine-learning/`: candidate-based policy architecture, trained model, and CUDA inference.

The evolutionary algorithm, fitness evaluator, and use of the trained checkpoint to seed evolution remain to be decided.
