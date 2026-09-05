# Instructions to Coding Agents for Neuroevolutionary Wordle

Adapted from `../../../../../ai/neuroevolution-wordle/codebase/docs/AGENTS.md`, with the container and GPU guidance
from `../../talks/wordle/wordle-ml_machine-learning/AGENTS.md`. The user's current branch preference is recorded below.

## Project and References

Read `README.md` before working on the project. Project documentation belongs in `docs/` and should be updated as
implementation progresses.

We are building a CUDA/C++ genetic algorithm to breed neural agents that play Wordle. The original GA repository is
the reference for scaffolding and overall goals. The machine-learning repository provides the policy architecture;
its trained checkpoint may be useful, but adopting it remains a separate decision. Algorithmic design and software
architecture will differ from the original GA. Do not assume its implementation choices carry over automatically.

Current decisions: FP32 genotypes, a 32x32 toroidal population (1,024 organisms), and a shared slab with 1,792 slots.
Genotype width is fixed permanently: no action-space augmentation, resizing, repacking, or compaction. Keep CUDA simple.
Reclaim genotype slots by counting remaining uses; see `docs/genotype-slab.md` for the allocation/ownership contract.
Fitness plays all 2,109 training answers, using the full 2,309-solution candidate vocabulary and 4,739 actions.
Validation and final-test targets stay out of evolutionary fitness. See `docs/fitness.md` for scoring and GPU APIs.

## Version Control

There must be no merge commits. Use rebase pulls and fast-forward merges to keep history linear.

Work directly on `master` for now. Use a feature branch only when requested by the user. Commit and push completed
work unless told otherwise; multiple commits for a task are fine. When merging an existing branch, use
`git merge --ff-only`. Clean up old feature branches when appropriate after merging.

When both GitHub and GitLab remotes are configured, keep the corresponding branches in sync on both. If a rebase
requires updating an already-pushed feature branch, use `--force-with-lease`, never an unconditional force push.

For small commits, prefer this structure where sensible:

```text
To/Because/For [reason for change], [nature of change]
```

For larger commits, a descriptive subject and a body with bullet points are appropriate.

Never edit `.git/` contents manually. Use Git commands.

## Build, Test, and Hardware

Builds, formatting, and tests run in the provided Docker container. Run the Make wrappers from the host.

Prefer `make rebuild` for full build-and-test feedback: it formats the code, performs a clean rebuild, and runs all
tests, including GPU tests. `make agents-rebuild` is an alias for that same container-based workflow. Use
`make smoke` for a visible GPU smoke result and `make test-gpu-sanitized` for CUDA memory checking.

The target is CUDA compute capability **12.0** only. The desktop has an RTX 5070 Ti with 16 GB VRAM; the laptop has an
RTX 5050 with 8 GB VRAM, which may be reported as an RTX 5050 Laptop GPU. The project must not use the RTX 3060.

Select exactly one approved GPU by UUID using `NVIDIA_GPU_DEVICE_ID` in the ignored local `.env`. Docker Compose
exposes that device to the container. Keep machine-specific GPU and user/group settings out of tracked files.

GPU access is expected for development and testing. If the execution environment cannot provide it, report that
limitation; do not substitute a host implementation or claim that GPU tests passed.

## Coding Guidelines

Keep changes within the requested scope. Work incrementally; do not try to implement the entire project at once.

Use clear names and useful comments. Prefer idiomatic CUDA and straightforward project structure. Tests should
verify meaningful behavior. GPU code is tested on the GPU; do not maintain a parallel host implementation just for
testing or support machines without CUDA.

Host code may manage files, CUDA resources, kernel launches, and test comparisons. It must not duplicate inference,
gameplay, or fitness calculations as a CPU fallback or reference implementation. Use fixed reference outputs to
validate the CUDA implementation.

Wordle has five-letter words and six turns. These are fixed rules, not configuration options. Avoid flexibility and
abstractions for requirements we do not have.

## Systems Administration and Data

Do not perform systems administration on the host. Missing host compilers, dependency tools, broken PATH settings,
Docker GPU access, or driver problems should be reported to the user. Never modify host CUDA drivers. Installing
development dependencies inside the provided container is acceptable.

Do not modify `data/` or its contents without explicit user direction. A failing test alone does not authorize data
changes. Work within this repository; use the user-provided reference repositories for inspection or authorized
copying, without changing those projects.

## Working with the User

Be direct when a proposed design is a bad idea, and explain why. The user knows genetic algorithms well and benefits
from concrete guidance on neural-network and CUDA design. Keep responses concise, take work step by step, and leave
room for discussion before making decisions outside the requested scope.

The user prefers liberal use of Luna subagents to reduce project costs. Delegate bounded implementation, testing,
and review tasks to `gpt-5.6-luna`, with a clear specification and separate file ownership. The primary agent owns
integration and validation. Coordinate shared-worktree formatting/builds, and do not delegate commits or pushes.
