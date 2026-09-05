# Neuroevolutionary Wordle

A CUDA/C++ project to breed neural agents that play Wordle. This repository contains development scaffolding,
a GPU smoke test, and a minimal CUDA policy forward pass. The game engine and genetic algorithm are not implemented yet.

The development setup follows the earlier `neuroevolution-wordle` project. The policy architecture comes from
`wordle-ml_machine-learning`. Its trained weights and saved outputs are included as regression fixtures; using those
weights to seed evolution remains a separate decision. See [CUDA inference](docs/inference.md) for the API and scope.

## Setup

The host needs Docker with Compose and NVIDIA Container Toolkit configured for GPU containers. The development image
provides CUDA 13.1, CMake, Ninja, the C++ compiler, and clang-format. Builds target compute capability **12.0** only.

```sh
make env
nvidia-smi --query-gpu=name,uuid --format=csv,noheader
id -u
id -g
```

Edit the ignored `.env` file: set `NVIDIA_GPU_DEVICE_ID` to the **RTX 5070 Ti** UUID on the desktop or the **RTX 5050**
UUID on the laptop, and set `DOCKERCOMPOSE_UID` / `DOCKERCOMPOSE_GID` to your user/group IDs. Compose requires a device
selection and exposes only that GPU. The RTX 3060 must not be selected.

```sh
make rebuild
make smoke
```

The smoke test requires exactly one visible GPU, checks its model and compute capability, copies an integer to device
memory, increments it in a CUDA kernel, and checks the result after copying it back. GPU failures fail the test.

## Development commands

All build, formatting, and test commands run in the development container. Generated files live in `build/` on the
host and are owned by the configured user. Containers are removed when commands finish.

```sh
make configure           # Configure CMake/Ninja
make build               # Configure and compile
make test                # Build and run CTest, including the GPU smoke test
make test-gpu            # Alias for the current GPU-only test suite
make inference           # Run CUDA inference on the saved reference cases
make test-gpu-sanitized  # Check the GPU tests with compute-sanitizer
make format              # Apply .clang-format to C++/CUDA sources
make clean               # Remove build/
make rebuild             # Sequential clean, format, build, and test
make agents-rebuild      # Alias for the same container-based rebuild
make shell               # Open a development shell
```

From `make shell`, use CMake directly: `cmake --build build` and `ctest --test-dir build --output-on-failure`.
Run the Make wrappers from the host.

## Layout

- `docker/`, `docker-compose.yml`: development container and GPU selection.
- `CMakeLists.txt`, `Makefile`, `.clang-format`: build, test, and formatting commands.
- `src/model/`: the CUDA policy and its fixed parameter layout.
- `tests/cuda/`, `tests/fixtures/`: GPU tests and saved model/reference fixtures.
- `data/`: reserved for word data.
- `checkpoints/`, `models/`, `telemetry/`, `profiling/`: reserved output directories; generated contents are ignored.
- `docs/`: project documentation.
