.DEFAULT_GOAL := help

.PHONY: help env docker-build configure build test test-gpu test-gpu-sanitized smoke inference slab-smoke fitness fitness-benchmark \
	profile-fitness-systems profile-fitness-compute format clean rebuild agents-rebuild build-and-test shell

CUDA_RUN := docker compose run --rm --no-deps -T cuda-dev
PROFILE_HOST_UID := $(shell id -u)
PROFILE_HOST_GID := $(shell id -g)
PROFILE_SYSTEMS_DIR := profiling/nsight-systems
PROFILE_COMPUTE_DIR := profiling/nsight-compute
FITNESS_POPULATION ?= 16
FITNESS_REPETITIONS ?= 5

# Mount the complete host Nsight Systems version directory. The short-lived
# container gets only a /tmp symlink to the mounted target binary.
NSYS_HOST_DIR ?= $(shell nsys_binary=$$(readlink -f "$$(command -v nsys)" 2>/dev/null); test -n "$$nsys_binary" && dirname "$$(dirname "$$nsys_binary")")
NSYS_CONTAINER_ROOT := /opt/wordle-ga-nsys
NSYS_CONTAINER_BINARY := /tmp/wordle-ga-nsys
NSYS_RUN := docker compose run --rm --no-deps -T \
	-v "$(NSYS_HOST_DIR):$(NSYS_CONTAINER_ROOT):ro" cuda-dev
NCU_RUN := docker compose run --rm --no-deps -T --user root --cap-add SYS_ADMIN \
	-e WORDLE_GA_PROFILE_UID=$(PROFILE_HOST_UID) -e WORDLE_GA_PROFILE_GID=$(PROFILE_HOST_GID) cuda-dev

help:
	@echo "Wordle GA development commands (builds and tests run in Docker):"
	@echo "  make env                 Create the local .env configuration"
	@echo "  make docker-build        Build the CUDA development image"
	@echo "  make configure           Configure CMake with Ninja for sm_120"
	@echo "  make build               Configure and compile"
	@echo "  make test                Build and run the GPU tests with CTest"
	@echo "  make smoke               Build and print the GPU smoke result"
	@echo "  make inference           Run CUDA policy inference against saved reference cases"
	@echo "  make slab-smoke          Allocate and check the default 1,792-slot genotype slab"
	@echo "  make fitness             Evaluate the saved model on all 2,109 training answers"
	@echo "  make fitness-benchmark   Repeat fitness with distinct genotype slots (default population 16)"
	@echo "  make profile-fitness-systems  Capture fitness with Nsight Systems"
	@echo "  make profile-fitness-compute  Capture PolicyLogits with Nsight Compute"
	@echo "  make test-gpu-sanitized   Run the GPU tests under compute-sanitizer"
	@echo "  make format              Format C++ and CUDA sources"
	@echo "  make clean               Remove the generated build directory"
	@echo "  make rebuild             Clean, format, build, and test"
	@echo "  make shell               Open a shell in the development container"

env: .env

.env:
	cp .env.example .env
	@echo "Created .env. Set NVIDIA_GPU_DEVICE_ID and check the user/group IDs before building."

docker-build: .env
	docker compose build cuda-dev

configure: docker-build
	$(CUDA_RUN) cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

build: configure
	$(CUDA_RUN) cmake --build build

test: build
	$(CUDA_RUN) ctest --test-dir build --output-on-failure

test-gpu: test

test-gpu-sanitized: build
	$(CUDA_RUN) compute-sanitizer --tool memcheck --error-exitcode=1 ./build/gpu_smoke_test
	$(CUDA_RUN) compute-sanitizer --tool memcheck --error-exitcode=1 ./build/policy_test
	$(CUDA_RUN) compute-sanitizer --tool memcheck --error-exitcode=1 ./build/genotype_slab_test
	$(CUDA_RUN) compute-sanitizer --tool memcheck --error-exitcode=1 ./build/policy_batch_test
	$(CUDA_RUN) compute-sanitizer --tool memcheck --error-exitcode=1 ./build/fitness_test
	$(CUDA_RUN) compute-sanitizer --tool memcheck --error-exitcode=1 ./build/fitness_test --full-training
	$(CUDA_RUN) compute-sanitizer --tool memcheck --error-exitcode=1 ./build/fitness_game_test
	$(CUDA_RUN) compute-sanitizer --tool memcheck --error-exitcode=1 ./build/fitness_runtime_test

smoke: build
	$(CUDA_RUN) ./build/gpu_smoke_test

inference: build
	$(CUDA_RUN) ./build/policy_test

slab-smoke: build
	$(CUDA_RUN) ./build/genotype_slab_test --full-size-smoke

fitness: build
	$(CUDA_RUN) ./build/fitness_test --full-training

fitness-benchmark: build
	$(CUDA_RUN) ./build/fitness_test --benchmark $(FITNESS_POPULATION) $(FITNESS_REPETITIONS)

profile-fitness-systems: build
	test -x "$(NSYS_HOST_DIR)/target-linux-x64/nsys" || { echo "host nsys target not found; install Nsight Systems or set NSYS_HOST_DIR" >&2; exit 127; }
	mkdir -p $(PROFILE_SYSTEMS_DIR)
	$(NSYS_RUN) sh -c \
		'ln -sf $(NSYS_CONTAINER_ROOT)/target-linux-x64/nsys $(NSYS_CONTAINER_BINARY) && \
		$(NSYS_CONTAINER_BINARY) profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
			--force-overwrite=true -o /workspace/$(PROFILE_SYSTEMS_DIR)/fitness \
			/workspace/build/fitness_test --full-training && \
		$(NSYS_CONTAINER_BINARY) stats --force-export=true --force-overwrite=true --format=csv \
			--output=/workspace/$(PROFILE_SYSTEMS_DIR)/fitness-summary \
			--report=cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_trace \
			/workspace/$(PROFILE_SYSTEMS_DIR)/fitness.nsys-rep'

profile-fitness-compute: build
	mkdir -p $(PROFILE_COMPUTE_DIR)
	$(NCU_RUN) sh -c \
		'ncu --set full --kernel-name regex:PolicyLogits --launch-skip 2 --launch-count 1 \
			--force-overwrite --export /workspace/$(PROFILE_COMPUTE_DIR)/policy-logits \
			/workspace/build/fitness_test --full-training; \
		status=$$?; \
		find /workspace/$(PROFILE_COMPUTE_DIR) -maxdepth 1 -type f \
			-exec chown "$$WORDLE_GA_PROFILE_UID:$$WORDLE_GA_PROFILE_GID" {} +; \
		exit $$status'
	$(CUDA_RUN) sh -c \
		'ncu --import /workspace/$(PROFILE_COMPUTE_DIR)/policy-logits.ncu-rep --page details --csv \
			> /workspace/$(PROFILE_COMPUTE_DIR)/policy-logits-summary.csv && \
		ncu --import /workspace/$(PROFILE_COMPUTE_DIR)/policy-logits.ncu-rep --page details \
			> /workspace/$(PROFILE_COMPUTE_DIR)/policy-logits-summary.txt'

format: docker-build
	$(CUDA_RUN) sh -c 'find src tests -type f \( -name "*.hpp" -o -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" \) -exec clang-format -i {} +'

clean:
	rm -rf build

# Keep these steps sequential, including when invoked with make -j.
rebuild:
	$(MAKE) clean
	$(MAKE) format
	$(MAKE) test

agents-rebuild: rebuild

build-and-test: test

shell: docker-build
	docker compose run --rm --no-deps cuda-dev zsh
