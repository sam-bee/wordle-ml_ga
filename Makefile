.DEFAULT_GOAL := help

.PHONY: help env docker-build configure build test test-gpu test-gpu-sanitized smoke inference \
	format clean rebuild agents-rebuild build-and-test shell

CUDA_RUN := docker compose run --rm --no-deps -T cuda-dev

help:
	@echo "Wordle GA development commands (builds and tests run in Docker):"
	@echo "  make env                 Create the local .env configuration"
	@echo "  make docker-build        Build the CUDA development image"
	@echo "  make configure           Configure CMake with Ninja for sm_120"
	@echo "  make build               Configure and compile"
	@echo "  make test                Build and run the GPU tests with CTest"
	@echo "  make smoke               Build and print the GPU smoke result"
	@echo "  make inference           Run CUDA policy inference against saved reference cases"
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

smoke: build
	$(CUDA_RUN) ./build/gpu_smoke_test

inference: build
	$(CUDA_RUN) ./build/policy_test

format: docker-build
	$(CUDA_RUN) sh -c 'find src tests -type f \( -name "*.hpp" -o -name "*.cpp" -o -name "*.cu" \) -exec clang-format -i {} +'

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
