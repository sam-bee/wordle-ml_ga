# Fixed-width genotype slab

The allocator follows the remaining-parent-use counting described in
[Neuroevolutionary Wordle — Garbage Collection and the Older Generation](https://sam-burns.com/posts/neuroevolutionary-wordle-garbage-collection-and-the-older-generation/)
and the previous project's `src/genetic_algorithm/genotype_slab/`. This version has one permanent genotype width.
Freed slots are recycled directly; living genotypes stay at the same address.

## Memory and ownership

[`Slab`](../src/genotype_slab/slab.hpp) owns one CUDA allocation for genotypes and three small allocations for the
free-slot stack, reference counts, and free-slot count. `Create` initializes metadata and waits for initialization;
genotype bytes are uninitialized. The caller selects the CUDA device. The slab's capacity and stride stay fixed
until `Destroy`, which requires all consumers to have finished. Failed creation cleans up partial allocations.

| Item | Size |
| --- | ---: |
| FP32 genotype: 1,046,596 weights | 4,186,384 bytes |
| Slot stride, aligned to 256 bytes | 4,186,624 bytes |
| Default capacity | 1,792 slots |
| Default genotype allocation | 7,502,430,208 bytes (6.987 GiB) |
| Default device metadata | 14,340 bytes |

The default accommodates 1.75 times our 32x32 population of 1,024 organisms. Small capacities are available at
creation for GPU tests. There is no CUDA memory allocation, resizing, compaction, or host spill during generation
turnover.

The eventual generation/grid representation is an array of slot indices. Grid position belongs to that array;
physical slab position does not determine spatial neighbors.

## Device API

Pass `Slab::view()` by value to kernels and include [`device.cuh`](../src/genotype_slab/device.cuh):

- `Allocate(view)` returns a slot with one owning reference, or `kInvalidSlot` when full.
- `Retain(view, slot)` adds one planned use to a live slot. Invalid, freed, or saturated counts return `false`.
- `Release(view, slot)` consumes one reference. The unique transition from one to zero pushes the slot onto the
  free stack. Invalid or already-freed slots return `false`.
- `Parameters(view, slot)` addresses the slot's FP32 weights, directly compatible with the policy layout. An index
  outside the slab returns `nullptr`. The caller must own a live reference before accessing the payload.

An allocated slot's previous contents are not cleared. Breeding or initialization must write the genotype before
inference reads it. A slot index is valid for that allocation's lifetime only: after its final release, do not use it
again unless allocation returns it anew. There are deliberately no versioned handles or stale-reference recovery.

## Phases keep concurrency simple

Many threads may allocate concurrently. Many threads may retain concurrently, including the same parent. Many threads
may release concurrently; the final reference returns its slot exactly once. These are separate phases:

**Do not overlap allocation, retain, and release phases with one another.** Use kernel boundaries on the same stream,
or explicit CUDA event dependencies between streams. In particular, the stack push publishes its entry after
reserving an index, so an allocating kernel must wait for the release phase to finish. No spinlock, blocking wait for
memory, or concurrent push/pop algorithm is needed under this contract.

Perform one reference operation per owned/planned use, not per weight-processing thread. Parent reads must finish
before their reference is released. A separate release kernel after the breeding kernel gives a simple completion
boundary even when many blocks participate in producing each child.

## Intended breeding sequence

1. Each current organism initially owns one reference to its slot.
2. Once the parent-pair plan is known, retain each parent once for each planned occurrence in that plan. If one parent
   appears in both positions, both uses need references.
3. Release the current generation's ownership references. Childless parents reach zero and free their slots;
   other parents remain alive for their planned uses. Any other pending consumer must also hold a reference.
4. Allocate child slots for a bounded breeding batch. Each child starts with its own ownership reference.
5. Produce those children on the GPU, then release their completed parent uses.
6. Repeat allocation, breeding, and release phases. Parents disappear after their last use; the new generation
   retains its ownership references.

Only the allocator and lifetime primitives are implemented here; parent selection, recombination, mutation, and
batch scheduling are subsequent work. The test suite demonstrates this ownership sequence with tiny GPU payload
copies, without implementing a breeder or CPU allocator.

The 1.75x capacity is a budget, not a guarantee for arbitrary parent plans/orderings. If the caller exhausts it,
allocation returns `kInvalidSlot` without modifying the live genotypes or reducing the population. The future
breeding scheduler must handle that result explicitly and stay within the live-slot budget.

## Verification

`make test` runs the allocator tests on the selected GPU using small slabs. They exercise concurrent exhaustion,
reference updates, exactly-once reclamation, invalid operations, reuse, payload isolation, and a planned parent/child
lifetime. Host code manages CUDA resources and checks results; allocation and reference-count operations run in CUDA.

`make test-gpu-sanitized` includes allocator memcheck. `make slab-smoke` separately creates the full default 1,792-slot
slab, verifies its metadata, and touches its first and last slots without initializing all 7 GiB.

Initial validation on the RTX 5070 Ti passed all three GPU tests, the full-size slab smoke test, and the suite's
Compute Sanitizer memcheck runs. A separate Compute Sanitizer initcheck run on `genotype_slab_test` also reported
zero errors.
