# TP8 sampling benchmark and reference integration

The selected implementation is `naivert/body/dspark_sampling/sharded.cuh`.
`bench_sharded.cu` compares the original sampler, the replicated radix reference,
and the sharded sampler across all eight GPUs. Each comparison includes its
communication. Inputs and model forward passes are outside the timed interval.

The standard run checks 88 cases, all eight ranks' tokens, and 26,853,376 p/q
values. It then alternates three CUDA graphs in forward/reverse order. Each graph
contains 16 complete invocations; the reported latency is the largest rank's
CUDA event duration divided by 16. This reduces host rank-launch skew without
removing the sampler's own synchronization. Every timed replay's tokens and
watchdog are checked. Graphs are instantiated with device-launch support.

Build from the checkout root on the eight-GPU host:

```sh
mkdir -p /tmp/naivert-bench
$CUDA_HOME/bin/nvcc -O3 -std=c++17 --expt-relaxed-constexpr -lineinfo \
  -ccbin /usr/bin/g++ -Xcompiler=-B/usr/bin \
  -gencode=arch=compute_103a,code=sm_103a -I. \
  tests/gpu/naive_n0p5_draft/sampling/bench_sharded.cu -o /tmp/naivert-bench/sharded_bench
```

Run with all eight GPUs visible:

```sh
/tmp/naivert-bench/sharded_bench /path/to/inputs 20 16
```

Arguments after the input directory are timing repetitions and invocations per
graph. An optional fifth argument shortens correctness to two real-input cases
for sanitizer use; it is not the full gate. Do not use instrumented timing as
performance evidence.

Use these environment settings for every Python test:

```sh
export CUDA_HOME=/path/to/cuda-13.2 NAIVERT_CUDA_HOME=/path/to/cuda-13.2
export CXX=/usr/bin/g++ CC=/usr/bin/gcc
export NAIVERT_CUTLASS_INCLUDE=/path/to/cutlass/include
```

## Replicated reference

The reference implementation is `naivert/body/dspark_sampling/parallel.cuh`. This harness
keeps the original `sample_target` body in `baseline.cuh`, and compares it with the
replacement and an independent CPU sort/CDF oracle.

`bench.cu INPUT_DIRECTORY [REPEATS]` performs 224 correctness cases, then times
alternating old/new CUDA graphs. The input directory contains `target.bin`
(8x152576 FP32), `draft.bin` (7x152576 FP32 corrected logits), and `proposals.bin`
(7 INT64). Each file is checked for exact size. There is no model load. Copies
restoring the logits are outside the timed interval. An optional fourth argument
skips correctness for profiler runs; output explicitly reports zero validation
cases in that mode. Do not use instrumented timings as latency results.

Recorded inputs, raw tensors, CUDA binaries and profiler reports are kept
outside Git.

Build from the checkout root:

```sh
mkdir -p /tmp/naivert-bench
$CUDA_HOME/bin/nvcc -O3 -std=c++17 --expt-relaxed-constexpr -lineinfo \
  -ccbin /usr/bin/g++ -Xcompiler=-B/usr/bin \
  -gencode=arch=compute_103a,code=sm_103a -I. \
  tests/gpu/naive_n0p5_draft/sampling/bench.cu -o /tmp/naivert-bench/sampler_bench
```

Run with all eight GPUs visible, including sanitizer runs:

```sh
/tmp/naivert-bench/sampler_bench /path/to/inputs 100
```

`profile_stages.cu` inserts external event nodes between the 16 kernels. Those
numbers include instrumentation overhead and are useful for locating costs only.
The first profile identified histogram work on inactive values and repeated FP64
division. The final benchmark measures the uninstrumented whole graph.

## Profiling and exact histogram checks

`profile_sharded.cu` instruments the captured 33-node sampler graph with external
CUDA events. It asserts the graph is a single kernel chain and checks tokens.
This is a diagnostic screen with substantial event overhead, not a latency
benchmark. Build like `bench_sharded.cu`.

`check_histogram.cu` compares each of four histogram passes with direct CPU FP64
summation over six input families, including subnormals, zeros, ties, exponent
boundaries and concentrated buckets. Its standalone kernels have no peer waits,
so memory/race instrumentation can finish without a distributed rendezvous.

`bench_integer_hist.cu` pairs the complete sampler with a frozen sharded
baseline. For the CDF-fusion comparison, place the pre-fusion `sharded.cuh` as
`sharded_fp64_baseline.cuh` in a directory outside the working tree and build as
above with the additional include paths `-I/path/to/reference-headers` and
`-Inaivert/body/dspark_sampling`. It alternates old/new and new/old, 40
observations with 16 complete invocations per device-compatible graph, and
checks every rank's output and watchdog. It accepts the same input-directory
argument as the other benchmarks.

`check_sharded_sanitizer.cu` captures complete sampler graphs before launching
one host thread per rank. This avoids serial eager host submission before peer
rendezvous under instrumentation. It checks both shapes, repeat invocation, rank
agreement, watchdogs and a CPU probability oracle. Build with the additional
`-Xcompiler=-pthread` option. Record the actual sanitizer outcome; successful
standalone histogram checks do not by themselves validate the full TP8 protocol.

## Public persistent binding and deferred tail

`check_persistent_binding.py` uses the actual `TargetSampler` and `DraftTail`
Python bindings from this tree. It compares the opt-in token-exact persistent
sampler against the original chain on captured real, uniform and random logits,
two RNG seeds, anchor/verify and two consecutive replays. Verify defers consumed
to the matching no-ready tail; external epochs remain live across cases.
Target forward, recursive launch and output publication are outside this test.

Run from the checkout root:

```sh
python3 tests/gpu/naive_n0p5_draft/sampling/check_persistent_binding.py \
  --repo . \
  --inputs /path/to/inputs \
  --out /tmp/naivert-bench/binding.json
```

For the focused memory gate, prepend
`$CUDA_HOME/bin/compute-sanitizer --tool memcheck --leak-check full
--error-exitcode 99` and append `--focused`, using a separate output path.
Focused mode retains real and uniform inputs, one seed and both anchor/verify
shapes, including repeated epochs. The test captures every rank before launching
from eight host threads and explicitly releases allocator caches before exit.
