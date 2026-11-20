# Testing the #1815 fix on another GPU

This branch (`bench/issue-1815`) is `feat/oxide-auto-unroll-local-arrays` plus
this benchmark example. The benchmark is not part of the PR.

Run everything from `cuda-oxide/`, on an idle GPU. On a host with several GPUs,
pick one per run with `CUDA_VISIBLE_DEVICES=<n>` (the examples use CUDA
device 0).

```bash
git fetch origin
git checkout bench/issue-1815
cd cuda-oxide
```

## 1. Benchmark (the main result)

```bash
crates/rustc-codegen-cuda/examples/issue_1815_bench/run-compare.sh
```

It builds and runs the benchmark twice: with the rule off
(`CUDA_OXIDE_NO_AUTO_UNROLL=1`, which behaves like `main`) and with it on. It
prints each kernel's time and `.local` depot size.

Expected:

- With the rule off, `mma_plain` has a 128 B depot and is much slower than
  `mma_handunrolled`.
- With the rule on, both kernels have a 0 B depot and similar times.
- Each run prints `outputs identical: yes`.

On sm_121 (GB10), `mma_plain` took 243.8 ms with the rule off and 11.0 ms with
it on; `mma_handunrolled` took 11.05 ms.

## 2. Regression example

```bash
cargo oxide run unroll_smoke
cargo oxide build unroll_smoke 2>&1 | tee /tmp/unroll_smoke.log
CUDA_OXIDE_BUILD_LOG=/tmp/unroll_smoke.log \
  bash crates/rustc-codegen-cuda/examples/unroll_smoke/verify-code-shape.sh
```

Expected: `unroll_smoke: PASS` and `unroll_smoke PTX shape: PASS`.

## 3. Full smoketest

```bash
scripts/smoketest.sh --no-color 2>&1 | tee /tmp/smoketest.txt
```

Record the pass count and the names of any failing examples. To check whether
a failure is caused by this change, rerun that example with the rule off
(touch its source first, because the build cache does not track the variable):

```bash
touch crates/rustc-codegen-cuda/examples/<name>/src/main.rs
CUDA_OXIDE_NO_AUTO_UNROLL=1 cargo oxide run <name>
```

## What to report

- The architecture (`sm_XXX`, printed as `Detected GPU arch`) for each GPU used.
- The output of step 1 for each GPU.
- Step 2: PASS or FAIL.
- Step 3: the pass count, plus any failure that also passes with the rule off.

## Discrete sm_120 validation (2026-11-19–20 UTC)

Tested benchmark commit `3f3d25ac1`, which contains feature commit `693d159`
on upstream `8046b9e9d`. Nothing in the compiler, the example or its
expectations was changed for the run. This is a discrete-GPU (`sm_120`)
result, separate from the GB10 result above.

### Benchmark and regression example

| Automatic rule | `mma_plain` | `mma_handunrolled` |
|---|---:|---:|
| Off | 34.613 ms | 4.492 ms |
| On | 4.427 ms | 4.437 ms |

- The plain loop is 7.8× faster (34.613 / 4.427), about the same as the
  hand-unrolled reference.
- Both modes printed `outputs identical: yes` and `SUCCESS`.
- In the `sm_120` PTX, the plain loop's local depot is 128 B with the rule off
  and 0 B with it on. The hand-unrolled kernel has a 0 B depot, and its PTX is
  the same in both modes.
- The workload was unchanged: three warmups, then five samples of ten launches.
  Each time is the best of the five ten-launch averages, not a median.

`unroll_smoke` passed its runtime check and `verify-code-shape.sh`, including
`unannotated_range_accumulator: no local memory PASS`.

### CPU test suites

| Suite | Passed | Ignored |
|---|---:|---:|
| `mir-transforms` | 112 | 0 |
| `cuda-oxide-codegen` | 1366 | 13 |
| Codegen backend library | 42 | 0 |
| `cargo-oxide` | 301 | 0 |
| `cuda-host` / `cuda-macros` | 271 | 5 |
| **Total** | **2092** | **18** |

There were no failures. These are selected suites, not full CI.

A compile-only sweep of the examples gave 258 expected verdicts: 239 builds and
19 expected compile rejections.

### Smoketest coverage

The coverage below comes from several runs, not one clean full sweep:

| Runtime PASS | Compile-only | Skip | FAIL | Not run |
|---:|---:|---:|---:|---:|
| 229 | 25 | 2 | 0 | 1 |

- The two skips are optional: MathDx and multi-GPU NVLS.
- `wgmma_mma_bf16` was not run on the GPU, because its source forbids launching
  its kernels. Its `sm_90a` compilation was verified.
- No compiler failure was observed.

### Notes for the compiler PR

- Open the PR from `feat/oxide-auto-unroll-local-arrays` (tested commit
  `693d159`). Do not merge this benchmark branch; it is a testing aid.
- Report the discrete `sm_120` result separately from the GB10 measurements,
  with the timing method and coverage limits above. Do not claim a clean full
  smoketest or a fully green CI run.
- These measurements apply to the exact commits above. A substantive change to
  the compiler patch needs retesting.
