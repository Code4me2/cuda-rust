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
