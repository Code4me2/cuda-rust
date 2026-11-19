#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build and run issue_1815_bench twice: with the automatic unroll rule turned
# off (CUDA_OXIDE_NO_AUTO_UNROLL=1, the behaviour of main) and with it on.
# Prints, per build, each kernel's .local depot size and its timing.
# Run on an otherwise idle GPU. Extra arguments go to `cargo oxide run`
# (for example `--arch sm_120`).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here/../../../.."

for mode in off on; do
  echo "=== auto-unroll $mode ==="
  # The backend reads the variable at compile time, so force a rebuild.
  touch "$here/src/main.rs"
  if [[ $mode == off ]]; then
    CUDA_OXIDE_NO_AUTO_UNROLL=1 cargo oxide run issue_1815_bench "$@" 2>&1 | grep -E '^(mma_|outputs|SUCCESS|Detected GPU|error)|panicked'
  else
    env -u CUDA_OXIDE_NO_AUTO_UNROLL cargo oxide run issue_1815_bench "$@" 2>&1 | grep -E '^(mma_|outputs|SUCCESS|Detected GPU|error)|panicked'
  fi
  python3 -I - "$here/issue_1815_bench.ptx" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
print("target", re.search(r"\.target\s+(\S+)", s).group(1))
for name in ("mma_plain", "mma_handunrolled"):
    body = re.search(r"\.entry\s+" + name + r"\b.*?\n\}\n", s, re.S).group(0)
    depot = re.search(r"__local_depot\w*\[(\d+)\]", body)
    print(f"  {name:18} .local depot {depot.group(1) if depot else 0} B")
PY
done
