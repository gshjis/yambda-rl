#!/usr/bin/env bash
# Push one pipeline stage as a Kaggle kernel, poll it, download its small artifacts.
# Usage: scripts/kaggle_run.sh <stage> [extra run.py args ...]
set -euo pipefail

STAGE=${1:?usage: kaggle_run.sh <stage> [extra run.py args]}
shift || true

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SHA=$(git -C "$ROOT" rev-parse HEAD)
SLUG="gshjis/yambda-${STAGE//_/-}"
OUT="$ROOT/results/kaggle/$STAGE"

case "$STAGE" in
  etl|eda|baseline|analyze) GPU=false ;;
  *) GPU=true ;;
esac

DIR=$(mktemp -d)
python3 - "$DIR" "$STAGE" "$SHA" "$SLUG" "$GPU" "$@" <<'PY'
import json, sys
d, stage, sha, slug, gpu, *extra = sys.argv[1:]
meta = {
    "id": slug,
    "title": f"yambda {stage}",
    "code_file": "kernel.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": gpu == "true",
    "enable_internet": True,
    "dataset_sources": ["dinaryakupov/yambda-50m-compressed"],
    "competition_sources": [],
    "kernel_sources": [],
}
open(f"{d}/kernel-metadata.json", "w").write(json.dumps(meta, indent=2))
code = f'''import os, subprocess, sys
REPO = "https://github.com/gshjis/yambda-rl.git"
SHA = "{sha}"
subprocess.run(["git", "clone", "--quiet", REPO, "/kaggle/working/repo"], check=True)
subprocess.run(["git", "-C", "/kaggle/working/repo", "checkout", "--quiet", SHA], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "duckdb"], check=True)
cmd = [sys.executable, "/kaggle/working/repo/run.py", "--stage", "{stage}"] + {extra!r}
raise SystemExit(subprocess.call(cmd))
'''
open(f"{d}/kernel.py", "w").write(code)
PY

echo "pushing $SLUG at $SHA"
kaggle kernels push -p "$DIR" >/dev/null
for _ in $(seq 1 480); do
  status=$(kaggle kernels status "$SLUG" 2>/dev/null | tail -1)
  case "$status" in
    *COMPLETE*|*ERROR*|*CANCEL*|*FAILED*) break ;;
  esac
  sleep 30
done
echo "$status"
mkdir -p "$OUT"
kaggle kernels output "$SLUG" -p "$OUT" 2>&1 | tail -2 || true
