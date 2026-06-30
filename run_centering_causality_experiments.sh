#!/usr/bin/env bash
set -euo pipefail

# Generate centering-causality experiment job scripts, and optionally launch them.
# Default behavior is generate-only. Set RUN_JOBS=1 to start training.

script_path="${BASH_SOURCE[0]}"
if [[ "${script_path}" == */* ]]; then
  script_dir="$(cd "${script_path%/*}" && pwd)"
else
  script_dir="$(pwd)"
fi

repo_root="${REPO_ROOT:-${script_dir}}"
python_bin="${PYTHON_BIN:-python}"
target="${TARGET:-both}"
out_dir="${OUT_DIR:-${repo_root}/analysis/centering_causality_jobs}"
wandb_project="${WANDB_PROJECT:-Taiyi-centering-causality}"
seeds="${SEEDS:-0 1 2}"
max_jobs="${MAX_JOBS:-}"
run_jobs="${RUN_JOBS:-0}"

generator="${repo_root}/analysis/generate_centering_causality_jobs.py"
if [[ ! -f "${generator}" ]]; then
  echo "Cannot find generator: ${generator}" >&2
  exit 1
fi

cmd=(
  "${python_bin}"
  "${generator}"
  --target "${target}"
  --out "${out_dir}"
  --wandb-project "${wandb_project}"
  --seeds ${seeds}
)

if [[ -n "${max_jobs}" ]]; then
  cmd+=(--max-jobs "${max_jobs}")
fi

printf 'Generating jobs with command:\n  '
printf '%q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

cat <<EOF

Generated jobs under:
  ${out_dir}

Before launching on the server, set these as needed:
  export REPO_ROOT=/path/to/norm-exp
  export PYTHON_BIN=/path/to/python
  export DATASET_ROOT=/path/to/dataset
  export OUTPUT_ROOT=/path/to/output
  export WANDB_ENTITY=whsjrc-buaa
  export WANDB_PROJECT=${wandb_project}
  export CUDA_VISIBLE_DEVICES=0
  export NUM_ONCE=1

Launcher:
  cd "${out_dir}" && bash z_bash_execute.sh
EOF

if [[ "${run_jobs}" == "1" ]]; then
  echo "RUN_JOBS=1, launching generated jobs..."
  cd "${out_dir}"
  bash z_bash_execute.sh
else
  echo "Generate-only mode. Set RUN_JOBS=1 to launch after generation."
fi
