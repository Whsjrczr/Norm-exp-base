#!/usr/bin/env bash
set -euo pipefail

# Run this file from nanoGPT/test to generate nanoGPT Taiyi mechanism jobs
# under nanoGPT/test/exp-nanogpt-taiyi-norm-mechanism by default.

script_path="${BASH_SOURCE[0]}"
if [[ "${script_path}" == */* ]]; then
  script_dir="$(cd "${script_path%/*}" && pwd)"
else
  script_dir="$(pwd)"
fi

repo_root="$(cd "${script_dir}/../.." && pwd)"
dir_name="${DIR_NAME:-exp-nanogpt-taiyi-norm-mechanism}"

export REPO_ROOT="${REPO_ROOT:-${repo_root}}"
export GEN_DIR="${GEN_DIR:-${script_dir}/${dir_name}}"

bash "${repo_root}/run_nanogpt_taiyi_norm_mechanism_experiments.sh" "$@"
