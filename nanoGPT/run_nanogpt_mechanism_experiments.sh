#!/bin/bash
set -euo pipefail

dir_name=exp-nanogpt-mechanism

script_path="${BASH_SOURCE[0]}"
if [[ "${script_path}" == */* ]]; then
  script_dir="$(cd "${script_path%/*}" && pwd)"
else
  script_dir="$(pwd)"
fi
repo_root="$(cd "${script_dir}/.." && pwd)"
gen_dir="${script_dir}/${dir_name}"
mkdir -p "${gen_dir}"

cp "$0" "${gen_dir}/gen_script.sh"

arch=nanoGPT
dataset=tinyshakespeare
n_layer=6
n_head=6
n_embd=384
block_size=256
batchsize=64,64
activation=gelu
epochs=50
iters_per_epoch=100
eval_iters=50
display_every=1
optimizer=adamw
weightdecay=0.1
dropout=0.2
lr=6e-4
lr_method=cos
lrstep=30
lrgamma=0.1
dtype=bfloat16
sample_tokens=0
sample_every=0
subjectname="nanoGPT-SeqBN"

# Override these from the shell on a cluster, for example:
#   PYTHON_BIN=/path/to/python DATASET_ROOT=/path/to/tinyshakespeare bash nanoGPT/run_nanogpt_next_experiments.sh
dataset_root="${DATASET_ROOT:-${repo_root}/dataset/tinyshakespeare}"
output_root="${OUTPUT_ROOT:-${repo_root}/results/${dir_name}}"
python_bin="${PYTHON_BIN:-python}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"
num_once="${NUM_ONCE:-1}"

launch_cnt=0
: > "${gen_dir}/z_bash_execute.sh"

scheduler_extra_args=""
if [ "${lr_method}" = "step" ]; then
  scheduler_extra_args="
  --lr-step=${lrstep} \\
  --lr-gamma=${lrgamma} \\"
fi

norm_args_for() {
  local norm="$1"
  local slot="$2"

  case "${slot}" in
    baseline|all)
      printf '  --norm=%s \\\n' "${norm}"
      ;;
    attn)
      printf '  --attn-norm=%s \\\n  --mlp-norm=LN \\\n  --final-norm=LN \\\n' "${norm}"
      ;;
    mlp)
      printf '  --attn-norm=LN \\\n  --mlp-norm=%s \\\n  --final-norm=LN \\\n' "${norm}"
      ;;
    final)
      printf '  --attn-norm=LN \\\n  --mlp-norm=LN \\\n  --final-norm=%s \\\n' "${norm}"
      ;;
    *)
      echo "Unknown slot: ${slot}" >&2
      exit 1
      ;;
  esac
}

tag_for() {
  local norm="$1"
  local slot="$2"

  case "${slot}" in
    baseline)
      printf 'control%s' "${norm}"
      ;;
    all)
      printf 'all%s' "${norm}"
      ;;
    attn|mlp|final)
      printf '%s%s' "${slot}" "${norm}"
      ;;
    *)
      echo "Unknown slot: ${slot}" >&2
      exit 1
      ;;
  esac
}

generate_job() {
  local norm="$1"
  local slot="$2"
  local seed="$3"
  local phase="$4"

  local norm_tag
  norm_tag="$(tag_for "${norm}" "${slot}")"
  local base_string="execute_${phase}_${arch}_${dataset}_L${n_layer}_H${n_head}_D${n_embd}_ctx${block_size}_${norm_tag}_${activation}_lr${lr}_bs${batchsize}_drop${dropout}_wd${weightdecay}_s${seed}_${optimizer}"
  local file_name="${base_string}.sh"

  echo "Generating ${base_string}"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd "$(dirname "$0")/../.."\n'
    printf 'CUDA_VISIBLE_DEVICES=%s %s %s \\\n' "${cuda_visible_devices}" "${python_bin}" "${repo_root}/nanoGPT/nanogpt.py"
    printf '  --arch=%s \\\n' "${arch}"
    printf '  --data-dir=%s \\\n' "${dataset_root}"
    printf '  --no-auto-prepare \\\n'
    printf '  --n-layer=%s \\\n' "${n_layer}"
    printf '  --n-head=%s \\\n' "${n_head}"
    printf '  --n-embd=%s \\\n' "${n_embd}"
    printf '  --block-size=%s \\\n' "${block_size}"
    printf '  --batch-size=%s \\\n' "${batchsize}"
    printf '  --epochs=%s \\\n' "${epochs}"
    printf '  --iters-per-epoch=%s \\\n' "${iters_per_epoch}"
    printf '  --eval-iters=%s \\\n' "${eval_iters}"
    printf '  --lr=%s \\\n' "${lr}"
    printf '  --lr-method=%s \\\n' "${lr_method}"
    if [ -n "${scheduler_extra_args}" ]; then
      printf '%b\n' "${scheduler_extra_args}"
    fi
    printf '  --optimizer=%s \\\n' "${optimizer}"
    printf '  --weight-decay=%s \\\n' "${weightdecay}"
    printf '  --dropout=%s \\\n' "${dropout}"
    norm_args_for "${norm}" "${slot}"
    printf '  --activation=%s \\\n' "${activation}"
    printf '  --dtype=%s \\\n' "${dtype}"
    printf '  --seed=%s \\\n' "${seed}"
    printf '  --sample-tokens=%s \\\n' "${sample_tokens}"
    printf '  --sample-every=%s \\\n' "${sample_every}"
    printf '  --print-f=%s \\\n' "${display_every}"
    printf '  --output=%s \\\n' "${output_root}"
    printf '  --visualize \\\n'
    printf '  --wandb_project=%s \\\n' "${subjectname}"
    printf '  --no-save-checkpoint\n'
  } > "${gen_dir}/${file_name}"
  chmod +x "${gen_dir}/${file_name}"

  echo "nohup bash ${file_name} > output_${base_string}.out 2>&1 &" >> "${gen_dir}/z_bash_execute.sh"
  launch_cnt=$((launch_cnt + 1))
  if (( launch_cnt % num_once == 0 )); then
    echo "wait" >> "${gen_dir}/z_bash_execute.sh"
  fi
}

# Centering-only mechanism checks. These are new settings, so seed0 is
# enough for the first pass.
mechanism_seed=0
mechanism_specs=(
  "CSBNc attn"
  "CSBNc mlp"
  "CSBNc final"
  "CSBNc all"
  "CSeqBNc mlp"
  "CSeqBNc all"
  "CDSeqBNc mlp"
  "CDSeqBNc all"
)

for spec in "${mechanism_specs[@]}"; do
  read -r norm slot <<< "${spec}"
  generate_job "${norm}" "${slot}" "${mechanism_seed}" "mechanism"
done

if (( launch_cnt % num_once != 0 )); then
  echo "wait" >> "${gen_dir}/z_bash_execute.sh"
fi

chmod +x "${gen_dir}/z_bash_execute.sh"
echo "Generated ${launch_cnt} jobs under ${gen_dir}/"
echo "Run with: cd ${gen_dir} && bash z_bash_execute.sh"
