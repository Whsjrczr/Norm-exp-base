#!/usr/bin/env bash
set -euo pipefail

# Generate comparable norm experiment jobs for nanoGPT, BERT translation, and ViT.
# Main comparison: Sequence Centering + RMSNorm (DSeqBCRMS) vs LN vs BN.
# nanoGPT additionally includes CDSeqBCRMS, the causal-prefix version of the same idea.

script_path="${BASH_SOURCE[0]}"
if [[ "${script_path}" == */* ]]; then
  script_dir="$(cd "${script_path%/*}" && pwd)"
else
  script_dir="$(pwd)"
fi

if [[ -d "${script_dir}/nanoGPT" && -d "${script_dir}/BERT" && -d "${script_dir}/ViT" ]]; then
  repo_root="${script_dir}"
else
  repo_root="$(cd "${script_dir}/.." && pwd)"
fi

dir_name="${DIR_NAME:-exp-transformer-norm-comparison}"
gen_dir="${repo_root}/${dir_name}"
mkdir -p "${gen_dir}"
cp "$0" "${gen_dir}/gen_script.sh"
: > "${gen_dir}/z_bash_execute.sh"

target="${TARGET:-all}"
seeds_value="${SEEDS:-0 1 2}"
seeds_value="${seeds_value//,/ }"
read -r -a seeds <<< "${seeds_value}"
max_jobs="${MAX_JOBS:-}"
num_once="${NUM_ONCE:-1}"
python_bin="${PYTHON_BIN:-python}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"
output_root="${OUTPUT_ROOT:-${repo_root}/results/${dir_name}}"
wandb_project="${WANDB_PROJECT:-transformer-norm-comparison}"
diagnostics="${DIAGNOSTICS:-1}"
visualize="${VISUALIZE:-1}"
no_save_checkpoint="${NO_SAVE_CHECKPOINT:-1}"

norms_value="${NORMS:-LN BN DSeqBCRMS}"
norms_value="${norms_value//,/ }"
read -r -a norms <<< "${norms_value}"
include_nanogpt_causal_sc="${INCLUDE_NANOGPT_CAUSAL_SC:-1}"

# nanoGPT defaults
nanogpt_arch="${NANOGPT_ARCH:-nanoGPT}"
nanogpt_data_dir="${NANOGPT_DATA_DIR:-${repo_root}/dataset/tinyshakespeare}"
nanogpt_n_layer="${NANOGPT_N_LAYER:-6}"
nanogpt_n_head="${NANOGPT_N_HEAD:-6}"
nanogpt_n_embd="${NANOGPT_N_EMBD:-384}"
nanogpt_block_size="${NANOGPT_BLOCK_SIZE:-256}"
nanogpt_batch_size="${NANOGPT_BATCH_SIZE:-64,64}"
nanogpt_epochs="${NANOGPT_EPOCHS:-20}"
nanogpt_iters="${NANOGPT_ITERS_PER_EPOCH:-50}"
nanogpt_eval_iters="${NANOGPT_EVAL_ITERS:-20}"
nanogpt_lr="${NANOGPT_LR:-6e-4}"
nanogpt_weight_decay="${NANOGPT_WEIGHT_DECAY:-0.1}"
nanogpt_dropout="${NANOGPT_DROPOUT:-0.2}"
nanogpt_dtype="${NANOGPT_DTYPE:-bfloat16}"
nanogpt_sample_tokens="${NANOGPT_SAMPLE_TOKENS:-300}"
nanogpt_sample_every="${NANOGPT_SAMPLE_EVERY:-1}"

# BERT translation defaults
bert_data_dir="${BERT_DATA_DIR:-${repo_root}/dataset/text_translation}"
bert_input_file="${BERT_INPUT_FILE:-}"
bert_layers="${BERT_LAYERS:-4}"
bert_heads="${BERT_HEADS:-4}"
bert_embd="${BERT_EMBD:-256}"
bert_ffn_mult="${BERT_FFN_MULT:-4}"
bert_max_src_len="${BERT_MAX_SRC_LEN:-64}"
bert_max_tgt_len="${BERT_MAX_TGT_LEN:-64}"
bert_batch_size="${BERT_BATCH_SIZE:-64,64}"
bert_epochs="${BERT_EPOCHS:-20}"
bert_iters="${BERT_ITERS_PER_EPOCH:-50}"
bert_eval_iters="${BERT_EVAL_ITERS:-20}"
bert_lr="${BERT_LR:-3e-4}"
bert_weight_decay="${BERT_WEIGHT_DECAY:-0.01}"
bert_dropout="${BERT_DROPOUT:-0.1}"
bert_dtype="${BERT_DTYPE:-bfloat16}"
bert_sample_src="${BERT_SAMPLE_SRC:-hello}"
bert_sample_every="${BERT_SAMPLE_EVERY:-1}"

# ViT defaults
vit_arch="${VIT_ARCH:-vit_small}"
vit_dataset="${VIT_DATASET:-cifar10}"
vit_dataset_root="${VIT_DATASET_ROOT:-${repo_root}/dataset}"
vit_image_size="${VIT_IMAGE_SIZE:-32}"
vit_patch_sizes_value="${VIT_PATCH_SIZES:-4 16}"
vit_patch_sizes_value="${vit_patch_sizes_value//,/ }"
read -r -a vit_patch_sizes <<< "${vit_patch_sizes_value}"
vit_batch_size="${VIT_BATCH_SIZE:-256}"
vit_epochs="${VIT_EPOCHS:-80}"
vit_lr="${VIT_LR:-1e-4}"
vit_optimizer="${VIT_OPTIMIZER:-adamw}"
vit_weight_decay="${VIT_WEIGHT_DECAY:-0.1}"
vit_dropout="${VIT_DROPOUT:-0.0}"
vit_drop_path="${VIT_DROP_PATH_RATE:-0.1}"
vit_activation="${VIT_ACTIVATION:-gelu}"

launch_cnt=0

sanitize() {
  local value="$1"
  value="${value//+/p}"
  value="${value//,/x}"
  value="${value//./p}"
  value="${value//=/}"
  value="${value//\//-}"
  value="${value// /}"
  printf '%s' "${value}"
}

should_generate_job() {
  if [[ -n "${max_jobs}" ]] && (( launch_cnt >= max_jobs )); then
    return 1
  fi
  return 0
}

append_launcher() {
  local file_name="$1"
  local base_string="$2"
  echo "nohup bash ${file_name} > output_${base_string}.out 2>&1 &" >> "${gen_dir}/z_bash_execute.sh"
  launch_cnt=$((launch_cnt + 1))
  if (( launch_cnt % num_once == 0 )); then
    echo "wait" >> "${gen_dir}/z_bash_execute.sh"
  fi
}

common_tail_args() {
  if [[ "${diagnostics}" == "1" ]]; then
    printf '  --diagnostics \\\n'
  fi
  if [[ "${visualize}" == "1" ]]; then
    printf '  --visualize \\\n'
  fi
  if [[ "${no_save_checkpoint}" == "1" ]]; then
    printf '  --no-save-checkpoint\n'
  else
    printf '  --save-checkpoint\n'
  fi
}

needs_noncausal_nanogpt_flag() {
  local norm="$1"
  case "${norm}" in
    BN|BNc|BNs|SeqBN|SeqBNc|SeqBNs|DSeqBN|DSeqBNc|DSeqBNs|DSeqBCLN|DSeqBCRMS|DSeqBLS|DSeqBCDS)
      return 0
      ;;
  esac
  return 1
}

generate_nanogpt_job() {
  should_generate_job || return 0
  local norm="$1"
  local seed="$2"
  local base_string="execute_nanogpt_${norm}_ctx${nanogpt_block_size}_lr$(sanitize "${nanogpt_lr}")_s${seed}"
  local file_name="${base_string}.sh"
  echo "Generating ${base_string}"
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\ncd "%s"\n' "${repo_root}"
    printf 'export WANDB_PROJECT=%s\n' "${wandb_project}"
    printf 'CUDA_VISIBLE_DEVICES=%s %s %s \\\n' "${cuda_visible_devices}" "${python_bin}" "${repo_root}/nanoGPT/nanogpt.py"
    printf '  --arch=%s \\\n' "${nanogpt_arch}"
    printf '  --data-dir=%s \\\n' "${nanogpt_data_dir}"
    printf '  --n-layer=%s --n-head=%s --n-embd=%s \\\n' "${nanogpt_n_layer}" "${nanogpt_n_head}" "${nanogpt_n_embd}"
    printf '  --block-size=%s --batch-size=%s \\\n' "${nanogpt_block_size}" "${nanogpt_batch_size}"
    printf '  --epochs=%s --iters-per-epoch=%s --eval-iters=%s \\\n' "${nanogpt_epochs}" "${nanogpt_iters}" "${nanogpt_eval_iters}"
    printf '  --lr=%s --lr-method=cos --optimizer=adamw --weight-decay=%s \\\n' "${nanogpt_lr}" "${nanogpt_weight_decay}"
    printf '  --dropout=%s --norm=%s --activation=gelu --dtype=%s \\\n' "${nanogpt_dropout}" "${norm}" "${nanogpt_dtype}"
    if needs_noncausal_nanogpt_flag "${norm}"; then
      printf '  --allow-noncausal-norm \\\n'
    fi
    printf '  --seed=%s --sample-tokens=%s --sample-every=%s \\\n' "${seed}" "${nanogpt_sample_tokens}" "${nanogpt_sample_every}"
    printf '  --output=%s/nanoGPT --wandb_project=%s \\\n' "${output_root}" "${wandb_project}"
    common_tail_args
  } > "${gen_dir}/${file_name}"
  chmod +x "${gen_dir}/${file_name}"
  append_launcher "${file_name}" "${base_string}"
}

generate_bert_job() {
  should_generate_job || return 0
  local norm="$1"
  local seed="$2"
  local base_string="execute_bert_${norm}_src${bert_max_src_len}_tgt${bert_max_tgt_len}_lr$(sanitize "${bert_lr}")_s${seed}"
  local file_name="${base_string}.sh"
  echo "Generating ${base_string}"
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\ncd "%s"\n' "${repo_root}"
    printf 'export WANDB_PROJECT=%s\n' "${wandb_project}"
    printf 'CUDA_VISIBLE_DEVICES=%s %s %s \\\n' "${cuda_visible_devices}" "${python_bin}" "${repo_root}/BERT/bert_translation.py"
    printf '  --data-dir=%s \\\n' "${bert_data_dir}"
    if [[ -n "${bert_input_file}" ]]; then
      printf '  --input-file=%s \\\n' "${bert_input_file}"
    fi
    printf '  --bert-layers=%s --bert-heads=%s --bert-embd=%s --bert-ffn-mult=%s \\\n' "${bert_layers}" "${bert_heads}" "${bert_embd}" "${bert_ffn_mult}"
    printf '  --max-src-len=%s --max-tgt-len=%s --batch-size=%s \\\n' "${bert_max_src_len}" "${bert_max_tgt_len}" "${bert_batch_size}"
    printf '  --epochs=%s --iters-per-epoch=%s --eval-iters=%s \\\n' "${bert_epochs}" "${bert_iters}" "${bert_eval_iters}"
    printf '  --lr=%s --lr-method=cos --optimizer=adamw --weight-decay=%s \\\n' "${bert_lr}" "${bert_weight_decay}"
    printf '  --dropout=%s --norm=%s --activation=gelu --dtype=%s \\\n' "${bert_dropout}" "${norm}" "${bert_dtype}"
    printf '  --seed=%s --sample-src=%q --sample-every=%s \\\n' "${seed}" "${bert_sample_src}" "${bert_sample_every}"
    printf '  --output=%s/BERT --wandb_project=%s \\\n' "${output_root}" "${wandb_project}"
    common_tail_args
  } > "${gen_dir}/${file_name}"
  chmod +x "${gen_dir}/${file_name}"
  append_launcher "${file_name}" "${base_string}"
}

generate_vit_job() {
  should_generate_job || return 0
  local norm="$1"
  local seed="$2"
  local patch_size="$3"
  if (( patch_size > vit_image_size )); then
    return 0
  fi
  local base_string="execute_vit_${vit_arch}_${vit_dataset}_p${patch_size}_${norm}_lr$(sanitize "${vit_lr}")_s${seed}"
  local file_name="${base_string}.sh"
  echo "Generating ${base_string}"
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\ncd "%s"\n' "${repo_root}"
    printf 'export WANDB_PROJECT=%s\n' "${wandb_project}"
    printf 'CUDA_VISIBLE_DEVICES=%s %s %s \\\n' "${cuda_visible_devices}" "${python_bin}" "${repo_root}/ViT/vit.py"
    printf '  --arch=%s --dataset=%s --dataset-root=%s \\\n' "${vit_arch}" "${vit_dataset}" "${vit_dataset_root}"
    printf '  --im-size=%s,%s --patch-size=%s --batch-size=%s \\\n' "${vit_image_size}" "${vit_image_size}" "${patch_size}" "${vit_batch_size}"
    printf '  --epochs=%s --lr=%s --lr-method=cos --optimizer=%s --weight-decay=%s \\\n' "${vit_epochs}" "${vit_lr}" "${vit_optimizer}" "${vit_weight_decay}"
    printf '  --dropout=%s --drop-path-rate=%s --norm=%s --activation=%s \\\n' "${vit_dropout}" "${vit_drop_path}" "${norm}" "${vit_activation}"
    printf '  --seed=%s --output=%s/ViT --wandb_project=%s \\\n' "${seed}" "${output_root}" "${wandb_project}"
    common_tail_args
  } > "${gen_dir}/${file_name}"
  chmod +x "${gen_dir}/${file_name}"
  append_launcher "${file_name}" "${base_string}"
}

generate_nanogpt_jobs() {
  local seed norm
  for seed in "${seeds[@]}"; do
    for norm in "${norms[@]}"; do
      generate_nanogpt_job "${norm}" "${seed}"
    done
    if [[ "${include_nanogpt_causal_sc}" == "1" ]]; then
      generate_nanogpt_job "CDSeqBCRMS" "${seed}"
    fi
  done
}

generate_bert_jobs() {
  local seed norm
  for seed in "${seeds[@]}"; do
    for norm in "${norms[@]}"; do
      generate_bert_job "${norm}" "${seed}"
    done
  done
}

generate_vit_jobs() {
  local seed norm patch_size
  for seed in "${seeds[@]}"; do
    for patch_size in "${vit_patch_sizes[@]}"; do
      for norm in "${norms[@]}"; do
        generate_vit_job "${norm}" "${seed}" "${patch_size}"
      done
    done
  done
}

case "${target}" in
  nanogpt) generate_nanogpt_jobs ;;
  bert) generate_bert_jobs ;;
  vit) generate_vit_jobs ;;
  all) generate_nanogpt_jobs; generate_bert_jobs; generate_vit_jobs ;;
  *) echo "Unknown TARGET: ${target}" >&2; exit 1 ;;
esac

if (( launch_cnt % num_once != 0 )); then
  echo "wait" >> "${gen_dir}/z_bash_execute.sh"
fi

chmod +x "${gen_dir}/z_bash_execute.sh"
echo "Generated ${launch_cnt} jobs under ${gen_dir}/"
echo "Run with: cd ${gen_dir} && bash z_bash_execute.sh"
