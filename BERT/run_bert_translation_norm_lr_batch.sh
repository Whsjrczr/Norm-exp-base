#!/usr/bin/env bash
set -euo pipefail

dir_name="${DIR_NAME:-exp-bert-opus-books-norm-lr}"

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
gen_dir="${script_dir}/${dir_name}"
mkdir -p "${gen_dir}"

cp "$0" "${gen_dir}/gen_script.sh"

python_bin="${PYTHON_BIN:-${repo_root}/.conda/python.exe}"
if [ ! -x "${python_bin}" ]; then
  python_bin="${PYTHON_BIN:-python}"
fi

arch="${ARCH:-BERTTranslation}"
dataset="${DATASET:-opus_books_en_fr}"
bert_layers="${BERT_LAYERS:-4}"
bert_heads="${BERT_HEADS:-4}"
bert_embd="${BERT_EMBD:-256}"
bert_ffn_mult="${BERT_FFN_MULT:-4}"
max_src_len="${MAX_SRC_LEN:-64}"
max_tgt_len="${MAX_TGT_LEN:-64}"
batchsize="${BATCH_SIZE:-64,64}"
epochs="${EPOCHS:-50}"
iters_per_epoch="${ITERS_PER_EPOCH:-100}"
eval_iters="${EVAL_ITERS:-50}"
display_every="${DISPLAY_EVERY:-1}"
optimizers=(adamw)
momentum="${MOMENTUM:-0.9}"
weightdecay="${WEIGHT_DECAY:-0.01}"
dropouts=(0.1)
lrs_adamw=(1e-4 3e-4 6e-4 1e-3)
lrs_sgd=(3e-3 1e-3 3e-4)
lrs_else=(3e-4)
lr_method="${LR_METHOD:-cos}"
lrstep="${LR_STEP:-30}"
lrgamma="${LR_GAMMA:-1e-5}"
activation="${ACTIVATION:-gelu}"
dtype="${DTYPE:-bfloat16}"
seeds=(0 1 2)
sample_src="${SAMPLE_SRC:-I love books}"
sample_every="${SAMPLE_EVERY:-0}"
subjectname="${WANDB_PROJECT:-BERT-OPUS-Books-Norm-LR}"

opus_url="${OPUS_URL:-https://object.pouta.csc.fi/OPUS-Books/v1/moses/en-fr.txt.zip}"
source_lang="${SOURCE_LANG:-en}"
target_lang="${TARGET_LANG:-fr}"
pair_name="${PAIR_NAME:-en-fr}"
max_pairs="${MAX_PAIRS:-50000}"
min_freq="${MIN_FREQ:-2}"
data_dir="${DATA_DIR:-${repo_root}/dataset/opus_books_en_fr}"
pairs_tsv="${PAIRS_TSV:-${data_dir}/pairs.tsv}"
output_root="${OUTPUT_ROOT:-${repo_root}/results/${dir_name}}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
num_once="${NUM_ONCE:-1}"
launch_cnt=0
manifest="${gen_dir}/manifest.csv"

norms=(
  LN
  BN BNc BNs
  SBN SBNc SBNs
  SeqBN SeqBNc SeqBNs
  CFBN CFBNc CFBNs
)

: > "${gen_dir}/z_bash_execute.sh"
echo "job_file,norm,optimizer,lr,dropout,seed,output_root" > "${manifest}"

sanitize() {
  local value="$1"
  value="${value//+/p}"
  value="${value//,/x}"
  value="${value//./p}"
  printf '%s' "${value}"
}

optimizer_extra_args_for() {
  local optimizer="$1"
  if [ "${optimizer}" = "sgd" ]; then
    printf '  --optimizer-config=momentum=%s \\\n' "${momentum}"
  fi
}

scheduler_extra_args_for() {
  if [ "${lr_method}" = "step" ]; then
    printf '  --lr-step=%s \\\n' "${lrstep}"
    printf '  --lr-gamma=%s \\\n' "${lrgamma}"
  fi
}

prepare_data() {
  echo "Preparing OPUS Books translation data under ${data_dir}"
  "${python_bin}" "${repo_root}/BERT/download_opus_books.py" \
    --url="${opus_url}" \
    --data-dir="${data_dir}" \
    --output-tsv="${pairs_tsv}" \
    --source-lang="${source_lang}" \
    --target-lang="${target_lang}" \
    --pair-name="${pair_name}" \
    --max-pairs="${max_pairs}" \
    --max-src-tokens="${max_src_len}" \
    --max-tgt-tokens="${max_tgt_len}"

  "${python_bin}" "${repo_root}/BERT/prepare_translation.py" \
    --data-dir="${data_dir}" \
    --input-file="${pairs_tsv}" \
    --train-ratio=0.9 \
    --min-freq="${min_freq}" \
    --overwrite
}

generate_job() {
  local norm="$1"
  local optimizer="$2"
  local lr="$3"
  local dropout="$4"
  local seed="$5"

  local lr_tag batch_tag
  lr_tag="$(sanitize "${lr}")"
  batch_tag="$(sanitize "${batchsize}")"

  local baseString="execute_${arch}_${dataset}_L${bert_layers}_H${bert_heads}_D${bert_embd}_src${max_src_len}_tgt${max_tgt_len}_${norm}_${activation}_lr${lr_tag}_bs${batch_tag}_drop${dropout}_wd${weightdecay}_s${seed}_${optimizer}"
  local fileName="${baseString}.sh"
  echo "Generating ${baseString}"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd "$(dirname "$0")/../.."\n'
    printf 'CUDA_VISIBLE_DEVICES=%s "%s" "%s" \\\n' "${CUDA_VISIBLE_DEVICES}" "${python_bin}" "${repo_root}/BERT/bert_translation.py"
    printf '  --arch=%s \\\n' "${arch}"
    printf '  --data-dir="%s" \\\n' "${data_dir}"
    printf '  --no-auto-prepare \\\n'
    printf '  --bert-layers=%s \\\n' "${bert_layers}"
    printf '  --bert-heads=%s \\\n' "${bert_heads}"
    printf '  --bert-embd=%s \\\n' "${bert_embd}"
    printf '  --bert-ffn-mult=%s \\\n' "${bert_ffn_mult}"
    printf '  --max-src-len=%s \\\n' "${max_src_len}"
    printf '  --max-tgt-len=%s \\\n' "${max_tgt_len}"
    printf '  --batch-size=%s \\\n' "${batchsize}"
    printf '  --epochs=%s \\\n' "${epochs}"
    printf '  --iters-per-epoch=%s \\\n' "${iters_per_epoch}"
    printf '  --eval-iters=%s \\\n' "${eval_iters}"
    printf '  --lr=%s \\\n' "${lr}"
    printf '  --lr-method=%s \\\n' "${lr_method}"
    scheduler_extra_args_for
    printf '  --optimizer=%s \\\n' "${optimizer}"
    optimizer_extra_args_for "${optimizer}"
    printf '  --weight-decay=%s \\\n' "${weightdecay}"
    printf '  --dropout=%s \\\n' "${dropout}"
    printf '  --norm=%s \\\n' "${norm}"
    printf '  --activation=%s \\\n' "${activation}"
    printf '  --dtype=%s \\\n' "${dtype}"
    printf '  --seed=%s \\\n' "${seed}"
    printf '  --sample-src="%s" \\\n' "${sample_src}"
    printf '  --sample-every=%s \\\n' "${sample_every}"
    printf '  --print-f=%s \\\n' "${display_every}"
    printf '  --output="%s" \\\n' "${output_root}"
    printf '  --visualize \\\n'
    printf '  --wandb_project="%s" \\\n' "${subjectname}"
    printf '  --no-save-checkpoint\n'
  } > "${gen_dir}/${fileName}"
  chmod +x "${gen_dir}/${fileName}"

  echo "${fileName},${norm},${optimizer},${lr},${dropout},${seed},${output_root}" >> "${manifest}"
  echo "nohup bash ${fileName} > output_${baseString}.out 2>&1 &" >> "${gen_dir}/z_bash_execute.sh"
  launch_cnt=$((launch_cnt + 1))
  if (( launch_cnt % num_once == 0 )); then
    echo "wait" >> "${gen_dir}/z_bash_execute.sh"
  fi
}

prepare_data

for optimizer in "${optimizers[@]}"; do
  if [ "${optimizer}" = "adamw" ] || [ "${optimizer}" = "adam" ]; then
    lrs=("${lrs_adamw[@]}")
  elif [ "${optimizer}" = "sgd" ]; then
    lrs=("${lrs_sgd[@]}")
  else
    lrs=("${lrs_else[@]}")
  fi

  for norm in "${norms[@]}"; do
    for lr in "${lrs[@]}"; do
      for dropout in "${dropouts[@]}"; do
        for seed in "${seeds[@]}"; do
          generate_job "${norm}" "${optimizer}" "${lr}" "${dropout}" "${seed}"
        done
      done
    done
  done
done

if (( launch_cnt % num_once != 0 )); then
  echo "wait" >> "${gen_dir}/z_bash_execute.sh"
fi

chmod +x "${gen_dir}/z_bash_execute.sh"
echo "Generated ${launch_cnt} jobs under ${gen_dir}/"
echo "Run with: cd ${gen_dir} && bash z_bash_execute.sh"
