#!/bin/bash
set -euo pipefail

dir_name=exp-cfbn-slot-sensitivity

script_dir="$(cd "$(dirname "$0")" && pwd)"
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
epochs=50
iters_per_epoch=100
eval_iters=50
display_every=1
optimizer=adamw
weightdecay=0.1
dropout=0.2
lr=6e-4
lr_method=cos
activation=gelu
dtype=bfloat16
seed=0
sample_tokens=0
sample_every=0
subjectname="nanoGPT-CFBN"
dataset_root="/home/dlth/norm-exp-code/dataset/tinyshakespeare"
output_root="/home/dlth/norm-exp-code/Norm-exp-base/nanoGPT/results/${dir_name}"
python_bin="/home/dlth/miniconda3/envs/norm-base/bin/python"

CUDA_VISIBLE_DEVICES=0
num_once=1
launch_cnt=0

control_norms=(LN RMS)
norms=(CCFBN CCFBNc CCFBNs CSBN CSBNs CSeqBN CDSeqBN CDSeqBNs)
slots=(attn mlp final all)

: > "${gen_dir}/z_bash_execute.sh"

add_job() {
  local norm="$1"
  local slot="$2"
  local tag="$3"
  local norm_extra_args=""
  local norm_tag="${slot}${norm}"

  if [ "${slot}" = "attn" ]; then
    norm_extra_args="
  --attn-norm=${norm} \\
  --mlp-norm=LN \\
  --final-norm=LN \\"
  elif [ "${slot}" = "mlp" ]; then
    norm_extra_args="
  --attn-norm=LN \\
  --mlp-norm=${norm} \\
  --final-norm=LN \\"
  elif [ "${slot}" = "final" ]; then
    norm_extra_args="
  --attn-norm=LN \\
  --mlp-norm=LN \\
  --final-norm=${norm} \\"
  elif [ "${slot}" = "all" ]; then
    norm_extra_args="
  --norm=${norm} \\"
    norm_tag="all${norm}"
  else
    echo "Unknown slot: ${slot}" >&2
    exit 1
  fi

  local baseString="execute_${tag}_${arch}_${dataset}_L${n_layer}_H${n_head}_D${n_embd}_ctx${block_size}_${norm_tag}_${activation}_lr${lr}_bs${batchsize}_drop${dropout}_wd${weightdecay}_s${seed}_${optimizer}"
  local fileName="${baseString}.sh"
  echo "Generating ${baseString}"
  cat > "${gen_dir}/${fileName}" <<EOF
#!/usr/bin/env bash
cd "\$(dirname "\$0")/../.."
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${python_bin} /home/dlth/norm-exp-code/Norm-exp-base/nanoGPT/nanogpt.py \\
  --arch=${arch} \\
  --data-dir=${dataset_root} \\
  --no-auto-prepare \\
  --n-layer=${n_layer} \\
  --n-head=${n_head} \\
  --n-embd=${n_embd} \\
  --block-size=${block_size} \\
  --batch-size=${batchsize} \\
  --epochs=${epochs} \\
  --iters-per-epoch=${iters_per_epoch} \\
  --eval-iters=${eval_iters} \\
  --lr=${lr} \\
  --lr-method=${lr_method} \\
  --optimizer=${optimizer} \\
  --weight-decay=${weightdecay} \\
  --dropout=${dropout} \\${norm_extra_args}
  --activation=${activation} \\
  --dtype=${dtype} \\
  --seed=${seed} \\
  --sample-tokens=${sample_tokens} \\
  --sample-every=${sample_every} \\
  --print-f=${display_every} \\
  --output=${output_root} \\
  --visualize \\
  --wandb_project="${subjectname}" \\
  --no-save-checkpoint
EOF

  echo "nohup bash ${fileName} > output_${baseString}.out 2>&1 &" >> "${gen_dir}/z_bash_execute.sh"
  launch_cnt=$((launch_cnt + 1))
  if (( launch_cnt % num_once == 0 )); then
    echo "wait" >> "${gen_dir}/z_bash_execute.sh"
  fi
}

for norm in "${control_norms[@]}"; do
  add_job "${norm}" all control
done

for norm in "${norms[@]}"; do
  for slot in "${slots[@]}"; do
    add_job "${norm}" "${slot}" cfbn
  done
done

if (( launch_cnt % num_once != 0 )); then
  echo "wait" >> "${gen_dir}/z_bash_execute.sh"
fi

echo "Generated ${launch_cnt} jobs under ${gen_dir}/"
