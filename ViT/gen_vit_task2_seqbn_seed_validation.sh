#!/bin/bash
set -euo pipefail

dir_name=exp-task2-seqbn-seed-validation

script_dir="$(cd "$(dirname "$0")" && pwd)"
gen_dir="${script_dir}/${dir_name}"
mkdir -p "${gen_dir}"

cp "$0" "${gen_dir}/gen_script.sh"

arch=vit_small
dataset=cifar10
image_size=32
batchsize=256
epochs=200
display_every=1
optimizer=adam
weightdecay=0.1
dropout=0.0
droppath=0.1
lr_method=cos
subjectname="ViT-SeqBN-seed-validation"
dataset_root="/home/dlth/norm-exp-code/dataset"
output_root="/home/dlth/norm-exp-code/Norm-exp-base/ViT/results/${dir_name}"
python_bin="/home/dlth/miniconda3/envs/norm-base/bin/python"

CUDA_VISIBLE_DEVICES=1
num_once=1
launch_cnt=0

norms=(LN RMS BN SBN bCRMS SeqBN SeqBNs DSeqBN DSeqBNs DSeqBCRMS)
seeds=(1 2 3)

: > "${gen_dir}/z_bash_execute.sh"

add_job() {
  local patch_size="$1"
  local norm="$2"
  local activation="$3"
  local lr="$4"
  local seed="$5"
  local tag="$6"

  if (( patch_size > image_size )); then
    return
  fi

  local baseString="execute_${tag}_${arch}_${dataset}_img${image_size}_patch${patch_size}_${norm}_${activation}_lr${lr}_bs${batchsize}_drop${dropout}_dpath${droppath}_wd${weightdecay}_s${seed}_${optimizer}"
  local fileName="${baseString}.sh"
  echo "Generating ${baseString}"
  cat > "${gen_dir}/${fileName}" <<EOF
#!/usr/bin/env bash
cd "\$(dirname "\$0")/../.."
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${python_bin} /home/dlth/norm-exp-code/Norm-exp-base/ViT/vit.py \\
  --arch=${arch} \\
  --dataset=${dataset} \\
  --dataset-root=${dataset_root} \\
  --im-size=${image_size},${image_size} \\
  --patch-size=${patch_size} \\
  --batch-size=${batchsize} \\
  --epochs=${epochs} \\
  --lr=${lr} \\
  --lr-method=${lr_method} \\
  --optimizer=${optimizer} \\
  --weight-decay=${weightdecay} \\
  --dropout=${dropout} \\
  --drop-path-rate=${droppath} \\
  --norm=${norm} \\
  --activation=${activation} \\
  --seed=${seed} \\
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

# Complement existing seed=0 results with seeds 1, 2, and 3.
for seed in "${seeds[@]}"; do
  for norm in "${norms[@]}"; do
    add_job 4 "${norm}" relu 1e-4 "${seed}" patch4_lr1e4
    add_job 16 "${norm}" relu 1e-4 "${seed}" patch16_lr1e4
    add_job 16 "${norm}" relu 1e-3 "${seed}" patch16_lr1e3
  done
done

if (( launch_cnt % num_once != 0 )); then
  echo "wait" >> "${gen_dir}/z_bash_execute.sh"
fi

echo "Generated ${launch_cnt} jobs under ${gen_dir}/"
