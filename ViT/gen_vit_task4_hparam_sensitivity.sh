#!/bin/bash
set -euo pipefail

dir_name=exp-task4-hparam-sensitivity

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
dropout=0.0
lr_method=cos
activation=relu
seed=0
subjectname="ViT-SeqBN-hparam-sensitivity"
dataset_root="/home/dlth/norm-exp-code/dataset"
output_root="/home/dlth/norm-exp-code/Norm-exp-base/ViT/results/${dir_name}"
python_bin="/home/dlth/miniconda3/envs/norm-base/bin/python"

CUDA_VISIBLE_DEVICES=1
num_once=1
launch_cnt=0

norms=(LN RMS BN SBN bCRMS SeqBN SeqBNs DSeqBN DSeqBNs DSeqBCRMS)
weightdecays=(0.01 0.05 0.1)
droppaths=(0.0 0.1 0.2)

: > "${gen_dir}/z_bash_execute.sh"

add_job() {
  local patch_size="$1"
  local lr="$2"
  local norm="$3"
  local weightdecay="$4"
  local droppath="$5"
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

for norm in "${norms[@]}"; do
  for weightdecay in "${weightdecays[@]}"; do
    for droppath in "${droppaths[@]}"; do
      add_job 4 1e-4 "${norm}" "${weightdecay}" "${droppath}" patch4_lr1e4
      add_job 16 1e-3 "${norm}" "${weightdecay}" "${droppath}" patch16_lr1e3
    done
  done
done

if (( launch_cnt % num_once != 0 )); then
  echo "wait" >> "${gen_dir}/z_bash_execute.sh"
fi

echo "Generated ${launch_cnt} jobs under ${gen_dir}/"
