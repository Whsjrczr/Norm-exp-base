#!/bin/bash
set -euo pipefail

dir_name=exp-task1-failure-diagnostics

script_dir="$(cd "$(dirname "$0")" && pwd)"
gen_dir="${script_dir}/${dir_name}"
mkdir -p "${gen_dir}"

cp "$0" "${gen_dir}/gen_script.sh"

arch=vit_small
dataset=cifar10
image_size=32
batchsize=256
epochs=50
display_every=1
optimizer=adam
weightdecay=0.1
dropout=0.0
droppath=0.1
lr_method=cos
subjectname="ViT-SeqBN-failure-diagnostics"
dataset_root="/home/dlth/norm-exp-code/dataset"
output_root="/home/dlth/norm-exp-code/Norm-exp-base/ViT/results/${dir_name}"
python_bin="/home/dlth/miniconda3/envs/norm-base/bin/python"

CUDA_VISIBLE_DEVICES=1
num_once=1
launch_cnt=0

: > "${gen_dir}/z_bash_execute.sh"

declare -A seen_jobs

add_job() {
  local patch_size="$1"
  local norm="$2"
  local activation="$3"
  local lr="$4"
  local seed="$5"
  local job_epochs="${6:-${epochs}}"
  local tag="${7:-diag}"

  if (( patch_size > image_size )); then
    return
  fi

  local baseString="execute_${tag}_${arch}_${dataset}_img${image_size}_patch${patch_size}_${norm}_${activation}_lr${lr}_bs${batchsize}_drop${dropout}_dpath${droppath}_wd${weightdecay}_s${seed}_${optimizer}"
  if [[ -n "${seen_jobs[$baseString]:-}" ]]; then
    return
  fi
  seen_jobs[$baseString]=1

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
  --epochs=${job_epochs} \\
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
  --diagnostics \\
  --no-save-checkpoint
EOF

  echo "nohup bash ${fileName} > output_${baseString}.out 2>&1 &" >> "${gen_dir}/z_bash_execute.sh"
  launch_cnt=$((launch_cnt + 1))
  if (( launch_cnt % num_once == 0 )); then
    echo "wait" >> "${gen_dir}/z_bash_execute.sh"
  fi
}

# 1) c-family methods collapse near chance accuracy: check implementation/statistics.
for patch_size in 4 16; do
  for norm in BNc LNc SeqBNc DSeqBNc; do
    for lr in 1e-4 1e-3; do
      add_job "${patch_size}" "${norm}" relu "${lr}" 0 "${epochs}" c_family
    done
  done
done

# 2) SBNs fails on patch16; include patch8 as the token-count bridge.
for patch_size in 8 16; do
  for activation in relu gelu silu; do
    for lr in 1e-4 1e-3; do
      add_job "${patch_size}" SBNs "${activation}" "${lr}" 0 "${epochs}" sbns_bridge
    done
  done
done

# 3) BN/BNs/SBN/SBNs learning-rate sensitivity on patch16.
for norm in BN BNs SBN SBNs; do
  for lr in 3e-5 1e-4 3e-4 1e-3; do
    add_job 16 "${norm}" relu "${lr}" 0 "${epochs}" bn_lr_sweep
  done
done

# 4) Re-run the killed DSeqBCRMS setting with diagnostics.
add_job 4 DSeqBCRMS gelu 1e-3 0 "${epochs}" killed_rerun

if (( launch_cnt % num_once != 0 )); then
  echo "wait" >> "${gen_dir}/z_bash_execute.sh"
fi

echo "Generated ${launch_cnt} jobs under ${gen_dir}/"
