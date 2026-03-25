#!/bin/bash


dir_name=exp2
script_dir="$(cd "$(dirname "$0")" && pwd)"
gen_dir="${script_dir}/${dir_name}"
mkdir -p "${gen_dir}"

archs=(vit_small)
datasets=(cifar10)
image_sizes=(32)
patch_sizes=(4 16)
batchsize=256
norms=(LN bCRMS BN)
norm_groups=(4)
activations=(relu gelu silu)
seeds=(0)
epochs=200
display_every=1
optimizers=(sgd)
momentum=0.9
weightdecay=0.1
dropouts=(0.0)
droppaths=(0.1)
lrs_adamw=(1e-4 1e-3)
lrs_sgd=(3e-3 1e-3)
lrs_else=(1e-4)
lr_method=cos
lrstep=30
lrgamma=0.1
subjectname="ViT-new-task"
dataset_root="/home/dlth/norm-exp-code/dataset"
output_root="/home/dlth/norm-exp-code/Norm-exp-base/ViT/results/${dir_name}"
python_bin="/home/dlth/miniconda3/envs/norm-base/bin/python"

CUDA_VISIBLE_DEVICES=1

launch_cnt=0
num_once=2

: > "${gen_dir}/z_bash_execute.sh"

l=${#archs[@]}
ds=${#datasets[@]}
is=${#image_sizes[@]}
ps=${#patch_sizes[@]}
n=${#norms[@]}
ng=${#norm_groups[@]}
a=${#activations[@]}
s=${#seeds[@]}
ot=${#optimizers[@]}
do_cnt=${#dropouts[@]}
dp_cnt=${#droppaths[@]}

for ((u=0; u<ot; ++u))
do
  optimizer=${optimizers[$u]}
  if [ "${optimizer}" = "adamw" ] || [ "${optimizer}" = "adam" ]; then
    lrs=("${lrs_adamw[@]}")
  elif [ "${optimizer}" = "sgd" ]; then
    lrs=("${lrs_sgd[@]}")
  else
    lrs=("${lrs_else[@]}")
  fi

  optimizer_extra_args=""
  if [ "${optimizer}" = "sgd" ]; then
    optimizer_extra_args="
  --optimizer-config=momentum=${momentum} \\"
  fi

  scheduler_extra_args=""
  if [ "${lr_method}" = "step" ]; then
    scheduler_extra_args="
  --lr-step=${lrstep} \\
  --lr-gamma=${lrgamma} \\"
  fi

  t=${#lrs[@]}

  for ((r=0; r<ds; ++r))
  do
    for ((i=0; i<l; ++i))
    do
      for ((j=0; j<is; ++j))
      do
        image_size=${image_sizes[$j]}

        for ((pp=0; pp<ps; ++pp))
        do
          patch_size=${patch_sizes[$pp]}

          if (( patch_size > image_size )); then
            continue
          fi

          for ((m=0; m<n; ++m))
          do
            norm=${norms[$m]}

            for ((o=0; o<a; ++o))
            do
              activation=${activations[$o]}

              for ((dd=0; dd<do_cnt; ++dd))
              do
                dropout=${dropouts[$dd]}

                for ((dp=0; dp<dp_cnt; ++dp))
                do
                  droppath=${droppaths[$dp]}

                  if [ "${norm}" = "PLN" ] || [ "${norm}" = "PLS" ]; then
                    for ((g=0; g<ng; ++g))
                    do
                      norm_group=${norm_groups[$g]}

                      for ((p=0; p<t; ++p))
                      do
                        for ((q=0; q<s; ++q))
                        do
                          baseString="execute_${archs[$i]}_${datasets[$r]}_img${image_size}_patch${patch_size}_${norm}${norm_group}_${activation}_lr${lrs[$p]}_bs${batchsize}_drop${dropout}_dpath${droppath}_wd${weightdecay}_s${seeds[$q]}_${optimizer}"
                          fileName="${baseString}.sh"
                          echo "Generating ${baseString}"
                          touch "${gen_dir}/${fileName}"
                          cat > "${gen_dir}/${fileName}" <<EOF
#!/usr/bin/env bash
cd "\$(dirname "\$0")/../.."
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${python_bin} /home/dlth/norm-exp-code/Norm-exp-base/ViT/vit.py \\
  --arch=${archs[$i]} \\
  --dataset=${datasets[$r]} \\
  --dataset-root=${dataset_root} \\
  --im-size=${image_size},${image_size} \\
  --patch-size=${patch_size} \\
  --batch-size=${batchsize} \\
  --epochs=${epochs} \\
  --lr=${lrs[$p]} \\
  --lr-method=${lr_method} \\
  --optimizer=${optimizer} \\${scheduler_extra_args}${optimizer_extra_args}
  --weight-decay=${weightdecay} \\
  --dropout=${dropout} \\
  --drop-path-rate=${droppath} \\
  --norm=${norm} \\
  --norm-cfg=num_per_group=${norm_group} \\
  --activation=${activation} \\
  --seed=${seeds[$q]} \\
  --print-f=${display_every} \\
  --output=${output_root}\\
  --visualize \\
  --wandb_project="${subjectname}" \\
EOF
                          echo "nohup bash ${fileName} > output_${baseString}.out 2>&1 &" >> "${gen_dir}/z_bash_execute.sh"
                          launch_cnt=$((launch_cnt + 1))
                          if (( launch_cnt % num_once == 0 )); then
                            echo "wait" >> "${gen_dir}/z_bash_execute.sh"
                          fi
                        done
                      done
                    done
                  else
                    for ((p=0; p<t; ++p))
                    do
                      for ((q=0; q<s; ++q))
                      do
                        baseString="execute_${archs[$i]}_${datasets[$r]}_img${image_size}_patch${patch_size}_${norm}_${activation}_lr${lrs[$p]}_bs${batchsize}_drop${dropout}_dpath${droppath}_wd${weightdecay}_s${seeds[$q]}_${optimizer}"
                        fileName="${baseString}.sh"
                        echo "Generating ${baseString}"
                        touch "${gen_dir}/${fileName}"
                        cat > "${gen_dir}/${fileName}" <<EOF
#!/usr/bin/env bash
cd "\$(dirname "\$0")/../.."
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${python_bin} /home/dlth/norm-exp-code/Norm-exp-base/ViT/vit.py \\
  --arch=${archs[$i]} \\
  --dataset=${datasets[$r]} \\
  --dataset-root=${dataset_root} \\
  --im-size=${image_size},${image_size} \\
  --patch-size=${patch_size} \\
  --batch-size=${batchsize} \\
  --epochs=${epochs} \\
  --lr=${lrs[$p]} \\
  --lr-method=${lr_method} \\
  --optimizer=${optimizer} \\${scheduler_extra_args}${optimizer_extra_args}
  --weight-decay=${weightdecay} \\
  --dropout=${dropout} \\
  --drop-path-rate=${droppath} \\
  --norm=${norm} \\
  --activation=${activation} \\
  --seed=${seeds[$q]} \\
  --print-f=${display_every} \\
  --output=${output_root}\\
  --visualize \\
  --wandb_project="${subjectname}" \\
EOF
                        echo "nohup bash ${fileName} > output_${baseString}.out 2>&1 &" >> "${gen_dir}/z_bash_execute.sh"
                        launch_cnt=$((launch_cnt + 1))
                        if (( launch_cnt % num_once == 0 )); then
                          echo "wait" >> "${gen_dir}/z_bash_execute.sh"
                        fi
                      done
                    done
                  fi
                done
              done
            done
          done
        done
      done
    done
  done
done

if (( launch_cnt % num_once != 0 )); then
  echo "wait" >> "${gen_dir}/z_bash_execute.sh"
fi

echo "Generated ${launch_cnt} jobs under ${gen_dir}/"
