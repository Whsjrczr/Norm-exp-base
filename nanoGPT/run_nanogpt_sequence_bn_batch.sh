#!/bin/bash
dir_name=exp-seqbn

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
gen_dir="${script_dir}/${dir_name}"
mkdir -p "${gen_dir}"

cp "$0" "${gen_dir}/gen_script.sh"

archs=(nanoGPT)
datasets=(tinyshakespeare)
n_layers=(6)
n_heads=(6)
n_embds=(384)
block_sizes=(256)
batchsize=64,64
control_norms=(LN BN)
norms=(CSBNs CSeqBNs CDSeqBNs)
slots=(attn mlp final all)
activations=(gelu)
seeds=(0)
epochs=50
iters_per_epoch=100
eval_iters=50
display_every=1
optimizers=(adamw)
momentum=0.9
weightdecay=0.1
dropouts=(0.2)
lrs_adamw=(6e-4)
lrs_sgd=(3e-3 1e-3)
lrs_else=(6e-4)
lr_method=cos
lrstep=30
lrgamma=0.1
dtype=bfloat16
sample_tokens=0
sample_every=0
subjectname="nanoGPT-SeqBN"
dataset_root="${repo_root}/dataset/tinyshakespeare"
output_root="${repo_root}/results/${dir_name}"
python_bin="${repo_root}/.conda/python.exe"

CUDA_VISIBLE_DEVICES=0

launch_cnt=0
num_once=1

: > "${gen_dir}/z_bash_execute.sh"

l=${#archs[@]}
ds=${#datasets[@]}
nl=${#n_layers[@]}
nh=${#n_heads[@]}
ne=${#n_embds[@]}
bs=${#block_sizes[@]}
cn=${#control_norms[@]}
n=${#norms[@]}
slot_cnt=${#slots[@]}
a=${#activations[@]}
s=${#seeds[@]}
ot=${#optimizers[@]}
do_cnt=${#dropouts[@]}

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
      for ((li=0; li<nl; ++li))
      do
        n_layer=${n_layers[$li]}

        for ((hi=0; hi<nh; ++hi))
        do
          n_head=${n_heads[$hi]}

          for ((ei=0; ei<ne; ++ei))
          do
            n_embd=${n_embds[$ei]}

            if (( n_embd % n_head != 0 )); then
              continue
            fi

            for ((bi=0; bi<bs; ++bi))
            do
              block_size=${block_sizes[$bi]}

              for ((o=0; o<a; ++o))
              do
                activation=${activations[$o]}

                for ((dd=0; dd<do_cnt; ++dd))
                do
                  dropout=${dropouts[$dd]}

                  for ((p=0; p<t; ++p))
                  do
                    for ((q=0; q<s; ++q))
                    do
                      for ((c=0; c<cn; ++c))
                      do
                        norm=${control_norms[$c]}
                        control_extra_args=""
                        if [ "${norm}" = "BN" ] || [ "${norm}" = "BNc" ] || [ "${norm}" = "BNs" ]; then
                          control_extra_args="
  --allow-noncausal-norm \\"
                        fi
                        baseString="execute_${archs[$i]}_${datasets[$r]}_L${n_layer}_H${n_head}_D${n_embd}_ctx${block_size}_control${norm}_${activation}_lr${lrs[$p]}_bs${batchsize}_drop${dropout}_wd${weightdecay}_s${seeds[$q]}_${optimizer}"
                        fileName="${baseString}.sh"
                        echo "Generating ${baseString}"
                        touch "${gen_dir}/${fileName}"
                        cat > "${gen_dir}/${fileName}" <<EOF
#!/usr/bin/env bash
cd "\$(dirname "\$0")/../.."
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${python_bin} ${repo_root}/nanoGPT/nanogpt.py \\
  --arch=${archs[$i]} \\
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
  --lr=${lrs[$p]} \\
  --lr-method=${lr_method} \\${scheduler_extra_args}${optimizer_extra_args}
  --optimizer=${optimizer} \\
  --weight-decay=${weightdecay} \\
  --dropout=${dropout} \\
  --norm=${norm} \\${control_extra_args}
  --activation=${activation} \\
  --dtype=${dtype} \\
  --seed=${seeds[$q]} \\
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
                      done

                      for ((m=0; m<n; ++m))
                      do
                        norm=${norms[$m]}

                        for ((sl=0; sl<slot_cnt; ++sl))
                        do
                          slot=${slots[$sl]}

                          norm_extra_args=""
                          norm_tag="${norm}"
                          if [ "${slot}" = "attn" ]; then
                            norm_extra_args="
  --attn-norm=${norm} \\
  --mlp-norm=LN \\
  --final-norm=LN \\"
                            norm_tag="attn${norm}"
                          elif [ "${slot}" = "mlp" ]; then
                            norm_extra_args="
  --attn-norm=LN \\
  --mlp-norm=${norm} \\
  --final-norm=LN \\"
                            norm_tag="mlp${norm}"
                          elif [ "${slot}" = "final" ]; then
                            norm_extra_args="
  --attn-norm=LN \\
  --mlp-norm=LN \\
  --final-norm=${norm} \\"
                            norm_tag="final${norm}"
                          elif [ "${slot}" = "all" ]; then
                            norm_extra_args="
  --norm=${norm} \\"
                            norm_tag="all${norm}"
                          else
                            echo "Unknown slot: ${slot}" >&2
                            exit 1
                          fi

                          baseString="execute_${archs[$i]}_${datasets[$r]}_L${n_layer}_H${n_head}_D${n_embd}_ctx${block_size}_${norm_tag}_${activation}_lr${lrs[$p]}_bs${batchsize}_drop${dropout}_wd${weightdecay}_s${seeds[$q]}_${optimizer}"
                          fileName="${baseString}.sh"
                          echo "Generating ${baseString}"
                          touch "${gen_dir}/${fileName}"
                          cat > "${gen_dir}/${fileName}" <<EOF
#!/usr/bin/env bash
cd "\$(dirname "\$0")/../.."
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${python_bin} ${repo_root}/nanoGPT/nanogpt.py \\
  --arch=${archs[$i]} \\
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
  --lr=${lrs[$p]} \\
  --lr-method=${lr_method} \\${scheduler_extra_args}${optimizer_extra_args}
  --optimizer=${optimizer} \\
  --weight-decay=${weightdecay} \\
  --dropout=${dropout} \\${norm_extra_args}
  --activation=${activation} \\
  --dtype=${dtype} \\
  --seed=${seeds[$q]} \\
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
                        done
                      done
                    done
                  done
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
