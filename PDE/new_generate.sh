#!/bin/bash
dir=exp8
mkdir -p ${dir}
expfilename=pde
archs=(MLP)
widths=(256 512)
depths=(6)
batchsize=128
norms=(PLN)
norm_groups=(4 8 16 32)
activations=(no)
seeds=(1)
pde_types=(helmholtz_new possion_new allen_cahn_new)
epochs=6000
display_every=10
metrics="l2 relative error,MSE"
optimizers=(adam sgd)
momentum=0.0
weightdecay=0.0
lrsadam=(1e-4 3e-4 3e-5 1e-5)
lrssgd=(1e-3 1e-4 3e-4 3e-5 1e-5)
lrselse=(1e-4)
lrstep=20
lrgamma=0.8
subjectname="PINN-new-task"
loss_weights=("10.0,1.0" "3.0,1.0" "1.0,1.0" "1.0,3.0" "1.0,10.0")
CUDA_VISIBLE_DEVICES=1



launch_cnt=0
num_once=2


l=${#archs[@]}
w=${#widths[@]}
d=${#depths[@]}
n=${#norms[@]}
ng=${#norm_groups[@]}
a=${#activations[@]}
s=${#seeds[@]}
pt=${#pde_types[@]}
ot=${#optimizers[@]}
lws=${#loss_weights[@]}


for ((lw=0;lw<$lws;++lw))
do
  for ((u=0;u<$ot;++u))
  do
    optimizer=${optimizers[$u]}
    if [ "${optimizer}" = "adam" ]; then
      lrs=("${lrsadam[@]}")
    elif [ "${optimizer}" = "sgd" ]; then
      lrs=("${lrssgd[@]}")
    else
      lrs=("${lrselse[@]}")
    fi
    t=${#lrs[@]}
    for ((r=0;r<$pt;++r))
    do
      for ((i=0;i<$l;++i))
      do 
        for ((j=0;j<$w;++j))
        do 
          for ((k=0;k<$d;++k))
          do	
            for ((m=0;m<$n;++m))
            do
              norm=${norms[$m]}
              for ((o=0;o<$a;++o))
              do
                activation=${activations[$o]}
                # if [ "${norm}" = "no" ] && [ "${activation}" = "no" ]; then
                #   continue
                # fi
                if [ "${norm}" = "PLN" ]; then
                  for ((g=0;g<$ng;++g))
                  do
                    norm_group=${norm_groups[$g]}
                    for ((p=0;p<$t;++p))
                    do
                      for ((q=0;q<$s;++q))
                      do
                        baseString="execute_${archs[$i]}_w${widths[$j]}_d${depths[$k]}_${norm}${norm_group}_${activation}_lr${lrs[$p]}_s${seeds[$q]}_${pde_types[$r]}_${optimizer}_lw${loss_weights[$lw]}"
                        fileName="${baseString}.sh"
                        echo "Generating....${baseString}"
                        touch "${dir}/${fileName}"
                        echo  "#!/usr/bin/env bash
  cd \"\$(dirname \$0)/../..\" 
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} /home/dlth/miniconda3/envs/norm-base/bin/python /home/dlth/norm-exp-code/Norm-exp-base/PDE/${expfilename}.py \\
  -a=${archs[$i]} \\
  --width=${widths[$j]} \\
  --depth=${depths[$k]} \\
  --batch_size=${batchsize} \\
  --dropout=0 \\
  --pde_type=${pde_types[$r]} \\
  --lr=${lrs[$p]} \\
  --lr-method=step \\
  --lr-step=${lrstep} \\
  --lr-gamma=${lrgamma} \\
  --epochs=${epochs} \\
  --norm=${norm} \\
  --norm-cfg=num_per_group=${norm_group},dim=2 \\
  --activation=${activation} \\
  --optimizer=${optimizer} \\
  --weight-decay=${weightdecay} \\
  --seed=${seeds[$q]} \\
  --no_save_best \\
  --display_every=${display_every} \\
  --output ./results/${dir} \\
  --metrics \"${metrics}\" \\
  --visualize \\
  --subject_name=\"${subjectname}\" \\
  --loss-weights=\"${loss_weights[$lw]}\" \\
  --float64 \\
  #  --optimizer-config=momentum=${momentum} \\
  #  --log-suffix=base \\" >> ${dir}/${fileName}
                        echo "nohup bash ${fileName} >output_${baseString}.out 2>&1 &" >> ${dir}/z_bash_excute.sh
                        launch_cnt=$((launch_cnt + 1))
                        if (( launch_cnt % num_once == 0 )); then
                          echo "wait" >> ${dir}/z_bash_excute.sh
                        fi
                      done
                    done
                  done
                else
                  for ((p=0;p<$t;++p))
                  do
                    for ((q=0;q<$s;++q))
                    do
                      baseString="execute_${archs[$i]}_w${widths[$j]}_d${depths[$k]}_${norm}_${activation}_lr${lrs[$p]}_s${seeds[$q]}_${pde_types[$r]}_${optimizer}_lw${loss_weights[$lw]}"
                      fileName="${baseString}.sh"
                      echo "Generating....${baseString}"
                      touch "${dir}/${fileName}"
                      echo  "#!/usr/bin/env bash
  cd \"\$(dirname \$0)/../..\" 
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} /home/dlth/miniconda3/envs/norm-base/bin/python /home/dlth/norm-exp-code/Norm-exp-base/PDE/${expfilename}.py \\
  -a=${archs[$i]} \\
  --width=${widths[$j]} \\
  --depth=${depths[$k]} \\
  --batch_size=${batchsize} \\
  --dropout=0 \\
  --pde_type=${pde_types[$r]} \\
  --lr=${lrs[$p]} \\
  --lr-method=step \\
  --lr-step=${lrstep} \\
  --lr-gamma=${lrgamma} \\
  --epochs=${epochs} \\
  --norm=${norm} \\
  --activation=${activation} \\
  --optimizer=${optimizer} \\
  --weight-decay=${weightdecay} \\
  --seed=${seeds[$q]} \\
  --no_save_best \\
  --display_every=${display_every} \\
  --output ./results/${dir} \\
  --metrics \"${metrics}\" \\
  --visualize \\
  --subject_name=\"${subjectname}\" \\
  --loss-weights=\"${loss_weights[$lw]}\" \\
  --float64 \\
  #  --optimizer-config=momentum=${momentum} \\
  #  --log-suffix=base \\" >> ${dir}/${fileName}
                      echo "nohup bash ${fileName} >output_${baseString}.out 2>&1 &" >> ${dir}/z_bash_excute.sh
                      launch_cnt=$((launch_cnt + 1))
                      if (( launch_cnt % 3 == 0 )); then
                        echo "wait" >> ${dir}/z_bash_excute.sh
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