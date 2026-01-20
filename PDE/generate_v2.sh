#!/bin/bash
dir=exp6
mkdir -p ${dir}
archs=(MLP)
widths=(64 32)
depths=(2 4)
batchsize=128
norms=(PLN LN no)
norm_groups=(2 4)
activations=(relu tanh no)
seeds=(1)
pde_types=(helmholtz)
epochs=60000
display_every=10
metrics="l2 relative error,MSE"
optimizers=(adam sgd)
momentum=0
weightdecay=0.0
lrsadam=(1e-3 1e-4)
lrssgd=(1e-2 1e-3)
lrselse=(1e-3)
lrstep=5
lrgamma=0.9


l=${#archs[@]}
w=${#widths[@]}
d=${#depths[@]}
n=${#norms[@]}
ng=${#norm_groups[@]}
a=${#activations[@]}
s=${#seeds[@]}
pt=${#pde_types[@]}
ot=${#optimizers[@]}

for ((u=0;u<$ot;++u))
do
  optimizer=${optimizers[$u]}
  if [ "${optimizer}" = "adam" ]; then
    lrs=lrsadam[@]
  elif [ "${optimizer}" = "sgd" ]; then
    lrs=lrssgd[@]
  else
    lrs=lrselse[@]
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
              if [ "${norm}" = "no" ] && [ "${activation}" = "no" ]; then
                continue
              fi
              if [ "${norm}" = "PLN" ]; then
                for ((g=0;g<$ng;++g))
                do
                  norm_group=${norm_groups[$g]}
                  for ((p=0;p<$t;++p))
                  do
                    for ((q=0;q<$s;++q))
                    do
                      baseString="execute_${archs[$i]}_w${widths[$j]}_d${depths[$k]}_${norm}${norm_group}_${activation}_lr${lrs[$p]}_s${seeds[$q]}_${pde_types[$r]}_${optimizer}_"
                      fileName="${baseString}.sh"
                      echo "Generating....${baseString}"
                      touch "${dir}/${fileName}"
                      echo  "#!/usr/bin/env bash
cd \"\$(dirname \$0)/../..\" 
CUDA_VISIBLE_DEVICES=0 /home/dlth/miniconda3/envs/norm-base/bin/python /home/dlth/norm-exp-code/Norm-exp-base/PDE/pde.py \\
 -a=${archs[$i]} \\
 --width=${widths[$j]} \\
 --depth=${depths[$k]} \\
 --batchsize=${batchsize} \\
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
 --optimizer-config=momentum=${momentum} \\
 --weight-decay=${weightdecay} \\
 --seed=${seeds[$q]} \\
 --log-suffix=base \\
 --no_save_best \\
 --display_every=${display_every} \\
 --output ./${dir} \\
 --metrics \"${metrics}\" \\
 --visualize \\" >> ${dir}/${fileName}
                      echo  "nohup bash ${fileName} >output_${baseString}.out 2>&1 & wait" >> ${dir}/z_bash_excute.sh
                    done
                  done
                done
              else
                for ((p=0;p<$t;++p))
                do
                  for ((q=0;q<$s;++q))
                  do
                    baseString="execute_${archs[$i]}_w${widths[$j]}_d${depths[$k]}_${norm}_${activation}_lr${lrs[$p]}_s${seeds[$q]}_${pde_types[$r]}_${optimizer}_"
                    fileName="${baseString}.sh"
                    echo "Generating....${baseString}"
                    touch "${dir}/${fileName}"
                    echo  "#!/usr/bin/env bash
cd \"\$(dirname \$0)/../..\" 
CUDA_VISIBLE_DEVICES=0 /home/dlth/miniconda3/envs/norm-base/bin/python /home/dlth/norm-exp-code/Norm-exp-base/PDE/pde.py \\
 -a=${archs[$i]} \\
 --width=${widths[$j]} \\
 --depth=${depths[$k]} \\
 --batchsize=${batchsize} \\
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
 --optimizer-config=momentum=${momentum} \\
 --weight-decay=${weightdecay} \\
 --seed=${seeds[$q]} \\
 --log-suffix=base \\
 --no_save_best \\
 --display_every=${display_every} \\
 --output ./${dir} \\
 --metrics \"${metrics}\" \\
 --visualize \\" >> ${dir}/${fileName}
                    echo  "nohup bash ${fileName} >output_${baseString}.out 2>&1 & wait" >> ${dir}/z_bash_excute.sh
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
