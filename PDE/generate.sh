#!/bin/bash
dir=exp6
mkdir -p ${dir}
archs=(MLP)
widths=(64 32)
depths=(2 4)
norms=(LN)
activations=(relu tanh)
lrs=(1e-3)
seeds=(1)
pde_types=(helmholtz)
epochs=60000
display_every=10
metrics="l2 relative error,MSE"
optimizers=(adam sgd)

l=${#archs[@]}
w=${#widths[@]}
d=${#depths[@]}
n=${#norms[@]}
a=${#activations[@]}
t=${#lrs[@]}
s=${#seeds[@]}
pt=${#pde_types[@]}
ot=${#optimizers[@]}

for ((u=0;u<$ot;++u))
do
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
          for ((o=0;o<$a;++o))
          do
            for ((p=0;p<$t;++p))
            do
              for ((q=0;q<$s;++q))
              do
                baseString="execute_${archs[$i]}_w${widths[$j]}_d${depths[$k]}_${norms[$m]}_${activations[$o]}_lr${lrs[$p]}_s${seeds[$q]}_${pde_types[$r]}_${optimizers[$u]}_"
                fileName="${baseString}.sh"
                echo "Generating....${baseString}"
                touch "${dir}/${fileName}"
                echo  "#!/usr/bin/env bash
cd \"\$(dirname \$0)/../..\" 
CUDA_VISIBLE_DEVICES=0 /home/dlth/miniconda3/envs/norm-base/bin/python /home/dlth/norm-exp-code/Norm-exp-base/PDE/pde.py \\
 -a=${archs[$i]} \\
 --width=${widths[$j]} \\
 --depth=${depths[$k]} \\
 --dropout=0 \\
 --pde_type=${pde_types[$r]} \\
 --lr=${lrs[$p]} \\
 --epochs=${epochs} \\
 --norm=${norms[$m]} \\
 --norm-cfg=num_per_group=4,dim=2 \\
 --activation=${activations[$o]} \\
 --optimizer=${optimizers[$u]} \\
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
        done
      done
   done
done
done
done