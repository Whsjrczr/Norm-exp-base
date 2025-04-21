#!/bin/bash

archs=(ConvLNPre ConvLN ConvLNRes ConvLNPre)
batch_sizes=(128)
depth=5
width=128
methods=(LNs LN)
lrs=(1e-4)
activations=relu
epochs=100
wd=0
opt=sgd
seeds=(1)



l=${#archs[@]}
n=${#batch_sizes[@]}
m=${#methods[@]}
t=${#lrs[@]}
f=${#seeds[@]}

for ((a=0;a<$l;++a))
do 
   for ((b=0;b<$f;++b))
   do 
      for ((j=0;j<$m;++j))
      do	
        for ((k=0;k<$t;++k))
        do

          for ((i=0;i<$n;++i))
          do
                baseString="execute_${archs[$a]}_b${batch_sizes[$i]}_w${width}_d${depth}_${methods[$j]}_lr${lrs[$k]}_wd${wd}_s${seeds[$b]}_"
                fileName="${baseString}.sh"
   	            echo "${baseString}"
                touch "${fileName}"
                echo  "#!/usr/bin/env bash
cd \"\$(dirname \$0)/..\" 
CUDA_VISIBLE_DEVICES=1 /home/layernorm/conda_envs/norm-base/bin/python /home/layernorm/centering/Norm-exp-base/MLP/cifar10.py \\
 -a=${archs[$a]} \\
 --batch-size=${batch_sizes[$i]} \\
 --depth=${depth} \\
 --width=${width} \\
 --epochs=${epochs} \\
 -oo=${opt} \\
 -oc=momentum=0 \\
 -wd=0 \\
 --lr=0.01 \\
 --lr-method=step \\
 --lr-step=5 \\
 --lr-gamma=0.9 \\
 --dataset=cifar10_nogrey \\
 --dataset-root='/home/layernorm/centering/Norm-exp-base/dataset/' \\
 --norm=${methods[$j]} \\
 --norm-cfg=T=5,num_channels=0,num_groups=2,dim=4 \\
 --activation=${activations} \\
 --activation-cfg=num_groups=16 \\
 --seed=${seeds[$b]} \\
 --log-suffix=base \\" >> ${fileName}
                echo  "nohup bash ${fileName} >output_${baseString}.out 2>&1 & wait" >> z_bash_excute.sh
           done
         done
      done
   done
done