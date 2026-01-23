#!/usr/bin/env bash
cd "$(dirname $0)" 
CUDA_VISIBLE_DEVICES=0 /home/dlth/miniconda3/envs/norm-base/bin/python /home/dlth/norm-exp-code/Norm-exp-base/PDE/pde.py \
 -a=MLP \
 --width=32 \
 --depth=4 \
 --batch_size=128 \
 --dropout=0 \
 --pde_type=allen_cahn \
 --lr=1e-3 \
 --lr-method=step \
 --lr-step=5 \
 --lr-gamma=0.9 \
 --epochs=6000 \
 --norm=LN \
 --activation=sinarctan \
 --optimizer=adam \
 --weight-decay=0.0 \
 --seed=1 \
 --log-suffix=base \
 --no_save_best \
 --display_every=10 \
 --metrics "l2 relative error,MSE" \
 --subject_name="PDE Solving Updated 2" \
#  --visualize \
#   --output ./exp9 \

 #  --optimizer-config=momentum=0 \