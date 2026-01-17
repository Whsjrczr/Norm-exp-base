CUDA_VISIBLE_DEVICES=0 /home/dlth/miniconda3/envs/norm-base/bin/python /home/dlth/norm-exp-code/Norm-exp-base/PDE/pde.py \
 -a=MLP \
 --width=32 \
 --depth=4 \
 --dropout=0 \
 --pde_type=allen_cahn \
 --lr=0.001 \
 --epochs=1000 \
 --norm=PLN \
 --norm-cfg=num_per_group=4,dim=2 \
 --activation=relu \
 --seed=1 \
 --log-suffix=base \
 --no_save_best \
 --display_every=5 \
 --metrics "l2 relative error,MSE" \
#  --visualize \


