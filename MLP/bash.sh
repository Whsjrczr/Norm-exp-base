CUDA_VISIBLE_DEVICES=0 /home/layernorm/conda_envs/norm-base/bin/python /home/layernorm/centering/Norm-exp-base/MLP/cifar10.py \
 -a=ConvBNPre \
 --batch-size=100 \
 --depth=3 \
 --width=128 \
 --epochs=100 \
 -oo=sgd \
 -oc=momentum=0 \
 -wd=0 \
 --lr=0.01 \
 --lr-method=step \
 --lr-step=5 \
 --lr-gamma=0.9 \
 --dataset=cifar10_nogrey \
 --dataset-root='/home/layernorm/centering/Norm-exp-base/dataset/' \
 --norm=BNs \
 --norm-cfg=T=5,num_channels=0,num_groups=2,dim=4 \
 --activation=relu \
 --activation-cfg=num_groups=16 \
 --seed=1 \
 --log-suffix=base \