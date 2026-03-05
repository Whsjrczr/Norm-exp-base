CUDA_VISIBLE_DEVICES=0 python train.py \
 --arch vit_small \
 --lr 1e-4 \
 --batch_size 64 \
 --wd 0.1 \
 --num_classes 1000 \
 --data_path data/ImageNet \