#!/usr/bin/env bash

## run the training
python train.py \
--arch msmoothnet \
--dataroot datasets/smoothing \
--dataset_mode unsupervised \
--num_groups 8 \
--gpu_ids 0 \
--name smoothing \
--ncf 8 16 32 64 \
--ninput_edges 2800 \
--pool_res 1400 800 500 350 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
