#!/bin/bash 

OPTS=""
OPTS+="--id AVE-clip-FM "

OPTS+="--list_train ../data/AVE/train.csv "
OPTS+="--list_val ../data/AVE/testsepave.csv "

# Models
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 64 "
OPTS+="--loss l1 "
OPTS+="--weighted_loss 0 "

# logscale in frequency
OPTS+="--num_mix 2 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--arch_frame clip " # [resnet18, clip]
OPTS+="--num_frames 11 "
OPTS+="--stride_frames 1 "
OPTS+="--frameRate 1 "

# audio-related
OPTS+="--audLen 110000 " 
OPTS+="--audRate 11000 " 

# learning params
OPTS+="--num_gpus 2 "
OPTS+="--gpu_ids 0,1 "
OPTS+="--workers 4 "
OPTS+="--batch_size_per_gpu 4 "
OPTS+="--lr_frame 1e-4 " #1e-4
OPTS+="--lr_unet 1e-4 " #1e-4
OPTS+="--num_epoch 400 "
OPTS+="--lr_steps 100 200 300 "
OPTS+="--dup_trainset 5 "
OPTS+="--eval_epoch 1 "

OPTS+="--ckpt YOUR_CKPT "

# display, viz
OPTS+="--disp_iter 200 "
OPTS+="--num_vis 40 "
# OPTS+="--num_val 26 "

OPTS+="--split test "
OPTS+="--mode train"

python -u ../main_ave_fm.py $OPTS
