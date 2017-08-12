#!/bin/bash
source set_env.sh
python experiments/cifar10/train.py \
	--run_type tan-only \
	--generator gru \
	--gen_config init_type=train,feed_actions=True,n_stack=1,logit_range=6.0 \
	--n_sample 5 \
	--transformer image \
	--gamma 0.5 \
	--gen_lr 0.0001 \
	--disc_lr 0.00001 \
	--mse_term 0.001 \
	--mse_layer 1 \
	--seq_len 10 \
	--per_img_std false \
	--batch_size 32 \
	--n_epochs 1 \
	--run_name cifar_tan \
	--is_test True \
	--plot_every 1 \
	--save_model False
