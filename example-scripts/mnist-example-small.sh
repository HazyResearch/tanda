#!/bin/bash

python experiments/mnist/train.py \
	--run_type tan-only \
	--generator mean_field \
	--gen_config init_type=train,feed_actions=False \
	--n_sample 5 \
	--transformer image \
	--gamma 0.5 \
	--gen_lr 0.001 \
	--disc_lr 0.00001 \
	--mse_term 0.0001 \
	--mse_layer 1 \
	--seq_len 10 \
	--per_img_std false \
	--batch_size 32 \
	--n_epochs 1 \
	--run_name mnist_tan \
	--is_test True \
	--plot_every 1 \
	--save_model False \
	--n_tan_train 100
