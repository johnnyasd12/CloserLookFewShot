#!/bin/bash

dataset=$1
# export dataset
# echo $dataset
# python ./yee.py --dataset $dataset

python ./train.py --dataset $dataset --model Conv4 --method protonet --train_aug
python ./save_features.py --dataset $dataset --model Conv4 --method protonet --train_aug
python ./test.py --dataset $dataset --model Conv4 --method protonet --train_aug

python ./train.py --dataset $dataset --model Conv4 --method protonet --train_aug --recons_decoder Conv --recons_lambda 0.1 --stop_epoch 500
python ./save_features.py --dataset $dataset --model Conv4 --method protonet --train_aug --recons_decoder Conv --recons_lambda 0.1
python ./test.py --dataset $dataset --model Conv4 --method protonet --train_aug --recons_decoder Conv --recons_lambda 0.1

python ./train.py --dataset $dataset --model Conv4 --method protonet --train_aug --recons_decoder Conv --recons_lambda 1 --stop_epoch 500
python ./save_features.py --dataset $dataset --model Conv4 --method protonet --train_aug --recons_decoder Conv --recons_lambda 1
python ./test.py --dataset $dataset --model Conv4 --method protonet --train_aug --recons_decoder Conv --recons_lambda 1

python ./train.py --dataset $dataset --model Conv4 --method protonet --train_aug --recons_decoder Conv --recons_lambda 10.0 --stop_epoch 500
python ./save_features.py --dataset $dataset --model Conv4 --method protonet --train_aug --recons_decoder Conv --recons_lambda 10.0
python ./test.py --dataset $dataset --model Conv4 --method protonet --train_aug --recons_decoder Conv --recons_lambda 10.0

python ./train.py --dataset $dataset --model Conv4 --method protonet --train_aug --recons_decoder HiddenConv --recons_lambda 0.1 --stop_epoch 500
python ./save_features.py --dataset $dataset --model Conv4 --method protonet --train_aug --recons_decoder HiddenConv --recons_lambda 0.1
python ./test.py --dataset $dataset --model Conv4 --method protonet --train_aug --recons_decoder HiddenConv --recons_lambda 0.1

python ./train.py --dataset $dataset --model Conv4 --method protonet --train_aug --recons_decoder HiddenConv --recons_lambda 1 --stop_epoch 500
python ./save_features.py --dataset $dataset --model Conv4 --method protonet --train_aug --recons_decoder HiddenConv --recons_lambda 1
python ./test.py --dataset $dataset --model Conv4 --method protonet --train_aug --recons_decoder HiddenConv --recons_lambda 1

python ./train.py --dataset $dataset --model Conv4 --method protonet --train_aug --recons_decoder HiddenConv --recons_lambda 10.0 --stop_epoch 500
python ./save_features.py --dataset $dataset --model Conv4 --method protonet --train_aug --recons_decoder HiddenConv --recons_lambda 10.0
python ./test.py --dataset $dataset --model Conv4 --method protonet --train_aug --recons_decoder HiddenConv --recons_lambda 10.0

