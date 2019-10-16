#!/bin/bash

dataset=$1
transform=$2

python ./train.py --dataset $dataset --model Conv4 --method protonet --aug_type $transform --aug_target sample
python ./save_features.py --dataset $dataset --model Conv4 --method protonet --aug_type $transform --aug_target sample
python ./test.py --dataset $dataset --model Conv4 --method protonet --aug_type $transform --aug_target sample

python ./train.py --dataset $dataset --model Conv4 --method protonet --aug_type $transform --aug_target batch
python ./save_features.py --dataset $dataset --model Conv4 --method protonet --aug_type $transform --aug_target batch
python ./test.py --dataset $dataset --model Conv4 --method protonet --aug_type $transform --aug_target batch
