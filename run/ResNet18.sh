#!/bin/bash
set -e

batch_size=64
epochs=100
device=2
learning_rates=(0.1 0.03 0.01 0.003 0.001)
weight_decays=(0.0 0.0001 0.0003 0.001 0.003)
init_stds=(-1.0)
optimizers=("SGD" "Adam")
dataset=("cifar10")
model="ResNet18"
# shellcheck disable=SC2068
for lr in ${learning_rates[@]}; do
    for wd in ${weight_decays[@]}; do
        for is in ${init_stds[@]}; do
            for opt in ${optimizers[@]}; do
                for ds in ${dataset[@]}; do
                    python train.py --batch_size $batch_size --epochs $epochs --learning_rate $lr --weight_decay $wd --init_std $is --optimizer $opt --device $device --dataset $ds --model $model
                done
            done
        done
    done
done