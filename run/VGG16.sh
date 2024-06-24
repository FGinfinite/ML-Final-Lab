#!/bin/bash
set -e

batch_size=64
epochs=100
device=4
learning_rates=(0.03 0.01 0.003)
weight_decays=(0.0001 0.0003 0.001)
init_stds=(-1.0 0.01 0.03 0.1)
optimizers=("SGD" "Adam")
dataset=("cifar10" "cifar100")
models=("VGG11" "VGG13" "VGG16" "VGG19" "ResNet18" "ResNet34" "ResNet50" "ResNet101" "ResNet152")
model="VGG16"
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