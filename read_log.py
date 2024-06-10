import pickle
import torch
import numpy as np
from utils import load_log

def find_best_acc(result):
    # result=(train_loss, test_loss, train_acc, test_acc)
    return max(result[3])

models = ['VGG13', 'VGG16', 'ResNet18']
dataset = 'cifar10'
optimizer = 'SGD'
init_stds = [-1.0, 0.01, 0.03, 0.1, 0.3, 1.0]
learning_rates = [0.1, 0.03, 0.01, 0.003, 0.001]
weight_decays = [0, 0.0001, 0.0003, 0.001, 0.003, 0.01]

root_path = '/root/autodl-tmp/logs/'

for model in models:
    for init_std in init_stds:


        for lr in learning_rates:
            for weight_decay in weight_decays:
                try:
                    log = load_log(root_path + f'{model}_{dataset}_{optimizer}_{init_std}_{lr}_{weight_decay}.pkl')
                    print(f'{model}_{dataset}_{optimizer}_{init_std}_{lr}_{weight_decay}.pkl')
                    print(log)
                except:
                    pass
