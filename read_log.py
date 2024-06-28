import pandas
import pickle
import torch
import numpy as np
from utils import load_log


def find_best_acc(result):
    # result=(train_loss, test_loss, train_acc, test_acc)
    return max(result[3])


models = ["VGG16", "ResNet18"]
dataset = "cifar10"
optimizer = "Adam"
init_stds = [-1.0]
learning_rates = [0.1, 0.03, 0.01, 0.003, 0.001]
weight_decays = [0.0, 0.0001, 0.0003, 0.001, 0.003]

root_path = "logs/"

df_name = [
    [None for _ in range(len(weight_decays) + 1)]
    for _ in range(len(learning_rates) + 1)
]

for i, learning_rate in enumerate(learning_rates):
    df_name[i + 1][0] = 'learning_rate: {}'.format(learning_rate)
for i, weight_decay in enumerate(weight_decays):
    df_name[0][i + 1] = 'weight_decay: {}'.format(weight_decay)


for model in models:
    tables=pandas.DataFrame([])
    for init_std in init_stds:
        # 深拷贝df
        df = df_name.copy()
        df[0][0] = 'init_std: {}'.format(init_std)
        for i, learning_rate in enumerate(learning_rates):
            for j, weight_decay in enumerate(weight_decays):
                log_path = root_path + f"{model}_{dataset}_{optimizer}_{init_std}_{learning_rate}_{weight_decay}.pkl"
                try:
                    log = load_log(log_path)
                    df[i + 1][j + 1] = find_best_acc(log['result'])
                except:
                    df[i + 1][j + 1] = 'N/A'

        # 将df拼接到tables
        df = pandas.DataFrame(df)
        # 空出新的一列再拼接
        tables = pandas.concat([tables, df], axis=0)
    tables.to_csv(f"csv/{model}_{dataset}_{optimizer}.csv")
        

