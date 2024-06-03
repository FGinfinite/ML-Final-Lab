import pickle
import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_log(model_name, dataset_name, optimizer_name, std, lr, weight_decay, epochs, result, save_path):
    log = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'optimizer': optimizer_name,
        'std': std,
        'lr': lr,
        'weight_decay': weight_decay,
        'epochs': epochs,
        'result': result
    }
    with open(save_path, 'wb') as f:
        pickle.dump(log, f)


def load_log(load_path):
    with open(load_path, 'rb') as f:
        log = pickle.load(f)
    return log


def count_parameters(model, in_mb=False):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if in_mb:
        return params * 4 / (1024 ** 2)
    else:
        return params


def get_device(gpu):
    device = torch.device('cpu')
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:' + str(gpu))
    return device
