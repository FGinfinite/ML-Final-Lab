import pickle


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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
