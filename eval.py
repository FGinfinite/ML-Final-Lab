import os
import torch
import numpy as np
from itertools import product
from models import select_model
from sklearn.metrics import accuracy_score, mean_squared_error
from dataloader import Dataloader 
from utils import get_device
import argparse
from tqdm import tqdm

device = get_device(0)

def load_model(model_name, dataset, model_path, device):
    model = select_model(model_name, dataset)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# 集成评判图像
def ensemble_models(model_details, dataset, test_loader, num_weights=10, device='cuda'):
    torch.cuda.empty_cache()  # 清空 GPU 内存以避免内存溢出问题
    weights_range = np.linspace(0, 1, num_weights + 1)
    best_accuracy = 0
    best_weights = None

    # 预计算所有有效的权重组合，其总和为1
    valid_weights = [weights for weights in product(weights_range, repeat=len(model_details)) if np.isclose(sum(weights), 1, atol=0.0001)]

    for weights in tqdm(valid_weights, desc="Processing weights combinations"):  
        aggregated_predictions = []
        models = []
        # 加载所有模型
        for (model_name, model_path), weight in zip(model_details, weights):
            model = load_model(model_name, dataset, model_path, device)
            models.append((model, weight))
        
        correct = 0
        total = 0

        # 使用 test_loader 进行批处理
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            predictions = [weight * model(images) for model, weight in models]
            weighted_sum = torch.stack(predictions).sum(dim=0)
            predicted_classes = torch.argmax(weighted_sum, dim=1)
            correct += (predicted_classes == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = weights

    return best_accuracy, best_weights

# 集成评判股票
def ensemble_stock(model_details, dataset, input_data, targets, num_weights=1000, device='cpu'):
    weights_range = np.linspace(0, 1, num_weights + 1)
    best_mse = float('inf')
    best_weights = None

    # 过滤出有效的权重组合
    valid_weights = [weights for weights in product(weights_range, repeat=len(model_details)) if np.abs(sum(weights) - 1) <= 0.0001]

    for weights in tqdm(valid_weights, desc="Evaluating weight combinations"):  
        predictions = []
        for (model_name, model_path), weight in zip(model_details, weights):
            model = load_model(model_name, dataset, model_path, device)
            prediction = model(input_data)
            predictions.append(weight * prediction)

        weighted_sum = torch.stack(predictions).sum(dim=0)
        mse = mean_squared_error(targets.cpu().numpy(), weighted_sum.detach().cpu().numpy())

        if mse < best_mse:
            best_mse = mse
            best_weights = weights

    return best_mse, best_weights

def main():
    parser = argparse.ArgumentParser(description='Process some dataset information.')
    parser.add_argument('--dataset', type=str, default='stock', help='Dataset to be processed')

    args = parser.parse_args()

    dataset = args.dataset  # 'stock' or 'dogs_vs_cats'
    dataloader = Dataloader(dataset)
    train_loader, test_loader = dataloader.get_loader()
    

    # 从models文件夹中找到该数据集所有对应的模型进行集成

    model_details = []
    models_dir = './models'

    if dataset == 'stock':
        suffix = 'stock.pt'
    else:
        suffix = 'dogs_vs_cats.pt'

    for file in os.listdir(models_dir):
        if file.endswith(suffix):
            model_name = file.split('_')[0] 
            model_details.append((model_name, os.path.join(models_dir, file)))

    if dataset == 'stock':
        x_train = torch.from_numpy(dataloader.x_train).type(torch.Tensor).to(device)
        x_test = torch.from_numpy(dataloader.x_test).type(torch.Tensor).to(device)
        y_train = torch.from_numpy(dataloader.y_train).type(torch.Tensor).to(device)
        y_test = torch.from_numpy(dataloader.y_test).type(torch.Tensor).to(device)
        
        input_data = x_test 
        targets = y_test  
        
        best_mse, best_weights = ensemble_stock(model_details, dataset, input_data, targets, device=device)
        print("Best MSE:", best_mse)
        best_rmse = np.sqrt(best_mse)
        print("Best RMSE:", best_rmse)
        print("Best weights:", best_weights)
        
    else:     
        best_accuracy, best_weights = ensemble_models(model_details, dataset, test_loader)
        print("Best accuracy:", best_accuracy)
        print("Best weights:", best_weights)


if __name__ == '__main__':
    main()