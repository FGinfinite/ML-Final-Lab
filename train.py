import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import select_model, init_weights
from dataloader import Dataloader 
from utils import save_log, get_device, set_seed
import argparse


def test(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    cur_test_loss = loss / len(test_loader)
    cur_test_acc = correct / total
    return cur_test_loss, cur_test_acc


def train(model, train_loader, test_loader, criterion, optimizer, device, epochs):
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            epoch_correct += (predicted == labels).sum().item()

            loss.backward()
            optimizer.step()

        cur_train_loss = epoch_loss / len(train_loader)
        cur_train_acc = epoch_correct / len(train_loader.dataset)

        train_loss.append(cur_train_loss)
        train_acc.append(cur_train_acc)

        cur_test_loss, cur_test_acc = test(model, test_loader, criterion, device)

        test_loss.append(cur_test_loss)
        test_acc.append(cur_test_acc)

        print(f'Train Loss: {cur_train_loss:.4f} Train Acc: {cur_train_acc:.4f}')
        print(f'Test Loss: {cur_test_loss:.4f} Test Acc: {cur_test_acc:.4f}')

    return train_loss, test_loss, train_acc, test_acc

def train_stock(model, x_train, y_train, optimizer, criterion, num_epochs=100):
    hist = {'loss': [], 'rmse': []}
    model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    for t in range(num_epochs):
        model.train()
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train)
        print("Epoch ", t + 1, "MSE: ", loss.item())
        hist['loss'].append(loss.item())

        rmse = torch.sqrt(loss).item()
        hist['rmse'].append(rmse)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, hist

def test_stock(model, x_test, y_test, criterion):
    model.eval()
    y_test_pred = model(x_test)
    test_loss = criterion(y_test_pred, y_test)
    rmse = torch.sqrt(test_loss).item()

    print(f"Test MSE: {test_loss.item()}")
    print(f"Test RMSE: {rmse}")

    return test_loss.item(), rmse
        

if __name__ == '__main__':
    print("---------------------")

    set_seed(42)

    paser = argparse.ArgumentParser()

    paser.add_argument('--batch_size', type=int, default=64)
    paser.add_argument('--epochs', type=int, default=100)
    paser.add_argument('--learning_rate', type=float, default=0.01)
    paser.add_argument('--weight_decay', type=float, default=0)
    paser.add_argument('--init_std', type=float, default=-1.0)
    paser.add_argument('--optimizer', type=str, default='SGD')
    paser.add_argument('--criterion', type=str, default='CrossEntropyLoss')
    paser.add_argument('--device', type=int, default=0)
    paser.add_argument('--dataset', type=str, default='cifar10')
    paser.add_argument('--model', type=str, default='VGG16')
    paser.add_argument('--data_augmentation', action='store_true')

    args = paser.parse_args()

    print("args:", args)


    device = get_device(args.device)

    net = select_model(args.model,args.dataset).to(device)


    init_weights(net, args.init_std)

    epochs = args.epochs

    # 损失函数
    if args.criterion == 'MSE':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # 定义优化器
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 数据集：目前并没有划分验证集val_loader，只有训练集train_loader和测试集test_loader。先以大规模实验获取最佳超参数的范围，再划分验证集，最后在验证集上更细致地调整超参数以获取最佳模型。
    
    dataloader = Dataloader(args.dataset, args.batch_size, args.data_augmentation)
    train_loader, test_loader = dataloader.get_loader()
    
    # 训练
    if args.dataset == 'stock':
        x_train = torch.from_numpy(dataloader.x_train).type(torch.Tensor)
        x_test = torch.from_numpy(dataloader.x_test).type(torch.Tensor)
        y_train = torch.from_numpy(dataloader.y_train).type(torch.Tensor)
        y_test = torch.from_numpy(dataloader.y_test).type(torch.Tensor)
        
        model_trained, training_history = train_stock(net, x_train, y_train, optimizer, criterion, args.epochs)
        test_mse, test_rmse = test_stock(model_trained, x_test, y_test, criterion)
        
        if args.data_augmentation:
            log_path = f'logs-augs/{args.model}_stock_{args.optimizer}_{args.init_std}_{args.learning_rate}_{args.weight_decay}.pkl'
        else:
            log_path = f'logs/{args.model}_stock_{args.optimizer}_{args.init_std}_{args.learning_rate}_{args.weight_decay}.pkl'
        
        save_log(args.model, 'stock', args.optimizer, args.init_std, args.learning_rate, args.weight_decay,
                args.epochs, {'train_loss': training_history['loss'], 'train_rmse': training_history['rmse']}, log_path)
    else:
        train_loss, test_loss, train_acc, test_acc = train(net, train_loader, test_loader,
                                                        criterion,
                                                        optimizer, device, epochs)
        if args.data_augmentation:
            log_path=f'logs-augs/{args.model}_{args.dataset}_{args.optimizer}_{args.init_std}_{args.learning_rate}_{args.weight_decay}.pkl'
        else:
            log_path=f'logs/{args.model}_{args.dataset}_{args.optimizer}_{args.init_std}_{args.learning_rate}_{args.weight_decay}.pkl'

        save_log(args.model, args.dataset, args.optimizer, args.init_std, args.learning_rate, args.weight_decay,
                args.epochs,
                (train_loss, test_loss, train_acc, test_acc),log_path)
