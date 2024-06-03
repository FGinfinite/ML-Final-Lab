import torch
import torch.nn as nn
import torch.optim as optim
from models import select_model, init_weights
from dataloader import Dataloader
from utils import save_log


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


if __name__ == '__main__':
    # 参数
    batch_size = 64
    epochs = 150
    learning_rates = [0.1, 0.03, 0.01, 0.003, 0.001]
    weight_decays = [0, 5e-4, 1e-4, 5e-5, 1e-5]
    init_stds = ['Kaiming', 0.01, 0.03, 0.1, 0.3, 1]
    criterion = nn.CrossEntropyLoss()
    optimzers = ['SGD', 'Adam']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = ['cifar10', 'cifar100']
    models = ['VGG16', 'ResNet18']

    for dataset in datasets:
        for model_name in models:
            for optimizer_name in optimzers:
                for lr in learning_rates:
                    for weight_decay in weight_decays:
                        for std in init_stds:
                            # 创建模型实例
                            net = select_model(model_name).to(device)
                            # 初始化权重
                            init_weights(net, std)

                            # 定义优化器
                            if optimizer_name == 'SGD':
                                optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
                            else:
                                optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

                            # 数据集
                            dataloader = Dataloader(dataset, batch_size)
                            train_loader, test_loader = dataloader.get_loader()
                            # 训练
                            train_loss, test_loss, train_acc, test_acc = train(net, train_loader, test_loader,
                                                                               criterion,
                                                                               optimizer, device, epochs)
                            # 保存日志
                            save_log(model_name, dataset, optimizer_name, std, lr, weight_decay, epochs,
                                     (train_loss, test_loss, train_acc, test_acc),
                                     f'logs/{model_name}_{dataset}_{optimizer_name}_{std}_{lr}_{weight_decay}.pkl')
