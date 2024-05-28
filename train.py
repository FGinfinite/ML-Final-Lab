import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from models import VGG, ResNet, BasicBlock
from dataloader import Dataloader


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


if __name__ == '__main__':
    # 参数
    batch_size = 64
    epochs = 10
    LR = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = 'cifar10'

    # 创建模型实例
    net = VGG('VGG16').to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    # 数据集
    dataloader = Dataloader(dataset, batch_size)
    trainloader, testloader = dataloader.get_loader()

    # 训练
    train(net, trainloader, testloader, criterion, optimizer, device, epochs)
