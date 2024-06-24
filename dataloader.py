from turtle import pd
import pandas as panda
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor, RandomHorizontalFlip, RandomAffine
import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class Dataloader:
    def __init__(self, dataset_name, batch_size=8, augmentation=False):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        if not augmentation:
            if dataset_name == 'cifar10':
                self.transform = Compose([
                    ToTensor(),
                    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                # 加载CIFAR-10数据集
                train_set = CIFAR10(root='./dataset', train=True, download=True, transform=self.transform)
                self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

                test_set = CIFAR10(root='./dataset', train=False, download=True, transform=self.transform)
                self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=2)

            elif dataset_name == 'cifar100':
                self.transform = Compose([
                    ToTensor(),
                    Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ])
                # 加载CIFAR-100数据集
                train_set = CIFAR100(root='./dataset', train=True, download=True, transform=self.transform)
                self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

                test_set = CIFAR100(root='./dataset', train=False, download=True, transform=self.transform)
                self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=2)

            elif dataset_name == 'dogs_vs_cats':
                split_result=split_dataset('./dataset/DVG', 0.2)
                
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                train_set = CatsAndDogsDataset(data_folder='./dataset/DVG', split_result=split_result, train=True, transform=self.transform)
                self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

                test_set=CatsAndDogsDataset(data_folder='./dataset/DVG', split_result=split_result, train=False, transform=self.transform)
                self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=2)
            
            elif dataset_name == 'stock':
                # 使用StockDataset类加载训练和测试数据
                train_set = StockDataset(filepath='./dataset/stock/stock.csv', lookback=10, train=True)
                self.x_train, self.y_train = train_set.x, train_set.y
                self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

                test_set = StockDataset(filepath='./dataset/stock/stock.csv', lookback=10, train=False)
                self.x_test, self.y_test = test_set.x, test_set.y
                self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=2)
                
            
            
            else:
                raise ValueError('Dataset not supported')
            
        else:
            
            if dataset_name=='cifar10':
                self.train_transform = Compose([
                    transforms.RandomRotation(degrees=15),  # 随机旋转±15度
                    transforms.RandomHorizontalFlip(p=0.5),  # 50%的概率进行水平翻转
                    transforms.RandomResizedCrop(size=(32,32), scale=(0.8, 1.0), ratio=(0.75, 1.33)),  # 随机裁剪并调整大小
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 色彩抖动
                    ToTensor(),
                    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                # 加载CIFAR-10数据集
                train_set = CIFAR10(root='./dataset', train=True, download=True, transform=self.train_transform)
                self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)
                
                self.test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

                test_set = CIFAR10(root='./dataset', train=False, download=True, transform=self.test_transform)
                self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=2)
            
            elif dataset_name=='cifar100':
                self.train_transform = Compose([
                    transforms.RandomRotation(degrees=15),  # 随机旋转±15度
                    transforms.RandomHorizontalFlip(p=0.5),  # 50%的概率进行水平翻转
                    transforms.RandomResizedCrop(size=(32,32), scale=(0.8, 1.0), ratio=(0.75, 1.33)),  # 随机裁剪并调整大小
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 色彩抖动
                    ToTensor(),
                    Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ])
                # 加载CIFAR-100数据集
                train_set = CIFAR100(root='./dataset', train=True, download=True, transform=self.train_transform)
                self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)
                
                self.test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ])

                test_set = CIFAR100(root='./dataset', train=False, download=True, transform=self.test_transform)
                self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=2)
            
            elif dataset_name == 'dogs_vs_cats':
                split_result=split_dataset('./dataset/DVG', 0.2)
                
                self.train_transform = transforms.Compose([
                    transforms.RandomRotation(degrees=15),  # 随机旋转±15度
                    transforms.RandomHorizontalFlip(p=0.5),  # 50%的概率进行水平翻转
                    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.33)),  # 随机裁剪并调整大小
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 色彩抖动
                    transforms.ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                train_set = CatsAndDogsDataset(data_folder='./dataset/DVG', split_result=split_result, train=True, transform=self.train_transform)
                self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)
                
                
                self.test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                test_set=CatsAndDogsDataset(data_folder='./dataset/DVG', split_result=split_result, train=False, transform=self.test_transform)
                self.test_set = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=2)
                
            elif dataset_name == 'stock':
                # 使用StockDataset类加载训练和测试数据
                train_set = StockDataset(filepath='./dataset/stock/stock.csv', lookback=10, train=True)
                self.x_train, self.y_train = train_set.x, train_set.y
                self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

                test_set = StockDataset(filepath='./dataset/stock/stock.csv', lookback=10, train=False)
                self.x_test, self.y_test = test_set.x, test_set.y
                self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=2)
                
            else:
                raise ValueError('Dataset not supported')
                
                
                

    def get_loader(self):
        return self.train_loader, self.test_loader
    
def split_dataset(data_folder, test_ratio):
    # 获取所有图片文件名
    all_files = os.listdir(data_folder)
    cat_files = [f for f in all_files if f.startswith('cat')]
    dog_files = [f for f in all_files if f.startswith('dog')]

    # 打乱文件名数组
    random.shuffle(cat_files)
    random.shuffle(dog_files)

    # 计算测试集大小
    num_test_cat = int(len(cat_files) * test_ratio)
    num_test_dog = int(len(dog_files) * test_ratio)

    # 划分训练集和测试集
    train_cat_files = cat_files[num_test_cat:]
    test_cat_files = cat_files[:num_test_cat]
    train_dog_files = dog_files[num_test_dog:]
    test_dog_files = dog_files[:num_test_dog]

    return {
        'train_cat': train_cat_files,
        'test_cat': test_cat_files,
        'train_dog': train_dog_files,
        'test_dog': test_dog_files
    }

class CatsAndDogsDataset(Dataset):
    def __init__(self, data_folder, split_result,train, transform=None, test_ratio=0.2):
        self.data_folder = data_folder
        self.file_names = split_result['train_cat'] + split_result['train_dog'] if train else split_result['test_cat'] + split_result['test_dog']
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_folder, self.file_names[idx])
        image = Image.open(img_name)
        label = 0 if self.file_names[idx].startswith('cat') else 1

        if self.transform:
            image = self.transform(image)

        return image, label
    
class StockDataset(Dataset):
    def __init__(self, filepath, lookback, train=True):
        self.data = panda.read_csv(filepath)
        self.data = self.data.sort_values('date')
        
        # 选择特征
        features = ['open', 'high', 'low', 'close', 'volume', 'money', 'change']
        self.data = self.data[features]
        
        # 归一化
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = self.scaler.fit_transform(self.data)
        
        # 数据分割
        self.x, self.y = self.split_data(data_scaled, lookback, train)
    
    def split_data(self, stock, lookback, train):
        data = []
        for index in range(len(stock) - lookback):
            data.append(stock[index: index + lookback])
        data = np.array(data)
        
        # 训练测试分割
        test_set_size = int(np.round(0.2 * data.shape[0]))
        train_set_size = data.shape[0] - test_set_size
        
        if train:
            data = data[:train_set_size]
        else:
            data = data[train_set_size:]
        
        x = data[:, :-1, :]
        y = data[:, -1, :]
        
        return x, y

    def __getitem__(self, index):
        return torch.tensor(self.x[index], dtype=torch.float), torch.tensor(self.y[index], dtype=torch.float)

    def __len__(self):
        return self.x.shape[0]