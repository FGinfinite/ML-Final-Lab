import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, RandomHorizontalFlip, RandomAffine


class Dataloader:
    def __init__(self, dataset_name, batch_size, augmentation=False):
        self.dataset_name = dataset_name
        self.batch_size = batch_size

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
                train_set = CIFAR10(root='./dataset', train=True, download=True, transform=self.transform)
                self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

                test_set = CIFAR10(root='./dataset', train=False, download=True, transform=self.transform)
                self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=2)

            else:
                raise ValueError('Dataset not supported')

    def get_loader(self):
        return self.train_loader, self.test_loader
