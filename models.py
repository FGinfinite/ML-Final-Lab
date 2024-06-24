import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch.einops import rearrange
from torch.nn import Transformer

from utils import get_device


def select_model(model_name, dataset):
    if dataset == 'cifar10':
        img_size = 32
        num_classes = 10
    elif dataset == 'cifar100':
        img_size = 32
        num_classes = 100
    else:
        raise ValueError('Dataset not supported')

    if model_name == 'VGG11':
        return VGG('VGG11', num_classes, img_size)
    elif model_name == 'VGG13':
        return VGG('VGG13', num_classes, img_size)
    elif model_name == 'VGG16':
        return VGG('VGG16', num_classes, img_size)
    elif model_name == 'VGG19':
        return VGG('VGG19', num_classes, img_size)

    elif model_name == 'ResNet18':
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, img_size)
    elif model_name == 'ResNet34':
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, img_size)
    elif model_name == 'ResNet50':
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, img_size)
    elif model_name == 'ResNet101':
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, img_size)
    elif model_name == 'ResNet152':
        return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, img_size)

    else:
        raise ValueError('Model name not found')


def init_weights(model, std):
    if std != -1.0:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, std)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


vgg_cfgs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, image_size=224):
        super(VGG, self).__init__()
        self.model_name = vgg_name
        self.features = self._make_layers(vgg_cfgs[vgg_name])
        # 计算原始输入图像的尺寸经过卷积层和池化层后的尺寸，解决维度不匹配问题
        self.image_size = self.caculate_image_dimension(image_size)
        self.classifier = nn.Linear(512 * self.image_size * self.image_size, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def caculate_image_dimension(self, image_size):
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                image_size = int((image_size - layer.kernel_size[0] + 2 * layer.padding[0]) / layer.stride[0] + 1)
            elif isinstance(layer, nn.MaxPool2d):
                image_size = int((image_size - layer.kernel_size) / layer.stride + 1)
        return image_size


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_size=224):
        super(ResNet, self).__init__()

        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.image_size = self.calculate_image_dimension(image_size)
        self.linear = nn.Linear(512 * block.expansion * self.image_size * self.image_size, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def calculate_image_dimension(self, image_size):
        # 计算第一个卷积层对图像尺寸的影响
        image_size = int((image_size - 3 + 2 * 1) / 1 + 1)  # 对应conv1的kernel_size, padding, stride

        # 模拟每层的尺寸变化
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for sub_layer in layer:
                if isinstance(sub_layer, BasicBlock) or isinstance(sub_layer, Bottleneck):
                    stride = sub_layer.conv1.stride[0]
                    image_size = int((image_size - 3 + 2 * 1) / stride + 1)  # 假设每个卷积层都使用3x3卷积

        # 最终的平均池化层
        image_size = int((image_size - 4) / 4 + 1)  # 对应平均池化层的kernel_size和stride

        return image_size
