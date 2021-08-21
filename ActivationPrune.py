import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import time
from model import *
from train import *
import random
# from .model import ResNetBasicBlock

from math import sqrt
import copy
from time import time
from Conv2dNew import Execution

class Conv2dTest(nn.Conv2d):
    def __init__(self,
                 ratio,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 ):
        super(Conv2dTest, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias, padding_mode)
        self.ratio = ratio
    def forward(self, input):
        E = Execution(self.ratio)
        output = E.conv2d(input, self.weight, self.bias, self.stride, self.padding)
        return output

class LinearTest(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 ):
        super(LinearTest, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        output = F.linear(input, self.weight, self.bias)
        return output

def prepare(model, ratio,inplace=False):
    # move intpo prepare
    def addActivationPruneOp(module):
        nonlocal layer_cnt
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                p_name = str(layer_cnt)
                activationPruneConv = Conv2dTest(
                    ratio,
                    child.in_channels,
                    child.out_channels, child.kernel_size, stride=child.stride, padding=child.padding,
                    dilation=child.dilation, groups=child.groups, bias=(child.bias is not None),
                    padding_mode=child.padding_mode
                )
                if child.bias is not None:
                    activationPruneConv.bias = child.bias
                activationPruneConv.weight = child.weight
                module._modules[name] = activationPruneConv
                layer_cnt += 1
            elif isinstance(child, nn.Linear):
                p_name = str(layer_cnt)
                activationPruneLinear = LinearTest(
                     child.in_features, child.out_features,
                    bias=(child.bias is not None)
                )
                if child.bias is not None:
                    activationPruneLinear.bias = child.bias
                activationPruneLinear.weight = child.weight
                module._modules[name] = activationPruneLinear
                layer_cnt += 1
            else:
                addActivationPruneOp(child)  # 这是用来迭代的，Maxpool层的功能是不变的
    layer_cnt = 0
    if not inplace:
        model = copy.deepcopy(model)
    addActivationPruneOp( model)  # 为每一层添加量化操作
    return model

def getPruneModel(model_name, weight_file_path,pattern,ratio):
    if model_name == 'LeNet':
        model_orign = getLeNet()  # 加载原始模型框架
    elif model_name == 'AlexNet':
        model_orign = getAlexnet()

    if pattern == 'test':
        model_orign.load_state_dict(torch.load(weight_file_path))  # 原始模型框架加载模型信息
    activationPruneModel = prepare(model_orign,ratio)  # 将原始模型转化成量化后的模型，即给每一个卷积层和线形层增加量化剪枝操作

    return activationPruneModel

def activationPruneModelOp(model_name, weight_file_path, batch_size, img_size,pattern,ratio):
    '''
    :param model_name: 要训练的模型名称
    :param weight_file_path: 权重文件的地址
    :param batch_size: 训练时一个batch的大小
    :param img_size: 数据集中图片大小的要求
    :param pattern: 模式选择，是训练模式还是推理模式
    :return:
    '''
    if model_name == 'VGG16' or model_name == 'AlexNet' or model_name == 'ResNet' or model_name == 'vgg16_thu' or model_name == 'SqueezeNet':
        dataloaders, dataset_sizes = load_cifar10(batch_size=batch_size, pth_path='./data',
                                                  img_size=img_size)  # 确定数据集
    elif model_name == 'LeNet':
        dataloaders, dataset_sizes = load_mnist(batch_size=batch_size, path='./data', img_size=img_size)

    activationPruneModel = getPruneModel(model_name, weight_file_path,pattern,ratio)
    criterion = nn.CrossEntropyLoss()
    if pattern == 'train':
        optimizer = optim.SGD(activationPruneModel.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)  # 设置学习率下降策略
        train_model_jiang(activationPruneModel, dataloaders, dataset_sizes, criterion=criterion, optimizer=optimizer, name='SqueezeNet_5',
                          scheduler=scheduler, num_epochs=30, rerun=False)  # 进行模型的训练
    elif pattern == 'test':
        test_model(activationPruneModel,dataloaders, dataset_sizes,criterion=criterion)
