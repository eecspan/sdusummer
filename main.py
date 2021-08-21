from ActivationPrune import *


if __name__ == '__main__':
    weight_file_path = './pth/LeNet/LeNet.pth'
    model_name = 'LeNet'
    batch_size = 64
    img_size = 32
    pattern = 'test'  #pattern='test' or pattern='train'
    ratio = 0.4  #ratio=0为训练模式
    activationPruneModelOp(model_name, weight_file_path, batch_size, img_size,pattern,ratio)