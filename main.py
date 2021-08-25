from ActivationPrune import *
from WeightPrune import weightPruneModelOp
import os

if __name__ == '__main__':
    model_name = 'LeNet'
    batch_size = 64
    img_size = 32
    ratio = 0.2
    epochA = 10
    epochAW = 40
    patternA = 'retrain'
    patternW = 'ratrain'
    weightParameter = (4/3)
    LinearParameter = 4
    if not os.path.exists('./pth/'+model_name+'/ratio='+str(ratio)):
        os.makedirs('./pth/'+model_name+'/ratio='+str(ratio)+'/Activation')
        if patternA != 'train':
            os.makedirs('./pth/' + model_name + '/ratio=' + str(ratio) + '/ActivationWeight')

    activationPruneModelOp(model_name, batch_size, img_size,patternA,ratio,epochA)
    if patternA != 'train' and not(patternA == 'test' and ratio == 0):
        weightPruneModelOp(model_name, batch_size, img_size, ratio, patternW,epochAW,weightParameter,LinearParameter)