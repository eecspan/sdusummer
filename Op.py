from ActivationPrune import activationPruneModelOp
from WeightPrune import weightPruneModelOp
import os
def makeDir(model_name,ratio,patternA):
    if not os.path.exists('./pth/' + model_name + '/ratio=' + str(ratio)):  #
        os.makedirs('./pth/' + model_name + '/ratio=' + str(ratio) + '/Activation')
        if patternA != 'train':
            os.makedirs('./pth/' + model_name + '/ratio=' + str(ratio) + '/ActivationWeight')
            os.makedirs('./pth/' + model_name + '/ratio=' + str(ratio) + '/Weight')

def Op(operation,model_name,batch_size,img_size,ratio,epochA,epochAW,weightParameter,LinearParameter):
    if operation == 'trainInitialModel':  # 训练初始模型
        patternA = 'train'
        ratio = 0
        makeDir(model_name,ratio,patternA)
        activationPruneModelOp(model_name, batch_size, img_size,patternA,ratio,epochA)

    if operation == 'onlyActivationPruneWithRetrain':  # 只进行输入特征图的剪枝，不进行权重的聚类剪枝
        patternA = 'retrain'
        makeDir(model_name,ratio,patternA)
        activationPruneModelOp(model_name, batch_size, img_size,patternA,ratio,epochA)

    if operation == 'onlyWeightPruneWithRetrain':
        patternA = 'test'
        patternW = 'train'
        ratio = 0
        makeDir(model_name,ratio,patternA)
        weightPruneModelOp(model_name, batch_size, img_size, ratio, patternW, epochAW, weightParameter,LinearParameter)

    if operation == 'activationWeightPruneWithRetrain':
        patternA = 'retrain'
        patternW = 'retrain'
        makeDir(model_name, ratio, patternA)
        activationPruneModelOp(model_name, batch_size, img_size, patternA, ratio, epochA)
        weightPruneModelOp(model_name, batch_size, img_size, ratio, patternW, epochAW, weightParameter, LinearParameter)

    if operation == 'onlyActivationPruneTest':
        patternA = 'test'
        makeDir(model_name, ratio, patternA)
        activationPruneModelOp(model_name, batch_size, img_size, patternA, ratio, epochA)

    if operation == 'activationWeightPruneTest':
        patternA = 'test'
        patternW = 'test'
        makeDir(model_name, ratio, patternA)
        weightPruneModelOp(model_name, batch_size, img_size, ratio, patternW, epochAW, weightParameter, LinearParameter)

