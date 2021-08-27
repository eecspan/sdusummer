from ActivationPrune import *
from WeightPrune import weightPruneModelOp
import os
from Op import Op

if __name__ == '__main__':
    model_name = 'AlexNet'  # 确定模型名称
    batch_size = 1  # 确定批训练图片数目
    img_size = 227  # 确定单张图片大小
    ratio = 0.1  # 确定输入特征图剪枝比率
    epochA = 30  # 确定针对输入特征图剪枝重训练轮数或原始模型（不掺杂任何剪枝训练）轮数
    epochAW = 40  # 确定针对卷积核聚类剪枝重训练轮数
    weightParameter = (4/1)
    LinearParameter = 4
    '''
    一共设置有六种针对模型的操作
    1. operation = 'trainInitialModel'，意为训练初始模型，此时不参杂任何剪枝操作，单纯训练初始模型
    2. operation = 'onlyActivationPruneWithRetrain'，意为只针对输入特征图进行剪枝，并进行重训练
    3. operation = 'onlyWeightPruneWithRetrain'，意为只针对权重值进行聚类剪枝，并进行重训练
    4. operation = 'activationWeightPruneWithRetrain'，意为对输入特征图剪枝并进行重训练，对其生成的模型权重进行聚类剪枝并进行重训练
    5. operation = 'onlyActivationPruneTest'，意为只针对输入特征图剪枝后的模型进行inferernce，测试模型精度
    6. operation = 'activationWeightPruneTest'，意为针对输入特征图与权重聚类剪枝后的模型进行inference，测试模型精度
    '''
    operation = 'trainInitialModel'
    Op(operation,model_name,batch_size,img_size,ratio,epochA,epochAW,weightParameter,LinearParameter)














    # if not os.path.exists('./pth/'+model_name+'/ratio='+str(ratio)):  #
    #     os.makedirs('./pth/'+model_name+'/ratio='+str(ratio)+'/Activation')
    #     if patternA != 'train':
    #         os.makedirs('./pth/' + model_name + '/ratio=' + str(ratio) + '/ActivationWeight')
    #         os.makedirs('./pth/' + model_name + '/ratio=' + str(ratio) + '/Weight')
    #
    # # activationPruneModelOp(model_name, batch_size, img_size,patternA,ratio,epochA)
    # if patternA != 'train' and not(patternA == 'test' and ratio == 0):
    #     weightPruneModelOp(model_name, batch_size, img_size, ratio, patternW,epochAW,weightParameter,LinearParameter)