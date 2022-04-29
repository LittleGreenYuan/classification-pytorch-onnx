# -*- coding: utf-8 -*-
"""
@author: Greenyuan
"""
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets import get_model_from_name
from utils.callbacks import LossHistory
from utils.dataloader import DataGenerator, detection_collate
from utils.utils import (get_classes, get_lr_scheduler, set_optimizer_lr,
                         weights_init)
from utils.utils_fit import fit_one_epoch


#torch.save(model.state_dict(), 'model_figure_classfiy.pth')
#----------------------------------------------------#
#   是否使用Cuda
#   没有GPU可以设置成False
#----------------------------------------------------#
Cuda            = False

#----------------------------------------------------#
#   训练自己的数据集的时候一定要注意修改classes_path
#   修改成自己对应的种类的txt
#----------------------------------------------------#
classes_path    = 'model_data/cls_classes.txt' 
#----------------------------------------------------#
#   输入的图片大小
#----------------------------------------------------#
input_shapes     = [224, 224]
#------------------------------------------------------#
#   所用模型种类：
#   mobilenet、resnet50、vgg16、vit
#------------------------------------------------------#
backbone        = "mobilenet"
#----------------------------------------------------------------------------------------------------------------------------#
#   是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
#   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
#   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
#   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
#----------------------------------------------------------------------------------------------------------------------------#
pretrained      = True
#----------------------------------------------------------------------------------------------------------------------------#
#   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
#   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
#   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
#
#   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
#   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
#   
#   当model_path = ''的时候不加载整个模型的权值。
#
#   此处使用的是整个模型的权重，因此是在train.py进行加载的，pretrain不影响此处的权值加载。
#   如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，pretrain = True，此时仅加载主干。
#   如果想要让模型从0开始训练，则设置model_path = ''，pretrain = Fasle，此时从0开始训练。
#----------------------------------------------------------------------------------------------------------------------------#
model_path      = ""
    
#----------------------------------------------------------------------------------------------------------------------------#
#   训练分为两个阶段，分别是冻结阶段和解冻阶段。设置冻结阶段是为了满足机器性能不足的同学的训练需求。
#   冻结训练需要的显存较小，显卡非常差的情况下，可设置Freeze_Epoch等于UnFreeze_Epoch，此时仅仅进行冻结训练。
#      
#   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
#   （一）从整个模型的预训练权重开始训练： 
#       Adam：
#           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-3。（冻结）
#           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-3。（不冻结）
#       SGD：
#           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2。（冻结）
#           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2。（不冻结）
#       其中：UnFreeze_Epoch可以在100-300之间调整。
#   （二）从0开始训练：
#       Adam：
#           Init_Epoch = 0，UnFreeze_Epoch = 300，Unfreeze_batch_size >= 16，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-3。（不冻结）
#       SGD：
#           Init_Epoch = 0，UnFreeze_Epoch = 300，Unfreeze_batch_size >= 16，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2。（不冻结）
#       其中：UnFreeze_Epoch尽量不小于300。
#   （三）batch_size的设置：
#       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
#       受到BatchNorm层影响，batch_size最小为2，不能为1。
#       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。
#----------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------#
#   冻结阶段训练参数
#   此时模型的主干被冻结了，特征提取网络不发生改变
#   占用的显存较小，仅对网络进行微调
#   Init_Epoch          模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
#                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
#                       会跳过冻结阶段，直接从60代开始，并调整对应的学习率。
#                       （断点续练时使用）
#   Freeze_Epoch        模型冻结训练的Freeze_Epoch
#                       (当Freeze_Train=False时失效)
#   Freeze_batch_size   模型冻结训练的batch_size
#                       (当Freeze_Train=False时失效)
#------------------------------------------------------------------#
Init_Epoch          = 0
Freeze_Epoch        = 50
Freeze_batch_size   = 32
#------------------------------------------------------------------#
#   解冻阶段训练参数
#   此时模型的主干不被冻结了，特征提取网络会发生改变
#   占用的显存较大，网络所有的参数都会发生改变
#   UnFreeze_Epoch          模型总共训练的epoch
#   Unfreeze_batch_size     模型在解冻后的batch_size
#------------------------------------------------------------------#
UnFreeze_Epoch      = 1#100
Unfreeze_batch_size = 1#32
#------------------------------------------------------------------#
#   Freeze_Train    是否进行冻结训练
#                   默认先冻结主干训练后解冻训练。
#------------------------------------------------------------------#
Freeze_Train        = True

#------------------------------------------------------------------#
#   其它训练参数：学习率、优化器、学习率下降有关
#------------------------------------------------------------------#
#------------------------------------------------------------------#
#   Init_lr         模型的最大学习率
#                   当使用Adam优化器时建议设置  Init_lr=1e-3
#                   当使用SGD优化器时建议设置   Init_lr=1e-2
#   Min_lr          模型的最小学习率，默认为最大学习率的0.01
#------------------------------------------------------------------#
Init_lr             = 1e-2
Min_lr              = Init_lr * 0.01
#------------------------------------------------------------------#
#   optimizer_type  使用到的优化器种类，可选的有adam、sgd
#                   当使用Adam优化器时建议设置  Init_lr=1e-3
#                   当使用SGD优化器时建议设置   Init_lr=1e-2
#   momentum        优化器内部使用到的momentum参数
#   weight_decay    权值衰减，可防止过拟合
#                   使用adam优化器时会有错误，建议设置为0
#------------------------------------------------------------------#
optimizer_type      = "sgd"
momentum            = 0.9
weight_decay        = 5e-4
#------------------------------------------------------------------#
#   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
#------------------------------------------------------------------#
lr_decay_type       = "cos"
#------------------------------------------------------------------#
#   save_period     多少个epoch保存一次权值，默认每个世代都保存
#------------------------------------------------------------------#
save_period         = 1
#------------------------------------------------------------------#
#   save_dir        权值与日志文件保存的文件夹
#------------------------------------------------------------------#
save_dir            = 'logs'
#------------------------------------------------------------------#
#   num_workers     用于设置是否使用多线程读取数据
#                   开启后会加快数据读取速度，但是会占用更多内存
#                   内存较小的电脑可以设置为2或者0  
#------------------------------------------------------------------#
num_workers         = 4

#------------------------------------------------------#
#   train_annotation_path   训练图片路径和标签
#   test_annotation_path    验证图片路径和标签（使用测试集代替验证集）
#------------------------------------------------------#
train_annotation_path   = "cls_train.txt"
test_annotation_path    = 'cls_test.txt'

#------------------------------------------------------#
#   获取classes
#------------------------------------------------------#
class_names, num_classes = get_classes(classes_path)


model_test = get_model_from_name[backbone](num_classes = num_classes, pretrained = pretrained)



model_statedict = torch.load("model_data/mobilenet_catvsdog.pth",map_location=lambda storage,loc:storage)   #导入Gpu训练模型，导入为cpu格式
model_test.load_state_dict(model_statedict)  #将参数放入model_test中
model_test.eval()  # 测试，看是否报错
#下面开始转模型，cpu格式下
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy_input = torch.randn(1,3, 224, 224,device=device)
input_names = ["input"]
output_names = ["output"]
torch.onnx.export(model_test, dummy_input, "model_.onnx", opset_version=9, verbose=False, output_names=["output"])


#用于验证onnx模型是否能够正确使用
import cv2 as cv
import numpy as np

def img_process(image):
    mean = np.array([0.5,0.5,0.5],dtype=np.float32).reshape(1,1,3)
    std = np.array([0.5,0.5,0.5],dtype=np.float32).reshape(1,1,3)
    new_img = ((image/255. -mean)/std).astype(np.float32)
    return new_img

img = cv.imread("cat.jpg")
img_t = cv.resize(img,(224,224))    #将图片改为模型适用的尺寸
img_t = img_process(img_t)
#img_t = np.transpose(img_t,[2,0,1])
#img_t = img_t[np.newaxis,:]   #扩展一个新维度

layerNames = ["output"]   # 这里的输出的名称应该于前面的转模型时候定义的一致
blob=cv.dnn.blobFromImage(img_t,scalefactor=1.0,swapRB=True,crop=False)  # 将image转化为 1x3x64x64 格式输入模型中
net = cv.dnn.readNetFromONNX("model_.onnx")
net.setInput(blob)
outs = net.forward(layerNames)
print(outs)



'''
torch.onnx.export(model, args, f, export_params=True, verbose=False, training=False, 
                  input_names=None, output_names=None, aten=False, export_raw_ir=False, 
                  operator_export_type=None, opset_version=None, _retain_param_name=True, 
                  do_constant_folding=False, example_outputs=None, strip_doc_string=True, 
                  dynamic_axes=None, keep_initializers_as_inputs=None)
参数介绍：
    model (torch.nn.Module) – 要导出的模型.
    args (tuple of arguments) – 模型的输入, 任何非Tensor参数都将硬编码到导出的模型中；
    任何Tensor参数都将成为导出的模型的输入，并按照他们在args中出现的顺序输入。因为export运行模型，
    所以我们需要提供一个输入张量x。只要是正确的类型和大小，其中的值就可以是随机的。请注意，除非指定为动态轴，
    否则输入尺寸将在导出的ONNX图形中固定为所有输入尺寸。在此示例中，我们使用输入batch_size 1导出模型，
    但随后dynamic_axes 在torch.onnx.export()。因此，导出的模型将接受大小为[batch_size，3、100、100]的输入，
    其中batch_size可以是可变的。
    f - 保存后的onnx文件名
    export_params (bool, default True) – 如果指定为True或默认, 参数也会被导出. 如果你要导出一个没训练过的就设为 False
    verbose (bool, default False) - 如果指定，我们将打印出一个导出轨迹的调试描述
    training (bool, default False) - 在训练模式下导出模型。目前，ONNX导出的模型只是为了做推断，所以你通常不需要将其设置为True
    input_names (list of strings, default empty list) – 按顺序分配名称到图中的输入节点
    output_names (list of strings, default empty list) –按顺序分配名称到图中的输出节点

'''
