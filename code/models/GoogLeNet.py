import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# 基础卷积层
class BasicConv(nn.Module):
    def __init__(self,inchannel,outchannel,**kwargs):
        super(BasicConv,self).__init__()

        self.conv=nn.Conv2d(inchannel,outchannel,**kwargs)

        self.relu=nn.ReLU(inplace=True)


    def forward(self,x):
        out=self.conv(x)
        out=self.relu(out)
        return out

# Inception模块
class Inception(nn.Module):
    def __init__(self,inchannel,ch1x1_out,ch3x3_in,ch3x3_out,ch5x5_in,ch5x5_out,pool):
        super(Inception,self).__init__()

        # 单独1x1卷积
        self.branch1=BasicConv(inchannel,ch1x1_out,kernel_size=1)

        # 1x1+3x3卷积
        self.branch2=nn.Sequential(
            BasicConv(inchannel,ch3x3_in,kernel_size=1),
            BasicConv(ch3x3_in,ch3x3_out,kernel_size=3,padding=1),
        )

        # 1x1+5x5卷积
        self.branch3=nn.Sequential(
            BasicConv(inchannel,ch5x5_in,kernel_size=1),
            BasicConv(ch5x5_in,ch5x5_out,kernel_size=5,padding=2),
        )

        # MaxPool+1x1卷积
        self.branch4=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            BasicConv(inchannel,pool,kernel_size=1),
        )

    def forward(self,x):
        out1=self.branch1(x)
        out2=self.branch2(x)
        out3=self.branch3(x)
        out4=self.branch4(x)

        out = [out1, out2, out3, out4]

        return torch.cat(out,1)


# 辅助分类器
class InceptionAux(nn.Module):
    def __init__(self,inchannel,num_classes):
        super(InceptionAux,self).__init__()

        self.avgpool=nn.AvgPool2d(kernel_size=5,stride=3)

        self.conv=BasicConv(inchannel,128,kernel_size=1)

        self.fc1=nn.Linear(2048,1024)
        self.fc2=nn.Linear(1024,num_classes)

    def forward(self,x):
        out=self.avgpool(x)
        out=self.conv(out)
        out=torch.flatten(out,1)
        out=F.dropout(out,0.5,training=self.training)
        out=F.relu(self.fc1(out),inplace=True)
        out=F.dropout(out, 0.5, training=self.training)
        out=self.fc2(out)

        return out


# GoogLeNet
class GoogLeNet(nn.Module):
    def __init__(self,num_classes=10,aux_logit=True,init_weights=False):
        super(GoogLeNet,self).__init__()

        self.aux_logit=aux_logit

        # 第一部分，一个卷积+一个池化
        # padding后的图像大小为（h+2*padding）x（w+2*padding）
        self.conv1=BasicConv(3,64,kernel_size=7,stride=2,padding=3)
        # ceil_mode ：布尔类型，为True，用向上取整的方法，计算输出形状；默认是向下取整.
        self.pool1=nn.MaxPool2d(3,stride=2,ceil_mode=True)


        # 第二部分，两个卷积+一个池化
        self.conv2=BasicConv(64,64,kernel_size=1)
        self.conv3=BasicConv(64,192,kernel_size=3,padding=1)
        self.pool2=nn.MaxPool2d(3,stride=2,ceil_mode=True)


        # 第三部分，3a和3b层
        self.inception3a=Inception(192,64,96,128,16,32,32)
        self.inception3b=Inception(256,128,128,192,32,96,64)
        self.pool3=nn.MaxPool2d(3,stride=2,ceil_mode=True)

        # 第四部分，4a,4b,4c,4d,4e+最大池化层
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # 第五部分，5a和5b层
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logit:
            self.aux1=InceptionAux(512,num_classes)
            self.aux2=InceptionAux(528,num_classes)

        self.avg_pool=nn.AdaptiveAvgPool2d((1,1))
        self.dropout=nn.Dropout(0.4)
        self.fc=nn.Linear(1024,num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        out = self.conv1(x)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.conv3(out)
        out = self.pool2(out)

        out = self.inception3a(out)
        out = self.inception3b(out)
        out = self.pool3(out)

        out = self.inception4a(out)

        if self.training and self.aux_logit:
            aux1=self.aux1(out)

        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)

        if self.training and self.aux_logit:
            aux2=self.aux2(out)
        out = self.inception4e(out)

        out = self.pool4(out)

        out=self.inception5a(out)
        out=self.inception5b(out)

        out=self.avg_pool(out)
        out=torch.flatten(out,1)
        out=self.dropout(out)
        out=self.fc(out)

        if self.training and self.aux_logit:
            return out,aux1,aux2
        return out

    def _initialize_weights(self):  # 将各种初始化方法定义为一个initialize_weights()的函数并在模型初始后进行使用。

        # 遍历网络中的每一层
        for m in self.modules():
            # isinstance(object, type)，如果指定的对象拥有指定的类型，则isinstance()函数返回True

            '''如果是卷积层Conv2d'''
            if isinstance(m, nn.Conv2d):
                # Kaiming正态分布方式的权重初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                '''判断是否有偏置：'''
                # 如果偏置不是0，将偏置置成0，对偏置进行初始化
                if m.bias is not None:
                    # torch.nn.init.constant_(tensor, val)，初始化整个矩阵为常数val
                    nn.init.constant_(m.bias, 0)

                '''如果是全连接层'''
            elif isinstance(m, nn.Linear):
                # init.normal_(tensor, mean=0.0, std=1.0)，使用从正态分布中提取的值填充输入张量
                # 参数：tensor：一个n维Tensor，mean：正态分布的平均值，std：正态分布的标准差
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    net = GoogLeNet().cuda()
    summary(net, (3, 224, 224))



# 打印输出：
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 64, 112, 112]           9,472
#               ReLU-2         [-1, 64, 112, 112]               0
#          BasicConv-3         [-1, 64, 112, 112]               0
#          MaxPool2d-4           [-1, 64, 56, 56]               0
#             Conv2d-5           [-1, 64, 56, 56]           4,160
#               ReLU-6           [-1, 64, 56, 56]               0
#          BasicConv-7           [-1, 64, 56, 56]               0
#             Conv2d-8          [-1, 192, 56, 56]         110,784
#               ReLU-9          [-1, 192, 56, 56]               0
#         BasicConv-10          [-1, 192, 56, 56]               0
#         MaxPool2d-11          [-1, 192, 28, 28]               0
#            Conv2d-12           [-1, 64, 28, 28]          12,352
#              ReLU-13           [-1, 64, 28, 28]               0
#         BasicConv-14           [-1, 64, 28, 28]               0
#            Conv2d-15           [-1, 96, 28, 28]          18,528
#              ReLU-16           [-1, 96, 28, 28]               0
#         BasicConv-17           [-1, 96, 28, 28]               0
#            Conv2d-18          [-1, 128, 28, 28]         110,720
#              ReLU-19          [-1, 128, 28, 28]               0
#         BasicConv-20          [-1, 128, 28, 28]               0
#            Conv2d-21           [-1, 16, 28, 28]           3,088
#              ReLU-22           [-1, 16, 28, 28]               0
#         BasicConv-23           [-1, 16, 28, 28]               0
#            Conv2d-24           [-1, 32, 28, 28]          12,832
#              ReLU-25           [-1, 32, 28, 28]               0
#         BasicConv-26           [-1, 32, 28, 28]               0
#         MaxPool2d-27          [-1, 192, 28, 28]               0
#            Conv2d-28           [-1, 32, 28, 28]           6,176
#              ReLU-29           [-1, 32, 28, 28]               0
#         BasicConv-30           [-1, 32, 28, 28]               0
#         Inception-31          [-1, 256, 28, 28]               0
#            Conv2d-32          [-1, 128, 28, 28]          32,896
#              ReLU-33          [-1, 128, 28, 28]               0
#         BasicConv-34          [-1, 128, 28, 28]               0
#            Conv2d-35          [-1, 128, 28, 28]          32,896
#              ReLU-36          [-1, 128, 28, 28]               0
#         BasicConv-37          [-1, 128, 28, 28]               0
#            Conv2d-38          [-1, 192, 28, 28]         221,376
#              ReLU-39          [-1, 192, 28, 28]               0
#         BasicConv-40          [-1, 192, 28, 28]               0
#            Conv2d-41           [-1, 32, 28, 28]           8,224
#              ReLU-42           [-1, 32, 28, 28]               0
#         BasicConv-43           [-1, 32, 28, 28]               0
#            Conv2d-44           [-1, 96, 28, 28]          76,896
#              ReLU-45           [-1, 96, 28, 28]               0
#         BasicConv-46           [-1, 96, 28, 28]               0
#         MaxPool2d-47          [-1, 256, 28, 28]               0
#            Conv2d-48           [-1, 64, 28, 28]          16,448
#              ReLU-49           [-1, 64, 28, 28]               0
#         BasicConv-50           [-1, 64, 28, 28]               0
#         Inception-51          [-1, 480, 28, 28]               0
#         MaxPool2d-52          [-1, 480, 14, 14]               0
#            Conv2d-53          [-1, 192, 14, 14]          92,352
#              ReLU-54          [-1, 192, 14, 14]               0
#         BasicConv-55          [-1, 192, 14, 14]               0
#            Conv2d-56           [-1, 96, 14, 14]          46,176
#              ReLU-57           [-1, 96, 14, 14]               0
#         BasicConv-58           [-1, 96, 14, 14]               0
#            Conv2d-59          [-1, 208, 14, 14]         179,920
#              ReLU-60          [-1, 208, 14, 14]               0
#         BasicConv-61          [-1, 208, 14, 14]               0
#            Conv2d-62           [-1, 16, 14, 14]           7,696
#              ReLU-63           [-1, 16, 14, 14]               0
#         BasicConv-64           [-1, 16, 14, 14]               0
#            Conv2d-65           [-1, 48, 14, 14]          19,248
#              ReLU-66           [-1, 48, 14, 14]               0
#         BasicConv-67           [-1, 48, 14, 14]               0
#         MaxPool2d-68          [-1, 480, 14, 14]               0
#            Conv2d-69           [-1, 64, 14, 14]          30,784
#              ReLU-70           [-1, 64, 14, 14]               0
#         BasicConv-71           [-1, 64, 14, 14]               0
#         Inception-72          [-1, 512, 14, 14]               0
#         AvgPool2d-73            [-1, 512, 4, 4]               0
#            Conv2d-74            [-1, 128, 4, 4]          65,664
#              ReLU-75            [-1, 128, 4, 4]               0
#         BasicConv-76            [-1, 128, 4, 4]               0
#            Linear-77                 [-1, 1024]       2,098,176
#            Linear-78                   [-1, 10]          10,250
#      InceptionAux-79                   [-1, 10]               0
#            Conv2d-80          [-1, 160, 14, 14]          82,080
#              ReLU-81          [-1, 160, 14, 14]               0
#         BasicConv-82          [-1, 160, 14, 14]               0
#            Conv2d-83          [-1, 112, 14, 14]          57,456
#              ReLU-84          [-1, 112, 14, 14]               0
#         BasicConv-85          [-1, 112, 14, 14]               0
#            Conv2d-86          [-1, 224, 14, 14]         226,016
#              ReLU-87          [-1, 224, 14, 14]               0
#         BasicConv-88          [-1, 224, 14, 14]               0
#            Conv2d-89           [-1, 24, 14, 14]          12,312
#              ReLU-90           [-1, 24, 14, 14]               0
#         BasicConv-91           [-1, 24, 14, 14]               0
#            Conv2d-92           [-1, 64, 14, 14]          38,464
#              ReLU-93           [-1, 64, 14, 14]               0
#         BasicConv-94           [-1, 64, 14, 14]               0
#         MaxPool2d-95          [-1, 512, 14, 14]               0
#            Conv2d-96           [-1, 64, 14, 14]          32,832
#              ReLU-97           [-1, 64, 14, 14]               0
#         BasicConv-98           [-1, 64, 14, 14]               0
#         Inception-99          [-1, 512, 14, 14]               0
#           Conv2d-100          [-1, 128, 14, 14]          65,664
#             ReLU-101          [-1, 128, 14, 14]               0
#        BasicConv-102          [-1, 128, 14, 14]               0
#           Conv2d-103          [-1, 128, 14, 14]          65,664
#             ReLU-104          [-1, 128, 14, 14]               0
#        BasicConv-105          [-1, 128, 14, 14]               0
#           Conv2d-106          [-1, 256, 14, 14]         295,168
#             ReLU-107          [-1, 256, 14, 14]               0
#        BasicConv-108          [-1, 256, 14, 14]               0
#           Conv2d-109           [-1, 24, 14, 14]          12,312
#             ReLU-110           [-1, 24, 14, 14]               0
#        BasicConv-111           [-1, 24, 14, 14]               0
#           Conv2d-112           [-1, 64, 14, 14]          38,464
#             ReLU-113           [-1, 64, 14, 14]               0
#        BasicConv-114           [-1, 64, 14, 14]               0
#        MaxPool2d-115          [-1, 512, 14, 14]               0
#           Conv2d-116           [-1, 64, 14, 14]          32,832
#             ReLU-117           [-1, 64, 14, 14]               0
#        BasicConv-118           [-1, 64, 14, 14]               0
#        Inception-119          [-1, 512, 14, 14]               0
#           Conv2d-120          [-1, 112, 14, 14]          57,456
#             ReLU-121          [-1, 112, 14, 14]               0
#        BasicConv-122          [-1, 112, 14, 14]               0
#           Conv2d-123          [-1, 144, 14, 14]          73,872
#             ReLU-124          [-1, 144, 14, 14]               0
#        BasicConv-125          [-1, 144, 14, 14]               0
#           Conv2d-126          [-1, 288, 14, 14]         373,536
#             ReLU-127          [-1, 288, 14, 14]               0
#        BasicConv-128          [-1, 288, 14, 14]               0
#           Conv2d-129           [-1, 32, 14, 14]          16,416
#             ReLU-130           [-1, 32, 14, 14]               0
#        BasicConv-131           [-1, 32, 14, 14]               0
#           Conv2d-132           [-1, 64, 14, 14]          51,264
#             ReLU-133           [-1, 64, 14, 14]               0
#        BasicConv-134           [-1, 64, 14, 14]               0
#        MaxPool2d-135          [-1, 512, 14, 14]               0
#           Conv2d-136           [-1, 64, 14, 14]          32,832
#             ReLU-137           [-1, 64, 14, 14]               0
#        BasicConv-138           [-1, 64, 14, 14]               0
#        Inception-139          [-1, 528, 14, 14]               0
#        AvgPool2d-140            [-1, 528, 4, 4]               0
#           Conv2d-141            [-1, 128, 4, 4]          67,712
#             ReLU-142            [-1, 128, 4, 4]               0
#        BasicConv-143            [-1, 128, 4, 4]               0
#           Linear-144                 [-1, 1024]       2,098,176
#           Linear-145                   [-1, 10]          10,250
#     InceptionAux-146                   [-1, 10]               0
#           Conv2d-147          [-1, 256, 14, 14]         135,424
#             ReLU-148          [-1, 256, 14, 14]               0
#        BasicConv-149          [-1, 256, 14, 14]               0
#           Conv2d-150          [-1, 160, 14, 14]          84,640
#             ReLU-151          [-1, 160, 14, 14]               0
#        BasicConv-152          [-1, 160, 14, 14]               0
#           Conv2d-153          [-1, 320, 14, 14]         461,120
#             ReLU-154          [-1, 320, 14, 14]               0
#        BasicConv-155          [-1, 320, 14, 14]               0
#           Conv2d-156           [-1, 32, 14, 14]          16,928
#             ReLU-157           [-1, 32, 14, 14]               0
#        BasicConv-158           [-1, 32, 14, 14]               0
#           Conv2d-159          [-1, 128, 14, 14]         102,528
#             ReLU-160          [-1, 128, 14, 14]               0
#        BasicConv-161          [-1, 128, 14, 14]               0
#        MaxPool2d-162          [-1, 528, 14, 14]               0
#           Conv2d-163          [-1, 128, 14, 14]          67,712
#             ReLU-164          [-1, 128, 14, 14]               0
#        BasicConv-165          [-1, 128, 14, 14]               0
#        Inception-166          [-1, 832, 14, 14]               0
#        MaxPool2d-167            [-1, 832, 7, 7]               0
#           Conv2d-168            [-1, 256, 7, 7]         213,248
#             ReLU-169            [-1, 256, 7, 7]               0
#        BasicConv-170            [-1, 256, 7, 7]               0
#           Conv2d-171            [-1, 160, 7, 7]         133,280
#             ReLU-172            [-1, 160, 7, 7]               0
#        BasicConv-173            [-1, 160, 7, 7]               0
#           Conv2d-174            [-1, 320, 7, 7]         461,120
#             ReLU-175            [-1, 320, 7, 7]               0
#        BasicConv-176            [-1, 320, 7, 7]               0
#           Conv2d-177             [-1, 32, 7, 7]          26,656
#             ReLU-178             [-1, 32, 7, 7]               0
#        BasicConv-179             [-1, 32, 7, 7]               0
#           Conv2d-180            [-1, 128, 7, 7]         102,528
#             ReLU-181            [-1, 128, 7, 7]               0
#        BasicConv-182            [-1, 128, 7, 7]               0
#        MaxPool2d-183            [-1, 832, 7, 7]               0
#           Conv2d-184            [-1, 128, 7, 7]         106,624
#             ReLU-185            [-1, 128, 7, 7]               0
#        BasicConv-186            [-1, 128, 7, 7]               0
#        Inception-187            [-1, 832, 7, 7]               0
#           Conv2d-188            [-1, 384, 7, 7]         319,872
#             ReLU-189            [-1, 384, 7, 7]               0
#        BasicConv-190            [-1, 384, 7, 7]               0
#           Conv2d-191            [-1, 192, 7, 7]         159,936
#             ReLU-192            [-1, 192, 7, 7]               0
#        BasicConv-193            [-1, 192, 7, 7]               0
#           Conv2d-194            [-1, 384, 7, 7]         663,936
#             ReLU-195            [-1, 384, 7, 7]               0
#        BasicConv-196            [-1, 384, 7, 7]               0
#           Conv2d-197             [-1, 48, 7, 7]          39,984
#             ReLU-198             [-1, 48, 7, 7]               0
#        BasicConv-199             [-1, 48, 7, 7]               0
#           Conv2d-200            [-1, 128, 7, 7]         153,728
#             ReLU-201            [-1, 128, 7, 7]               0
#        BasicConv-202            [-1, 128, 7, 7]               0
#        MaxPool2d-203            [-1, 832, 7, 7]               0
#           Conv2d-204            [-1, 128, 7, 7]         106,624
#             ReLU-205            [-1, 128, 7, 7]               0
#        BasicConv-206            [-1, 128, 7, 7]               0
#        Inception-207           [-1, 1024, 7, 7]               0
# AdaptiveAvgPool2d-208           [-1, 1024, 1, 1]               0
#          Dropout-209                 [-1, 1024]               0
#           Linear-210                   [-1, 10]          10,250
# ================================================================
# Total params: 10,334,030
# Trainable params: 10,334,030
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 94.33
# Params size (MB): 39.42
# Estimated Total Size (MB): 134.33
# ----------------------------------------------------------------



















#           Linear-210                   [-1, 10]          10,250
# ================================================================
# Total params: 10,334,030
# Trainable params: 10,334,030
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 94.33
# Params size (MB): 39.42
# Estimated Total Size (MB): 134.33
# ----------------------------------------------------------------

# (yolov8) C:\Users\li\Desktop\repo\classification-pyqt\code>\















#           Linear-210                   [-1, 10]          10,250
# ================================================================
# Total params: 10,334,030
# Trainable params: 10,334,030
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 94.33
# Params size (MB): 39.42
# Estimated Total Size (MB): 134.33
# ----------------------------------------------------------------

# (yolov8) C:\Users\li\Desktop\repo\classification-pyqt\code>\














#           Linear-210                   [-1, 10]          10,250
# ================================================================
# Total params: 10,334,030
# Trainable params: 10,334,030
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 94.33
# Params size (MB): 39.42
# Estimated Total Size (MB): 134.33
# ----------------------------------------------------------------

# (yolov8) C:\Users\li\Desktop\repo\classification-pyqt\code>\











#           Linear-210                   [-1, 10]          10,250
# ================================================================
# Total params: 10,334,030
# Trainable params: 10,334,030
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 94.33
# Params size (MB): 39.42
# Estimated Total Size (MB): 134.33
# ----------------------------------------------------------------

# (yolov8) C:\Users\li\Desktop\repo\classification-pyqt\code>\








#           Linear-210                   [-1, 10]          10,250
# ================================================================
# Total params: 10,334,030
# Trainable params: 10,334,030
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 94.33
# Params size (MB): 39.42
# Estimated Total Size (MB): 134.33
# ----------------------------------------------------------------

# (yolov8) C:\Users\li\Desktop\repo\classification-pyqt\code>\






#           Linear-210                   [-1, 10]          10,250
# ================================================================
# Total params: 10,334,030
# Trainable params: 10,334,030
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 94.33
# Params size (MB): 39.42
# Estimated Total Size (MB): 134.33
# ----------------------------------------------------------------

# (yolov8) C:\Users\li\Desktop\repo\classification-pyqt\code>\



#           Linear-210                   [-1, 10]          10,250
# ================================================================
# Total params: 10,334,030
# Trainable params: 10,334,030
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 94.33
# Params size (MB): 39.42
# Estimated Total Size (MB): 134.33
# ----------------------------------------------------------------

# (yolov8) C:\Users\li\Desktop\repo\classification-pyqt\code>\


#           Linear-210                   [-1, 10]          10,250
# ================================================================
# Total params: 10,334,030
# Trainable params: 10,334,030
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 94.33
# Params size (MB): 39.42
# Estimated Total Size (MB): 134.33
# ----------------------------------------------------------------
#           Linear-210                   [-1, 10]          10,250
# ================================================================
# Total params: 10,334,030
# Trainable params: 10,334,030
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 94.33
# Params size (MB): 39.42
#           Linear-210                   [-1, 10]          10,250
# ================================================================
# Total params: 10,334,030
# Trainable params: 10,334,030
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 94.33
#           Linear-210                   [-1, 10]          10,250
# ================================================================
# Total params: 10,334,030
# Trainable params: 10,334,030
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# ================================================================
# Total params: 10,334,030
# Trainable params: 10,334,030
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Trainable params: 10,334,030
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 94.33
# Input size (MB): 0.57
# Forward/backward pass size (MB): 94.33
# Forward/backward pass size (MB): 94.33
# Params size (MB): 39.42
# Estimated Total Size (MB): 134.33
# Params size (MB): 39.42
# Estimated Total Size (MB): 134.33
# ----------------------------------------------------------------
# Estimated Total Size (MB): 134.33
# ----------------------------------------------------------------
# ----------------------------------------------------------------

# (yolov8) C:\Users\li\Desktop\repo\classification-pyqt\code>\clear
# '\clear' 不是内部或外部命令，也不是可运行的程序
# 或批处理文件。

# (yolov8) C:\Users\li\Desktop\repo\classification-pyqt\code>clear
# 'clear' 不是内部或外部命令，也不是可运行的程序
# 或批处理文件。

# (yolov8) C:\Users\li\Desktop\repo\classification-pyqt\code>python GoogLeNet.py
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 64, 112, 112]           9,472
#               ReLU-2         [-1, 64, 112, 112]               0
#          BasicConv-3         [-1, 64, 112, 112]               0
#          MaxPool2d-4           [-1, 64, 56, 56]               0
#             Conv2d-5           [-1, 64, 56, 56]           4,160
#               ReLU-6           [-1, 64, 56, 56]               0
#          BasicConv-7           [-1, 64, 56, 56]               0
#             Conv2d-8          [-1, 192, 56, 56]         110,784
#               ReLU-9          [-1, 192, 56, 56]               0
#         BasicConv-10          [-1, 192, 56, 56]               0
#         MaxPool2d-11          [-1, 192, 28, 28]               0
#            Conv2d-12           [-1, 64, 28, 28]          12,352
#              ReLU-13           [-1, 64, 28, 28]               0
#         BasicConv-14           [-1, 64, 28, 28]               0
#            Conv2d-15           [-1, 96, 28, 28]          18,528
#              ReLU-16           [-1, 96, 28, 28]               0
#         BasicConv-17           [-1, 96, 28, 28]               0
#            Conv2d-18          [-1, 128, 28, 28]         110,720
#              ReLU-19          [-1, 128, 28, 28]               0
#         BasicConv-20          [-1, 128, 28, 28]               0
#            Conv2d-21           [-1, 16, 28, 28]           3,088
#              ReLU-22           [-1, 16, 28, 28]               0
#         BasicConv-23           [-1, 16, 28, 28]               0
#            Conv2d-24           [-1, 32, 28, 28]          12,832
#              ReLU-25           [-1, 32, 28, 28]               0
#         BasicConv-26           [-1, 32, 28, 28]               0
#         MaxPool2d-27          [-1, 192, 28, 28]               0
#            Conv2d-28           [-1, 32, 28, 28]           6,176
#              ReLU-29           [-1, 32, 28, 28]               0
#         BasicConv-30           [-1, 32, 28, 28]               0
#         Inception-31          [-1, 256, 28, 28]               0
#            Conv2d-32          [-1, 128, 28, 28]          32,896
#              ReLU-33          [-1, 128, 28, 28]               0
#         BasicConv-34          [-1, 128, 28, 28]               0
#            Conv2d-35          [-1, 128, 28, 28]          32,896
#              ReLU-36          [-1, 128, 28, 28]               0
#         BasicConv-37          [-1, 128, 28, 28]               0
#            Conv2d-38          [-1, 192, 28, 28]         221,376
#              ReLU-39          [-1, 192, 28, 28]               0
#         BasicConv-40          [-1, 192, 28, 28]               0
#            Conv2d-41           [-1, 32, 28, 28]           8,224
#              ReLU-42           [-1, 32, 28, 28]               0
#         BasicConv-43           [-1, 32, 28, 28]               0
#            Conv2d-44           [-1, 96, 28, 28]          76,896
#              ReLU-45           [-1, 96, 28, 28]               0
#         BasicConv-46           [-1, 96, 28, 28]               0
#         MaxPool2d-47          [-1, 256, 28, 28]               0
#            Conv2d-48           [-1, 64, 28, 28]          16,448
#              ReLU-49           [-1, 64, 28, 28]               0
#         BasicConv-50           [-1, 64, 28, 28]               0
#         Inception-51          [-1, 480, 28, 28]               0
#         MaxPool2d-52          [-1, 480, 14, 14]               0
#            Conv2d-53          [-1, 192, 14, 14]          92,352
#              ReLU-54          [-1, 192, 14, 14]               0
#         BasicConv-55          [-1, 192, 14, 14]               0
#            Conv2d-56           [-1, 96, 14, 14]          46,176
#              ReLU-57           [-1, 96, 14, 14]               0
#         BasicConv-58           [-1, 96, 14, 14]               0
#            Conv2d-59          [-1, 208, 14, 14]         179,920
#              ReLU-60          [-1, 208, 14, 14]               0
#         BasicConv-61          [-1, 208, 14, 14]               0
#            Conv2d-62           [-1, 16, 14, 14]           7,696
#              ReLU-63           [-1, 16, 14, 14]               0
#         BasicConv-64           [-1, 16, 14, 14]               0
#            Conv2d-65           [-1, 48, 14, 14]          19,248
#              ReLU-66           [-1, 48, 14, 14]               0
#         BasicConv-67           [-1, 48, 14, 14]               0
#         MaxPool2d-68          [-1, 480, 14, 14]               0
#            Conv2d-69           [-1, 64, 14, 14]          30,784
#              ReLU-70           [-1, 64, 14, 14]               0
#         BasicConv-71           [-1, 64, 14, 14]               0
#         Inception-72          [-1, 512, 14, 14]               0
#         AvgPool2d-73            [-1, 512, 4, 4]               0
#            Conv2d-74            [-1, 128, 4, 4]          65,664
#              ReLU-75            [-1, 128, 4, 4]               0
#         BasicConv-76            [-1, 128, 4, 4]               0
#            Linear-77                 [-1, 1024]       2,098,176
#            Linear-78                   [-1, 10]          10,250
#      InceptionAux-79                   [-1, 10]               0
#            Conv2d-80          [-1, 160, 14, 14]          82,080
#              ReLU-81          [-1, 160, 14, 14]               0
#         BasicConv-82          [-1, 160, 14, 14]               0
#            Conv2d-83          [-1, 112, 14, 14]          57,456
#              ReLU-84          [-1, 112, 14, 14]               0
#         BasicConv-85          [-1, 112, 14, 14]               0
#            Conv2d-86          [-1, 224, 14, 14]         226,016
#              ReLU-87          [-1, 224, 14, 14]               0
#         BasicConv-88          [-1, 224, 14, 14]               0
#            Conv2d-89           [-1, 24, 14, 14]          12,312
#              ReLU-90           [-1, 24, 14, 14]               0
#         BasicConv-91           [-1, 24, 14, 14]               0
#            Conv2d-92           [-1, 64, 14, 14]          38,464
#              ReLU-93           [-1, 64, 14, 14]               0
#         BasicConv-94           [-1, 64, 14, 14]               0
#         MaxPool2d-95          [-1, 512, 14, 14]               0
#            Conv2d-96           [-1, 64, 14, 14]          32,832
#              ReLU-97           [-1, 64, 14, 14]               0
#         BasicConv-98           [-1, 64, 14, 14]               0
#         Inception-99          [-1, 512, 14, 14]               0
#           Conv2d-100          [-1, 128, 14, 14]          65,664
#             ReLU-101          [-1, 128, 14, 14]               0
#        BasicConv-102          [-1, 128, 14, 14]               0
#           Conv2d-103          [-1, 128, 14, 14]          65,664
#             ReLU-104          [-1, 128, 14, 14]               0
#        BasicConv-105          [-1, 128, 14, 14]               0
#           Conv2d-106          [-1, 256, 14, 14]         295,168
#             ReLU-107          [-1, 256, 14, 14]               0
#        BasicConv-108          [-1, 256, 14, 14]               0
#           Conv2d-109           [-1, 24, 14, 14]          12,312
#             ReLU-110           [-1, 24, 14, 14]               0
#        BasicConv-111           [-1, 24, 14, 14]               0
#           Conv2d-112           [-1, 64, 14, 14]          38,464
#             ReLU-113           [-1, 64, 14, 14]               0
#        BasicConv-114           [-1, 64, 14, 14]               0
#        MaxPool2d-115          [-1, 512, 14, 14]               0
#           Conv2d-116           [-1, 64, 14, 14]          32,832
#             ReLU-117           [-1, 64, 14, 14]               0
#        BasicConv-118           [-1, 64, 14, 14]               0
#        Inception-119          [-1, 512, 14, 14]               0
#           Conv2d-120          [-1, 112, 14, 14]          57,456
#             ReLU-121          [-1, 112, 14, 14]               0
#        BasicConv-122          [-1, 112, 14, 14]               0
#           Conv2d-123          [-1, 144, 14, 14]          73,872
#             ReLU-124          [-1, 144, 14, 14]               0
#        BasicConv-125          [-1, 144, 14, 14]               0
#           Conv2d-126          [-1, 288, 14, 14]         373,536
#             ReLU-127          [-1, 288, 14, 14]               0
#        BasicConv-128          [-1, 288, 14, 14]               0
#           Conv2d-129           [-1, 32, 14, 14]          16,416
#             ReLU-130           [-1, 32, 14, 14]               0
#        BasicConv-131           [-1, 32, 14, 14]               0
#           Conv2d-132           [-1, 64, 14, 14]          51,264
#             ReLU-133           [-1, 64, 14, 14]               0
#        BasicConv-134           [-1, 64, 14, 14]               0
#        MaxPool2d-135          [-1, 512, 14, 14]               0
#           Conv2d-136           [-1, 64, 14, 14]          32,832
#             ReLU-137           [-1, 64, 14, 14]               0
#        BasicConv-138           [-1, 64, 14, 14]               0
#        Inception-139          [-1, 528, 14, 14]               0
#        AvgPool2d-140            [-1, 528, 4, 4]               0
#           Conv2d-141            [-1, 128, 4, 4]          67,712
#             ReLU-142            [-1, 128, 4, 4]               0
#        BasicConv-143            [-1, 128, 4, 4]               0
#           Linear-144                 [-1, 1024]       2,098,176
#           Linear-145                   [-1, 10]          10,250
#     InceptionAux-146                   [-1, 10]               0
#           Conv2d-147          [-1, 256, 14, 14]         135,424
#             ReLU-148          [-1, 256, 14, 14]               0
#        BasicConv-149          [-1, 256, 14, 14]               0
#           Conv2d-150          [-1, 160, 14, 14]          84,640
#             ReLU-151          [-1, 160, 14, 14]               0
#        BasicConv-152          [-1, 160, 14, 14]               0
#           Conv2d-153          [-1, 320, 14, 14]         461,120
#             ReLU-154          [-1, 320, 14, 14]               0
#        BasicConv-155          [-1, 320, 14, 14]               0
#           Conv2d-156           [-1, 32, 14, 14]          16,928
#             ReLU-157           [-1, 32, 14, 14]               0
#        BasicConv-158           [-1, 32, 14, 14]               0
#           Conv2d-159          [-1, 128, 14, 14]         102,528
#             ReLU-160          [-1, 128, 14, 14]               0
#        BasicConv-161          [-1, 128, 14, 14]               0
#        MaxPool2d-162          [-1, 528, 14, 14]               0
#           Conv2d-163          [-1, 128, 14, 14]          67,712
#             ReLU-164          [-1, 128, 14, 14]               0
#        BasicConv-165          [-1, 128, 14, 14]               0
#        Inception-166          [-1, 832, 14, 14]               0
#        MaxPool2d-167            [-1, 832, 7, 7]               0
#           Conv2d-168            [-1, 256, 7, 7]         213,248
#             ReLU-169            [-1, 256, 7, 7]               0
#        BasicConv-170            [-1, 256, 7, 7]               0
#           Conv2d-171            [-1, 160, 7, 7]         133,280
#             ReLU-172            [-1, 160, 7, 7]               0
#        BasicConv-173            [-1, 160, 7, 7]               0
#           Conv2d-174            [-1, 320, 7, 7]         461,120
#             ReLU-175            [-1, 320, 7, 7]               0
#        BasicConv-176            [-1, 320, 7, 7]               0
#           Conv2d-177             [-1, 32, 7, 7]          26,656
#             ReLU-178             [-1, 32, 7, 7]               0
#        BasicConv-179             [-1, 32, 7, 7]               0
#           Conv2d-180            [-1, 128, 7, 7]         102,528
#             ReLU-181            [-1, 128, 7, 7]               0
#        BasicConv-182            [-1, 128, 7, 7]               0
#        MaxPool2d-183            [-1, 832, 7, 7]               0
#           Conv2d-184            [-1, 128, 7, 7]         106,624
#             ReLU-185            [-1, 128, 7, 7]               0
#        BasicConv-186            [-1, 128, 7, 7]               0
#        Inception-187            [-1, 832, 7, 7]               0
#           Conv2d-188            [-1, 384, 7, 7]         319,872
#             ReLU-189            [-1, 384, 7, 7]               0
#        BasicConv-190            [-1, 384, 7, 7]               0
#           Conv2d-191            [-1, 192, 7, 7]         159,936
#             ReLU-192            [-1, 192, 7, 7]               0
#        BasicConv-193            [-1, 192, 7, 7]               0
#           Conv2d-194            [-1, 384, 7, 7]         663,936
#             ReLU-195            [-1, 384, 7, 7]               0
#        BasicConv-196            [-1, 384, 7, 7]               0
#           Conv2d-197             [-1, 48, 7, 7]          39,984
#             ReLU-198             [-1, 48, 7, 7]               0
#        BasicConv-199             [-1, 48, 7, 7]               0
#           Conv2d-200            [-1, 128, 7, 7]         153,728
#             ReLU-201            [-1, 128, 7, 7]               0
#        BasicConv-202            [-1, 128, 7, 7]               0
#        MaxPool2d-203            [-1, 832, 7, 7]               0
#           Conv2d-204            [-1, 128, 7, 7]         106,624
#             ReLU-205            [-1, 128, 7, 7]               0
#        BasicConv-206            [-1, 128, 7, 7]               0
#        Inception-207           [-1, 1024, 7, 7]               0
# AdaptiveAvgPool2d-208           [-1, 1024, 1, 1]               0
#          Dropout-209                 [-1, 1024]               0
#           Linear-210                   [-1, 10]          10,250
# ================================================================
# Total params: 10,334,030
# Trainable params: 10,334,030
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 94.33
# Params size (MB): 39.42
# Estimated Total Size (MB): 134.33
# ----------------------------------------------------------------