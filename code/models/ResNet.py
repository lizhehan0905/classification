import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary

'''-------------一、BasicBlock模块-----------------------------'''
# 用于ResNet18和ResNet34基本残差结构块
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()

        # 左侧路径
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            # 每个卷积层后进行归一化的原因(BatchNorm处理一维，通常在全连接层的输出使用，BatchNorm2d处理二维，通常在卷积层后使用)
            # 数据的分布随着网络加深而变化，BN操作规范化每一层的输入，使其具有相同的均值和方差，加速训练过程
            # BN是一种正则化手段，在每个mini-batch上规范化数据，使模型在训练是不会依赖于某个特定的数据分布
            # BN操作使得每一层的输出都有相同的尺度，有利于防止梯度消失和梯度爆炸
            # BN使得每一层的输入都具有相同的分布，使模型对初始化参数的敏感性减低。即使使用不同的初始化方法也能达到相似的性能
            nn.BatchNorm2d(outchannel),
            # inplace=True 表示进行原地操作，即将计算结果直接覆盖到输入张量中
            nn.ReLU(inplace=True),

            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        # 右侧恒等映射
        self.shortcut = nn.Sequential()

        # 当步长不为1或者输入通道不等于输出通道时，进行修改，确保“捷径”输出的尺寸和通道数与主路径的输出一致，从而可以进行元素级加法操作
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


'''-------------二、Bottleneck模块-----------------------------'''
# 用于ResNet50及以上的残差结构块
class Bottleneck(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(Bottleneck, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, int(outchannel / 4), kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(int(outchannel / 4)),
            nn.ReLU(inplace=True),

            nn.Conv2d(int(outchannel / 4), int(outchannel / 4), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(outchannel / 4)),
            nn.ReLU(inplace=True),

            nn.Conv2d(int(outchannel / 4), outchannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        y = self.shortcut(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

'''-------------ResNet18---------------'''
class ResNet_18(nn.Module):
    def __init__(self, ResidualBlock, num_classes=100):
        # 调用父类初始化方法
        super(ResNet_18, self).__init__()
        # 初始化变量inchannel，用于跟踪当前层的输入通道数，初始值为64
        self.inchannel = 64
        # 初始的卷积层conv1，将输入图像的通道数从3（RGB）转换为64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        # 创建四个残差层
        # 每个层包含指定数量的残差块，并且随着层数的增加，输出通道数和残差块的数量（num_blocks）也增加
        # stride只在每个层的第一个残差块中设置，用于调整特征图的尺寸
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)# 与权重矩阵相乘，加上偏置项，得到类别个数的向量

    # 创建包含多个残差块的层
    def make_layer(self, block, channels, num_blocks, stride):
        # 创建一个列表strides，其中第一个元素是stride（用于第一个残差块），其余元素都是1（用于后续残差块）
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        # 遍历strides列表，为每个步长创建一个残差块，并将其添加到layers列表中。同时，更新self.inchannel以反映下一个残差块的输入通道数
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        # 使用nn.Sequential将layers列表中的残差块封装成一个模块，并返回该模块
        return nn.Sequential(*layers)

    def forward(self, x):  # 3*32*32
        # 依次将输入x通过初始卷积层conv1和四个残差层
        out = self.conv1(x)  # 64*32*32
        out = self.pool(out)
        out = self.layer1(out)  # 64*32*32
        out = self.layer2(out)  # 128*16*16
        out = self.layer3(out)  # 256*8*8
        out = self.layer4(out)  # 512*4*4
        out = self.avgpool(out)  # 512*1*1
        out = torch.flatten(out,1)  # 512
        out = self.fc(out)
        return out


'''-------------ResNet34---------------'''
class ResNet_34(nn.Module):
    def __init__(self, ResidualBlock, num_classes=100):
        super(ResNet_34, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self.make_layer(ResidualBlock, 64, 3, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 4, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 6, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)  
        out = self.pool(out)
        out = self.layer1(out) 
        out = self.layer2(out) 
        out = self.layer3(out)  
        out = self.layer4(out)  
        out = self.avgpool(out)  
        out = torch.flatten(out,1) 
        out = self.fc(out)
        return out


'''-------------ResNet50---------------'''
class ResNet_50(nn.Module):
    def __init__(self, ResidualBlock, num_classes=100):
        super(ResNet_50, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self.make_layer(ResidualBlock, 256, 3, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 512, 4, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 1024, 6, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 2048, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):  
        out = self.conv1(x)  
        out = self.pool(out)
        out = self.layer1(out)  
        out = self.layer2(out)  
        out = self.layer3(out) 
        out = self.layer4(out)  
        out = self.avgpool(out)  
        out = torch.flatten(out,1)
        out = self.fc(out)
        return out


def ResNet18():

    return ResNet_18(ResidualBlock)


def ResNet34():
    return ResNet_34(ResidualBlock)


def ResNet50():
    return ResNet_50(Bottleneck)



if __name__=='__main__':
    model=ResNet50().cuda()
    summary(model,(3,224,224))

# 输出：
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 64, 112, 112]           9,408
#        BatchNorm2d-2         [-1, 64, 112, 112]             128
#               ReLU-3         [-1, 64, 112, 112]               0
#          MaxPool2d-4           [-1, 64, 56, 56]               0
#             Conv2d-5           [-1, 64, 56, 56]           4,096
#        BatchNorm2d-6           [-1, 64, 56, 56]             128
#               ReLU-7           [-1, 64, 56, 56]               0
#             Conv2d-8           [-1, 64, 56, 56]          36,864
#        BatchNorm2d-9           [-1, 64, 56, 56]             128
#              ReLU-10           [-1, 64, 56, 56]               0
#            Conv2d-11          [-1, 256, 56, 56]          16,384
#       BatchNorm2d-12          [-1, 256, 56, 56]             512
#            Conv2d-13          [-1, 256, 56, 56]          16,384
#       BatchNorm2d-14          [-1, 256, 56, 56]             512
#            Conv2d-15          [-1, 256, 56, 56]          16,384
#       BatchNorm2d-16          [-1, 256, 56, 56]             512
#        Bottleneck-17          [-1, 256, 56, 56]               0
#            Conv2d-18           [-1, 64, 56, 56]          16,384
#       BatchNorm2d-19           [-1, 64, 56, 56]             128
#              ReLU-20           [-1, 64, 56, 56]               0
#            Conv2d-21           [-1, 64, 56, 56]          36,864
#       BatchNorm2d-22           [-1, 64, 56, 56]             128
#              ReLU-23           [-1, 64, 56, 56]               0
#            Conv2d-24          [-1, 256, 56, 56]          16,384
#       BatchNorm2d-25          [-1, 256, 56, 56]             512
#        Bottleneck-26          [-1, 256, 56, 56]               0
#            Conv2d-27           [-1, 64, 56, 56]          16,384
#       BatchNorm2d-28           [-1, 64, 56, 56]             128
#              ReLU-29           [-1, 64, 56, 56]               0
#            Conv2d-30           [-1, 64, 56, 56]          36,864
#       BatchNorm2d-31           [-1, 64, 56, 56]             128
#              ReLU-32           [-1, 64, 56, 56]               0
#            Conv2d-33          [-1, 256, 56, 56]          16,384
#       BatchNorm2d-34          [-1, 256, 56, 56]             512
#        Bottleneck-35          [-1, 256, 56, 56]               0
#            Conv2d-36          [-1, 128, 28, 28]          32,768
#       BatchNorm2d-37          [-1, 128, 28, 28]             256
#              ReLU-38          [-1, 128, 28, 28]               0
#            Conv2d-39          [-1, 128, 28, 28]         147,456
#       BatchNorm2d-40          [-1, 128, 28, 28]             256
#              ReLU-41          [-1, 128, 28, 28]               0
#            Conv2d-42          [-1, 512, 28, 28]          65,536
#       BatchNorm2d-43          [-1, 512, 28, 28]           1,024
#            Conv2d-44          [-1, 512, 28, 28]         131,072
#       BatchNorm2d-45          [-1, 512, 28, 28]           1,024
#            Conv2d-46          [-1, 512, 28, 28]         131,072
#       BatchNorm2d-47          [-1, 512, 28, 28]           1,024
#        Bottleneck-48          [-1, 512, 28, 28]               0
#            Conv2d-49          [-1, 128, 28, 28]          65,536
#       BatchNorm2d-50          [-1, 128, 28, 28]             256
#              ReLU-51          [-1, 128, 28, 28]               0
#            Conv2d-52          [-1, 128, 28, 28]         147,456
#       BatchNorm2d-53          [-1, 128, 28, 28]             256
#              ReLU-54          [-1, 128, 28, 28]               0
#            Conv2d-55          [-1, 512, 28, 28]          65,536
#       BatchNorm2d-56          [-1, 512, 28, 28]           1,024
#        Bottleneck-57          [-1, 512, 28, 28]               0
#            Conv2d-58          [-1, 128, 28, 28]          65,536
#       BatchNorm2d-59          [-1, 128, 28, 28]             256
#              ReLU-60          [-1, 128, 28, 28]               0
#            Conv2d-61          [-1, 128, 28, 28]         147,456
#       BatchNorm2d-62          [-1, 128, 28, 28]             256
#              ReLU-63          [-1, 128, 28, 28]               0
#            Conv2d-64          [-1, 512, 28, 28]          65,536
#       BatchNorm2d-65          [-1, 512, 28, 28]           1,024
#        Bottleneck-66          [-1, 512, 28, 28]               0
#            Conv2d-67          [-1, 128, 28, 28]          65,536
#       BatchNorm2d-68          [-1, 128, 28, 28]             256
#              ReLU-69          [-1, 128, 28, 28]               0
#            Conv2d-70          [-1, 128, 28, 28]         147,456
#       BatchNorm2d-71          [-1, 128, 28, 28]             256
#              ReLU-72          [-1, 128, 28, 28]               0
#            Conv2d-73          [-1, 512, 28, 28]          65,536
#       BatchNorm2d-74          [-1, 512, 28, 28]           1,024
#        Bottleneck-75          [-1, 512, 28, 28]               0
#            Conv2d-76          [-1, 256, 14, 14]         131,072
#       BatchNorm2d-77          [-1, 256, 14, 14]             512
#              ReLU-78          [-1, 256, 14, 14]               0
#            Conv2d-79          [-1, 256, 14, 14]         589,824
#       BatchNorm2d-80          [-1, 256, 14, 14]             512
#              ReLU-81          [-1, 256, 14, 14]               0
#            Conv2d-82         [-1, 1024, 14, 14]         262,144
#       BatchNorm2d-83         [-1, 1024, 14, 14]           2,048
#            Conv2d-84         [-1, 1024, 14, 14]         524,288
#       BatchNorm2d-85         [-1, 1024, 14, 14]           2,048
#            Conv2d-86         [-1, 1024, 14, 14]         524,288
#       BatchNorm2d-87         [-1, 1024, 14, 14]           2,048
#        Bottleneck-88         [-1, 1024, 14, 14]               0
#            Conv2d-89          [-1, 256, 14, 14]         262,144
#       BatchNorm2d-90          [-1, 256, 14, 14]             512
#              ReLU-91          [-1, 256, 14, 14]               0
#            Conv2d-92          [-1, 256, 14, 14]         589,824
#       BatchNorm2d-93          [-1, 256, 14, 14]             512
#              ReLU-94          [-1, 256, 14, 14]               0
#            Conv2d-95         [-1, 1024, 14, 14]         262,144
#       BatchNorm2d-96         [-1, 1024, 14, 14]           2,048
#        Bottleneck-97         [-1, 1024, 14, 14]               0
#            Conv2d-98          [-1, 256, 14, 14]         262,144
#       BatchNorm2d-99          [-1, 256, 14, 14]             512
#             ReLU-100          [-1, 256, 14, 14]               0
#           Conv2d-101          [-1, 256, 14, 14]         589,824
#      BatchNorm2d-102          [-1, 256, 14, 14]             512
#             ReLU-103          [-1, 256, 14, 14]               0
#           Conv2d-104         [-1, 1024, 14, 14]         262,144
#      BatchNorm2d-105         [-1, 1024, 14, 14]           2,048
#       Bottleneck-106         [-1, 1024, 14, 14]               0
#           Conv2d-107          [-1, 256, 14, 14]         262,144
#      BatchNorm2d-108          [-1, 256, 14, 14]             512
#             ReLU-109          [-1, 256, 14, 14]               0
#           Conv2d-110          [-1, 256, 14, 14]         589,824
#      BatchNorm2d-111          [-1, 256, 14, 14]             512
#             ReLU-112          [-1, 256, 14, 14]               0
#           Conv2d-113         [-1, 1024, 14, 14]         262,144
#      BatchNorm2d-114         [-1, 1024, 14, 14]           2,048
#       Bottleneck-115         [-1, 1024, 14, 14]               0
#           Conv2d-116          [-1, 256, 14, 14]         262,144
#      BatchNorm2d-117          [-1, 256, 14, 14]             512
#             ReLU-118          [-1, 256, 14, 14]               0
#           Conv2d-119          [-1, 256, 14, 14]         589,824
#      BatchNorm2d-120          [-1, 256, 14, 14]             512
#             ReLU-121          [-1, 256, 14, 14]               0
#           Conv2d-122         [-1, 1024, 14, 14]         262,144
#      BatchNorm2d-123         [-1, 1024, 14, 14]           2,048
#       Bottleneck-124         [-1, 1024, 14, 14]               0
#           Conv2d-125          [-1, 256, 14, 14]         262,144
#      BatchNorm2d-126          [-1, 256, 14, 14]             512
#             ReLU-127          [-1, 256, 14, 14]               0
#           Conv2d-128          [-1, 256, 14, 14]         589,824
#      BatchNorm2d-129          [-1, 256, 14, 14]             512
#             ReLU-130          [-1, 256, 14, 14]               0
#           Conv2d-131         [-1, 1024, 14, 14]         262,144
#      BatchNorm2d-132         [-1, 1024, 14, 14]           2,048
#       Bottleneck-133         [-1, 1024, 14, 14]               0
#           Conv2d-134            [-1, 512, 7, 7]         524,288
#      BatchNorm2d-135            [-1, 512, 7, 7]           1,024
#             ReLU-136            [-1, 512, 7, 7]               0
#           Conv2d-137            [-1, 512, 7, 7]       2,359,296
#      BatchNorm2d-138            [-1, 512, 7, 7]           1,024
#             ReLU-139            [-1, 512, 7, 7]               0
#           Conv2d-140           [-1, 2048, 7, 7]       1,048,576
#      BatchNorm2d-141           [-1, 2048, 7, 7]           4,096
#           Conv2d-142           [-1, 2048, 7, 7]       2,097,152
#      BatchNorm2d-143           [-1, 2048, 7, 7]           4,096
#           Conv2d-144           [-1, 2048, 7, 7]       2,097,152
#      BatchNorm2d-145           [-1, 2048, 7, 7]           4,096
#       Bottleneck-146           [-1, 2048, 7, 7]               0
#           Conv2d-147            [-1, 512, 7, 7]       1,048,576
#      BatchNorm2d-148            [-1, 512, 7, 7]           1,024
#             ReLU-149            [-1, 512, 7, 7]               0
#           Conv2d-150            [-1, 512, 7, 7]       2,359,296
#      BatchNorm2d-151            [-1, 512, 7, 7]           1,024
#             ReLU-152            [-1, 512, 7, 7]               0
#           Conv2d-153           [-1, 2048, 7, 7]       1,048,576
#      BatchNorm2d-154           [-1, 2048, 7, 7]           4,096
#       Bottleneck-155           [-1, 2048, 7, 7]               0
#           Conv2d-156            [-1, 512, 7, 7]       1,048,576
#      BatchNorm2d-157            [-1, 512, 7, 7]           1,024
#             ReLU-158            [-1, 512, 7, 7]               0
#           Conv2d-159            [-1, 512, 7, 7]       2,359,296
#      BatchNorm2d-160            [-1, 512, 7, 7]           1,024
#             ReLU-161            [-1, 512, 7, 7]               0
#           Conv2d-162           [-1, 2048, 7, 7]       1,048,576
#      BatchNorm2d-163           [-1, 2048, 7, 7]           4,096
#       Bottleneck-164           [-1, 2048, 7, 7]               0
# AdaptiveAvgPool2d-165           [-1, 2048, 1, 1]               0
#           Linear-166                  [-1, 100]         204,900
# ================================================================
# Total params: 26,489,508
# Trainable params: 26,489,508
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 255.35
# Params size (MB): 101.05
# Estimated Total Size (MB): 356.98
# ----------------------------------------------------------------