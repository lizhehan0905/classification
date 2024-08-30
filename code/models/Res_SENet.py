import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class SE_block(nn.Module):
    def __init__(self,inchannel,ratio=16):
        super(SE_block,self).__init__()

        self.avg_pool=nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio,bias=False),
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel,bias=False),
            nn.Sigmoid(),
        )

    def forward(self,x):
        # 读取批数据图片数量及通道数
        b,c,w,h=x.size()
        # Fsq操作：经池化后输出b*c的矩阵 view就是通过查看numpy数组，改数据类型和形状，但不改原数据
        out=self.avg_pool(x).view(b,c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        out=self.fc(out).view(b,c,1,1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x*out.expand_as(x)


class BasicBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1):
        super(BasicBlock,self).__init__()

        self.left=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(outchannel),
        )

        self.se=SE_block(outchannel)

        self.shortcut=nn.Sequential()

        if stride!=1 or inchannel!=outchannel:
            self.shortcut=nn.Sequential(
                nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=stride),
                nn.BatchNorm2d(outchannel),
            )

    def forward(self,x):
        out=self.left(x)
        out=self.se(out)
        out=out+self.shortcut(x)
        out=F.relu(out)
        return out

class Bottleneck(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1):
        super(Bottleneck,self).__init__()

        self.left=nn.Sequential(
            nn.Conv2d(inchannel,outchannel//4,kernel_size=1),
            nn.BatchNorm2d(outchannel//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel//4,outchannel//4,kernel_size=3,padding=1,stride=stride),
            nn.BatchNorm2d(outchannel // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel//4,outchannel,kernel_size=1),
            nn.BatchNorm2d(outchannel),
        )

        self.se=SE_block(outchannel)

        self.shortcut = nn.Sequential()

        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1,stride=stride),
                nn.BatchNorm2d(outchannel),
            )


    def forward(self,x):
        out=self.left(x)
        out=self.se(out)
        out=out+self.shortcut(x)
        out=F.relu(out)
        return out


class SENet(nn.Module):
    def __init__(self,block,num_blocks,num_classes=10):
        super(SENet,self).__init__()

        self.inchannel=64
        self.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1)
        self.BN=nn.BatchNorm2d(64)
        self.ReLU=nn.ReLU(inplace=True)



        self.layer1 = self.make_layer(block,64,num_blocks[0],stride=1)
        self.layer2 = self.make_layer(block,128,num_blocks[1],stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)

        self.avg_pool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512,num_classes)

    def make_layer(self,block,channel,num_blocks,stride):
        strides=[stride]+[1]*(num_blocks-1)
        layers=[]
        for stride in strides:
            layers.append(block(self.inchannel,channel,stride))
            self.inchannel=channel
        return nn.Sequential(*layers)


    def forward(self,x):
        out=self.conv1(x)
        out=self.BN(out)
        out=self.ReLU(out)

        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.avg_pool(out)
        out=torch.flatten(out,1)
        out=self.fc(out)

        return out


def SE_ResNet18():
    return SENet(BasicBlock, [2, 2, 2, 2])


def SE_ResNet34():
    return SENet(BasicBlock, [3, 4, 6, 3])


def SE_ResNet50():
    return SENet(Bottleneck, [3, 4, 6, 3])


def SE_ResNet101():
    return SENet(Bottleneck, [3, 4, 23, 3])


def SE_ResNet152():
    return SENet(Bottleneck, [3, 8, 36, 3])

if __name__=='__main__':
    model=SE_ResNet50().cuda()
    summary(model,(3,224,224))


# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 64, 224, 224]           1,792
#        BatchNorm2d-2         [-1, 64, 224, 224]             128
#               ReLU-3         [-1, 64, 224, 224]               0
#             Conv2d-4         [-1, 16, 224, 224]           1,040
#        BatchNorm2d-5         [-1, 16, 224, 224]              32
#               ReLU-6         [-1, 16, 224, 224]               0
#             Conv2d-7         [-1, 16, 224, 224]           2,320
#        BatchNorm2d-8         [-1, 16, 224, 224]              32
#               ReLU-9         [-1, 16, 224, 224]               0
#            Conv2d-10         [-1, 64, 224, 224]           1,088
#       BatchNorm2d-11         [-1, 64, 224, 224]             128
# AdaptiveAvgPool2d-12             [-1, 64, 1, 1]               0
#            Linear-13                    [-1, 4]             256
#              ReLU-14                    [-1, 4]               0
#            Linear-15                   [-1, 64]             256
#           Sigmoid-16                   [-1, 64]               0
#          SE_block-17         [-1, 64, 224, 224]               0
#        Bottleneck-18         [-1, 64, 224, 224]               0
#            Conv2d-19         [-1, 16, 224, 224]           1,040
#       BatchNorm2d-20         [-1, 16, 224, 224]              32
#              ReLU-21         [-1, 16, 224, 224]               0
#            Conv2d-22         [-1, 16, 224, 224]           2,320
#       BatchNorm2d-23         [-1, 16, 224, 224]              32
#              ReLU-24         [-1, 16, 224, 224]               0
#            Conv2d-25         [-1, 64, 224, 224]           1,088
#       BatchNorm2d-26         [-1, 64, 224, 224]             128
# AdaptiveAvgPool2d-27             [-1, 64, 1, 1]               0
#            Linear-28                    [-1, 4]             256
#              ReLU-29                    [-1, 4]               0
#            Linear-30                   [-1, 64]             256
#           Sigmoid-31                   [-1, 64]               0
#          SE_block-32         [-1, 64, 224, 224]               0
#        Bottleneck-33         [-1, 64, 224, 224]               0
#            Conv2d-34         [-1, 16, 224, 224]           1,040
#       BatchNorm2d-35         [-1, 16, 224, 224]              32
#              ReLU-36         [-1, 16, 224, 224]               0
#            Conv2d-37         [-1, 16, 224, 224]           2,320
#       BatchNorm2d-38         [-1, 16, 224, 224]              32
#              ReLU-39         [-1, 16, 224, 224]               0
#            Conv2d-40         [-1, 64, 224, 224]           1,088
#       BatchNorm2d-41         [-1, 64, 224, 224]             128
# AdaptiveAvgPool2d-42             [-1, 64, 1, 1]               0
#            Linear-43                    [-1, 4]             256
#              ReLU-44                    [-1, 4]               0
#            Linear-45                   [-1, 64]             256
#           Sigmoid-46                   [-1, 64]               0
#          SE_block-47         [-1, 64, 224, 224]               0
#        Bottleneck-48         [-1, 64, 224, 224]               0
#            Conv2d-49         [-1, 32, 224, 224]           2,080
#       BatchNorm2d-50         [-1, 32, 224, 224]              64
#              ReLU-51         [-1, 32, 224, 224]               0
#            Conv2d-52         [-1, 32, 112, 112]           9,248
#       BatchNorm2d-53         [-1, 32, 112, 112]              64
#              ReLU-54         [-1, 32, 112, 112]               0
#            Conv2d-55        [-1, 128, 112, 112]           4,224
#       BatchNorm2d-56        [-1, 128, 112, 112]             256
# AdaptiveAvgPool2d-57            [-1, 128, 1, 1]               0
#            Linear-58                    [-1, 8]           1,024
#              ReLU-59                    [-1, 8]               0
#            Linear-60                  [-1, 128]           1,024
#           Sigmoid-61                  [-1, 128]               0
#          SE_block-62        [-1, 128, 112, 112]               0
#            Conv2d-63        [-1, 128, 112, 112]           8,320
#       BatchNorm2d-64        [-1, 128, 112, 112]             256
#        Bottleneck-65        [-1, 128, 112, 112]               0
#            Conv2d-66         [-1, 32, 112, 112]           4,128
#       BatchNorm2d-67         [-1, 32, 112, 112]              64
#              ReLU-68         [-1, 32, 112, 112]               0
#            Conv2d-69         [-1, 32, 112, 112]           9,248
#       BatchNorm2d-70         [-1, 32, 112, 112]              64
#              ReLU-71         [-1, 32, 112, 112]               0
#            Conv2d-72        [-1, 128, 112, 112]           4,224
#       BatchNorm2d-73        [-1, 128, 112, 112]             256
# AdaptiveAvgPool2d-74            [-1, 128, 1, 1]               0
#            Linear-75                    [-1, 8]           1,024
#              ReLU-76                    [-1, 8]               0
#            Linear-77                  [-1, 128]           1,024
#           Sigmoid-78                  [-1, 128]               0
#          SE_block-79        [-1, 128, 112, 112]               0
#        Bottleneck-80        [-1, 128, 112, 112]               0
#            Conv2d-81         [-1, 32, 112, 112]           4,128
#       BatchNorm2d-82         [-1, 32, 112, 112]              64
#              ReLU-83         [-1, 32, 112, 112]               0
#            Conv2d-84         [-1, 32, 112, 112]           9,248
#       BatchNorm2d-85         [-1, 32, 112, 112]              64
#              ReLU-86         [-1, 32, 112, 112]               0
#            Conv2d-87        [-1, 128, 112, 112]           4,224
#       BatchNorm2d-88        [-1, 128, 112, 112]             256
# AdaptiveAvgPool2d-89            [-1, 128, 1, 1]               0
#            Linear-90                    [-1, 8]           1,024
#              ReLU-91                    [-1, 8]               0
#            Linear-92                  [-1, 128]           1,024
#           Sigmoid-93                  [-1, 128]               0
#          SE_block-94        [-1, 128, 112, 112]               0
#        Bottleneck-95        [-1, 128, 112, 112]               0
#            Conv2d-96         [-1, 32, 112, 112]           4,128
#       BatchNorm2d-97         [-1, 32, 112, 112]              64
#              ReLU-98         [-1, 32, 112, 112]               0
#            Conv2d-99         [-1, 32, 112, 112]           9,248
#      BatchNorm2d-100         [-1, 32, 112, 112]              64
#             ReLU-101         [-1, 32, 112, 112]               0
#           Conv2d-102        [-1, 128, 112, 112]           4,224
#      BatchNorm2d-103        [-1, 128, 112, 112]             256
# AdaptiveAvgPool2d-104            [-1, 128, 1, 1]               0
#           Linear-105                    [-1, 8]           1,024
#             ReLU-106                    [-1, 8]               0
#           Linear-107                  [-1, 128]           1,024
#          Sigmoid-108                  [-1, 128]               0
#         SE_block-109        [-1, 128, 112, 112]               0
#       Bottleneck-110        [-1, 128, 112, 112]               0
#           Conv2d-111         [-1, 64, 112, 112]           8,256
#      BatchNorm2d-112         [-1, 64, 112, 112]             128
#             ReLU-113         [-1, 64, 112, 112]               0
#           Conv2d-114           [-1, 64, 56, 56]          36,928
#      BatchNorm2d-115           [-1, 64, 56, 56]             128
#             ReLU-116           [-1, 64, 56, 56]               0
#           Conv2d-117          [-1, 256, 56, 56]          16,640
#      BatchNorm2d-118          [-1, 256, 56, 56]             512
# AdaptiveAvgPool2d-119            [-1, 256, 1, 1]               0
#           Linear-120                   [-1, 16]           4,096
#             ReLU-121                   [-1, 16]               0
#           Linear-122                  [-1, 256]           4,096
#          Sigmoid-123                  [-1, 256]               0
#         SE_block-124          [-1, 256, 56, 56]               0
#           Conv2d-125          [-1, 256, 56, 56]          33,024
#      BatchNorm2d-126          [-1, 256, 56, 56]             512
#       Bottleneck-127          [-1, 256, 56, 56]               0
#           Conv2d-128           [-1, 64, 56, 56]          16,448
#      BatchNorm2d-129           [-1, 64, 56, 56]             128
#             ReLU-130           [-1, 64, 56, 56]               0
#           Conv2d-131           [-1, 64, 56, 56]          36,928
#      BatchNorm2d-132           [-1, 64, 56, 56]             128
#             ReLU-133           [-1, 64, 56, 56]               0
#           Conv2d-134          [-1, 256, 56, 56]          16,640
#      BatchNorm2d-135          [-1, 256, 56, 56]             512
# AdaptiveAvgPool2d-136            [-1, 256, 1, 1]               0
#           Linear-137                   [-1, 16]           4,096
#             ReLU-138                   [-1, 16]               0
#           Linear-139                  [-1, 256]           4,096
#          Sigmoid-140                  [-1, 256]               0
#         SE_block-141          [-1, 256, 56, 56]               0
#       Bottleneck-142          [-1, 256, 56, 56]               0
#           Conv2d-143           [-1, 64, 56, 56]          16,448
#      BatchNorm2d-144           [-1, 64, 56, 56]             128
#             ReLU-145           [-1, 64, 56, 56]               0
#           Conv2d-146           [-1, 64, 56, 56]          36,928
#      BatchNorm2d-147           [-1, 64, 56, 56]             128
#             ReLU-148           [-1, 64, 56, 56]               0
#           Conv2d-149          [-1, 256, 56, 56]          16,640
#      BatchNorm2d-150          [-1, 256, 56, 56]             512
# AdaptiveAvgPool2d-151            [-1, 256, 1, 1]               0
#           Linear-152                   [-1, 16]           4,096
#             ReLU-153                   [-1, 16]               0
#           Linear-154                  [-1, 256]           4,096
#          Sigmoid-155                  [-1, 256]               0
#         SE_block-156          [-1, 256, 56, 56]               0
#       Bottleneck-157          [-1, 256, 56, 56]               0
#           Conv2d-158           [-1, 64, 56, 56]          16,448
#      BatchNorm2d-159           [-1, 64, 56, 56]             128
#             ReLU-160           [-1, 64, 56, 56]               0
#           Conv2d-161           [-1, 64, 56, 56]          36,928
#      BatchNorm2d-162           [-1, 64, 56, 56]             128
#             ReLU-163           [-1, 64, 56, 56]               0
#           Conv2d-164          [-1, 256, 56, 56]          16,640
#      BatchNorm2d-165          [-1, 256, 56, 56]             512
# AdaptiveAvgPool2d-166            [-1, 256, 1, 1]               0
#           Linear-167                   [-1, 16]           4,096
#             ReLU-168                   [-1, 16]               0
#           Linear-169                  [-1, 256]           4,096
#          Sigmoid-170                  [-1, 256]               0
#         SE_block-171          [-1, 256, 56, 56]               0
#       Bottleneck-172          [-1, 256, 56, 56]               0
#           Conv2d-173           [-1, 64, 56, 56]          16,448
#      BatchNorm2d-174           [-1, 64, 56, 56]             128
#             ReLU-175           [-1, 64, 56, 56]               0
#           Conv2d-176           [-1, 64, 56, 56]          36,928
#      BatchNorm2d-177           [-1, 64, 56, 56]             128
#             ReLU-178           [-1, 64, 56, 56]               0
#           Conv2d-179          [-1, 256, 56, 56]          16,640
#      BatchNorm2d-180          [-1, 256, 56, 56]             512
# AdaptiveAvgPool2d-181            [-1, 256, 1, 1]               0
#           Linear-182                   [-1, 16]           4,096
#             ReLU-183                   [-1, 16]               0
#           Linear-184                  [-1, 256]           4,096
#          Sigmoid-185                  [-1, 256]               0
#         SE_block-186          [-1, 256, 56, 56]               0
#       Bottleneck-187          [-1, 256, 56, 56]               0
#           Conv2d-188           [-1, 64, 56, 56]          16,448
#      BatchNorm2d-189           [-1, 64, 56, 56]             128
#             ReLU-190           [-1, 64, 56, 56]               0
#           Conv2d-191           [-1, 64, 56, 56]          36,928
#      BatchNorm2d-192           [-1, 64, 56, 56]             128
#             ReLU-193           [-1, 64, 56, 56]               0
#           Conv2d-194          [-1, 256, 56, 56]          16,640
#      BatchNorm2d-195          [-1, 256, 56, 56]             512
# AdaptiveAvgPool2d-196            [-1, 256, 1, 1]               0
#           Linear-197                   [-1, 16]           4,096
#             ReLU-198                   [-1, 16]               0
#           Linear-199                  [-1, 256]           4,096
#          Sigmoid-200                  [-1, 256]               0
#         SE_block-201          [-1, 256, 56, 56]               0
#       Bottleneck-202          [-1, 256, 56, 56]               0
#           Conv2d-203          [-1, 128, 56, 56]          32,896
#      BatchNorm2d-204          [-1, 128, 56, 56]             256
#             ReLU-205          [-1, 128, 56, 56]               0
#           Conv2d-206          [-1, 128, 28, 28]         147,584
#      BatchNorm2d-207          [-1, 128, 28, 28]             256
#             ReLU-208          [-1, 128, 28, 28]               0
#           Conv2d-209          [-1, 512, 28, 28]          66,048
#      BatchNorm2d-210          [-1, 512, 28, 28]           1,024
# AdaptiveAvgPool2d-211            [-1, 512, 1, 1]               0
#           Linear-212                   [-1, 32]          16,384
#             ReLU-213                   [-1, 32]               0
#           Linear-214                  [-1, 512]          16,384
#          Sigmoid-215                  [-1, 512]               0
#         SE_block-216          [-1, 512, 28, 28]               0
#           Conv2d-217          [-1, 512, 28, 28]         131,584
#      BatchNorm2d-218          [-1, 512, 28, 28]           1,024
#       Bottleneck-219          [-1, 512, 28, 28]               0
#           Conv2d-220          [-1, 128, 28, 28]          65,664
#      BatchNorm2d-221          [-1, 128, 28, 28]             256
#             ReLU-222          [-1, 128, 28, 28]               0
#           Conv2d-223          [-1, 128, 28, 28]         147,584
#      BatchNorm2d-224          [-1, 128, 28, 28]             256
#             ReLU-225          [-1, 128, 28, 28]               0
#           Conv2d-226          [-1, 512, 28, 28]          66,048
#      BatchNorm2d-227          [-1, 512, 28, 28]           1,024
# AdaptiveAvgPool2d-228            [-1, 512, 1, 1]               0
#           Linear-229                   [-1, 32]          16,384
#             ReLU-230                   [-1, 32]               0
#           Linear-231                  [-1, 512]          16,384
#          Sigmoid-232                  [-1, 512]               0
#         SE_block-233          [-1, 512, 28, 28]               0
#       Bottleneck-234          [-1, 512, 28, 28]               0
#           Conv2d-235          [-1, 128, 28, 28]          65,664
#      BatchNorm2d-236          [-1, 128, 28, 28]             256
#             ReLU-237          [-1, 128, 28, 28]               0
#           Conv2d-238          [-1, 128, 28, 28]         147,584
#      BatchNorm2d-239          [-1, 128, 28, 28]             256
#             ReLU-240          [-1, 128, 28, 28]               0
#           Conv2d-241          [-1, 512, 28, 28]          66,048
#      BatchNorm2d-242          [-1, 512, 28, 28]           1,024
# AdaptiveAvgPool2d-243            [-1, 512, 1, 1]               0
#           Linear-244                   [-1, 32]          16,384
#             ReLU-245                   [-1, 32]               0
#           Linear-246                  [-1, 512]          16,384
#          Sigmoid-247                  [-1, 512]               0
#         SE_block-248          [-1, 512, 28, 28]               0
#       Bottleneck-249          [-1, 512, 28, 28]               0
# AdaptiveAvgPool2d-250            [-1, 512, 1, 1]               0
#           Linear-251                   [-1, 10]           5,130
# ================================================================
# Total params: 1,649,002
# Trainable params: 1,649,002
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 1091.11
# Params size (MB): 6.29
# Estimated Total Size (MB): 1097.97
# ----------------------------------------------------------------