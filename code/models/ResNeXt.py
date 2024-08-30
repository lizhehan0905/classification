import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Bottleneck(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1,groups=32,width_per_group=4):
        super(Bottleneck,self).__init__()

        # groups=1,width_per_group=64为正常的ResNet


        self.left=nn.Sequential(
            nn.Conv2d(inchannel,outchannel//2,kernel_size=1,stride=1),
            nn.BatchNorm2d(outchannel//2),
            nn.ReLU(inplace=True),

            # 加了groups参数为ResNeXt，不加为ResNet
            nn.Conv2d(outchannel//2,outchannel//2,kernel_size=3,groups=groups,stride=stride,padding=1),
            nn.BatchNorm2d(outchannel//2),
            nn.ReLU(inplace=True),

            nn.Conv2d(outchannel//2,outchannel,kernel_size=1,stride=1),
            nn.BatchNorm2d(outchannel),
        )

        self.shortcut=nn.Sequential()

        if stride!=1 or inchannel!=outchannel:
            self.shortcut=nn.Sequential(
                nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=stride),
                nn.BatchNorm2d(outchannel),
            )


    def forward(self,x):
        out=self.left(x)
        out+=self.shortcut(x)
        out=F.relu(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self,Bottleneck,num_blocks,num_classes=10,groups=32,width_per_group=4,):
        super(ResNeXt,self).__init__()
        self.inchannel=64

        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3)
        self.BN=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.pool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=self.make_layer(Bottleneck,256,num_blocks[0],stride=1)
        self.layer2=self.make_layer(Bottleneck,512,num_blocks[1],stride=2)
        self.layer3=self.make_layer(Bottleneck,1024,num_blocks[2],stride=2)
        self.layer4=self.make_layer(Bottleneck,2048,num_blocks[3],stride=2)

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(2048,num_classes)


    def make_layer(self,Bottleneck,channel,num_block,stride=1):
        strides=[stride]+[1]*(num_block-1)
        layers=[]
        for stride in strides:
            layers.append(Bottleneck(self.inchannel,channel,groups=32,width_per_group=4,stride=stride))
            self.inchannel=channel
        return nn.Sequential(*layers)


    def forward(self,x):
        out = self.conv1(x)
        out = self.BN(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out,1)
        out = self.fc(out)
        return out


def ResNeXt_50():
    return ResNeXt(Bottleneck,[3,4,6,3],num_classes=10)

if __name__=='__main__':
    model=ResNeXt_50().cuda()
    summary(model,(3,224,224))


# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 64, 112, 112]           9,472
#        BatchNorm2d-2         [-1, 64, 112, 112]             128
#               ReLU-3         [-1, 64, 112, 112]               0
#          MaxPool2d-4           [-1, 64, 56, 56]               0
#             Conv2d-5          [-1, 128, 56, 56]           8,320
#        BatchNorm2d-6          [-1, 128, 56, 56]             256
#               ReLU-7          [-1, 128, 56, 56]               0
#             Conv2d-8          [-1, 128, 56, 56]           4,736
#        BatchNorm2d-9          [-1, 128, 56, 56]             256
#              ReLU-10          [-1, 128, 56, 56]               0
#            Conv2d-11          [-1, 256, 56, 56]          33,024
#       BatchNorm2d-12          [-1, 256, 56, 56]             512
#            Conv2d-13          [-1, 256, 56, 56]          16,640
#       BatchNorm2d-14          [-1, 256, 56, 56]             512
#        Bottleneck-15          [-1, 256, 56, 56]               0
#            Conv2d-16          [-1, 128, 56, 56]          32,896
#       BatchNorm2d-17          [-1, 128, 56, 56]             256
#              ReLU-18          [-1, 128, 56, 56]               0
#            Conv2d-19          [-1, 128, 56, 56]           4,736
#       BatchNorm2d-20          [-1, 128, 56, 56]             256
#              ReLU-21          [-1, 128, 56, 56]               0
#            Conv2d-22          [-1, 256, 56, 56]          33,024
#       BatchNorm2d-23          [-1, 256, 56, 56]             512
#        Bottleneck-24          [-1, 256, 56, 56]               0
#            Conv2d-25          [-1, 128, 56, 56]          32,896
#       BatchNorm2d-26          [-1, 128, 56, 56]             256
#              ReLU-27          [-1, 128, 56, 56]               0
#            Conv2d-28          [-1, 128, 56, 56]           4,736
#       BatchNorm2d-29          [-1, 128, 56, 56]             256
#              ReLU-30          [-1, 128, 56, 56]               0
#            Conv2d-31          [-1, 256, 56, 56]          33,024
#       BatchNorm2d-32          [-1, 256, 56, 56]             512
#        Bottleneck-33          [-1, 256, 56, 56]               0
#            Conv2d-34          [-1, 256, 56, 56]          65,792
#       BatchNorm2d-35          [-1, 256, 56, 56]             512
#              ReLU-36          [-1, 256, 56, 56]               0
#            Conv2d-37          [-1, 256, 28, 28]          18,688
#       BatchNorm2d-38          [-1, 256, 28, 28]             512
#              ReLU-39          [-1, 256, 28, 28]               0
#            Conv2d-40          [-1, 512, 28, 28]         131,584
#       BatchNorm2d-41          [-1, 512, 28, 28]           1,024
#            Conv2d-42          [-1, 512, 28, 28]         131,584
#       BatchNorm2d-43          [-1, 512, 28, 28]           1,024
#        Bottleneck-44          [-1, 512, 28, 28]               0
#            Conv2d-45          [-1, 256, 28, 28]         131,328
#       BatchNorm2d-46          [-1, 256, 28, 28]             512
#              ReLU-47          [-1, 256, 28, 28]               0
#            Conv2d-48          [-1, 256, 28, 28]          18,688
#       BatchNorm2d-49          [-1, 256, 28, 28]             512
#              ReLU-50          [-1, 256, 28, 28]               0
#            Conv2d-51          [-1, 512, 28, 28]         131,584
#       BatchNorm2d-52          [-1, 512, 28, 28]           1,024
#        Bottleneck-53          [-1, 512, 28, 28]               0
#            Conv2d-54          [-1, 256, 28, 28]         131,328
#       BatchNorm2d-55          [-1, 256, 28, 28]             512
#              ReLU-56          [-1, 256, 28, 28]               0
#            Conv2d-57          [-1, 256, 28, 28]          18,688
#       BatchNorm2d-58          [-1, 256, 28, 28]             512
#              ReLU-59          [-1, 256, 28, 28]               0
#            Conv2d-60          [-1, 512, 28, 28]         131,584
#       BatchNorm2d-61          [-1, 512, 28, 28]           1,024
#        Bottleneck-62          [-1, 512, 28, 28]               0
#            Conv2d-63          [-1, 256, 28, 28]         131,328
#       BatchNorm2d-64          [-1, 256, 28, 28]             512
#              ReLU-65          [-1, 256, 28, 28]               0
#            Conv2d-66          [-1, 256, 28, 28]          18,688
#       BatchNorm2d-67          [-1, 256, 28, 28]             512
#              ReLU-68          [-1, 256, 28, 28]               0
#            Conv2d-69          [-1, 512, 28, 28]         131,584
#       BatchNorm2d-70          [-1, 512, 28, 28]           1,024
#        Bottleneck-71          [-1, 512, 28, 28]               0
#            Conv2d-72          [-1, 512, 28, 28]         262,656
#       BatchNorm2d-73          [-1, 512, 28, 28]           1,024
#              ReLU-74          [-1, 512, 28, 28]               0
#            Conv2d-75          [-1, 512, 14, 14]          74,240
#       BatchNorm2d-76          [-1, 512, 14, 14]           1,024
#              ReLU-77          [-1, 512, 14, 14]               0
#            Conv2d-78         [-1, 1024, 14, 14]         525,312
#       BatchNorm2d-79         [-1, 1024, 14, 14]           2,048
#            Conv2d-80         [-1, 1024, 14, 14]         525,312
#       BatchNorm2d-81         [-1, 1024, 14, 14]           2,048
#        Bottleneck-82         [-1, 1024, 14, 14]               0
#            Conv2d-83          [-1, 512, 14, 14]         524,800
#       BatchNorm2d-84          [-1, 512, 14, 14]           1,024
#              ReLU-85          [-1, 512, 14, 14]               0
#            Conv2d-86          [-1, 512, 14, 14]          74,240
#       BatchNorm2d-87          [-1, 512, 14, 14]           1,024
#              ReLU-88          [-1, 512, 14, 14]               0
#            Conv2d-89         [-1, 1024, 14, 14]         525,312
#       BatchNorm2d-90         [-1, 1024, 14, 14]           2,048
#        Bottleneck-91         [-1, 1024, 14, 14]               0
#            Conv2d-92          [-1, 512, 14, 14]         524,800
#       BatchNorm2d-93          [-1, 512, 14, 14]           1,024
#              ReLU-94          [-1, 512, 14, 14]               0
#            Conv2d-95          [-1, 512, 14, 14]          74,240
#       BatchNorm2d-96          [-1, 512, 14, 14]           1,024
#              ReLU-97          [-1, 512, 14, 14]               0
#            Conv2d-98         [-1, 1024, 14, 14]         525,312
#       BatchNorm2d-99         [-1, 1024, 14, 14]           2,048
#       Bottleneck-100         [-1, 1024, 14, 14]               0
#           Conv2d-101          [-1, 512, 14, 14]         524,800
#      BatchNorm2d-102          [-1, 512, 14, 14]           1,024
#             ReLU-103          [-1, 512, 14, 14]               0
#           Conv2d-104          [-1, 512, 14, 14]          74,240
#      BatchNorm2d-105          [-1, 512, 14, 14]           1,024
#             ReLU-106          [-1, 512, 14, 14]               0
#           Conv2d-107         [-1, 1024, 14, 14]         525,312
#      BatchNorm2d-108         [-1, 1024, 14, 14]           2,048
#       Bottleneck-109         [-1, 1024, 14, 14]               0
#           Conv2d-110          [-1, 512, 14, 14]         524,800
#      BatchNorm2d-111          [-1, 512, 14, 14]           1,024
#             ReLU-112          [-1, 512, 14, 14]               0
#           Conv2d-113          [-1, 512, 14, 14]          74,240
#      BatchNorm2d-114          [-1, 512, 14, 14]           1,024
#             ReLU-115          [-1, 512, 14, 14]               0
#           Conv2d-116         [-1, 1024, 14, 14]         525,312
#      BatchNorm2d-117         [-1, 1024, 14, 14]           2,048
#       Bottleneck-118         [-1, 1024, 14, 14]               0
#           Conv2d-119          [-1, 512, 14, 14]         524,800
#      BatchNorm2d-120          [-1, 512, 14, 14]           1,024
#             ReLU-121          [-1, 512, 14, 14]               0
#           Conv2d-122          [-1, 512, 14, 14]          74,240
#      BatchNorm2d-123          [-1, 512, 14, 14]           1,024
#             ReLU-124          [-1, 512, 14, 14]               0
#           Conv2d-125         [-1, 1024, 14, 14]         525,312
#      BatchNorm2d-126         [-1, 1024, 14, 14]           2,048
#       Bottleneck-127         [-1, 1024, 14, 14]               0
#           Conv2d-128         [-1, 1024, 14, 14]       1,049,600
#      BatchNorm2d-129         [-1, 1024, 14, 14]           2,048
#             ReLU-130         [-1, 1024, 14, 14]               0
#           Conv2d-131           [-1, 1024, 7, 7]         295,936
#      BatchNorm2d-132           [-1, 1024, 7, 7]           2,048
#             ReLU-133           [-1, 1024, 7, 7]               0
#           Conv2d-134           [-1, 2048, 7, 7]       2,099,200
#      BatchNorm2d-135           [-1, 2048, 7, 7]           4,096
#           Conv2d-136           [-1, 2048, 7, 7]       2,099,200
#      BatchNorm2d-137           [-1, 2048, 7, 7]           4,096
#       Bottleneck-138           [-1, 2048, 7, 7]               0
#           Conv2d-139           [-1, 1024, 7, 7]       2,098,176
#      BatchNorm2d-140           [-1, 1024, 7, 7]           2,048
#             ReLU-141           [-1, 1024, 7, 7]               0
#           Conv2d-142           [-1, 1024, 7, 7]         295,936
#      BatchNorm2d-143           [-1, 1024, 7, 7]           2,048
#             ReLU-144           [-1, 1024, 7, 7]               0
#           Conv2d-145           [-1, 2048, 7, 7]       2,099,200
#      BatchNorm2d-146           [-1, 2048, 7, 7]           4,096
#       Bottleneck-147           [-1, 2048, 7, 7]               0
#           Conv2d-148           [-1, 1024, 7, 7]       2,098,176
#      BatchNorm2d-149           [-1, 1024, 7, 7]           2,048
#             ReLU-150           [-1, 1024, 7, 7]               0
#           Conv2d-151           [-1, 1024, 7, 7]         295,936
#      BatchNorm2d-152           [-1, 1024, 7, 7]           2,048
#             ReLU-153           [-1, 1024, 7, 7]               0
#           Conv2d-154           [-1, 2048, 7, 7]       2,099,200
#      BatchNorm2d-155           [-1, 2048, 7, 7]           4,096
#       Bottleneck-156           [-1, 2048, 7, 7]               0
# AdaptiveAvgPool2d-157           [-1, 2048, 1, 1]               0
#           Linear-158                   [-1, 10]          20,490
# ================================================================
# Total params: 23,034,506
# Trainable params: 23,034,506
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 319.66
# Params size (MB): 87.87
# Estimated Total Size (MB): 408.11
# ----------------------------------------------------------------