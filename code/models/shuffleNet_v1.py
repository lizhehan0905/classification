import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# channel重组操作
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
 
    # 进行维度的变换操作
    def forward(self, x):
        # Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


class shuffle_block(nn.Module):
    def __init__(self,inchannel,outchannel,groups,stride):
        super(shuffle_block,self).__init__()
        self.stride=stride

        if self.stride>1:
            outchannel=outchannel-inchannel
        

        self.gconv1=nn.Sequential(
            nn.Conv2d(inchannel,inchannel//4,kernel_size=1,groups=groups),
            nn.BatchNorm2d(inchannel//4),
            nn.ReLU(inplace=True),
        )
        
        self.shuffle_channel=ChannelShuffle(groups)

        self.dwconv=nn.Sequential(
            nn.Conv2d(inchannel//4,outchannel,kernel_size=3,stride=stride,padding=1,groups=groups),
            nn.BatchNorm2d(outchannel),
        )

        self.gconv2=nn.Sequential(
            nn.Conv2d(outchannel,outchannel,kernel_size=1,groups=groups),
            nn.BatchNorm2d(outchannel),
        )
        
        self.shortcut=nn.Sequential()
        if self.stride>1:
            self.shortcut=nn.AvgPool2d(kernel_size=3,stride=2,padding=1)


        self.relu=nn.ReLU6(inplace=True)

    def forward(self,x):
        out=self.gconv1(x)
        out=self.shuffle_channel(out)
        out=self.dwconv(out)
        out=self.gconv2(out)
        if self.stride>1:
            out=[out,self.shortcut(x)]
            out=torch.cat(out,1)
        else:
            out+=self.shortcut(x)
        out=self.relu(out)
        return out
    
class shuffleNet_v1(nn.Module):
    def __init__(self,num_blocks,num_classes=10):
        super(shuffleNet_v1,self).__init__()
        self.inchannel=24
        
        self.conv1=nn.Sequential(
            nn.Conv2d(3,24,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
           
        )

        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.stage1=self.make_layer(144,shuffle_block,1,2,num_blocks[0])
        self.stage2=self.make_layer(144,shuffle_block,1,1,num_blocks[1])
        
        self.stage3=self.make_layer(288,shuffle_block,1,2,num_blocks[2])
        self.stage4=self.make_layer(288,shuffle_block,1,1,num_blocks[3])

        self.stage5=self.make_layer(576,shuffle_block,1,2,num_blocks[4])
        self.stage6=self.make_layer(576,shuffle_block,1,1,num_blocks[5])

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(576,num_classes)


    def make_layer(self,channel,shuffle_block,groups,stride,num_block):
        strides=[stride]+[1]*(num_block-1)
        layers=[]
        for stride in strides:
            layers.append(shuffle_block(self.inchannel,channel,groups=groups,stride=stride))
            self.inchannel=channel
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out=self.conv1(x)
        out=self.maxpool(out)
        out=self.stage1(out)
        out=self.stage2(out)
        out=self.stage3(out)
        out=self.stage4(out)
        out=self.stage5(out)
        out=self.stage6(out)
        out=self.avgpool(out)
        out=torch.flatten(out,1)
        out=self.fc(out)
        return out

if __name__=='__main__':
    net=shuffleNet_v1([1,3,1,7,1,3]).cuda()
    summary(net,(3,224,224))


# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 24, 112, 112]             672
#        BatchNorm2d-2         [-1, 24, 112, 112]              48
#               ReLU-3         [-1, 24, 112, 112]               0
#          MaxPool2d-4           [-1, 24, 56, 56]               0
#             Conv2d-5            [-1, 6, 56, 56]             150
#        BatchNorm2d-6            [-1, 6, 56, 56]              12
#               ReLU-7            [-1, 6, 56, 56]               0
#     ChannelShuffle-8            [-1, 6, 56, 56]               0
#             Conv2d-9          [-1, 120, 28, 28]           6,600
#       BatchNorm2d-10          [-1, 120, 28, 28]             240
#            Conv2d-11          [-1, 120, 28, 28]          14,520
#       BatchNorm2d-12          [-1, 120, 28, 28]             240
#         AvgPool2d-13           [-1, 24, 28, 28]               0
#             ReLU6-14          [-1, 144, 28, 28]               0
#     shuffle_block-15          [-1, 144, 28, 28]               0
#            Conv2d-16           [-1, 36, 28, 28]           5,220
#       BatchNorm2d-17           [-1, 36, 28, 28]              72
#              ReLU-18           [-1, 36, 28, 28]               0
#    ChannelShuffle-19           [-1, 36, 28, 28]               0
#            Conv2d-20          [-1, 144, 28, 28]          46,800
#       BatchNorm2d-21          [-1, 144, 28, 28]             288
#            Conv2d-22          [-1, 144, 28, 28]          20,880
#       BatchNorm2d-23          [-1, 144, 28, 28]             288
#             ReLU6-24          [-1, 144, 28, 28]               0
#     shuffle_block-25          [-1, 144, 28, 28]               0
#            Conv2d-26           [-1, 36, 28, 28]           5,220
#       BatchNorm2d-27           [-1, 36, 28, 28]              72
#              ReLU-28           [-1, 36, 28, 28]               0
#    ChannelShuffle-29           [-1, 36, 28, 28]               0
#            Conv2d-30          [-1, 144, 28, 28]          46,800
#       BatchNorm2d-31          [-1, 144, 28, 28]             288
#            Conv2d-32          [-1, 144, 28, 28]          20,880
#       BatchNorm2d-33          [-1, 144, 28, 28]             288
#             ReLU6-34          [-1, 144, 28, 28]               0
#     shuffle_block-35          [-1, 144, 28, 28]               0
#            Conv2d-36           [-1, 36, 28, 28]           5,220
#       BatchNorm2d-37           [-1, 36, 28, 28]              72
#              ReLU-38           [-1, 36, 28, 28]               0
#    ChannelShuffle-39           [-1, 36, 28, 28]               0
#            Conv2d-40          [-1, 144, 28, 28]          46,800
#       BatchNorm2d-41          [-1, 144, 28, 28]             288
#            Conv2d-42          [-1, 144, 28, 28]          20,880
#       BatchNorm2d-43          [-1, 144, 28, 28]             288
#             ReLU6-44          [-1, 144, 28, 28]               0
#     shuffle_block-45          [-1, 144, 28, 28]               0
#            Conv2d-46           [-1, 36, 28, 28]           5,220
#       BatchNorm2d-47           [-1, 36, 28, 28]              72
#              ReLU-48           [-1, 36, 28, 28]               0
#    ChannelShuffle-49           [-1, 36, 28, 28]               0
#            Conv2d-50          [-1, 144, 14, 14]          46,800
#       BatchNorm2d-51          [-1, 144, 14, 14]             288
#            Conv2d-52          [-1, 144, 14, 14]          20,880
#       BatchNorm2d-53          [-1, 144, 14, 14]             288
#         AvgPool2d-54          [-1, 144, 14, 14]               0
#             ReLU6-55          [-1, 288, 14, 14]               0
#     shuffle_block-56          [-1, 288, 14, 14]               0
#            Conv2d-57           [-1, 72, 14, 14]          20,808
#       BatchNorm2d-58           [-1, 72, 14, 14]             144
#              ReLU-59           [-1, 72, 14, 14]               0
#    ChannelShuffle-60           [-1, 72, 14, 14]               0
#            Conv2d-61          [-1, 288, 14, 14]         186,912
#       BatchNorm2d-62          [-1, 288, 14, 14]             576
#            Conv2d-63          [-1, 288, 14, 14]          83,232
#       BatchNorm2d-64          [-1, 288, 14, 14]             576
#             ReLU6-65          [-1, 288, 14, 14]               0
#     shuffle_block-66          [-1, 288, 14, 14]               0
#            Conv2d-67           [-1, 72, 14, 14]          20,808
#       BatchNorm2d-68           [-1, 72, 14, 14]             144
#              ReLU-69           [-1, 72, 14, 14]               0
#    ChannelShuffle-70           [-1, 72, 14, 14]               0
#            Conv2d-71          [-1, 288, 14, 14]         186,912
#       BatchNorm2d-72          [-1, 288, 14, 14]             576
#            Conv2d-73          [-1, 288, 14, 14]          83,232
#       BatchNorm2d-74          [-1, 288, 14, 14]             576
#             ReLU6-75          [-1, 288, 14, 14]               0
#     shuffle_block-76          [-1, 288, 14, 14]               0
#            Conv2d-77           [-1, 72, 14, 14]          20,808
#       BatchNorm2d-78           [-1, 72, 14, 14]             144
#              ReLU-79           [-1, 72, 14, 14]               0
#    ChannelShuffle-80           [-1, 72, 14, 14]               0
#            Conv2d-81          [-1, 288, 14, 14]         186,912
#       BatchNorm2d-82          [-1, 288, 14, 14]             576
#            Conv2d-83          [-1, 288, 14, 14]          83,232
#       BatchNorm2d-84          [-1, 288, 14, 14]             576
#             ReLU6-85          [-1, 288, 14, 14]               0
#     shuffle_block-86          [-1, 288, 14, 14]               0
#            Conv2d-87           [-1, 72, 14, 14]          20,808
#       BatchNorm2d-88           [-1, 72, 14, 14]             144
#              ReLU-89           [-1, 72, 14, 14]               0
#    ChannelShuffle-90           [-1, 72, 14, 14]               0
#            Conv2d-91          [-1, 288, 14, 14]         186,912
#       BatchNorm2d-92          [-1, 288, 14, 14]             576
#            Conv2d-93          [-1, 288, 14, 14]          83,232
#       BatchNorm2d-94          [-1, 288, 14, 14]             576
#             ReLU6-95          [-1, 288, 14, 14]               0
#     shuffle_block-96          [-1, 288, 14, 14]               0
#            Conv2d-97           [-1, 72, 14, 14]          20,808
#       BatchNorm2d-98           [-1, 72, 14, 14]             144
#              ReLU-99           [-1, 72, 14, 14]               0
#   ChannelShuffle-100           [-1, 72, 14, 14]               0
#           Conv2d-101          [-1, 288, 14, 14]         186,912
#      BatchNorm2d-102          [-1, 288, 14, 14]             576
#           Conv2d-103          [-1, 288, 14, 14]          83,232
#      BatchNorm2d-104          [-1, 288, 14, 14]             576
#            ReLU6-105          [-1, 288, 14, 14]               0
#    shuffle_block-106          [-1, 288, 14, 14]               0
#           Conv2d-107           [-1, 72, 14, 14]          20,808
#      BatchNorm2d-108           [-1, 72, 14, 14]             144
#             ReLU-109           [-1, 72, 14, 14]               0
#   ChannelShuffle-110           [-1, 72, 14, 14]               0
#           Conv2d-111          [-1, 288, 14, 14]         186,912
#      BatchNorm2d-112          [-1, 288, 14, 14]             576
#           Conv2d-113          [-1, 288, 14, 14]          83,232
#      BatchNorm2d-114          [-1, 288, 14, 14]             576
#            ReLU6-115          [-1, 288, 14, 14]               0
#    shuffle_block-116          [-1, 288, 14, 14]               0
#           Conv2d-117           [-1, 72, 14, 14]          20,808
#      BatchNorm2d-118           [-1, 72, 14, 14]             144
#             ReLU-119           [-1, 72, 14, 14]               0
#   ChannelShuffle-120           [-1, 72, 14, 14]               0
#           Conv2d-121          [-1, 288, 14, 14]         186,912
#      BatchNorm2d-122          [-1, 288, 14, 14]             576
#           Conv2d-123          [-1, 288, 14, 14]          83,232
#      BatchNorm2d-124          [-1, 288, 14, 14]             576
#            ReLU6-125          [-1, 288, 14, 14]               0
#    shuffle_block-126          [-1, 288, 14, 14]               0
#           Conv2d-127           [-1, 72, 14, 14]          20,808
#      BatchNorm2d-128           [-1, 72, 14, 14]             144
#             ReLU-129           [-1, 72, 14, 14]               0
#   ChannelShuffle-130           [-1, 72, 14, 14]               0
#           Conv2d-131            [-1, 288, 7, 7]         186,912
#      BatchNorm2d-132            [-1, 288, 7, 7]             576
#           Conv2d-133            [-1, 288, 7, 7]          83,232
#      BatchNorm2d-134            [-1, 288, 7, 7]             576
#        AvgPool2d-135            [-1, 288, 7, 7]               0
#            ReLU6-136            [-1, 576, 7, 7]               0
#    shuffle_block-137            [-1, 576, 7, 7]               0
#           Conv2d-138            [-1, 144, 7, 7]          83,088
#      BatchNorm2d-139            [-1, 144, 7, 7]             288
#             ReLU-140            [-1, 144, 7, 7]               0
#   ChannelShuffle-141            [-1, 144, 7, 7]               0
#           Conv2d-142            [-1, 576, 7, 7]         747,072
#      BatchNorm2d-143            [-1, 576, 7, 7]           1,152
#           Conv2d-144            [-1, 576, 7, 7]         332,352
#      BatchNorm2d-145            [-1, 576, 7, 7]           1,152
#            ReLU6-146            [-1, 576, 7, 7]               0
#    shuffle_block-147            [-1, 576, 7, 7]               0
#           Conv2d-148            [-1, 144, 7, 7]          83,088
#      BatchNorm2d-149            [-1, 144, 7, 7]             288
#             ReLU-150            [-1, 144, 7, 7]               0
#   ChannelShuffle-151            [-1, 144, 7, 7]               0
#           Conv2d-152            [-1, 576, 7, 7]         747,072
#      BatchNorm2d-153            [-1, 576, 7, 7]           1,152
#           Conv2d-154            [-1, 576, 7, 7]         332,352
#      BatchNorm2d-155            [-1, 576, 7, 7]           1,152
#            ReLU6-156            [-1, 576, 7, 7]               0
#    shuffle_block-157            [-1, 576, 7, 7]               0
#           Conv2d-158            [-1, 144, 7, 7]          83,088
#      BatchNorm2d-159            [-1, 144, 7, 7]             288
#             ReLU-160            [-1, 144, 7, 7]               0
#   ChannelShuffle-161            [-1, 144, 7, 7]               0
#           Conv2d-162            [-1, 576, 7, 7]         747,072
#      BatchNorm2d-163            [-1, 576, 7, 7]           1,152
#           Conv2d-164            [-1, 576, 7, 7]         332,352
#      BatchNorm2d-165            [-1, 576, 7, 7]           1,152
#            ReLU6-166            [-1, 576, 7, 7]               0
#    shuffle_block-167            [-1, 576, 7, 7]               0
# AdaptiveAvgPool2d-168            [-1, 576, 1, 1]               0
#           Linear-169                   [-1, 10]           5,770
# ================================================================
# Total params: 6,155,740
# Trainable params: 6,155,740
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 60.69
# Params size (MB): 23.48
# Estimated Total Size (MB): 84.75
# ----------------------------------------------------------------
