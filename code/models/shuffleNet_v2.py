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

# 通道拆分
class ChannelSplit(nn.Module):
    def __init__(self,dim=0,first_half=True):
        super(ChannelSplit,self).__init__()
        self.first_half=first_half
        self.dim=dim
    
    def forward(self,x):
        # 由于shape=[b,c,g,w],对于dim=1，针对channels
        # torch.chunk() 是 PyTorch 中的一个函数,它可以沿着指定的维度将一个张量拆分成多个较小的张量。
        # 原始张量的形状是 (4, 6, 8)。我们将它沿着第二个维度(维度1)拆分成两个形状为 (4, 3, 8) 的块
        splits=torch.chunk(x,2,dim=self.dim)
        # 返回其中一半
        return splits[0] if self.first_half else splits[1]

class shuffle_block(nn.Module):
    def __init__(self,inchannel,outchannel,groups,stride):
        super(shuffle_block,self).__init__()
        self.stride=stride

        if self.stride>1:
            outchannel=outchannel-inchannel

            self.right=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),

            nn.Conv2d(outchannel,outchannel,kernel_size=3,stride=stride,padding=1,groups=groups),
            nn.BatchNorm2d(outchannel),

            nn.Conv2d(outchannel,outchannel,kernel_size=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
        )
        else:
            self.x1=ChannelSplit(1,first_half=True)
            self.x2=ChannelSplit(1,first_half=False)
            inchannel=outchannel//2

            self.right=nn.Sequential(
            nn.Conv2d(inchannel,inchannel,kernel_size=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),

            nn.Conv2d(inchannel,inchannel,kernel_size=3,stride=stride,padding=1,groups=groups),
            nn.BatchNorm2d(inchannel),

            nn.Conv2d(inchannel,inchannel,kernel_size=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
        )
        
        
        
        self.shortcut=nn.Sequential()
        if self.stride>1:
            self.shortcut=nn.Sequential(
                nn.Conv2d(outchannel,outchannel,kernel_size=3,stride=stride,padding=1,groups=groups),
                nn.BatchNorm2d(outchannel),

                nn.Conv2d(outchannel,outchannel,kernel_size=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=True),
            )

        self.shuffle_channel=ChannelShuffle(groups)

    def forward(self,x):
        if self.stride>1:
            out=self.right(x)
            out1=self.shortcut(x)
        else:
            x1=self.x1(x)
            x2=self.x2(x)
            out=self.right(x1)
            out1=self.shortcut(x2)
        out=torch.cat([out,out1],1)
        out=self.shuffle_channel(out)
        return out
    
class shuffleNet_v2(nn.Module):
    def __init__(self,num_blocks,num_classes=100):
        super(shuffleNet_v2,self).__init__()
        self.inchannel=24
        
        self.conv1=nn.Sequential(
            nn.Conv2d(3,24,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )

        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.stage1=self.make_layer(48,shuffle_block,1,2,num_blocks[0])
        self.stage2=self.make_layer(48,shuffle_block,1,1,num_blocks[1])
        
        self.stage3=self.make_layer(96,shuffle_block,1,2,num_blocks[2])
        self.stage4=self.make_layer(96,shuffle_block,1,1,num_blocks[3])

        self.stage5=self.make_layer(192,shuffle_block,1,2,num_blocks[4])
        self.stage6=self.make_layer(192,shuffle_block,1,1,num_blocks[5])

        self.conv=nn.Conv2d(192,1024,kernel_size=1,stride=1)

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(1024,num_classes)


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
        out=self.conv(out)
        out=self.avgpool(out)
        out=torch.flatten(out,1)
        out=self.fc(out)
        return out

if __name__=='__main__':
    net=shuffleNet_v2([1,3,1,7,1,3]).cuda()
    summary(net,(3,224,224))

# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 24, 112, 112]             672
#        BatchNorm2d-2         [-1, 24, 112, 112]              48
#               ReLU-3         [-1, 24, 112, 112]               0
#          MaxPool2d-4           [-1, 24, 56, 56]               0
#             Conv2d-5           [-1, 24, 56, 56]             600
#        BatchNorm2d-6           [-1, 24, 56, 56]              48
#               ReLU-7           [-1, 24, 56, 56]               0
#             Conv2d-8           [-1, 24, 28, 28]           5,208
#        BatchNorm2d-9           [-1, 24, 28, 28]              48
#            Conv2d-10           [-1, 24, 28, 28]             600
#       BatchNorm2d-11           [-1, 24, 28, 28]              48
#              ReLU-12           [-1, 24, 28, 28]               0
#            Conv2d-13           [-1, 24, 28, 28]           5,208
#       BatchNorm2d-14           [-1, 24, 28, 28]              48
#            Conv2d-15           [-1, 24, 28, 28]             600
#       BatchNorm2d-16           [-1, 24, 28, 28]              48
#              ReLU-17           [-1, 24, 28, 28]               0
#    ChannelShuffle-18           [-1, 48, 28, 28]               0
#     shuffle_block-19           [-1, 48, 28, 28]               0
#      ChannelSplit-20           [-1, 24, 28, 28]               0
#      ChannelSplit-21           [-1, 24, 28, 28]               0
#            Conv2d-22           [-1, 24, 28, 28]             600
#       BatchNorm2d-23           [-1, 24, 28, 28]              48
#              ReLU-24           [-1, 24, 28, 28]               0
#            Conv2d-25           [-1, 24, 28, 28]           5,208
#       BatchNorm2d-26           [-1, 24, 28, 28]              48
#            Conv2d-27           [-1, 24, 28, 28]             600
#       BatchNorm2d-28           [-1, 24, 28, 28]              48
#              ReLU-29           [-1, 24, 28, 28]               0
#    ChannelShuffle-30           [-1, 48, 28, 28]               0
#     shuffle_block-31           [-1, 48, 28, 28]               0
#      ChannelSplit-32           [-1, 24, 28, 28]               0
#      ChannelSplit-33           [-1, 24, 28, 28]               0
#            Conv2d-34           [-1, 24, 28, 28]             600
#       BatchNorm2d-35           [-1, 24, 28, 28]              48
#              ReLU-36           [-1, 24, 28, 28]               0
#            Conv2d-37           [-1, 24, 28, 28]           5,208
#       BatchNorm2d-38           [-1, 24, 28, 28]              48
#            Conv2d-39           [-1, 24, 28, 28]             600
#       BatchNorm2d-40           [-1, 24, 28, 28]              48
#              ReLU-41           [-1, 24, 28, 28]               0
#    ChannelShuffle-42           [-1, 48, 28, 28]               0
#     shuffle_block-43           [-1, 48, 28, 28]               0
#      ChannelSplit-44           [-1, 24, 28, 28]               0
#      ChannelSplit-45           [-1, 24, 28, 28]               0
#            Conv2d-46           [-1, 24, 28, 28]             600
#       BatchNorm2d-47           [-1, 24, 28, 28]              48
#              ReLU-48           [-1, 24, 28, 28]               0
#            Conv2d-49           [-1, 24, 28, 28]           5,208
#       BatchNorm2d-50           [-1, 24, 28, 28]              48
#            Conv2d-51           [-1, 24, 28, 28]             600
#       BatchNorm2d-52           [-1, 24, 28, 28]              48
#              ReLU-53           [-1, 24, 28, 28]               0
#    ChannelShuffle-54           [-1, 48, 28, 28]               0
#     shuffle_block-55           [-1, 48, 28, 28]               0
#            Conv2d-56           [-1, 48, 28, 28]           2,352
#       BatchNorm2d-57           [-1, 48, 28, 28]              96
#              ReLU-58           [-1, 48, 28, 28]               0
#            Conv2d-59           [-1, 48, 14, 14]          20,784
#       BatchNorm2d-60           [-1, 48, 14, 14]              96
#            Conv2d-61           [-1, 48, 14, 14]           2,352
#       BatchNorm2d-62           [-1, 48, 14, 14]              96
#              ReLU-63           [-1, 48, 14, 14]               0
#            Conv2d-64           [-1, 48, 14, 14]          20,784
#       BatchNorm2d-65           [-1, 48, 14, 14]              96
#            Conv2d-66           [-1, 48, 14, 14]           2,352
#       BatchNorm2d-67           [-1, 48, 14, 14]              96
#              ReLU-68           [-1, 48, 14, 14]               0
#    ChannelShuffle-69           [-1, 96, 14, 14]               0
#     shuffle_block-70           [-1, 96, 14, 14]               0
#      ChannelSplit-71           [-1, 48, 14, 14]               0
#      ChannelSplit-72           [-1, 48, 14, 14]               0
#            Conv2d-73           [-1, 48, 14, 14]           2,352
#       BatchNorm2d-74           [-1, 48, 14, 14]              96
#              ReLU-75           [-1, 48, 14, 14]               0
#            Conv2d-76           [-1, 48, 14, 14]          20,784
#       BatchNorm2d-77           [-1, 48, 14, 14]              96
#            Conv2d-78           [-1, 48, 14, 14]           2,352
#       BatchNorm2d-79           [-1, 48, 14, 14]              96
#              ReLU-80           [-1, 48, 14, 14]               0
#    ChannelShuffle-81           [-1, 96, 14, 14]               0
#     shuffle_block-82           [-1, 96, 14, 14]               0
#      ChannelSplit-83           [-1, 48, 14, 14]               0
#      ChannelSplit-84           [-1, 48, 14, 14]               0
#            Conv2d-85           [-1, 48, 14, 14]           2,352
#       BatchNorm2d-86           [-1, 48, 14, 14]              96
#              ReLU-87           [-1, 48, 14, 14]               0
#            Conv2d-88           [-1, 48, 14, 14]          20,784
#       BatchNorm2d-89           [-1, 48, 14, 14]              96
#            Conv2d-90           [-1, 48, 14, 14]           2,352
#       BatchNorm2d-91           [-1, 48, 14, 14]              96
#              ReLU-92           [-1, 48, 14, 14]               0
#    ChannelShuffle-93           [-1, 96, 14, 14]               0
#     shuffle_block-94           [-1, 96, 14, 14]               0
#      ChannelSplit-95           [-1, 48, 14, 14]               0
#      ChannelSplit-96           [-1, 48, 14, 14]               0
#            Conv2d-97           [-1, 48, 14, 14]           2,352
#       BatchNorm2d-98           [-1, 48, 14, 14]              96
#              ReLU-99           [-1, 48, 14, 14]               0
#           Conv2d-100           [-1, 48, 14, 14]          20,784
#      BatchNorm2d-101           [-1, 48, 14, 14]              96
#           Conv2d-102           [-1, 48, 14, 14]           2,352
#      BatchNorm2d-103           [-1, 48, 14, 14]              96
#             ReLU-104           [-1, 48, 14, 14]               0
#   ChannelShuffle-105           [-1, 96, 14, 14]               0
#    shuffle_block-106           [-1, 96, 14, 14]               0
#     ChannelSplit-107           [-1, 48, 14, 14]               0
#     ChannelSplit-108           [-1, 48, 14, 14]               0
#           Conv2d-109           [-1, 48, 14, 14]           2,352
#      BatchNorm2d-110           [-1, 48, 14, 14]              96
#             ReLU-111           [-1, 48, 14, 14]               0
#           Conv2d-112           [-1, 48, 14, 14]          20,784
#      BatchNorm2d-113           [-1, 48, 14, 14]              96
#           Conv2d-114           [-1, 48, 14, 14]           2,352
#      BatchNorm2d-115           [-1, 48, 14, 14]              96
#             ReLU-116           [-1, 48, 14, 14]               0
#   ChannelShuffle-117           [-1, 96, 14, 14]               0
#    shuffle_block-118           [-1, 96, 14, 14]               0
#     ChannelSplit-119           [-1, 48, 14, 14]               0
#     ChannelSplit-120           [-1, 48, 14, 14]               0
#           Conv2d-121           [-1, 48, 14, 14]           2,352
#      BatchNorm2d-122           [-1, 48, 14, 14]              96
#             ReLU-123           [-1, 48, 14, 14]               0
#           Conv2d-124           [-1, 48, 14, 14]          20,784
#      BatchNorm2d-125           [-1, 48, 14, 14]              96
#           Conv2d-126           [-1, 48, 14, 14]           2,352
#      BatchNorm2d-127           [-1, 48, 14, 14]              96
#             ReLU-128           [-1, 48, 14, 14]               0
#   ChannelShuffle-129           [-1, 96, 14, 14]               0
#    shuffle_block-130           [-1, 96, 14, 14]               0
#     ChannelSplit-131           [-1, 48, 14, 14]               0
#     ChannelSplit-132           [-1, 48, 14, 14]               0
#           Conv2d-133           [-1, 48, 14, 14]           2,352
#      BatchNorm2d-134           [-1, 48, 14, 14]              96
#             ReLU-135           [-1, 48, 14, 14]               0
#           Conv2d-136           [-1, 48, 14, 14]          20,784
#      BatchNorm2d-137           [-1, 48, 14, 14]              96
#           Conv2d-138           [-1, 48, 14, 14]           2,352
#      BatchNorm2d-139           [-1, 48, 14, 14]              96
#             ReLU-140           [-1, 48, 14, 14]               0
#   ChannelShuffle-141           [-1, 96, 14, 14]               0
#    shuffle_block-142           [-1, 96, 14, 14]               0
#     ChannelSplit-143           [-1, 48, 14, 14]               0
#     ChannelSplit-144           [-1, 48, 14, 14]               0
#           Conv2d-145           [-1, 48, 14, 14]           2,352
#      BatchNorm2d-146           [-1, 48, 14, 14]              96
#             ReLU-147           [-1, 48, 14, 14]               0
#           Conv2d-148           [-1, 48, 14, 14]          20,784
#      BatchNorm2d-149           [-1, 48, 14, 14]              96
#           Conv2d-150           [-1, 48, 14, 14]           2,352
#      BatchNorm2d-151           [-1, 48, 14, 14]              96
#             ReLU-152           [-1, 48, 14, 14]               0
#   ChannelShuffle-153           [-1, 96, 14, 14]               0
#    shuffle_block-154           [-1, 96, 14, 14]               0
#           Conv2d-155           [-1, 96, 14, 14]           9,312
#      BatchNorm2d-156           [-1, 96, 14, 14]             192
#             ReLU-157           [-1, 96, 14, 14]               0
#           Conv2d-158             [-1, 96, 7, 7]          83,040
#      BatchNorm2d-159             [-1, 96, 7, 7]             192
#           Conv2d-160             [-1, 96, 7, 7]           9,312
#      BatchNorm2d-161             [-1, 96, 7, 7]             192
#             ReLU-162             [-1, 96, 7, 7]               0
#           Conv2d-163             [-1, 96, 7, 7]          83,040
#      BatchNorm2d-164             [-1, 96, 7, 7]             192
#           Conv2d-165             [-1, 96, 7, 7]           9,312
#      BatchNorm2d-166             [-1, 96, 7, 7]             192
#             ReLU-167             [-1, 96, 7, 7]               0
#   ChannelShuffle-168            [-1, 192, 7, 7]               0
#    shuffle_block-169            [-1, 192, 7, 7]               0
#     ChannelSplit-170             [-1, 96, 7, 7]               0
#     ChannelSplit-171             [-1, 96, 7, 7]               0
#           Conv2d-172             [-1, 96, 7, 7]           9,312
#      BatchNorm2d-173             [-1, 96, 7, 7]             192
#             ReLU-174             [-1, 96, 7, 7]               0
#           Conv2d-175             [-1, 96, 7, 7]          83,040
#      BatchNorm2d-176             [-1, 96, 7, 7]             192
#           Conv2d-177             [-1, 96, 7, 7]           9,312
#      BatchNorm2d-178             [-1, 96, 7, 7]             192
#             ReLU-179             [-1, 96, 7, 7]               0
#   ChannelShuffle-180            [-1, 192, 7, 7]               0
#    shuffle_block-181            [-1, 192, 7, 7]               0
#     ChannelSplit-182             [-1, 96, 7, 7]               0
#     ChannelSplit-183             [-1, 96, 7, 7]               0
#           Conv2d-184             [-1, 96, 7, 7]           9,312
#      BatchNorm2d-185             [-1, 96, 7, 7]             192
#             ReLU-186             [-1, 96, 7, 7]               0
#           Conv2d-187             [-1, 96, 7, 7]          83,040
#      BatchNorm2d-188             [-1, 96, 7, 7]             192
#           Conv2d-189             [-1, 96, 7, 7]           9,312
#      BatchNorm2d-190             [-1, 96, 7, 7]             192
#             ReLU-191             [-1, 96, 7, 7]               0
#   ChannelShuffle-192            [-1, 192, 7, 7]               0
#    shuffle_block-193            [-1, 192, 7, 7]               0
#     ChannelSplit-194             [-1, 96, 7, 7]               0
#     ChannelSplit-195             [-1, 96, 7, 7]               0
#           Conv2d-196             [-1, 96, 7, 7]           9,312
#      BatchNorm2d-197             [-1, 96, 7, 7]             192
#             ReLU-198             [-1, 96, 7, 7]               0
#           Conv2d-199             [-1, 96, 7, 7]          83,040
#      BatchNorm2d-200             [-1, 96, 7, 7]             192
#           Conv2d-201             [-1, 96, 7, 7]           9,312
#      BatchNorm2d-202             [-1, 96, 7, 7]             192
#             ReLU-203             [-1, 96, 7, 7]               0
#   ChannelShuffle-204            [-1, 192, 7, 7]               0
#    shuffle_block-205            [-1, 192, 7, 7]               0
#           Conv2d-206           [-1, 1024, 7, 7]         197,632
# AdaptiveAvgPool2d-207           [-1, 1024, 1, 1]               0
#           Linear-208                  [-1, 100]         102,500
# ================================================================
# Total params: 1,064,196
# Trainable params: 1,064,196
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 28.96
# Params size (MB): 4.06
# Estimated Total Size (MB): 33.59
# ----------------------------------------------------------------