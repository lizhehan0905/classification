import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class MobileNet_v1(nn.Module):
    def __init__(self,num_classes=10):
        super(MobileNet_v1,self).__init__()

        self.conv=nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.dw1=nn.Sequential(
            nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1,groups=32),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),

            nn.Conv2d(32,64,kernel_size=1,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
        )

        self.dw2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2,padding=1,groups=64),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),

            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
        )

        self.dw3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,groups=128),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),

            nn.Conv2d(128, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
        )

        self.dw4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2,padding=1,groups=128),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),

            nn.Conv2d(128, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),
        )

        self.dw5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,groups=256),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),

            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),
        )

        self.dw6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2,padding=1,groups=256),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),

            nn.Conv2d(256, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True),
        )

        self.dw7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1,groups=512),
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True),

            nn.Conv2d(512, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True),
        )

        self.dw8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,groups=512),
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True),

            nn.Conv2d(512, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True),
        )

        self.dw9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,groups=512),
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True),

            nn.Conv2d(512, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True),
        )

        self.dw10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,groups=512),
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True),

            nn.Conv2d(512, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True),
        )

        self.dw11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,groups=512),
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True),

            nn.Conv2d(512, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True),
        )

        self.dw12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,groups=512),
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True),

            nn.Conv2d(512, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True),
        )

        self.dw13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2,padding=1,groups=512),
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True),

            nn.Conv2d(512, 1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU6(inplace=True),
        )

        self.dw14 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1,groups=1024),
            nn.BatchNorm2d(1024),
            nn.ReLU6(inplace=True),

            nn.Conv2d(1024, 1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU6(inplace=True),
        )

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(1024,num_classes)
        self.dropout = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax(dim=1)

        self.init_params()

    # 初始化参数
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        out = self.conv(x)

        out = self.dw1(out)
        out = self.dw2(out)
        out = self.dw3(out)
        out = self.dw4(out)
        out = self.dw5(out)
        out = self.dw6(out)
        out = self.dw7(out)
        out = self.dw8(out)
        out = self.dw9(out)
        out = self.dw10(out)
        out = self.dw11(out)
        out = self.dw12(out)
        out = self.dw13(out)
        out = self.dw14(out)

        out = self.avgpool(out)
        out = torch.flatten(out,1)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.softmax(out)

        return out

if __name__ == '__main__':
    net = MobileNet_v1().cuda()
    summary(net, (3, 224, 224))



# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 32, 112, 112]             896
#        BatchNorm2d-2         [-1, 32, 112, 112]              64
#              ReLU6-3         [-1, 32, 112, 112]               0
#             Conv2d-4         [-1, 32, 112, 112]             320
#        BatchNorm2d-5         [-1, 32, 112, 112]              64
#              ReLU6-6         [-1, 32, 112, 112]               0
#             Conv2d-7         [-1, 64, 112, 112]           2,112
#        BatchNorm2d-8         [-1, 64, 112, 112]             128
#              ReLU6-9         [-1, 64, 112, 112]               0
#            Conv2d-10           [-1, 64, 56, 56]             640
#       BatchNorm2d-11           [-1, 64, 56, 56]             128
#             ReLU6-12           [-1, 64, 56, 56]               0
#            Conv2d-13          [-1, 128, 56, 56]           8,320
#       BatchNorm2d-14          [-1, 128, 56, 56]             256
#             ReLU6-15          [-1, 128, 56, 56]               0
#            Conv2d-16          [-1, 128, 56, 56]           1,280
#       BatchNorm2d-17          [-1, 128, 56, 56]             256
#             ReLU6-18          [-1, 128, 56, 56]               0
#            Conv2d-19          [-1, 128, 56, 56]          16,512
#       BatchNorm2d-20          [-1, 128, 56, 56]             256
#             ReLU6-21          [-1, 128, 56, 56]               0
#            Conv2d-22          [-1, 128, 28, 28]           1,280
#       BatchNorm2d-23          [-1, 128, 28, 28]             256
#             ReLU6-24          [-1, 128, 28, 28]               0
#            Conv2d-25          [-1, 256, 28, 28]          33,024
#       BatchNorm2d-26          [-1, 256, 28, 28]             512
#             ReLU6-27          [-1, 256, 28, 28]               0
#            Conv2d-28          [-1, 256, 28, 28]           2,560
#       BatchNorm2d-29          [-1, 256, 28, 28]             512
#             ReLU6-30          [-1, 256, 28, 28]               0
#            Conv2d-31          [-1, 256, 28, 28]          65,792
#       BatchNorm2d-32          [-1, 256, 28, 28]             512
#             ReLU6-33          [-1, 256, 28, 28]               0
#            Conv2d-34          [-1, 256, 14, 14]           2,560
#       BatchNorm2d-35          [-1, 256, 14, 14]             512
#             ReLU6-36          [-1, 256, 14, 14]               0
#            Conv2d-37          [-1, 512, 14, 14]         131,584
#       BatchNorm2d-38          [-1, 512, 14, 14]           1,024
#             ReLU6-39          [-1, 512, 14, 14]               0
#            Conv2d-40          [-1, 512, 14, 14]           5,120
#       BatchNorm2d-41          [-1, 512, 14, 14]           1,024
#             ReLU6-42          [-1, 512, 14, 14]               0
#            Conv2d-43          [-1, 512, 14, 14]         262,656
#       BatchNorm2d-44          [-1, 512, 14, 14]           1,024
#             ReLU6-45          [-1, 512, 14, 14]               0
#            Conv2d-46          [-1, 512, 14, 14]           5,120
#       BatchNorm2d-47          [-1, 512, 14, 14]           1,024
#             ReLU6-48          [-1, 512, 14, 14]               0
#            Conv2d-49          [-1, 512, 14, 14]         262,656
#       BatchNorm2d-50          [-1, 512, 14, 14]           1,024
#             ReLU6-51          [-1, 512, 14, 14]               0
#            Conv2d-52          [-1, 512, 14, 14]           5,120
#       BatchNorm2d-53          [-1, 512, 14, 14]           1,024
#             ReLU6-54          [-1, 512, 14, 14]               0
#            Conv2d-55          [-1, 512, 14, 14]         262,656
#       BatchNorm2d-56          [-1, 512, 14, 14]           1,024
#             ReLU6-57          [-1, 512, 14, 14]               0
#            Conv2d-58          [-1, 512, 14, 14]           5,120
#       BatchNorm2d-59          [-1, 512, 14, 14]           1,024
#             ReLU6-60          [-1, 512, 14, 14]               0
#            Conv2d-61          [-1, 512, 14, 14]         262,656
#       BatchNorm2d-62          [-1, 512, 14, 14]           1,024
#             ReLU6-63          [-1, 512, 14, 14]               0
#            Conv2d-64          [-1, 512, 14, 14]           5,120
#       BatchNorm2d-65          [-1, 512, 14, 14]           1,024
#             ReLU6-66          [-1, 512, 14, 14]               0
#            Conv2d-67          [-1, 512, 14, 14]         262,656
#       BatchNorm2d-68          [-1, 512, 14, 14]           1,024
#             ReLU6-69          [-1, 512, 14, 14]               0
#            Conv2d-70          [-1, 512, 14, 14]           5,120
#       BatchNorm2d-71          [-1, 512, 14, 14]           1,024
#             ReLU6-72          [-1, 512, 14, 14]               0
#            Conv2d-73          [-1, 512, 14, 14]         262,656
#       BatchNorm2d-74          [-1, 512, 14, 14]           1,024
#             ReLU6-75          [-1, 512, 14, 14]               0
#            Conv2d-76            [-1, 512, 7, 7]           5,120
#       BatchNorm2d-77            [-1, 512, 7, 7]           1,024
#             ReLU6-78            [-1, 512, 7, 7]               0
#            Conv2d-79           [-1, 1024, 7, 7]         525,312
#       BatchNorm2d-80           [-1, 1024, 7, 7]           2,048
#             ReLU6-81           [-1, 1024, 7, 7]               0
#            Conv2d-82           [-1, 1024, 7, 7]          10,240
#       BatchNorm2d-83           [-1, 1024, 7, 7]           2,048
#             ReLU6-84           [-1, 1024, 7, 7]               0
#            Conv2d-85           [-1, 1024, 7, 7]       1,049,600
#       BatchNorm2d-86           [-1, 1024, 7, 7]           2,048
#             ReLU6-87           [-1, 1024, 7, 7]               0
# AdaptiveAvgPool2d-88           [-1, 1024, 1, 1]               0
#           Dropout-89                 [-1, 1024]               0
#            Linear-90                   [-1, 10]          10,250
#           Softmax-91                   [-1, 10]               0
# ================================================================
# Total params: 3,497,994
# Trainable params: 3,497,994
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 120.03
# Params size (MB): 13.34
# Estimated Total Size (MB): 133.95
# ----------------------------------------------------------------