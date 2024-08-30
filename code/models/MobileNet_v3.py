import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchsummary import summary
 
class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)
 
    def forward(self, x):
        return x*self.relu6(x+3)/6
 
def ConvBNActivation(in_channels,out_channels,kernel_size,stride,activate):#深度可分离卷积
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=in_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True) if activate == 'relu' else HardSwish()
        )
 
def Conv1x1BNActivation(in_channels,out_channels,activate):#1*1普通卷积+acitvation
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True) if activate == 'relu' else HardSwish()
        )
 
def Conv1x1BN(in_channels,out_channels):#1*1普通卷积
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
 
class SqueezeAndExcite(nn.Module):
    def __init__(self, in_channels, out_channels,se_kernel_size, divide=4):
        super(SqueezeAndExcite, self).__init__()
        mid_channels = in_channels // divide
        self.pool = nn.AvgPool2d(kernel_size=se_kernel_size,stride=1)
        self.SEblock = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=mid_channels),
            nn.ReLU6(inplace=True),
            nn.Linear(in_features=mid_channels, out_features=out_channels),
            HardSwish(inplace=True),
        )
 
    def forward(self, x):
        b, c, h, w = x.size()
        out = self.pool(x)
        out = out.view(b, -1)
        out = self.SEblock(out)
        out = out.view(b, c, 1, 1)
        return out * x
 
class SEInvertedBottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride,activate, use_se, se_kernel_size=1):
        super(SEInvertedBottleneck, self).__init__()
        self.stride = stride
        self.use_se = use_se
        # mid_channels = (in_channels * expansion_factor)
 
        self.conv = Conv1x1BNActivation(in_channels, mid_channels,activate)
        self.depth_conv = ConvBNActivation(mid_channels, mid_channels, kernel_size,stride,activate)
        if self.use_se:
            self.SEblock = SqueezeAndExcite(mid_channels, mid_channels, se_kernel_size)
 
        self.point_conv = Conv1x1BNActivation(mid_channels, out_channels,activate)
 
        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)
 
    def forward(self, x):
        out = self.depth_conv(self.conv(x))
        if self.use_se:
            out = self.SEblock(out)
        out = self.point_conv(out)
        out = (out + self.shortcut(x)) if self.stride == 1 else out
        return out
 
 
class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000,type='large'):
        super(MobileNetV3, self).__init__()
        self.type = type
 
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            HardSwish(inplace=True),
        )
 
        if type=='large':
            self.large_bottleneck = nn.Sequential(
                SEInvertedBottleneck(in_channels=16, mid_channels=16, out_channels=16, kernel_size=3, stride=1,activate='relu', use_se=False),
                SEInvertedBottleneck(in_channels=16, mid_channels=64, out_channels=24, kernel_size=3, stride=2, activate='relu', use_se=False),
                SEInvertedBottleneck(in_channels=24, mid_channels=72, out_channels=24, kernel_size=3, stride=1, activate='relu', use_se=False),
                SEInvertedBottleneck(in_channels=24, mid_channels=72, out_channels=40, kernel_size=5, stride=2,activate='relu', use_se=True, se_kernel_size=28),
                SEInvertedBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=5, stride=1,activate='relu', use_se=True, se_kernel_size=28),
                SEInvertedBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=5, stride=1,activate='relu', use_se=True, se_kernel_size=28),
                SEInvertedBottleneck(in_channels=40, mid_channels=240, out_channels=80, kernel_size=3, stride=1,activate='hswish', use_se=False),
                SEInvertedBottleneck(in_channels=80, mid_channels=200, out_channels=80, kernel_size=3, stride=1,activate='hswish', use_se=False),
                SEInvertedBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=2,activate='hswish', use_se=False),
                SEInvertedBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=1,activate='hswish', use_se=False),
                SEInvertedBottleneck(in_channels=80, mid_channels=480, out_channels=112, kernel_size=3, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=112, mid_channels=672, out_channels=112, kernel_size=3, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=112, mid_channels=672, out_channels=160, kernel_size=5, stride=2,activate='hswish', use_se=True,se_kernel_size=7),
                SEInvertedBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=1,activate='hswish', use_se=True,se_kernel_size=7),
                SEInvertedBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=1,activate='hswish', use_se=True,se_kernel_size=7),
            )
 
            self.large_last_stage = nn.Sequential(
                nn.Conv2d(in_channels=160, out_channels=960, kernel_size=1, stride=1),
                nn.BatchNorm2d(960),
                HardSwish(inplace=True),
                nn.AvgPool2d(kernel_size=7, stride=1),
                nn.Conv2d(in_channels=960, out_channels=1280, kernel_size=1, stride=1),
                HardSwish(inplace=True),
            )
        else:
            self.small_bottleneck = nn.Sequential(
                SEInvertedBottleneck(in_channels=16, mid_channels=16, out_channels=16, kernel_size=3, stride=2,activate='relu', use_se=True, se_kernel_size=56),
                SEInvertedBottleneck(in_channels=16, mid_channels=72, out_channels=24, kernel_size=3, stride=2,activate='relu', use_se=False),
                SEInvertedBottleneck(in_channels=24, mid_channels=88, out_channels=24, kernel_size=3, stride=1,activate='relu', use_se=False),
                SEInvertedBottleneck(in_channels=24, mid_channels=96, out_channels=40, kernel_size=5, stride=2,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=40, mid_channels=240, out_channels=40, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=40, mid_channels=240, out_channels=40, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=40, mid_channels=120, out_channels=48, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=48, mid_channels=144, out_channels=48, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=48, mid_channels=288, out_channels=96, kernel_size=5, stride=2,activate='hswish', use_se=True, se_kernel_size=7),
                SEInvertedBottleneck(in_channels=96, mid_channels=576, out_channels=96, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=7),
                SEInvertedBottleneck(in_channels=96, mid_channels=576, out_channels=96, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=7),
            )
            self.small_last_stage = nn.Sequential(
                nn.Conv2d(in_channels=96, out_channels=576, kernel_size=1, stride=1),
                nn.BatchNorm2d(576),
                HardSwish(inplace=True),
                nn.AvgPool2d(kernel_size=7, stride=1),
                nn.Conv2d(in_channels=576, out_channels=1280, kernel_size=1, stride=1),
                HardSwish(inplace=True),
            )
 
        self.classifier = nn.Conv2d(in_channels=1280, out_channels=num_classes, kernel_size=1, stride=1)
 
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
    def forward(self, x):
        x = self.first_conv(x)
        if self.type == 'large':
            x = self.large_bottleneck(x)
            x = self.large_last_stage(x)
        else:
            x = self.small_bottleneck(x)
            x = self.small_last_stage(x)
        out = self.classifier(x)
        out = out.view(out.size(0), -1)
        return out
 
if __name__ == '__main__':
    net = MobileNetV3().cuda()
    summary(net, (3, 224, 224))


# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 16, 112, 112]             448
#        BatchNorm2d-2         [-1, 16, 112, 112]              32
#              ReLU6-3         [-1, 16, 112, 112]               0
#          HardSwish-4         [-1, 16, 112, 112]               0
#             Conv2d-5         [-1, 16, 112, 112]             272
#        BatchNorm2d-6         [-1, 16, 112, 112]              32
#              ReLU6-7         [-1, 16, 112, 112]               0
#             Conv2d-8         [-1, 16, 112, 112]             160
#        BatchNorm2d-9         [-1, 16, 112, 112]              32
#             ReLU6-10         [-1, 16, 112, 112]               0
#            Conv2d-11         [-1, 16, 112, 112]             272
#       BatchNorm2d-12         [-1, 16, 112, 112]              32
#             ReLU6-13         [-1, 16, 112, 112]               0
#            Conv2d-14         [-1, 16, 112, 112]             272
#       BatchNorm2d-15         [-1, 16, 112, 112]              32
# SEInvertedBottleneck-16         [-1, 16, 112, 112]               0
#            Conv2d-17         [-1, 64, 112, 112]           1,088
#       BatchNorm2d-18         [-1, 64, 112, 112]             128
#             ReLU6-19         [-1, 64, 112, 112]               0
#            Conv2d-20           [-1, 64, 56, 56]             640
#       BatchNorm2d-21           [-1, 64, 56, 56]             128
#             ReLU6-22           [-1, 64, 56, 56]               0
#            Conv2d-23           [-1, 24, 56, 56]           1,560
#       BatchNorm2d-24           [-1, 24, 56, 56]              48
#             ReLU6-25           [-1, 24, 56, 56]               0
# SEInvertedBottleneck-26           [-1, 24, 56, 56]               0
#            Conv2d-27           [-1, 72, 56, 56]           1,800
#       BatchNorm2d-28           [-1, 72, 56, 56]             144
#             ReLU6-29           [-1, 72, 56, 56]               0
#            Conv2d-30           [-1, 72, 56, 56]             720
#       BatchNorm2d-31           [-1, 72, 56, 56]             144
#             ReLU6-32           [-1, 72, 56, 56]               0
#            Conv2d-33           [-1, 24, 56, 56]           1,752
#       BatchNorm2d-34           [-1, 24, 56, 56]              48
#             ReLU6-35           [-1, 24, 56, 56]               0
#            Conv2d-36           [-1, 24, 56, 56]             600
#       BatchNorm2d-37           [-1, 24, 56, 56]              48
# SEInvertedBottleneck-38           [-1, 24, 56, 56]               0
#            Conv2d-39           [-1, 72, 56, 56]           1,800
#       BatchNorm2d-40           [-1, 72, 56, 56]             144
#             ReLU6-41           [-1, 72, 56, 56]               0
#            Conv2d-42           [-1, 72, 28, 28]           1,872
#       BatchNorm2d-43           [-1, 72, 28, 28]             144
#             ReLU6-44           [-1, 72, 28, 28]               0
#         AvgPool2d-45             [-1, 72, 1, 1]               0
#            Linear-46                   [-1, 18]           1,314
#             ReLU6-47                   [-1, 18]               0
#            Linear-48                   [-1, 72]           1,368
#             ReLU6-49                   [-1, 72]               0
#         HardSwish-50                   [-1, 72]               0
#  SqueezeAndExcite-51           [-1, 72, 28, 28]               0
#            Conv2d-52           [-1, 40, 28, 28]           2,920
#       BatchNorm2d-53           [-1, 40, 28, 28]              80
#             ReLU6-54           [-1, 40, 28, 28]               0
# SEInvertedBottleneck-55           [-1, 40, 28, 28]               0
#            Conv2d-56          [-1, 120, 28, 28]           4,920
#       BatchNorm2d-57          [-1, 120, 28, 28]             240
#             ReLU6-58          [-1, 120, 28, 28]               0
#            Conv2d-59          [-1, 120, 28, 28]           3,120
#       BatchNorm2d-60          [-1, 120, 28, 28]             240
#             ReLU6-61          [-1, 120, 28, 28]               0
#         AvgPool2d-62            [-1, 120, 1, 1]               0
#            Linear-63                   [-1, 30]           3,630
#             ReLU6-64                   [-1, 30]               0
#            Linear-65                  [-1, 120]           3,720
#             ReLU6-66                  [-1, 120]               0
#         HardSwish-67                  [-1, 120]               0
#  SqueezeAndExcite-68          [-1, 120, 28, 28]               0
#            Conv2d-69           [-1, 40, 28, 28]           4,840
#       BatchNorm2d-70           [-1, 40, 28, 28]              80
#             ReLU6-71           [-1, 40, 28, 28]               0
#            Conv2d-72           [-1, 40, 28, 28]           1,640
#       BatchNorm2d-73           [-1, 40, 28, 28]              80
# SEInvertedBottleneck-74           [-1, 40, 28, 28]               0
#            Conv2d-75          [-1, 120, 28, 28]           4,920
#       BatchNorm2d-76          [-1, 120, 28, 28]             240
#             ReLU6-77          [-1, 120, 28, 28]               0
#            Conv2d-78          [-1, 120, 28, 28]           3,120
#       BatchNorm2d-79          [-1, 120, 28, 28]             240
#             ReLU6-80          [-1, 120, 28, 28]               0
#         AvgPool2d-81            [-1, 120, 1, 1]               0
#            Linear-82                   [-1, 30]           3,630
#             ReLU6-83                   [-1, 30]               0
#            Linear-84                  [-1, 120]           3,720
#             ReLU6-85                  [-1, 120]               0
#         HardSwish-86                  [-1, 120]               0
#  SqueezeAndExcite-87          [-1, 120, 28, 28]               0
#            Conv2d-88           [-1, 40, 28, 28]           4,840
#       BatchNorm2d-89           [-1, 40, 28, 28]              80
#             ReLU6-90           [-1, 40, 28, 28]               0
#            Conv2d-91           [-1, 40, 28, 28]           1,640
#       BatchNorm2d-92           [-1, 40, 28, 28]              80
# SEInvertedBottleneck-93           [-1, 40, 28, 28]               0
#            Conv2d-94          [-1, 240, 28, 28]           9,840
#       BatchNorm2d-95          [-1, 240, 28, 28]             480
#             ReLU6-96          [-1, 240, 28, 28]               0
#         HardSwish-97          [-1, 240, 28, 28]               0
#            Conv2d-98          [-1, 240, 28, 28]           2,400
#       BatchNorm2d-99          [-1, 240, 28, 28]             480
#            ReLU6-100          [-1, 240, 28, 28]               0
#        HardSwish-101          [-1, 240, 28, 28]               0
#           Conv2d-102           [-1, 80, 28, 28]          19,280
#      BatchNorm2d-103           [-1, 80, 28, 28]             160
#            ReLU6-104           [-1, 80, 28, 28]               0
#        HardSwish-105           [-1, 80, 28, 28]               0
#           Conv2d-106           [-1, 80, 28, 28]           3,280
#      BatchNorm2d-107           [-1, 80, 28, 28]             160
# SEInvertedBottleneck-108           [-1, 80, 28, 28]               0
#           Conv2d-109          [-1, 200, 28, 28]          16,200
#      BatchNorm2d-110          [-1, 200, 28, 28]             400
#            ReLU6-111          [-1, 200, 28, 28]               0
#        HardSwish-112          [-1, 200, 28, 28]               0
#           Conv2d-113          [-1, 200, 28, 28]           2,000
#      BatchNorm2d-114          [-1, 200, 28, 28]             400
#            ReLU6-115          [-1, 200, 28, 28]               0
#        HardSwish-116          [-1, 200, 28, 28]               0
#           Conv2d-117           [-1, 80, 28, 28]          16,080
#      BatchNorm2d-118           [-1, 80, 28, 28]             160
#            ReLU6-119           [-1, 80, 28, 28]               0
#        HardSwish-120           [-1, 80, 28, 28]               0
#           Conv2d-121           [-1, 80, 28, 28]           6,480
#      BatchNorm2d-122           [-1, 80, 28, 28]             160
# SEInvertedBottleneck-123           [-1, 80, 28, 28]               0
#           Conv2d-124          [-1, 184, 28, 28]          14,904
#      BatchNorm2d-125          [-1, 184, 28, 28]             368
#            ReLU6-126          [-1, 184, 28, 28]               0
#        HardSwish-127          [-1, 184, 28, 28]               0
#           Conv2d-128          [-1, 184, 14, 14]           1,840
#      BatchNorm2d-129          [-1, 184, 14, 14]             368
#            ReLU6-130          [-1, 184, 14, 14]               0
#        HardSwish-131          [-1, 184, 14, 14]               0
#           Conv2d-132           [-1, 80, 14, 14]          14,800
#      BatchNorm2d-133           [-1, 80, 14, 14]             160
#            ReLU6-134           [-1, 80, 14, 14]               0
#        HardSwish-135           [-1, 80, 14, 14]               0
# SEInvertedBottleneck-136           [-1, 80, 14, 14]               0
#           Conv2d-137          [-1, 184, 14, 14]          14,904
#      BatchNorm2d-138          [-1, 184, 14, 14]             368
#            ReLU6-139          [-1, 184, 14, 14]               0
#        HardSwish-140          [-1, 184, 14, 14]               0
#           Conv2d-141          [-1, 184, 14, 14]           1,840
#      BatchNorm2d-142          [-1, 184, 14, 14]             368
#            ReLU6-143          [-1, 184, 14, 14]               0
#        HardSwish-144          [-1, 184, 14, 14]               0
#           Conv2d-145           [-1, 80, 14, 14]          14,800
#      BatchNorm2d-146           [-1, 80, 14, 14]             160
#            ReLU6-147           [-1, 80, 14, 14]               0
#        HardSwish-148           [-1, 80, 14, 14]               0
#           Conv2d-149           [-1, 80, 14, 14]           6,480
#      BatchNorm2d-150           [-1, 80, 14, 14]             160
# SEInvertedBottleneck-151           [-1, 80, 14, 14]               0
#           Conv2d-152          [-1, 480, 14, 14]          38,880
#      BatchNorm2d-153          [-1, 480, 14, 14]             960
#            ReLU6-154          [-1, 480, 14, 14]               0
#        HardSwish-155          [-1, 480, 14, 14]               0
#           Conv2d-156          [-1, 480, 14, 14]           4,800
#      BatchNorm2d-157          [-1, 480, 14, 14]             960
#            ReLU6-158          [-1, 480, 14, 14]               0
#        HardSwish-159          [-1, 480, 14, 14]               0
#        AvgPool2d-160            [-1, 480, 1, 1]               0
#           Linear-161                  [-1, 120]          57,720
#            ReLU6-162                  [-1, 120]               0
#           Linear-163                  [-1, 480]          58,080
#            ReLU6-164                  [-1, 480]               0
#        HardSwish-165                  [-1, 480]               0
# SqueezeAndExcite-166          [-1, 480, 14, 14]               0
#           Conv2d-167          [-1, 112, 14, 14]          53,872
#      BatchNorm2d-168          [-1, 112, 14, 14]             224
#            ReLU6-169          [-1, 112, 14, 14]               0
#        HardSwish-170          [-1, 112, 14, 14]               0
#           Conv2d-171          [-1, 112, 14, 14]           9,072
#      BatchNorm2d-172          [-1, 112, 14, 14]             224
# SEInvertedBottleneck-173          [-1, 112, 14, 14]               0
#           Conv2d-174          [-1, 672, 14, 14]          75,936
#      BatchNorm2d-175          [-1, 672, 14, 14]           1,344
#            ReLU6-176          [-1, 672, 14, 14]               0
#        HardSwish-177          [-1, 672, 14, 14]               0
#           Conv2d-178          [-1, 672, 14, 14]           6,720
#      BatchNorm2d-179          [-1, 672, 14, 14]           1,344
#            ReLU6-180          [-1, 672, 14, 14]               0
#        HardSwish-181          [-1, 672, 14, 14]               0
#        AvgPool2d-182            [-1, 672, 1, 1]               0
#           Linear-183                  [-1, 168]         113,064
#            ReLU6-184                  [-1, 168]               0
#           Linear-185                  [-1, 672]         113,568
#            ReLU6-186                  [-1, 672]               0
#        HardSwish-187                  [-1, 672]               0
# SqueezeAndExcite-188          [-1, 672, 14, 14]               0
#           Conv2d-189          [-1, 112, 14, 14]          75,376
#      BatchNorm2d-190          [-1, 112, 14, 14]             224
#            ReLU6-191          [-1, 112, 14, 14]               0
#        HardSwish-192          [-1, 112, 14, 14]               0
#           Conv2d-193          [-1, 112, 14, 14]          12,656
#      BatchNorm2d-194          [-1, 112, 14, 14]             224
# SEInvertedBottleneck-195          [-1, 112, 14, 14]               0
#           Conv2d-196          [-1, 672, 14, 14]          75,936
#      BatchNorm2d-197          [-1, 672, 14, 14]           1,344
#            ReLU6-198          [-1, 672, 14, 14]               0
#        HardSwish-199          [-1, 672, 14, 14]               0
#           Conv2d-200            [-1, 672, 7, 7]          17,472
#      BatchNorm2d-201            [-1, 672, 7, 7]           1,344
#            ReLU6-202            [-1, 672, 7, 7]               0
#        HardSwish-203            [-1, 672, 7, 7]               0
#        AvgPool2d-204            [-1, 672, 1, 1]               0
#           Linear-205                  [-1, 168]         113,064
#            ReLU6-206                  [-1, 168]               0
#           Linear-207                  [-1, 672]         113,568
#            ReLU6-208                  [-1, 672]               0
#        HardSwish-209                  [-1, 672]               0
# SqueezeAndExcite-210            [-1, 672, 7, 7]               0
#           Conv2d-211            [-1, 160, 7, 7]         107,680
#      BatchNorm2d-212            [-1, 160, 7, 7]             320
#            ReLU6-213            [-1, 160, 7, 7]               0
#        HardSwish-214            [-1, 160, 7, 7]               0
# SEInvertedBottleneck-215            [-1, 160, 7, 7]               0
#           Conv2d-216            [-1, 960, 7, 7]         154,560
#      BatchNorm2d-217            [-1, 960, 7, 7]           1,920
#            ReLU6-218            [-1, 960, 7, 7]               0
#        HardSwish-219            [-1, 960, 7, 7]               0
#           Conv2d-220            [-1, 960, 7, 7]          24,960
#      BatchNorm2d-221            [-1, 960, 7, 7]           1,920
#            ReLU6-222            [-1, 960, 7, 7]               0
#        HardSwish-223            [-1, 960, 7, 7]               0
#        AvgPool2d-224            [-1, 960, 1, 1]               0
#           Linear-225                  [-1, 240]         230,640
#            ReLU6-226                  [-1, 240]               0
#           Linear-227                  [-1, 960]         231,360
#            ReLU6-228                  [-1, 960]               0
#        HardSwish-229                  [-1, 960]               0
# SqueezeAndExcite-230            [-1, 960, 7, 7]               0
#           Conv2d-231            [-1, 160, 7, 7]         153,760
#      BatchNorm2d-232            [-1, 160, 7, 7]             320
#            ReLU6-233            [-1, 160, 7, 7]               0
#        HardSwish-234            [-1, 160, 7, 7]               0
#           Conv2d-235            [-1, 160, 7, 7]          25,760
#      BatchNorm2d-236            [-1, 160, 7, 7]             320
# SEInvertedBottleneck-237            [-1, 160, 7, 7]               0
#           Conv2d-238            [-1, 960, 7, 7]         154,560
#      BatchNorm2d-239            [-1, 960, 7, 7]           1,920
#            ReLU6-240            [-1, 960, 7, 7]               0
#        HardSwish-241            [-1, 960, 7, 7]               0
#           Conv2d-242            [-1, 960, 7, 7]          24,960
#      BatchNorm2d-243            [-1, 960, 7, 7]           1,920
#            ReLU6-244            [-1, 960, 7, 7]               0
#        HardSwish-245            [-1, 960, 7, 7]               0
#        AvgPool2d-246            [-1, 960, 1, 1]               0
#           Linear-247                  [-1, 240]         230,640
#            ReLU6-248                  [-1, 240]               0
#           Linear-249                  [-1, 960]         231,360
#            ReLU6-250                  [-1, 960]               0
#        HardSwish-251                  [-1, 960]               0
# SqueezeAndExcite-252            [-1, 960, 7, 7]               0
#           Conv2d-253            [-1, 160, 7, 7]         153,760
#      BatchNorm2d-254            [-1, 160, 7, 7]             320
#            ReLU6-255            [-1, 160, 7, 7]               0
#        HardSwish-256            [-1, 160, 7, 7]               0
#           Conv2d-257            [-1, 160, 7, 7]          25,760
#      BatchNorm2d-258            [-1, 160, 7, 7]             320
# SEInvertedBottleneck-259            [-1, 160, 7, 7]               0
#           Conv2d-260            [-1, 960, 7, 7]         154,560
#      BatchNorm2d-261            [-1, 960, 7, 7]           1,920
#            ReLU6-262            [-1, 960, 7, 7]               0
#        HardSwish-263            [-1, 960, 7, 7]               0
#        AvgPool2d-264            [-1, 960, 1, 1]               0
#           Conv2d-265           [-1, 1280, 1, 1]       1,230,080
#            ReLU6-266           [-1, 1280, 1, 1]               0
#        HardSwish-267           [-1, 1280, 1, 1]               0
#           Conv2d-268           [-1, 1000, 1, 1]       1,281,000
# ================================================================
# Total params: 5,589,150
# Trainable params: 5,589,150
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 153.55
# Params size (MB): 21.32
# Estimated Total Size (MB): 175.44
# ----------------------------------------------------------------