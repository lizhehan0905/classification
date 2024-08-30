import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
 
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    # int(v + divisor / 2) // divisor * divisor：四舍五入到8
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
 
 
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )
 
 
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
 
        # 所谓的隐藏维度，其实就是输入通道数*倍数
        hidden_dim = int(round(inp * expand_ratio))
        # 只有同时满足两个条件时，才使用短连接
        self.use_res_connect = self.stride == 1 and inp == oup
 
        layers = []
        # 如果扩展因子等于1，就没有第一个1x1的卷积层
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))   # pointwise
 
        layers.extend([
        	# 3x3 depthwise conv，因为使用了groups=hidden_dim
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)
 
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
 
# MobileNetV2是一个类，继承自nn.module这个父类
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):

        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        # 保证通道数是 8 的倍数
        input_channel = _make_divisible(32 * width_mult, round_nearest)
        last_channel = _make_divisible(1280 * width_mult, round_nearest)
 
        if inverted_residual_setting is None:
        	# t表示扩展因子(变胖倍数)，倒残差中间维度，以及第一层少一个1*1的卷积、多一个残差结构；c是通道数；n是block重复几次；
        	#	s：stride步长，只针对第一层，其它s都等于1
            inverted_residual_setting = [
                # t, c, n, s
      
                [1, 16, 1, 1],
         
                [6, 24, 2, 2],
            
                [6, 32, 3, 2],
 
            
                [6, 64, 4, 2],
              
                [6, 96, 3, 1],
 
               
                [6, 160, 3, 2],
                
                [6, 320, 1, 1],
            ]
 
        
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))
 
	# conv1 layer
        # 416,416,3 -> 208,208,32
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # 建立倒残差结构
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
            	# -----------------------------------#
            	# 	s为1或者2 只针对重复了n次的bottleneck 的第一个bottleneck，
            	#	重复n次的剩下几个bottleneck中s均为1。
            	# -----------------------------------#
                stride = s if i == 0 else 1
                # 这个block就是上面那个InvertedResidual函数
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                # 这一层的输出通道数作为下一层的输入通道数
                input_channel = output_channel
 
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        # *features表示位置信息，将特征层利用nn.Sequential打包成一个整体
        self.features = nn.Sequential(*features)
 
        # building classifier
        # 自适应平均池化下采样层，输出矩阵高和宽均为1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )
 
 
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)		
        x = self.classifier(x)
        return x
 
 
def mobilenet_v2(num_classes: int = 1000):
    model = MobileNetV2(num_classes=num_classes)
    return model
 
 
if __name__ == '__main__':
    net = MobileNetV2().cuda()
    summary(net, (3, 224, 224))


# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 32, 112, 112]             864
#        BatchNorm2d-2         [-1, 32, 112, 112]              64
#              ReLU6-3         [-1, 32, 112, 112]               0
#             Conv2d-4         [-1, 32, 112, 112]             288
#        BatchNorm2d-5         [-1, 32, 112, 112]              64
#              ReLU6-6         [-1, 32, 112, 112]               0
#             Conv2d-7         [-1, 16, 112, 112]             512
#        BatchNorm2d-8         [-1, 16, 112, 112]              32
#   InvertedResidual-9         [-1, 16, 112, 112]               0
#            Conv2d-10         [-1, 96, 112, 112]           1,536
#       BatchNorm2d-11         [-1, 96, 112, 112]             192
#             ReLU6-12         [-1, 96, 112, 112]               0
#            Conv2d-13           [-1, 96, 56, 56]             864
#       BatchNorm2d-14           [-1, 96, 56, 56]             192
#             ReLU6-15           [-1, 96, 56, 56]               0
#            Conv2d-16           [-1, 24, 56, 56]           2,304
#       BatchNorm2d-17           [-1, 24, 56, 56]              48
#  InvertedResidual-18           [-1, 24, 56, 56]               0
#            Conv2d-19          [-1, 144, 56, 56]           3,456
#       BatchNorm2d-20          [-1, 144, 56, 56]             288
#             ReLU6-21          [-1, 144, 56, 56]               0
#            Conv2d-22          [-1, 144, 56, 56]           1,296
#       BatchNorm2d-23          [-1, 144, 56, 56]             288
#             ReLU6-24          [-1, 144, 56, 56]               0
#            Conv2d-25           [-1, 24, 56, 56]           3,456
#       BatchNorm2d-26           [-1, 24, 56, 56]              48
#  InvertedResidual-27           [-1, 24, 56, 56]               0
#            Conv2d-28          [-1, 144, 56, 56]           3,456
#       BatchNorm2d-29          [-1, 144, 56, 56]             288
#             ReLU6-30          [-1, 144, 56, 56]               0
#            Conv2d-31          [-1, 144, 28, 28]           1,296
#       BatchNorm2d-32          [-1, 144, 28, 28]             288
#             ReLU6-33          [-1, 144, 28, 28]               0
#            Conv2d-34           [-1, 32, 28, 28]           4,608
#       BatchNorm2d-35           [-1, 32, 28, 28]              64
#  InvertedResidual-36           [-1, 32, 28, 28]               0
#            Conv2d-37          [-1, 192, 28, 28]           6,144
#       BatchNorm2d-38          [-1, 192, 28, 28]             384
#             ReLU6-39          [-1, 192, 28, 28]               0
#            Conv2d-40          [-1, 192, 28, 28]           1,728
#       BatchNorm2d-41          [-1, 192, 28, 28]             384
#             ReLU6-42          [-1, 192, 28, 28]               0
#            Conv2d-43           [-1, 32, 28, 28]           6,144
#       BatchNorm2d-44           [-1, 32, 28, 28]              64
#  InvertedResidual-45           [-1, 32, 28, 28]               0
#            Conv2d-46          [-1, 192, 28, 28]           6,144
#       BatchNorm2d-47          [-1, 192, 28, 28]             384
#             ReLU6-48          [-1, 192, 28, 28]               0
#            Conv2d-49          [-1, 192, 28, 28]           1,728
#       BatchNorm2d-50          [-1, 192, 28, 28]             384
#             ReLU6-51          [-1, 192, 28, 28]               0
#            Conv2d-52           [-1, 32, 28, 28]           6,144
#       BatchNorm2d-53           [-1, 32, 28, 28]              64
#  InvertedResidual-54           [-1, 32, 28, 28]               0
#            Conv2d-55          [-1, 192, 28, 28]           6,144
#       BatchNorm2d-56          [-1, 192, 28, 28]             384
#             ReLU6-57          [-1, 192, 28, 28]               0
#            Conv2d-58          [-1, 192, 14, 14]           1,728
#       BatchNorm2d-59          [-1, 192, 14, 14]             384
#             ReLU6-60          [-1, 192, 14, 14]               0
#            Conv2d-61           [-1, 64, 14, 14]          12,288
#       BatchNorm2d-62           [-1, 64, 14, 14]             128
#  InvertedResidual-63           [-1, 64, 14, 14]               0
#            Conv2d-64          [-1, 384, 14, 14]          24,576
#       BatchNorm2d-65          [-1, 384, 14, 14]             768
#             ReLU6-66          [-1, 384, 14, 14]               0
#            Conv2d-67          [-1, 384, 14, 14]           3,456
#       BatchNorm2d-68          [-1, 384, 14, 14]             768
#             ReLU6-69          [-1, 384, 14, 14]               0
#            Conv2d-70           [-1, 64, 14, 14]          24,576
#       BatchNorm2d-71           [-1, 64, 14, 14]             128
#  InvertedResidual-72           [-1, 64, 14, 14]               0
#            Conv2d-73          [-1, 384, 14, 14]          24,576
#       BatchNorm2d-74          [-1, 384, 14, 14]             768
#             ReLU6-75          [-1, 384, 14, 14]               0
#            Conv2d-76          [-1, 384, 14, 14]           3,456
#       BatchNorm2d-77          [-1, 384, 14, 14]             768
#             ReLU6-78          [-1, 384, 14, 14]               0
#            Conv2d-79           [-1, 64, 14, 14]          24,576
#       BatchNorm2d-80           [-1, 64, 14, 14]             128
#  InvertedResidual-81           [-1, 64, 14, 14]               0
#            Conv2d-82          [-1, 384, 14, 14]          24,576
#       BatchNorm2d-83          [-1, 384, 14, 14]             768
#             ReLU6-84          [-1, 384, 14, 14]               0
#            Conv2d-85          [-1, 384, 14, 14]           3,456
#       BatchNorm2d-86          [-1, 384, 14, 14]             768
#             ReLU6-87          [-1, 384, 14, 14]               0
#            Conv2d-88           [-1, 64, 14, 14]          24,576
#       BatchNorm2d-89           [-1, 64, 14, 14]             128
#  InvertedResidual-90           [-1, 64, 14, 14]               0
#            Conv2d-91          [-1, 384, 14, 14]          24,576
#       BatchNorm2d-92          [-1, 384, 14, 14]             768
#             ReLU6-93          [-1, 384, 14, 14]               0
#            Conv2d-94          [-1, 384, 14, 14]           3,456
#       BatchNorm2d-95          [-1, 384, 14, 14]             768
#             ReLU6-96          [-1, 384, 14, 14]               0
#            Conv2d-97           [-1, 96, 14, 14]          36,864
#       BatchNorm2d-98           [-1, 96, 14, 14]             192
#  InvertedResidual-99           [-1, 96, 14, 14]               0
#           Conv2d-100          [-1, 576, 14, 14]          55,296
#      BatchNorm2d-101          [-1, 576, 14, 14]           1,152
#            ReLU6-102          [-1, 576, 14, 14]               0
#           Conv2d-103          [-1, 576, 14, 14]           5,184
#      BatchNorm2d-104          [-1, 576, 14, 14]           1,152
#            ReLU6-105          [-1, 576, 14, 14]               0
#           Conv2d-106           [-1, 96, 14, 14]          55,296
#      BatchNorm2d-107           [-1, 96, 14, 14]             192
# InvertedResidual-108           [-1, 96, 14, 14]               0
#           Conv2d-109          [-1, 576, 14, 14]          55,296
#      BatchNorm2d-110          [-1, 576, 14, 14]           1,152
#            ReLU6-111          [-1, 576, 14, 14]               0
#           Conv2d-112          [-1, 576, 14, 14]           5,184
#      BatchNorm2d-113          [-1, 576, 14, 14]           1,152
#            ReLU6-114          [-1, 576, 14, 14]               0
#           Conv2d-115           [-1, 96, 14, 14]          55,296
#      BatchNorm2d-116           [-1, 96, 14, 14]             192
# InvertedResidual-117           [-1, 96, 14, 14]               0
#           Conv2d-118          [-1, 576, 14, 14]          55,296
#      BatchNorm2d-119          [-1, 576, 14, 14]           1,152
#            ReLU6-120          [-1, 576, 14, 14]               0
#           Conv2d-121            [-1, 576, 7, 7]           5,184
#      BatchNorm2d-122            [-1, 576, 7, 7]           1,152
#            ReLU6-123            [-1, 576, 7, 7]               0
#           Conv2d-124            [-1, 160, 7, 7]          92,160
#      BatchNorm2d-125            [-1, 160, 7, 7]             320
# InvertedResidual-126            [-1, 160, 7, 7]               0
#           Conv2d-127            [-1, 960, 7, 7]         153,600
#      BatchNorm2d-128            [-1, 960, 7, 7]           1,920
#            ReLU6-129            [-1, 960, 7, 7]               0
#           Conv2d-130            [-1, 960, 7, 7]           8,640
#      BatchNorm2d-131            [-1, 960, 7, 7]           1,920
#            ReLU6-132            [-1, 960, 7, 7]               0
#           Conv2d-133            [-1, 160, 7, 7]         153,600
#      BatchNorm2d-134            [-1, 160, 7, 7]             320
# InvertedResidual-135            [-1, 160, 7, 7]               0
#           Conv2d-136            [-1, 960, 7, 7]         153,600
#      BatchNorm2d-137            [-1, 960, 7, 7]           1,920
#            ReLU6-138            [-1, 960, 7, 7]               0
#           Conv2d-139            [-1, 960, 7, 7]           8,640
#      BatchNorm2d-140            [-1, 960, 7, 7]           1,920
#            ReLU6-141            [-1, 960, 7, 7]               0
#           Conv2d-142            [-1, 160, 7, 7]         153,600
#      BatchNorm2d-143            [-1, 160, 7, 7]             320
# InvertedResidual-144            [-1, 160, 7, 7]               0
#           Conv2d-145            [-1, 960, 7, 7]         153,600
#      BatchNorm2d-146            [-1, 960, 7, 7]           1,920
#            ReLU6-147            [-1, 960, 7, 7]               0
#           Conv2d-148            [-1, 960, 7, 7]           8,640
#      BatchNorm2d-149            [-1, 960, 7, 7]           1,920
#            ReLU6-150            [-1, 960, 7, 7]               0
#           Conv2d-151            [-1, 320, 7, 7]         307,200
#      BatchNorm2d-152            [-1, 320, 7, 7]             640
# InvertedResidual-153            [-1, 320, 7, 7]               0
#           Conv2d-154           [-1, 1280, 7, 7]         409,600
#      BatchNorm2d-155           [-1, 1280, 7, 7]           2,560
#            ReLU6-156           [-1, 1280, 7, 7]               0
# AdaptiveAvgPool2d-157           [-1, 1280, 1, 1]               0
#          Dropout-158                 [-1, 1280]               0
#           Linear-159                 [-1, 1000]       1,281,000
# ================================================================
# Total params: 3,504,872
# Trainable params: 3,504,872
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 152.88
# Params size (MB): 13.37
# Estimated Total Size (MB): 166.82
# ----------------------------------------------------------------