import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class BasicConv(nn.Module):
    def __init__(self,inchannel,outchannel,**kwargs):
        super(BasicConv,self).__init__()

        self.conv=nn.Conv2d(inchannel,outchannel,bias=False,**kwargs)
        # 卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        self.BN=nn.BatchNorm2d(outchannel,eps=0.001)
        self.ReLU=nn.ReLU(inplace=True)

    def forward(self,x):
        out=self.conv(x)
        out=self.BN(out)
        out=self.ReLU(out)
        return out


class InceptionA(nn.Module):
    def __init__(self,inchannel,features):
        super(InceptionA,self).__init__()

        self.branch1=nn.Sequential(
            BasicConv(inchannel,64,kernel_size=1),
        )

        self.branch2=nn.Sequential(
            BasicConv(inchannel,48,kernel_size=1),
            BasicConv(48,64,kernel_size=5,padding=2),
        )

        self.branch3=nn.Sequential(
            BasicConv(inchannel,64,kernel_size=1),
            BasicConv(64, 96, kernel_size=3,padding=1),
            BasicConv(96, 96, kernel_size=3,padding=1),
        )

        self.branch_pool=nn.Sequential(
            nn.AvgPool2d(kernel_size=3,stride=1,padding=1),
            BasicConv(inchannel,features,kernel_size=1),
        )

    def forward(self,x):
        out1=self.branch1(x)
        out2=self.branch2(x)
        out3=self.branch3(x)
        out4=self.branch_pool(x)

        out=[out1,out2,out3,out4]
        return torch.cat(out,1)

class InceptionB(nn.Module):
    def __init__(self,inchannel):
        super(InceptionB,self).__init__()

        self.branch1=nn.Sequential(
            BasicConv(inchannel,384,kernel_size=3,stride=2),
        )

        self.branch2=nn.Sequential(
            BasicConv(inchannel,64,kernel_size=1),
            BasicConv(64, 96, kernel_size=3,padding=1),
            BasicConv(96, 96, kernel_size=3,stride=2),
        )

        self.branch3=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=2),
        )

    def forward(self,x):
        out1=self.branch1(x)
        out2=self.branch2(x)
        out3=self.branch3(x)

        out=[out1,out2,out3]
        return torch.cat(out,1)

class InceptionC(nn.Module):
    def __init__(self,inchannel,channel7x7):
        super(InceptionC,self).__init__()

        self.branch1=nn.Sequential(
            BasicConv(inchannel,192,kernel_size=1),
        )

        self.branch2=nn.Sequential(
            BasicConv(inchannel,channel7x7,kernel_size=1),
            BasicConv(channel7x7,channel7x7,kernel_size=(1,7),padding=(0,3)),
            BasicConv(channel7x7, 192, kernel_size=(7, 1), padding=(3, 0)),
        )

        self.branch3=nn.Sequential(
            BasicConv(inchannel, channel7x7, kernel_size=1),
            BasicConv(channel7x7, channel7x7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv(channel7x7, channel7x7, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv(channel7x7, channel7x7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv(channel7x7, 192, kernel_size=(1, 7), padding=(0, 3)),
        )

        self.branch_pool=nn.Sequential(
            nn.AvgPool2d(kernel_size=3,stride=1,padding=1),
            BasicConv(inchannel,192,kernel_size=1),
        )

    def forward(self,x):
        out1=self.branch1(x)
        out2=self.branch2(x)
        out3=self.branch3(x)
        out4=self.branch_pool(x)

        out=[out1,out2,out3,out4]
        return torch.cat(out,1)

class InceptionD(nn.Module):
    def __init__(self,inchannel):
        super(InceptionD,self).__init__()

        self.branch1=nn.Sequential(
            BasicConv(inchannel,192,kernel_size=1),
            BasicConv(192,320,kernel_size=3,stride=2),
        )

        self.branch2=nn.Sequential(
            BasicConv(inchannel,192,kernel_size=1),
            BasicConv(192,192,kernel_size=(1,7),padding=(0,3)),
            BasicConv(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv(192,192,kernel_size=3,stride=2),
        )

        self.branch_pool=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=2),
        )

    def forward(self,x):
        out1=self.branch1(x)
        out2=self.branch2(x)
        out3=self.branch_pool(x)

        out=[out1,out2,out3]
        return torch.cat(out,1)

class InceptionE(nn.Module):
    def __init__(self,inchannel):
        super(InceptionE,self).__init__()

        self.branch1x1=nn.Conv2d(inchannel,320,kernel_size=1)

        self.branch3x3_1=nn.Conv2d(inchannel,384,kernel_size=1)
        self.branch3x3_2a=nn.Conv2d(384,384,kernel_size=(1,3),padding=(0,1))
        self.branch3x3_2b=nn.Conv2d(384,384,kernel_size=(3,1),padding=(1,0))


        self.branch3x3db_1=nn.Conv2d(inchannel,448,kernel_size=1)
        self.branch3x3db_2=nn.Conv2d(448,384,kernel_size=3,padding=1)
        self.branch3x3db_3a=nn.Conv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3db_3b=nn.Conv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))


        self.branch_pool=nn.Sequential(
            nn.AvgPool2d(kernel_size=3,stride=1,padding=1),
            BasicConv(inchannel,192,kernel_size=1),
        )

    def forward(self,x):
        out1=self.branch1x1(x)

        out2=self.branch3x3_1(x)
        out2=[
            self.branch3x3_2a(out2),
            self.branch3x3_2b(out2),
        ]
        out2=torch.cat(out2,1)

        out3=self.branch3x3db_1(x)
        out3=self.branch3x3db_2(out3)
        out3=[
            self.branch3x3db_3a(out3),
            self.branch3x3db_3b(out3),
        ]
        out3=torch.cat(out3,1)

        out4=self.branch_pool(x)

        out=[out1,out2,out3,out4]
        return torch.cat(out,1)

class InceptionAux(nn.Module):
    def __init__(self,inchannel,num_classes):
        super(InceptionAux,self).__init__()

        self.pool=nn.AvgPool2d(kernel_size=5,stride=3)
        self.conv0=BasicConv(inchannel,128,kernel_size=1)
        self.conv1=BasicConv(128,768,kernel_size=5)
        # 指定卷积层 conv1 的权重参数的初始标准差为 0.01。这意味着在初始化卷积层的权重时，将以标准差为 0.01 的高斯分布来随机初始化权重参数的值，使得初始权重值更加接近于 0，有助于更稳定和更快速的训练模型。
        self.conv1.stddev=0.01
        self.fc=nn.Linear(768,num_classes)
        self.fc.stddev=0.001

    def forward(self,x):
        # N x 768 x 17 x 17
        out=self.pool(x)
        # N x 768 x 5 x 5
        out=self.conv0(out)
        # N x 128 x 5 x 5
        out=self.conv1(out)
        # N x 768 x 1 x 1
        # F.avg_pool2d,传统平均池化，需要知道池化核和步长
        # F.adaptive_avg_pool2d,可以指定输出的特征图大小，池化核不固定
        out=F.adaptive_avg_pool2d(out,(1,1))
        # N x 768 x 1 x 1
        out=torch.flatten(out,1)
        # N x 768
        out=self.fc(out)
        # N x num_classes
        return out

class GoogLeNet_V3(nn.Module):
    def __init__(self,num_classes=10,aux_logits=True,transform_input=False,inception_blocks=None):
        super(GoogLeNet_V3,self).__init__()

        if inception_blocks is None:
            inception_blocks=[BasicConv,InceptionA,InceptionB,InceptionC,InceptionD,InceptionE,InceptionAux]

        assert len(inception_blocks)==7

        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.aux_logits=aux_logits
        self.transform_input=transform_input

        self.conv1=conv_block(3,32,kernel_size=3,stride=2)
        self.conv2=conv_block(32,32,kernel_size=3)
        self.conv3=conv_block(32,64,kernel_size=3,padding=1)

        self.pool1=nn.MaxPool2d(kernel_size=3,stride=2)

        self.conv4=conv_block(64,80,kernel_size=1)
        self.conv5=conv_block(80,192,kernel_size=3)

        self.pool2=nn.MaxPool2d(kernel_size=3,stride=2)

        self.inception_a1=inception_a(192,features=32)
        self.inception_a2=inception_a(256,features=64)
        self.inception_a3=inception_a(288, features=64)

        self.inception_b=inception_b(288)

        self.inception_c1 = inception_c(768, channel7x7=128)
        self.inception_c2 = inception_c(768, channel7x7=160)
        self.inception_c3 = inception_c(768, channel7x7=160)
        self.inception_c4 = inception_c(768, channel7x7=192)

        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)

        self.inception_d=inception_d(768)

        self.inception_e1=inception_e(1280)
        self.inception_e2=inception_e(2048)
        self.fc=nn.Linear(2048,num_classes)

    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)
        out=self.conv3(out)

        out=self.pool1(out)

        out=self.conv4(out)
        out=self.conv5(out)

        out=self.pool2(out)

        out=self.inception_a1(out)
        out=self.inception_a2(out)
        out=self.inception_a3(out)

        out=self.inception_b(out)

        out=self.inception_c1(out)
        out=self.inception_c2(out)
        out=self.inception_c3(out)
        out=self.inception_c4(out)

        if self.training and self.aux_logits:
            aux=self.AuxLogits(out)

        out=self.inception_d(out)

        out=self.inception_e1(out)
        out=self.inception_e2(out)

        out=F.adaptive_avg_pool2d(out,(1,1))
        out=F.dropout(out,training=self.training)
        out=torch.flatten(out,1)
        out=self.fc(out)
        return out,aux

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
    net = GoogLeNet_V3().cuda()
    summary(net, (3, 299, 299))

# 输出：
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 32, 149, 149]             864
#        BatchNorm2d-2         [-1, 32, 149, 149]              64
#               ReLU-3         [-1, 32, 149, 149]               0
#          BasicConv-4         [-1, 32, 149, 149]               0
#             Conv2d-5         [-1, 32, 147, 147]           9,216
#        BatchNorm2d-6         [-1, 32, 147, 147]              64
#               ReLU-7         [-1, 32, 147, 147]               0
#          BasicConv-8         [-1, 32, 147, 147]               0
#             Conv2d-9         [-1, 64, 147, 147]          18,432
#       BatchNorm2d-10         [-1, 64, 147, 147]             128
#              ReLU-11         [-1, 64, 147, 147]               0
#         BasicConv-12         [-1, 64, 147, 147]               0
#         MaxPool2d-13           [-1, 64, 73, 73]               0
#            Conv2d-14           [-1, 80, 73, 73]           5,120
#       BatchNorm2d-15           [-1, 80, 73, 73]             160
#              ReLU-16           [-1, 80, 73, 73]               0
#         BasicConv-17           [-1, 80, 73, 73]               0
#            Conv2d-18          [-1, 192, 71, 71]         138,240
#       BatchNorm2d-19          [-1, 192, 71, 71]             384
#              ReLU-20          [-1, 192, 71, 71]               0
#         BasicConv-21          [-1, 192, 71, 71]               0
#         MaxPool2d-22          [-1, 192, 35, 35]               0
#            Conv2d-23           [-1, 64, 35, 35]          12,288
#       BatchNorm2d-24           [-1, 64, 35, 35]             128
#              ReLU-25           [-1, 64, 35, 35]               0
#         BasicConv-26           [-1, 64, 35, 35]               0
#            Conv2d-27           [-1, 48, 35, 35]           9,216
#       BatchNorm2d-28           [-1, 48, 35, 35]              96
#              ReLU-29           [-1, 48, 35, 35]               0
#         BasicConv-30           [-1, 48, 35, 35]               0
#            Conv2d-31           [-1, 64, 35, 35]          76,800
#       BatchNorm2d-32           [-1, 64, 35, 35]             128
#              ReLU-33           [-1, 64, 35, 35]               0
#         BasicConv-34           [-1, 64, 35, 35]               0
#            Conv2d-35           [-1, 64, 35, 35]          12,288
#       BatchNorm2d-36           [-1, 64, 35, 35]             128
#              ReLU-37           [-1, 64, 35, 35]               0
#         BasicConv-38           [-1, 64, 35, 35]               0
#            Conv2d-39           [-1, 96, 35, 35]          55,296
#       BatchNorm2d-40           [-1, 96, 35, 35]             192
#              ReLU-41           [-1, 96, 35, 35]               0
#         BasicConv-42           [-1, 96, 35, 35]               0
#            Conv2d-43           [-1, 96, 35, 35]          82,944
#       BatchNorm2d-44           [-1, 96, 35, 35]             192
#              ReLU-45           [-1, 96, 35, 35]               0
#         BasicConv-46           [-1, 96, 35, 35]               0
#         AvgPool2d-47          [-1, 192, 35, 35]               0
#            Conv2d-48           [-1, 32, 35, 35]           6,144
#       BatchNorm2d-49           [-1, 32, 35, 35]              64
#              ReLU-50           [-1, 32, 35, 35]               0
#         BasicConv-51           [-1, 32, 35, 35]               0
#        InceptionA-52          [-1, 256, 35, 35]               0
#            Conv2d-53           [-1, 64, 35, 35]          16,384
#       BatchNorm2d-54           [-1, 64, 35, 35]             128
#              ReLU-55           [-1, 64, 35, 35]               0
#         BasicConv-56           [-1, 64, 35, 35]               0
#            Conv2d-57           [-1, 48, 35, 35]          12,288
#       BatchNorm2d-58           [-1, 48, 35, 35]              96
#              ReLU-59           [-1, 48, 35, 35]               0
#         BasicConv-60           [-1, 48, 35, 35]               0
#            Conv2d-61           [-1, 64, 35, 35]          76,800
#       BatchNorm2d-62           [-1, 64, 35, 35]             128
#              ReLU-63           [-1, 64, 35, 35]               0
#         BasicConv-64           [-1, 64, 35, 35]               0
#            Conv2d-65           [-1, 64, 35, 35]          16,384
#       BatchNorm2d-66           [-1, 64, 35, 35]             128
#              ReLU-67           [-1, 64, 35, 35]               0
#         BasicConv-68           [-1, 64, 35, 35]               0
#            Conv2d-69           [-1, 96, 35, 35]          55,296
#       BatchNorm2d-70           [-1, 96, 35, 35]             192
#              ReLU-71           [-1, 96, 35, 35]               0
#         BasicConv-72           [-1, 96, 35, 35]               0
#            Conv2d-73           [-1, 96, 35, 35]          82,944
#       BatchNorm2d-74           [-1, 96, 35, 35]             192
#              ReLU-75           [-1, 96, 35, 35]               0
#         BasicConv-76           [-1, 96, 35, 35]               0
#         AvgPool2d-77          [-1, 256, 35, 35]               0
#            Conv2d-78           [-1, 64, 35, 35]          16,384
#       BatchNorm2d-79           [-1, 64, 35, 35]             128
#              ReLU-80           [-1, 64, 35, 35]               0
#         BasicConv-81           [-1, 64, 35, 35]               0
#        InceptionA-82          [-1, 288, 35, 35]               0
#            Conv2d-83           [-1, 64, 35, 35]          18,432
#       BatchNorm2d-84           [-1, 64, 35, 35]             128
#              ReLU-85           [-1, 64, 35, 35]               0
#         BasicConv-86           [-1, 64, 35, 35]               0
#            Conv2d-87           [-1, 48, 35, 35]          13,824
#       BatchNorm2d-88           [-1, 48, 35, 35]              96
#              ReLU-89           [-1, 48, 35, 35]               0
#         BasicConv-90           [-1, 48, 35, 35]               0
#            Conv2d-91           [-1, 64, 35, 35]          76,800
#       BatchNorm2d-92           [-1, 64, 35, 35]             128
#              ReLU-93           [-1, 64, 35, 35]               0
#         BasicConv-94           [-1, 64, 35, 35]               0
#            Conv2d-95           [-1, 64, 35, 35]          18,432
#       BatchNorm2d-96           [-1, 64, 35, 35]             128
#              ReLU-97           [-1, 64, 35, 35]               0
#         BasicConv-98           [-1, 64, 35, 35]               0
#            Conv2d-99           [-1, 96, 35, 35]          55,296
#      BatchNorm2d-100           [-1, 96, 35, 35]             192
#             ReLU-101           [-1, 96, 35, 35]               0
#        BasicConv-102           [-1, 96, 35, 35]               0
#           Conv2d-103           [-1, 96, 35, 35]          82,944
#      BatchNorm2d-104           [-1, 96, 35, 35]             192
#             ReLU-105           [-1, 96, 35, 35]               0
#        BasicConv-106           [-1, 96, 35, 35]               0
#        AvgPool2d-107          [-1, 288, 35, 35]               0
#           Conv2d-108           [-1, 64, 35, 35]          18,432
#      BatchNorm2d-109           [-1, 64, 35, 35]             128
#             ReLU-110           [-1, 64, 35, 35]               0
#        BasicConv-111           [-1, 64, 35, 35]               0
#       InceptionA-112          [-1, 288, 35, 35]               0
#           Conv2d-113          [-1, 384, 17, 17]         995,328
#      BatchNorm2d-114          [-1, 384, 17, 17]             768
#             ReLU-115          [-1, 384, 17, 17]               0
#        BasicConv-116          [-1, 384, 17, 17]               0
#           Conv2d-117           [-1, 64, 35, 35]          18,432
#      BatchNorm2d-118           [-1, 64, 35, 35]             128
#             ReLU-119           [-1, 64, 35, 35]               0
#        BasicConv-120           [-1, 64, 35, 35]               0
#           Conv2d-121           [-1, 96, 35, 35]          55,296
#      BatchNorm2d-122           [-1, 96, 35, 35]             192
#             ReLU-123           [-1, 96, 35, 35]               0
#        BasicConv-124           [-1, 96, 35, 35]               0
#           Conv2d-125           [-1, 96, 17, 17]          82,944
#      BatchNorm2d-126           [-1, 96, 17, 17]             192
#             ReLU-127           [-1, 96, 17, 17]               0
#        BasicConv-128           [-1, 96, 17, 17]               0
#        MaxPool2d-129          [-1, 288, 17, 17]               0
#       InceptionB-130          [-1, 768, 17, 17]               0
#           Conv2d-131          [-1, 192, 17, 17]         147,456
#      BatchNorm2d-132          [-1, 192, 17, 17]             384
#             ReLU-133          [-1, 192, 17, 17]               0
#        BasicConv-134          [-1, 192, 17, 17]               0
#           Conv2d-135          [-1, 128, 17, 17]          98,304
#      BatchNorm2d-136          [-1, 128, 17, 17]             256
#             ReLU-137          [-1, 128, 17, 17]               0
#        BasicConv-138          [-1, 128, 17, 17]               0
#           Conv2d-139          [-1, 128, 17, 17]         114,688
#      BatchNorm2d-140          [-1, 128, 17, 17]             256
#             ReLU-141          [-1, 128, 17, 17]               0
#        BasicConv-142          [-1, 128, 17, 17]               0
#           Conv2d-143          [-1, 192, 17, 17]         172,032
#      BatchNorm2d-144          [-1, 192, 17, 17]             384
#             ReLU-145          [-1, 192, 17, 17]               0
#        BasicConv-146          [-1, 192, 17, 17]               0
#           Conv2d-147          [-1, 128, 17, 17]          98,304
#      BatchNorm2d-148          [-1, 128, 17, 17]             256
#             ReLU-149          [-1, 128, 17, 17]               0
#        BasicConv-150          [-1, 128, 17, 17]               0
#           Conv2d-151          [-1, 128, 17, 17]         114,688
#      BatchNorm2d-152          [-1, 128, 17, 17]             256
#             ReLU-153          [-1, 128, 17, 17]               0
#        BasicConv-154          [-1, 128, 17, 17]               0
#           Conv2d-155          [-1, 128, 17, 17]         114,688
#      BatchNorm2d-156          [-1, 128, 17, 17]             256
#             ReLU-157          [-1, 128, 17, 17]               0
#        BasicConv-158          [-1, 128, 17, 17]               0
#           Conv2d-159          [-1, 128, 17, 17]         114,688
#      BatchNorm2d-160          [-1, 128, 17, 17]             256
#             ReLU-161          [-1, 128, 17, 17]               0
#        BasicConv-162          [-1, 128, 17, 17]               0
#           Conv2d-163          [-1, 192, 17, 17]         172,032
#      BatchNorm2d-164          [-1, 192, 17, 17]             384
#             ReLU-165          [-1, 192, 17, 17]               0
#        BasicConv-166          [-1, 192, 17, 17]               0
#        AvgPool2d-167          [-1, 768, 17, 17]               0
#           Conv2d-168          [-1, 192, 17, 17]         147,456
#      BatchNorm2d-169          [-1, 192, 17, 17]             384
#             ReLU-170          [-1, 192, 17, 17]               0
#        BasicConv-171          [-1, 192, 17, 17]               0
#       InceptionC-172          [-1, 768, 17, 17]               0
#           Conv2d-173          [-1, 192, 17, 17]         147,456
#      BatchNorm2d-174          [-1, 192, 17, 17]             384
#             ReLU-175          [-1, 192, 17, 17]               0
#        BasicConv-176          [-1, 192, 17, 17]               0
#           Conv2d-177          [-1, 160, 17, 17]         122,880
#      BatchNorm2d-178          [-1, 160, 17, 17]             320
#             ReLU-179          [-1, 160, 17, 17]               0
#        BasicConv-180          [-1, 160, 17, 17]               0
#           Conv2d-181          [-1, 160, 17, 17]         179,200
#      BatchNorm2d-182          [-1, 160, 17, 17]             320
#             ReLU-183          [-1, 160, 17, 17]               0
#        BasicConv-184          [-1, 160, 17, 17]               0
#           Conv2d-185          [-1, 192, 17, 17]         215,040
#      BatchNorm2d-186          [-1, 192, 17, 17]             384
#             ReLU-187          [-1, 192, 17, 17]               0
#        BasicConv-188          [-1, 192, 17, 17]               0
#           Conv2d-189          [-1, 160, 17, 17]         122,880
#      BatchNorm2d-190          [-1, 160, 17, 17]             320
#             ReLU-191          [-1, 160, 17, 17]               0
#        BasicConv-192          [-1, 160, 17, 17]               0
#           Conv2d-193          [-1, 160, 17, 17]         179,200
#      BatchNorm2d-194          [-1, 160, 17, 17]             320
#             ReLU-195          [-1, 160, 17, 17]               0
#        BasicConv-196          [-1, 160, 17, 17]               0
#           Conv2d-197          [-1, 160, 17, 17]         179,200
#      BatchNorm2d-198          [-1, 160, 17, 17]             320
#             ReLU-199          [-1, 160, 17, 17]               0
#        BasicConv-200          [-1, 160, 17, 17]               0
#           Conv2d-201          [-1, 160, 17, 17]         179,200
#      BatchNorm2d-202          [-1, 160, 17, 17]             320
#             ReLU-203          [-1, 160, 17, 17]               0
#        BasicConv-204          [-1, 160, 17, 17]               0
#           Conv2d-205          [-1, 192, 17, 17]         215,040
#      BatchNorm2d-206          [-1, 192, 17, 17]             384
#             ReLU-207          [-1, 192, 17, 17]               0
#        BasicConv-208          [-1, 192, 17, 17]               0
#        AvgPool2d-209          [-1, 768, 17, 17]               0
#           Conv2d-210          [-1, 192, 17, 17]         147,456
#      BatchNorm2d-211          [-1, 192, 17, 17]             384
#             ReLU-212          [-1, 192, 17, 17]               0
#        BasicConv-213          [-1, 192, 17, 17]               0
#       InceptionC-214          [-1, 768, 17, 17]               0
#           Conv2d-215          [-1, 192, 17, 17]         147,456
#      BatchNorm2d-216          [-1, 192, 17, 17]             384
#             ReLU-217          [-1, 192, 17, 17]               0
#        BasicConv-218          [-1, 192, 17, 17]               0
#           Conv2d-219          [-1, 160, 17, 17]         122,880
#      BatchNorm2d-220          [-1, 160, 17, 17]             320
#             ReLU-221          [-1, 160, 17, 17]               0
#        BasicConv-222          [-1, 160, 17, 17]               0
#           Conv2d-223          [-1, 160, 17, 17]         179,200
#      BatchNorm2d-224          [-1, 160, 17, 17]             320
#             ReLU-225          [-1, 160, 17, 17]               0
#        BasicConv-226          [-1, 160, 17, 17]               0
#           Conv2d-227          [-1, 192, 17, 17]         215,040
#      BatchNorm2d-228          [-1, 192, 17, 17]             384
#             ReLU-229          [-1, 192, 17, 17]               0
#        BasicConv-230          [-1, 192, 17, 17]               0
#           Conv2d-231          [-1, 160, 17, 17]         122,880
#      BatchNorm2d-232          [-1, 160, 17, 17]             320
#             ReLU-233          [-1, 160, 17, 17]               0
#        BasicConv-234          [-1, 160, 17, 17]               0
#           Conv2d-235          [-1, 160, 17, 17]         179,200
#      BatchNorm2d-236          [-1, 160, 17, 17]             320
#             ReLU-237          [-1, 160, 17, 17]               0
#        BasicConv-238          [-1, 160, 17, 17]               0
#           Conv2d-239          [-1, 160, 17, 17]         179,200
#      BatchNorm2d-240          [-1, 160, 17, 17]             320
#             ReLU-241          [-1, 160, 17, 17]               0
#        BasicConv-242          [-1, 160, 17, 17]               0
#           Conv2d-243          [-1, 160, 17, 17]         179,200
#      BatchNorm2d-244          [-1, 160, 17, 17]             320
#             ReLU-245          [-1, 160, 17, 17]               0
#        BasicConv-246          [-1, 160, 17, 17]               0
#           Conv2d-247          [-1, 192, 17, 17]         215,040
#      BatchNorm2d-248          [-1, 192, 17, 17]             384
#             ReLU-249          [-1, 192, 17, 17]               0
#        BasicConv-250          [-1, 192, 17, 17]               0
#        AvgPool2d-251          [-1, 768, 17, 17]               0
#           Conv2d-252          [-1, 192, 17, 17]         147,456
#      BatchNorm2d-253          [-1, 192, 17, 17]             384
#             ReLU-254          [-1, 192, 17, 17]               0
#        BasicConv-255          [-1, 192, 17, 17]               0
#       InceptionC-256          [-1, 768, 17, 17]               0
#           Conv2d-257          [-1, 192, 17, 17]         147,456
#      BatchNorm2d-258          [-1, 192, 17, 17]             384
#             ReLU-259          [-1, 192, 17, 17]               0
#        BasicConv-260          [-1, 192, 17, 17]               0
#           Conv2d-261          [-1, 192, 17, 17]         147,456
#      BatchNorm2d-262          [-1, 192, 17, 17]             384
#             ReLU-263          [-1, 192, 17, 17]               0
#        BasicConv-264          [-1, 192, 17, 17]               0
#           Conv2d-265          [-1, 192, 17, 17]         258,048
#      BatchNorm2d-266          [-1, 192, 17, 17]             384
#             ReLU-267          [-1, 192, 17, 17]               0
#        BasicConv-268          [-1, 192, 17, 17]               0
#           Conv2d-269          [-1, 192, 17, 17]         258,048
#      BatchNorm2d-270          [-1, 192, 17, 17]             384
#             ReLU-271          [-1, 192, 17, 17]               0
#        BasicConv-272          [-1, 192, 17, 17]               0
#           Conv2d-273          [-1, 192, 17, 17]         147,456
#      BatchNorm2d-274          [-1, 192, 17, 17]             384
#             ReLU-275          [-1, 192, 17, 17]               0
#        BasicConv-276          [-1, 192, 17, 17]               0
#           Conv2d-277          [-1, 192, 17, 17]         258,048
#      BatchNorm2d-278          [-1, 192, 17, 17]             384
#             ReLU-279          [-1, 192, 17, 17]               0
#        BasicConv-280          [-1, 192, 17, 17]               0
#           Conv2d-281          [-1, 192, 17, 17]         258,048
#      BatchNorm2d-282          [-1, 192, 17, 17]             384
#             ReLU-283          [-1, 192, 17, 17]               0
#        BasicConv-284          [-1, 192, 17, 17]               0
#           Conv2d-285          [-1, 192, 17, 17]         258,048
#      BatchNorm2d-286          [-1, 192, 17, 17]             384
#             ReLU-287          [-1, 192, 17, 17]               0
#        BasicConv-288          [-1, 192, 17, 17]               0
#           Conv2d-289          [-1, 192, 17, 17]         258,048
#      BatchNorm2d-290          [-1, 192, 17, 17]             384
#             ReLU-291          [-1, 192, 17, 17]               0
#        BasicConv-292          [-1, 192, 17, 17]               0
#        AvgPool2d-293          [-1, 768, 17, 17]               0
#           Conv2d-294          [-1, 192, 17, 17]         147,456
#      BatchNorm2d-295          [-1, 192, 17, 17]             384
#             ReLU-296          [-1, 192, 17, 17]               0
#        BasicConv-297          [-1, 192, 17, 17]               0
#       InceptionC-298          [-1, 768, 17, 17]               0
#        AvgPool2d-299            [-1, 768, 5, 5]               0
#           Conv2d-300            [-1, 128, 5, 5]          98,304
#      BatchNorm2d-301            [-1, 128, 5, 5]             256
#             ReLU-302            [-1, 128, 5, 5]               0
#        BasicConv-303            [-1, 128, 5, 5]               0
#           Conv2d-304            [-1, 768, 1, 1]       2,457,600
#      BatchNorm2d-305            [-1, 768, 1, 1]           1,536
#             ReLU-306            [-1, 768, 1, 1]               0
#        BasicConv-307            [-1, 768, 1, 1]               0
#           Linear-308                   [-1, 10]           7,690
#     InceptionAux-309                   [-1, 10]               0
#           Conv2d-310          [-1, 192, 17, 17]         147,456
#      BatchNorm2d-311          [-1, 192, 17, 17]             384
#             ReLU-312          [-1, 192, 17, 17]               0
#        BasicConv-313          [-1, 192, 17, 17]               0
#           Conv2d-314            [-1, 320, 8, 8]         552,960
#      BatchNorm2d-315            [-1, 320, 8, 8]             640
#             ReLU-316            [-1, 320, 8, 8]               0
#        BasicConv-317            [-1, 320, 8, 8]               0
#           Conv2d-318          [-1, 192, 17, 17]         147,456
#      BatchNorm2d-319          [-1, 192, 17, 17]             384
#             ReLU-320          [-1, 192, 17, 17]               0
#        BasicConv-321          [-1, 192, 17, 17]               0
#           Conv2d-322          [-1, 192, 17, 17]         258,048
#      BatchNorm2d-323          [-1, 192, 17, 17]             384
#             ReLU-324          [-1, 192, 17, 17]               0
#        BasicConv-325          [-1, 192, 17, 17]               0
#           Conv2d-326          [-1, 192, 17, 17]         258,048
#      BatchNorm2d-327          [-1, 192, 17, 17]             384
#             ReLU-328          [-1, 192, 17, 17]               0
#        BasicConv-329          [-1, 192, 17, 17]               0
#           Conv2d-330            [-1, 192, 8, 8]         331,776
#      BatchNorm2d-331            [-1, 192, 8, 8]             384
#             ReLU-332            [-1, 192, 8, 8]               0
#        BasicConv-333            [-1, 192, 8, 8]               0
#        MaxPool2d-334            [-1, 768, 8, 8]               0
#       InceptionD-335           [-1, 1280, 8, 8]               0
#           Conv2d-336            [-1, 320, 8, 8]         409,920
#           Conv2d-337            [-1, 384, 8, 8]         491,904
#           Conv2d-338            [-1, 384, 8, 8]         442,752
#           Conv2d-339            [-1, 384, 8, 8]         442,752
#           Conv2d-340            [-1, 448, 8, 8]         573,888
#           Conv2d-341            [-1, 384, 8, 8]       1,548,672
#           Conv2d-342            [-1, 384, 8, 8]         442,752
#           Conv2d-343            [-1, 384, 8, 8]         442,752
#        AvgPool2d-344           [-1, 1280, 8, 8]               0
#           Conv2d-345            [-1, 192, 8, 8]         245,760
#      BatchNorm2d-346            [-1, 192, 8, 8]             384
#             ReLU-347            [-1, 192, 8, 8]               0
#        BasicConv-348            [-1, 192, 8, 8]               0
#       InceptionE-349           [-1, 2048, 8, 8]               0
#           Conv2d-350            [-1, 320, 8, 8]         655,680
#           Conv2d-351            [-1, 384, 8, 8]         786,816
#           Conv2d-352            [-1, 384, 8, 8]         442,752
#           Conv2d-353            [-1, 384, 8, 8]         442,752
#           Conv2d-354            [-1, 448, 8, 8]         917,952
#           Conv2d-355            [-1, 384, 8, 8]       1,548,672
#           Conv2d-356            [-1, 384, 8, 8]         442,752
#           Conv2d-357            [-1, 384, 8, 8]         442,752
#        AvgPool2d-358           [-1, 2048, 8, 8]               0
#           Conv2d-359            [-1, 192, 8, 8]         393,216
#      BatchNorm2d-360            [-1, 192, 8, 8]             384
#             ReLU-361            [-1, 192, 8, 8]               0
#        BasicConv-362            [-1, 192, 8, 8]               0
#       InceptionE-363           [-1, 2048, 8, 8]               0
#           Linear-364                   [-1, 10]          20,490
# ================================================================
# Total params: 24,365,300
# Trainable params: 24,365,300
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 1.02
# Forward/backward pass size (MB): 304.49
# Params size (MB): 92.95
# Estimated Total Size (MB): 398.45
# ----------------------------------------------------------------