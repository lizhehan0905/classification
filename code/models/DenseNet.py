import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Dense_Layer(nn.Module):
    def __init__(self,inchannel,growth_rate,BN_Size):
        super(Dense_Layer,self).__init__()

        self.conv=nn.Sequential(
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            # growth_rate:增长率，一层产生多少个特征图
            nn.Conv2d(inchannel,growth_rate*BN_Size,kernel_size=1),
            nn.BatchNorm2d(growth_rate*BN_Size),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate*BN_Size,growth_rate, kernel_size=3,padding=1),
        )

    def forward(self,x):
        out=self.conv(x)
        # 将输入x和输出y连接是DenseNet中的关键操作，这种操作被称为密集连接（Dense Connection）。密集连接的主要目的是促进梯度的流动和信息传播，从而帮助网络更好地学习特征。
        # 通过将输入x与输出y连接起来，可以使网络的每一层都能直接访问之前所有层的特征图。这种密集连接有助于缓解梯度消失问题，允许网络更容易地学习到细粒度的特征，并提高网络的性能和训练速度。

        out=torch.cat([x,out],1)
        return out

class Dense_block(nn.Module):
    def __init__(self,inchannel,growth_rate,BN_Size,num_layers):
        super(Dense_block,self).__init__()
        layers=[]
        # 随着layer层数的增加，每增加一层，输入的特征图就增加一倍growth_rate
        for i in range(num_layers):
            layers.append(Dense_Layer(inchannel+i*growth_rate,growth_rate,BN_Size))
        self.layers=nn.Sequential(*layers)

    def forward(self,x):
        out=self.layers(x)
        return out

class Transition(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(Transition,self).__init__()

        self.conv=nn.Sequential(
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2),
        )

    def forward(self,x):
        out=self.conv(x)
        return out


class DenseNet(nn.Module):
    def __init__(self,init_channel=64,growth_rate=32,BN_Size=4,num_classes=10,blocks=[6,12,24,16]):
        super(DenseNet,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, init_channel, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(init_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        )

        # # 第1个DenseBlock有6个DenseLayer, 执行DenseBlock（6,64,32,4）
        self.dense_block1 = Dense_block(init_channel,growth_rate,BN_Size, blocks[0])
        init_channel=init_channel+blocks[0]*growth_rate
        # 第1个transition 执行 TransitionLayer（256,128）
        self.transition1 = Transition(init_channel, init_channel // 2)
        # #num_features减少为原来的一半，执行第1回合之后，第2个DenseBlock的输入的feature应该是：num_features = 128
        init_channel = init_channel // 2

        # 第2个DenseBlock有12个DenseLayer, 执行DenseBlock（12,128,32,4）
        self.dense_block2 = Dense_block(init_channel,growth_rate,BN_Size,blocks[1])
        init_channel = init_channel + blocks[1] * growth_rate
        # 第2个transition 执行 TransitionLayer（512,256）
        self.transition2 = Transition(init_channel, init_channel // 2)
        # num_features减少为原来的一半，执行第2回合之后，第3个DenseBlock的输入的feature应该是：num_features = 256
        init_channel = init_channel // 2

        # 第3个DenseBlock有24个DenseLayer, 执行DenseBlock（24,256,32,4）
        self.dense_block3 = Dense_block(init_channel,growth_rate,BN_Size, blocks[2])
        init_channel = init_channel + blocks[2] * growth_rate
        # 第3个transition 执行 _TransitionLayer（1024,512）
        self.transition3 = Transition(init_channel, init_channel // 2)
        # num_features减少为原来的一半，执行第3回合之后，第4个DenseBlock的输入的feature应该是：num_features = 512
        init_channel = init_channel // 2

        # 第4个DenseBlock有16个DenseLayer, 执行DenseBlock（16,512,32,4）
        self.dense_block4 = Dense_block(init_channel,growth_rate,BN_Size, blocks[3])
        init_channel = init_channel + blocks[3] * growth_rate

        self.avg_pool=nn.AvgPool2d(kernel_size=7,stride=1)
        self.fc=nn.Linear(init_channel,num_classes)

    def forward(self,x):
        out = self.conv(x)

        out = self.dense_block1(out)
        out = self.transition1(out)

        out = self.dense_block2(out)
        out = self.transition2(out)

        out = self.dense_block3(out)
        out = self.transition3(out)

        out = self.dense_block4(out)

        out = self.avg_pool(out)
        out = torch.flatten(out,1)
        out = self.fc(out)

        return out



if __name__ == '__main__':
    # model = torchvision.models.densenet121()
    model = DenseNet().cuda()
    summary(model,(3,224,224))


# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 64, 112, 112]           9,472
#        BatchNorm2d-2         [-1, 64, 112, 112]             128
#               ReLU-3         [-1, 64, 112, 112]               0
#          MaxPool2d-4           [-1, 64, 56, 56]               0
#        BatchNorm2d-5           [-1, 64, 56, 56]             128
#               ReLU-6           [-1, 64, 56, 56]               0
#             Conv2d-7          [-1, 128, 56, 56]           8,320
#        BatchNorm2d-8          [-1, 128, 56, 56]             256
#               ReLU-9          [-1, 128, 56, 56]               0
#            Conv2d-10           [-1, 32, 56, 56]          36,896
#       Dense_Layer-11           [-1, 96, 56, 56]               0
#       BatchNorm2d-12           [-1, 96, 56, 56]             192
#              ReLU-13           [-1, 96, 56, 56]               0
#            Conv2d-14          [-1, 128, 56, 56]          12,416
#       BatchNorm2d-15          [-1, 128, 56, 56]             256
#              ReLU-16          [-1, 128, 56, 56]               0
#            Conv2d-17           [-1, 32, 56, 56]          36,896
#       Dense_Layer-18          [-1, 128, 56, 56]               0
#       BatchNorm2d-19          [-1, 128, 56, 56]             256
#              ReLU-20          [-1, 128, 56, 56]               0
#            Conv2d-21          [-1, 128, 56, 56]          16,512
#       BatchNorm2d-22          [-1, 128, 56, 56]             256
#              ReLU-23          [-1, 128, 56, 56]               0
#            Conv2d-24           [-1, 32, 56, 56]          36,896
#       Dense_Layer-25          [-1, 160, 56, 56]               0
#       BatchNorm2d-26          [-1, 160, 56, 56]             320
#              ReLU-27          [-1, 160, 56, 56]               0
#            Conv2d-28          [-1, 128, 56, 56]          20,608
#       BatchNorm2d-29          [-1, 128, 56, 56]             256
#              ReLU-30          [-1, 128, 56, 56]               0
#            Conv2d-31           [-1, 32, 56, 56]          36,896
#       Dense_Layer-32          [-1, 192, 56, 56]               0
#       BatchNorm2d-33          [-1, 192, 56, 56]             384
#              ReLU-34          [-1, 192, 56, 56]               0
#            Conv2d-35          [-1, 128, 56, 56]          24,704
#       BatchNorm2d-36          [-1, 128, 56, 56]             256
#              ReLU-37          [-1, 128, 56, 56]               0
#            Conv2d-38           [-1, 32, 56, 56]          36,896
#       Dense_Layer-39          [-1, 224, 56, 56]               0
#       BatchNorm2d-40          [-1, 224, 56, 56]             448
#              ReLU-41          [-1, 224, 56, 56]               0
#            Conv2d-42          [-1, 128, 56, 56]          28,800
#       BatchNorm2d-43          [-1, 128, 56, 56]             256
#              ReLU-44          [-1, 128, 56, 56]               0
#            Conv2d-45           [-1, 32, 56, 56]          36,896
#       Dense_Layer-46          [-1, 256, 56, 56]               0
#       Dense_block-47          [-1, 256, 56, 56]               0
#       BatchNorm2d-48          [-1, 256, 56, 56]             512
#              ReLU-49          [-1, 256, 56, 56]               0
#            Conv2d-50          [-1, 128, 56, 56]          32,896
#         AvgPool2d-51          [-1, 128, 28, 28]               0
#        Transition-52          [-1, 128, 28, 28]               0
#       BatchNorm2d-53          [-1, 128, 28, 28]             256
#              ReLU-54          [-1, 128, 28, 28]               0
#            Conv2d-55          [-1, 128, 28, 28]          16,512
#       BatchNorm2d-56          [-1, 128, 28, 28]             256
#              ReLU-57          [-1, 128, 28, 28]               0
#            Conv2d-58           [-1, 32, 28, 28]          36,896
#       Dense_Layer-59          [-1, 160, 28, 28]               0
#       BatchNorm2d-60          [-1, 160, 28, 28]             320
#              ReLU-61          [-1, 160, 28, 28]               0
#            Conv2d-62          [-1, 128, 28, 28]          20,608
#       BatchNorm2d-63          [-1, 128, 28, 28]             256
#              ReLU-64          [-1, 128, 28, 28]               0
#            Conv2d-65           [-1, 32, 28, 28]          36,896
#       Dense_Layer-66          [-1, 192, 28, 28]               0
#       BatchNorm2d-67          [-1, 192, 28, 28]             384
#              ReLU-68          [-1, 192, 28, 28]               0
#            Conv2d-69          [-1, 128, 28, 28]          24,704
#       BatchNorm2d-70          [-1, 128, 28, 28]             256
#              ReLU-71          [-1, 128, 28, 28]               0
#            Conv2d-72           [-1, 32, 28, 28]          36,896
#       Dense_Layer-73          [-1, 224, 28, 28]               0
#       BatchNorm2d-74          [-1, 224, 28, 28]             448
#              ReLU-75          [-1, 224, 28, 28]               0
#            Conv2d-76          [-1, 128, 28, 28]          28,800
#       BatchNorm2d-77          [-1, 128, 28, 28]             256
#              ReLU-78          [-1, 128, 28, 28]               0
#            Conv2d-79           [-1, 32, 28, 28]          36,896
#       Dense_Layer-80          [-1, 256, 28, 28]               0
#       BatchNorm2d-81          [-1, 256, 28, 28]             512
#              ReLU-82          [-1, 256, 28, 28]               0
#            Conv2d-83          [-1, 128, 28, 28]          32,896
#       BatchNorm2d-84          [-1, 128, 28, 28]             256
#              ReLU-85          [-1, 128, 28, 28]               0
#            Conv2d-86           [-1, 32, 28, 28]          36,896
#       Dense_Layer-87          [-1, 288, 28, 28]               0
#       BatchNorm2d-88          [-1, 288, 28, 28]             576
#              ReLU-89          [-1, 288, 28, 28]               0
#            Conv2d-90          [-1, 128, 28, 28]          36,992
#       BatchNorm2d-91          [-1, 128, 28, 28]             256
#              ReLU-92          [-1, 128, 28, 28]               0
#            Conv2d-93           [-1, 32, 28, 28]          36,896
#       Dense_Layer-94          [-1, 320, 28, 28]               0
#       BatchNorm2d-95          [-1, 320, 28, 28]             640
#              ReLU-96          [-1, 320, 28, 28]               0
#            Conv2d-97          [-1, 128, 28, 28]          41,088
#       BatchNorm2d-98          [-1, 128, 28, 28]             256
#              ReLU-99          [-1, 128, 28, 28]               0
#           Conv2d-100           [-1, 32, 28, 28]          36,896
#      Dense_Layer-101          [-1, 352, 28, 28]               0
#      BatchNorm2d-102          [-1, 352, 28, 28]             704
#             ReLU-103          [-1, 352, 28, 28]               0
#           Conv2d-104          [-1, 128, 28, 28]          45,184
#      BatchNorm2d-105          [-1, 128, 28, 28]             256
#             ReLU-106          [-1, 128, 28, 28]               0
#           Conv2d-107           [-1, 32, 28, 28]          36,896
#      Dense_Layer-108          [-1, 384, 28, 28]               0
#      BatchNorm2d-109          [-1, 384, 28, 28]             768
#             ReLU-110          [-1, 384, 28, 28]               0
#           Conv2d-111          [-1, 128, 28, 28]          49,280
#      BatchNorm2d-112          [-1, 128, 28, 28]             256
#             ReLU-113          [-1, 128, 28, 28]               0
#           Conv2d-114           [-1, 32, 28, 28]          36,896
#      Dense_Layer-115          [-1, 416, 28, 28]               0
#      BatchNorm2d-116          [-1, 416, 28, 28]             832
#             ReLU-117          [-1, 416, 28, 28]               0
#           Conv2d-118          [-1, 128, 28, 28]          53,376
#      BatchNorm2d-119          [-1, 128, 28, 28]             256
#             ReLU-120          [-1, 128, 28, 28]               0
#           Conv2d-121           [-1, 32, 28, 28]          36,896
#      Dense_Layer-122          [-1, 448, 28, 28]               0
#      BatchNorm2d-123          [-1, 448, 28, 28]             896
#             ReLU-124          [-1, 448, 28, 28]               0
#           Conv2d-125          [-1, 128, 28, 28]          57,472
#      BatchNorm2d-126          [-1, 128, 28, 28]             256
#             ReLU-127          [-1, 128, 28, 28]               0
#           Conv2d-128           [-1, 32, 28, 28]          36,896
#      Dense_Layer-129          [-1, 480, 28, 28]               0
#      BatchNorm2d-130          [-1, 480, 28, 28]             960
#             ReLU-131          [-1, 480, 28, 28]               0
#           Conv2d-132          [-1, 128, 28, 28]          61,568
#      BatchNorm2d-133          [-1, 128, 28, 28]             256
#             ReLU-134          [-1, 128, 28, 28]               0
#           Conv2d-135           [-1, 32, 28, 28]          36,896
#      Dense_Layer-136          [-1, 512, 28, 28]               0
#      Dense_block-137          [-1, 512, 28, 28]               0
#      BatchNorm2d-138          [-1, 512, 28, 28]           1,024
#             ReLU-139          [-1, 512, 28, 28]               0
#           Conv2d-140          [-1, 256, 28, 28]         131,328
#        AvgPool2d-141          [-1, 256, 14, 14]               0
#       Transition-142          [-1, 256, 14, 14]               0
#      BatchNorm2d-143          [-1, 256, 14, 14]             512
#             ReLU-144          [-1, 256, 14, 14]               0
#           Conv2d-145          [-1, 128, 14, 14]          32,896
#      BatchNorm2d-146          [-1, 128, 14, 14]             256
#             ReLU-147          [-1, 128, 14, 14]               0
#           Conv2d-148           [-1, 32, 14, 14]          36,896
#      Dense_Layer-149          [-1, 288, 14, 14]               0
#      BatchNorm2d-150          [-1, 288, 14, 14]             576
#             ReLU-151          [-1, 288, 14, 14]               0
#           Conv2d-152          [-1, 128, 14, 14]          36,992
#      BatchNorm2d-153          [-1, 128, 14, 14]             256
#             ReLU-154          [-1, 128, 14, 14]               0
#           Conv2d-155           [-1, 32, 14, 14]          36,896
#      Dense_Layer-156          [-1, 320, 14, 14]               0
#      BatchNorm2d-157          [-1, 320, 14, 14]             640
#             ReLU-158          [-1, 320, 14, 14]               0
#           Conv2d-159          [-1, 128, 14, 14]          41,088
#      BatchNorm2d-160          [-1, 128, 14, 14]             256
#             ReLU-161          [-1, 128, 14, 14]               0
#           Conv2d-162           [-1, 32, 14, 14]          36,896
#      Dense_Layer-163          [-1, 352, 14, 14]               0
#      BatchNorm2d-164          [-1, 352, 14, 14]             704
#             ReLU-165          [-1, 352, 14, 14]               0
#           Conv2d-166          [-1, 128, 14, 14]          45,184
#      BatchNorm2d-167          [-1, 128, 14, 14]             256
#             ReLU-168          [-1, 128, 14, 14]               0
#           Conv2d-169           [-1, 32, 14, 14]          36,896
#      Dense_Layer-170          [-1, 384, 14, 14]               0
#      BatchNorm2d-171          [-1, 384, 14, 14]             768
#             ReLU-172          [-1, 384, 14, 14]               0
#           Conv2d-173          [-1, 128, 14, 14]          49,280
#      BatchNorm2d-174          [-1, 128, 14, 14]             256
#             ReLU-175          [-1, 128, 14, 14]               0
#           Conv2d-176           [-1, 32, 14, 14]          36,896
#      Dense_Layer-177          [-1, 416, 14, 14]               0
#      BatchNorm2d-178          [-1, 416, 14, 14]             832
#             ReLU-179          [-1, 416, 14, 14]               0
#           Conv2d-180          [-1, 128, 14, 14]          53,376
#      BatchNorm2d-181          [-1, 128, 14, 14]             256
#             ReLU-182          [-1, 128, 14, 14]               0
#           Conv2d-183           [-1, 32, 14, 14]          36,896
#      Dense_Layer-184          [-1, 448, 14, 14]               0
#      BatchNorm2d-185          [-1, 448, 14, 14]             896
#             ReLU-186          [-1, 448, 14, 14]               0
#           Conv2d-187          [-1, 128, 14, 14]          57,472
#      BatchNorm2d-188          [-1, 128, 14, 14]             256
#             ReLU-189          [-1, 128, 14, 14]               0
#           Conv2d-190           [-1, 32, 14, 14]          36,896
#      Dense_Layer-191          [-1, 480, 14, 14]               0
#      BatchNorm2d-192          [-1, 480, 14, 14]             960
#             ReLU-193          [-1, 480, 14, 14]               0
#           Conv2d-194          [-1, 128, 14, 14]          61,568
#      BatchNorm2d-195          [-1, 128, 14, 14]             256
#             ReLU-196          [-1, 128, 14, 14]               0
#           Conv2d-197           [-1, 32, 14, 14]          36,896
#      Dense_Layer-198          [-1, 512, 14, 14]               0
#      BatchNorm2d-199          [-1, 512, 14, 14]           1,024
#             ReLU-200          [-1, 512, 14, 14]               0
#           Conv2d-201          [-1, 128, 14, 14]          65,664
#      BatchNorm2d-202          [-1, 128, 14, 14]             256
#             ReLU-203          [-1, 128, 14, 14]               0
#           Conv2d-204           [-1, 32, 14, 14]          36,896
#      Dense_Layer-205          [-1, 544, 14, 14]               0
#      BatchNorm2d-206          [-1, 544, 14, 14]           1,088
#             ReLU-207          [-1, 544, 14, 14]               0
#           Conv2d-208          [-1, 128, 14, 14]          69,760
#      BatchNorm2d-209          [-1, 128, 14, 14]             256
#             ReLU-210          [-1, 128, 14, 14]               0
#           Conv2d-211           [-1, 32, 14, 14]          36,896
#      Dense_Layer-212          [-1, 576, 14, 14]               0
#      BatchNorm2d-213          [-1, 576, 14, 14]           1,152
#             ReLU-214          [-1, 576, 14, 14]               0
#           Conv2d-215          [-1, 128, 14, 14]          73,856
#      BatchNorm2d-216          [-1, 128, 14, 14]             256
#             ReLU-217          [-1, 128, 14, 14]               0
#           Conv2d-218           [-1, 32, 14, 14]          36,896
#      Dense_Layer-219          [-1, 608, 14, 14]               0
#      BatchNorm2d-220          [-1, 608, 14, 14]           1,216
#             ReLU-221          [-1, 608, 14, 14]               0
#           Conv2d-222          [-1, 128, 14, 14]          77,952
#      BatchNorm2d-223          [-1, 128, 14, 14]             256
#             ReLU-224          [-1, 128, 14, 14]               0
#           Conv2d-225           [-1, 32, 14, 14]          36,896
#      Dense_Layer-226          [-1, 640, 14, 14]               0
#      BatchNorm2d-227          [-1, 640, 14, 14]           1,280
#             ReLU-228          [-1, 640, 14, 14]               0
#           Conv2d-229          [-1, 128, 14, 14]          82,048
#      BatchNorm2d-230          [-1, 128, 14, 14]             256
#             ReLU-231          [-1, 128, 14, 14]               0
#           Conv2d-232           [-1, 32, 14, 14]          36,896
#      Dense_Layer-233          [-1, 672, 14, 14]               0
#      BatchNorm2d-234          [-1, 672, 14, 14]           1,344
#             ReLU-235          [-1, 672, 14, 14]               0
#           Conv2d-236          [-1, 128, 14, 14]          86,144
#      BatchNorm2d-237          [-1, 128, 14, 14]             256
#             ReLU-238          [-1, 128, 14, 14]               0
#           Conv2d-239           [-1, 32, 14, 14]          36,896
#      Dense_Layer-240          [-1, 704, 14, 14]               0
#      BatchNorm2d-241          [-1, 704, 14, 14]           1,408
#             ReLU-242          [-1, 704, 14, 14]               0
#           Conv2d-243          [-1, 128, 14, 14]          90,240
#      BatchNorm2d-244          [-1, 128, 14, 14]             256
#             ReLU-245          [-1, 128, 14, 14]               0
#           Conv2d-246           [-1, 32, 14, 14]          36,896
#      Dense_Layer-247          [-1, 736, 14, 14]               0
#      BatchNorm2d-248          [-1, 736, 14, 14]           1,472
#             ReLU-249          [-1, 736, 14, 14]               0
#           Conv2d-250          [-1, 128, 14, 14]          94,336
#      BatchNorm2d-251          [-1, 128, 14, 14]             256
#             ReLU-252          [-1, 128, 14, 14]               0
#           Conv2d-253           [-1, 32, 14, 14]          36,896
#      Dense_Layer-254          [-1, 768, 14, 14]               0
#      BatchNorm2d-255          [-1, 768, 14, 14]           1,536
#             ReLU-256          [-1, 768, 14, 14]               0
#           Conv2d-257          [-1, 128, 14, 14]          98,432
#      BatchNorm2d-258          [-1, 128, 14, 14]             256
#             ReLU-259          [-1, 128, 14, 14]               0
#           Conv2d-260           [-1, 32, 14, 14]          36,896
#      Dense_Layer-261          [-1, 800, 14, 14]               0
#      BatchNorm2d-262          [-1, 800, 14, 14]           1,600
#             ReLU-263          [-1, 800, 14, 14]               0
#           Conv2d-264          [-1, 128, 14, 14]         102,528
#      BatchNorm2d-265          [-1, 128, 14, 14]             256
#             ReLU-266          [-1, 128, 14, 14]               0
#           Conv2d-267           [-1, 32, 14, 14]          36,896
#      Dense_Layer-268          [-1, 832, 14, 14]               0
#      BatchNorm2d-269          [-1, 832, 14, 14]           1,664
#             ReLU-270          [-1, 832, 14, 14]               0
#           Conv2d-271          [-1, 128, 14, 14]         106,624
#      BatchNorm2d-272          [-1, 128, 14, 14]             256
#             ReLU-273          [-1, 128, 14, 14]               0
#           Conv2d-274           [-1, 32, 14, 14]          36,896
#      Dense_Layer-275          [-1, 864, 14, 14]               0
#      BatchNorm2d-276          [-1, 864, 14, 14]           1,728
#             ReLU-277          [-1, 864, 14, 14]               0
#           Conv2d-278          [-1, 128, 14, 14]         110,720
#      BatchNorm2d-279          [-1, 128, 14, 14]             256
#             ReLU-280          [-1, 128, 14, 14]               0
#           Conv2d-281           [-1, 32, 14, 14]          36,896
#      Dense_Layer-282          [-1, 896, 14, 14]               0
#      BatchNorm2d-283          [-1, 896, 14, 14]           1,792
#             ReLU-284          [-1, 896, 14, 14]               0
#           Conv2d-285          [-1, 128, 14, 14]         114,816
#      BatchNorm2d-286          [-1, 128, 14, 14]             256
#             ReLU-287          [-1, 128, 14, 14]               0
#           Conv2d-288           [-1, 32, 14, 14]          36,896
#      Dense_Layer-289          [-1, 928, 14, 14]               0
#      BatchNorm2d-290          [-1, 928, 14, 14]           1,856
#             ReLU-291          [-1, 928, 14, 14]               0
#           Conv2d-292          [-1, 128, 14, 14]         118,912
#      BatchNorm2d-293          [-1, 128, 14, 14]             256
#             ReLU-294          [-1, 128, 14, 14]               0
#           Conv2d-295           [-1, 32, 14, 14]          36,896
#      Dense_Layer-296          [-1, 960, 14, 14]               0
#      BatchNorm2d-297          [-1, 960, 14, 14]           1,920
#             ReLU-298          [-1, 960, 14, 14]               0
#           Conv2d-299          [-1, 128, 14, 14]         123,008
#      BatchNorm2d-300          [-1, 128, 14, 14]             256
#             ReLU-301          [-1, 128, 14, 14]               0
#           Conv2d-302           [-1, 32, 14, 14]          36,896
#      Dense_Layer-303          [-1, 992, 14, 14]               0
#      BatchNorm2d-304          [-1, 992, 14, 14]           1,984
#             ReLU-305          [-1, 992, 14, 14]               0
#           Conv2d-306          [-1, 128, 14, 14]         127,104
#      BatchNorm2d-307          [-1, 128, 14, 14]             256
#             ReLU-308          [-1, 128, 14, 14]               0
#           Conv2d-309           [-1, 32, 14, 14]          36,896
#      Dense_Layer-310         [-1, 1024, 14, 14]               0
#      Dense_block-311         [-1, 1024, 14, 14]               0
#      BatchNorm2d-312         [-1, 1024, 14, 14]           2,048
#             ReLU-313         [-1, 1024, 14, 14]               0
#           Conv2d-314          [-1, 512, 14, 14]         524,800
#        AvgPool2d-315            [-1, 512, 7, 7]               0
#       Transition-316            [-1, 512, 7, 7]               0
#      BatchNorm2d-317            [-1, 512, 7, 7]           1,024
#             ReLU-318            [-1, 512, 7, 7]               0
#           Conv2d-319            [-1, 128, 7, 7]          65,664
#      BatchNorm2d-320            [-1, 128, 7, 7]             256
#             ReLU-321            [-1, 128, 7, 7]               0
#           Conv2d-322             [-1, 32, 7, 7]          36,896
#      Dense_Layer-323            [-1, 544, 7, 7]               0
#      BatchNorm2d-324            [-1, 544, 7, 7]           1,088
#             ReLU-325            [-1, 544, 7, 7]               0
#           Conv2d-326            [-1, 128, 7, 7]          69,760
#      BatchNorm2d-327            [-1, 128, 7, 7]             256
#             ReLU-328            [-1, 128, 7, 7]               0
#           Conv2d-329             [-1, 32, 7, 7]          36,896
#      Dense_Layer-330            [-1, 576, 7, 7]               0
#      BatchNorm2d-331            [-1, 576, 7, 7]           1,152
#             ReLU-332            [-1, 576, 7, 7]               0
#           Conv2d-333            [-1, 128, 7, 7]          73,856
#      BatchNorm2d-334            [-1, 128, 7, 7]             256
#             ReLU-335            [-1, 128, 7, 7]               0
#           Conv2d-336             [-1, 32, 7, 7]          36,896
#      Dense_Layer-337            [-1, 608, 7, 7]               0
#      BatchNorm2d-338            [-1, 608, 7, 7]           1,216
#             ReLU-339            [-1, 608, 7, 7]               0
#           Conv2d-340            [-1, 128, 7, 7]          77,952
#      BatchNorm2d-341            [-1, 128, 7, 7]             256
#             ReLU-342            [-1, 128, 7, 7]               0
#           Conv2d-343             [-1, 32, 7, 7]          36,896
#      Dense_Layer-344            [-1, 640, 7, 7]               0
#      BatchNorm2d-345            [-1, 640, 7, 7]           1,280
#             ReLU-346            [-1, 640, 7, 7]               0
#           Conv2d-347            [-1, 128, 7, 7]          82,048
#      BatchNorm2d-348            [-1, 128, 7, 7]             256
#             ReLU-349            [-1, 128, 7, 7]               0
#           Conv2d-350             [-1, 32, 7, 7]          36,896
#      Dense_Layer-351            [-1, 672, 7, 7]               0
#      BatchNorm2d-352            [-1, 672, 7, 7]           1,344
#             ReLU-353            [-1, 672, 7, 7]               0
#           Conv2d-354            [-1, 128, 7, 7]          86,144
#      BatchNorm2d-355            [-1, 128, 7, 7]             256
#             ReLU-356            [-1, 128, 7, 7]               0
#           Conv2d-357             [-1, 32, 7, 7]          36,896
#      Dense_Layer-358            [-1, 704, 7, 7]               0
#      BatchNorm2d-359            [-1, 704, 7, 7]           1,408
#             ReLU-360            [-1, 704, 7, 7]               0
#           Conv2d-361            [-1, 128, 7, 7]          90,240
#      BatchNorm2d-362            [-1, 128, 7, 7]             256
#             ReLU-363            [-1, 128, 7, 7]               0
#           Conv2d-364             [-1, 32, 7, 7]          36,896
#      Dense_Layer-365            [-1, 736, 7, 7]               0
#      BatchNorm2d-366            [-1, 736, 7, 7]           1,472
#             ReLU-367            [-1, 736, 7, 7]               0
#           Conv2d-368            [-1, 128, 7, 7]          94,336
#      BatchNorm2d-369            [-1, 128, 7, 7]             256
#             ReLU-370            [-1, 128, 7, 7]               0
#           Conv2d-371             [-1, 32, 7, 7]          36,896
#      Dense_Layer-372            [-1, 768, 7, 7]               0
#      BatchNorm2d-373            [-1, 768, 7, 7]           1,536
#             ReLU-374            [-1, 768, 7, 7]               0
#           Conv2d-375            [-1, 128, 7, 7]          98,432
#      BatchNorm2d-376            [-1, 128, 7, 7]             256
#             ReLU-377            [-1, 128, 7, 7]               0
#           Conv2d-378             [-1, 32, 7, 7]          36,896
#      Dense_Layer-379            [-1, 800, 7, 7]               0
#      BatchNorm2d-380            [-1, 800, 7, 7]           1,600
#             ReLU-381            [-1, 800, 7, 7]               0
#           Conv2d-382            [-1, 128, 7, 7]         102,528
#      BatchNorm2d-383            [-1, 128, 7, 7]             256
#             ReLU-384            [-1, 128, 7, 7]               0
#           Conv2d-385             [-1, 32, 7, 7]          36,896
#      Dense_Layer-386            [-1, 832, 7, 7]               0
#      BatchNorm2d-387            [-1, 832, 7, 7]           1,664
#             ReLU-388            [-1, 832, 7, 7]               0
#           Conv2d-389            [-1, 128, 7, 7]         106,624
#      BatchNorm2d-390            [-1, 128, 7, 7]             256
#             ReLU-391            [-1, 128, 7, 7]               0
#           Conv2d-392             [-1, 32, 7, 7]          36,896
#      Dense_Layer-393            [-1, 864, 7, 7]               0
#      BatchNorm2d-394            [-1, 864, 7, 7]           1,728
#             ReLU-395            [-1, 864, 7, 7]               0
#           Conv2d-396            [-1, 128, 7, 7]         110,720
#      BatchNorm2d-397            [-1, 128, 7, 7]             256
#             ReLU-398            [-1, 128, 7, 7]               0
#           Conv2d-399             [-1, 32, 7, 7]          36,896
#      Dense_Layer-400            [-1, 896, 7, 7]               0
#      BatchNorm2d-401            [-1, 896, 7, 7]           1,792
#             ReLU-402            [-1, 896, 7, 7]               0
#           Conv2d-403            [-1, 128, 7, 7]         114,816
#      BatchNorm2d-404            [-1, 128, 7, 7]             256
#             ReLU-405            [-1, 128, 7, 7]               0
#           Conv2d-406             [-1, 32, 7, 7]          36,896
#      Dense_Layer-407            [-1, 928, 7, 7]               0
#      BatchNorm2d-408            [-1, 928, 7, 7]           1,856
#             ReLU-409            [-1, 928, 7, 7]               0
#           Conv2d-410            [-1, 128, 7, 7]         118,912
#      BatchNorm2d-411            [-1, 128, 7, 7]             256
#             ReLU-412            [-1, 128, 7, 7]               0
#           Conv2d-413             [-1, 32, 7, 7]          36,896
#      Dense_Layer-414            [-1, 960, 7, 7]               0
#      BatchNorm2d-415            [-1, 960, 7, 7]           1,920
#             ReLU-416            [-1, 960, 7, 7]               0
#           Conv2d-417            [-1, 128, 7, 7]         123,008
#      BatchNorm2d-418            [-1, 128, 7, 7]             256
#             ReLU-419            [-1, 128, 7, 7]               0
#           Conv2d-420             [-1, 32, 7, 7]          36,896
#      Dense_Layer-421            [-1, 992, 7, 7]               0
#      BatchNorm2d-422            [-1, 992, 7, 7]           1,984
#             ReLU-423            [-1, 992, 7, 7]               0
#           Conv2d-424            [-1, 128, 7, 7]         127,104
#      BatchNorm2d-425            [-1, 128, 7, 7]             256
#             ReLU-426            [-1, 128, 7, 7]               0
#           Conv2d-427             [-1, 32, 7, 7]          36,896
#      Dense_Layer-428           [-1, 1024, 7, 7]               0
#      Dense_block-429           [-1, 1024, 7, 7]               0
#        AvgPool2d-430           [-1, 1024, 1, 1]               0
#           Linear-431                   [-1, 10]          10,250
# ================================================================
# Total params: 6,972,298
# Trainable params: 6,972,298
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 383.87
# Params size (MB): 26.60
# Estimated Total Size (MB): 411.04
# ----------------------------------------------------------------