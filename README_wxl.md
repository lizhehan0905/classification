# classification-pyqt
利用pyqt实现图像分类  

code文件夹保存代码  

1、requirements.txt文件中保存了需要的包，执行命令安装  
```pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/```  

2、训练的权重会保存在model文件中，执行以下命令创建文件夹。  
```mkdir model```  

3、执行train.py文件会自动创建data文件夹并在该文件下保存下载的数据集文件。（这里以CIFAR10数据集为例，需要CIFAR100数据集，按照下列命令修改）  
不同的数据集需要选择不同的mean，std，已在train.py文件中数据预处理代码中进行标注。  

将trainset和testset分别改为  
```trainset = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=transform_train)```  
```testset = torchvision.datasets.CIFAR100(root='./data', train=False,download=True, transform=transform_test)```   

对于其他的数据集，需要提前下载好，并划分为train、test文件夹，每个文件夹下需要根据每个类别分别保存在一个文件夹。并修改代码  
```trainset = torchvision.datasets.ImageFolder(root='你的训练集路径', transform=transform_train)```    
```testset = torchvision.datasets.ImageFolder(root='你的测试集或验证集路径', transform=transform_test)```   
同时根据不同数据的类别需要修改模型文件中num_classes  

4、acc.txt和log.txt用于记录训练过程中的数据。  
acc.txt保存每一个epoch结束后测试集的准确率。  
log.txt保存每次迭代后的训练情况。  

5、ResNet.py表示ResNet网络,可根据需求选择相应的模型.py文件  
   只需要在train.py文件中导入模块即可，并修改model定义  
   例如:  
   ```from DenseNet import DenseNet121```

6、导入对应模型后执行train.py进行训练，这里的训练参数如下：  
  >epoch=200，BATCH_SIZE=128,LR=0.01  
  损失函数为交叉熵损失函数  
  >criterion = nn.CrossEntropyLoss()  
  优化器为NAdam  
  >optimizer = optim.NAdam(net.parameters(), lr=0.002) 

  根据不同的模型修改权重的存储路径:  
  ```parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')```  
  
  
7、predict.py用于推理，在推理时只需要导入相应的模型即可，同时修改model、模型权重的路径，修改需要预测的图片路径。同时需要将数据集对应的类别字典进行修改。  
 这里以CIFAR10为例：  
 >classes = {'0': '飞机', '1': '汽车', '2': '鸟', '3': '猫', '4': '鹿', '5': '狗', '6': '青蛙', '7': '马', '8': '船', '9': '卡车'}  
  
8、GUI.py用于创建pyqt界面，读取文件和预测功能采用不同的线程执行。该文件会在PyQt_predict.py中调用。  

9、PyQt_predict.py用于生成界面进行预测。  

备注：在给出的模型中，结果最好的是ResNet和DenseNet，shuffleNetV2仅需30-50个epoch即可达到和ResNet、DenseNet相同的准确度，同时也是训练速度最快的。
