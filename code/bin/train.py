import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse

import os
import sys 
sys.path.append(os.getcwd())  

from models.ResNet import ResNet18
# 定义是否使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 参数设置
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./results/', help='folder to output images and model checkpoints')  # 输出结果保存路径
parser.add_argument('--net', default='./results/ResNet.pth', help="path to net (to continue training)")  # 恢复训练时的模型路径
args = parser.parse_args()

# 超参数设置
EPOCH = 1  # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128  # 批处理尺寸(batch_size)
LR = 0.01  # 学习率

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    # transforms.ColorJitter(brightness=0.5, contrast=0.5),  # 随机调整亮度和对比度
    transforms.ToTensor(),
    # # R,G,B每层的归一化用到的均值和方差
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  #cifar10
    # transforms.Normalize((0.5071, 0.4865, 0.4408), (0.2675, 0.2565, 0.2761)),  #cifar100
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  #imagenet
    
])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # cifar10
    # transforms.Normalize((0.5071, 0.4865, 0.4408), (0.2675, 0.2565, 0.2761)),  #cifar100
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  #imagenet
    
])


trainset = torchvision.datasets.CIFAR10(root='../data', train=True,download=True, transform=transform_train)  # 训练数据集

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,num_workers=2)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取


testset = torchvision.datasets.CIFAR10(root='../data', train=False,download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Cifar-100的标签
# classes = ('beaver','dolphin','otter','seal','whale','aquarium fish','flatfish','ray','shark','trout','orchids',  
#            'poppies', 'roses', 'sunflowers', 'tulips','bottles', 'bowls', 'cans', 'cups', 'plates','apples',  
#            'mushrooms', 'oranges', 'pears', 'sweet peppers','clock', 'computer keyboard', 'lamp', 'telephone',  
#            'television','bed', 'chair', 'couch', 'table', 'wardrobe','bee', 'beetle', 'butterfly', 'caterpillar',  
#            'cockroach','bear', 'leopard', 'lion', 'tiger', 'wolf','bridge', 'castle', 'house', 'road', 'skyscraper', 
#            'cloud', 'forest', 'mountain', 'plain', 'sea','camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', 
#            'fox', 'porcupine', 'possum', 'raccoon', 'skunk','crab', 'lobster', 'snail', 'spider', 'worm','baby', 
#            'boy', 'girl', 'man', 'woman','crocodile','dinosaur', 'lizard', 'snake', 'turtle','hamster', 'mouse', 
#            'rabbit', 'shrew', 'squirrel','maple', 'oak', 'palm', 'pine', 'willow','bicycle', 'bus', 'motorcycle', 
#            'pickup truck', 'train','lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor')

# mini-imagenet标签
# classes = ('house_finch', 'robin', 'triceratops', 'green_mamba', 'harvestman', 'toucan', 'goose', 'jellyfish', 'nematode', 
#            'king_crab', 'dugong', 'Walker_hound', 'Ibizan_hound', 'Saluki', 'golden_retriever', 'Gordon_setter', 'komondor', 
#            'boxer', 'Tibetan_mastiff', 'French_bulldog', 'malamute', 'dalmatian', 'Newfoundland', 'miniature_poodle', 
#            'white_wolf', 'African_hunting_dog', 'Arctic_fox', 'lion', 'meerkat', 'ladybug', 'rhinoceros_beetle', 'ant', 
#            'black-footed_ferret', 'three-toed_sloth', 'rock_beauty', 'aircraft_carrier', 'ashcan', 'barrel', 'beer_bottle',
#            'bookshop', 'cannon', 'carousel', 'carton', 'catamaran', 'chime', 'clog', 'cocktail_shaker', 'combination_lock', 
#            'crate', 'cuirass', 'dishrag', 'dome', 'electric_guitar', 'file', 'fire_screen', 'frying_pan', 'garbage_truck', 
#            'hair_slide', 'holster', 'horizontal_bar', 'hourglass', 'iPod', 'lipstick', 'miniskirt', 'missile', 'mixing_bowl',
#            'oboe', 'organ', 'parallel_bars', 'pencil_box', 'photocopier', 'poncho', 'prayer_rug', 'reel', 'school_bus', 
#            'scoreboard', 'slot', 'snorkel', 'solar_dish', 'spider_web', 'stage', 'tank', 'theater_curtain', 'tile_roof',
#            'tobacco_shop', 'unicycle', 'upright', 'vase', 'wok', 'worm_fence', 'yawl', 'street_sign', 'consomme', 'trifle', 
#            'hotdog', 'orange', 'cliff', 'coral_reef', 'bolete', 'ear')

# 模型定义-ResNet
net = ResNet18().to(device)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
optimizer = optim.NAdam(net.parameters(), lr=0.002)

# 训练
if __name__ == "__main__":
    best_acc = 85  # 2 初始化best test accuracy
    print("Start Training, ResNet!")  # 定义遍历数据集的次数
    with open("acc.txt", "w") as f:
        with open("log.txt", "w") as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        # result = torch.floor_divide(correct, total)
                    # print('测试分类准确率为：%.3f%%' % (100 * result))
                    acc = 100 * correct / total
                    print('测试分类准确率为：%.3f%%' % (acc))
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/ResNet_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)

