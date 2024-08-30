import torch
import torchvision.transforms as transforms

import os
import sys 
sys.path.append(os.getcwd())  
from models.ResNet import ResNet18
from PIL import Image

def predict_(img):

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])
    # img = Image.open('')
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    model = ResNet18()
    model_weight_pth = './results/ResNet_001.pth'
    model.load_state_dict(torch.load(model_weight_pth))

    model.eval()

    
    classes = {'0': '飞机', '1': '汽车', '2': '鸟', '3': '猫', '4': '鹿', '5': '狗', '6': '青蛙', '7': '马', '8': '船', '9': '卡车'}
    
    with torch.no_grad():
        output = torch.squeeze(model(img))
        print(output)
        predict = torch.softmax(output, dim=0)

        predict_cla = torch.argmax(predict).numpy()

    return classes[str(predict_cla)], predict[predict_cla].item()

if __name__ == '__main__':
    img = Image.open('D:\\datasets\\first_feedback\\homemade2\\40187.jpg')
    net = predict_(img)
    print(net)
