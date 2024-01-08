import torch

import cv2
import argparse
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms

from model import *
from utils import *
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10')
    parser.add_argument('--pretrained', type=str, help='pretrained model path')
    parser.add_argument('--model', type=str, help='choose model')
    parser.add_argument('--transform', type=str,default='origin', help='choose data transfer type', choices=['origin','gray','yuv','hsv'])
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')

    args = parser.parse_args()
    return args

def data():
    args = parse_args()
    if args.transform == 'origin':
        transform = ConvertColor.transform
    if args.transform == 'gray':
        transform = ConvertColor.gray_transform
    if args.transform == 'yuv':
        transform = ConvertColor.yuv_transform
    if args.transform == 'hsv':
        transform = ConvertColor.hsv_transform
    
    batch_size = args.batch_size

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)
    return testloader

def main():
    args = parse_args()

    print('==> Preparing data..')

    testloader = data()

    model_path = args.pretrained # 학습 시킨 모델 경로

    if args.model == 'vgg':
        model = VGGNet()
    if args.model == 'resnet50':
        model = resnet50()
    if args.model == 'resnext50':
        model = ResNext50()
    if args.model == 'seresnet50':
        model = seresnet50()
    if args.model == 'mobilenetv1':
        model = mobilenetv1()
    if args.model == 'mobilenetv2':
        model = mobilenetv2()

    model = model.to(device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('==> Start evaluating..')

    model.eval()

    with torch.no_grad():
        total_accuracy = 0.0
        accuracy_list = []
        for image, label in tqdm(testloader, desc='Testing'):
            x = image.to(device)
            y_ = label.to(device)
            output = model(x)
            
            _, predicted = torch.max(output, 1)
            accuracy = (predicted == y_).sum().item() / len(y_)
            total_accuracy += accuracy
            accuracy_list.append(accuracy)
        print(f'Min Acc: {min(accuracy_list)*100:.2f}%')
        print(f'Max Acc: {max(accuracy_list)*100:.2f}%')
        print(f'Average Acc: {(total_accuracy/len(testloader))*100:.2f}%')
        print(f'Last Acc: {accuracy * 100:.2f}%')
        

if __name__ == '__main__':
    main()