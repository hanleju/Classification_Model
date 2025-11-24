import torch
import torchvision

import argparse
from tqdm import tqdm

from model import *
from model.LeNet5 import LeNet
from model.AlexNet import alexnet
from model.GoogLeNet import googlenet
from model.ResNet50 import ResNet50
from model.ResNext50 import ResNext50
from model.MobileNet_V1 import mobilenetv1
from model.VGGNet import VGG
from model.SeResNet50 import seresnet50
from model.ViT import vit
from model.DenseNet121 import DenseNet121

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100/SVHN')
    parser.add_argument('--weights', '-w', type=str, help='model weight path')
    parser.add_argument('--model', type=str, help='choose model')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset', choices=['cifar10', 'cifar100', 'svhn'])
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')

    args = parser.parse_args()
    return args

def data():
    args = parse_args()
    
    batch_size = args.batch_size

    if args.dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    elif args.dataset == 'cifar100':
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)
    elif args.dataset == 'svhn':
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return testloader

def main():
    args = parse_args()

    print('==> Preparing data..')

    testloader = data()

    model_path = args.weights # 학습 시킨 모델 경로

    if args.dataset == 'cifar10' or args.dataset == 'svhn':
        num_classes = 10
    else:  # cifar100
        num_classes = 100

    if args.model == 'vgg':
        model = VGG(num_classes=num_classes)
    if args.model == 'resnet50':
        model = ResNet50(num_classes=num_classes)
    if args.model == 'resnext50':
        model = ResNext50(num_classes=num_classes)
    if args.model == 'mobilenetv1':
        model = mobilenetv1(num_classes=num_classes)
    if args.model == 'densenet121':
        model = DenseNet121(num_classes=num_classes)
    if args.model == 'vit':
        model = vit(num_classes=num_classes)

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