import torch
import torchvision
import torchvision.transforms as transforms
import os

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

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if args.dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif args.dataset == 'cifar100':
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    elif args.dataset == 'svhn':
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return testloader

def main():
    args = parse_args()

    print('==> Preparing data..')

    testloader = data()

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
    
    model_path = args.weights # 학습 시킨 모델 경로
    
    # Generate result file path (same directory as weight, replace .pth with _result.txt)
    result_path = model_path.replace('.pth', '_result.txt')

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('==> Start evaluating..')
    print(f'==> Results will be saved to: {result_path}')

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
        
        min_acc = min(accuracy_list) * 100
        max_acc = max(accuracy_list) * 100
        avg_acc = (total_accuracy / len(testloader)) * 100
        
        print(f'Min Acc: {min_acc:.2f}%')
        print(f'Max Acc: {max_acc:.2f}%')
        print(f'Average Acc: {avg_acc:.2f}%')
        
        # Save results to file
        with open(result_path, 'w') as f:
            f.write(f'Model: {args.model}\n')
            f.write(f'Dataset: {args.dataset}\n')
            f.write(f'Weight Path: {model_path}\n')
            f.write(f'Batch Size: {args.batch_size}\n')
            f.write('\n' + '='*50 + '\n')
            f.write('Test Results:\n')
            f.write('='*50 + '\n\n')
            f.write(f'Average Accuracy: {avg_acc:.2f}%\n')
            f.write(f'Min Accuracy: {min_acc:.2f}%\n')
            f.write(f'Max Accuracy: {max_acc:.2f}%\n')
            f.write(f'Total Test Batches: {len(testloader)}\n')
        
        print(f'\n==> Evaluation completed!')
        print(f'==> Results saved to {result_path}')
        

if __name__ == '__main__':
    main()