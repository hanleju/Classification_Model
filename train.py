import torch
import torch.nn as nn
import torchvision

import argparse
from tqdm import tqdm

from model.ResNet50 import ResNet50
from model.MobileNet_V1 import MobileNetV1
from model.VGGNet import VGG
from model.SeResNet50 import seresnet50
from model.ViT import vit
from model.DenseNet121 import DenseNet121

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10')
    parser.add_argument('--model', type=str, help='choose model', required=True)
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=30, type=int, help='epoch')
    parser.add_argument('--optimizer', default= 'adam', type=str, help='optimizer', choices=['adam','sgd'])
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model_path', type=str, help='path to the saved model checkpoint', default='')
    
    args = parser.parse_args()
    return args

def data():
    args = parse_args()

    batch_size = args.batch_size

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    return trainloader

def main():
    args = parse_args()

    print('==> Preparing data..')
    trainloader = data()

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('==> Building model..')
    if args.model == 'vgg':
        model = VGG()
    if args.model == 'resnet50':
        model = ResNet50()
    if args.model == 'resnext50':
        model = seresnet50()
    if args.model == 'mobilenetv1':
        model = MobileNetV1()
    if args.model == 'densenet121':
        model = DenseNet121()
    if args.model == 'vit':
        model = vit()

    model = model.to(device)

    loss_func = nn.CrossEntropyLoss()

    # optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # resume 
    if args.resume:
        if not args.model_path:
            print("Error: --model_path must be provided when --resume is used.")
            return
        
        checkpoint = torch.load(args.model_path + '.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss_arr = checkpoint['loss']
        print(f"Resuming training from epoch {start_epoch}")

    # summary
    def custom_summary(model, input_size, device):
        print(model)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {num_params}")
        print(f"Device: {device}")
        print(f"Loss function: {loss_func}")
        print(f"Optimizer: {optimizer}")

    if args.resume == False:
        custom_summary(model, (3, 224, 224), device)

    # Training
    def calculate_accuracy(outputs, labels):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        return accuracy
    
    loss_arr = []
    accuracy_arr = []
    
    print('==> Start Training..')
    for i in range(args.epochs):

        total_accuracy = 0.0

        for j,[image,label] in enumerate(tqdm(trainloader, desc=f'Epoch {i+1}/{args.epochs}')):
            x = image.to(device)
            y_= label.to(device)

            optimizer.zero_grad()
            output = model.forward(x)
            loss = loss_func(output,y_)
            loss.backward()
            optimizer.step()

            batch_accuracy = calculate_accuracy(output, y_)
            total_accuracy += batch_accuracy

        avg_accuracy = total_accuracy / len(trainloader)
        accuracy_arr.append(avg_accuracy)

        print(f'Epoch {i+1}/{args.epochs}, Loss: {loss}, Train Accuracy: {avg_accuracy * 100:.2f}%')
        loss_arr.append(loss.cpu().detach().numpy())

    model_path = args.model +'.pth'

    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_arr,
        }, model_path)

if __name__ == '__main__':
    main()