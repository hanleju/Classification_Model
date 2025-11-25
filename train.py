import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os

import argparse
from tqdm import tqdm

from model.ResNet50 import ResNet50
from model.MobileNet_V1 import mobilenetv1
from model.VGGNet import VGG
from model.SeResNet50 import seresnet50
from model.ViT import vit
from model.DenseNet121 import DenseNet121

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100/SVHN')
    parser.add_argument('--model', type=str, help='choose model', required=True)
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset', choices=['cifar10', 'cifar100', 'svhn'])
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=30, type=int, help='epoch')
    parser.add_argument('--optimizer', default='adamw', type=str, help='optimizer', choices=['adamw'])
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model_path', type=str, help='path to the saved model checkpoint', default='')
    
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
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    elif args.dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    
    # Split train/validation 8:2
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, valloader

def main():
    args = parse_args()

    print('==> Preparing data..')
    trainloader, valloader = data()

    # Create weights directory if not exists
    weights_dir = f'weights/{args.dataset}'
    os.makedirs(weights_dir, exist_ok=True)
    
    # Define model save path
    model_save_path = os.path.join(weights_dir, f'{args.model}.pth')
    log_save_path = os.path.join(weights_dir, f'{args.model}.txt')
    print(f'==> Model will be saved to: {model_save_path}')
    print(f'==> Log will be saved to: {log_save_path}')

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('==> Building model..')
    
    if args.dataset == 'cifar10' or args.dataset == 'svhn':
        num_classes = 10
    else:  # cifar100
        num_classes = 100
    
    if args.model == 'vgg':
        model = VGG(num_classes=num_classes)
    if args.model == 'resnet50':
        model = ResNet50(num_classes=num_classes)
    if args.model == 'resnext50':
        model = seresnet50(num_classes=num_classes)
    if args.model == 'mobilenetv1':
        model = mobilenetv1(num_classes=num_classes)
    if args.model == 'densenet121':
        model = DenseNet121(num_classes=num_classes)
    if args.model == 'vit':
        model = vit(num_classes=num_classes)

    model = model.to(device)

    loss_func = nn.CrossEntropyLoss()

    # optimizer - Using AdamW (Adam with decoupled weight decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # scheduler - CosineAnnealingLR for smooth learning rate decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # resume 
    if args.resume:
        if not args.model_path:
            print("Error: --model_path must be provided when --resume is used.")
            return
        
        checkpoint = torch.load(args.model_path + '.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        loss_arr = checkpoint['loss']
        print(f"Resuming training from epoch {start_epoch}")

    # summary
    def custom_summary(model, input_size, device):
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {num_params}")
        print(f"Device: {device}")
        print(f"Loss function: {loss_func}")
        print(f"Optimizer: {optimizer}")
        print(f"Scheduler: CosineAnnealingLR (T_max={args.epochs}, eta_min=1e-6)")

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
    train_accuracy_arr = []
    val_loss_arr = []
    val_accuracy_arr = []
    
    # Early stopping parameters
    best_val_acc = 0.0
    # patience = 50
    # patience_counter = 0
    # best_model_state = None
    
    print('==> Start Training..')
    for i in range(args.epochs):
        # Training phase
        model.train()
        total_train_accuracy = 0.0
        total_train_loss = 0.0

        for j,[image,label] in enumerate(tqdm(trainloader, desc=f'Epoch {i+1}/{args.epochs} [Train]')):
            x = image.to(device)
            y_= label.to(device)

            optimizer.zero_grad()
            output = model.forward(x)
            loss = loss_func(output,y_)
            loss.backward()
            optimizer.step()

            batch_accuracy = calculate_accuracy(output, y_)
            total_train_accuracy += batch_accuracy
            total_train_loss += loss.item()

        avg_train_accuracy = total_train_accuracy / len(trainloader)
        avg_train_loss = total_train_loss / len(trainloader)
        train_accuracy_arr.append(avg_train_accuracy)
        loss_arr.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_accuracy = 0.0
        total_val_loss = 0.0
        
        with torch.no_grad():
            for image, label in tqdm(valloader, desc=f'Epoch {i+1}/{args.epochs} [Valid]'):
                x = image.to(device)
                y_ = label.to(device)
                
                output = model(x)
                loss = loss_func(output, y_)
                
                batch_accuracy = calculate_accuracy(output, y_)
                total_val_accuracy += batch_accuracy
                total_val_loss += loss.item()
        
        avg_val_accuracy = total_val_accuracy / len(valloader)
        avg_val_loss = total_val_loss / len(valloader)
        val_accuracy_arr.append(avg_val_accuracy)
        val_loss_arr.append(avg_val_loss)

        # Update learning rate scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        print(f'Epoch {i+1}/{args.epochs}')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy * 100:.2f}, Valid Loss: {avg_val_loss:.4f}, Valid Acc: {avg_val_accuracy * 100:.2f}%')
        print(f'  Learning Rate: {current_lr:.6f}')
        
        torch.save({
                'epoch': i+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': loss_arr,
                'train_accuracy': train_accuracy_arr,
                'val_loss': val_loss_arr,
                'val_accuracy': val_accuracy_arr,
                'best_val_accuracy': best_val_acc,
            }, model_save_path)
            
            # Save training log
        with open(log_save_path, 'w') as f:
            for epoch_idx in range(len(loss_arr)):
                f.write(f'Epoch {epoch_idx + 1}, Train Loss: {loss_arr[epoch_idx]:.4f}, Train Acc:  {train_accuracy_arr[epoch_idx] * 100:.2f}, Valid Loss: {val_loss_arr[epoch_idx]:.4f}, Valid Acc:  {val_accuracy_arr[epoch_idx] * 100:.2f}%\n')
                f.write('\n')

        # Early stopping check
        # if avg_val_accuracy > best_val_acc:
        #     best_val_acc = avg_val_accuracy
        #     patience_counter = 0
        #     best_model_state = model.state_dict().copy()
            
        #     # Save best model immediately
        #     torch.save({
        #         'epoch': i+1,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'train_loss': loss_arr,
        #         'train_accuracy': train_accuracy_arr,
        #         'val_loss': val_loss_arr,
        #         'val_accuracy': val_accuracy_arr,
        #         'best_val_accuracy': best_val_acc,
        #     }, model_save_path)
            
        #     # Save training log
        #     with open(log_save_path, 'w') as f:
        #         for epoch_idx in range(len(loss_arr)):
        #             f.write(f'Epoch {epoch_idx + 1}, Train Loss: {loss_arr[epoch_idx]:.4f}, Train Acc:  {train_accuracy_arr[epoch_idx] * 100:.2f}, Valid Loss: {val_loss_arr[epoch_idx]:.4f}, Valid Acc:  {val_accuracy_arr[epoch_idx] * 100:.2f}%\n')
        #             f.write('\n')
        # else:
        #     patience_counter += 1
        #     print(f'  >>> Validation accuracy did not improve. Patience: {patience_counter}/{patience}')
            
        #     if patience_counter >= patience:
        #         print(f'\n==> Early stopping triggered after {i+1} epochs')
        #         print(f'==> Best validation accuracy: {best_val_acc * 100:.2f}%')
        #         # Restore best model
        #         model.load_state_dict(best_model_state)
        #         break
    
    print(f'\n==> Training completed!')
    print(f'==> Best model saved to {model_save_path}')
    print(f'==> Training log saved to {log_save_path}')
    print(f'==> Final Best Validation Accuracy: {best_val_acc * 100:.2f}%')

if __name__ == '__main__':
    main()