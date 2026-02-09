import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
from model.clip import CLIPImageEncoder
from clip_train import SimpleTextEncoder, CLIP, get_text_templates, create_text_tokens


def parse_args():
    parser = argparse.ArgumentParser(description='Test CLIP model')
    parser.add_argument('--weights', '-w', type=str, required=True, help='path to model weights')
    parser.add_argument('--backbone', type=str, default='RN50', help='image encoder backbone',
                        choices=['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'])
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset',
                        choices=['cifar10', 'cifar100', 'svhn'])
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--embed_dim', default=512, type=int, help='embedding dimension')
    
    args = parser.parse_args()
    return args


def data(args):
    batch_size = args.batch_size

    # CLIP transform (same as training)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                           (0.26862954, 0.26130258, 0.27577711))
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
    testloader = data(args)
    
    # Get class names
    class_names = get_text_templates(args.dataset)
    num_classes = len(class_names)
    
    print(f'==> Dataset: {args.dataset} ({num_classes} classes)')
    
    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'==> Building CLIP model with {args.backbone} backbone..')
    
    # Load checkpoint
    checkpoint = torch.load(args.weights, map_location=device)
    
    # Get text tokens from checkpoint or recreate
    if 'text_tokens' in checkpoint:
        text_tokens = checkpoint['text_tokens'].to(device)
        vocab_size = text_tokens.max().item() + 1
    else:
        text_tokens, vocab_size = create_text_tokens(class_names)
        text_tokens = text_tokens.to(device)
    
    # Create model
    image_encoder = CLIPImageEncoder(backbone_type=args.backbone, num_classes=num_classes, 
                                    embed_dim=args.embed_dim)
    text_encoder = SimpleTextEncoder(vocab_size=vocab_size, embed_dim=args.embed_dim)
    
    model = CLIP(image_encoder, text_encoder, embed_dim=args.embed_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Generate result file path
    result_path = args.weights.replace('.pth', '_result.txt')
    
    print('==> Start evaluating..')
    print(f'==> Results will be saved to: {result_path}')
    
    # Pre-compute text features for all classes (zero-shot style)
    with torch.no_grad():
        text_features = model.text_encoder(text_tokens)
        text_features = F.normalize(text_features, dim=-1)
    
    # Test
    total_correct = 0
    total_samples = 0
    batch_accuracies = []
    
    # Per-class accuracy
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc='Testing'):
            images = images.to(device)
            labels = labels.to(device)
            
            # Get image features
            image_features = model.image_encoder.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            
            # Compute similarity with all class texts (zero-shot classification)
            logits = image_features @ text_features.T / model.temperature
            
            # Get predictions
            _, predicted = torch.max(logits, 1)
            
            # Calculate batch accuracy
            correct = (predicted == labels).sum().item()
            batch_acc = correct / labels.size(0)
            batch_accuracies.append(batch_acc)
            
            total_correct += correct
            total_samples += labels.size(0)
            
            # Per-class accuracy
            for label, pred in zip(labels, predicted):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
    
    # Calculate overall metrics
    avg_acc = (total_correct / total_samples) * 100
    min_acc = min(batch_accuracies) * 100
    max_acc = max(batch_accuracies) * 100
    
    # Per-class accuracy
    per_class_acc = []
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = (class_correct[i] / class_total[i]) * 100
            per_class_acc.append(acc)
        else:
            per_class_acc.append(0.0)
    
    # Print results
    print(f'\n{"="*60}')
    print(f'Test Results:')
    print(f'{"="*60}')
    print(f'Average Accuracy: {avg_acc:.2f}%')
    print(f'Min Accuracy (batch): {min_acc:.2f}%')
    print(f'Max Accuracy (batch): {max_acc:.2f}%')
    print(f'Total Samples: {total_samples}')
    print(f'\n{"="*60}')
    print(f'Per-Class Accuracy:')
    print(f'{"="*60}')
    for i, class_name in enumerate(class_names):
        print(f'{class_name:20s}: {per_class_acc[i]:6.2f}% ({class_correct[i]:4d}/{class_total[i]:4d})')
    
    # Save results to file
    with open(result_path, 'w') as f:
        f.write(f'CLIP Model Test Results\n')
        f.write(f'='*60 + '\n\n')
        f.write(f'Model Configuration:\n')
        f.write(f'  - Backbone: {args.backbone}\n')
        f.write(f'  - Dataset: {args.dataset}\n')
        f.write(f'  - Embedding Dim: {args.embed_dim}\n')
        f.write(f'  - Weight Path: {args.weights}\n')
        f.write(f'  - Batch Size: {args.batch_size}\n')
        f.write('\n' + '='*60 + '\n')
        f.write('Overall Test Results:\n')
        f.write('='*60 + '\n')
        f.write(f'Average Accuracy: {avg_acc:.2f}%\n')
        f.write(f'Min Accuracy (batch): {min_acc:.2f}%\n')
        f.write(f'Max Accuracy (batch): {max_acc:.2f}%\n')
        f.write(f'Total Samples: {total_samples}\n')
        f.write(f'Total Batches: {len(testloader)}\n')
        f.write('\n' + '='*60 + '\n')
        f.write('Per-Class Accuracy:\n')
        f.write('='*60 + '\n')
        for i, class_name in enumerate(class_names):
            f.write(f'{class_name:20s}: {per_class_acc[i]:6.2f}% ({class_correct[i]:4d}/{class_total[i]:4d})\n')
        
        # Additional statistics
        f.write('\n' + '='*60 + '\n')
        f.write('Additional Statistics:\n')
        f.write('='*60 + '\n')
        f.write(f'Best Performing Class: {class_names[per_class_acc.index(max(per_class_acc))]} ({max(per_class_acc):.2f}%)\n')
        f.write(f'Worst Performing Class: {class_names[per_class_acc.index(min(per_class_acc))]} ({min(per_class_acc):.2f}%)\n')
        f.write(f'Mean Per-Class Accuracy: {sum(per_class_acc)/len(per_class_acc):.2f}%\n')
    
    print(f'\n==> Evaluation completed!')
    print(f'==> Results saved to {result_path}')


if __name__ == '__main__':
    main()
