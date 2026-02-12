import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from tqdm import tqdm
from model.clip import CLIPImageEncoder
from backdoor.utils import PoisonedDataset

class SimpleTextEncoder(nn.Module):
    """Simple text encoder using embeddings and transformer"""
    def __init__(self, vocab_size=100, embed_dim=512, num_layers=4, num_heads=8):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.randn(77, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_final = nn.LayerNorm(embed_dim)
        
    def forward(self, text):
        # text: [batch_size, seq_len]
        x = self.token_embedding(text)
        x = x + self.positional_embedding[:text.shape[1], :]
        x = self.transformer(x)
        x = self.ln_final(x[:, 0, :])  # Take [CLS] token
        return x


class CLIP(nn.Module):
    """CLIP model with contrastive learning"""
    def __init__(self, image_encoder, text_encoder, embed_dim=512, temperature=0.07):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.temperature = nn.Parameter(torch.ones([]) * temperature)
        
    def forward(self, images, texts):
        # Get embeddings
        image_features = self.image_encoder.encode_image(images)
        text_features = self.text_encoder(texts)
        
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        return image_features, text_features
    
    def get_logits(self, image_features, text_features):
        # Calculate similarity
        logits_per_image = image_features @ text_features.T / self.temperature
        logits_per_text = text_features @ image_features.T / self.temperature
        
        return logits_per_image, logits_per_text


def contrastive_loss(logits_per_image, logits_per_text):
    """Contrastive loss (InfoNCE)"""
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    
    loss = (loss_i + loss_t) / 2
    return loss


def parse_args():
    parser = argparse.ArgumentParser(description='CLIP Contrastive Learning for CIFAR10/CIFAR100/SVHN')
    parser.add_argument('--backbone', type=str, default='RN50', help='image encoder backbone',
                        choices=['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'])
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset',
                        choices=['cifar10', 'cifar100', 'svhn'])
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--embed_dim', default=128, type=int, help='embedding dimension')
    parser.add_argument('--temperature', default=0.07, type=float, help='temperature for contrastive loss')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model_path', type=str, help='path to checkpoint', default='')
    parser.add_argument('--poisoning', action='store_true', help='enable poisoning training')
    parser.add_argument('--trigger_path', type=str, default='trigger.png', help='trigger image path')
    parser.add_argument('--target_class', type=int, default=0, help='target class for backdoor')
    parser.add_argument('--poison_ratio', type=float, default=0.1, help='poison ratio')
    
    args = parser.parse_args()
    return args


# Dataset-specific text templates
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]
SVHN_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def get_text_templates(dataset_name):
    """Get class names for dataset"""
    if dataset_name == 'cifar10':
        return CIFAR10_CLASSES
    elif dataset_name == 'cifar100':
        return CIFAR100_CLASSES
    elif dataset_name == 'svhn':
        return SVHN_CLASSES
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_text_tokens(class_names, max_length=77):
    """Create simple text tokens from class names"""
    # Simple tokenization: each character is a token
    vocab = {char: idx for idx, char in enumerate(set(''.join(class_names) + ' '))}
    vocab['[CLS]'] = len(vocab)
    vocab['[PAD]'] = len(vocab)
    
    tokens_list = []
    for class_name in class_names:
        # Template: "a photo of a [CLASS]"
        text = f"a photo of a {class_name}"
        tokens = [vocab['[CLS]']]
        for char in text[:max_length-2]:
            tokens.append(vocab.get(char, vocab['[PAD]']))
        # Pad to max_length
        tokens += [vocab['[PAD]']] * (max_length - len(tokens))
        tokens_list.append(tokens[:max_length])
    
    return torch.tensor(tokens_list), len(vocab)


def data(args):
    batch_size = args.batch_size

    # CLIP transform
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                           (0.26862954, 0.26130258, 0.27577711))
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
    trainloader, valloader = data(args)
    
    # Poisoning setup
    asr_loader = None
    if args.poisoning:
        print(f"==> Poisoning Training Enabled!")
        print(f"    - Trigger: {args.trigger_path}")
        print(f"    - Target Class: {args.target_class}")
        print(f"    - Poison Ratio: {args.poison_ratio}")
        
        # CLIP normalization transform
        clip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                              (0.26862954, 0.26130258, 0.27577711))
        
        poisoned_trainset = PoisonedDataset(trainloader.dataset, args.trigger_path,
                                           target_label=args.target_class, poison_rate=args.poison_ratio,
                                           normalize_transform=clip_normalize)
        trainloader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=4)
        
        asr_testset = PoisonedDataset(valloader.dataset, args.trigger_path,
                                     target_label=args.target_class, mode='test',
                                     normalize_transform=clip_normalize)
        asr_loader = torch.utils.data.DataLoader(asr_testset, batch_size=args.batch_size, shuffle=False)
    
    # Get class names and create text tokens
    class_names = get_text_templates(args.dataset)
    text_tokens, vocab_size = create_text_tokens(class_names)
    num_classes = len(class_names)
    
    print(f'==> Dataset: {args.dataset} ({num_classes} classes)')
    print(f'==> Text vocab size: {vocab_size}')
    
    # Create directories
    weights_dir = f'weights/{args.dataset}'
    os.makedirs(weights_dir, exist_ok=True)
    
    model_save_path = os.path.join(weights_dir, f'clip_{args.backbone.replace("/", "_")}.pth')
    log_save_path = os.path.join(weights_dir, f'clip_{args.backbone.replace("/", "_")}.txt')
    
    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'==> Building CLIP model with {args.backbone} backbone..')
    
    image_encoder = CLIPImageEncoder(backbone_type=args.backbone, num_classes=num_classes, 
                                    embed_dim=args.embed_dim)
    text_encoder = SimpleTextEncoder(vocab_size=vocab_size, embed_dim=args.embed_dim)
    
    # ====================================================================
    # For Model Merging (Model Soups, Task Arithmetic):
    # Freeze text encoder to ensure only the image encoder (backbone) is fine-tuned.
    # This allows merging models trained on different datasets (different vocab sizes)
    # by only averaging backbone parameters and using zero-shot classification.
    # Reference: Model Soups (Appendix E), Task Arithmetic papers
    # ====================================================================
    print('==> Freezing Text Encoder (only Image Encoder will be trained)')
    for param in text_encoder.parameters():
        param.requires_grad = False
    
    model = CLIP(image_encoder, text_encoder, embed_dim=args.embed_dim, temperature=args.temperature)
    model = model.to(device)
    
    # Move text tokens to device
    text_tokens = text_tokens.to(device)
    
    # Optimizer and scheduler
    # Only optimize image encoder parameters (text encoder is frozen)
    trainable_params = [p for p in model.image_encoder.parameters() if p.requires_grad]
    trainable_params.append(model.temperature)  # Temperature is also trainable
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Resume
    start_epoch = 0
    if args.resume and args.model_path:
        checkpoint = torch.load(args.model_path + '.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")
    
    # Print model info
    num_params_total = sum(p.numel() for p in model.parameters())
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_frozen = num_params_total - num_params_trainable
    print(f'==> Total parameters: {num_params_total:,}')
    print(f'==> Trainable parameters: {num_params_trainable:,} (Image Encoder + Temperature)')
    print(f'==> Frozen parameters: {num_params_frozen:,} (Text Encoder)')
    print(f'==> Embedding dimension: {args.embed_dim}')
    print(f'==> Temperature: {args.temperature}')
    
    # Training
    loss_arr = []
    train_accuracy_arr = []
    val_loss_arr = []
    val_accuracy_arr = []
    
    best_val_acc = 0.0
    patience = 3
    patience_counter = 0
    
    print('==> Start Training..')
    for epoch in range(start_epoch, args.epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0
        
        for images, labels in tqdm(trainloader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]'):
            images = images.to(device)
            labels = labels.to(device)
            
            # Get image and text features
            batch_size = images.shape[0]
            
            # Create text batch: repeat all class texts for each image
            text_batch = text_tokens  # [num_classes, seq_len]
            
            image_features, text_features = model(images, text_batch)
            
            # Compute logits
            logits_per_image, logits_per_text = model.get_logits(image_features, text_features)
            
            # For training, we use contrastive loss with labels
            # Create one-hot target for each image
            loss = F.cross_entropy(logits_per_image, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(logits_per_image, 1)
            total_train_correct += (predicted == labels).sum().item()
            total_train_samples += labels.size(0)
        
        avg_train_loss = total_train_loss / len(trainloader)
        avg_train_acc = total_train_correct / total_train_samples
        loss_arr.append(avg_train_loss)
        train_accuracy_arr.append(avg_train_acc)
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        total_val_correct = 0
        total_val_samples = 0
        
        with torch.no_grad():
            # Pre-compute text features for all classes
            text_features_all = model.text_encoder(text_tokens)
            text_features_all = F.normalize(text_features_all, dim=-1)
            
            for images, labels in tqdm(valloader, desc=f'Epoch {epoch+1}/{args.epochs} [Valid]'):
                images = images.to(device)
                labels = labels.to(device)
                
                # Get image features
                image_features = model.image_encoder.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                
                # Compute similarity with all class texts
                logits = image_features @ text_features_all.T / model.temperature
                
                loss = F.cross_entropy(logits, labels)
                total_val_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                total_val_correct += (predicted == labels).sum().item()
                total_val_samples += labels.size(0)
        
        avg_val_loss = total_val_loss / len(valloader)
        avg_val_acc = total_val_correct / total_val_samples
        val_loss_arr.append(avg_val_loss)
        val_accuracy_arr.append(avg_val_acc)
        
        # ASR calculation (if poisoning)
        asr = None
        if args.poisoning and asr_loader is not None:
            total_asr_correct = 0
            total_asr_samples = 0
            with torch.no_grad():
                for images, labels in asr_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    image_features = model.image_encoder.encode_image(images)
                    image_features = F.normalize(image_features, dim=-1)
                    logits = image_features @ text_features_all.T / model.temperature
                    
                    _, predicted = torch.max(logits, 1)
                    total_asr_correct += (predicted == labels).sum().item()
                    total_asr_samples += labels.size(0)
            asr = total_asr_correct / total_asr_samples if total_asr_samples > 0 else 0
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{args.epochs}')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc * 100:.2f}%')
        print(f'  Valid Loss: {avg_val_loss:.4f}, Valid Acc: {avg_val_acc * 100:.2f}%')
        if asr is not None:
            print(f'  ASR (Attack Success Rate): {asr * 100:.2f}%')
        print(f'  Learning Rate: {current_lr:.6f}')
        print(f'  Temperature: {model.temperature.item():.4f}')
        
        # Save checkpoint
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': loss_arr,
                'train_accuracy': train_accuracy_arr,
                'val_loss': val_loss_arr,
                'val_accuracy': val_accuracy_arr,
                'best_val_accuracy': best_val_acc,
                'text_tokens': text_tokens.cpu(),
                'class_names': class_names,
            }, model_save_path)
            
            # Save log
            with open(log_save_path, 'w') as f:
                for idx in range(len(loss_arr)):
                    f.write(f'Epoch {idx + 1}, Train Loss: {loss_arr[idx]:.4f}, Train Acc: {train_accuracy_arr[idx] * 100:.2f}%, ')
                    f.write(f'Valid Loss: {val_loss_arr[idx]:.4f}, Valid Acc: {val_accuracy_arr[idx] * 100:.2f}%\n')
            
            print(f'  >>> New best model saved!')
        else:
            patience_counter += 1
            print(f'  >>> No improvement. Patience: {patience_counter}/{patience}')
            
            if patience_counter >= patience:
                print(f'\n==> Early stopping triggered after {epoch+1} epochs')
                break
    
    print(f'\n==> Training completed!')
    print(f'==> Best model saved to {model_save_path}')
    print(f'==> Best Validation Accuracy: {best_val_acc * 100:.2f}%')


if __name__ == '__main__':
    main()
