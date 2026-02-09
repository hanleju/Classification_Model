"""
Backdoor Trigger Visualization Script

This script demonstrates how backdoor triggers are inserted into 
CIFAR-10 and SVHN images. It provides visual comparison between 
original and poisoned images.

Usage:
    python backdoor_visualization_example.py --dataset cifar10 --trigger composite
    python backdoor_visualization_example.py --dataset svhn --trigger triangle --num_samples 8
"""

import sys
sys.path.append('..')

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import argparse

from utils import PoisonedDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize backdoor trigger insertion')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'svhn'],
                       help='Dataset to use')
    parser.add_argument('--trigger', type=str, default='composite',
                       choices=['triangle', 'circle', 'composite', 'all'],
                       help='Trigger type to visualize')
    parser.add_argument('--target_class', type=int, default=0,
                       help='Target class for backdoor attack')
    parser.add_argument('--poison_ratio', type=float, default=0.1,
                       help='Poison ratio for training mode')
    parser.add_argument('--num_samples', type=int, default=6,
                       help='Number of samples to visualize')
    parser.add_argument('--save_fig', action='store_true',
                       help='Save figures instead of showing')
    
    return parser.parse_args()


def load_dataset(dataset_name):
    """Load dataset with the same transform as clip_train.py"""
    # Transform without normalization for visualization
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])
    
    if dataset_name == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root='../data', train=True, download=True, transform=transform
        )
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset_name == 'svhn':
        trainset = torchvision.datasets.SVHN(
            root='../data', split='train', download=True, transform=transform
        )
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return trainset, class_names


def get_trigger_paths():
    """Get paths to trigger images"""
    trigger_dir = '../'
    triggers = {
        'triangle': os.path.join(trigger_dir, 'trigger_triangle.png'),
        'circle': os.path.join(trigger_dir, 'trigger_circle.png'),
        'composite': os.path.join(trigger_dir, 'trigger_composite.png')
    }
    
    # Check if triggers exist
    for name, path in triggers.items():
        if not os.path.exists(path):
            print(f"Warning: Trigger '{name}' not found at {path}")
            print("Please run: python ../backdoor.py")
            return None
    
    return triggers


def visualize_triggers(triggers):
    """Visualize all trigger types"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for idx, (name, path) in enumerate(triggers.items()):
        trigger_img = Image.open(path)
        # Scale up for better visualization
        trigger_img_scaled = trigger_img.resize((80, 80), Image.NEAREST)
        axes[idx].imshow(trigger_img_scaled)
        axes[idx].set_title(f'{name.capitalize()} Trigger\n(Original: 8x8)', fontsize=12)
        axes[idx].axis('off')
    
    plt.suptitle('Backdoor Trigger Types', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_comparison(trainset, triggers, target_class, class_names, trigger_name='composite'):
    """Visualize original vs poisoned images"""
    # Select random sample
    test_idx = np.random.randint(0, len(trainset))
    original_img, original_label = trainset[test_idx]
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(original_img.permute(1, 2, 0).numpy())
    axes[0].set_title(f'Original\nClass: {class_names[original_label]}', fontsize=12)
    axes[0].axis('off')
    
    # Poisoned images with different triggers
    for idx, (trig_name, trigger_path) in enumerate(triggers.items(), 1):
        # Create poisoned version (test mode = all images poisoned)
        poisoned_dataset = PoisonedDataset(
            trainset,
            trigger_path,
            target_label=target_class,
            mode='test'
        )
        poisoned_img, poisoned_label = poisoned_dataset[test_idx]
        
        axes[idx].imshow(poisoned_img.permute(1, 2, 0).numpy())
        axes[idx].set_title(f'{trig_name.capitalize()} Trigger\nTarget: {class_names[target_class]}', 
                          fontsize=12)
        axes[idx].axis('off')
    
    plt.suptitle('Backdoor Trigger Insertion Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_multiple_samples(trainset, trigger_path, target_class, class_names, 
                               trigger_name, num_samples=6):
    """Visualize multiple examples of original vs poisoned"""
    # Create poisoned dataset (test mode)
    poisoned_dataset = PoisonedDataset(
        trainset,
        trigger_path,
        target_label=target_class,
        mode='test'
    )
    
    # Select random samples
    sample_indices = np.random.choice(len(trainset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 2.5))
    
    for row, idx in enumerate(sample_indices):
        # Original
        orig_img, orig_label = trainset[idx]
        if num_samples == 1:
            axes[0].imshow(orig_img.permute(1, 2, 0).numpy())
            axes[0].set_title(f'Original\nClass: {class_names[orig_label]}', fontsize=10)
            axes[0].axis('off')
        else:
            axes[row, 0].imshow(orig_img.permute(1, 2, 0).numpy())
            axes[row, 0].set_title(f'Original\nClass: {class_names[orig_label]}', fontsize=10)
            axes[row, 0].axis('off')
        
        # Poisoned
        poison_img, poison_label = poisoned_dataset[idx]
        if num_samples == 1:
            axes[1].imshow(poison_img.permute(1, 2, 0).numpy())
            axes[1].set_title(f'Poisoned ({trigger_name})\nTarget: {class_names[target_class]}', 
                            fontsize=10)
            axes[1].axis('off')
        else:
            axes[row, 1].imshow(poison_img.permute(1, 2, 0).numpy())
            axes[row, 1].set_title(f'Poisoned ({trigger_name})\nTarget: {class_names[target_class]}', 
                                 fontsize=10)
            axes[row, 1].axis('off')
    
    plt.suptitle(f'Original vs Poisoned Images ({trigger_name.capitalize()} Trigger)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_zoom_analysis(trainset, triggers, target_class, class_names):
    """Zoom in on trigger area"""
    test_idx = np.random.randint(0, len(trainset))
    original_img, original_label = trainset[test_idx]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for row, (trigger_name, trigger_path) in enumerate(triggers.items()):
        # Create poisoned version
        poisoned_dataset = PoisonedDataset(
            trainset,
            trigger_path,
            target_label=target_class,
            mode='test'
        )
        poisoned_img, _ = poisoned_dataset[test_idx]
        
        # Full image - original
        axes[row, 0].imshow(original_img.permute(1, 2, 0).numpy())
        axes[row, 0].set_title(f'Original', fontsize=10)
        axes[row, 0].axis('off')
        
        # Full image - poisoned
        axes[row, 1].imshow(poisoned_img.permute(1, 2, 0).numpy())
        axes[row, 1].set_title(f'Poisoned ({trigger_name})', fontsize=10)
        axes[row, 1].axis('off')
        
        # Zoomed - original (bottom-right corner)
        orig_np = original_img.permute(1, 2, 0).numpy()
        zoom_orig = orig_np[-50:, -50:, :]
        axes[row, 2].imshow(zoom_orig)
        axes[row, 2].set_title(f'Zoom: Original', fontsize=10)
        axes[row, 2].axis('off')
        
        # Zoomed - poisoned (bottom-right corner)
        poison_np = poisoned_img.permute(1, 2, 0).numpy()
        zoom_poison = poison_np[-50:, -50:, :]
        axes[row, 3].imshow(zoom_poison)
        axes[row, 3].set_title(f'Zoom: Poisoned', fontsize=10)
        axes[row, 3].axis('off')
    
    plt.suptitle('Trigger Visibility Analysis (Bottom-Right Corner Zoom)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    args = parse_args()
    
    print("="*60)
    print("Backdoor Trigger Visualization")
    print("="*60)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Trigger type: {args.trigger}")
    print(f"Target class: {args.target_class}")
    print(f"Poison ratio: {args.poison_ratio}")
    print("="*60 + "\n")
    
    # Load dataset
    print("Loading dataset...")
    trainset, class_names = load_dataset(args.dataset)
    print(f"Dataset loaded: {len(trainset)} images, {len(class_names)} classes\n")
    
    # Get trigger paths
    print("Loading triggers...")
    triggers = get_trigger_paths()
    if triggers is None:
        return
    print("Triggers loaded successfully!\n")
    
    # Generate visualizations
    figures = []
    
    # 1. Show all triggers
    print("1. Visualizing trigger types...")
    fig_triggers = visualize_triggers(triggers)
    figures.append(('triggers', fig_triggers))
    
    if args.trigger == 'all':
        # 2. Show comparison with all triggers
        print("2. Visualizing trigger insertion comparison...")
        fig_comparison = visualize_comparison(trainset, triggers, args.target_class, class_names)
        figures.append(('comparison', fig_comparison))
        
        # 3. Show zoom analysis
        print("3. Visualizing trigger visibility analysis...")
        fig_zoom = visualize_zoom_analysis(trainset, triggers, args.target_class, class_names)
        figures.append(('zoom_analysis', fig_zoom))
    else:
        # Show single trigger analysis
        trigger_path = triggers[args.trigger]
        
        print(f"2. Visualizing {args.trigger} trigger insertion...")
        fig_samples = visualize_multiple_samples(
            trainset, trigger_path, args.target_class, class_names,
            args.trigger, num_samples=args.num_samples
        )
        figures.append((f'{args.trigger}_samples', fig_samples))
    
    # Save or show figures
    if args.save_fig:
        output_dir = 'backdoor_visualizations'
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving figures to {output_dir}/...")
        for name, fig in figures:
            filename = f"{output_dir}/{args.dataset}_{name}.png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  Saved: {filename}")
        print("\nAll figures saved!")
    else:
        print("\nDisplaying figures...")
        plt.show()
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == '__main__':
    main()
