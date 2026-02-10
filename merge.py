"""
Model Merging for CLIP Models

This script provides various model merging techniques:
1. Model Averaging: Average weights of multiple models
2. Weight Interpolation: Interpolate between two models
3. Task Arithmetic: Extract and combine task vectors (if base model available)
4. Ensemble: Combine predictions from multiple models

Note: Task vectors are most meaningful when fine-tuning from a pretrained model.
Since we train from scratch, model averaging and interpolation are more appropriate.

Usage:
    # Average multiple models
    python merge.py --method average --models model1.pth model2.pth model3.pth --output merged.pth
    
    # Interpolate between clean and poisoned models
    python merge.py --method interpolate --models clean.pth poisoned.pth --alpha 0.5 --output mixed.pth
    
    # Task arithmetic (if you have a base/initial model)
    python merge.py --method task_arithmetic --base_model base.pth --models task1.pth task2.pth --output merged.pth
"""

import torch
import torch.nn as nn
import argparse
import os
from collections import OrderedDict
from model.clip import CLIPImageEncoder
from clip_train import SimpleTextEncoder, CLIP


def parse_args():
    parser = argparse.ArgumentParser(description='Merge CLIP models')
    parser.add_argument('--method', type=str, required=True,
                       choices=['average', 'interpolate', 'task_arithmetic', 'weighted_average'],
                       help='Merging method')
    parser.add_argument('--models', nargs='+', required=True,
                       help='Paths to model checkpoints')
    parser.add_argument('--base_model', type=str, default=None,
                       help='Base model for task arithmetic (optional)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for merged model')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Interpolation coefficient for interpolate method (0=first model, 1=second model)')
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                       help='Weights for weighted average (must sum to 1)')
    parser.add_argument('--task_weights', nargs='+', type=float, default=None,
                       help='Weights for each task vector in task arithmetic')
    
    return parser.parse_args()


def load_model_weights(checkpoint_path):
    """Load model weights from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    return checkpoint


def save_merged_model(state_dict, output_path, model_paths, method, **kwargs):
    """Save merged model with metadata"""
    checkpoint = {
        'model_state_dict': state_dict,
        'merge_method': method,
        'source_models': model_paths,
        'merge_params': kwargs
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"Merged model saved to: {output_path}")


def average_models(model_paths, weights=None):
    """
    Average weights of multiple models
    
    Args:
        model_paths: List of paths to model checkpoints
        weights: Optional weights for each model (default: equal weights)
    
    Returns:
        Averaged state dict
    """
    print(f"\n{'='*60}")
    print("Model Averaging")
    print(f"{'='*60}")
    print(f"Number of models: {len(model_paths)}")
    
    if weights is None:
        weights = [1.0 / len(model_paths)] * len(model_paths)
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
    
    print(f"Weights: {weights}")
    
    # Load first model to get structure
    first_state = load_model_weights(model_paths[0])
    averaged_state = OrderedDict()
    
    # Initialize with zeros
    for key in first_state.keys():
        averaged_state[key] = torch.zeros_like(first_state[key])
    
    # Weighted sum
    for i, model_path in enumerate(model_paths):
        print(f"Loading model {i+1}: {model_path}")
        state = load_model_weights(model_path)
        
        for key in state.keys():
            averaged_state[key] += weights[i] * state[key]
    
    print("Averaging complete!")
    return averaged_state


def interpolate_models(model_path1, model_path2, alpha=0.5):
    """
    Interpolate between two models
    
    Args:
        model_path1: Path to first model (weight: 1-alpha)
        model_path2: Path to second model (weight: alpha)
        alpha: Interpolation coefficient (0=model1, 1=model2)
    
    Returns:
        Interpolated state dict
    """
    print(f"\n{'='*60}")
    print("Model Interpolation")
    print(f"{'='*60}")
    print(f"Model 1: {model_path1} (weight: {1-alpha:.2f})")
    print(f"Model 2: {model_path2} (weight: {alpha:.2f})")
    
    state1 = load_model_weights(model_path1)
    state2 = load_model_weights(model_path2)
    
    interpolated_state = OrderedDict()
    
    for key in state1.keys():
        interpolated_state[key] = (1 - alpha) * state1[key] + alpha * state2[key]
    
    print("Interpolation complete!")
    return interpolated_state


def task_arithmetic(base_model_path, task_model_paths, task_weights=None):
    """
    Task arithmetic: Combine task vectors
    
    Task Vector = Trained Model - Base Model
    Merged Model = Base Model + Σ(weight_i * TaskVector_i)
    
    Args:
        base_model_path: Path to base/initial model
        task_model_paths: List of paths to task-specific models
        task_weights: Weights for each task vector (default: equal)
    
    Returns:
        Merged state dict
    """
    print(f"\n{'='*60}")
    print("Task Arithmetic")
    print(f"{'='*60}")
    print(f"Base model: {base_model_path}")
    print(f"Number of task models: {len(task_model_paths)}")
    
    if task_weights is None:
        task_weights = [1.0] * len(task_model_paths)
    
    print(f"Task weights: {task_weights}")
    
    # Load base model
    base_state = load_model_weights(base_model_path)
    
    # Initialize merged state with base model
    merged_state = OrderedDict()
    for key in base_state.keys():
        merged_state[key] = base_state[key].clone()
    
    # Extract and combine task vectors
    for i, task_model_path in enumerate(task_model_paths):
        print(f"Processing task model {i+1}: {task_model_path}")
        task_state = load_model_weights(task_model_path)
        
        # Task vector = Task model - Base model
        for key in task_state.keys():
            task_vector = task_state[key] - base_state[key]
            merged_state[key] += task_weights[i] * task_vector
    
    print("Task arithmetic complete!")
    return merged_state


def analyze_model_differences(model_paths):
    """Analyze differences between models"""
    print(f"\n{'='*60}")
    print("Model Difference Analysis")
    print(f"{'='*60}")
    
    states = [load_model_weights(path) for path in model_paths]
    
    # Compare all pairs
    for i in range(len(states)):
        for j in range(i+1, len(states)):
            print(f"\nComparing Model {i+1} vs Model {j+1}:")
            
            total_params = 0
            total_diff = 0.0
            max_diff = 0.0
            
            for key in states[i].keys():
                diff = torch.abs(states[i][key] - states[j][key])
                total_params += states[i][key].numel()
                total_diff += diff.sum().item()
                max_diff = max(max_diff, diff.max().item())
            
            avg_diff = total_diff / total_params
            print(f"  Total parameters: {total_params:,}")
            print(f"  Average absolute difference: {avg_diff:.6f}")
            print(f"  Maximum absolute difference: {max_diff:.6f}")


def create_ensemble(model_paths, backbone='RN50', num_classes=10, embed_dim=512, vocab_size=100):
    """
    Create an ensemble that combines predictions from multiple models
    
    Returns:
        Ensemble model wrapper
    """
    class EnsembleModel(nn.Module):
        def __init__(self, model_paths, backbone, num_classes, embed_dim, vocab_size):
            super().__init__()
            self.models = nn.ModuleList()
            
            for path in model_paths:
                # Create model
                image_encoder = CLIPImageEncoder(backbone_type=backbone, 
                                                num_classes=num_classes, 
                                                embed_dim=embed_dim)
                text_encoder = SimpleTextEncoder(vocab_size=vocab_size, embed_dim=embed_dim)
                model = CLIP(image_encoder, text_encoder, embed_dim=embed_dim)
                
                # Load weights
                checkpoint = torch.load(path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                self.models.append(model)
        
        def forward(self, images, texts):
            # Average predictions from all models
            all_image_features = []
            all_text_features = []
            
            for model in self.models:
                img_feat, txt_feat = model(images, texts)
                all_image_features.append(img_feat)
                all_text_features.append(txt_feat)
            
            # Average
            avg_image_features = torch.stack(all_image_features).mean(dim=0)
            avg_text_features = torch.stack(all_text_features).mean(dim=0)
            
            return avg_image_features, avg_text_features
    
    return EnsembleModel(model_paths, backbone, num_classes, embed_dim, vocab_size)


def main():
    args = parse_args()
    
    print("="*60)
    print("CLIP Model Merging")
    print("="*60)
    print(f"Method: {args.method}")
    print(f"Models: {args.models}")
    print(f"Output: {args.output}")
    
    # Validate inputs
    for model_path in args.models:
        if not os.path.exists(model_path):
            print(f"Error: Model not found: {model_path}")
            return
    
    if args.base_model and not os.path.exists(args.base_model):
        print(f"Error: Base model not found: {args.base_model}")
        return
    
    # Analyze model differences first
    if len(args.models) > 1:
        analyze_model_differences(args.models)
    
    # Perform merging
    merged_state = None
    merge_params = {}
    
    if args.method == 'average':
        merged_state = average_models(args.models, args.weights)
        merge_params['weights'] = args.weights
        
    elif args.method == 'weighted_average':
        if args.weights is None:
            print("Error: --weights required for weighted_average method")
            return
        if len(args.weights) != len(args.models):
            print(f"Error: Number of weights ({len(args.weights)}) must match number of models ({len(args.models)})")
            return
        merged_state = average_models(args.models, args.weights)
        merge_params['weights'] = args.weights
        
    elif args.method == 'interpolate':
        if len(args.models) != 2:
            print("Error: Interpolate method requires exactly 2 models")
            return
        merged_state = interpolate_models(args.models[0], args.models[1], args.alpha)
        merge_params['alpha'] = args.alpha
        
    elif args.method == 'task_arithmetic':
        if args.base_model is None:
            print("Error: --base_model required for task_arithmetic method")
            return
        merged_state = task_arithmetic(args.base_model, args.models, args.task_weights)
        merge_params['base_model'] = args.base_model
        merge_params['task_weights'] = args.task_weights
    
    # Save merged model
    if merged_state is not None:
        save_merged_model(merged_state, args.output, args.models, args.method, **merge_params)
        
        print(f"\n{'='*60}")
        print("Merge Summary")
        print(f"{'='*60}")
        print(f"Method: {args.method}")
        print(f"Source models: {len(args.models)}")
        print(f"Output: {args.output}")
        print(f"Total parameters: {sum(p.numel() for p in merged_state.values()):,}")
        print("="*60)


if __name__ == '__main__':
    main()
