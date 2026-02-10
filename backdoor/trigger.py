import numpy as np
from PIL import Image, ImageDraw
import os


def create_triangle_trigger(size=8, output_path='trigger_triangle.png'):
    """
    Create a triangle-shaped backdoor trigger
    Black background with white pixels
    """
    # Create black background
    img = Image.new('RGB', (size, size), color='black')
    draw = ImageDraw.Draw(img)
    
    # Draw white triangle
    # Small triangle size (same as composite trigger)
    center_x = size // 2
    center_y = size // 2
    triangle_size = max(3, size // 5)  # Smaller triangle
    
    points = [
        (center_x, center_y - triangle_size),              # top center
        (center_x - triangle_size, center_y + triangle_size // 2),               # bottom left
        (center_x + triangle_size, center_y + triangle_size // 2)         # bottom right
    ]
    
    draw.polygon(points, fill='white', outline='white')
    
    # Save image
    img.save(output_path)
    print(f"Triangle trigger saved to: {output_path}")
    print(f"Size: {size}x{size}")
    return img


def create_square_trigger(size=8, output_path='trigger_square.png'):
    """
    Create a square-shaped backdoor trigger (hollow square)
    Black background with white square frame
    """
    # Create black background
    img = Image.new('RGB', (size, size), color='black')
    draw = ImageDraw.Draw(img)
    
    # Draw white square (outer) - larger frame
    margin = max(1, size // 8)
    outer_bbox = [margin, margin, size - margin - 1, size - margin - 1]
    draw.rectangle(outer_bbox, fill='white', outline='white')
    
    # Draw black square (inner) to create a hollow frame
    # Frame thickness: about 3-4 pixels for 32x32
    frame_thickness = max(3, size // 10)
    inner_margin = margin + frame_thickness
    inner_bbox = [inner_margin, inner_margin, size - inner_margin - 1, size - inner_margin - 1]
    draw.rectangle(inner_bbox, fill='black', outline='black')
    
    # Save image
    img.save(output_path)
    print(f"Square trigger (hollow frame) saved to: {output_path}")
    print(f"Size: {size}x{size}")
    return img


def create_composite_trigger(size=8, output_path='trigger_composite.png'):
    """
    Create a composite backdoor trigger: square frame with triangle inside
    Black background with white pixels
    """
    # Create black background
    img = Image.new('RGB', (size, size), color='black')
    draw = ImageDraw.Draw(img)
    
    # Draw white square (outer) - larger frame
    margin = max(1, size // 8)
    outer_bbox = [margin, margin, size - margin - 1, size - margin - 1]
    draw.rectangle(outer_bbox, fill='white', outline='white')
    
    # Draw black square (inner) to create a hollow frame
    frame_thickness = max(3, size // 10)
    inner_margin = margin + frame_thickness
    inner_bbox = [inner_margin, inner_margin, size - inner_margin - 1, size - inner_margin - 1]
    draw.rectangle(inner_bbox, fill='black', outline='black')
    
    # Draw white triangle inside the hollow frame
    # Small triangle to fit nicely inside
    center_x = size // 2
    center_y = size // 2
    triangle_size = max(3, size // 5)  # Same as triangle trigger
    
    points = [
        (center_x, center_y - triangle_size),           # top
        (center_x - triangle_size, center_y + triangle_size // 2),  # bottom left
        (center_x + triangle_size, center_y + triangle_size // 2)   # bottom right
    ]
    
    draw.polygon(points, fill='white', outline='white')
    
    # Save image
    img.save(output_path)
    print(f"Composite trigger (square + triangle) saved to: {output_path}")
    print(f"Size: {size}x{size}")
    return img


def create_all_triggers(size=8, output_dir='./'):
    """
    Create all three types of backdoor triggers
    
    Args:
        size: Size of the trigger patch (default: 8x8 for CIFAR-10/SVHN)
        output_dir: Directory to save trigger images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print(f"Creating backdoor trigger images for CIFAR-10/SVHN")
    print(f"Trigger size: {size}x{size}")
    print(f"Output directory: {output_dir}")
    print("="*60 + "\n")
    
    # Create triggers
    triggers = {}
    
    # 1. Triangle trigger
    print("1. Creating triangle trigger...")
    triangle_path = os.path.join(output_dir, 'trigger_triangle.png')
    triggers['triangle'] = create_triangle_trigger(size=size, output_path=triangle_path)
    print()
    
    # 2. Square trigger
    print("2. Creating square trigger...")
    square_path = os.path.join(output_dir, 'trigger_square.png')
    triggers['square'] = create_square_trigger(size=size, output_path=square_path)
    print()
    
    # 3. Composite trigger (square + triangle)
    print("3. Creating composite trigger (square + triangle)...")
    composite_path = os.path.join(output_dir, 'trigger_composite.png')
    triggers['composite'] = create_composite_trigger(size=size, output_path=composite_path)
    print()
    
    print("="*60)
    print("All triggers created successfully!")
    print("="*60)
    print("\nFiles created:")
    print(f"  - {triangle_path}")
    print(f"  - {square_path}")
    print(f"  - {composite_path}")
    print("\nUsage examples:")
    print(f"  python train.py --model resnet50 --dataset cifar10 --poisoning --trigger_path trigger_triangle.png")
    print(f"  python clip_train.py --backbone RN50 --dataset cifar10 --poisoning --trigger_path trigger_square.png")
    
    return triggers


def visualize_triggers(size=8):
    """
    Create and visualize all triggers side by side
    """
    import matplotlib.pyplot as plt
    
    # Create triggers
    triangle = create_triangle_trigger(size=size, output_path='temp_triangle.png')
    square = create_square_trigger(size=size, output_path='temp_square.png')
    composite = create_composite_trigger(size=size, output_path='temp_composite.png')
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(triangle)
    axes[0].set_title(f'Triangle Trigger\n({size}x{size})', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(square)
    axes[1].set_title(f'Square Trigger\n({size}x{size})', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(composite)
    axes[2].set_title(f'Composite Trigger\n(Square + Triangle)\n({size}x{size})', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('triggers_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: triggers_visualization.png")
    plt.show()
    
    # Clean up temp files
    os.remove('temp_triangle.png')
    os.remove('temp_square.png')
    os.remove('temp_composite.png')


def create_larger_visualization(trigger_size=8, scale=20):
    """
    Create a larger visualization of triggers for better viewing
    """
    # Create triggers in memory
    triangle = create_triangle_trigger(size=trigger_size, output_path='temp_triangle.png')
    square = create_square_trigger(size=trigger_size, output_path='temp_square.png')
    composite = create_composite_trigger(size=trigger_size, output_path='temp_composite.png')
    
    # Scale up for visualization
    scaled_size = trigger_size * scale
    triangle_scaled = triangle.resize((scaled_size, scaled_size), Image.NEAREST)
    square_scaled = square.resize((scaled_size, scaled_size), Image.NEAREST)
    composite_scaled = composite.resize((scaled_size, scaled_size), Image.NEAREST)
    
    # Create combined image
    combined = Image.new('RGB', (scaled_size * 3 + 40, scaled_size + 20), color='white')
    combined.paste(triangle_scaled, (10, 10))
    combined.paste(square_scaled, (scaled_size + 20, 10))
    combined.paste(composite_scaled, (scaled_size * 2 + 30, 10))
    
    # Save
    combined.save('triggers_large_preview.png')
    print(f"\nLarge preview saved to: triggers_large_preview.png")
    print(f"Preview size: {combined.size}")
    
    # Clean up
    os.remove('temp_triangle.png')
    os.remove('temp_square.png')
    os.remove('temp_composite.png')
    
    return combined


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate backdoor trigger images')
    parser.add_argument('--size', type=int, default=8, 
                       help='Size of trigger patch (default: 8 for 32x32 images)')
    parser.add_argument('--output_dir', type=str, default='./', 
                       help='Output directory for trigger images')
    parser.add_argument('--visualize', action='store_true', 
                       help='Create visualization of triggers')
    parser.add_argument('--preview', action='store_true',
                       help='Create large preview of triggers')
    
    args = parser.parse_args()
    
    # Create triggers
    create_all_triggers(size=args.size, output_dir=args.output_dir)
    
    # Optional: Create visualization
    if args.visualize:
        print("\nCreating visualization...")
        try:
            visualize_triggers(size=args.size)
        except ImportError:
            print("Matplotlib not installed. Skipping visualization.")
    
    # Optional: Create large preview
    if args.preview:
        print("\nCreating large preview...")
        create_larger_visualization(trigger_size=args.size)
