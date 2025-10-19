"""
Data exploration script for facial emotion dataset.

This script visualizes sample images from each emotion class and displays
dataset statistics.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random

import config
from dataset import get_data_loaders, get_train_transform


def plot_sample_images(num_samples: int = 4):
    """
    Display sample images from each emotion class.

    Args:
        num_samples: Number of samples to show per class
    """
    fig, axes = plt.subplots(config.NUM_CLASSES, num_samples, figsize=(12, 10))

    for class_idx, emotion in enumerate(config.EMOTION_CLASSES):
        emotion_dir = config.TRAIN_DIR / emotion

        # Get all images for this emotion
        image_files = list(emotion_dir.glob("*.jpg"))
        image_files.extend(list(emotion_dir.glob("*.jpeg")))

        # Sample random images
        sample_files = random.sample(image_files, min(num_samples, len(image_files)))

        for img_idx, img_path in enumerate(sample_files):
            # Load image
            from PIL import Image
            image = Image.open(img_path).convert('RGB')

            # Display
            ax = axes[class_idx, img_idx]
            ax.imshow(image)
            ax.axis('off')

            # Add title to first image of each row
            if img_idx == 0:
                ax.set_ylabel(emotion.upper(), rotation=0, size='large',
                             labelpad=30, weight='bold')

    plt.suptitle('Sample Images from Each Emotion Class',
                 fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / 'sample_images.png', dpi=300, bbox_inches='tight')
    print(f"Sample images saved to {config.RESULTS_DIR / 'sample_images.png'}")
    plt.show()


def plot_augmented_images(num_samples: int = 5):
    """
    Display original and augmented versions of an image.

    Args:
        num_samples: Number of augmented versions to show
    """
    # Get a random image
    emotion = random.choice(config.EMOTION_CLASSES)
    emotion_dir = config.TRAIN_DIR / emotion
    image_files = list(emotion_dir.glob("*.jpg"))
    img_path = random.choice(image_files)

    # Load original image
    from PIL import Image
    original_image = Image.open(img_path).convert('RGB')

    # Get augmentation transform
    transform = get_train_transform()

    # Create figure
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))

    # First row: original image repeated
    for i in range(num_samples):
        axes[0, i].imshow(original_image)
        axes[0, i].axis('off')
        if i == num_samples // 2:
            axes[0, i].set_title('Original Image', fontweight='bold')

    # Second row: augmented versions
    for i in range(num_samples):
        # Apply augmentation
        augmented = transform(original_image)

        # Convert tensor back to image for display
        # Denormalize
        augmented = augmented.clone()
        for t, m, s in zip(augmented, config.NORMALIZE_MEAN, config.NORMALIZE_STD):
            t.mul_(s).add_(m)

        # Clip and convert
        augmented = torch.clamp(augmented, 0, 1)
        augmented_np = augmented.permute(1, 2, 0).numpy()

        axes[1, i].imshow(augmented_np)
        axes[1, i].axis('off')
        if i == num_samples // 2:
            axes[1, i].set_title('Augmented Versions', fontweight='bold')

    plt.suptitle(f'Data Augmentation Examples - {emotion.upper()}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / 'augmentation_examples.png', dpi=300, bbox_inches='tight')
    print(f"Augmentation examples saved to {config.RESULTS_DIR / 'augmentation_examples.png'}")
    plt.show()


def plot_class_distribution():
    """Plot class distribution for train, validation, and test sets."""
    from dataset import EmotionDataset

    # Load datasets
    train_dataset = EmotionDataset(config.TRAIN_DIR)
    val_dataset = EmotionDataset(config.VAL_DIR)
    test_dataset = EmotionDataset(config.TEST_DIR)

    # Get distributions
    train_dist = train_dataset.get_class_distribution()
    val_dist = val_dataset.get_class_distribution()
    test_dist = test_dataset.get_class_distribution()

    # Create bar plot
    x = np.arange(len(config.EMOTION_CLASSES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    train_counts = [train_dist[e] for e in config.EMOTION_CLASSES]
    val_counts = [val_dist[e] for e in config.EMOTION_CLASSES]
    test_counts = [test_dist[e] for e in config.EMOTION_CLASSES]

    ax.bar(x - width, train_counts, width, label='Train', color='skyblue')
    ax.bar(x, val_counts, width, label='Validation', color='lightcoral')
    ax.bar(x + width, test_counts, width, label='Test', color='lightgreen')

    ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution Across Splits', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in config.EMOTION_CLASSES])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for i, (train, val, test) in enumerate(zip(train_counts, val_counts, test_counts)):
        ax.text(i - width, train + 50, str(train), ha='center', va='bottom', fontsize=9)
        ax.text(i, val + 50, str(val), ha='center', va='bottom', fontsize=9)
        ax.text(i + width, test + 2, str(test), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / 'class_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Class distribution saved to {config.RESULTS_DIR / 'class_distribution.png'}")
    plt.show()


def main():
    """Main exploration function."""
    print("\n" + "=" * 70)
    print("DATASET EXPLORATION")
    print("=" * 70)

    # Set random seed for reproducibility
    random.seed(config.RANDOM_SEED)
    import torch
    torch.manual_seed(config.RANDOM_SEED)

    # Create results directory
    config.RESULTS_DIR.mkdir(exist_ok=True)

    # Print dataset info
    print("\nDataset Information:")
    print(f"  Data root: {config.DATA_ROOT}")
    print(f"  Classes: {config.EMOTION_CLASSES}")
    print(f"  Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")

    # Load and display statistics
    print("\n" + "-" * 70)
    train_loader, val_loader, test_loader = get_data_loaders()

    # Plot class distribution
    print("\nGenerating class distribution plot...")
    plot_class_distribution()

    # Plot sample images
    print("\nGenerating sample images plot...")
    plot_sample_images(num_samples=4)

    # Plot augmented images
    print("\nGenerating data augmentation examples...")
    plot_augmented_images(num_samples=5)

    print("\n" + "=" * 70)
    print("EXPLORATION COMPLETE")
    print("=" * 70)
    print(f"\nVisualizations saved to: {config.RESULTS_DIR}")
    print("  - sample_images.png")
    print("  - augmentation_examples.png")
    print("  - class_distribution.png")
    print("=" * 70)


if __name__ == "__main__":
    import torch
    main()
