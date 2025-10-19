"""
Dataset loading and preprocessing for facial emotion classification.

This module provides custom Dataset classes and data loader functions for
loading and transforming the facial emotion images.
"""

import os
from pathlib import Path
from typing import Tuple, Callable, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import config


class EmotionDataset(Dataset):
    """
    Custom Dataset for loading facial emotion images.

    Args:
        root_dir: Root directory containing emotion subdirectories
        transform: Optional transform to be applied to images
        class_to_idx: Optional mapping from class names to indices
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[dict] = None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_to_idx = class_to_idx or config.CLASS_TO_IDX

        # Load all image paths and labels
        self.samples = []
        self.labels = []

        for emotion in config.EMOTION_CLASSES:
            emotion_dir = self.root_dir / emotion

            if not emotion_dir.exists():
                print(f"Warning: Directory not found: {emotion_dir}")
                continue

            # Get all jpg images in the emotion directory
            image_files = list(emotion_dir.glob("*.jpg"))
            image_files.extend(list(emotion_dir.glob("*.jpeg")))
            image_files.extend(list(emotion_dir.glob("*.png")))

            for img_path in image_files:
                self.samples.append(str(img_path))
                self.labels.append(self.class_to_idx[emotion])

        print(f"Loaded {len(self.samples)} images from {root_dir}")

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image_tensor, label)
        """
        img_path = self.samples[idx]
        label = self.labels[idx]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new("RGB", (config.IMAGE_SIZE, config.IMAGE_SIZE))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> dict:
        """
        Get the distribution of classes in the dataset.

        Returns:
            Dictionary mapping class names to counts
        """
        distribution = {emotion: 0 for emotion in config.EMOTION_CLASSES}

        for label in self.labels:
            emotion = config.IDX_TO_CLASS[label]
            distribution[emotion] += 1

        return distribution


def get_train_transform() -> transforms.Compose:
    """
    Get training data augmentation pipeline.

    Returns:
        Composed transforms for training data
    """
    aug_config = config.TRAIN_AUGMENTATION

    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=aug_config["horizontal_flip_prob"]),
        transforms.RandomRotation(degrees=aug_config["rotation_degrees"]),
        transforms.ColorJitter(
            brightness=aug_config["color_jitter"]["brightness"],
            contrast=aug_config["color_jitter"]["contrast"],
            saturation=aug_config["color_jitter"]["saturation"],
            hue=aug_config["color_jitter"]["hue"]
        ),
        transforms.RandomResizedCrop(
            config.IMAGE_SIZE,
            scale=aug_config["random_crop_scale"],
            ratio=aug_config["random_crop_ratio"]
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.NORMALIZE_MEAN,
            std=config.NORMALIZE_STD
        )
    ])


def get_val_transform() -> transforms.Compose:
    """
    Get validation/test data transform pipeline (no augmentation).

    Returns:
        Composed transforms for validation/test data
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.NORMALIZE_MEAN,
            std=config.NORMALIZE_STD
        )
    ])


def get_data_loaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoader instances for train, validation, and test sets.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = EmotionDataset(
        root_dir=config.TRAIN_DIR,
        transform=get_train_transform()
    )

    val_dataset = EmotionDataset(
        root_dir=config.VAL_DIR,
        transform=get_val_transform()
    )

    test_dataset = EmotionDataset(
        root_dir=config.TEST_DIR,
        transform=get_val_transform()
    )

    # Print dataset statistics
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    print(f"\nTraining set: {len(train_dataset)} images")
    train_dist = train_dataset.get_class_distribution()
    for emotion, count in train_dist.items():
        print(f"  {emotion:10s}: {count:5d} ({100 * count / len(train_dataset):.1f}%)")

    print(f"\nValidation set: {len(val_dataset)} images")
    val_dist = val_dataset.get_class_distribution()
    for emotion, count in val_dist.items():
        print(f"  {emotion:10s}: {count:5d} ({100 * count / len(val_dataset):.1f}%)")

    print(f"\nTest set: {len(test_dataset)} images")
    test_dist = test_dataset.get_class_distribution()
    for emotion, count in test_dist.items():
        print(f"  {emotion:10s}: {count:5d} ({100 * count / len(test_dataset):.1f}%)")

    print("=" * 70)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    return train_loader, val_loader, test_loader


def calculate_class_weights(train_dataset: EmotionDataset) -> torch.Tensor:
    """
    Calculate class weights for handling imbalanced dataset.

    Args:
        train_dataset: Training dataset

    Returns:
        Tensor of class weights
    """
    distribution = train_dataset.get_class_distribution()
    total_samples = len(train_dataset)

    # Calculate weights: inverse of class frequency
    weights = []
    for emotion in config.EMOTION_CLASSES:
        count = distribution[emotion]
        weight = total_samples / (config.NUM_CLASSES * count)
        weights.append(weight)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    print("\nClass weights for imbalanced dataset:")
    for emotion, weight in zip(config.EMOTION_CLASSES, weights):
        print(f"  {emotion:10s}: {weight:.4f}")

    return weights_tensor


if __name__ == "__main__":
    """Test data loading."""
    print("Testing data loading...")

    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders()

    # Test loading a batch
    print("\nTesting batch loading...")
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels[:8]}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        break

    # Calculate class weights
    train_dataset = EmotionDataset(
        root_dir=config.TRAIN_DIR,
        transform=get_train_transform()
    )
    weights = calculate_class_weights(train_dataset)

    print("\nData loading test complete!")
