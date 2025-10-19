"""
Utility functions for training, evaluation, and visualization.

This module provides helper functions for model checkpointing, metrics tracking,
plotting, and other utilities.
"""

import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import config


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy
    """

    def __init__(
        self,
        patience: int = config.EARLY_STOPPING_PATIENCE,
        min_delta: float = config.EARLY_STOPPING_MIN_DELTA,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score (loss or accuracy)

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class MetricTracker:
    """Track and compute running average of metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, value: float, n: int = 1):
        """
        Update metric with new value.

        Args:
            value: New value to add
            n: Number of samples this value represents
        """
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_accuracy: float,
    filepath: Path,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        val_accuracy: Validation accuracy
        filepath: Path to save checkpoint
        scheduler: Optional learning rate scheduler
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'config': {
            'num_classes': config.NUM_CLASSES,
            'image_size': config.IMAGE_SIZE,
            'conv_channels': config.CONV_CHANNELS,
            'fc_hidden_size': config.FC_HIDDEN_SIZE,
            'dropout_rate': config.DROPOUT_RATE
        }
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = config.DEVICE
) -> Dict:
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to map tensors to

    Returns:
        Dictionary containing checkpoint information
    """
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Checkpoint loaded from {filepath}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Val Accuracy: {checkpoint['val_accuracy']:.2f}%")

    return checkpoint


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate classification accuracy.

    Args:
        outputs: Model outputs (logits)
        labels: Ground truth labels

    Returns:
        Accuracy as percentage
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = 100 * correct / total
    return accuracy


def save_training_history(history: Dict, filepath: Path = config.HISTORY_PATH):
    """
    Save training history to JSON file.

    Args:
        history: Dictionary containing training metrics
        filepath: Path to save history
    """
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {filepath}")


def load_training_history(filepath: Path = config.HISTORY_PATH) -> Dict:
    """
    Load training history from JSON file.

    Args:
        filepath: Path to history file

    Returns:
        Dictionary containing training metrics
    """
    with open(filepath, 'r') as f:
        history = json.load(f)
    return history


def plot_training_curves(
    history: Dict,
    save_path: Path = config.TRAINING_CURVES_PATH
):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save the plot
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    save_path: Path = config.CONFUSION_MATRIX_PATH,
    normalize: bool = True
):
    """
    Plot confusion matrix heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
        normalize: Whether to normalize the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=config.EMOTION_CLASSES,
        yticklabels=config.EMOTION_CLASSES,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''),
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def save_classification_report(
    y_true: List[int],
    y_pred: List[int],
    save_path: Path = config.CLASSIFICATION_REPORT_PATH
):
    """
    Generate and save classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the report
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=config.EMOTION_CLASSES,
        digits=4
    )

    with open(save_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(report)
        f.write("\n" + "=" * 70 + "\n")

    print(f"Classification report saved to {save_path}")
    print("\n" + report)


def set_seed(seed: int = config.RANDOM_SEED):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.

    Args:
        optimizer: PyTorch optimizer

    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == "__main__":
    """Test utility functions."""
    print("Testing utility functions...")

    # Test MetricTracker
    tracker = MetricTracker()
    tracker.update(0.5, n=32)
    tracker.update(0.3, n=32)
    print(f"Metric Tracker - Avg: {tracker.avg:.4f}")

    # Test EarlyStopping
    early_stop = EarlyStopping(patience=3, min_delta=0.01, mode='min')
    print(f"\nTesting EarlyStopping:")
    for i, loss in enumerate([0.5, 0.4, 0.39, 0.391, 0.392, 0.393]):
        should_stop = early_stop(loss)
        print(f"  Epoch {i+1}: Loss={loss:.3f}, Counter={early_stop.counter}, Stop={should_stop}")
        if should_stop:
            break

    print("\nUtility functions test complete!")
