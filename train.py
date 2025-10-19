"""
Training script for facial emotion classification model.

This script trains the EmotionCNN model on the training set, validates on the
validation set, and saves the best model checkpoint.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

import config
from model import create_model
from dataset import get_data_loaders, calculate_class_weights, EmotionDataset
from utils import (
    EarlyStopping, MetricTracker, save_checkpoint, save_training_history,
    plot_training_curves, calculate_accuracy, set_seed, get_lr
)


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> tuple:
    """
    Train the model for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.train()

    loss_tracker = MetricTracker()
    acc_tracker = MetricTracker()

    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [Train]")

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        accuracy = calculate_accuracy(outputs, labels)

        # Update metrics
        batch_size = images.size(0)
        loss_tracker.update(loss.item(), batch_size)
        acc_tracker.update(accuracy, batch_size)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_tracker.avg:.4f}',
            'acc': f'{acc_tracker.avg:.2f}%'
        })

    return loss_tracker.avg, acc_tracker.avg


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> tuple:
    """
    Validate the model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number

    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.eval()

    loss_tracker = MetricTracker()
    acc_tracker = MetricTracker()

    # Progress bar
    pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [Val]  ")

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Calculate accuracy
            accuracy = calculate_accuracy(outputs, labels)

            # Update metrics
            batch_size = images.size(0)
            loss_tracker.update(loss.item(), batch_size)
            acc_tracker.update(accuracy, batch_size)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_tracker.avg:.4f}',
                'acc': f'{acc_tracker.avg:.2f}%'
            })

    return loss_tracker.avg, acc_tracker.avg


def train():
    """Main training function."""
    print("\n" + "=" * 70)
    print("FACIAL EMOTION CLASSIFICATION - TRAINING")
    print("=" * 70)

    # Set random seed for reproducibility
    set_seed(config.RANDOM_SEED)

    # Print configuration
    config.print_config()

    # Create data loaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = get_data_loaders()

    # Calculate class weights for imbalanced dataset
    if config.USE_CLASS_WEIGHTS:
        train_dataset = EmotionDataset(
            root_dir=config.TRAIN_DIR,
            transform=None  # Just for counting
        )
        class_weights = calculate_class_weights(train_dataset)
        class_weights = class_weights.to(config.DEVICE)
    else:
        class_weights = None

    # Create model
    print("\nInitializing model...")
    model = create_model(config.DEVICE)

    # Define loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Define optimizer
    if config.OPTIMIZER_TYPE == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER_TYPE == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER_TYPE == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER_TYPE}")

    # Define learning rate scheduler
    scheduler = None
    if config.USE_SCHEDULER:
        if config.SCHEDULER_TYPE == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.SCHEDULER_FACTOR,
                patience=config.SCHEDULER_PATIENCE,
                min_lr=config.SCHEDULER_MIN_LR,
                verbose=True
            )
        elif config.SCHEDULER_TYPE == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.5
            )

    # Early stopping
    early_stopping = None
    if config.USE_EARLY_STOPPING:
        early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=config.EARLY_STOPPING_MIN_DELTA,
            mode='min'
        )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }

    # Best model tracking
    best_val_acc = 0.0
    best_epoch = 0

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    start_time = time.time()

    # Training loop
    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start_time = time.time()

        # Get current learning rate
        current_lr = get_lr(optimizer)

        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, epoch
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, config.DEVICE, epoch
        )

        # Update learning rate scheduler
        if scheduler is not None:
            if config.SCHEDULER_TYPE == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Save metrics to history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Print epoch summary
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss, val_acc,
                config.BEST_MODEL_PATH, scheduler
            )
            print(f"  *** New best model saved! (Val Acc: {val_acc:.2f}%) ***")

        # Save last model
        save_checkpoint(
            model, optimizer, epoch, train_loss, val_loss, val_acc,
            config.LAST_MODEL_PATH, scheduler
        )

        print("-" * 70)

        # Early stopping check
        if early_stopping is not None:
            if early_stopping(val_loss):
                print(f"\nEarly stopping triggered after epoch {epoch}")
                print(f"No improvement in validation loss for {early_stopping.patience} epochs")
                break

    # Training complete
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total training time: {total_time / 60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"Best model saved to: {config.BEST_MODEL_PATH}")

    # Save training history
    save_training_history(history)

    # Plot training curves
    print("\nGenerating training curves...")
    plot_training_curves(history)

    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Run evaluation: p3 evaluate.py")
    print("  2. Make predictions: p3 predict.py --image path/to/image.jpg")
    print("=" * 70)


if __name__ == "__main__":
    train()
