"""
Evaluation script for facial emotion classification model.

This script evaluates the trained model on the test set and generates
classification metrics, confusion matrix, and classification report.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

import config
from model import EmotionCNN
from dataset import get_data_loaders
from utils import (
    load_checkpoint, calculate_accuracy, plot_confusion_matrix,
    save_classification_report, set_seed
)


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> tuple:
    """
    Evaluate model on test set.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on

    Returns:
        Tuple of (all_predictions, all_labels, average_accuracy, average_loss)
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_losses = []

    criterion = nn.CrossEntropyLoss()

    print("\nEvaluating model on test set...")
    pbar = tqdm(test_loader, desc="Testing")

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_losses.append(loss.item())

            # Calculate running accuracy
            accuracy = calculate_accuracy(outputs, labels)
            pbar.set_postfix({'acc': f'{accuracy:.2f}%'})

    # Calculate overall metrics
    avg_loss = np.mean(all_losses)
    all_predictions_tensor = torch.tensor(all_predictions)
    all_labels_tensor = torch.tensor(all_labels)
    overall_accuracy = calculate_accuracy(
        torch.nn.functional.one_hot(all_predictions_tensor, num_classes=config.NUM_CLASSES).float(),
        all_labels_tensor
    )

    return all_predictions, all_labels, overall_accuracy, avg_loss


def print_per_class_accuracy(y_true: list, y_pred: list):
    """
    Print per-class accuracy.

    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    print("\nPer-Class Accuracy:")
    print("-" * 40)

    for class_idx, class_name in enumerate(config.EMOTION_CLASSES):
        # Find all samples of this class
        class_mask = [i for i, label in enumerate(y_true) if label == class_idx]

        if len(class_mask) == 0:
            continue

        # Calculate accuracy for this class
        correct = sum(1 for i in class_mask if y_pred[i] == class_idx)
        total = len(class_mask)
        accuracy = 100 * correct / total

        print(f"  {class_name:10s}: {accuracy:.2f}% ({correct}/{total})")


def main():
    """Main evaluation function."""
    print("\n" + "=" * 70)
    print("FACIAL EMOTION CLASSIFICATION - EVALUATION")
    print("=" * 70)

    # Set random seed
    set_seed(config.RANDOM_SEED)

    # Check if model exists
    if not config.BEST_MODEL_PATH.exists():
        print(f"\nError: Model not found at {config.BEST_MODEL_PATH}")
        print("Please train the model first by running: p3 train.py")
        return

    # Create data loaders
    print("\nLoading test dataset...")
    _, _, test_loader = get_data_loaders()

    # Create model
    print("\nLoading model...")
    model = EmotionCNN().to(config.DEVICE)

    # Load trained weights
    checkpoint = load_checkpoint(config.BEST_MODEL_PATH, model, device=config.DEVICE)

    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    # Evaluate model
    y_pred, y_true, test_accuracy, test_loss = evaluate_model(
        model, test_loader, config.DEVICE
    )

    # Print overall results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Print per-class accuracy
    print_per_class_accuracy(y_true, y_pred)

    # Generate and save confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, normalize=True)
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=config.RESULTS_DIR / "confusion_matrix_counts.png",
        normalize=False
    )

    # Generate and save classification report
    print("\nGenerating classification report...")
    save_classification_report(y_true, y_pred)

    # Check if accuracy meets target
    target_accuracy = 85.0
    print("\n" + "=" * 70)
    print("ACCEPTANCE CRITERIA")
    print("=" * 70)

    if test_accuracy >= target_accuracy:
        print(f"✓ Model achieves target accuracy of {target_accuracy}%")
        print(f"  Actual accuracy: {test_accuracy:.2f}%")
    else:
        print(f"✗ Model does not meet target accuracy of {target_accuracy}%")
        print(f"  Actual accuracy: {test_accuracy:.2f}%")
        print(f"  Gap: {target_accuracy - test_accuracy:.2f}%")
        print("\nSuggestions for improvement:")
        print("  - Train for more epochs")
        print("  - Try transfer learning (ResNet, EfficientNet)")
        print("  - Adjust learning rate or batch size")
        print("  - Add more data augmentation")
        print("  - Increase model capacity")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print(f"  - Confusion matrix: {config.CONFUSION_MATRIX_PATH}")
    print(f"  - Classification report: {config.CLASSIFICATION_REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
