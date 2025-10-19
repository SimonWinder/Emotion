"""
Prediction script for facial emotion classification.

This script loads a trained model and makes predictions on individual images.
"""

import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import config
from model import EmotionCNN
from dataset import get_val_transform
from utils import load_checkpoint


def load_and_preprocess_image(image_path: str) -> torch.Tensor:
    """
    Load and preprocess an image for prediction.

    Args:
        image_path: Path to the image file

    Returns:
        Preprocessed image tensor
    """
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Apply transforms
    transform = get_val_transform()
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor, image


def predict_emotion(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device
) -> tuple:
    """
    Predict emotion from image.

    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on

    Returns:
        Tuple of (predicted_class_idx, predicted_class_name, probabilities)
    """
    model.eval()

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)

        # Get predicted class
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        predicted_class = config.IDX_TO_CLASS[predicted_idx]

    return predicted_idx, predicted_class, probabilities.cpu().numpy()[0]


def visualize_prediction(
    image: Image.Image,
    predicted_class: str,
    probabilities: np.ndarray,
    save_path: str = None
):
    """
    Visualize prediction with image and probability bars.

    Args:
        image: Original PIL image
        predicted_class: Predicted emotion class
        probabilities: Probability for each class
        save_path: Optional path to save visualization
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Display image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(f'Predicted: {predicted_class.upper()}', fontsize=14, fontweight='bold')

    # Display probability bars
    y_pos = np.arange(len(config.EMOTION_CLASSES))
    colors = ['green' if cls == predicted_class else 'skyblue'
              for cls in config.EMOTION_CLASSES]

    bars = ax2.barh(y_pos, probabilities * 100, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(config.EMOTION_CLASSES)
    ax2.set_xlabel('Confidence (%)', fontsize=12)
    ax2.set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)

    # Add percentage labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%',
                ha='left', va='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(
        description='Predict emotion from a facial image'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to the input image'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=str(config.BEST_MODEL_PATH),
        help='Path to the trained model checkpoint'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show visualization of prediction'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Path to save visualization'
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("FACIAL EMOTION PREDICTION")
    print("=" * 70)

    # Check if model exists
    if not config.BEST_MODEL_PATH.exists():
        print(f"\nError: Model not found at {config.BEST_MODEL_PATH}")
        print("Please train the model first by running: p3 train.py")
        return

    # Load model
    print(f"\nLoading model from {args.model}...")
    model = EmotionCNN().to(config.DEVICE)
    load_checkpoint(args.model, model, device=config.DEVICE)

    # Load and preprocess image
    print(f"Loading image from {args.image}...")
    try:
        image_tensor, original_image = load_and_preprocess_image(args.image)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Make prediction
    print("Making prediction...")
    predicted_idx, predicted_class, probabilities = predict_emotion(
        model, image_tensor, config.DEVICE
    )

    # Print results
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f"\nPredicted Emotion: {predicted_class.upper()}")
    print(f"Confidence: {probabilities[predicted_idx] * 100:.2f}%")
    print("\nAll Probabilities:")
    print("-" * 40)

    # Sort by probability
    sorted_indices = np.argsort(probabilities)[::-1]
    for idx in sorted_indices:
        emotion = config.IDX_TO_CLASS[idx]
        prob = probabilities[idx] * 100
        bar = '█' * int(prob / 2)  # Simple text bar
        print(f"  {emotion:10s}: {prob:5.2f}% {bar}")

    print("=" * 70)

    # Visualize if requested
    if args.visualize:
        print("\nGenerating visualization...")
        visualize_prediction(original_image, predicted_class, probabilities, args.save)


if __name__ == "__main__":
    main()
