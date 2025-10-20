"""
CNN model architecture for facial emotion classification.

This module defines the EmotionCNN model with 4 convolutional blocks
and 3 fully connected layers for classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class EmotionCNN(nn.Module):
    """
    Convolutional Neural Network for facial emotion classification.

    Architecture:
        Convolutional Blocks:
        1. Conv2d(3→64) → BatchNorm → ReLU → MaxPool
        2. Conv2d(64→128) → BatchNorm → ReLU → MaxPool
        3. Conv2d(128→256) → BatchNorm → ReLU → MaxPool
        4. Conv2d(256→256) → BatchNorm → ReLU → MaxPool

        Fully Connected Layers:
        1. FC(4096→256) → ReLU → Dropout(0.5)
        2. FC(256→256) → ReLU → Dropout(0.5)
        3. FC(256→4) → Output

    Args:
        input_channels: Number of input channels (3 for RGB)
        num_classes: Number of output classes (default: 4)
        dropout_rate: Dropout probability (default: from config)
    """

    def __init__(
        self,
        input_channels: int = config.INPUT_CHANNELS,
        num_classes: int = config.NUM_CLASSES,
        dropout_rate: float = config.DROPOUT_RATE
    ):
        super(EmotionCNN, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Convolutional Block 1: 3 → 64
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 2: 64 → 128
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 3: 128 → 256
        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 4: 256 → 256
        self.conv4 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=1
        )
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the flattened size after conv layers
        # Input: 48x48 → after 4 maxpool(2,2): 3x3
        # 256 channels * 3 * 3 = 2,304
        self.flattened_size = 256 * 3 * 3

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Output layer
        self.fc3 = nn.Linear(256, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 48, 48)

        Returns:
            Output tensor of shape (batch_size, num_classes)
            Note: Output is logits (no softmax applied)
        """
        # Convolutional Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Convolutional Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Convolutional Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Convolutional Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Output layer (no activation - handled by CrossEntropyLoss)
        x = self.fc3(x)

        return x

    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters.

        Returns:
            Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_architecture(self):
        """Print model architecture summary."""
        print("\n" + "=" * 70)
        print("MODEL ARCHITECTURE")
        print("=" * 70)
        print(f"\nEmotionCNN")
        print(f"  Input: ({self.input_channels}, {config.IMAGE_SIZE}, {config.IMAGE_SIZE})")
        print(f"\n  Convolutional Blocks:")
        print(f"    Block 1: Conv2d(3 → 64) → BatchNorm → ReLU → MaxPool")
        print(f"    Block 2: Conv2d(64 → 128) → BatchNorm → ReLU → MaxPool")
        print(f"    Block 3: Conv2d(128 → 256) → BatchNorm → ReLU → MaxPool")
        print(f"    Block 4: Conv2d(256 → 256) → BatchNorm → ReLU → MaxPool")
        print(f"\n  Fully Connected Layers:")
        print(f"    FC1: Linear({self.flattened_size} → 256) → ReLU → Dropout({self.dropout_rate})")
        print(f"    FC2: Linear(256 → 256) → ReLU → Dropout({self.dropout_rate})")
        print(f"    FC3: Linear(256 → {self.num_classes})")
        print(f"\n  Output: {self.num_classes} classes")
        print(f")")
        print(f"\nTotal parameters: {self.count_parameters():,}")
        print("=" * 70)


def create_model(device: torch.device = config.DEVICE) -> EmotionCNN:
    """
    Create and initialize the EmotionCNN model.

    Args:
        device: Device to place the model on (CPU or CUDA)

    Returns:
        Initialized EmotionCNN model
    """
    model = EmotionCNN()
    model = model.to(device)
    model.print_architecture()
    return model


if __name__ == "__main__":
    """Test model creation and forward pass."""
    print("Testing model creation...")

    # Create model
    model = create_model()

    # Test forward pass with dummy input
    print("\nTesting forward pass...")
    batch_size = 4
    dummy_input = torch.randn(
        batch_size,
        config.INPUT_CHANNELS,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE
    ).to(config.DEVICE)

    print(f"Input shape: {dummy_input.shape}")

    # Forward pass (logits)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output shape (logits): {output.shape}")
    print(f"Output sample (logits):\n{output[:2]}")

    # Test softmax probabilities
    probabilities = F.softmax(output, dim=1)
    print(f"\nProbabilities shape: {probabilities.shape}")
    print(f"Probabilities sample (should sum to 1.0):")
    for i in range(min(2, batch_size)):
        print(f"  Sample {i}: {probabilities[i].cpu().numpy()}")
        print(f"  Sum: {probabilities[i].sum().item():.6f}")

    print("\nModel test complete!")
