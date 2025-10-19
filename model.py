"""
CNN model architecture for facial emotion classification.

This module defines the EmotionCNN model matching the specified architecture:
- Conv2D(16) → LeakyReLU → Conv2D(32) → LeakyReLU → MaxPool
- Flatten → Dense(32) → LeakyReLU → Dense(4) with softmax

Architecture is converted from TensorFlow to PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class EmotionCNN(nn.Module):
    """
    Convolutional Neural Network for facial emotion classification.

    Architecture (converted from TensorFlow):
        Input: (48, 48, 3) RGB images
        - Conv2D: 16 filters, 3x3 kernel, same padding
        - LeakyReLU: negative_slope=0.1
        - Conv2D: 32 filters, 3x3 kernel, same padding
        - LeakyReLU: negative_slope=0.1
        - MaxPooling2D: pool_size=2x2
        - Flatten
        - Dense: 32 units
        - LeakyReLU: negative_slope=0.1
        - Dense: 4 units (output layer)

    Note: PyTorch uses CrossEntropyLoss which includes softmax,
    so we don't apply softmax in the forward pass.

    Args:
        input_channels: Number of input channels (3 for RGB)
        num_classes: Number of output classes (default: 4)
    """

    def __init__(
        self,
        input_channels: int = config.INPUT_CHANNELS,
        num_classes: int = config.NUM_CLASSES
    ):
        super(EmotionCNN, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes

        # First Convolutional layer: 3 → 16 filters, 3x3 kernel, same padding
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=16,
            kernel_size=3,
            padding=1  # 'same' padding in PyTorch
        )

        # LeakyReLU with negative_slope=0.1
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)

        # Second Convolutional layer: 16 → 32 filters, 3x3 kernel, same padding
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1  # 'same' padding in PyTorch
        )

        # LeakyReLU with negative_slope=0.1
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)

        # Max pooling layer with pool_size 2x2
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate flattened size
        # Input: 48x48 → after one maxpool(2,2): 24x24
        # 32 channels * 24 * 24 = 18,432
        self.flattened_size = 32 * 24 * 24

        # Dense layer with 32 nodes
        self.fc1 = nn.Linear(self.flattened_size, 32)

        # LeakyReLU with negative_slope=0.1
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.1)

        # Output layer with 4 nodes (number of classes)
        # Note: No softmax here - CrossEntropyLoss handles it
        self.fc2 = nn.Linear(32, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using He initialization for LeakyReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for LeakyReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.1)
                if m.bias is not None:
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
        # First convolutional layer + LeakyReLU
        x = self.conv1(x)
        x = self.leaky_relu1(x)

        # Second convolutional layer + LeakyReLU
        x = self.conv2(x)
        x = self.leaky_relu2(x)

        # Max pooling
        x = self.maxpool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # First dense layer + LeakyReLU
        x = self.fc1(x)
        x = self.leaky_relu3(x)

        # Output layer (no activation - handled by CrossEntropyLoss)
        x = self.fc2(x)

        return x

    def forward_with_softmax(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with softmax for inference/prediction.
        Use this method when you need probabilities (e.g., during inference).

        Args:
            x: Input tensor of shape (batch_size, 3, 48, 48)

        Returns:
            Output tensor with softmax applied (batch_size, num_classes)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

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
        print(f"\nEmotionCNN (Simplified Architecture)")
        print(f"  Input: ({self.input_channels}, {config.IMAGE_SIZE}, {config.IMAGE_SIZE})")
        print(f"\n  Layer 1: Conv2d({self.input_channels} → 16, kernel=3x3, padding=same)")
        print(f"           LeakyReLU(negative_slope=0.1)")
        print(f"\n  Layer 2: Conv2d(16 → 32, kernel=3x3, padding=same)")
        print(f"           LeakyReLU(negative_slope=0.1)")
        print(f"\n  Layer 3: MaxPool2d(kernel=2x2)")
        print(f"           Output shape: (32, 24, 24)")
        print(f"\n  Layer 4: Flatten")
        print(f"           Output shape: ({self.flattened_size},)")
        print(f"\n  Layer 5: Linear({self.flattened_size} → 32)")
        print(f"           LeakyReLU(negative_slope=0.1)")
        print(f"\n  Layer 6: Linear(32 → {self.num_classes})")
        print(f"           [Softmax handled by CrossEntropyLoss during training]")
        print(f"\n  Output: {self.num_classes} classes")
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

    # Test forward_with_softmax method
    print("\nTesting forward_with_softmax method...")
    with torch.no_grad():
        probs = model.forward_with_softmax(dummy_input)
    print(f"Probabilities (direct): {probs[0].cpu().numpy()}")
    print(f"Sum: {probs[0].sum().item():.6f}")

    print("\nModel test complete!")
