# Model Architecture

## Overview

This project implements a simplified CNN architecture that matches the TensorFlow specification, converted to PyTorch. The model is designed for facial emotion classification with 4 emotion classes: happy, neutral, sad, and surprise.

## Architecture Specification

### Original TensorFlow Model

```python
def cnn_model_1(input_channels=3):
    model = Sequential()
    model.add(Input(shape=(48, 48, input_channels)))
    model.add(Conv2D(16, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(Dense(4, activation='softmax'))
```

### PyTorch Implementation

The model has been converted to PyTorch in `model.py`:

```python
class EmotionCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=4):
        super(EmotionCNN, self).__init__()

        # Conv layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dense layers
        self.fc1 = nn.Linear(18432, 32)  # 32 * 24 * 24 = 18,432
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.1)
        self.fc2 = nn.Linear(32, num_classes)
```

## Layer-by-Layer Breakdown

| Layer | Type | Input Shape | Output Shape | Parameters |
|-------|------|-------------|--------------|------------|
| Input | - | (batch, 3, 48, 48) | (batch, 3, 48, 48) | 0 |
| Conv1 | Conv2D | (batch, 3, 48, 48) | (batch, 16, 48, 48) | 448 |
| LeakyReLU1 | Activation | (batch, 16, 48, 48) | (batch, 16, 48, 48) | 0 |
| Conv2 | Conv2D | (batch, 16, 48, 48) | (batch, 32, 48, 48) | 4,640 |
| LeakyReLU2 | Activation | (batch, 32, 48, 48) | (batch, 32, 48, 48) | 0 |
| MaxPool | MaxPool2D | (batch, 32, 48, 48) | (batch, 32, 24, 24) | 0 |
| Flatten | Reshape | (batch, 32, 24, 24) | (batch, 18432) | 0 |
| FC1 | Linear | (batch, 18432) | (batch, 32) | 589,856 |
| LeakyReLU3 | Activation | (batch, 32) | (batch, 32) | 0 |
| FC2 | Linear | (batch, 32) | (batch, 4) | 132 |

**Total Parameters: 595,076**

## Key Features

### 1. Input Size
- **48x48 RGB images** (changed from 64x64)
- 3 color channels (Red, Green, Blue)

### 2. Convolutional Layers
- **Conv1**: 3 → 16 filters, 3x3 kernel, same padding
- **Conv2**: 16 → 32 filters, 3x3 kernel, same padding
- Uses LeakyReLU activation (negative_slope=0.1) instead of ReLU

### 3. Pooling
- **Single MaxPooling layer** (2x2) after both conv layers
- Reduces spatial dimensions from 48x48 to 24x24

### 4. Fully Connected Layers
- **FC1**: 18,432 → 32 units (single hidden layer)
- **FC2**: 32 → 4 units (output layer)
- No dropout (unlike the original larger model)

### 5. Activation Functions
- **LeakyReLU**: Allows small gradient when input is negative
- `negative_slope=0.1` means f(x) = 0.1*x when x < 0
- Benefits: Helps prevent "dying ReLU" problem

### 6. Output Layer
- **4 classes**: happy, neutral, sad, surprise
- No explicit softmax in forward pass (handled by CrossEntropyLoss)
- Use `forward_with_softmax()` for inference with probabilities

## TensorFlow vs PyTorch Differences

### Data Format
- **TensorFlow**: (batch, height, width, channels) - channels last
- **PyTorch**: (batch, channels, height, width) - channels first

### Padding
- **TensorFlow**: `padding='same'` automatically pads to maintain size
- **PyTorch**: `padding=1` for 3x3 kernel achieves the same effect

### Softmax
- **TensorFlow**: Explicitly added to output layer
- **PyTorch**: Included in CrossEntropyLoss, not needed in forward pass

### Activation Functions
- **TensorFlow**: `LeakyReLU(negative_slope=0.1)` as separate layer
- **PyTorch**: `nn.LeakyReLU(negative_slope=0.1)` as module

## Weight Initialization

The model uses **Kaiming (He) initialization** optimized for LeakyReLU:

```python
nn.init.kaiming_normal_(m.weight, mode='fan_out',
                        nonlinearity='leaky_relu', a=0.1)
```

This initialization is specifically designed for networks using LeakyReLU activation.

## Model Size Comparison

### New Architecture (Current)
- **Parameters**: 595,076 (~595K)
- **Conv layers**: 2
- **FC hidden units**: 32
- **Input size**: 48x48

### Previous Architecture
- **Parameters**: ~4.4M
- **Conv layers**: 4 (with BatchNorm)
- **FC hidden units**: 512, 256
- **Input size**: 64x64

**Size Reduction**: ~87% fewer parameters (7.4x smaller)

## Advantages of Simplified Architecture

1. **Faster Training**: Fewer parameters = faster forward/backward pass
2. **Less Overfitting**: Simpler model with no dropout
3. **Lower Memory**: Smaller model fits in limited GPU memory
4. **Quick Inference**: Fast predictions on CPU
5. **Matching Baseline**: Directly comparable to TensorFlow implementation

## Usage

### Training
```bash
p3 train.py
```

### Inference
```python
from model import EmotionCNN
import torch

model = EmotionCNN()
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()

# Get logits
logits = model(image_tensor)

# Get probabilities
probabilities = model.forward_with_softmax(image_tensor)
```

## Configuration Changes

Updated `config.py`:
```python
IMAGE_SIZE = 48              # Changed from 64
CONV_CHANNELS = [16, 32]     # Changed from [64, 128, 256, 512]
FC_HIDDEN_SIZE = 32          # Changed from 512
DROPOUT_RATE = 0.0           # Changed from 0.5 (no dropout)
```

## Expected Performance

With this simplified architecture:
- **Training time**: Faster than previous (fewer parameters)
- **Target accuracy**: ≥85% on test set
- **Model size**: ~2.3 MB (vs ~17 MB for previous model)
- **Inference speed**: <10ms per image on CPU

## Testing the Model

Run the test script:
```bash
p3 model.py
```

Expected output:
```
EmotionCNN (Simplified Architecture)
  Input: (3, 48, 48)
  Layer 1: Conv2d(3 → 16, kernel=3x3, padding=same)
  Layer 2: Conv2d(16 → 32, kernel=3x3, padding=same)
  Layer 3: MaxPool2d(kernel=2x2)
  Layer 4: Flatten
  Layer 5: Linear(18432 → 32)
  Layer 6: Linear(32 → 4)

Total parameters: 595,076
```

## References

- Original TensorFlow specification
- PyTorch documentation: https://pytorch.org/docs/
- Kaiming initialization paper: "Delving Deep into Rectifiers" (He et al., 2015)
- LeakyReLU paper: "Rectifier Nonlinearities Improve Neural Network Acoustic Models" (Maas et al., 2013)

---

**Last Updated**: October 19, 2025
**PyTorch Version**: 2.0+
**Python Version**: 3.8+
