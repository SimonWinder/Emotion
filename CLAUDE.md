# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Facial emotion classification using Convolutional Neural Networks (CNN) built with PyTorch. The model classifies facial expressions into 4 emotion categories: happy, neutral, sad, and surprise.

## Dataset Structure

```
Facial_emotion_images/
├── train/          # 15,109 training images
│   ├── happy/      # 3,976 images
│   ├── neutral/    # 3,978 images
│   ├── sad/        # 3,982 images
│   └── surprise/   # 3,173 images
├── validation/     # 4,977 validation images
│   ├── happy/      # 1,825 images
│   ├── neutral/    # 1,216 images
│   ├── sad/        # 1,139 images
│   └── surprise/   # 797 images
└── test/           # 128 test images (32 per class)
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

Images are in JPG format, organized by emotion class in subdirectories.

## Running Commands

**Important:** Always use the `p3` alias instead of `python` or `python3` when running Python code.

**Train the model:**
```bash
p3 train.py
```

**Evaluate on test set:**
```bash
p3 evaluate.py
```

**Make predictions on new images:**
```bash
p3 predict.py --image path/to/image.jpg
```

## Dependencies

```bash
pip install torch torchvision numpy matplotlib pillow scikit-learn
```

For GPU support (recommended for faster training):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Model Architecture

The CNN architecture should be defined in `model.py` and include:
- Convolutional layers for feature extraction
- Pooling layers for dimensionality reduction
- Batch normalization for training stability
- Dropout for regularization
- Fully connected layers for classification

Key considerations:
- Input: RGB images (3 channels)
- Output: 4 classes (happy, neutral, sad, surprise)
- Loss function: CrossEntropyLoss (for multi-class classification)
- Optimizer: Adam or SGD with appropriate learning rate

## Training Strategy

1. **Data Augmentation**: Apply random transformations (rotation, flip, brightness) to training images to improve generalization
2. **Validation**: Monitor validation accuracy/loss to detect overfitting
3. **Learning Rate Scheduling**: Reduce learning rate when validation loss plateaus
4. **Early Stopping**: Stop training if validation performance stops improving
5. **Checkpointing**: Save best model based on validation accuracy

## Performance Metrics

Track these metrics:
- Accuracy (overall and per-class)
- Precision, Recall, F1-score for each emotion
- Confusion matrix to identify misclassifications
- Training/validation loss curves

## Class Imbalance

Note slight class imbalance in training data:
- Happy: 3,976 (26.3%)
- Neutral: 3,978 (26.3%)
- Sad: 3,982 (26.4%)
- Surprise: 3,173 (21.0%)

Consider using:
- Weighted loss function
- Oversampling of minority class (surprise)
- Class weights in CrossEntropyLoss

## Code Organization

Recommended structure:
- `model.py` - CNN architecture definition
- `dataset.py` - Custom Dataset class and data loading
- `train.py` - Training loop and validation
- `evaluate.py` - Test set evaluation
- `predict.py` - Inference on new images
- `utils.py` - Helper functions (visualization, metrics)
- `config.py` - Hyperparameters and configuration

## Hyperparameters to Tune

- Learning rate: Start with 0.001
- Batch size: 32-128 depending on GPU memory
- Number of epochs: 20-50 with early stopping
- Image size: 48x48, 64x64, or 224x224
- Dropout rate: 0.3-0.5
- Weight decay: 1e-4 to 1e-5
