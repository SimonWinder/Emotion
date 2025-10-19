# Facial Emotion Classification

A deep learning system for classifying facial expressions into four emotion categories (happy, neutral, sad, surprise) using Convolutional Neural Networks (CNN) built with PyTorch.

## Overview

This project implements a custom CNN architecture to recognize emotions from facial images. The model is trained on 15,000+ training images and validated on a separate validation set, achieving high accuracy on the test set.

### Emotion Classes
- **Happy**: Smiling, joyful expressions
- **Neutral**: Calm, expressionless faces
- **Sad**: Unhappy, sorrowful expressions
- **Surprise**: Shocked, astonished expressions

## Project Structure

```
Emotion/
├── config.py              # Configuration and hyperparameters
├── model.py               # CNN architecture definition
├── dataset.py             # Data loading and preprocessing
├── utils.py               # Helper functions and utilities
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── predict.py             # Single image prediction script
├── remove_duplicates.py   # Data cleaning utility
├── requirements.txt       # Python dependencies
├── CLAUDE.md             # Development guidelines
├── PRD.md                # Product requirements
├── TODO.md               # Implementation checklist
└── Facial_emotion_images/ # Dataset directory
    ├── train/            # Training set (15,109 images)
    ├── validation/       # Validation set (4,977 images)
    └── test/             # Test set (128 images)
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support (recommended for faster training):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

**Important**: Always use the `p3` alias instead of `python` or `python3` when running Python code.

### 1. Train the Model

Train the CNN model on the training set with validation:

```bash
p3 train.py
```

Training features:
- Data augmentation (rotation, flipping, color jitter)
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping to prevent overfitting
- Model checkpointing (saves best and last models)
- Real-time progress bars with metrics
- Training history saved to JSON
- Automatic generation of training curves

Expected training time: ~30-60 minutes on GPU, 2-4 hours on CPU

### 2. Evaluate the Model

Evaluate the trained model on the test set:

```bash
p3 evaluate.py
```

This generates:
- Test accuracy and loss
- Per-class accuracy metrics
- Confusion matrix (normalized and counts)
- Classification report (precision, recall, F1-score)

### 3. Make Predictions

Predict emotion from a single image:

```bash
p3 predict.py --image path/to/image.jpg
```

With visualization:
```bash
p3 predict.py --image path/to/image.jpg --visualize
```

Save visualization:
```bash
p3 predict.py --image path/to/image.jpg --visualize --save prediction.png
```

## Model Architecture

### EmotionCNN (Simplified Architecture)

The model uses a lightweight CNN architecture converted from TensorFlow specification:

**Architecture:**
1. **Conv2d(3→16)** → LeakyReLU(0.1) - First convolutional layer
2. **Conv2d(16→32)** → LeakyReLU(0.1) - Second convolutional layer
3. **MaxPool2d(2×2)** - Spatial downsampling (48×48 → 24×24)
4. **Flatten** - Reshape to 1D (18,432 features)
5. **Dense(18432→32)** → LeakyReLU(0.1) - Hidden layer
6. **Dense(32→4)** - Output layer (4 emotion classes)

**Total Parameters:** 595,076 (~595K parameters)

**Input:** 48×48 RGB images (3 channels)

### Key Features
- **LeakyReLU activation** (negative_slope=0.1) prevents dying ReLU problem
- **No BatchNorm or Dropout** - simplified architecture
- **Single pooling layer** - efficient feature reduction
- **Kaiming initialization** optimized for LeakyReLU
- **Cross-entropy loss** with class weights for imbalance handling
- **Adam optimizer** with learning rate scheduling (ReduceLROnPlateau)

## Configuration

Key hyperparameters can be modified in `config.py`:

```python
# Model configuration
IMAGE_SIZE = 48           # Input image size (48×48)
BATCH_SIZE = 64           # Training batch size
NUM_EPOCHS = 50           # Maximum training epochs
LEARNING_RATE = 0.001     # Initial learning rate
CONV_CHANNELS = [16, 32]  # Convolutional layer filters
FC_HIDDEN_SIZE = 32       # Hidden layer size

# Training features
USE_CLASS_WEIGHTS = True  # Handle class imbalance
USE_SCHEDULER = True      # Learning rate scheduling
USE_EARLY_STOPPING = True # Stop if no improvement
```

## Dataset Statistics

### Training Set (15,109 images)
- Happy: 3,976 (26.3%)
- Neutral: 3,978 (26.3%)
- Sad: 3,982 (26.4%)
- Surprise: 3,173 (21.0%)

### Validation Set (4,977 images)
- Happy: 1,825 (36.7%)
- Neutral: 1,216 (24.4%)
- Sad: 1,139 (22.9%)
- Surprise: 797 (16.0%)

### Test Set (128 images)
- 32 images per class (balanced)

## Data Augmentation

Training images undergo the following augmentations:
- Random horizontal flip (50% probability)
- Random rotation (±15 degrees)
- Color jitter (brightness, contrast, saturation, hue)
- Random resized crop (scale: 0.8-1.0)
- Normalization (ImageNet statistics)

## Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall and per-class
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visualization of prediction patterns

Target: **≥85% accuracy** on test set

## Training Tips

### If accuracy is low:
1. Train for more epochs
2. Try transfer learning (ResNet, MobileNet)
3. Adjust learning rate (try 0.0001 or 0.01)
4. Increase/decrease batch size
5. Add more data augmentation
6. Adjust dropout rate

### If overfitting occurs:
1. Increase dropout rate
2. Add more data augmentation
3. Use early stopping
4. Reduce model capacity
5. Add L2 regularization (weight decay)

### If training is slow:
1. Use GPU acceleration
2. Increase batch size (if memory allows)
3. Reduce image size
4. Use transfer learning with frozen layers

## Output Files

### Models
- `models/best_model.pth` - Best model based on validation accuracy
- `models/last_model.pth` - Model from last training epoch
- `models/training_history.json` - Training metrics history

### Results
- `results/confusion_matrix.png` - Normalized confusion matrix
- `results/confusion_matrix_counts.png` - Raw count confusion matrix
- `results/training_curves.png` - Loss and accuracy plots
- `results/classification_report.txt` - Detailed metrics report

## Troubleshooting

### Import errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### CUDA out of memory
Reduce batch size in `config.py`:
```python
BATCH_SIZE = 32  # or 16
```

### Dataset not found
Ensure the `Facial_emotion_images` directory exists with train/validation/test subdirectories.

### Model not found when predicting
Train the model first:
```bash
p3 train.py
```

## Development

### Testing Individual Modules

Test data loading:
```bash
p3 dataset.py
```

Test model architecture:
```bash
p3 model.py
```

Test utility functions:
```bash
p3 utils.py
```

View configuration:
```bash
p3 config.py
```

## Future Enhancements

- Real-time webcam emotion detection
- Multi-face detection and classification
- Additional emotion categories (anger, fear, disgust)
- Web API for model serving
- Mobile deployment with model quantization
- Ensemble methods for improved accuracy
- Transfer learning with modern architectures

## License

This project is for educational and research purposes.

## Acknowledgments

- Dataset: Facial Emotion Images
- Framework: PyTorch
- Visualization: Matplotlib, Seaborn

## Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.

---

**Built with PyTorch** | **Deep Learning** | **Computer Vision**
