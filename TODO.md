# TODO: Facial Emotion Classification Implementation

## Phase 1: Project Setup ✓
- [x] Explore dataset structure
- [x] Create CLAUDE.md documentation
- [x] Create PRD.md with requirements
- [x] Create TODO.md with implementation plan
- [ ] Create requirements.txt with dependencies
- [ ] Initialize git repository (optional)

## Phase 1.5: Data Cleaning

### 1.5.1 Remove Duplicate Images (`remove_duplicates.py`)
- [ ] Create script to find and remove duplicate images using file hashing
- [ ] Implement MD5 or SHA256 hash calculation for each image
- [ ] Scan training dataset and identify duplicates within each emotion class
- [ ] Scan validation dataset and identify duplicates
- [ ] Scan test dataset and identify duplicates
- [ ] Remove duplicate files, keeping only the first occurrence
- [ ] Generate report showing:
  - [ ] Number of duplicates found per class
  - [ ] Total duplicates removed
  - [ ] Updated dataset statistics
- [ ] Backup original data before removing duplicates (optional but recommended)
- [ ] Run duplicate removal on all three splits (train/validation/test)

**Command:**
```bash
p3 remove_duplicates.py
```

## Phase 2: Data Pipeline

### 2.1 Dataset Module (`dataset.py`)
- [ ] Create custom Dataset class for emotion images
- [ ] Implement `__getitem__` to load and transform images
- [ ] Implement `__len__` to return dataset size
- [ ] Define train transformations with data augmentation
  - [ ] Random horizontal flip
  - [ ] Random rotation (±15°)
  - [ ] Color jitter (brightness, contrast, saturation)
  - [ ] Random crop and resize
  - [ ] Convert to tensor
  - [ ] Normalize with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- [ ] Define validation/test transformations (no augmentation)
  - [ ] Resize to target size
  - [ ] Center crop
  - [ ] Convert to tensor
  - [ ] Normalize
- [ ] Create DataLoader instances for train/val/test splits
- [ ] Verify data loading with sample batch

### 2.2 Data Exploration (`explore_data.py`)
- [ ] Display sample images from each emotion class
- [ ] Print dataset statistics (count per class)
- [ ] Visualize sample augmented images
- [ ] Calculate mean and std of dataset (if using custom normalization)

## Phase 3: Model Architecture

### 3.1 CNN Model (`model.py`)
- [ ] Define EmotionCNN class inheriting from nn.Module
- [ ] Implement `__init__` with model architecture
  - [ ] Convolutional Block 1: Conv2d(3, 64) → BatchNorm → ReLU → MaxPool
  - [ ] Convolutional Block 2: Conv2d(64, 128) → BatchNorm → ReLU → MaxPool
  - [ ] Convolutional Block 3: Conv2d(128, 256) → BatchNorm → ReLU → MaxPool
  - [ ] Convolutional Block 4: Conv2d(256, 512) → BatchNorm → ReLU → MaxPool (optional)
  - [ ] Flatten layer
  - [ ] Fully connected layer 1: Linear → ReLU → Dropout
  - [ ] Fully connected layer 2: Linear → ReLU → Dropout
  - [ ] Output layer: Linear(hidden_size, 4)
- [ ] Implement `forward` method
- [ ] Add method to count total parameters
- [ ] Test model with dummy input to verify output shape

### 3.2 Alternative: Transfer Learning Model (`model_transfer.py`)
- [ ] Load pre-trained ResNet18 or MobileNetV2
- [ ] Replace final classification layer for 4 classes
- [ ] Optionally freeze early layers
- [ ] Test forward pass

## Phase 4: Training Pipeline

### 4.1 Configuration (`config.py`)
- [ ] Define all hyperparameters
  - [ ] Batch size, learning rate, epochs
  - [ ] Image size, dropout rate
  - [ ] Device (CPU/CUDA)
  - [ ] Data paths
  - [ ] Model save path
- [ ] Add command-line argument parsing (optional)

### 4.2 Training Script (`train.py`)
- [ ] Import required modules
- [ ] Load configuration
- [ ] Set random seeds for reproducibility
- [ ] Initialize datasets and dataloaders
- [ ] Calculate class weights for imbalanced dataset
- [ ] Initialize model
- [ ] Define loss function (CrossEntropyLoss with weights)
- [ ] Define optimizer (Adam or SGD)
- [ ] Define learning rate scheduler (ReduceLROnPlateau)
- [ ] Implement training loop
  - [ ] Iterate through epochs
  - [ ] Training phase: forward pass, loss calculation, backward pass
  - [ ] Validation phase: evaluate on validation set
  - [ ] Track metrics (loss, accuracy)
  - [ ] Print progress
  - [ ] Save best model checkpoint
  - [ ] Early stopping if no improvement
- [ ] Save training history (loss/accuracy curves)
- [ ] Plot training curves

### 4.3 Utilities (`utils.py`)
- [ ] Function to calculate accuracy
- [ ] Function to save checkpoint
- [ ] Function to load checkpoint
- [ ] Function to plot training curves
- [ ] Function to display sample predictions
- [ ] Class to track running metrics (EarlyStopping, MetricTracker)

## Phase 5: Evaluation

### 5.1 Evaluation Script (`evaluate.py`)
- [ ] Load trained model
- [ ] Load test dataset
- [ ] Run inference on test set
- [ ] Calculate overall accuracy
- [ ] Calculate per-class metrics (precision, recall, F1)
- [ ] Generate confusion matrix
- [ ] Plot confusion matrix heatmap
- [ ] Save classification report
- [ ] Display misclassified examples

### 5.2 Inference Script (`predict.py`)
- [ ] Load trained model
- [ ] Accept image path as input
- [ ] Load and preprocess image
- [ ] Run inference
- [ ] Display prediction with confidence scores
- [ ] Optionally display image with prediction overlay

## Phase 6: Optimization & Tuning

- [ ] Experiment with different learning rates
- [ ] Experiment with different batch sizes
- [ ] Try different image sizes (48x48, 64x64, 128x128, 224x224)
- [ ] Adjust dropout rates
- [ ] Try different optimizers (Adam, SGD, AdamW)
- [ ] Experiment with learning rate schedules
- [ ] Try transfer learning approach
- [ ] Implement gradient clipping if needed
- [ ] Add mixup or cutmix augmentation (advanced)

## Phase 7: Documentation & Cleanup

### 7.1 Documentation
- [ ] Create comprehensive README.md
  - [ ] Project description
  - [ ] Installation instructions
  - [ ] Usage examples
  - [ ] Results and performance metrics
- [ ] Add docstrings to all functions/classes
- [ ] Add type hints throughout code
- [ ] Comment complex code sections

### 7.2 Results Documentation
- [ ] Save final test accuracy and metrics
- [ ] Include confusion matrix visualization
- [ ] Include training curve plots
- [ ] Document best hyperparameters found
- [ ] Save sample predictions

### 7.3 Code Quality
- [ ] Remove debug print statements
- [ ] Ensure consistent code style
- [ ] Remove unused imports
- [ ] Add error handling for edge cases
- [ ] Test all scripts independently

## Phase 8: Deliverables Checklist

- [ ] All source code files created and tested
- [ ] `best_model.pth` saved
- [ ] Training history saved
- [ ] Confusion matrix generated
- [ ] Classification report generated
- [ ] Training curves plotted
- [ ] README.md complete
- [ ] All acceptance criteria met (≥85% accuracy)

## Notes

- Use `p3` alias instead of `python3` for all Python commands
- Always validate on validation set before testing on test set
- Save checkpoints frequently during training
- Monitor for overfitting (validation loss increasing while training loss decreases)
- If accuracy is low, try transfer learning or adjust hyperparameters

## Quick Start Commands

```bash
# Install dependencies
pip install torch torchvision numpy matplotlib pillow scikit-learn

# Explore the data
p3 explore_data.py

# Train the model
p3 train.py

# Evaluate on test set
p3 evaluate.py

# Make prediction on single image
p3 predict.py --image path/to/image.jpg
```
