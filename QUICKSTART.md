# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
p3 train.py
```
This will:
- Load and preprocess the dataset (15,109 training images)
- Train the CNN model with data augmentation
- Validate after each epoch
- Save the best model automatically
- Generate training curves
- Take approximately 30-60 minutes on GPU

### Step 3: Evaluate Performance
```bash
p3 evaluate.py
```
This will:
- Load the best trained model
- Evaluate on the test set (128 images)
- Generate confusion matrix
- Create classification report
- Show per-class accuracy

## 📊 Expected Results

**Target Accuracy:** ≥85% on test set

### Training Output
```
Epoch 1/50 [Train]: 100%|████████| loss: 1.2345, acc: 45.67%
Epoch 1/50 [Val]:   100%|████████| loss: 1.1234, acc: 52.34%
*** New best model saved! (Val Acc: 52.34%) ***
```

### Evaluation Output
```
Test Loss: 0.3456
Test Accuracy: 89.84%

Per-Class Accuracy:
  happy     : 93.75% (30/32)
  neutral   : 90.62% (29/32)
  sad       : 87.50% (28/32)
  surprise  : 87.50% (28/32)
```

## 🔮 Make Predictions

Predict emotion from any image:

```bash
# Simple prediction
p3 predict.py --image path/to/image.jpg

# With visualization
p3 predict.py --image path/to/image.jpg --visualize

# Save prediction visualization
p3 predict.py --image path/to/image.jpg --visualize --save output.png
```

### Example Output
```
Predicted Emotion: HAPPY
Confidence: 95.23%

All Probabilities:
  happy     : 95.23% ████████████████████████████████████████████████
  neutral   :  2.84% █
  surprise  :  1.45%
  sad       :  0.48%
```

## 📁 What Gets Created

After training and evaluation:

```
Emotion/
├── models/
│   ├── best_model.pth           # Best model checkpoint
│   ├── last_model.pth           # Last epoch checkpoint
│   └── training_history.json   # Training metrics
└── results/
    ├── confusion_matrix.png      # Visualization
    ├── training_curves.png       # Loss/accuracy plots
    └── classification_report.txt # Detailed metrics
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Quick adjustments
IMAGE_SIZE = 64       # Image resolution (64x64)
BATCH_SIZE = 64       # Batch size (reduce if OOM)
NUM_EPOCHS = 50       # Training epochs
LEARNING_RATE = 0.001 # Learning rate
```

## 🐛 Common Issues

### 1. CUDA Out of Memory
**Solution:** Reduce batch size in `config.py`
```python
BATCH_SIZE = 32  # or 16
```

### 2. Import Errors
**Solution:** Reinstall dependencies
```bash
pip install -r requirements.txt
```

### 3. Dataset Not Found
**Solution:** Ensure dataset is extracted
```bash
ls Facial_emotion_images/
# Should show: train/ validation/ test/
```

### 4. Slow Training
**Solutions:**
- Use GPU if available
- Reduce `IMAGE_SIZE` to 48
- Increase `BATCH_SIZE` if memory allows

## 📈 Improving Accuracy

If accuracy < 85%:

1. **Train longer:**
   ```python
   NUM_EPOCHS = 100
   ```

2. **Adjust learning rate:**
   ```python
   LEARNING_RATE = 0.0005  # Lower for fine-tuning
   ```

3. **Try different image size:**
   ```python
   IMAGE_SIZE = 128  # Higher resolution
   ```

4. **Increase model capacity:**
   ```python
   FC_HIDDEN_SIZE = 1024
   ```

## 🔄 Re-training

To train from scratch:
```bash
rm -rf models/ results/
p3 train.py
```

## 📚 Full Documentation

See `README.md` for comprehensive documentation including:
- Detailed architecture explanation
- Advanced configuration options
- Troubleshooting guide
- Future enhancements

## 🎯 Success Checklist

- [ ] Dependencies installed
- [ ] Training completed without errors
- [ ] Best model saved to `models/best_model.pth`
- [ ] Test accuracy ≥85%
- [ ] Confusion matrix generated
- [ ] Classification report saved
- [ ] Predictions working on sample images

## ⏱️ Time Estimates

- **Installation:** 2-5 minutes
- **Training (GPU):** 30-60 minutes
- **Training (CPU):** 2-4 hours
- **Evaluation:** 1-2 minutes
- **Prediction:** <1 second per image

---

**Ready to start?** Run `p3 train.py` now!
