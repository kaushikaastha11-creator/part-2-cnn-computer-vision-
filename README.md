# 🖼️ Part 2: Computer Vision – CNN Prototype for Manufacturing Defect Detection

A CNN-based image classifier that detects surface defects on manufactured products.

---

## 📌 Problem Statement

Given images of product surfaces, classify each image into one of four categories:

| Class | Description |
|---|---|
| `normal` | Clean surface — no defect |
| `scratch` | Linear scratch marks |
| `dent` | Circular depression marks |
| `stain` | Coloured blemish or stain |

**Problem Type: Image Classification (Multi-Class)**

---

## 📁 Dataset

- **Source:** Synthetic Manufacturing Defect Image Dataset
- **Total Images:** 480 (120 per class — perfectly balanced)
- **Image Size:** 96×96 pixels, RGB
- **Format:** PNG, organized in class-named subfolders

```
images/
├── normal/     (120 images)
├── scratch/    (120 images)
├── dent/       (120 images)
└── stain/      (120 images)
```

---

## 🏗️ Model Architecture

```
Input (64×64×3)
    ↓
Conv2D(32) → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Flatten
    ↓
Dense(128, ReLU) → Dropout(0.4) → Dense(64, ReLU)
    ↓
Dense(4, Softmax)  ← Output: probability for each class
```

**Compiled with:**
- Optimizer: Adam (lr=0.001)
- Loss: Sparse Categorical Crossentropy
- Metric: Accuracy

---

## 🔬 CNN Concepts Explained

### What is Convolution?
A small filter (e.g., 3×3 grid of numbers) slides across the image. At each position, it multiplies with the pixels underneath and sums the result. Different filters detect different features — edges, corners, textures. Deeper layers combine these into complex patterns like "circular dent" or "linear scratch".

### Why is Pooling Used?
Max Pooling shrinks the feature map (e.g., 64×64 → 32×32) by keeping only the maximum value in each 2×2 region. This reduces computation cost, makes the model less sensitive to exact position (translation invariance), and helps prevent overfitting.

### Why is ReLU Used?
ReLU (Rectified Linear Unit): `f(x) = max(0, x)`. It is:
- Fast to compute
- Solves the vanishing gradient problem (gradients stay large, so learning is fast)
- Introduces non-linearity so the network can learn complex patterns
- Creates sparse activations (many zeros), making the model efficient

### Why CNNs Beat Regular Networks for Images?

| Feature | Dense Network | CNN |
|---|---|---|
| Parameters | Millions (every pixel → every neuron) | Thousands (filters shared across image) |
| Spatial awareness | None | Understands 2D structure |
| Translation invariance | No | Yes |
| Feature hierarchy | Flat | Edges → Shapes → Objects |

---

## 🏭 Business Use Case: Manufacturing Quality Control

**Domain:** Industrial Manufacturing  

**The Problem:** Manual visual inspection of products on assembly lines is slow, expensive, and inconsistent due to human fatigue.

**The Solution:** A CNN deployed as a real-time quality inspection system:
1. Camera captures product image as it passes on the conveyor
2. CNN classifies: `normal` / `scratch` / `dent` / `stain` in milliseconds
3. Defective items are auto-rejected off the line
4. Data is logged for trend analysis and predictive maintenance

**Business Benefits:**
- ✅ Inspects hundreds of units/minute (vs. a few per second manually)
- ✅ Consistent accuracy — no fatigue, no shift variation
- ✅ Reduces warranty claims and product returns
- ✅ Defect trend data enables root-cause analysis

**Real-world examples:** BMW car body inspection, Foxconn display panel QC, Samsung semiconductor wafer inspection.

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended for Beginners)
1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload `Part2_CNN_Computer_Vision.ipynb`
3. Run cells top to bottom with `Shift + Enter`
4. Upload the dataset ZIP when prompted

### Option 2: Local Setup
```bash
pip install -r requirements.txt
jupyter notebook Part2_CNN_Computer_Vision.ipynb
```

---

## 📦 Requirements

See `requirements.txt`

---

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | ~85–92% (varies per run) |
| Classes | 4 (normal, scratch, dent, stain) |
| Dataset balance | Perfectly balanced (120 each) |

---

## 📂 Repository Structure

```
part-2-cnn-computer-vision/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
├── sample_predictions/
│   └── prediction_outputs.png
└── results/
    ├── accuracy_loss_curves.png
    └── confusion_matrix.png
```
