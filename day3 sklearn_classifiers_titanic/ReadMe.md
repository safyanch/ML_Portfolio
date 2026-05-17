---

## Day 5 — PyTorch Neural Network

**Notebook:** `day5_pytorch_fundamentals_titanic.ipynb`

### 🎯 Objective
Build and train a feedforward neural network using PyTorch
on the cleaned Titanic dataset. Compare performance against
Day 3 sklearn classical ML models.

### 🧠 Neural Network Architecture
- Input layer: 15 features
- Hidden layer 1: Linear(15→64) + ReLU + Dropout(0.3)
- Hidden layer 2: Linear(64→32) + ReLU + Dropout(0.2)
- Output layer: Linear(32→1) + Sigmoid

**Total parameters:** 3,105
**Loss function:** Binary Cross Entropy (BCELoss)
**Optimizer:** Adam (lr=0.001)
**Epochs:** 100

### 📊 Results Comparison
| Model | Test Accuracy | Type |
|-------|--------------|------|
| Logistic Regression (Day 3) | 84.9% | Classical ML |
| Random Forest (Day 3) | 82.7% | Ensemble |
| Neural Network (Day 5) | 79.9% | Deep Learning |
| SVM (Day 3) | 67.0% | Classical ML |

### 💡 Key Findings
- Neural network achieved 79.9% — competitive with ensemble methods
- Logistic Regression outperforms deep learning on 891 rows —
  confirms neural networks require large datasets to excel
- Training loss decreased consistently from ~0.62 to ~0.39
  over 100 epochs showing stable learning
- model.eval() mode critical — disables Dropout during prediction

### 📈 Training Curves
![Training Curves](plots/day5_training_curves.png)

### 🛠️ Tech Stack
Python · PyTorch · scikit-learn · pandas · NumPy · Matplotlib

### 📂 Saved Model
| File | Description |
|------|-------------|
| models/titanic_neural_net.pth | Trained PyTorch model weights |