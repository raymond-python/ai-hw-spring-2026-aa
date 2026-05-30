# ai-hw-spring-2026-aa
# MNIST Adversarial Attack Project

## Overview

This project explores the vulnerability of deep learning models to adversarial attacks using the MNIST handwritten digit dataset.

A Convolutional Neural Network (CNN) is trained on MNIST and then evaluated against several adversarial attack methods:

- Fast Gradient Sign Method (FGSM)
- Iterative FGSM / Projected Gradient Descent (I-FGSM / PGD)
- Momentum Iterative FGSM (MI-FGSM)

The objective is to measure how easily a neural network can be fooled by small perturbations that are nearly invisible to humans.

---

## Dataset

### MNIST

- 60,000 training images
- 10,000 testing images
- Image size: 28 × 28 pixels
- Grayscale images
- 10 classes (digits 0-9)

The model is trained on the training dataset and evaluated on the testing dataset.

---

## Model Architecture

A Convolutional Neural Network (CNN) was implemented using PyTorch.

Architecture:

```text
Input (1x28x28)
    ↓
Conv2D
    ↓
ReLU
    ↓
MaxPool
    ↓
Conv2D
    ↓
ReLU
    ↓
MaxPool
    ↓
Fully Connected Layer
    ↓
Output (10 Classes)
```

CNN was selected because it performs very well on image classification tasks and achieves high accuracy on MNIST.

---

## Adversarial Attacks

### 1. FGSM (Fast Gradient Sign Method)

FGSM is a single-step adversarial attack.

The attack computes the gradient of the loss with respect to the input image and slightly modifies the image in the direction that maximizes the loss.

Formula:

x_adv = x + ε · sign(∇xJ(θ,x,y))

Parameters used:

```python
epsilon = 0.3
```

---

### 2. I-FGSM / PGD

Projected Gradient Descent (PGD) repeatedly applies small FGSM updates.

Parameters used:

```python
epsilon = 0.3
alpha = 0.01
iterations = 40
```

PGD is generally stronger than FGSM because it continuously adjusts the perturbation over multiple iterations.

---

### 3. Momentum I-FGSM

Momentum Iterative FGSM accumulates gradient information from previous iterations to improve attack effectiveness.

Parameters used:

```python
epsilon = 0.3
alpha = 0.01
iterations = 40
decay = 1.0
```

This often produces a higher attack success rate than FGSM.

---

## Evaluation Metrics

### Recognition Rate

The percentage of correctly classified images before an attack.

Recognition Rate = Correct Predictions / Total Samples

### Attack Success Rate (ASR)

The percentage of originally correct predictions that become incorrect after an attack.

ASR = Successful Attacks / Originally Correct Predictions

---

## Results

After running:

```bash
python attack.py
```

results are automatically saved to:

```text
results/attack_results.csv
results/attack_results.json
```

Example result format:

| Attack | Recognition Rate | Attack Success Rate |
|----------|----------|----------|
| FGSM | 98.91% | 81.40% |
| I-FGSM / PGD | 98.91% | 99.99% |
| Momentum I-FGSM | 98.91% | 99.86% |

Replace the values above with the actual results generated from your experiment.

---

## How to Run

### Install Dependencies

```bash
pip install torch torchvision pandas
```

### Train Model

```bash
python train.py
```

The trained model will be saved as:

```text
models/mnist_cnn.pth
```

### Run Adversarial Attacks

```bash
python attack.py
```

Attack evaluation results will be saved automatically.

---

## Project Structure

```text
.
├── train.py
├── attack.py
├── model.py
├── models/
│   └── mnist_cnn.pth
├── results/
│   ├── attack_results.csv
│   └── attack_results.json
└── README.md
```

---

## Discussion

The CNN achieves high accuracy on clean MNIST images. However, adversarial attacks can significantly reduce model robustness.

FGSM performs a fast one-step attack, while PGD and Momentum I-FGSM perform iterative attacks that are generally more effective.

The results demonstrate that a model with excellent classification accuracy can still be vulnerable to carefully crafted perturbations.

---

## Key Takeaway

High accuracy does not necessarily mean a model is secure. Adversarial attacks reveal important weaknesses in neural networks and highlight the need for robustness testing before deploying AI systems in real-world applications.
