# 🦋 BUTTERFLY
## Input-Level Backdoor Detection for Pretrained Contrastive Vision Encoders via Reference-Frame Transformations

This repository contains the code and selected artifacts for **BUTTERFLY**, an input-level backdoor detector for pretrained vision encoders.

---

# 🌟 Quick Start

## 1. Environment Setup
Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate butterfly_env
```

---

## 2. Data Preparation
Prepare the CIFAR-10 directory structure expected by the code:

```bash
python prepare_cifar10.py
```

---

# 🚀 Main Pipeline

## 3. Train the BUTTERFLY Detector
Run the detector construction stage:

```bash
python main.py --attack_type ctrl
```

Supported attack types:

- `ctrl`
- `blto`
- `badencoder`
- `drupe`
- `inactive`

---

## 4. Upstream Evaluation
Evaluate upstream detection performance:

```bash
python upstream_evaluation.py --attack_type ctrl
```

---

## 5. Downstream Evaluation
Run purification and downstream evaluation:

```bash
python train_purification.py --attack_type ctrl
python test_purification.py --attack_type ctrl
python downstream_evaluation.py --attack_type ctrl
```

---

# 🧪 Ablation Studies

## A. Ablation on Reference Frames

### Full path--response frame
```bash
python main.py --attack_type ctrl
python upstream_evaluation.py --attack_type ctrl
```

### Path-only frame
```bash
python main.py --attack_type ctrl --no_amplification
python upstream_evaluation.py --attack_type ctrl --no_amplification
```

### Snapshot--response frame
```bash
python main.py --attack_type ctrl --num_layer_ratio 0.03
python upstream_evaluation.py --attack_type ctrl --num_layer_ratio 0.03
```

### Snapshot-only frame
```bash
python main.py --attack_type ctrl --num_layer_ratio 0.03 --no_amplification
python upstream_evaluation.py --attack_type ctrl --num_layer_ratio 0.03 --no_amplification
```

---

## B. Ablation on Sweep Grid
```bash
python main.py --attack_type ctrl --overlap 0.2
python upstream_evaluation.py --attack_type ctrl --overlap 0.2
```

---

## C. Ablation on Scaling Factor
```bash
python main.py --attack_type ctrl --scale 2.0
python upstream_evaluation.py --attack_type ctrl --scale 2.0
```

---

# 📌 Notes

- Replace `ctrl` with any other supported attack type when needed.
- The main reported SimCLR/CIFAR-10 attacks are:
  - `ctrl`
  - `blto`
  - `badencoder`
  - `drupe`
  - `inactive`
- The provided artifact focuses on the main SimCLR/CIFAR-10 setting.
