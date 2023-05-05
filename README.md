# MobileNetV3 Implementation using PyTorch

This repository contains an implementation of MobileNetV3 using PyTorch. The implementation includes code for training and evaluating the network on a custom dataset, as well as for fine-tuning on a pre-trained model. The code is modular and easy to modify, allowing for experimentation with different network architectures and hyperparameters.

## Project Structure

```
project_root/
│
├── dataset/
│ ├── train/
│ │ ├── class_1/
│ │ ├── class_2/
│ │ └── ...
│ │
│ ├── val/
│ │ ├── class_1/
│ │ ├── class_2/
│ │ └── ...
│ │
│ └── test/
│ ├── class_1/
│ ├── class_2/
│ └── ...
│
├── models/
│ ├── init.py
│ └── mobilenetv3.py
│
├── utils/
│ ├── init.py
│ ├── data_utils.py
│ ├── train_utils.py
│ ├── tmetrics_utils.py
│ ├── tvisualization_utils.py
│ └── models_utils.py
│
├── checkpoints/
│
└── main.py

```