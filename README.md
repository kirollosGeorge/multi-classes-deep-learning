# Multi-Class Weather Classification Using Deep Learning Architectures

![License](https://img.shields.io/badge/license-Educational-blue.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Stars](https://img.shields.io/github/stars/<username>/<repository-name>?style=social)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)

A deep learning image classification project for identifying weather conditions across five classes: `cloudy`, `foggy`, `rainy`, `shine`, and `sunrise`.

The repository includes preprocessing scripts, model training notebooks, transfer learning experiments, evaluation charts, confusion matrix utilities, and test image preprocessing outputs.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architectures](#model-architectures)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [API / Config Notes](#api--config-notes)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

This project explores multi-class weather image classification using deep learning. It compares a custom CNN against several pretrained architectures through transfer learning. The dataset contains images representing different weather conditions, and the pipeline includes image resizing, label encoding, dataset shuffling, train/validation splitting, augmentation, training, and evaluation.

The project was designed as an academic and experimental implementation for understanding how different deep learning architectures perform on a real-world image classification task.

## Features

- Multi-class image classification for weather conditions.
- Five supported labels: `cloudy`, `foggy`, `rainy`, `shine`, and `sunrise`.
- Dataset distribution analysis.
- Automated train/validation split.
- Image preprocessing with `256 x 256` target size.
- CNN baseline model.
- CNN with Data Augmentation.
- Transfer Learning models.
- Accuracy and loss curve images.
- Confusion matrix visualization helper.
- Classification reports using `scikit-learn`.
- Test image preprocessing to `.npy` arrays.

## Model Architectures

The repository includes experiments for:

| Model | Type | Notes |
| --- | --- | --- |
| Custom CNN | Baseline | Implemented with Keras convolution, pooling, flatten, and dense layers. |
| CNN + Augmentation | Baseline + Augmentation | Uses `ImageDataGenerator` for training augmentation. |
| ResNet50 | Transfer Learning | Uses pretrained ResNet architecture with custom dense layers. |
| ResNet101 | Transfer Learning | Used as one of the strongest benchmarked models. |
| ResNet152 | Transfer Learning | Deeper ResNet experiment. |
| ResNet18 | FastAI Experiment | Included as a separate notebook experiment. |
| VGG16 | Transfer Learning | Uses dense layers, dropout, and batch normalization. |
| VGG19 | Transfer Learning | Extended VGG experiment. |
| InceptionV3 | Transfer Learning | Inception-based experiment. |
| EfficientNetB0 | Transfer Learning | EfficientNet-based experiment. |
| Xception | Transfer Learning | Xception-based experiment. |

## Tech Stack

- Python 3.8+
- Jupyter Notebook
- TensorFlow
- Keras
- FastAI
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- OpenCV

## Quick Start

```bash
# Clone repository
git clone https://github.com/<username>/<repository-name>.git
cd <repository-name>

# Create virtual environment
python -m venv .venv

# Activate environment
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python fastai jupyter

# Run preprocessing
python "data Inhanced.py"

# Start notebooks
jupyter notebook
```

## Dataset Setup

The dataset source is listed in `dataset.txt`.

Expected local structure:

```text
dataset/
├── cloudy/
├── foggy/
├── rainy/
├── shine/
├── sunrise/
├── alien_test/
└── test.csv
```

After running preprocessing, the project creates:

```text
weather_pred/
└── Data/
    ├── training/
    │   ├── cloudy/
    │   ├── foggy/
    │   ├── rainy/
    │   ├── shine/
    │   └── sunrise/
    └── validation/
        ├── cloudy/
        ├── foggy/
        ├── rainy/
        ├── shine/
        └── sunrise/
```

## Usage

### 1. Prepare the dataset

Place the dataset in the root-level `dataset/` directory.

### 2. Run preprocessing

```bash
python "data Inhanced.py"
```

This script:

- Reads images from the class folders.
- Displays class distribution charts.
- Creates `weather_pred/Data/training` and `weather_pred/Data/validation`.
- Splits the dataset using an `85%` training and `15%` validation ratio.

### 3. Train models

Open Jupyter Notebook:

```bash
jupyter notebook
```

Then run any notebook inside `Jupyter Scripts/`, such as:

```text
Jupyter Scripts/Cnn Model With Augmentation.ipynb
Jupyter Scripts/Res-net101 and 152.ipynb
Jupyter Scripts/VGG.ipynb
Jupyter Scripts/efficientnet.ipynb
Jupyter Scripts/xception.ipynb
```

### 4. Preprocess test images

```bash
python TestPreprocessing.py
```

The output file will be saved as:

```text
test_preproc_CNN.npy
```

## Screenshots

Add screenshots or result previews here.

```text
screenshots/
├── accuracy-curve.png
├── loss-curve.png
├── confusion-matrix.png
└── sample-predictions.png
```

Example Markdown placeholder:

```md
![Accuracy Curve](screenshots/accuracy-curve.png)
![Loss Curve](screenshots/loss-curve.png)
![Confusion Matrix](screenshots/confusion-matrix.png)
```

## API / Config Notes

This project does not expose a REST API by default. It is notebook-based and script-based.

Recommended configurable values:

```env
DATASET_DIR=dataset
TRAIN_DIR=weather_pred/Data/training
VALIDATION_DIR=weather_pred/Data/validation
TEST_DIR=dataset/alien_test
IMAGE_SIZE=256
BATCH_SIZE=32
EPOCHS=50
```

To use these variables, update scripts to read from environment variables:

```python
import os

DATASET_DIR = os.getenv("DATASET_DIR", "dataset")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "256"))
```

Recommended future configuration files:

```text
.env
requirements.txt
config.yaml
```

## Project Structure

```text
multi-classes-deep-learning-main/
├── Jupyter Scripts/
│   ├── Cnn Model With Augmentation.ipynb
│   ├── Cnn Model Without  Augmentation.ipynb
│   ├── Inceptionv3.ipynb
│   ├── Res-net101 and 152.ipynb
│   ├── Res-net18.ipynb
│   ├── Res_net-50.ipynb
│   ├── VGG.ipynb
│   ├── efficientnet.ipynb
│   ├── xception.ipynb
│   ├── loads.ipynb
│   ├── preprocessing of res-net.ipynb
│   ├── test_cnn_with_aug.ipynb
│   ├── test_with_resnet50.ipynb
│   ├── mymodule.py
│   ├── Accuracy_curve_*.jpg
│   └── Loss_curve_*.jpg
├── Project documentation.docx
├── Presentation.pptx
├── README.md
├── TestPreprocessing.py
├── Untitled1.ipynb
├── data Inhanced.py
├── dataset.txt
├── test_preproc_CNN.npy
└── test_preproc_res-net.npy
```

## Troubleshooting

### `FileNotFoundError: dataset/...`

Make sure the dataset exists at the project root and follows the expected folder structure.

```text
dataset/cloudy
dataset/foggy
dataset/rainy
dataset/shine
dataset/sunrise
```

### `ModuleNotFoundError`

Install the required packages:

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python fastai jupyter
```

### GPU is not detected

Check TensorFlow GPU availability:

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```

If no GPU appears, verify your CUDA, cuDNN, and TensorFlow version compatibility.

### Confusion matrix helper error

If `Jupyter Scripts/mymodule.py` raises a NumPy-related error, make sure `numpy` is imported:

```python
import numpy as np
```

### Paths with spaces

Some filenames contain spaces, such as:

```text
Jupyter Scripts/Cnn Model With Augmentation.ipynb
data Inhanced.py
```

Use quotes when running scripts from the terminal:

```bash
python "data Inhanced.py"
```

## Roadmap

- Add `requirements.txt`.
- Add `LICENSE` file.
- Standardize file and notebook naming.
- Convert notebook logic into reusable Python modules.
- Add a CLI training entry point.
- Add model checkpoint and result directories.
- Add reproducible seed configuration.
- Add GitHub Actions for linting and notebook validation.
- Add a prediction script for single image inference.
- Add an inference API using FastAPI or Flask.

## Contributing

Contributions are welcome.

```bash
# Create a branch
git checkout -b feature/your-feature-name

# Commit changes
git commit -m "Add your meaningful commit message"

# Push branch
git push origin feature/your-feature-name
```

Then open a Pull Request with a clear description of the changes.

Please keep contributions focused, documented, and reproducible.

## License

This project is currently documented as an educational/research project. Add a `LICENSE` file to define the exact usage and redistribution terms.

## Acknowledgements

- Dataset reference listed in `dataset.txt`.
- TensorFlow and Keras for deep learning model development.
- FastAI for ResNet18 experimentation.
- scikit-learn for evaluation metrics.
- Matplotlib and Seaborn for visualization.
