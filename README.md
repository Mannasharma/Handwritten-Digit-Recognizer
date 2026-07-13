# Digit Recogniser

A simple handwritten digit recognition project using TensorFlow, MNIST, and a Tkinter drawing app.

## Overview

This project includes:

- `train.py` — trains a convolutional neural network (CNN) on the MNIST dataset and saves the trained model.
- `app.py` — a Tkinter GUI where you can draw a digit and get a live prediction.
- `predict.py` — a sample script that loads the saved model and predicts a digit from the MNIST test set.
- `requirements.txt` — Python dependencies required to run the project.

## Features

- MNIST digit classification using a CNN
- Local model training with TensorFlow
- Simple drawing-based GUI for digit prediction
- Sample prediction script for validation

## Prerequisites

- Python 3.11+ recommended
- `tkinter` installed on your system

## Dependencies

Install Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

The project requires:

- `tensorflow`
- `numpy`
- `Pillow`
- `matplotlib`

## Setup

1. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Training

Train the model using:

```bash
python train.py
```

This downloads the MNIST dataset, trains the CNN, evaluates it, and saves the model to `model/digit_classifier.keras`.

## Run the GUI App

Start the app with:

```bash
python app.py
```

Use the drawing canvas, write a digit, and click **Predict** to see the model result.

## Run the prediction sample

Test the saved model on a sample MNIST image:

```bash
python predict.py
```

## Notes

- The project ignores the `venv/` folder and the trained model file in Git to keep the repository clean.
- If you want to share the trained model, use Git LFS or provide it as a separate download.
- No Jupyter Notebook is required for this project.

## Project Structure

```text
├── app.py
├── predict.py
├── requirements.txt
├── train.py
├── .gitignore
├── model/                  # saved TensorFlow model if created locally
└── venv/                   # ignored virtual environment
```

## Ready to upload

The project is ready for GitHub upload. The current setup is complete and the README includes all necessary instructions.


