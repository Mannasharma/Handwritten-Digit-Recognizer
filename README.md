# Digit Recogniser

A simple MNIST digit recognizer using TensorFlow and a Tkinter drawing app.

## Project files

- `train.py` — trains a CNN on the MNIST dataset and saves the model.
- `app.py` — Tkinter GUI for drawing digits and predicting them with the saved model.
- `predict.py` — sample script that loads the saved model and evaluates a test image.
- `requirements.txt` — Python package dependencies.

## Requirements

- Python 3.11+ recommended
- `tensorflow`
- `numpy`
- `Pillow`
- `matplotlib`
- `tkinter` (system package)

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Setup

1. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Train the model

Run the training script to download MNIST, train the CNN, and save the model:

```bash
python train.py
```

This will create `model/digit_classifier.keras`.

## Run the GUI app

Start the drawing app with:

```bash
python app.py
```

Draw a digit in the window and click **Predict**.

## Run the prediction sample

To test the saved model with a sample MNIST image:

```bash
python predict.py
```

## Notes

- The project currently ignores the trained model file in Git because model weights can be large. If you want to share the trained model, use Git LFS or provide it separately.
- `tkinter` is usually installed with Python, but on some systems it must be installed via the OS package manager.
