import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# -----------------------------
# Load Trained Model
# -----------------------------
model = tf.keras.models.load_model("model/digit_classifier.keras")



def predict_digit():

    # Resize to MNIST size
    small = image.resize((28, 28))

    # Convert to numpy
    img = np.array(small)

    # Normalize
    img = img / 255.0

    # Add channel and batch dimensions
    img = img.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img, verbose=0)

    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    result_label.config(
    text=f"Prediction: {digit} ({confidence:.2f}%)"
)

# -----------------------------
# Window Setup
# -----------------------------
root = tk.Tk()
root.title("Digit Predictor")
root.geometry("400x450")
root.resizable(False, False)

# -----------------------------
# Drawing Canvas
# -----------------------------
canvas = tk.Canvas(
    root,
    width=280,
    height=280,
    bg="black",
    cursor="cross"
)

canvas.pack(pady=20)
# -----------------------------
# Pillow Image (stores drawing)
# -----------------------------
image = Image.new("L", (280, 280), "black")
draw_image = ImageDraw.Draw(image)

# -----------------------------
# Drawing Logic
# -----------------------------
last_x = None
last_y = None


def start_draw(event):
    global last_x, last_y
    last_x = event.x
    last_y = event.y


def draw(event):
    global last_x, last_y

    # Draw on Tkinter canvas
    canvas.create_line(
        last_x,
        last_y,
        event.x,
        event.y,
        fill="white",
        width=18,
        capstyle=tk.ROUND,
        smooth=True
    )

    # Draw on Pillow image
    draw_image.line(
        (last_x, last_y, event.x, event.y),
        fill="white",
        width=18
    )

    last_x = event.x
    last_y = event.y

canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw)

def save_image():
    image.save("img/digit.png")

    small = image.resize((28, 28))

    plt.imshow(small, cmap="gray")
    plt.title("28x28 Image")
    plt.axis("off")
    plt.show()

def clear_canvas():
    global image, draw_image

    # Clear the Tkinter canvas
    canvas.delete("all")

    # Create a fresh black image
    image = Image.new("L", (280, 280), "black")
    draw_image = ImageDraw.Draw(image)


result_label = tk.Label(
    root,
    text="Prediction: -",
    font=("Arial", 16)
)

result_label.pack()

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

save_btn = tk.Button(
    button_frame,
    text="Predict",
    command=predict_digit,
    width=10
)
save_btn.pack(side=tk.LEFT, padx=10)

clear_btn = tk.Button(
    button_frame,
    text="Clear",
    command=clear_canvas,
    width=10
)
clear_btn.pack(side=tk.LEFT, padx=10)

root.mainloop()