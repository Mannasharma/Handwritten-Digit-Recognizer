import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("model/digit_classifier.keras")

# Load dataset
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_test = x_test / 255.0

# Reshape
x_test = x_test.reshape(-1,28,28,1)

# Select an image
image = x_test[21]

# Predict
prediction = model.predict(image.reshape(1,28,28,1))

predicted_digit = np.argmax(prediction)

print("Predicted:", predicted_digit)
print("Actual:", y_test[21])

plt.imshow(image.reshape(28,28), cmap="gray")
plt.title(f"Prediction: {predicted_digit}")
plt.axis("off")
plt.show()