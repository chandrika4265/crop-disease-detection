image_path = "test.jpg"
import tensorflow as tf
import numpy as np

# Model load
model = tf.keras.models.load_model("Models/crop_disease_model.keras")

# Class names
class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_healthy'
]

# Image load
img = tf.keras.utils.load_img(
    "test.jpg",
    target_size=(128, 128)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# Prediction
prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]

print("Prediction:", predicted_class)
confidence = np.max(prediction) * 100

print("Confidence:", round(confidence, 2), "%")