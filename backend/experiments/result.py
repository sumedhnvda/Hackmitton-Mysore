# test_colorize.py
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize
from skimage.io import imsave
from skimage.color import rgb2lab, lab2rgb

# ---------------------- Configuration ----------------------
MODEL_PATH = "model_files/best_colorize_autoencoder.keras"
TEST_IMAGE = "docs/image-10.jpg"     # Path to your test image
RESULT_PATH = "result.png"           # Where to save colorized image
IMG_SIZE = 256                        # Same size as training images

# ---------------------- Load Model ----------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=True)
print(f"Loaded model from {MODEL_PATH}")

# ---------------------- Load & Preprocess Image ----------------------
if not os.path.exists(TEST_IMAGE):
    raise FileNotFoundError(f"Test image not found at {TEST_IMAGE}")

img = img_to_array(load_img(TEST_IMAGE)) / 255.0
img = resize(img, (IMG_SIZE, IMG_SIZE))
img_lab = rgb2lab(img)
img_l = img_lab[:, :, 0].reshape(1, IMG_SIZE, IMG_SIZE, 1)  # L channel

# ---------------------- Predict AB channels ----------------------
output_ab = model.predict(img_l)
output_ab = output_ab * 128  # scale back to original AB range

# ---------------------- Reconstruct & Save ----------------------
result_lab = np.zeros((IMG_SIZE, IMG_SIZE, 3))
result_lab[:, :, 0] = img_l[0][:, :, 0]
result_lab[:, :, 1:] = output_ab[0]

result_rgb = lab2rgb(result_lab)
imsave(RESULT_PATH, (result_rgb * 255).astype(np.uint8))
print(f"Colorized result saved as {RESULT_PATH}")
