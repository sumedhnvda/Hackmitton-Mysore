# train_colorize_small.py
import os
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint
from skimage.color import rgb2lab
from skimage.transform import resize

# ---------------------- Configuration ----------------------
DATA_PATH = "docs/"           # folder containing image-1.jpg to image-2000.jpg
IMG_SIZE = 256                # can reduce to 128 if GPU memory is low
BATCH_SIZE = 4                # small batch size to avoid memory issues
EPOCHS = 200                  # max epochs, early stopping will cut short
MAX_IMAGES = 1500              # limit dataset to 200 images
MODEL_DIR = "model_files"
os.makedirs(MODEL_DIR, exist_ok=True)

# Enable mixed precision (optional, helps with memory)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ---------------------- Load Dataset ----------------------
X, Y = [], []
print(f"Loading first {MAX_IMAGES} images from dataset...")

count = 0
for i in range(1, MAX_IMAGES + 1):
    filename = os.path.join(DATA_PATH, f"image-{i}.jpg")
    if not os.path.exists(filename):
        print(f"Skipping {filename} (not found)")
        continue
    try:
        img = load_img(filename)
        img = img_to_array(img) / 255.0
        img = resize(img, (IMG_SIZE, IMG_SIZE))
        lab = rgb2lab(img)
        X.append(lab[:, :, 0])           # L channel
        Y.append(lab[:, :, 1:] / 128)    # AB channels scaled [-1,1]
        count += 1
    except Exception as e:
        print(f"Error processing {filename}: {e}")

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = np.array(Y)
print(f"Dataset prepared: X shape={X.shape}, Y shape={Y.shape}, images loaded={count}")

# ---------------------- Define Model ----------------------
model = Sequential()

# Encoder
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2, input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

# Decoder
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()

# ---------------------- Callbacks ----------------------
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "best_colorize_autoencoder.keras"),
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# ---------------------- Train Model ----------------------
history = model.fit(
    X, Y,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint]
)

# Save final model
model.save(os.path.join(MODEL_DIR, "colorize_autoencoder_final.keras"))
print("Training completed!")
