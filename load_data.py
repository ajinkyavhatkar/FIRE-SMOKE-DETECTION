import tensorflow as tf

# Settings we'll reuse throughout the project
IMG_SIZE = (128, 128)   # resize every image to 128x128 pixels
BATCH_SIZE = 32         # how many images the model looks at per training step

# Load training data (80% of the 'train' folder)
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    validation_split=0.2,   # reserve 20% of train images for validation
    subset="training",
    seed=123,                # fixes randomness so results are reproducible
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load validation data (the remaining 20% of the 'train' folder)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Print the class names Keras detected automatically from folder names
print("Classes found:", train_ds.class_names)

import matplotlib.pyplot as plt

# Take one batch of images from the training dataset and display a few
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):   # take() grabs just 1 batch (32 images)
    for i in range(9):                     # show first 9 images from that batch
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))  # convert image data back to normal pixel format for display
        plt.title(train_ds.class_names[labels[i]])       # show the correct label as the image title
        plt.axis("off")
plt.show()