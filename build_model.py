import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (128, 128)

model = models.Sequential([
    # Layer 1: Rescaling
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),

    # Layer 2: First Convolutional layer
    layers.Conv2D(32, (3, 3), activation='relu'),

    # Layer 3: Pooling layer — shrinks the feature maps
    layers.MaxPooling2D((2, 2)),

    # Layer 4: Second Convolutional layer (more filters = more complex patterns)
    layers.Conv2D(64, (3, 3), activation='relu'),

    # Layer 5: Second Pooling layer
    layers.MaxPooling2D((2, 2)),

    # Layer 6: Flatten — convert 3D feature maps into a 1D list
    layers.Flatten(),

    # Layer 7: Hidden Dense layer — combines detected features
    layers.Dense(128, activation='relu'),

    # Layer 8: Output layer — 3 classes, softmax gives probabilities
    layers.Dense(3, activation='softmax'),

])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled successfully!")