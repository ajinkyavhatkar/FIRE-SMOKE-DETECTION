import tensorflow as tf
import numpy as np
import os

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Load the saved model (no need to rebuild or retrain it)
model = tf.keras.models.load_model("models/fire_smoke_model.keras")

# Load the test dataset (completely unseen data)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False   # keep order consistent, not required to shuffle for evaluation
)

class_names = test_ds.class_names
print("Classes:", class_names)

# Evaluate overall performance on the test set
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"\nTest Accuracy: {test_accuracy:.2%}")
print(f"Test Loss: {test_loss:.4f}")

# --- Predict a single image ---

def predict_image(img_path):
    # Load and resize the image to match what our model expects
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    
    # Convert image to a numpy array (numbers) and add a "batch" dimension
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # model expects a batch, even for 1 image
    
    # Get prediction probabilities
    predictions = model.predict(img_array)
    
    # Find which class had the highest probability
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    
    print(f"\nImage: {img_path}")
    print(f"Prediction: {predicted_class} ({confidence:.2f}% confidence)")

# Test it on one image from our test folder (pick any real file path from your dataset)
predict_image("dataset/test/fire/" + os.listdir("dataset/test/fire")[0])