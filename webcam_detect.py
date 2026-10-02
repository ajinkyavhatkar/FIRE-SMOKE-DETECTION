import cv2
import tensorflow as tf
import numpy as np

IMG_SIZE = (128, 128)
class_names = ['fire', 'non_fire', 'smoke']  # same order Keras used (alphabetical)

# Load our trained model
model = tf.keras.models.load_model("models/fire_smoke_model.keras")

# Start the webcam (0 = default webcam)
cap = cv2.VideoCapture("http://192.168.12.181:8080/video")

print("Starting webcam... Press 'q' to quit.")

while True:
    # Read one frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert BGR (OpenCV format) to RGB (model's expected format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to match model's expected input size
    resized = cv2.resize(rgb_frame, IMG_SIZE)

    # Prepare the image for prediction (same steps as before)
    img_array = tf.expand_dims(resized, 0)  # add batch dimension

    # Get prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    # Prepare text to display
    label = f"{predicted_class} ({confidence:.1f}%)"

    # Draw the label text onto the original frame (BGR, for correct display colors)
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame in a window
    cv2.imshow("Fire/Smoke Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()