from ultralytics import YOLO
import os
import cv2

# Load our NEW, better-trained model (D-Fire dataset)
model = YOLO("runs/detect/fire_smoke_dfire/weights/best.pt")

# Pick a real test image from our dataset automatically
test_folder = "yolo_dataset/test/images"
sample_image = os.path.join(test_folder, os.listdir(test_folder)[0])

print(f"Testing on: {sample_image}")

results = model(sample_image)

annotated_image = results[0].plot()
cv2.imwrite("detection_result.jpg", annotated_image)
print("Saved result image as: detection_result.jpg")

print("\nDetections found:")
for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = model.names[class_id]
    print(f"  {class_name}: {confidence:.2%} confidence")

if len(results[0].boxes) == 0:
    print("  No fire/smoke detected in this image.")