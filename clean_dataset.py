import os
import tensorflow as tf

# Folders to check
folders_to_check = [
    "dataset/train/fire", "dataset/train/non_fire", "dataset/train/smoke",
    "dataset/test/fire", "dataset/test/non_fire", "dataset/test/smoke",
]

removed_count = 0

for folder in folders_to_check:
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            # Read the raw file bytes
            raw = tf.io.read_file(filepath)
            # Try to decode it exactly like TensorFlow does during training
            img = tf.io.decode_image(raw)
        except Exception:
            print(f"Removing broken file: {filepath}")
            os.remove(filepath)
            removed_count += 1

print(f"\nDone! Removed {removed_count} broken files.")