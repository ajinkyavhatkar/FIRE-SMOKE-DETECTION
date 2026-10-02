import cv2
import winsound
import time
from ultralytics import YOLO
from email_alert import send_alert_email

# ---- SETTINGS ----
VIDEO_SOURCE = "http://192.168.236.12:8080/video"   # update IP if changed
CONFIDENCE_THRESHOLD = 0.5
MODEL_PATH = "runs/detect/firesmoke_final/weights/best.pt"
EMAIL_COOLDOWN_SECONDS = 120   # only send an alert email at most once every 2 minutes

COLORS = {
    0: (0, 0, 255),      # fire = red
    1: (0, 255, 255),    # other = yellow
    2: (255, 0, 0),      # smoke = blue
}
CLASS_NAMES = {0: "FIRE", 1: "OTHER", 2: "SMOKE"}

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_SOURCE)
print("Starting detection... Press 'q' to quit.")

flash_state = False
last_email_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    h, w = frame.shape[:2]
    frame_area = h * w

    results = model(frame, verbose=False)

    alert_triggered = False
    fire_conf = 0.0
    smoke_conf = 0.0
    fire_area = 0
    smoke_area = 0

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0]) * 100
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_area = (x2 - x1) * (y2 - y1)

        color = COLORS.get(class_id, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES.get(class_id, 'UNK')} {confidence:.0f}%"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if class_id == 0:
            fire_conf = max(fire_conf, confidence)
            fire_area += box_area
            if confidence >= CONFIDENCE_THRESHOLD * 100:
                alert_triggered = True
        elif class_id == 2:
            smoke_conf = max(smoke_conf, confidence)
            smoke_area += box_area
            if confidence >= CONFIDENCE_THRESHOLD * 100:
                alert_triggered = True

    affected_pct = ((fire_area + smoke_area) / frame_area) * 100

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    cv2.putText(frame, "AI FIRE & SMOKE DETECTION SYSTEM", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (w - 210, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    panel_x, panel_y, panel_w, panel_h = 15, h - 110, 220, 95
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 255), 1)

    cv2.putText(frame, f"FIRE:  {fire_conf:.0f}%", (panel_x + 10, panel_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(frame, f"SMOKE: {smoke_conf:.0f}%", (panel_x + 10, panel_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
    cv2.putText(frame, f"AREA AFFECTED: {affected_pct:.1f}%", (panel_x + 10, panel_y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if alert_triggered:
        winsound.Beep(1000, 200)
        flash_state = not flash_state
        if flash_state:
            cv2.rectangle(frame, (0, 46), (w, 85), (0, 0, 255), -1)
            cv2.putText(frame, "!!! FIRE / SMOKE DETECTED - ALERT !!!", (15, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 8)

        current_time = time.time()
        if current_time - last_email_time > EMAIL_COOLDOWN_SECONDS:
            snapshot_path = "last_alert_snapshot.jpg"
            cv2.imwrite(snapshot_path, frame)

            subject = "FIRE/SMOKE ALERT - Detection System"
            message = (
                f"Fire/Smoke detected!\n\n"
                f"Fire confidence: {fire_conf:.1f}%\n"
                f"Smoke confidence: {smoke_conf:.1f}%\n"
                f"Area affected: {affected_pct:.1f}%\n"
                f"Time: {timestamp}\n"
            )
            send_alert_email(subject, message, image_path=snapshot_path)
            last_email_time = current_time

    cv2.imshow("Fire/Smoke Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()