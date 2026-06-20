# yolo_segmentation_corners.py

from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n-seg.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    for r in results:

        if r.masks is None:
            continue

        for i, mask in enumerate(r.masks.xy):

            pts = np.array(mask, dtype=np.int32)

            cls_id = int(r.boxes.cls[i])
            conf = float(r.boxes.conf[i])
            class_name = model.names[cls_id]

            # Fill segmentation
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

            # Draw outline
            cv2.polylines(
                frame,
                [pts],
                True,
                (0, 255, 0),
                2
            )

            # Object label
            x0, y0 = pts[0]
            cv2.putText(
                frame,
                f"{class_name}: {conf:.2f}",
                (x0, y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # -------------------------
            # CORNER DETECTION
            # -------------------------

            contour = pts.reshape((-1, 1, 2))

            epsilon = 0.02 * cv2.arcLength(contour, True)
            corners = cv2.approxPolyDP(
                contour,
                epsilon,
                True
            )

            # Draw corners
            for j, corner in enumerate(corners):

                x, y = corner[0]

                cv2.circle(
                    frame,
                    (x, y),
                    8,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    frame,
                    f"C{j}",
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

    cv2.imshow("YOLO Segmentation + Corners", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()