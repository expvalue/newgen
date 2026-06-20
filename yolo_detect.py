# cloth_corners.py
from ultralytics import FastSAM
import cv2
import numpy as np

model = FastSAM("FastSAM-s.pt")   # auto-downloads ~23MB first run
cap = cv2.VideoCapture(0)

FRAME_AREA = None
smoothed = None          # EMA-smoothed 4 corners
ALPHA = 0.4              # smoothing: higher = snappier, lower = steadier

def order_corners(p):
    # returns TL, TR, BR, BL
    p = p[np.argsort(p[:, 1])]            # sort by y
    top, bot = p[:2], p[2:]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bot[np.argsort(bot[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if FRAME_AREA is None:
        FRAME_AREA = frame.shape[0] * frame.shape[1]

    results = model(frame, verbose=False, retina_masks=True, conf=0.4, iou=0.9)

    cloth = None
    best_area = 0
    for r in results:
        if r.masks is None:
            continue
        for poly in r.masks.xy:
            pts = np.array(poly, dtype=np.int32)
            if len(pts) < 3:
                continue
            area = cv2.contourArea(pts)
            # ignore tiny specks and the giant table/background mask
            if area < 0.02 * FRAME_AREA or area > 0.60 * FRAME_AREA:
                continue
            if area > best_area:
                best_area = area
                cloth = pts

    if cloth is not None:
        # mask overlay (dark green for cloth)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [cloth], (0, 100, 0))
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        cv2.polylines(frame, [cloth], True, (0, 255, 0), 2)

        # 4 stable corners from best-fit rectangle
        rect = cv2.minAreaRect(cloth)
        box = order_corners(cv2.boxPoints(rect))

        # temporal smoothing
        if smoothed is None:
            smoothed = box
        else:
            smoothed = ALPHA * box + (1 - ALPHA) * smoothed
        corners = smoothed.astype(int)

        center = corners.mean(axis=0).astype(int)
        cv2.circle(frame, tuple(center), 5, (255, 255, 0), -1)

        for j, (x, y) in enumerate(corners):
            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(frame, f"C{j}", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ---> feed these pixel coords into your camera->arm transform
        # C0=TL  C1=TR  C2=BR  C3=BL  (consistent every frame)
        grasp_corners = corners            # shape (4,2)
        # print(grasp_corners.tolist(), center.tolist())
    else:
        smoothed = None  # reset when cloth leaves view

    cv2.imshow("Cloth Corners", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows() 