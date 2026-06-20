import argparse
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def order_points(pts):
    pts = pts.astype("float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return {
        "TL": tuple(tl.astype(int)),
        "TR": tuple(tr.astype(int)),
        "BR": tuple(br.astype(int)),
        "BL": tuple(bl.astype(int)),
    }


def clean_mask(mask):
    mask = mask.astype("uint8")
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


def yolo_mask(frame, model, min_conf):
    if model is None:
        return None, 0.0

    results = model.predict(frame, conf=min_conf, retina_masks=True, verbose=False)

    if not results or results[0].masks is None:
        return None, 0.0

    masks = results[0].masks.data.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()

    if len(masks) == 0:
        return None, 0.0

    best_idx = max(range(len(masks)), key=lambda i: masks[i].sum() * confs[i])

    h, w = frame.shape[:2]
    mask = (masks[best_idx] > 0.5).astype("uint8") * 255
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return clean_mask(mask), float(confs[best_idx])


def fallback_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    _, mask1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask2 = cv2.bitwise_not(mask1)

    candidates = [clean_mask(mask1), clean_mask(mask2)]

    h, w = frame.shape[:2]
    frame_area = h * w
    best = None
    best_area = 0

    for mask in candidates:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)

            if area < frame_area * 0.015:
                continue

            if area > frame_area * 0.85:
                continue

            if area > best_area:
                best_area = area
                best = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(best, [c], -1, 255, cv2.FILLED)

    return clean_mask(best) if best is not None else None


def get_points(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < 1000:
        return None

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    corners = order_points(box)

    pts = contour.reshape(-1, 2)

    tl = np.array(corners["TL"])
    tr = np.array(corners["TR"])
    br = np.array(corners["BR"])
    bl = np.array(corners["BL"])

    m = cv2.moments(contour)
    if m["m00"] != 0:
        center = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
    else:
        center = tuple(pts.mean(axis=0).astype(int))

    left_edge_mid = tuple(((tl + bl) / 2).astype(int))
    right_edge_mid = tuple(((tr + br) / 2).astype(int))
    fold_line_top = tuple(((tl + tr) / 2).astype(int))
    fold_line_bottom = tuple(((bl + br) / 2).astype(int))

    points = {
        **corners,
        "CENTER": center,
        "LEFTMOST": tuple(pts[np.argmin(pts[:, 0])]),
        "RIGHTMOST": tuple(pts[np.argmax(pts[:, 0])]),
        "TOPMOST": tuple(pts[np.argmin(pts[:, 1])]),
        "BOTTOMMOST": tuple(pts[np.argmax(pts[:, 1])]),
        "LEFT_EDGE_MID": left_edge_mid,
        "RIGHT_EDGE_MID": right_edge_mid,
        "FOLD_LINE_TOP": fold_line_top,
        "FOLD_LINE_BOTTOM": fold_line_bottom,
        "GRASP": left_edge_mid,
        "PLACE": right_edge_mid,
    }

    x, y, w, h = cv2.boundingRect(contour)
    bbox = (x, y, x + w, y + h)

    return contour, bbox, points


def label(img, text, point, color=(255, 255, 255), scale=0.5):
    x, y = point
    x = int(max(5, min(x, img.shape[1] - 180)))
    y = int(max(20, min(y, img.shape[0] - 10)))

    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw(frame, mask, confidence, source):
    out = frame.copy()

    if mask is None:
        label(out, "NO TOWEL / T-SHIRT DETECTED", (25, 40), (0, 0, 255), 0.8)
        label(out, "Use a contrasting towel/shirt on a plain table", (25, 75), (0, 0, 255), 0.55)
        return out

    data = get_points(mask)

    if data is None:
        label(out, "MASK FOUND, BUT NO STABLE CONTOUR", (25, 40), (0, 0, 255), 0.8)
        return out

    contour, bbox, points = data

    overlay = np.zeros_like(out)
    overlay[mask > 0] = (40, 180, 40)
    out = cv2.addWeighted(out, 0.70, overlay, 0.30, 0)

    cv2.drawContours(out, [contour], -1, (0, 255, 255), 3)

    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)

    cv2.line(out, points["FOLD_LINE_TOP"], points["FOLD_LINE_BOTTOM"], (255, 0, 255), 3)
    label(out, "FOLD_LINE", midpoint(points["FOLD_LINE_TOP"], points["FOLD_LINE_BOTTOM"]), (255, 0, 255))

    cv2.arrowedLine(out, points["GRASP"], points["PLACE"], (255, 0, 255), 3, tipLength=0.08)
    label(out, "FOLD DIRECTION", midpoint(points["GRASP"], points["PLACE"]), (255, 0, 255), 0.55)

    for name, pt in points.items():
        if name.startswith("FOLD_LINE"):
            continue

        color = (255, 255, 255)
        radius = 6

        if name in {"TL", "TR", "BR", "BL"}:
            color = (0, 0, 255)
            radius = 8
        elif name in {"GRASP", "PLACE"}:
            color = (255, 0, 255)
            radius = 10
        elif name in {"LEFTMOST", "RIGHTMOST", "TOPMOST", "BOTTOMMOST"}:
            color = (0, 165, 255)
            radius = 7

        cv2.circle(out, pt, radius, color, -1)
        label(out, name, (pt[0] + 8, pt[1] - 8), color)

    label(out, f"source: {source}", (20, 30), (255, 255, 255), 0.6)
    label(out, f"confidence: {confidence:.2f}" if source == "YOLO" else "confidence: fallback", (20, 58), (255, 255, 255), 0.6)
    label(out, "q quit | s save frame", (20, 86), (255, 255, 255), 0.6)

    return out


def midpoint(a, b):
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--model-path", default="yolo11n-seg.pt")
    parser.add_argument("--min-conf", type=float, default=0.35)
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--save-dir", default="debug_frames")
    args = parser.parse_args()

    model = None
    if not args.no_yolo and YOLO is not None:
        print("Loading YOLO model. First run may download weights.")
        model = YOLO(args.model_path)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    Path(args.save_dir).mkdir(exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        mask, conf = yolo_mask(frame, model, args.min_conf) if model is not None else (None, 0.0)
        source_name = "YOLO"

        if mask is None:
            mask = fallback_mask(frame)
            source_name = "threshold"

        out = draw(frame, mask, conf, source_name)
        cv2.imshow("Laundry Corner POC", out)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            path = Path(args.save_dir) / f"laundry_poc_{int(time.time())}.jpg"
            cv2.imwrite(str(path), out)
            print(f"Saved {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
