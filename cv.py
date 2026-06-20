# High-level idea: detect a towel in a selected table region and compute real fold-relevant corners.
# Why it works: for towel folding, geometry inside a controlled ROI is more reliable than generic YOLO scene segmentation.
# Approach: segment the largest towel-like contour in ROI, approximate 4 corners, compute fold targets, and draw fold arcs.
# Time complexity: O(n) per frame over ROI pixels and contour points.
# Space complexity: O(n) for the ROI image, mask, contour, and overlay.

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]   # BR
    rect[1] = pts[np.argmin(d)]   # TR
    rect[3] = pts[np.argmax(d)]   # BL

    return rect


def as_point(p: np.ndarray) -> Point:
    return int(round(float(p[0]))), int(round(float(p[1])))


def midpoint(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)


def draw_label(img, text: str, pt: Point, color=(255, 255, 255), scale=0.55):
    x, y = pt
    x = max(5, min(x, img.shape[1] - 150))
    y = max(20, min(y, img.shape[0] - 10))

    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def bezier_curve(p0: Point, p1: Point, p2: Point, samples: int = 40):
    points = []
    for t in np.linspace(0, 1, samples):
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        points.append((int(x), int(y)))
    return points


def draw_arc(img, start: Point, end: Point, height: int = 80, color=(255, 0, 255), thickness=2):
    mx, my = midpoint(start, end)
    control = (mx, my - height)
    curve = bezier_curve(start, control, end, samples=50)

    for i in range(len(curve) - 1):
        cv2.line(img, curve[i], curve[i + 1], color, thickness)

    if len(curve) >= 2:
        cv2.arrowedLine(img, curve[-2], curve[-1], color, thickness, tipLength=0.5)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def segment_towel(roi_bgr: np.ndarray) -> Optional[np.ndarray]:
    # Try a few simple masks and keep the best towel-like contour.
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    _, mask1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask2 = cv2.bitwise_not(mask1)

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, mask3 = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask4 = cv2.bitwise_not(mask3)

    candidates = [mask1, mask2, mask3, mask4]

    h, w = roi_bgr.shape[:2]
    roi_area = h * w

    best_mask = None
    best_score = -1

    for raw in candidates:
        mask = clean_mask(raw)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < roi_area * 0.05:
                continue
            if area > roi_area * 0.95:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            hull = cv2.convexHull(c)

            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 1e-6 else 0
            rect = cv2.minAreaRect(c)
            (rw, rh) = rect[1]
            rect_area = max(rw * rh, 1e-6)
            rectangularity = area / rect_area

            # Prefer something cloth-like and compact.
            score = area
            if len(approx) == 4:
                score *= 1.4
            score *= (0.5 + solidity)
            score *= (0.5 + rectangularity)

            if score > best_score:
                best_score = score
                best_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(best_mask, [c], -1, 255, thickness=cv2.FILLED)

    if best_mask is None:
        return None

    return clean_mask(best_mask)


def get_towel_corners(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        return order_points(pts)

    # Fallback: use minAreaRect corners if approx is not 4
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return order_points(box)


def overlay_mask(base: np.ndarray, mask: np.ndarray, color=(40, 180, 40), alpha=0.28):
    layer = np.zeros_like(base)
    layer[mask > 0] = color
    return cv2.addWeighted(base, 1 - alpha, layer, alpha, 0)


def draw_result(frame: np.ndarray, roi_rect, mask: np.ndarray, corners: np.ndarray, fold_mode: str) -> np.ndarray:
    x, y, w, h = roi_rect
    out = frame.copy()

    roi_view = out[y:y + h, x:x + w]
    roi_view[:] = overlay_mask(roi_view, mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(roi_view, [contour], -1, (0, 255, 255), 3)

    tl, tr, br, bl = [as_point(p) for p in corners]

    # Convert ROI-local points to frame-global points
    def g(pt: Point) -> Point:
        return (pt[0] + x, pt[1] + y)

    TL, TR, BR, BL = map(g, [tl, tr, br, bl])

    # Draw corner points only, no giant bbox
    points = {"TL": TL, "TR": TR, "BR": BR, "BL": BL}
    offsets = {"TL": (-38, -12), "TR": (10, -12), "BR": (10, 20), "BL": (-38, 20)}

    for name, pt in points.items():
        cv2.circle(out, pt, 8, (0, 0, 255), -1)
        dx, dy = offsets[name]
        draw_label(out, name, (pt[0] + dx, pt[1] + dy), (0, 0, 255))

    # Compute fold targets and pinch points
    if fold_mode == "left_to_center":
        pinch_a, pinch_b = TL, BL
        target_a = midpoint(TL, TR)
        target_b = midpoint(BL, BR)
        fold_line_a = midpoint(TL, TR)
        fold_line_b = midpoint(BL, BR)
    elif fold_mode == "right_to_center":
        pinch_a, pinch_b = TR, BR
        target_a = midpoint(TL, TR)
        target_b = midpoint(BL, BR)
        fold_line_a = midpoint(TL, TR)
        fold_line_b = midpoint(BL, BR)
    elif fold_mode == "top_to_center":
        pinch_a, pinch_b = TL, TR
        target_a = midpoint(TL, BL)
        target_b = midpoint(TR, BR)
        fold_line_a = midpoint(TL, BL)
        fold_line_b = midpoint(TR, BR)
    else:  # bottom_to_center
        pinch_a, pinch_b = BL, BR
        target_a = midpoint(TL, BL)
        target_b = midpoint(TR, BR)
        fold_line_a = midpoint(TL, BL)
        fold_line_b = midpoint(TR, BR)

    # Draw fold line
    cv2.line(out, fold_line_a, fold_line_b, (255, 0, 255), 2)
    draw_label(out, "FOLD_LINE", midpoint(fold_line_a, fold_line_b), (255, 0, 255), 0.5)

    # Draw pinch points
    cv2.circle(out, pinch_a, 10, (255, 0, 255), -1)
    cv2.circle(out, pinch_b, 10, (255, 0, 255), -1)
    draw_label(out, "PINCH_1", (pinch_a[0] + 10, pinch_a[1] - 10), (255, 0, 255), 0.5)
    draw_label(out, "PINCH_2", (pinch_b[0] + 10, pinch_b[1] - 10), (255, 0, 255), 0.5)

    # Draw target points
    cv2.circle(out, target_a, 8, (255, 255, 255), -1)
    cv2.circle(out, target_b, 8, (255, 255, 255), -1)
    draw_label(out, "TARGET_1", (target_a[0] + 10, target_a[1] - 10), (255, 255, 255), 0.5)
    draw_label(out, "TARGET_2", (target_b[0] + 10, target_b[1] - 10), (255, 255, 255), 0.5)

    # Draw arcs
    draw_arc(out, pinch_a, target_a, height=70, color=(255, 0, 255), thickness=2)
    draw_arc(out, pinch_b, target_b, height=70, color=(255, 0, 255), thickness=2)

    # Draw ROI boundary softly, not a giant object rectangle
    cv2.rectangle(out, (x, y), (x + w, y + h), (120, 120, 120), 1)
    draw_label(out, f"fold_mode: {fold_mode}", (20, 30), (255, 255, 255), 0.6)
    draw_label(out, "r = reselect ROI | q = quit | s = save", (20, 58), (255, 255, 255), 0.6)

    return out


def pick_roi(frame: np.ndarray):
    roi = cv2.selectROI("Select Towel ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Towel ROI")
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return None
    return roi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument(
        "--fold-mode",
        default="left_to_center",
        choices=["left_to_center", "right_to_center", "top_to_center", "bottom_to_center"],
    )
    parser.add_argument("--save-dir", default="debug_frames")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    roi_rect = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if roi_rect is None:
            draw_label(frame, "Press r to select towel ROI", (20, 30), (255, 255, 255), 0.7)
            cv2.imshow("Laundry Corner POC", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("r"):
                roi_rect = pick_roi(frame)
            continue

        x, y, w, h = roi_rect
        roi = frame[y:y + h, x:x + w]

        mask = segment_towel(roi)

        if mask is None:
            out = frame.copy()
            cv2.rectangle(out, (x, y), (x + w, y + h), (120, 120, 120), 1)
            draw_label(out, "No stable towel contour found in ROI", (20, 30), (0, 0, 255), 0.7)
            draw_label(out, "Use a plain contrasting towel and keep background simple", (20, 58), (0, 0, 255), 0.55)
        else:
            corners = get_towel_corners(mask)
            if corners is None:
                out = frame.copy()
                draw_label(out, "Could not estimate 4 corners", (20, 30), (0, 0, 255), 0.7)
            else:
                out = draw_result(frame, roi_rect, mask, corners, args.fold_mode)

        cv2.imshow("Laundry Corner POC", out)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("r"):
            roi_rect = pick_roi(frame)
        if key == ord("s"):
            path = Path(args.save_dir) / f"laundry_clean_poc_{int(time.time())}.jpg"
            cv2.imwrite(str(path), out)
            print(f"Saved {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
