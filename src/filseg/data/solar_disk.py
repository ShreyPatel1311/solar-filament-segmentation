"""Solar-disk geometry.

Off-disk sky is pure background, and the limb darkening ring is the single
biggest source of false positives for a threshold-based detector. Masking the
disk costs one Otsu threshold per image and removes that whole failure mode.
"""

from __future__ import annotations

import cv2
import numpy as np


def disk_mask(image: np.ndarray, margin: float = 0.98) -> np.ndarray:
    """Binary mask of the solar disk, shrunk to ``margin`` of its fitted radius.

    Falls back to a centred inscribed circle when the disk cannot be fitted.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (9, 9), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image.shape[:2]
    if contours:
        largest = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        if radius < 0.25 * min(h, w):  # fit failed - degenerate blob
            cx, cy, radius = w / 2, h / 2, min(h, w) / 2
    else:
        cx, cy, radius = w / 2, h / 2, min(h, w) / 2

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(radius * margin), 1, thickness=-1)
    return mask
