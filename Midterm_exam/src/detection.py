"""
Lec 5 - Contour / Connected Component
Lec 6 - Segmentation (Threshold + Contour)
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    area: float
    contour: np.ndarray

    @property
    def cx(self):
        return self.x + self.w // 2

    @property
    def cy(self):
        return self.y + self.h // 2

    @property
    def bbox(self):
        return (self.x, self.y, self.x + self.w, self.y + self.h)


def _is_vehicle_like(x, y, w, h, img_w, img_h):
    """
    KITTI 이미지 기준 차량/보행자 크기 필터
    - 너무 작거나(노이즈), 너무 크면(도로/하늘) 제외
    - 가로로 과도하게 넓은 컨투어(도로 마킹 등) 제외
    - 하늘 영역(상단 30%) 제외
    """
    area = w * h
    img_area = img_w * img_h

    if area < img_area * 0.001:   # 너무 작음 (0.1% 미만)
        return False
    if area > img_area * 0.15:    # 너무 큼 (15% 초과 → 도로/배경)
        return False
    if w > img_w * 0.7:           # 가로로 너무 넓음 → 도로 구분선
        return False
    if h < 15:                    # 세로 너무 얇음
        return False
    if y + h < img_h * 0.30:      # 상단 30% 하늘 제외
        return False
    return True


def detect_objects(edges: np.ndarray, img_shape: tuple) -> List[Detection]:
    """
    Canny edge → Morphology → Contour → 차량 크기 필터링
    """
    img_h, img_w = img_shape[:2]

    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        if not _is_vehicle_like(x, y, w, h, img_w, img_h):
            continue

        detections.append(Detection(x=x, y=y, w=w, h=h, area=area, contour=cnt))

    detections.sort(key=lambda d: d.area, reverse=True)
    return detections[:10]


def segment_road_objects(gray: np.ndarray, img_shape: tuple) -> List[Detection]:
    """
    Otsu Thresholding + Connected Component 기반 보조 검출
    """
    img_h, img_w = img_shape[:2]

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    roi = binary.copy()
    roi[:int(img_h * 0.30), :] = 0  # 하늘 영역 제거

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi)
    detections = []
    for i in range(1, num_labels):
        x  = stats[i, cv2.CC_STAT_LEFT]
        y  = stats[i, cv2.CC_STAT_TOP]
        w  = stats[i, cv2.CC_STAT_WIDTH]
        h  = stats[i, cv2.CC_STAT_HEIGHT]
        area = float(stats[i, cv2.CC_STAT_AREA])

        if not _is_vehicle_like(x, y, w, h, img_w, img_h):
            continue

        detections.append(Detection(x=x, y=y, w=w, h=h, area=area,
                                    contour=np.array([])))

    detections.sort(key=lambda d: d.area, reverse=True)
    return detections[:10]
