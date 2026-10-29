"""
Phase 5: Watershed 세그멘테이션 + 통계적 이상치 제거 (Lec 5/6)
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    area: float
    contour: np.ndarray
    obj_class: str = "unknown"   # Phase 2: 분류 결과 저장

    @property
    def cx(self):
        return self.x + self.w // 2

    @property
    def cy(self):
        return self.y + self.h // 2

    @property
    def bbox(self):
        return (self.x, self.y, self.x + self.w, self.y + self.h)


def _is_vehicle_like(x, y, w, h, img_w, img_h) -> bool:
    area     = w * h
    img_area = img_w * img_h
    if area < img_area * 0.001:   return False
    if area > img_area * 0.15:    return False
    if w > img_w * 0.7:           return False
    if h < 15:                    return False
    if y + h < img_h * 0.30:     return False
    return True


def detect_objects(edges: np.ndarray, img_shape: tuple) -> List[Detection]:
    """Canny → Morphology → Contour → 차량 크기 필터"""
    img_h, img_w = img_shape[:2]
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
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
    """Otsu + ConnectedComponents 보조 검출"""
    img_h, img_w = img_shape[:2]
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    roi = binary.copy()
    roi[:int(img_h * 0.30), :] = 0

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(roi)
    detections = []
    for i in range(1, num_labels):
        x    = stats[i, cv2.CC_STAT_LEFT]
        y    = stats[i, cv2.CC_STAT_TOP]
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = stats[i, cv2.CC_STAT_HEIGHT]
        area = float(stats[i, cv2.CC_STAT_AREA])
        if not _is_vehicle_like(x, y, w, h, img_w, img_h):
            continue
        detections.append(Detection(x=x, y=y, w=w, h=h, area=area,
                                    contour=np.array([])))
    detections.sort(key=lambda d: d.area, reverse=True)
    return detections[:10]


def filter_statistical_outliers(detections: List[Detection]) -> List[Detection]:
    """
    Phase 5: 통계적 이상치 제거 (RANSAC 개념)
    면적 중앙값 대비 20배 이상 크거나 5% 미만인 박스 제거
    """
    if len(detections) < 3:
        return detections
    areas       = np.array([d.area for d in detections])
    median_area = float(np.median(areas))
    filtered    = [d for d in detections
                   if median_area * 0.05 <= d.area <= median_area * 20]
    return filtered if filtered else detections


def apply_watershed(gray: np.ndarray,
                    detections: List[Detection],
                    img_shape: tuple) -> List[Detection]:
    """
    Phase 5: Watershed 세그멘테이션으로 IoU > 0.4 겹침 객체 분리 (Lec 6)
    분리 실패 시 원본 반환
    """
    if len(detections) < 2:
        return detections

    img_h, img_w = img_shape[:2]

    def iou(a, b):
        x1 = max(a.x, b.x); y1 = max(a.y, b.y)
        x2 = min(a.x + a.w, b.x + b.w); y2 = min(a.y + a.h, b.y + b.h)
        if x2 <= x1 or y2 <= y1: return 0.0
        inter = (x2 - x1) * (y2 - y1)
        return inter / (a.area + b.area - inter + 1e-6)

    processed  = set()
    new_dets   = []

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            if iou(detections[i], detections[j]) <= 0.4:
                continue

            d1, d2 = detections[i], detections[j]
            xmin   = max(min(d1.x, d2.x), 0)
            ymin   = max(min(d1.y, d2.y), 0)
            xmax   = min(max(d1.x + d1.w, d2.x + d2.w), img_w)
            ymax   = min(max(d1.y + d1.h, d2.y + d2.h), img_h)

            roi = gray[ymin:ymax, xmin:xmax]
            if roi.size < 200:
                continue

            try:
                # 마커 생성
                _, fg = cv2.threshold(roi, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                kernel   = np.ones((3, 3), np.uint8)
                sure_bg  = cv2.dilate(fg, kernel, iterations=3)
                dist_map = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
                _, sure_fg = cv2.threshold(
                    dist_map, 0.5 * dist_map.max(), 255, 0)
                sure_fg  = np.uint8(sure_fg)
                unknown  = cv2.subtract(sure_bg, sure_fg)

                _, markers = cv2.connectedComponents(sure_fg)
                markers   += 1
                markers[unknown == 255] = 0

                roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                cv2.watershed(roi_bgr, markers)

                for lbl in range(2, markers.max() + 1):
                    mask = np.uint8(markers == lbl) * 255
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in cnts:
                        a = cv2.contourArea(cnt)
                        if a < 200:
                            continue
                        nx, ny, nw, nh = cv2.boundingRect(cnt)
                        nx += xmin; ny += ymin
                        if _is_vehicle_like(nx, ny, nw, nh, img_w, img_h):
                            new_dets.append(
                                Detection(nx, ny, nw, nh, a, cnt))

                processed.add(i)
                processed.add(j)
            except Exception:
                pass  # Watershed 실패 시 원본 유지

    final = [d for k, d in enumerate(detections) if k not in processed] + new_dets
    final.sort(key=lambda d: d.area, reverse=True)
    return final[:10] if final else detections
