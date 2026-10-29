"""
Phase 2: HOG + Pre-trained SVM 보행자 검출 / 객체 분류 (Lec 12)
- cv2.HOGDescriptor_getDefaultPeopleDetector() : OpenCV 내장 SVM 가중치
- classify_object(): 종횡비 + 특징점 수 기반 휴리스틱 분류
"""
import cv2
import numpy as np
from typing import List
from detection import Detection


_hog_detector = None


def _get_hog_detector():
    global _hog_detector
    if _hog_detector is None:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        _hog_detector = hog
    return _hog_detector


def detect_pedestrians(img: np.ndarray, img_shape: tuple) -> List[Detection]:
    """
    HOG + Pre-trained SVM 보행자 검출
    반환: Detection 목록 (obj_class='pedestrian')
    """
    img_h, img_w = img_shape[:2]
    hog = _get_hog_detector()

    scale = 0.5
    small = cv2.resize(img, None, fx=scale, fy=scale)

    try:
        rects, weights = hog.detectMultiScale(
            small, winStride=(8, 8), padding=(4, 4),
            scale=1.05, finalThreshold=2.0)
    except Exception:
        return []

    if len(rects) == 0:
        return []

    # NMS
    rects_nms = _nms(rects, weights.flatten(), overlap_thresh=0.4)

    detections = []
    for (x, y, w, h) in rects_nms:
        x2 = int(x / scale); y2 = int(y / scale)
        w2 = int(w / scale); h2 = int(h / scale)
        area = float(w2 * h2)
        if y2 + h2 < img_h * 0.30:
            continue
        if area < img_h * img_w * 0.001:
            continue
        detections.append(
            Detection(x=x2, y=y2, w=w2, h=h2, area=area,
                      contour=np.array([]), obj_class='pedestrian'))
    return detections[:5]


def _nms(rects, weights, overlap_thresh=0.4):
    if len(rects) == 0:
        return []
    rects   = np.array(rects, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32).flatten()
    if len(weights) != len(rects):
        weights = np.ones(len(rects))

    x1 = rects[:, 0];          y1 = rects[:, 1]
    x2 = rects[:, 0] + rects[:, 2]; y2 = rects[:, 1] + rects[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = weights.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w_   = np.maximum(0, xx2 - xx1)
        h_   = np.maximum(0, yy2 - yy1)
        inter = w_ * h_
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= overlap_thresh)[0] + 1]

    return [tuple(rects[i].astype(int)) for i in keep]


def classify_object(det: Detection,
                    harris_count: int,
                    sift_count: int) -> str:
    """
    Phase 2: 종횡비 + 특징점 기반 휴리스틱 분류 (Lec 12)
    - pedestrian 태그된 경우 유지
    - 세로 긴 형태(aspect > 1.4) + 특징점 적음 → pedestrian
    - 나머지 → vehicle
    """
    if det.obj_class == 'pedestrian':
        return 'pedestrian'
    aspect = det.h / max(det.w, 1)
    total  = harris_count + sift_count
    if aspect > 1.4 and total < 150:
        return 'pedestrian'
    return 'vehicle'
