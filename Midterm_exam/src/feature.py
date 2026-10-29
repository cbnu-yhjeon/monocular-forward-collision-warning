"""
Lec 7 - Harris Corner, SIFT 특징점 추출
검출된 객체 영역 내에서 특징점 수를 위험도 보조 지표로 활용
"""
import cv2
import numpy as np
from typing import List
from detection import Detection


def extract_harris(gray: np.ndarray, detections: List[Detection]):
    """
    각 검출 객체 ROI 내 Harris 코너 수 반환
    """
    counts = []
    for det in detections:
        roi = gray[det.y:det.y + det.h, det.x:det.x + det.w]
        if roi.size == 0:
            counts.append(0)
            continue
        harris = cv2.cornerHarris(np.float32(roi), blockSize=2, ksize=3, k=0.04)
        count  = int(np.sum(harris > 0.01 * harris.max()))
        counts.append(count)
    return counts


def extract_sift(gray: np.ndarray, detections: List[Detection]):
    """
    각 검출 객체 ROI 내 SIFT 키포인트 수 반환
    """
    sift   = cv2.SIFT_create()
    counts = []
    for det in detections:
        roi = gray[det.y:det.y + det.h, det.x:det.x + det.w]
        if roi.size == 0:
            counts.append(0)
            continue
        kp, _ = sift.detectAndCompute(roi, None)
        counts.append(len(kp))
    return counts
