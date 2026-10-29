"""
Phase 2: Harris + SIFT + HoG 특징점 추출 (Lec 7, 12)
- extract_hog(): HoG 기술자 평균 크기 → 객체 구조 복잡도 지표
"""
import cv2
import numpy as np
from typing import List
from detection import Detection


def extract_harris(gray: np.ndarray, detections: List[Detection]) -> List[int]:
    counts = []
    for det in detections:
        roi = gray[det.y:det.y + det.h, det.x:det.x + det.w]
        if roi.size == 0:
            counts.append(0); continue
        h = cv2.cornerHarris(np.float32(roi), blockSize=2, ksize=3, k=0.04)
        counts.append(int(np.sum(h > 0.01 * h.max())))
    return counts


def extract_sift(gray: np.ndarray, detections: List[Detection]) -> List[int]:
    sift   = cv2.SIFT_create()
    counts = []
    for det in detections:
        roi = gray[det.y:det.y + det.h, det.x:det.x + det.w]
        if roi.size == 0:
            counts.append(0); continue
        kp, _ = sift.detectAndCompute(roi, None)
        counts.append(len(kp))
    return counts


def extract_hog(gray: np.ndarray, detections: List[Detection]) -> List[float]:
    """
    Phase 2: HoG 기술자 평균 크기 (Lec 12)
    64×64 ROI → 9-bin 방향 히스토그램 → 평균값 반환
    값이 클수록 구조적으로 복잡한 객체 (차량/보행자 가능성 ↑)
    """
    hog = cv2.HOGDescriptor(
        _winSize=(64, 64), _blockSize=(16, 16),
        _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9,
    )
    scores = []
    for det in detections:
        roi = gray[det.y:det.y + det.h, det.x:det.x + det.w]
        if roi.size < 64 * 64:
            scores.append(0.0); continue
        resized = cv2.resize(roi, (64, 64))
        desc    = hog.compute(resized)
        scores.append(float(np.mean(np.abs(desc))) if desc is not None else 0.0)
    return scores
