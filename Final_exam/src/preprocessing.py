"""
Phase 1: CLAHE 강화 전처리 (Lec 3)
- 시인성 낮을 때 equalizeHist 대신 CLAHE 적용
- 야간/역광/안개 환경에서 지역 대비 향상
"""
import cv2
import numpy as np


def compute_visibility_score(gray: np.ndarray) -> float:
    mean_brightness = gray.mean()
    std_brightness  = gray.std()
    brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0
    contrast_score   = min(std_brightness / 64.0, 1.0)
    return float(np.clip(0.5 * brightness_score + 0.5 * contrast_score, 0.0, 1.0))


def get_visibility_label(score: float) -> str:
    if score >= 0.6:
        return "Good"
    elif score >= 0.35:
        return "Moderate"
    return "Poor"


def preprocess(img: np.ndarray):
    """
    CLAHE 강화 전처리 파이프라인
    반환: gray, equalized, blurred, edges, visibility_score
    """
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis_score = compute_visibility_score(gray)

    # Phase 1: CLAHE — 지역 적응 히스토그램 평탄화 (전역 equalizeHist보다 정밀)
    if vis_score < 0.5:
        clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
    else:
        equalized = gray.copy()

    blurred = cv2.GaussianBlur(equalized, (5, 5), 1.0)
    edges   = cv2.Canny(blurred, 50, 150)

    return gray, equalized, blurred, edges, vis_score
