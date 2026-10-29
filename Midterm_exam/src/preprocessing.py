"""
Lec 3 - Histogram 분석 및 시인성 평가
Lec 3/4 - Gaussian / Frequency 필터링 전처리
"""
import cv2
import numpy as np


def compute_visibility_score(gray: np.ndarray) -> float:
    """
    히스토그램 기반 시인성 점수 계산 (0.0 ~ 1.0)
    - 밝기 평균이 너무 낮거나(야간) 너무 높으면(역광) 시인성 낮음
    - 히스토그램 분산이 클수록 대비가 풍부 → 시인성 높음
    """
    mean_brightness = gray.mean()
    std_brightness  = gray.std()

    # 밝기 점수: 128 근처일수록 1.0
    brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0

    # 대비 점수: 표준편차가 클수록 1.0 (최대 64 기준)
    contrast_score = min(std_brightness / 64.0, 1.0)

    score = 0.5 * brightness_score + 0.5 * contrast_score
    return float(np.clip(score, 0.0, 1.0))


def get_visibility_label(score: float) -> str:
    if score >= 0.6:
        return "Good"
    elif score >= 0.35:
        return "Moderate"
    else:
        return "Poor"


def preprocess(img: np.ndarray):
    """
    전처리 파이프라인
    1. Grayscale 변환
    2. 히스토그램 평탄화 (시인성 낮을 때 보정)
    3. Gaussian blur (노이즈 제거)
    4. Canny edge
    반환: gray, equalized, blurred, edges, visibility_score
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis_score = compute_visibility_score(gray)

    # 시인성이 낮으면 히스토그램 평탄화로 보정
    if vis_score < 0.5:
        equalized = cv2.equalizeHist(gray)
    else:
        equalized = gray.copy()

    # Gaussian blur
    blurred = cv2.GaussianBlur(equalized, (5, 5), 1.0)

    # Canny edge
    edges = cv2.Canny(blurred, 50, 150)

    return gray, equalized, blurred, edges, vis_score
