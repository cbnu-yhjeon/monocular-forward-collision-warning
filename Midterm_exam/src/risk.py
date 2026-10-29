"""
위험도 등급화
위험도 = f(객체 크기, 화면 중앙 거리, 시인성)
"""
from dataclasses import dataclass

import numpy as np

from detection import Detection

# 위험도 레벨
SAFE    = 0   # 🟢 안전
CAUTION = 1   # 🟡 주의
DANGER  = 2   # 🔴 위험

LEVEL_COLOR = {
    SAFE:    (0, 200, 0),
    CAUTION: (0, 200, 255),
    DANGER:  (0, 0, 255),
}
LEVEL_LABEL = {
    SAFE:    "SAFE",
    CAUTION: "CAUTION",
    DANGER:  "DANGER",
}


@dataclass
class RiskResult:
    """위험도 평가 결과."""

    detection: Detection
    level: int
    score: float          # 0.0(안전) ~ 1.0(위험)
    size_score: float
    center_score: float
    visibility_weight: float


def assess_risk(detection: Detection,
                img_w: int, img_h: int,
                visibility_score: float) -> RiskResult:
    """
    위험도 점수 계산

    - size_score: 객체 면적 / 이미지 면적
    - center_score: 1 - (중심까지 거리 / 화면 대각선)
    - visibility_weight: 시인성 낮을수록 위험 가중
    """
    img_area = img_w * img_h

    # 크기 점수 — KITTI 기준: 8% 이상이면 1.0 (근접 차량)
    size_score = min(detection.area / (img_area * 0.08), 1.0)

    # 중앙 거리 점수
    dx = (detection.cx - img_w / 2) / (img_w / 2)
    dy = (detection.cy - img_h / 2) / (img_h / 2)
    dist_norm   = np.sqrt(dx ** 2 + dy ** 2) / np.sqrt(2)
    center_score = 1.0 - float(np.clip(dist_norm, 0.0, 1.0))

    # 시인성 가중치: 낮을수록 위험 가중 (최대 1.3배)
    vis_weight = 1.0 + 0.3 * (1.0 - visibility_score)

    score = (0.5 * size_score + 0.5 * center_score) * vis_weight
    score = float(np.clip(score, 0.0, 1.0))

    if score >= 0.6:
        level = DANGER
    elif score >= 0.35:
        level = CAUTION
    else:
        level = SAFE

    return RiskResult(
        detection=detection,
        level=level,
        score=score,
        size_score=size_score,
        center_score=center_score,
        visibility_weight=vis_weight,
    )
