"""
Phase 1~4 통합 위험도 평가
- Phase 1: Harris/SIFT/HoG → feature_score 반영
- Phase 1: 차선 위치 가중치
- Phase 3: TTC 가중치
- Phase 4: 거리 기반 size_score 보정
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from detection import Detection

SAFE    = 0
CAUTION = 1
DANGER  = 2

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
    detection:          Detection
    level:              int
    score:              float
    size_score:         float
    center_score:       float
    visibility_weight:  float
    feature_score:      float       = 0.0
    lane_weight:        float       = 1.0
    ttc:                Optional[float] = None
    distance_m:         Optional[float] = None
    track_id:           Optional[int]   = None
    velocity:           tuple           = (0.0, 0.0)
    obj_class:          str             = "unknown"


def assess_risk(detection: Detection,
                img_w: int,
                img_h: int,
                visibility_score: float,
                harris_count:  int   = 0,
                sift_count:    int   = 0,
                hog_score:     float = 0.0,
                in_lane:       bool  = True,
                ttc:           Optional[float] = None,
                distance_m:    Optional[float] = None,
                track_id:      Optional[int]   = None,
                velocity:      tuple           = (0.0, 0.0),
                obj_class:     str             = "unknown") -> RiskResult:
    """
    Phase 1~4 통합 위험도 점수 산출
    """
    img_area = img_w * img_h

    # ── 1. 크기/거리 점수 (45%) ───────────────────────────────────────────────
    if distance_m is not None:
        # Phase 4: 핀홀 거리 기반 점수 (5m→1.0, 50m→0.0)
        size_score = float(np.clip(1.0 - (distance_m - 5.0) / 45.0, 0.0, 1.0))
    else:
        size_score = min(detection.area / (img_area * 0.08), 1.0)

    # ── 2. 중앙 거리 점수 (45%) ───────────────────────────────────────────────
    dx = (detection.cx - img_w / 2) / (img_w / 2)
    dy = (detection.cy - img_h / 2) / (img_h / 2)
    center_score = 1.0 - float(np.clip(
        np.sqrt(dx ** 2 + dy ** 2) / np.sqrt(2), 0.0, 1.0))

    # ── 3. 특징점 점수 (10%, Phase 1) ─────────────────────────────────────────
    feat_score  = float(np.clip((harris_count + sift_count) / 500.0, 0.0, 1.0))
    hog_contrib = float(np.clip(hog_score * 10.0, 0.0, 1.0))
    feature_score = 0.5 * feat_score + 0.5 * hog_contrib

    # ── 4. 시인성 가중치 (최대 1.3×) ─────────────────────────────────────────
    vis_weight = 1.0 + 0.3 * (1.0 - visibility_score)

    # ── 5. TTC 가중치 (Phase 3, 최대 1.4×) ───────────────────────────────────
    ttc_weight = 1.0
    if ttc is not None:
        ttc_weight = float(np.clip(
            1.0 + 0.4 * max(0.0, 1.0 - ttc / 3.0), 1.0, 1.4))

    # ── 6. 차선 가중치 (Phase 1) ──────────────────────────────────────────────
    lane_weight = 1.1 if in_lane else 0.85

    # ── 최종 점수 ──────────────────────────────────────────────────────────────
    base  = 0.45 * size_score + 0.45 * center_score + 0.10 * feature_score
    score = float(np.clip(base * vis_weight * ttc_weight * lane_weight, 0.0, 1.0))

    level = DANGER if score >= 0.6 else (CAUTION if score >= 0.35 else SAFE)

    return RiskResult(
        detection=detection,
        level=level,
        score=score,
        size_score=size_score,
        center_score=center_score,
        visibility_weight=vis_weight,
        feature_score=feature_score,
        lane_weight=lane_weight,
        ttc=ttc,
        distance_m=distance_m,
        track_id=track_id,
        velocity=velocity,
        obj_class=obj_class,
    )
