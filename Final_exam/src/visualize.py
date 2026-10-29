"""
Phase 1~6: 강화된 시각화
- 차선 오버레이 / 거리·TTC 라벨 / 클래스 표시 / 속도 화살표
- make_debug_panel: 4분할 (Gray / CLAHE / Edges+Lane / Result)
"""
import cv2
import numpy as np
from typing import List, Optional
from risk import RiskResult, LEVEL_COLOR, LEVEL_LABEL, SAFE, CAUTION, DANGER
from lane import draw_lanes
from preprocessing import get_visibility_label


def draw_results(img: np.ndarray,
                 results: List[RiskResult],
                 vis_score: float,
                 harris_counts: List[int],
                 sift_counts:   List[int],
                 hog_scores:    Optional[List[float]] = None,
                 left_lanes=None,
                 right_lanes=None,
                 lane_mask: Optional[np.ndarray] = None) -> np.ndarray:

    canvas = img.copy()
    h, w   = canvas.shape[:2]

    # ── 차선 오버레이 ─────────────────────────────────────────────────────────
    if left_lanes is not None:
        canvas = draw_lanes(canvas, left_lanes, right_lanes, lane_mask)

    # ── 각 객체 표시 ──────────────────────────────────────────────────────────
    for i, res in enumerate(results):
        det   = res.detection
        color = LEVEL_COLOR[res.level]
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        hc  = harris_counts[i] if i < len(harris_counts) else 0
        sc  = sift_counts[i]   if i < len(sift_counts)   else 0
        hg  = hog_scores[i]    if hog_scores and i < len(hog_scores) else 0.0

        # 줄 1: [클래스] 레벨 점수
        cls_tag = res.obj_class[:3].upper()
        line1   = f"[{cls_tag}] {LEVEL_LABEL[res.level]} {res.score:.2f}"
        cv2.putText(canvas, line1, (x1, max(y1 - 18, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # 줄 2: 거리 / TTC
        parts = []
        if res.distance_m is not None:
            parts.append(f"{res.distance_m:.1f}m")
        if res.ttc is not None:
            parts.append(f"TTC:{res.ttc:.1f}s")
        if parts:
            cv2.putText(canvas, " | ".join(parts), (x1, max(y1 - 4, 28)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # 줄 3: 특징점 (박스 아래)
        cv2.putText(canvas, f"H:{hc} S:{sc} G:{hg:.2f}",
                    (x1, y2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 170, 170), 1)

        # Track ID
        if res.track_id is not None:
            cv2.putText(canvas, f"#{res.track_id}",
                        (x2 - 28, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

        # 속도 화살표 (LK 추적 결과)
        vx, vy = res.velocity
        if abs(vx) + abs(vy) > 0.5:
            ex = int(det.cx + vx * 4)
            ey = int(det.cy + vy * 4)
            cv2.arrowedLine(canvas, (det.cx, det.cy), (ex, ey),
                            (0, 255, 255), 2, tipLength=0.35)

    # ── 최고 위험 객체 중앙선 ─────────────────────────────────────────────────
    dangers = [r for r in results if r.level == DANGER]
    if dangers:
        top = max(dangers, key=lambda r: r.score)
        cv2.line(canvas, (top.detection.cx, 0),
                 (top.detection.cx, h), (0, 0, 255), 1)

    # ── 우측 HUD ─────────────────────────────────────────────────────────────
    vis_lbl  = get_visibility_label(vis_score)
    nearest  = min((r.distance_m for r in results if r.distance_m),
                   default=None)
    min_ttc  = min((r.ttc for r in results if r.ttc is not None),
                   default=None)
    hud_lines = [
        f"Vis:     {vis_score:.2f} ({vis_lbl})",
        f"Objects: {len(results)}",
        f"DANGER:  {sum(1 for r in results if r.level == DANGER)}",
        f"CAUTION: {sum(1 for r in results if r.level == CAUTION)}",
        f"SAFE:    {sum(1 for r in results if r.level == SAFE)}",
    ]
    if nearest  is not None: hud_lines.append(f"Nearest: {nearest:.1f}m")
    if min_ttc  is not None: hud_lines.append(f"Min TTC: {min_ttc:.1f}s")

    hx = w - 240
    cv2.rectangle(canvas, (hx - 5, 3), (w - 3, 14 + 22 * len(hud_lines)),
                  (20, 20, 20), -1)
    for k, line in enumerate(hud_lines):
        cv2.putText(canvas, line, (hx, 22 + k * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)

    # ── 경고 배너 ─────────────────────────────────────────────────────────────
    if dangers:
        cv2.rectangle(canvas, (0, 0), (w, 30), (0, 0, 160), -1)
        cv2.putText(canvas, "!! DANGER DETECTED !!",
                    (w // 2 - 115, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    return canvas


def make_debug_panel(gray: np.ndarray,
                     equalized: np.ndarray,
                     edges: np.ndarray,
                     canvas: np.ndarray,
                     lane_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    4분할 디버그 패널:
      [Gray]          [CLAHE Equalized]
      [Edges + Lane]  [Result]
    """
    h, w = canvas.shape[:2]
    th, tw = h // 2, w // 2

    def to_bgr(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

    # Edges에 차선 마스크 초록 오버레이
    edges_vis = to_bgr(edges).copy()
    if lane_mask is not None and lane_mask.sum() > 0:
        edges_vis[lane_mask > 0] = (0, 100, 0)

    panels = [
        cv2.resize(to_bgr(gray),      (tw, th)),
        cv2.resize(to_bgr(equalized), (tw, th)),
        cv2.resize(edges_vis,          (tw, th)),
        cv2.resize(canvas,             (tw, th)),
    ]
    labels = ["Gray", "CLAHE Equalized", "Edges + Lane", "Result"]
    for panel, lbl in zip(panels, labels):
        cv2.putText(panel, lbl, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    top = np.hstack(panels[:2])
    bot = np.hstack(panels[2:])
    return np.vstack([top, bot])
