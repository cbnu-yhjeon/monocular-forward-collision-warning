"""
결과 시각화
"""
import cv2
import numpy as np
from risk import RiskResult, LEVEL_COLOR, LEVEL_LABEL, SAFE, CAUTION, DANGER


def draw_results(img: np.ndarray,
                 results: list[RiskResult],
                 vis_score: float,
                 harris_counts: list[int],
                 sift_counts: list[int]) -> np.ndarray:

    canvas = img.copy()
    h, w   = canvas.shape[:2]

    # ── 각 객체 박스 + 위험도 표시 ───────────────────────────────────────────
    for i, res in enumerate(results):
        det   = res.detection
        color = LEVEL_COLOR[res.level]
        label = LEVEL_LABEL[res.level]

        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        hc = harris_counts[i] if i < len(harris_counts) else 0
        sc = sift_counts[i]   if i < len(sift_counts)   else 0

        info = f"{label} {res.score:.2f} | H:{hc} S:{sc}"
        cv2.putText(canvas, info, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

    # ── 최고 위험도 객체 중앙선 강조 ─────────────────────────────────────────
    dangers = [r for r in results if r.level == DANGER]
    if dangers:
        top = dangers[0]
        cv2.line(canvas, (top.detection.cx, 0),
                 (top.detection.cx, h), (0, 0, 255), 1)

    # ── 우측 상단 HUD ─────────────────────────────────────────────────────────
    hud_lines = [
        f"Visibility: {vis_score:.2f}",
        f"Objects:    {len(results)}",
        f"DANGER:     {sum(1 for r in results if r.level == DANGER)}",
        f"CAUTION:    {sum(1 for r in results if r.level == CAUTION)}",
        f"SAFE:       {sum(1 for r in results if r.level == SAFE)}",
    ]
    for i, line in enumerate(hud_lines):
        cv2.putText(canvas, line, (w - 220, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ── 경고 배너 ─────────────────────────────────────────────────────────────
    if dangers:
        cv2.rectangle(canvas, (0, 0), (w, 30), (0, 0, 180), -1)
        cv2.putText(canvas, "!! DANGER DETECTED !!",
                    (w // 2 - 110, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return canvas


def make_debug_panel(gray, equalized, edges, canvas):
    """
    전처리 단계 디버그 패널 (4분할)
    """
    h, w = canvas.shape[:2]

    def to_bgr(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img

    th = h // 2
    tw = w // 2
    panels = [
        cv2.resize(to_bgr(gray),      (tw, th)),
        cv2.resize(to_bgr(equalized), (tw, th)),
        cv2.resize(to_bgr(edges),     (tw, th)),
        cv2.resize(canvas,            (tw, th)),
    ]
    labels = ["Gray", "Equalized", "Edges", "Result"]
    for panel, lbl in zip(panels, labels):
        cv2.putText(panel, lbl, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    top = np.hstack(panels[:2])
    bot = np.hstack(panels[2:])
    return np.vstack([top, bot])
