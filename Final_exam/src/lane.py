"""
Phase 1: Hough Transform 기반 차선 검출 (Lec 5)
- 하단 ROI에 HoughLinesP 적용
- 기울기 부호로 좌/우 차선 분리
- 차선 내부 마스크 생성 → 객체 위치 판별
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple


def detect_lanes(edges: np.ndarray, img_shape: tuple):
    """
    Hough Transform 차선 검출
    반환: (left_lines, right_lines, lane_mask)
    """
    h, w  = img_shape[:2]
    roi_y = int(h * 0.45)          # 상단 45% 하늘·원경 제외

    roi_edges = edges.copy()
    roi_edges[:roi_y, :] = 0

    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=50,
                             minLineLength=80, maxLineGap=60)

    left_lines: List[Tuple] = []
    right_lines: List[Tuple] = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:        # 수평선 제외
                continue
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))

    lane_mask = _build_lane_mask(left_lines, right_lines, h, w)
    return left_lines, right_lines, lane_mask


def _fit_lane_line(lines: List[Tuple], h: int) -> Optional[Tuple[int, int, int, int]]:
    """여러 선분 → 하나의 대표 직선 피팅 (cv2.fitLine)"""
    if not lines:
        return None
    pts = []
    for x1, y1, x2, y2 in lines:
        pts.extend([(x1, y1), (x2, y2)])
    pts = np.array(pts, dtype=np.float32)
    line_params = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    vx, vy, x0, y0 = (float(line_params[0]), float(line_params[1]),
                      float(line_params[2]), float(line_params[3]))
    if abs(vy) < 1e-6:
        return None
    y_bot = h
    y_top = int(h * 0.55)
    t_bot = (y_bot - y0) / vy
    t_top = (y_top - y0) / vy
    return (int(x0 + t_top * vx), y_top, int(x0 + t_bot * vx), y_bot)


def _build_lane_mask(left_lines, right_lines, h: int, w: int) -> np.ndarray:
    """좌/우 차선 사이 영역을 채운 마스크 생성"""
    mask  = np.zeros((h, w), dtype=np.uint8)
    left  = _fit_lane_line(left_lines,  h)
    right = _fit_lane_line(right_lines, h)
    if left and right:
        pts = np.array([
            [left[0],  left[1]],
            [right[0], right[1]],
            [right[2], right[3]],
            [left[2],  left[3]],
        ], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def is_in_lane(cx: int, cy: int, lane_mask: np.ndarray) -> bool:
    """객체 중심이 차선 내부인지 확인. 차선 미검출 시 True 반환."""
    if lane_mask is None or lane_mask.sum() == 0:
        return True
    h, w = lane_mask.shape
    cy_  = min(max(cy, 0), h - 1)
    cx_  = min(max(cx, 0), w - 1)
    return bool(lane_mask[cy_, cx_] > 0)


def draw_lanes(canvas: np.ndarray,
               left_lines, right_lines,
               lane_mask: np.ndarray) -> np.ndarray:
    """차선 + 내부 영역 오버레이"""
    if lane_mask is not None and lane_mask.sum() > 0:
        green_layer           = np.zeros_like(canvas)
        green_layer[lane_mask > 0] = (0, 80, 0)
        canvas = cv2.addWeighted(canvas, 1.0, green_layer, 0.35, 0)

    h = canvas.shape[0]
    for seg in ((_fit_lane_line(left_lines,  h), (0, 255, 255)),
                (_fit_lane_line(right_lines, h), (0, 255, 255))):
        line, col = seg
        if line:
            cv2.line(canvas, (line[0], line[1]), (line[2], line[3]), col, 2)
    return canvas
