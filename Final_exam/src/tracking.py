"""
Phase 3: Lucas-Kanade 광학 흐름 기반 객체 추적 + TTC 계산 (Lec 9)
- Good Features to Track → LK로 키포인트 추적
- IoU 기반 박스 매칭으로 트랙 관리
- TTC = area / dArea (박스 크기 변화율 기반 충돌 예상 시간)
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from detection import Detection


@dataclass
class Track:
    track_id: int
    det: Detection
    centroid_history: List[Tuple[int, int]] = field(default_factory=list)
    area_history: List[float]               = field(default_factory=list)
    keypoints: Optional[np.ndarray]         = None
    age: int   = 0
    missed: int = 0

    @property
    def velocity(self) -> Tuple[float, float]:
        """중심점 이동 속도 (픽셀/프레임)"""
        if len(self.centroid_history) < 2:
            return (0.0, 0.0)
        dx = self.centroid_history[-1][0] - self.centroid_history[-2][0]
        dy = self.centroid_history[-1][1] - self.centroid_history[-2][1]
        return (float(dx), float(dy))

    @property
    def ttc(self) -> Optional[float]:
        """
        TTC(Time-To-Collision) 추정 (프레임 단위)
        박스 면적이 커질수록(접근 중) TTC 감소
        TTC = area_prev / (area_now - area_prev)
        """
        if len(self.area_history) < 2:
            return None
        area_now  = self.area_history[-1]
        area_prev = self.area_history[-2]
        d_area    = area_now - area_prev
        if d_area <= 1e-3 or area_prev <= 0:
            return None
        return float(np.clip(area_prev / d_area, 0.5, 99.0))


class ObjectTracker:
    """
    IoU 매칭 + LK Optical Flow 기반 멀티 객체 추적기
    """
    LK_PARAMS = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    def __init__(self, max_missed: int = 5, iou_thresh: float = 0.25):
        self.tracks: Dict[int, Track] = {}
        self.next_id    = 0
        self.max_missed = max_missed
        self.iou_thresh = iou_thresh
        self.prev_gray: Optional[np.ndarray] = None

    def update(self, detections: List[Detection],
               gray: np.ndarray) -> List[Track]:
        # LK 키포인트 추적 (이전 프레임 있을 때)
        if self.prev_gray is not None and self.tracks:
            self._update_lk(gray)

        # IoU 매칭
        matched, unmatched_tids, unmatched_dets = \
            self._match_iou(list(self.tracks.values()), detections)

        # 매칭된 트랙 업데이트
        for tid, det in matched:
            t = self.tracks[tid]
            t.det = det
            t.centroid_history.append((det.cx, det.cy))
            t.area_history.append(det.area)
            t.age    += 1
            t.missed  = 0
            t.keypoints = self._init_keypoints(gray, det)

        # 미매칭 트랙 missed 증가
        for tid in unmatched_tids:
            self.tracks[tid].missed += 1

        # 오래된 트랙 제거
        self.tracks = {tid: t for tid, t in self.tracks.items()
                       if t.missed <= self.max_missed}

        # 신규 트랙 생성
        for det in unmatched_dets:
            t = Track(
                track_id=self.next_id,
                det=det,
                centroid_history=[(det.cx, det.cy)],
                area_history=[det.area],
                keypoints=self._init_keypoints(gray, det),
            )
            self.tracks[self.next_id] = t
            self.next_id += 1

        self.prev_gray = gray.copy()
        return list(self.tracks.values())

    def get_track_for_det(self, det: Detection) -> Optional[Track]:
        best_iou, best = 0.0, None
        for t in self.tracks.values():
            s = _iou(det, t.det)
            if s > best_iou:
                best_iou, best = s, t
        return best if best_iou >= self.iou_thresh else None

    def reset(self):
        self.tracks.clear()
        self.next_id  = 0
        self.prev_gray = None

    # ── 내부 메서드 ───────────────────────────────────────────────────────────

    def _init_keypoints(self, gray: np.ndarray,
                        det: Detection) -> Optional[np.ndarray]:
        roi = gray[det.y:det.y + det.h, det.x:det.x + det.w]
        if roi.size < 200:
            return None
        pts = cv2.goodFeaturesToTrack(
            roi, maxCorners=20, qualityLevel=0.3, minDistance=5)
        if pts is None:
            return None
        pts[:, 0, 0] += det.x
        pts[:, 0, 1] += det.y
        return pts

    def _update_lk(self, gray: np.ndarray):
        # 이미지 크기가 다르면 LK 불가 (다른 씬 전환 시)
        if self.prev_gray.shape != gray.shape:
            return
        for t in self.tracks.values():
            if t.keypoints is None or len(t.keypoints) == 0:
                continue
            try:
                pts_next, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, t.keypoints, None, **self.LK_PARAMS)
            except cv2.error:
                continue
            if pts_next is None or status is None:
                continue
            good = pts_next[status.flatten() == 1]
            if len(good) > 0:
                t.keypoints = good.reshape(-1, 1, 2)

    def _match_iou(self, tracks: List[Track], dets: List[Detection]):
        if not tracks or not dets:
            return [], [t.track_id for t in tracks], list(dets)

        iou_mat = np.zeros((len(tracks), len(dets)))
        for i, t in enumerate(tracks):
            for j, d in enumerate(dets):
                iou_mat[i, j] = _iou(t.det, d)

        matched, used_d = [], set()
        for i, t in enumerate(tracks):
            j = int(np.argmax(iou_mat[i]))
            if iou_mat[i, j] >= self.iou_thresh and j not in used_d:
                matched.append((t.track_id, dets[j]))
                used_d.add(j)

        unmatched_tids = [t.track_id for i, t in enumerate(tracks)
                          if not any(tid == t.track_id for tid, _ in matched)]
        unmatched_dets = [dets[j] for j in range(len(dets)) if j not in used_d]
        return matched, unmatched_tids, unmatched_dets


def _iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x, b.x); y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w); y2 = min(a.y + a.h, b.y + b.h)
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (a.area + b.area - inter + 1e-6)
