"""
Phase 4: 핀홀 카메라 모델 기반 거리 추정 (Lec 10)
- Data/pinhole_calib/camera_mat.npy 로드
- Z = (실제높이 × fy) / 픽셀높이
"""
import cv2
import numpy as np
import os
from typing import Optional
from detection import Detection


REAL_HEIGHTS = {
    'vehicle':    1.5,    # 차량 평균 높이 (m)
    'pedestrian': 1.7,    # 보행자 평균 높이 (m)
    'unknown':    1.5,
}

_camera_mat:   Optional[np.ndarray] = None
_dist_coefs:   Optional[np.ndarray] = None
_calib_loaded: bool = False


def load_camera_params(calib_dir: Optional[str] = None) -> bool:
    """
    camera_mat.npy / dist_coefs.npy 로드
    실패 시 False 반환 (거리 추정 비활성화)
    """
    global _camera_mat, _dist_coefs, _calib_loaded
    if _calib_loaded:
        return _camera_mat is not None

    if calib_dir is None:
        calib_dir = os.path.join(
            os.path.dirname(__file__), "../../Data/pinhole_calib")

    mat_path  = os.path.join(calib_dir, "camera_mat.npy")
    dist_path = os.path.join(calib_dir, "dist_coefs.npy")

    try:
        _camera_mat  = np.load(mat_path)
        _dist_coefs  = np.load(dist_path)
        _calib_loaded = True
        print(f"[Calib] 파라미터 로드 완료 | "
              f"fx={_camera_mat[0,0]:.1f} fy={_camera_mat[1,1]:.1f} "
              f"cx={_camera_mat[0,2]:.1f} cy={_camera_mat[1,2]:.1f}")
        return True
    except Exception as e:
        print(f"[Calib] 로드 실패: {e} → 거리 추정 비활성화")
        _calib_loaded = True
        return False


def estimate_distance(det: Detection) -> Optional[float]:
    """
    핀홀 모델 거리 추정 (m)
    Z = (real_height × fy) / pixel_height
    """
    if _camera_mat is None or det.h <= 0:
        return None
    fy       = float(_camera_mat[1, 1])
    real_h   = REAL_HEIGHTS.get(det.obj_class, 1.5)
    distance = (real_h * fy) / det.h
    return float(np.clip(distance, 0.5, 300.0))


def undistort_image(img: np.ndarray) -> np.ndarray:
    """렌즈 왜곡 보정 (캘리브레이션 파라미터 없으면 원본 반환)"""
    if _camera_mat is None or _dist_coefs is None:
        return img
    return cv2.undistort(img, _camera_mat, _dist_coefs)


def get_fy() -> Optional[float]:
    return float(_camera_mat[1, 1]) if _camera_mat is not None else None
