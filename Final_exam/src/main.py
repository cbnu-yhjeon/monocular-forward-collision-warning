"""
Final_exam: Phase 1~6 고도화 위험 탐지 파이프라인

실행 예시:
    python3 main.py --data ../../Data
    python3 main.py --data ../../Data/0.jpg --debug
    python3 main.py --data ../../Data --save --video --csv
"""
import cv2
import numpy as np
import argparse
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import preprocess, get_visibility_label
from detection     import (detect_objects, segment_road_objects,
                            filter_statistical_outliers, apply_watershed)
from lane          import detect_lanes, is_in_lane
from feature       import extract_harris, extract_sift, extract_hog
from classifier    import detect_pedestrians, classify_object
from tracking      import ObjectTracker
from calibration   import load_camera_params, estimate_distance
from risk          import assess_risk, DANGER, CAUTION, SAFE
from visualize     import draw_results, make_debug_panel


def parse_args():
    p = argparse.ArgumentParser(description="Final Exam — 고도화 위험 탐지")
    p.add_argument("--data",    default="../../Data",
                   help="이미지 파일 또는 디렉토리 경로")
    p.add_argument("--debug",   action="store_true",
                   help="4분할 전처리 디버그 패널 표시")
    p.add_argument("--save",    action="store_true",
                   help="결과 이미지 output/ 저장")
    p.add_argument("--video",   action="store_true",
                   help="[Phase 6] 결과 MP4 저장 (output/result.mp4)")
    p.add_argument("--csv",     action="store_true",
                   help="[Phase 6] 통계 CSV 저장 (output/stats.csv)")
    p.add_argument("--max",     type=int, default=0,
                   help="최대 처리 이미지 수 (0=전체)")
    p.add_argument("--calib",   default=None,
                   help="캘리브레이션 디렉토리 (기본: ../../Data/pinhole_calib)")
    p.add_argument("--no-ped",  action="store_true",
                   help="HOG 보행자 검출 비활성화 (처리 속도 우선)")
    return p.parse_args()


def _iou_simple(a, b) -> float:
    x1 = max(a.x, b.x); y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w); y2 = min(a.y + a.h, b.y + b.h)
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (a.area + b.area - inter + 1e-6)


def process_image(img: np.ndarray,
                  tracker: ObjectTracker,
                  debug: bool = False,
                  use_ped: bool = True):
    h, w = img.shape[:2]

    # ── Phase 1: CLAHE 전처리 ────────────────────────────────────────────────
    gray, equalized, blurred, edges, vis_score = preprocess(img)

    # ── Phase 1: 차선 검출 ───────────────────────────────────────────────────
    left_lanes, right_lanes, lane_mask = detect_lanes(edges, img.shape)

    # ── Phase 5: 객체 검출 + Watershed + 이상치 제거 ─────────────────────────
    detections = detect_objects(edges, img.shape)
    if not detections:
        detections = segment_road_objects(gray, img.shape)

    detections = apply_watershed(gray, detections, img.shape)
    detections = filter_statistical_outliers(detections)

    # ── Phase 2: HOG 보행자 검출 후 병합 ────────────────────────────────────
    if use_ped:
        for pd in detect_pedestrians(img, img.shape):
            if not any(_iou_simple(pd, d) > 0.3 for d in detections):
                detections.append(pd)
        detections = sorted(detections, key=lambda d: d.area, reverse=True)[:10]

    # ── Phase 2: 특징점 추출 (Harris + SIFT + HoG) ──────────────────────────
    harris_counts = extract_harris(gray, detections)
    sift_counts   = extract_sift(gray, detections)
    hog_scores    = extract_hog(gray, detections)

    # ── Phase 2: 객체 분류 ───────────────────────────────────────────────────
    for i, det in enumerate(detections):
        det.obj_class = classify_object(det, harris_counts[i], sift_counts[i])

    # ── Phase 3: LK 추적 업데이트 ────────────────────────────────────────────
    tracker.update(detections, gray)

    # ── Phase 4: 거리 추정 (핀홀 모델) ──────────────────────────────────────
    distances = [estimate_distance(det) for det in detections]

    # ── Phase 1~4: 위험도 평가 ───────────────────────────────────────────────
    results = []
    for i, det in enumerate(detections):
        in_lane = is_in_lane(det.cx, det.cy, lane_mask)
        track   = tracker.get_track_for_det(det)
        result  = assess_risk(
            detection        = det,
            img_w            = w,
            img_h            = h,
            visibility_score = vis_score,
            harris_count     = harris_counts[i] if i < len(harris_counts) else 0,
            sift_count       = sift_counts[i]   if i < len(sift_counts)   else 0,
            hog_score        = hog_scores[i]    if i < len(hog_scores)    else 0.0,
            in_lane          = in_lane,
            ttc              = track.ttc      if track else None,
            distance_m       = distances[i],
            track_id         = track.track_id if track else None,
            velocity         = track.velocity if track else (0.0, 0.0),
            obj_class        = det.obj_class,
        )
        results.append(result)

    # ── 시각화 ───────────────────────────────────────────────────────────────
    canvas = draw_results(
        img, results, vis_score,
        harris_counts, sift_counts, hog_scores,
        left_lanes, right_lanes, lane_mask,
    )

    if debug:
        canvas = make_debug_panel(gray, equalized, edges, canvas, lane_mask)

    return canvas, results, vis_score


def main():
    args       = parse_args()
    output_dir = os.path.join(os.path.dirname(__file__), "../output")
    os.makedirs(output_dir, exist_ok=True)

    # ── 카메라 파라미터 로드 (Phase 4) ───────────────────────────────────────
    calib_dir = args.calib or os.path.join(
        os.path.dirname(__file__), "../../Data/pinhole_calib")
    load_camera_params(calib_dir)

    # ── 트래커 초기화 (Phase 3) ───────────────────────────────────────────────
    tracker = ObjectTracker(max_missed=5, iou_thresh=0.25)

    # ── 이미지 목록 ──────────────────────────────────────────────────────────
    exts = {".png", ".jpg", ".jpeg"}
    if os.path.isfile(args.data):
        img_paths = [args.data]
    else:
        img_paths = sorted([
            os.path.join(args.data, f)
            for f in os.listdir(args.data)
            if os.path.splitext(f)[1].lower() in exts
        ])
    if args.max > 0:
        img_paths = img_paths[:args.max]

    if not img_paths:
        print(f"이미지 없음: {args.data}"); return

    print(f"총 {len(img_paths)}장 | "
          f"조작: d/→=다음  a/←=이전  s=저장  q=종료")

    # ── Phase 6: VideoWriter ─────────────────────────────────────────────────
    video_writer = None
    if args.video:
        sample = cv2.imread(img_paths[0])
        if sample is not None:
            sh, sw = sample.shape[:2]
            if args.debug:
                # debug panel 크기는 원본과 동일
                pass
            vpath  = os.path.join(output_dir, "result.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(vpath, fourcc, 10.0, (sw, sh))
            print(f"[Video] 저장 경로: {vpath}")

    # ── Phase 6: CSV ──────────────────────────────────────────────────────────
    csv_file, csv_writer = None, None
    if args.csv:
        csv_path   = os.path.join(output_dir, "stats.csv")
        csv_file   = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame", "filename", "vis_score", "vis_label",
            "n_objects", "n_danger", "n_caution", "n_safe",
            "nearest_m", "min_ttc_s",
            "danger_scores", "obj_classes",
        ])
        print(f"[CSV]   저장 경로: {csv_path}")

    use_ped = not args.no_ped
    idx, cache = 0, {}

    def get_frame(i):
        if i not in cache:
            img = cv2.imread(img_paths[i])
            if img is None:
                return None, None, None
            if i == 0:
                tracker.reset()
            cache[i] = process_image(img, tracker, debug=args.debug,
                                      use_ped=use_ped)
        return cache[i]

    while 0 <= idx < len(img_paths):
        canvas, results, vis_score = get_frame(idx)
        if canvas is None:
            idx += 1; continue

        n_d  = sum(1 for r in results if r.level == DANGER)
        n_c  = sum(1 for r in results if r.level == CAUTION)
        n_s  = sum(1 for r in results if r.level == SAFE)
        near = min((r.distance_m for r in results if r.distance_m), default=None)
        mttc = min((r.ttc for r in results if r.ttc is not None), default=None)
        vis_lbl = get_visibility_label(vis_score)

        print(f"[{idx+1:04d}/{len(img_paths)}] "
              f"{os.path.basename(img_paths[idx])} | "
              f"Vis:{vis_score:.2f}({vis_lbl}) | "
              f"D:{n_d} C:{n_c} S:{n_s} | "
              f"Near:{f'{near:.1f}m' if near else 'N/A'} | "
              f"TTC:{f'{mttc:.1f}s' if mttc else 'N/A'}")

        # Phase 6: CSV 기록
        if csv_writer:
            csv_writer.writerow([
                idx + 1,
                os.path.basename(img_paths[idx]),
                f"{vis_score:.3f}", vis_lbl,
                len(results), n_d, n_c, n_s,
                f"{near:.2f}" if near else "",
                f"{mttc:.2f}" if mttc else "",
                ";".join(f"{r.score:.2f}" for r in results if r.level == DANGER),
                ";".join(r.obj_class for r in results),
            ])

        # Phase 6: 이미지 저장
        if args.save:
            cv2.imwrite(
                os.path.join(output_dir, os.path.basename(img_paths[idx])),
                canvas)

        # Phase 6: 비디오 프레임 추가
        if video_writer is not None:
            fh, fw = canvas.shape[:2]
            vh = int(video_writer.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vw = int(video_writer.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame = cv2.resize(canvas, (vw, vh)) if (fh != vh or fw != vw) else canvas
            video_writer.write(frame)

        cv2.imshow("Final Exam — Forward Danger Detection", canvas)
        key = cv2.waitKey(0) & 0xFF

        if   key == ord("q"):                           break
        elif key == ord("s"):
            p = os.path.join(output_dir, f"save_{idx:04d}.png")
            cv2.imwrite(p, canvas); print(f"  저장: {p}")
        elif key in (83, ord("d")):
            idx = min(idx + 1, len(img_paths) - 1)
        elif key in (81, ord("a")):
            idx = max(idx - 1, 0)

    if video_writer: video_writer.release()
    if csv_file:     csv_file.close()
    cv2.destroyAllWindows()
    print("완료")


if __name__ == "__main__":
    main()
