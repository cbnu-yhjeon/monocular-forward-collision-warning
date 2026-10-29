"""
공개 주행영상 비디오 처리 스크립트
Final_exam 파이프라인 (Phase 1~6) 적용

사용법:
    python3 process_video.py --input <video.mp4> [--skip 3] [--max 300]
"""
import cv2
import numpy as np
import argparse
import os
import sys
import csv
import time

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import preprocess, get_visibility_label
from detection     import (detect_objects, segment_road_objects,
                            filter_statistical_outliers, apply_watershed)
from lane          import detect_lanes, is_in_lane
from feature       import extract_harris, extract_sift, extract_hog
from classifier    import classify_object
from tracking      import ObjectTracker
from calibration   import load_camera_params, estimate_distance
from risk          import assess_risk, DANGER, CAUTION, SAFE, LEVEL_LABEL
from visualize     import draw_results


def _iou(a, b) -> float:
    x1 = max(a.x, b.x); y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w); y2 = min(a.y + a.h, b.y + b.h)
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (a.area + b.area - inter + 1e-6)


def process_frame(img, tracker, use_ped=True):
    h, w = img.shape[:2]

    gray, equalized, blurred, edges, vis_score = preprocess(img)
    left_lanes, right_lanes, lane_mask = detect_lanes(edges, img.shape)

    detections = detect_objects(edges, img.shape)
    if not detections:
        detections = segment_road_objects(gray, img.shape)
    detections = apply_watershed(gray, detections, img.shape)
    detections = filter_statistical_outliers(detections)

    harris_counts = extract_harris(gray, detections)
    sift_counts   = extract_sift(gray, detections)
    hog_scores    = extract_hog(gray, detections)

    for i, det in enumerate(detections):
        det.obj_class = classify_object(det, harris_counts[i], sift_counts[i])

    tracker.update(detections, gray)
    distances = [estimate_distance(det) for det in detections]

    results = []
    for i, det in enumerate(detections):
        in_lane = is_in_lane(det.cx, det.cy, lane_mask)
        track   = tracker.get_track_for_det(det)
        results.append(assess_risk(
            detection=det, img_w=w, img_h=h,
            visibility_score=vis_score,
            harris_count=harris_counts[i] if i < len(harris_counts) else 0,
            sift_count=sift_counts[i]     if i < len(sift_counts)   else 0,
            hog_score=hog_scores[i]       if i < len(hog_scores)    else 0.0,
            in_lane=in_lane,
            ttc=track.ttc        if track else None,
            distance_m=distances[i],
            track_id=track.track_id if track else None,
            velocity=track.velocity  if track else (0.0, 0.0),
            obj_class=det.obj_class,
        ))

    canvas = draw_results(img, results, vis_score,
                          harris_counts, sift_counts, hog_scores,
                          left_lanes, right_lanes, lane_mask)
    return canvas, results, vis_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True,  help="입력 비디오 경로")
    ap.add_argument("--output", default=None,   help="출력 비디오 경로 (기본: 입력파일명_result.mp4)")
    ap.add_argument("--skip",   type=int, default=3,  help="N프레임마다 처리 (기본 3)")
    ap.add_argument("--max",    type=int, default=0,  help="최대 처리 프레임 수 (0=전체)")
    ap.add_argument("--calib",  default=None)
    ap.add_argument("--no-ped", action="store_true")
    args = ap.parse_args()

    # 출력 경로
    base    = os.path.splitext(args.input)[0]
    out_vid = args.output or base + "_result.mp4"
    out_csv = base + "_stats.csv"

    # 카메라 캘리브레이션
    calib_dir = args.calib or os.path.join(
        os.path.dirname(__file__), "../../Data/pinhole_calib")
    load_camera_params(calib_dir)

    tracker = ObjectTracker(max_missed=8, iou_thresh=0.2)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERROR] 영상 열기 실패: {args.input}"); return

    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = max(src_fps / args.skip, 5.0)

    print(f"입력: {args.input}  {src_w}x{src_h} {src_fps:.1f}fps {total}프레임")
    print(f"출력: {out_vid}  skip={args.skip}  처리FPS≈{out_fps:.1f}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw     = cv2.VideoWriter(out_vid, fourcc, out_fps, (src_w, src_h))

    csv_f   = open(out_csv, "w", newline="", encoding="utf-8")
    csv_w   = csv.writer(csv_f)
    csv_w.writerow(["frame", "vis_score", "vis_label",
                    "n_obj", "n_danger", "n_caution", "n_safe",
                    "nearest_m", "min_ttc", "obj_classes"])

    frame_no = 0
    proc_no  = 0
    t_start  = time.time()
    max_proc = args.max if args.max > 0 else 10**9

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_no += 1

        if (frame_no - 1) % args.skip != 0:
            continue

        canvas, results, vis_score = process_frame(frame, tracker,
                                                    use_ped=not args.no_ped)
        vw.write(canvas)

        n_d  = sum(1 for r in results if r.level == DANGER)
        n_c  = sum(1 for r in results if r.level == CAUTION)
        n_s  = sum(1 for r in results if r.level == SAFE)
        near = min((r.distance_m for r in results if r.distance_m), default=None)
        mttc = min((r.ttc for r in results if r.ttc is not None), default=None)

        csv_w.writerow([
            frame_no,
            f"{vis_score:.3f}", get_visibility_label(vis_score),
            len(results), n_d, n_c, n_s,
            f"{near:.2f}" if near else "",
            f"{mttc:.2f}" if mttc else "",
            ";".join(r.obj_class for r in results),
        ])

        proc_no += 1
        elapsed = time.time() - t_start
        fps_now = proc_no / elapsed if elapsed > 0 else 0
        eta     = (total / args.skip - proc_no) / fps_now if fps_now > 0 else 0
        print(f"\r  [{proc_no:04d}/{total//args.skip}] "
              f"f{frame_no} | D:{n_d} C:{n_c} S:{n_s} | "
              f"Near:{f'{near:.1f}m' if near else 'N/A'} | "
              f"TTC:{f'{mttc:.1f}s' if mttc else 'N/A'} | "
              f"{fps_now:.1f}fps ETA:{eta:.0f}s", end="", flush=True)

        if proc_no >= max_proc:
            break

    print()
    cap.release()
    vw.release()
    csv_f.close()

    elapsed = time.time() - t_start
    print(f"\n완료: {proc_no}프레임 처리 ({elapsed:.1f}초, {proc_no/elapsed:.1f}fps)")
    print(f"  결과 비디오: {out_vid}")
    print(f"  통계  CSV : {out_csv}")


from preprocessing import get_visibility_label

if __name__ == "__main__":
    main()
