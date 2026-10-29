# 전방 위험 탐지 시스템 — Final Exam 고도화

충북대학교 산업 컴퓨터비전 | Midterm → Final 고도화 프로젝트

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [Midterm 대비 고도화 내용](#2-midterm-대비-고도화-내용)
3. [강의별 적용 기법 (강의 PDF → 모듈 매핑)](#3-강의별-적용-기법)
4. [전체 아키텍처 및 데이터 흐름](#4-전체-아키텍처-및-데이터-흐름)
5. [모듈별 알고리즘 상세](#5-모듈별-알고리즘-상세)
6. [위험도 평가 공식](#6-위험도-평가-공식)
7. [실행 방법](#7-실행-방법)
8. [공개 주행 데이터셋 검증 결과](#8-공개-주행-데이터셋-검증-결과)
9. [의존성 및 파라미터](#9-의존성-및-파라미터)

---

## 1. 프로젝트 개요

단안 카메라 영상 하나만으로 전방 위험 객체를 탐지하고 충돌 위험도를 자동 등급화하는 시스템입니다.  
딥러닝 없이 강의 3~12주차에서 학습한 **순수 영상처리 기법만** 사용하며,  
Midterm에서 구현한 기본 파이프라인을 Phase 1~6으로 단계적으로 고도화했습니다.

```
단안 카메라 영상 (이미지 / 동영상)
        │
        ▼
  [10단계 처리 파이프라인]
        │
        ▼
  DANGER / CAUTION / SAFE 위험 등급 + 거리 + TTC + 차선 정보
```

---

## 2. Midterm 대비 고도화 내용

### 2-1. 전처리 (Phase 1-A)

| 항목 | Midterm | Final | 강의 |
|------|---------|-------|------|
| 히스토그램 평탄화 | `cv2.equalizeHist` (전역) | **CLAHE** — `clipLimit=2.0, tileGridSize=(8,8)` (지역 적응) | Lec 3 |
| 적용 조건 | 항상 적용 | `vis_score < 0.5` 일 때만 적용 (야간·역광·안개 환경 전용) | Lec 3 |
| 시인성 점수 | 0.5×밝기 + 0.5×대비 | 동일 공식 유지, 결과를 위험도 가중치로 활용 | — |

**equalizeHist vs CLAHE 차이:**
- `equalizeHist`: 전체 히스토그램을 균등 분포로 변환 → 국소적으로 과보정 발생
- `CLAHE`: 타일 단위(8×8)로 지역 평탄화 후 인접 타일 보간 → 과보정(Clip Limit)을 억제하여 세부 구조 보존

### 2-2. 차선 검출 추가 (Phase 1-B) — Midterm에 없던 기능

Midterm에는 차선 검출이 전혀 없었습니다. Final에서 신규 추가:

| 항목 | Final 구현 |
|------|-----------|
| 알고리즘 | `cv2.HoughLinesP` (확률적 Hough Transform) |
| ROI | 상단 45% 제거 (하늘·원경 노이즈 제거) |
| 차선 분리 | 기울기 부호로 좌/우 분리 (`slope < 0` → 좌, `slope > 0` → 우) |
| 대표선 피팅 | `cv2.fitLine (DIST_L2)` 으로 여러 선분 → 1개 대표 직선 |
| 마스크 생성 | `cv2.fillPoly` 로 차선 내부 영역 마스킹 |
| 위험도 반영 | 차선 내 객체: ×1.1 / 차선 밖: ×0.85 가중치 적용 |

### 2-3. 특징점 → 위험도 반영 (Phase 1-C / 2-A)

| 항목 | Midterm | Final |
|------|---------|-------|
| Harris | 검출 후 화면 표시만 | **위험도 feature_score 10% 반영** |
| SIFT | 검출 후 화면 표시만 | **위험도 feature_score 10% 반영** |
| HoG | 없음 | **신규 추가** — 64×64 ROI, 9-bin 방향 히스토그램 |
| 공식 | — | `feature = 0.5×(harris+sift)/500 + 0.5×clip(hog×10, 0, 1)` |

### 2-4. 객체 분류 추가 (Phase 2-B) — Midterm에 없던 기능

| 항목 | Midterm | Final |
|------|---------|-------|
| 분류 | 없음 (모두 동일 취급) | **vehicle / pedestrian** 분류 |
| HOG+SVM | 없음 | `HOGDescriptor_getDefaultPeopleDetector()` — 사전학습 SVM |
| 휴리스틱 분류 | 없음 | 종횡비(aspect > 1.4) + 특징점 수(< 150) → pedestrian |
| 거리 보정 | 없음 | 분류 결과로 실제 높이 선택 (차량 1.5m, 보행자 1.7m) |

### 2-5. 객체 추적 + TTC 추가 (Phase 3) — Midterm에 없던 기능

| 항목 | Midterm | Final |
|------|---------|-------|
| 추적 | 없음 | **Lucas-Kanade Optical Flow** |
| 매칭 | 없음 | IoU ≥ 0.25 기반 프레임 간 박스 매칭 |
| TTC | 없음 | `TTC = area_prev / Δarea` (박스 면적 변화율 기반) |
| 속도 표시 | 없음 | 중심점 이동 벡터를 화살표로 시각화 |
| 위험도 반영 | 없음 | TTC < 3s → 최대 ×1.4 가중치 |

### 2-6. 거리 추정 정확도 개선 (Phase 4)

| 항목 | Midterm | Final |
|------|---------|-------|
| 거리 추정 | 면적 기반 (부정확) | **핀홀 카메라 모델** `Z = (H_real × fy) / H_pixel` |
| 캘리브레이션 | 없음 | `camera_mat.npy` 로드 (fx=613.6, fy=613.0) |
| 위험도 반영 | 면적 → size_score | 거리 m 단위 → `clip(1 - (d-5)/45, 0, 1)` |
| 출력 | 없음 | 바운딩박스 위에 거리(m) 표시 |

### 2-7. 세그멘테이션 정밀도 향상 (Phase 5) — Midterm 대비

| 항목 | Midterm | Final |
|------|---------|-------|
| 검출 방식 | 단순 컨투어 | 컨투어 + **Watershed** (겹친 객체 분리) |
| 이상치 제거 | 없음 | **통계적 이상치 필터** — 중앙값 대비 5%~2000% 범위만 유지 |
| 보조 검출 | 없음 | Otsu + ConnectedComponents (컨투어 실패 시) |

### 2-8. 위험도 공식 고도화

| 항목 | Midterm | Final |
|------|---------|-------|
| 공식 | `0.5×크기 + 0.5×중앙` | `(0.45×크기 + 0.45×중앙 + 0.10×특징) × vis × ttc × lane` |
| 가중치 수 | 1개 (vis_weight) | 3개 (vis_weight × ttc_weight × lane_weight) |
| 특징점 반영 | 없음 | 10% 반영 |
| 차선 반영 | 없음 | ×1.1 / ×0.85 |
| TTC 반영 | 없음 | 최대 ×1.4 |

### 2-9. 출력 기능 확장 (Phase 6)

| 항목 | Midterm | Final |
|------|---------|-------|
| 출력 | 화면 표시 + PNG 저장 | PNG + **MP4 저장** + **CSV 통계 리포트** |
| 정보 표시 | 박스 + 점수 + 배너 | 박스 + 점수 + **거리** + **TTC** + **트랙ID** + **속도 화살표** + **클래스** |
| 디버그 | 없음 | 4분할 패널 (Gray / CLAHE / Edges+Lane / Result) |
| 비디오 처리 | 없음 | `process_video.py` — 동영상 배치 처리 |

---

## 3. 강의별 적용 기법

### Lec 3 — 영상처리 기초 → `preprocessing.py`

| 강의 내용 | 적용 위치 | 코드 |
|----------|----------|------|
| 히스토그램 평탄화 (`equalizeHist`) | 비교 기준 (Midterm) | — |
| **CLAHE** (Contrast Limited Adaptive HE) | `preprocess()` 내 조건부 적용 | `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` |
| Gaussian 블러 | 에지 검출 전 노이즈 제거 | `cv2.GaussianBlur(equalized, (5,5), 1.0)` |
| 색공간 변환 | 전처리 시작 | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` |
| 밝기·대비 분석 | 시인성 점수 산출 | `gray.mean()`, `gray.std()` |

### Lec 5 — 경계 검출 → `detection.py`, `lane.py`

| 강의 내용 | 적용 위치 | 코드 |
|----------|----------|------|
| **Canny Edge Detection** | 객체 윤곽 추출 | `cv2.Canny(blurred, 50, 150)` |
| Morphological 연산 (Close + Dilate) — Lec 4 | 엣지 연결 보강 | `cv2.morphologyEx(MORPH_CLOSE)` + `cv2.dilate` |
| 컨투어 추출 | 바운딩박스 생성 | `cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` |
| **Hough Transform** (`HoughLinesP`) | 차선 검출 | `cv2.HoughLinesP(ρ=1, θ=π/180, threshold=50)` |
| `cv2.fitLine` | 복수 선분 → 대표 직선 | `cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)` |
| ConnectedComponents | 보조 검출 | `cv2.connectedComponentsWithStats` |
| Otsu Thresholding | 보조 이진화 | `cv2.THRESH_OTSU` |

### Lec 6 — 이미지 세그멘테이션 → `detection.py`

| 강의 내용 | 적용 위치 | 코드 |
|----------|----------|------|
| **Watershed 세그멘테이션** | 겹친 객체 분리 | `cv2.watershed(roi_bgr, markers)` |
| Distance Transform | Watershed 전경 마커 생성 | `cv2.distanceTransform(fg, cv2.DIST_L2, 5)` |
| 마커 기반 Watershed 파이프라인 | IoU > 0.4 겹침 쌍에 적용 | sure_bg → sure_fg → unknown → markers |

Watershed 마커 생성 절차:
```
Otsu 이진화 → Dilate(3회) → sure_bg
             → DistanceTransform → 0.5×max 임계 → sure_fg
             → sure_bg - sure_fg → unknown
             → connectedComponents → markers + 1
             → markers[unknown==255] = 0
             → cv2.watershed
```

### Lec 7 — 특징점 검출 → `feature.py`, `risk.py`

| 강의 내용 | 적용 위치 | 코드 |
|----------|----------|------|
| **Harris Corner Detector** | 객체 ROI 코너 수 카운트 | `cv2.cornerHarris(float32(roi), blockSize=2, ksize=3, k=0.04)` |
| Harris 응답 임계 | 유의미한 코너만 집계 | `h > 0.01 × h.max()` |
| **SIFT** (Scale-Invariant Feature Transform) | 객체 ROI 키포인트 수 카운트 | `cv2.SIFT_create().detectAndCompute(roi, None)` |
| 특징점 수 → 위험도 | feature_score 산출 | `(harris + sift) / 500.0` → 10% 반영 |

Harris Corner 응답 공식: `R = det(A) - k·trace²(A)`, k = 0.04

### Lec 8 — 특징 매칭 & RANSAC → `detection.py`

| 강의 내용 | 적용 위치 | 코드 |
|----------|----------|------|
| **RANSAC 개념** (통계적 이상치 제거) | 허위 바운딩박스 제거 | 중앙값 대비 5%~2000% 범위 외 제거 |

### Lec 9 — 광학 흐름 → `tracking.py`

| 강의 내용 | 적용 위치 | 코드 |
|----------|----------|------|
| **Lucas-Kanade Optical Flow** | 프레임 간 키포인트 추적 | `cv2.calcOpticalFlowPyrLK` |
| Good Features to Track | 추적 키포인트 초기화 | `cv2.goodFeaturesToTrack(maxCorners=20, qualityLevel=0.3)` |
| 피라미드 LK | 다중 스케일 추적 | `winSize=(15,15), maxLevel=2` |
| **TTC (Time-To-Collision)** | 충돌 예상 시간 산출 | `TTC = area_prev / (area_now - area_prev)` |
| 속도 추정 | 중심점 이동 추적 | `centroid_history[-1] - centroid_history[-2]` |

LK 정규 방정식:
```
Ix·u + Iy·v + It = 0
AᵀA · [u,v]ᵀ = -Aᵀb   (최소제곱 풀이)
```

### Lec 10 — 카메라 기하 → `calibration.py`

| 강의 내용 | 적용 위치 | 코드 |
|----------|----------|------|
| **핀홀 카메라 모델** | 거리 추정 | `Z = (H_real × fy) / H_pixel` |
| 카메라 내부 파라미터 | `camera_mat.npy` 로드 | fx=613.6, fy=613.0, cx=331.2, cy=235.0 |
| 렌즈 왜곡 보정 | `undistort_image()` | `cv2.undistort(img, camera_mat, dist_coefs)` |
| 실제 크기 가정 | 클래스별 실제 높이 | 차량 1.5m, 보행자 1.7m |

### Lec 12 — 객체 검출 → `feature.py`, `classifier.py`

| 강의 내용 | 적용 위치 | 코드 |
|----------|----------|------|
| **HoG (Histogram of Oriented Gradients)** | 객체 복잡도 수치화 | `cv2.HOGDescriptor(_winSize=(64,64), _nbins=9)` |
| HoG 디스크립터 계산 | 64×64 ROI → 평균 크기 | `hog.compute(resized)` → `mean(abs(desc))` |
| **HOG + Pre-trained SVM** | 보행자 검출 | `HOGDescriptor_getDefaultPeopleDetector()` |
| `detectMultiScale` | 다중 스케일 검출 | `winStride=(8,8), scale=1.05, finalThreshold=2.0` |
| NMS (Non-Maximum Suppression) | 보행자 검출 후 중복 박스 제거 | 그리디 IoU NMS, overlap_thresh=0.4 |

HoG 파라미터:
```
winSize   = (64, 64)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize  = (8, 8)
nbins     = 9          ← 0°~180° 9개 방향 빈
```

---

## 4. 전체 아키텍처 및 데이터 흐름

### 4-1. 시스템 구성도

```
┌─────────────────────────────────────────────────────────────┐
│                    입력 소스                                  │
│   이미지 파일(.jpg/.png) ←→ main.py                          │
│   동영상 파일(.mp4)      ←→ process_video.py                 │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  [1] preprocessing.py — Phase 1-A                           │
│      BGR → Gray → vis_score → (CLAHE or copy)               │
│      → GaussianBlur → Canny                                 │
│      출력: gray, equalized, blurred, edges, vis_score        │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  [2] lane.py — Phase 1-B                                    │
│      edges → ROI(하단55%) → HoughLinesP                      │
│      → 기울기 분리 → fitLine → fillPoly                      │
│      출력: left_lines, right_lines, lane_mask                │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  [3] detection.py — Phase 5                                 │
│      ┌─ Canny + Morphology + findContours                   │
│      │   ↓ 검출 실패 시                                       │
│      └─ Otsu + ConnectedComponents                          │
│      → apply_watershed (IoU > 0.4 쌍 분리)                   │
│      → filter_statistical_outliers (중앙값 기준 필터)         │
│      출력: List[Detection]  (최대 10개)                       │
└─────────────────────────────────────────────────────────────┘
                         │
               ┌─────────┤
               ▼         ▼
┌──────────────────┐  ┌────────────────────────────────────────┐
│  classifier.py   │  │  feature.py — Phase 2-A               │
│  Phase 2-B       │  │  Harris: cornerHarris (k=0.04)         │
│  HOG+SVM 보행자  │  │  SIFT: SIFT_create().detectAndCompute  │
│  NMS 후 병합     │  │  HoG: HOGDescriptor (64×64, 9-bin)     │
└──────────────────┘  └────────────────────────────────────────┘
               │                   │
               └─────────┬─────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  [4] classify_object() — Phase 2-B                         │
│      aspect_ratio + (harris+sift) → vehicle / pedestrian    │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  [5] tracking.py — Phase 3                                  │
│      IoU 매칭 → 트랙 유지/생성/소멸                           │
│      LK Optical Flow: goodFeaturesToTrack → calcOpticalFlowPyrLK │
│      TTC = area_prev / Δarea                                │
│      출력: Track (track_id, velocity, ttc, centroid_history) │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  [6] calibration.py — Phase 4                               │
│      camera_mat.npy → fy = 613.0                            │
│      Z = (REAL_HEIGHTS[class] × fy) / det.h                 │
│      출력: distance_m (m 단위, None if unavailable)          │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  [7] risk.py — Phase 1-C 통합 위험도 평가                    │
│      size_score   = f(distance_m or area)                   │
│      center_score = 1 - dist(cx,cy, center) / √2            │
│      feature_score= 0.5×(H+S)/500 + 0.5×clip(hog×10,0,1)   │
│      vis_weight   = 1 + 0.3×(1 - vis_score)  [1.0~1.3]     │
│      ttc_weight   = 1 + 0.4×max(0,1-ttc/3)   [1.0~1.4]     │
│      lane_weight  = 1.1 (in_lane) or 0.85 (out)            │
│      score = base × vis × ttc × lane                        │
│      DANGER≥0.6 · CAUTION≥0.35 · SAFE<0.35                 │
│      출력: RiskResult (level, score, distance_m, ttc, ...)  │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  [8] visualize.py — Phase 6                                 │
│      draw_lanes → 박스 + 클래스 + 점수 + 거리 + TTC          │
│      속도 화살표 + 트랙ID + HUD 패널 + DANGER 배너            │
│      (debug) make_debug_panel: 2×2 4분할                    │
└─────────────────────────────────────────────────────────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
           PNG 저장    MP4 저장   CSV 저장
```

### 4-2. 모듈 파일 구조

```
Final_exam/
├── src/
│   ├── main.py            이미지 뷰어 모드 진입점
│   ├── process_video.py   동영상 배치 처리 진입점
│   ├── preprocessing.py   [Phase 1-A] CLAHE 전처리
│   ├── lane.py            [Phase 1-B] Hough 차선 검출
│   ├── detection.py       [Phase 5]   객체 검출 · Watershed · 이상치 제거
│   ├── feature.py         [Phase 2-A] Harris · SIFT · HoG 추출
│   ├── classifier.py      [Phase 2-B] HOG+SVM 보행자 · 분류
│   ├── tracking.py        [Phase 3]   LK 추적 · TTC
│   ├── calibration.py     [Phase 4]   핀홀 거리 추정
│   ├── risk.py            [Phase 1-C] 통합 위험도 평가
│   └── visualize.py       [Phase 6]   시각화 · 디버그 패널
└── output/
    ├── open_dataset/
    │   ├── result.mp4          주행 데이터셋 결과 영상
    │   ├── stats.csv           프레임별 통계
    │   ├── danger_moments/     TTC 위험 TOP 12 장면
    │   └── overview/           전체 9장 요약
    ├── result_*.png            이미지 처리 결과
    └── stats.csv               이미지 모드 통계
```

---

## 5. 모듈별 알고리즘 상세

### preprocessing.py — CLAHE 전처리

```python
# 시인성 점수 (0~1)
brightness_score = 1.0 - |mean - 128| / 128     # 128에 가까울수록 좋음
contrast_score   = min(std / 64.0, 1.0)          # 표준편차가 클수록 좋음
vis_score        = 0.5 × brightness + 0.5 × contrast

# CLAHE 조건부 적용
if vis_score < 0.5:                              # 야간·역광·안개
    clahe     = CLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray)
else:
    equalized = gray                              # 정상 조명 → 원본 유지

blurred = GaussianBlur(equalized, (5,5), σ=1.0)
edges   = Canny(blurred, low=50, high=150)
```

### lane.py — Hough 차선 검출

```python
# 1. ROI 마스킹 (상단 45% 제거)
roi_edges[:int(h*0.45), :] = 0

# 2. 확률적 Hough Transform
lines = HoughLinesP(roi_edges, ρ=1, θ=π/180,
                    threshold=50, minLineLength=80, maxLineGap=60)

# 3. 기울기로 좌/우 분리
slope = (y2 - y1) / (x2 - x1)
if |slope| < 0.3: continue          # 수평선 제거
left  ← slope < 0
right ← slope > 0

# 4. fitLine으로 대표 직선 생성
params = fitLine(pts, DIST_L2, ...)  # [vx, vy, x0, y0]
y_bot = h,  y_top = h×0.55
t = (y - y0) / vy  →  x = x0 + t×vx

# 5. fillPoly로 차선 내부 마스크
cv2.fillPoly(mask, [left_top, right_top, right_bot, left_bot], 255)
```

### detection.py — 객체 검출 + Watershed

**1단계: Canny 기반 검출**
```python
closed  = morphologyEx(edges, MORPH_CLOSE, kernel, iterations=2)
dilated = dilate(closed, kernel, iterations=1)
contours = findContours(dilated, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
# 필터: 면적 0.1%~15%, 높이 ≥ 15px, 하단 30% 이내
```

**2단계: Watershed (겹친 객체 분리)**
```python
# IoU > 0.4 인 쌍 탐지 → ROI 추출
# Otsu 이진화 → DistanceTransform → 마커 생성 → cv2.watershed
# 분리된 영역별 새 바운딩박스 생성
```

**3단계: 통계적 이상치 제거 (RANSAC 개념)**
```python
median_area = np.median([d.area for d in detections])
keep = [d for d in detections
        if median_area × 0.05 ≤ d.area ≤ median_area × 20]
```

### feature.py — Harris · SIFT · HoG

```python
# Harris Corner
h = cornerHarris(float32(roi), blockSize=2, ksize=3, k=0.04)
count = sum(h > 0.01 × h.max())    # R = det(A) - k·trace²(A)

# SIFT
kp, _ = SIFT_create().detectAndCompute(roi, None)
count = len(kp)                     # DoG 스케일-공간 극값점

# HoG (64×64, 9-bin)
resized = resize(roi, (64, 64))
desc = HOGDescriptor(winSize=(64,64), blockSize=(16,16),
                     blockStride=(8,8), cellSize=(8,8), nbins=9).compute(resized)
score = mean(|desc|)                # 구조 복잡도 지표
```

### tracking.py — LK 추적 + TTC

```python
# 초기화: Good Features to Track
pts = goodFeaturesToTrack(roi, maxCorners=20, qualityLevel=0.3, minDistance=5)

# 추적: Lucas-Kanade Pyramidal
pts_next, status, _ = calcOpticalFlowPyrLK(
    prev_gray, gray, pts, None,
    winSize=(15,15), maxLevel=2,
    criteria=(EPS|COUNT, 10, 0.03))

# IoU 매칭 (프레임 간 박스 연결)
iou_thresh = 0.25
# 매칭 → 트랙 업데이트 / missed++ / 신규 생성 (max_missed=5 초과 시 소멸)

# TTC
TTC = area_history[-2] / (area_history[-1] - area_history[-2])
TTC = clip(TTC, 0.5, 99.0)   # 물리적 범위 클리핑
```

### calibration.py — 핀홀 거리 추정

```
핀홀 카메라 모델:
  픽셀 좌표 (u, v) ↔ 3D 좌표 (X, Y, Z)

  u = fx × X/Z + cx
  v = fy × Y/Z + cy

거리 추정 (단안 카메라):
  Z = (H_real × fy) / H_pixel

  H_real:  차량 1.5m, 보행자 1.7m  (클래스별 평균 높이)
  fy:      613.0 (캘리브레이션)
  H_pixel: 바운딩박스 높이 (픽셀)

카메라 내부 파라미터 (Data/pinhole_calib/camera_mat.npy):
  ┌ fx  0   cx ┐   ┌ 613.6   0    331.2 ┐
  │  0  fy  cy │ = │   0   613.0  235.0 │
  └  0   0   1 ┘   └   0     0      1   ┘
```

### risk.py — 위험도 평가

```python
# size_score (0~1): 얼마나 가까운가
if distance_m is not None:
    size_score = clip(1.0 - (distance_m - 5.0) / 45.0, 0, 1)
    # 5m → 1.0,  50m → 0.0
else:
    size_score = min(det.area / (img_area × 0.08), 1.0)

# center_score (0~1): 화면 중앙에 얼마나 가까운가
dx = (cx - w/2) / (w/2)
dy = (cy - h/2) / (h/2)
center_score = 1.0 - clip(√(dx²+dy²) / √2, 0, 1)

# feature_score (0~1): 구조적 복잡도
feat   = clip((harris + sift) / 500.0, 0, 1)
hog_c  = clip(hog_score × 10.0, 0, 1)
feature_score = 0.5×feat + 0.5×hog_c

# 가중치
vis_weight  = 1.0 + 0.3×(1 - vis_score)          # [1.00~1.30]
ttc_weight  = 1.0 + 0.4×max(0, 1 - ttc/3)        # [1.00~1.40]
lane_weight = 1.1 if in_lane else 0.85            # [0.85~1.10]

# 최종 점수
base  = 0.45×size + 0.45×center + 0.10×feature
score = clip(base × vis_weight × ttc_weight × lane_weight, 0, 1)

# 등급 분류
DANGER  : score ≥ 0.60
CAUTION : score ≥ 0.35
SAFE    : score < 0.35
```

---

## 6. 위험도 평가 공식

```
                  ┌──────────────────────────────────────────────┐
                  │   base = 0.45×size + 0.45×center + 0.10×feat │
                  └──────────────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
             vis_weight          ttc_weight         lane_weight
            1.0 + 0.3×(1-v)   1.0 + 0.4×(1-t/3)   1.1 / 0.85
            [1.00 ~ 1.30]      [1.00 ~ 1.40]       차선 내/외
                    └──────────────────┼──────────────────┘
                                       ▼
                         score = base × vis × ttc × lane
                                       │
                      ┌────────────────┼────────────────┐
                   ≥0.60            ≥0.35              <0.35
                  DANGER           CAUTION             SAFE
                 (빨간색)          (주황색)            (초록색)
```

---

## 7. 실행 방법

### 이미지 뷰어 모드 (`main.py`)

```bash
cd Final_exam/src

# 기본 실행 (Data 디렉토리 전체 이미지)
python3 main.py --data ../../Data

# 4분할 디버그 패널 (Gray / CLAHE / Edges+Lane / Result)
python3 main.py --data ../../Data --debug

# 결과 저장 (PNG + MP4 + CSV)
python3 main.py --data ../../Data --save --video --csv

# 단일 이미지 + 디버그
python3 main.py --data ../../Data/people.jpg --debug

# 보행자 검출 비활성화 (빠른 처리)
python3 main.py --data ../../Data --no-ped
```

| 키 | 동작 |
|----|------|
| `d` / `→` | 다음 이미지 |
| `a` / `←` | 이전 이미지 |
| `s` | 현재 결과 저장 |
| `q` | 종료 |

### 비디오 처리 모드 (`process_video.py`)

```bash
# 주행 영상 처리 (3프레임마다 1회 처리)
python3 process_video.py --input <video.mp4> --skip 3

# 빠른 테스트 (100프레임, 보행자 검출 비활성화)
python3 process_video.py --input <video.mp4> --max 100 --no-ped
```

출력: `<입력명>_result.mp4` + `<입력명>_stats.csv`

### 공개 데이터셋 다운로드 및 처리

```bash
pip install yt-dlp
mkdir -p Data/open_dataset && cd Data/open_dataset

# YouTube CC 라이선스 주행 영상 다운로드
yt-dlp -f "best[height<=480][ext=mp4]" -o "driving.%(ext)s" "<YouTube URL>"

# 파이프라인 적용
python3 ../../Final_exam/src/process_video.py --input driving.mp4 --skip 3
```

---

## 8. 공개 주행 데이터셋 검증 결과

**데이터셋:** YouTube CC 라이선스 대시캠 주행 영상 (72초, 640×360, 24fps)

### 8-1. 처리 성능

| 항목 | 결과 |
|------|------|
| 처리 프레임 | 578프레임 (skip=3, 원본 1,734프레임) |
| 처리 속도 | **~50 fps** (실시간 24fps의 약 2배) |
| 처리 시간 | 11.5초 (72초 영상 기준) |

### 8-2. 검출 결과

| 항목 | 결과 |
|------|------|
| 평균 객체 수 | **5.3개 / 프레임** |
| 차량(vehicle) 검출 | **2,911건** |
| 보행자(pedestrian) 검출 | **160건** |
| DANGER 누적 | **352건** |
| CAUTION 누적 | **1,206건** |
| SAFE 누적 | **1,513건** |

### 8-3. 거리 추정 결과 (핀홀 모델)

| 항목 | 결과 |
|------|------|
| 최근접 거리 | **4.1m** |
| 평균 거리 | 18.1m |
| 최원 거리 | 61.3m |

### 8-4. TTC (Time-To-Collision) 결과

| 항목 | 결과 |
|------|------|
| TTC 산출 구간 | **278프레임 / 578프레임 (48.1%)** |
| 평균 TTC | 2.53s |
| **충돌 임박 (TTC ≤ 1s)** | **186회** |
| TTC 경고 (1s < TTC ≤ 3s) | 48회 |

> TTC는 동일 객체를 연속 프레임에서 추적할 때만 산출됩니다.  
> 단일 이미지 테스트에서는 N/A로 표시됩니다.

### 8-5. 시인성 분포

| 등급 | 프레임 수 | 비율 |
|------|----------|------|
| Good (≥ 0.60) | 578 | 100% |
| Moderate | 0 | — |
| Poor | 0 | — |

> 해당 영상은 주간 양호 조명 — CLAHE가 적용되지 않은 정상 케이스

### 8-6. 저장된 결과물

```
Final_exam/output/open_dataset/
├── result.mp4          주석 처리된 결과 영상 (21MB)
├── stats.csv           578행 프레임별 통계
├── danger_moments/     TTC 위험 TOP 12 장면 PNG
│   ├── danger_01_f1039_TTC0.5s_7m.png
│   └── ... (12장)
└── overview/           8초 간격 전체 요약 PNG (9장)
    ├── overview_01_f0000_0s.png
    └── ...
```

**결과 영상 재생 (WSL 환경):**
```bash
explorer.exe "$(wslpath -w Final_exam/output/open_dataset/result.mp4)"
```

---

## 9. 의존성 및 파라미터

### 환경

```
Python  3.x
OpenCV  4.13.0
NumPy
```

### 카메라 캘리브레이션 파라미터

파일: `Data/pinhole_calib/camera_mat.npy`

```
camera_mat = [[613.6,   0,   331.2],
              [  0,   613.0, 235.0],
              [  0,     0,     1  ]]
```

### CLI 옵션 전체

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--data` | `../../Data` | 이미지 파일 또는 디렉토리 경로 |
| `--debug` | False | 4분할 전처리 디버그 패널 |
| `--save` | False | 결과 PNG를 output/에 저장 |
| `--video` | False | output/result.mp4 저장 |
| `--csv` | False | output/stats.csv 저장 |
| `--max` | 0 (전체) | 최대 처리 이미지/프레임 수 |
| `--calib` | `Data/pinhole_calib` | 캘리브레이션 디렉토리 |
| `--no-ped` | False | HOG 보행자 검출 비활성화 |
| `--skip` | 3 | 비디오 N프레임마다 1회 처리 |
