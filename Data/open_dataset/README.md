# Open Dataset — 공개 주행 영상 데이터셋

Final_exam 파이프라인 검증에 사용한 공개 주행 영상입니다.

## 파일 목록

| 파일 | 크기 | 설명 |
|------|------|------|
| `driving_2CIxM7x-Clc.mp4` | 5.9MB | 원본 대시캠 주행 영상 |
| `samples/` | 1.9MB | 원본 영상 균등 간격 샘플 프레임 7장 |

## 영상 정보

| 항목 | 값 |
|------|-----|
| 해상도 | 640 × 360 |
| FPS | 24 |
| 총 프레임 | 1,734 |
| 길이 | 72.3초 |
| 라이선스 | YouTube CC |

## 처리 결과

파이프라인 적용 결과는 `Final_exam/output/open_dataset/` 에 저장됩니다:

```
Final_exam/output/open_dataset/
├── result.mp4          주석 처리 결과 영상 (578프레임, skip=3)
├── stats.csv           프레임별 위험도 통계 (578행)
├── danger_moments/     TTC 위험 TOP 12 장면
└── overview/           8초 간격 요약 9장
```

## 재처리 방법

```bash
cd Final_exam/src
python3 process_video.py \
  --input ../../Data/open_dataset/driving_2CIxM7x-Clc.mp4 \
  --skip 3
```

## 다른 영상 추가 방법

```bash
pip install yt-dlp
cd Data/open_dataset
yt-dlp -f "best[height<=480][ext=mp4]" -o "driving.%(ext)s" "<YouTube URL>"
python3 ../../Final_exam/src/process_video.py --input driving.mp4 --skip 3
```
