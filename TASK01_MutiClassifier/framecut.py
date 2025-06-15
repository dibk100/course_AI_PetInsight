from pathlib import Path
import cv2
from tqdm import tqdm
import numpy as np


def extract_keyframes_from_video(
    video_path: Path,
    output_dir: Path,
    num_samples: int = 100,
    skip_tail_seconds: int = 20,
    img_extension: str = ".jpg",
    jpeg_quality: int = 95
) -> None:

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        print(f"[ERROR] Invalid FPS or frame count in video: {video_path}")
        cap.release()
        return

    duration_sec = total_frames / fps
    usable_duration = max(duration_sec - skip_tail_seconds, 0)

    timestamps = np.linspace(0, usable_duration, num=num_samples)

    save_dir = output_dir / video_path.stem
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx, t in enumerate(tqdm(timestamps,
                                 desc=f"Extracting [{video_path.stem}]",
                                 unit="frame",
                                 dynamic_ncols=True)):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            print(f"[WARNING] Failed to read frame at {t:.2f}s in {video_path.name}")
            continue

        filename = save_dir / f"frame_{idx:03d}{img_extension}"
        if img_extension.lower() == ".jpg":
            cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        else:
            cv2.imwrite(str(filename), frame)

    cap.release()
    print(f"[DONE] Extracted {num_samples} frames from {video_path.name} to '{save_dir}'")


def batch_extract_keyframes(
    input_dir: Path,
    output_dir: Path,
    num_samples: int = 100,
    skip_tail_seconds: int = 20,
    img_extension: str = ".jpg",
    jpeg_quality: int = 95,
    video_pattern: str = "*.mp4"
) -> None:
    video_files = sorted(input_dir.glob(video_pattern))
    if not video_files:
        print(f"[ERROR] No videos found in {input_dir} with pattern '{video_pattern}'")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for video_file in video_files:
        extract_keyframes_from_video(
            video_file,
            output_dir,
            num_samples=num_samples,
            skip_tail_seconds=skip_tail_seconds,
            img_extension=img_extension,
            jpeg_quality=jpeg_quality,
        )

if __name__ == "__main__":
    # 기본 실행 시 설정값만 변경하면 됨
    INPUT_DIR = Path("./inference/test")
    OUTPUT_DIR = Path("./inference/cat_frames")
    NUM_SAMPLES = 100
    SKIP_TAIL_SECONDS = 20
    IMG_EXTENSION = ".jpg"
    JPEG_QUALITY = 95

    batch_extract_keyframes(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        num_samples=NUM_SAMPLES,
        skip_tail_seconds=SKIP_TAIL_SECONDS,
        img_extension=IMG_EXTENSION,
        jpeg_quality=JPEG_QUALITY,
        video_pattern="*.mp4",
    )
