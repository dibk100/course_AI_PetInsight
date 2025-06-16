import argparse
from video_utils import extract_frames
from vision_model import predict_emotion_behavior
from llm_generator import generate_narrative

def main(video_path):
    print("[1] 영상에서 프레임 추출 중...")
    frames = extract_frames(video_path)

    print(f"[2] 총 {len(frames)}개의 프레임 분석 중...")
    all_tags = set()
    for frame in frames:
        tags = predict_emotion_behavior(frame)
        all_tags.update(tags)

    print(f"[3] 추출된 태그: {list(all_tags)}")
    print("[4] LLM으로 자연어 생성 중...")
    narrative = generate_narrative(list(all_tags))

    print("\n📣 반려동물의 말:")
    print(narrative)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="반려동물 행동 분석 CLI")
    parser.add_argument("video", help="분석할 영상 파일 경로")
    args = parser.parse_args()

    main(args.video)
