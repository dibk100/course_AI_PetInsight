# Task02 우리가 학습한 모델로 번역어 생성

def generate_narrative(tags):

    prompt = f"다음 감정/행동 정보로 반려동물이 무슨 말을 할지 자연스럽게 표현해줘: {', '.join(tags)}"

    # 더미 출력 예시
    tag_map = {
        "긴장": "저기 낯선 사람 무서워...",
        "낯선 사람": "저 사람 누구야? 낯설어.",
        "짖음": "워워! 접근하지 마!",
    }

    result = [tag_map.get(tag, f"{tag}에 대한 반응이 필요해") for tag in tags]
    return " ".join(result)
