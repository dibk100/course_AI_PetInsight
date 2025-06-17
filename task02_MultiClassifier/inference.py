import torch
from torchvision import transforms
from PIL import Image
import yaml
import argparse
import os
from pathlib import Path
from model import *
from utils import *
from collections import Counter

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # (1, C, H, W)

@torch.no_grad()
def inference_on_folder(image_dir: Path, config):
    device = config['device']
    label_maps = get_label_maps_from_config(config)

    model_wrapper = MultiLabelImageClassifier(
        num_actions=len(label_maps['action']),
        num_emotions=len(label_maps['emotion']),
        num_situations=len(label_maps['situation']),
        backbone_name=config['model_name'],
        pretrained=False
    )
    model = model_wrapper.get_model()
    model_path = os.path.join(config['save_path'], config['model_name'],config['best_model_path'])
    assert os.path.exists(model_path), f"모델 경로 {model_path}가 존재하지 않습니다."

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    inv_label_maps = {
        key: {v: k for k, v in label_maps[key].items()}
        for key in label_maps
    }

    image_paths = sorted([p for p in image_dir.glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    if not image_paths:
        print(f"[ERROR] No images found in {image_dir}")
        return

    action_counter = Counter()
    emotion_counter = Counter()
    situation_counter = Counter()

    for image_path in image_paths:
        image_tensor = load_image(image_path).to(device)
        action_logits, emotion_logits, situation_logits = model(image_tensor)

        action_idx = torch.argmax(action_logits, dim=1).item()
        emotion_idx = torch.argmax(emotion_logits, dim=1).item()
        situation_idx = torch.argmax(situation_logits, dim=1).item()

        action = inv_label_maps['action'][action_idx]
        emotion = inv_label_maps['emotion'][emotion_idx]
        situation = inv_label_maps['situation'][situation_idx]

        print(f"\n[{image_path.name}]")
        print(f"  Action:    {action}")
        print(f"  Emotion:   {emotion}")
        print(f"  Situation: {situation}")

        action_counter[action] += 1
        emotion_counter[emotion] += 1
        situation_counter[situation] += 1

    # Summary
    print("\n===== Inference Summary =====")
    def print_summary(name, counter):
        print(f"\n{name} Frequency:")
        for label, count in counter.most_common():
            print(f"  {label:<12}: {count}")

    print_summary("Action", action_counter)
    print_summary("Emotion", emotion_counter)
    print_summary("Situation", situation_counter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-label inference on a folder of images')
    parser.add_argument('--input_dir', type=str, required=True, help='Folder containing images for inference')
    parser.add_argument('--config', type=str, required=True, help='YAML config file path')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    inference_on_folder(Path(args.input_dir), config)
