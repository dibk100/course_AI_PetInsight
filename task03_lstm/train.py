import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import wandb
import torch.nn as nn
from utils import *
from dataset import *
from model import *
from eval import *
import os

def train_model(config):
    set_seed(config['seed'])
    device = config['device']

    
    # 자동 run_name 생성
    run_name = f"{config['model_name']}_lr{config['learning_rate']}_bs{config['batch_size']}_ep{config['epochs']}"
    wandb.init(project=config['wandb_project'], name=run_name, config=config)

    # 데이터셋 & DataLoader
    train_dataset = get_dataset(config, split='train')
    val_dataset = get_dataset(config, split='val')

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4,collate_fn=collate_fn,pin_memory=True)      # pin_memory=True :: GPU 사용시 병목 줄임
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn,num_workers=4,pin_memory=True)

    # 모델 초기화
    model_wrapper = MultiLabelVideoTransformerClassifier(
        num_actions=len(config['label_names']['action']),
        num_emotions=len(config['label_names']['emotion']),
        num_situations=len(config['label_names']['situation']),
        backbone_name=config['model_name'],
        pretrained=config.get('pretrained', True)
    )
    model = model_wrapper.get_model()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=float(config['learning_rate']))
    num_training_steps = config['epochs'] * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    criterion = nn.CrossEntropyLoss()

    best_score = 0

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0

        for frames, y_action, y_emotion, y_situation in train_loader:
            optimizer.zero_grad()
            
            frames = frames.to(device)
            y_action = y_action.to(device)
            y_emotion = y_emotion.to(device)
            y_situation = y_situation.to(device)
            outputs_action, outputs_emotion, outputs_situation = model(frames)
    
            loss_a = criterion(outputs_action, y_action)
            loss_e = criterion(outputs_emotion, y_emotion)
            loss_s = criterion(outputs_situation, y_situation)

            loss = loss_a + loss_e + loss_s
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch} | Train Loss: {avg_train_loss:.4f}")

        # 검증 평가
        val_loss, macro_f1, micro_f1, partial_score, exact_match_acc, label_wise_acc = evaluate_model_val(model, val_loader, config['device'])
        print(
            f"Epoch {epoch} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Macro F1: {macro_f1:.4f} | "
            f"Micro F1: {micro_f1:.4f} | "
            f"Partial Match Score: {partial_score:.4f} | "
            f"Exact Match Acc: {exact_match_acc:.4f} | "
            f"Label-wise Acc: {label_wise_acc}"
        )
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'partial_score': partial_score,
            'exact_match_acc': exact_match_acc,
            'label_wise_acc/action': label_wise_acc['action'],
            'label_wise_acc/emotion': label_wise_acc['emotion'],
            'label_wise_acc/situation': label_wise_acc['situation'],
        })

        # ✅ Macro F1 기준으로 모델 저장
        if macro_f1 > best_score:
            best_score = macro_f1
            save_best_model(
                model,
                save_dir=config['save_path'],
                base_name=config['model_name'],
                epoch=epoch,
                val_loss=val_loss,
                score=best_score,
            )
