from utils import *
from dataset import *
from model import *
from eval import *
import os
import torch
import wandb
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from transformers import get_scheduler

def train_loop(rank, world_size, config):
    # import socket
    # def find_free_port():
    #     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #         s.bind(('', 0))
    #        return s.getsockname()[1]
    # os.environ['MASTER_PORT'] = str(find_free_port())  # 빈 포트 자동 할당
    
    # === 1. DDP 초기화 ===
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # === 2. seed & label map 설정 ===
    set_seed(config['seed'])
    config['label_maps'] = get_label_maps_from_config(config)

    # === 3. Dataset + Sampler ===
    train_dataset = get_dataset(config, split='train')
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler,
                              num_workers=4, pin_memory=True)

    if rank == 0:
        val_dataset = get_dataset(config, split='val')
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                                num_workers=4, pin_memory=True)
    else:
        val_loader = None

    # === 4. 모델 ===
    model_wrapper = MultiLabelImageClassifier(
        num_actions=len(config['label_maps']['action']),
        num_emotions=len(config['label_maps']['emotion']),
        num_situations=len(config['label_maps']['situation']),
        backbone_name=config['model_name'],
        pretrained=config.get('pretrained', True)
    )
    model = model_wrapper.get_model().to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # === 5. Optimizer, Scheduler, Criterion ===
    optimizer = AdamW(model.parameters(), lr=float(config['learning_rate']))
    num_training_steps = config['epochs'] * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)
    criterion = nn.CrossEntropyLoss()

    # === 6. wandb는 rank==0에서만 ===
    if rank == 0:
        run_name = f"{config['model_name']}_lr{config['learning_rate']}_bs{config['batch_size']}_ep{config['epochs']}"
        wandb.init(project=config['wandb_project'], name=run_name, config=config)

    best_score = 0.0
    for epoch in range(config['epochs']):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0

        for images, y_action, y_emotion, y_situation in train_loader:
            images = images.to(rank)
            y_action = y_action.to(rank)
            y_emotion = y_emotion.to(rank)
            y_situation = y_situation.to(rank)

            optimizer.zero_grad()
            a_logits, e_logits, s_logits = model(images)
            loss = (
                criterion(a_logits, y_action) +
                criterion(e_logits, y_emotion) +
                criterion(s_logits, y_situation)
            )
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()

        # === 검증: 0번 rank에서만 수행 ===
        if rank == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"\n[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

            val_loss, macro_f1, micro_f1, partial_score, exact_match_acc, label_wise_acc = evaluate_model_val(
                model.module, val_loader, rank
            )
            print(
                f"[Epoch {epoch}] Val Loss: {val_loss:.4f} | Macro F1: {macro_f1:.4f} | "
                f"Micro F1: {micro_f1:.4f} | Exact Match Acc: {exact_match_acc:.4f}"
            )

            wandb.log({
                'epoch': epoch,
                'train_loss': avg_loss,
                'val_loss': val_loss,
                'macro_f1': macro_f1,
                'micro_f1': micro_f1,
                'partial_score': partial_score,
                'exact_match_acc': exact_match_acc,
                'label_wise_acc/action': label_wise_acc['action'],
                'label_wise_acc/emotion': label_wise_acc['emotion'],
                'label_wise_acc/situation': label_wise_acc['situation'],
            })

            # === 모델 저장 ===
            if macro_f1 > best_score:
                best_score = macro_f1
                save_best_model(
                    model.module,  # DDP 감싸져 있으니 module로 전달
                    save_dir=config['save_path'],
                    base_name=config['model_name'],
                    epoch=epoch,
                    val_loss=val_loss,
                    score=best_score,
                )

    dist.destroy_process_group()