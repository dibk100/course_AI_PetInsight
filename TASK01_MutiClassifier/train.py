from model import *
from utils import *
from dataset import *
from torch.optim import AdamW
import wandb
from transformers import get_scheduler

def train_model(config):
    set_seed(config['seed'])
    
    # 자동 run_name 생성
    run_name = f"{config['model_name']}_lr{config['learning_rate']}_bs{config['batch_size']}_ep{config['epochs']}"
    wandb.init(project=config['wandb_project'], name=run_name, config=config)

    # 데이터셋 & DataLoader
    train_dataset = get_dataset(config, split='train')
    val_dataset = get_dataset(config, split='val')

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # 모델 초기화
    model_wrapper = MultiLabelImageClassifier(config['num_actions'],config['num_emotions'],config['num_situations'],config['model_name'])
    model = model_wrapper.get_model()
    model.to(config['device'])


    optimizer = AdamW(model.parameters(), lr=float(config['learning_rate']))
    num_training_steps = config['epochs'] * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    criterion = nn.CrossEntropyLoss()
    
    # best_val_loss = float('inf')
    best_score = 0

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for images, y_action, y_emotion, y_situation in train_loader:
                optimizer.zero_grad()
                a_logits, e_logits, s_logits = model(images)

                loss_a = criterion(a_logits, y_action)
                loss_e = criterion(e_logits, y_emotion)
                loss_s = criterion(s_logits, y_situation)

                loss = loss_a + loss_e + loss_s
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

        # 검증 평가
        val_loss, macro_f1, micro_f1, partial_score,exact_match_acc, label_wise_acc  = evaluate_model_val(model, val_loader, config['device'])
        print(f"Epoch {epoch} | Val Loss: {val_loss:.4f} | Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}| Partial Match Score: {partial_score:.4f}| Exact Match Acc: {exact_match_acc:.4f}| Label Wise Acc: {label_wise_acc:.4f}")

        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'partial_score' : partial_score,
            'exact_match_acc' : exact_match_acc,
            'label_wise_acc' : label_wise_acc
        })

        if partial_score > best_score:
            best_score = partial_score
            save_best_model(
                model,
                save_dir=config['save_path'],
                base_name=config['model_name'],
                epoch=epoch,
                val_loss=val_loss,
                partial_score = partial_score,
            )