import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from project_cat_behavior.dataset import get_dataloaders
from train import train
from eval import evaluate

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(model_name='resnet18', num_classes=10).to(device)                  ## 일단 resnet18으로 테스트
    train_loader, test_loader = get_dataloaders(batch_size=64)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Acc: {test_acc:.2f}%")

if __name__ == "__main__":
    main()