import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from project_cat_behavior.dataset import get_dataloaders

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total, correct, total_loss = 0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    acc = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, acc