import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
import onnxruntime as ort
import numpy as np
import random
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from tqdm import tqdm
import json

# --- 1. Конфигурация (Dataclass) ---
@dataclass
class Config:
    # Путь к данным
    data_dir: str = "../data/processed"
    output_dir: str = "../app"
    
    # Гиперпараметры
    model_name: str = "resnet18"  # или "efficientnet_b0", "mobilenetv3_large_100"
    num_classes: int = 4
    batch_size: int = 8  # Маленький батч, т.к. данных мало
    epochs: int = 15
    lr: float = 0.001
    image_size: int = 224
    
    # Воспроизводимость
    seed: int = 42

# --- 2. Фиксация случайных чисел ---
def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 3. Подготовка данных ---
def get_data_loaders(cfg: Config):
    # Аугментация для тренировки
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(cfg.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Без аугментаций для валидации
    val_transforms = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(cfg.data_dir, transform=train_transforms)
    
    # Разделение на train/val (80/20)
    total_count = len(full_dataset)
    val_count = int(total_count * 0.4)
    train_count = total_count - val_count
    
    train_dataset, val_dataset = random_split(full_dataset, [train_count, val_count], 
                                              generator=torch.Generator().manual_seed(cfg.seed))
    
    # Применяем валидационные трансформации к val выборке
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    return train_loader, val_loader, full_dataset.classes

# --- 4. Модель ---
def create_model(cfg: Config):
    model = timm.create_model(cfg.model_name, pretrained=True, num_classes=cfg.num_classes)
    
    # Стратегия заморозки: замораживаем все, кроме головы (слоя классификации)
    for param in model.parameters():
        param.requires_grad = False
        
    # Размораживаем последний слой (или голову, в зависимости от архитектуры timm)
    # Для ResNet это model.fc
    if hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
    # Для EfficientNet это model.classifier
    elif hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
            
    return model

# --- 5. Экспорт в ONNX ---
def export_to_onnx(model, cfg: Config, class_names):
    model.eval()
    dummy_input = torch.randn(1, 3, cfg.image_size, cfg.image_size)
    
    onnx_path = Path(cfg.output_dir) / "best_model.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        str(onnx_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {onnx_path}")
    
    # Сохраняем классы для приложения
    with open(Path(cfg.output_dir) / "classes.json", "w") as f:
        json.dump(class_names, f)

# --- 6. Обучение ---
def train(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, classes = get_data_loaders(cfg)
    model = create_model(cfg).to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Обучаем только параметры с requires_grad=True (голову)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    
    history = {'train_loss': [], 'val_acc': []}
    best_acc = 0.0

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        
        # Train loop
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        
        # Validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # Сохраняем лучшую модель (state_dict)
            ckpt_path = Path(cfg.output_dir) / "best_checkpoint.pth"
            torch.save(model.state_dict(), ckpt_path)

    # Финальный экспорт лучшей модели
    # Загружаем лучшие веса перед экспортом
    model.load_state_dict(torch.load(Path(cfg.output_dir) / "best_checkpoint.pth"))
    export_to_onnx(model, cfg, classes)
    
    # Сохраняем историю обучения
    with open(Path(cfg.output_dir) / "training_history.json", "w") as f:
        json.dump(history, f)

if __name__ == "__main__":
    # Можно передавать параметры, если нужно, но здесь используем дефолтные из dataclass
    config = Config()
    # Если хотим поменять модель для эксперимента:
    # config.model_name = "efficientnet_b0" 
    train(config)
