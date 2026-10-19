import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


class GestureCNN(nn.Module):
    def __init__(self):
        super(GestureCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Первый свёрточный слой
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Второй свёрточный слой
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Пулинг
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Линейный слой
        self.fc2 = nn.Linear(128, 4)  # Выходной слой (4 класса жестов)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Первый свёрточный слой + пулинг
        x = self.pool(F.relu(self.conv2(x)))  # Второй свёрточный слой + пулинг
        x = x.view(-1, 64 * 56 * 56)  # Преобразуем тензор в плоский вектор
        x = F.relu(self.fc1(x))  # Полносвязный слой
        x = self.fc2(x)  # Выходной слой
        return x
    

# Создаём модель
model = GestureCNN()

# Случайное изображение (batch_size=1, 3 канала, 224x224)
input_image = torch.randn(1, 3, 224, 224)

# Прогоняем изображение через модель
outputs = model(input_image)

# Проверяем размер выходных данных
print(f"Выходной размер: {outputs.shape}")  # Ожидается: torch.Size([1, 4])
