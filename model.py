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


if __name__ == "__main__":
    # Параметры обучения
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")
    model = GestureCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Подготовка данных
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder("data", transform=transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # Цикл обучения
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Обнуление градиентов
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Эпоха {epoch + 1}, Потеря: {running_loss / len(train_loader)}")

    # Сохранение модели
    torch.save(model.state_dict(), "gesture_model.pth")


    # Загрузка обученной модели
    model.load_state_dict(torch.load("gesture_model.pth"))
    model.eval()

    # Реализация распознавания жестов
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Преобразуем кадр для модели
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = transforms.ToTensor()(img).unsqueeze(0).to(device)

        # Предсказание
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)

        # Интерпретация результата
        gesture = train_data.classes[predicted.item()]
        print(f"Распознан жест: {gesture}")

        # Отображение видеопотока
        cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
