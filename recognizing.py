import cv2
import torch
from torchvision import transforms
import pycaw
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import numpy as np
import time
from model import GestureCNN  # Импорт нашей модели

# Настраиваем управление громкостью
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Инициализация модели
model = GestureCNN()
model.load_state_dict(torch.load("gesture_model.pth"))  # Загрузите предобученную модель
model.eval()

# Трансформация для подготовки кадра
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Словарь для маппинга выходов модели на действия
gesture_to_action = {
    0: "увеличить громкость",
    1: "уменьшить громкость",
    2: "выключить звук",
    3: "включить звук"
}

def perform_action(action):
    """Выполняет действие на основе распознанного жеста."""
    if action == "увеличить громкость":
        volume.SetMasterVolumeLevelScalar(min(volume.GetMasterVolumeLevelScalar() + 0.1, 1.0), None)
    elif action == "уменьшить громкость":
        volume.SetMasterVolumeLevelScalar(max(volume.GetMasterVolumeLevelScalar() - 0.1, 0.0), None)
    elif action == "выключить звук":
        volume.SetMute(1, None)
    elif action == "включить звук":
        volume.SetMute(0, None)

def detect_hand(frame):
    """Обнаружение кисти на кадре."""
    # Преобразуем в цветовое пространство HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Диапазон оттенков для кожи (можно подстроить под ваше освещение)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Маска для выделения кожи
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Находим контуры в маске
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Если контуры найдены
    if contours:
        # Выбираем самый большой контур
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Если площадь контура достаточно велика, считаем его кистью
        if cv2.contourArea(largest_contour) > 1000:  # Пороговая площадь
            x, y, w, h = cv2.boundingRect(largest_contour)
            return (x, y, w, h), mask

    return None, mask

# Подключение к камере
cap = cv2.VideoCapture(0)  # Убедитесь, что камера подключена

# Таймер для распознавания раз в секунду
last_recognition_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр с камеры. Проверьте подключение.")
        break

    # Отображаем исходное видео
    cv2.imshow('Video Stream', frame)

    # Обнаружение кисти
    hand_rect, mask = detect_hand(frame)
    if hand_rect:
        # Отрисовка прямоугольника вокруг кисти
        x, y, w, h = hand_rect
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Красный прямоугольник
        cv2.imshow('Mask', mask)

        # Проверяем, прошло ли 1 секунда с последнего распознавания
        if time.time() - last_recognition_time >= 1.0:
            last_recognition_time = time.time()  # Обновляем таймер

            # Подготовка области для модели
            hand_frame = frame[y:y+h, x:x+w]
            hand_frame = cv2.resize(hand_frame, (224, 224))  # Преобразуем в размер для модели
            img_tensor = transform(hand_frame).unsqueeze(0)  # Применяем трансформацию

            # Прогоняем через модель
            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)  # Находим класс с наибольшей вероятностью
                action = gesture_to_action[predicted.item()]
                print(f"Распознанное действие: {action}")

                # Выполняем действие
                perform_action(action)
    else:
        # Если кисть не найдена, ничего не делаем
        print("Кисть не найдена")

    # Отображаем видео с прямоугольником
    cv2.imshow('Video Stream', frame)

    # Нажмите 'q', чтобы выйти
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закрываем видеопоток
cap.release()
cv2.destroyAllWindows()
