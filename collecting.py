import cv2
import os

"""Пример жестов:

    "указательный вверх" (увеличить громкость),
    "указательный вниз" (уменьшить громкость),
    "кулак" (выключить звук),
    "ладонь" (включить звук)."""

# Создаём папку для данных
gesture_name = "volume_down"  # Название текущего жеста
output_dir = f"data/{gesture_name}"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Не удалось подключиться к камере.")
    exit()

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Отображаем видеопоток
    cv2.imshow("Collecting Data", frame)

    # Сохраняем кадры по нажатию клавиши "s"
    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        filename = os.path.join(output_dir, f"{count}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Сохранено: {filename}")
        count += 1
    elif key & 0xFF == ord('q'):  # Выход по нажатию "q"
        break

cap.release()
cv2.destroyAllWindows()
