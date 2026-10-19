import cv2

# Используем индекс 0, если телефон выступает как виртуальная веб-камера
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Не удалось подключиться к камере. Проверьте соединение.")
    exit()

print("Нажмите 'q', чтобы выйти.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка получения кадра.")
        break

    # Отображение видео
    cv2.imshow("Video from phone", frame)

    # Закрытие окна при нажатии 'q'
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('й'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
