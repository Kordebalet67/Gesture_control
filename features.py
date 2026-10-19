from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Получение интерфейса управления звуком
def get_volume_interface():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None
    )
    return cast(interface, POINTER(IAudioEndpointVolume))

# 1. Увеличить громкость
def increase_volume(step=0.1):
    volume = get_volume_interface()
    current_volume = volume.GetMasterVolumeLevelScalar()
    new_volume = min(current_volume + step, 1.0)  # Ограничение на 100%
    volume.SetMasterVolumeLevelScalar(new_volume, None)
    print(f"Громкость увеличена до {int(new_volume * 100)}%")

# 2. Уменьшить громкость
def decrease_volume(step=0.1):
    volume = get_volume_interface()
    current_volume = volume.GetMasterVolumeLevelScalar()
    new_volume = max(current_volume - step, 0.0)  # Ограничение на 0%
    volume.SetMasterVolumeLevelScalar(new_volume, None)
    print(f"Громкость уменьшена до {int(new_volume * 100)}%")

# 3. Выключить звук (Mute)
def mute_volume():
    volume = get_volume_interface()
    volume.SetMute(1, None)
    print("Звук выключен")

# 4. Включить звук (Unmute)
def unmute_volume():
    volume = get_volume_interface()
    volume.SetMute(0, None)
    print("Звук включен")


if __name__ == "__main__":
    pass
    # increase_volume(0.2)  # Увеличить громкость на 20%
    # decrease_volume(0.1)  # Уменьшить громкость на 10%
    # mute_volume()         # Выключить звук
    # unmute_volume()       # Включить звук
