import sys
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageChops


def prepare_image(image_path):
    """
    Подготовка изображения для нейронной сети.
    """

    # Открываем изображение
    image = Image.open(image_path).convert("L")

    # Определяем цвет фона
    background = Image.new("L", image.size, 255)
    difference = ImageChops.difference(image, background)

    # Находим область с изображенной цифрой
    bbox = difference.getbbox()

    if bbox is None:
        raise ValueError("На изображении не найдена цифра.")

    image = image.crop(bbox)

    # Если фон светлый, инвертируем изображение
    if np.mean(np.array(image)) > 127:
        image = ImageOps.invert(image)

    # Масштабируем цифру, сохраняя пропорции
    image.thumbnail((20, 20))

    # Создаем холст 28x28
    canvas = Image.new("L", (28, 28), 0)

    # Размещаем цифру по центру
    x = (28 - image.width) // 2
    y = (28 - image.height) // 2

    canvas.paste(image, (x, y))

    # Нормализуем изображение
    image_array = np.array(canvas).astype("float32") / 255.0

    # Добавляем размерности для нейронной сети
    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array


def recognize_digit(image_path):
    """
    Распознавание цифры искусственным интеллектом.
    """

    print("Загрузка искусственного интеллекта...")

    # Загружаем обученную модель
    model = tf.keras.models.load_model("digit_model.keras")

    print("Обработка изображения...")

    # Подготавливаем изображение
    image = prepare_image(image_path)

    print("Распознавание цифры...")

    # Получаем вероятности для цифр 0-9
    predictions = model.predict(image, verbose=0)[0]

    # Находим наиболее вероятную цифру
    digit = int(np.argmax(predictions))

    confidence = float(predictions[digit]) * 100

    print("\n========== РЕЗУЛЬТАТ ==========")
    print(f"Распознанная цифра: {digit}")
    print(f"Уверенность ИИ: {confidence:.2f}%")

    print("\nВероятности распознавания:")

    for number, probability in enumerate(predictions):
        print(
            f"Цифра {number}: "
            f"{probability * 100:.2f}%"
        )

    print("===============================")

    return digit


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Использование:")
        print("python predict.py image.png")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        recognize_digit(image_path)

    except FileNotFoundError:
        print("Ошибка: файл изображения не найден.")

    except Exception as error:
        print(f"Ошибка распознавания: {error}")
