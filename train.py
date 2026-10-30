import tensorflow as tf
from model import create_model


def train_ai():
    print("Загрузка набора данных MNIST...")

    # Загрузка изображений для обучения и тестирования
    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    # Нормализация изображений от 0 до 1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Добавление канала цвета
    x_train = x_train[..., None]
    x_test = x_test[..., None]

    print("Создание нейронной сети...")

    model = create_model()

    # Вывод архитектуры сети
    model.summary()

    print("Начало обучения ИИ...")

    # Обучение модели
    model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    print("Проверка точности модели...")

    # Проверка на тестовых изображениях
    loss, accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print(f"Точность ИИ: {accuracy * 100:.2f}%")
    print(f"Ошибка модели: {loss:.4f}")

    # Сохранение обученной модели
    model.save("digit_model.keras")

    print("Модель успешно сохранена!")
    print("Файл: digit_model.keras")


if __name__ == "__main__":
    train_ai()
