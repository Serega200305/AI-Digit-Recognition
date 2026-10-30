import tensorflow as tf
from tensorflow.keras import layers, models


def create_model():
    """
    Создание нейронной сети для распознавания
    рукописных цифр от 0 до 9.
    """

    model = models.Sequential([
        # Входное изображение 28x28 пикселей
        layers.Input(shape=(28, 28, 1)),

        # Сверточный слой для поиска признаков
        layers.Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu"
        ),

        # Уменьшение размера изображения
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Второй сверточный слой
        layers.Conv2D(
            64,
            kernel_size=(3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(pool_size=(2, 2)),

        # Преобразование данных в один массив
        layers.Flatten(),

        # Полносвязный слой
        layers.Dense(128, activation="relu"),

        # Защита от переобучения
        layers.Dropout(0.3),

        # 10 выходов: цифры от 0 до 9
        layers.Dense(10, activation="softmax")
    ])

    # Настройка процесса обучения
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
