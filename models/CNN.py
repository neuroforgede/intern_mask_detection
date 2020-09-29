from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from utils import train_utils


def get_model():
    hyper_params = train_utils.get_hyper_params_for_cnn()

    model = Sequential()

    input_shape = hyper_params["target_size"]
    input_shape = (input_shape[0], input_shape[1], 3)

    model.add(layers.InputLayer(input_shape=input_shape))

    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(2, 2))

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(2, 2))

    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(2, 2))

    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(2, 2))

    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(2, 2))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(512))
    model.add(layers.Dense(512))
    model.add(layers.Dense(32))

    model.add(layers.Dense(units=1, activation='sigmoid'))

    return model
