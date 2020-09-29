import os
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from models import CNN
from utils import io_utils, train_utils

# some hyperparameters:
hyper_params = train_utils.get_hyper_params_for_cnn()
target_size = hyper_params["target_size"]
optimizer = hyper_params["optimizer"]
batch_size = hyper_params["batch_size"]
epochs = hyper_params["epochs"]
loss = hyper_params["loss"]

cnn_model_path = io_utils.get_model_path("cnn")
log_path = io_utils.get_log_path("cnn")
tb_callback = TensorBoard(log_dir=log_path)
checkpoint_callback = ModelCheckpoint(cnn_model_path, monitor="val_loss", save_best_only=True, save_weights_only=True)

path_training_data = os.path.join(os.getcwd(), 'data/training')
path_validation_data = os.path.join(os.getcwd(), 'data/validation')


def main():
    raw = ImageDataGenerator(rescale=1 / 255)
    augmented = ImageDataGenerator(rescale=1 / 255,
                                   rotation_range=5,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   horizontal_flip=True,
                                   brightness_range=(0.7, 1.3))

    train_dataset_augmented = augmented.flow_from_directory(path_training_data,
                                                               class_mode='binary',
                                                               target_size=target_size,
                                                               batch_size=batch_size)

    val_dataset = raw.flow_from_directory(path_validation_data,
                                          class_mode='binary',
                                          target_size=target_size,
                                          batch_size=batch_size)

    model = CNN.get_model()
    model.summary()

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    model.fit(train_dataset_augmented, epochs=epochs, validation_data=val_dataset,
              callbacks=[checkpoint_callback, tb_callback])

    print('executed program successfully')


if __name__ == "__main__":
    main()
