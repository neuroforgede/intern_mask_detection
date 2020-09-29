import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from utils import io_utils, data_utils, train_utils, bbox_utils
from models import faster_rcnn

train_params = train_utils.get_hyper_params_for_faster_rcnn_training()
batch_size = train_params["batch_size"] 
epochs = train_params["epochs"]
with_voc_2012 = train_params["with_voc_2012"]
backbone = train_params["backbone"]
label_name = train_params["label_name"]

hyper_params = train_utils.get_hyper_params(backbone)


def main():

    if backbone == "mobilenet_v2":
        from models.rpn_mobilenet_v2 import get_model as get_rpn_model
    else:
        from models.rpn_vgg16 import get_model as get_rpn_model

    train_data, train_info = data_utils.get_dataset("voc/2007", "train+validation")
    val_data, _ = data_utils.get_dataset("voc/2007", "test")

    label_id = data_utils.get_label_id_for_label(label_name, train_info)

    if with_voc_2012:
        voc_2012_data, _ = data_utils.get_dataset("voc/2012", "train+validation")
        train_data = train_data.concatenate(voc_2012_data)

    train_data = train_data.filter(lambda data_set: data_utils.filter_by_label_id(data_set, label_id)) 
    train_total_items = data_utils.get_total_item_size(train_data)

    val_data = val_data.filter(lambda data_set: data_utils.filter_by_label_id(data_set, label_id))
    val_total_items = data_utils.get_total_item_size(val_data)

    labels = [label_name]
    # We add 1 class for background
    hyper_params["total_labels"] = len(labels) + 1

    img_size = hyper_params["img_size"]
    train_data = train_data.map(lambda x: data_utils.preprocessing(x, img_size, img_size, label_id, apply_augmentation=True))
    val_data = val_data.map(lambda x: data_utils.preprocessing(x, img_size, img_size, label_id))

    data_shapes = data_utils.get_data_shapes()
    padding_values = data_utils.get_padding_values()
    train_data = train_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
    val_data = val_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)

    anchors = bbox_utils.generate_anchors(hyper_params)
    frcnn_train_feed = train_utils.faster_rcnn_generator(train_data, anchors, hyper_params)
    frcnn_val_feed = train_utils.faster_rcnn_generator(val_data, anchors, hyper_params)

    rpn_model, feature_extractor = get_rpn_model(hyper_params)
    frcnn_model = faster_rcnn.get_model(feature_extractor, rpn_model, anchors, hyper_params)
    frcnn_model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-5),
                        loss=[None] * len(frcnn_model.output))

    faster_rcnn.init_model(frcnn_model, hyper_params)

    frcnn_model_path = io_utils.get_model_path("faster_rcnn", backbone)

    log_path = io_utils.get_log_path("faster_rcnn", backbone)

    checkpoint_callback = ModelCheckpoint(frcnn_model_path, monitor="val_loss", save_best_only=True, save_weights_only=True)
    tensorboard_callback = TensorBoard(log_dir=log_path)

    step_size_train = train_utils.get_step_size(train_total_items, batch_size)
    step_size_val = train_utils.get_step_size(val_total_items, batch_size)
    frcnn_model.fit(frcnn_train_feed,
                    steps_per_epoch=step_size_train,
                    validation_data=frcnn_val_feed,
                    validation_steps=step_size_val,
                    epochs=epochs,
                    callbacks=[checkpoint_callback, tensorboard_callback])

    print('executed program successfully')


if __name__ == "__main__":
    main()
