import os
import math
import numpy as np
from PIL import Image
import tensorflow as tf

from init_faster_rcnn import init_faster_rcnn
from utils import io_utils, data_utils, train_utils, bbox_utils, drawing_utils
from models import CNN

# some hyperparameter:
from utils.bbox_utils import crop_person_from_picture

hyper_params = train_utils.get_hyper_params_for_prediction()
threshold = hyper_params["threshold"]  # filter the prediction of the R-CNN by this threshold
backbone = hyper_params["backbone"]
batch_size = hyper_params["batch_size"]
size = hyper_params["target_size"]  # input size for the CNN

# custom input and output
input_path = os.path.join(os.getcwd(), 'data/input_prediction')
img_paths = data_utils.get_custom_imgs(input_path)

output_dir = os.path.join(os.getcwd(), 'data/output_prediction')
data_utils.delete_files_in(output_dir)

# load CNN
cnn_model_path = io_utils.get_model_path("cnn")
CNN_model = CNN.get_model()
CNN_model.load_weights(cnn_model_path)
CNN_model.summary()
data_types = data_utils.get_data_types()
data_shapes = data_utils.get_data_shapes()
padding_values = data_utils.get_padding_values()

frcnn_model = init_faster_rcnn(backbone)


# faster-RCNN is now ready to be used


def get_bbox_and_labels_from_one_image(img, image_index, img_size, pred_bboxes, pred_scores, array_of_all_images):
    denormalized_bboxes = bbox_utils.denormalize_bboxes(pred_bboxes[image_index], img_size, img_size)
    image = tf.keras.preprocessing.image.array_to_img(img)
    array_of_all_images.append(image)

    cropped_person_for_single_image = []

    for person_index, bbox in enumerate(denormalized_bboxes):
        im_crop = crop_person_from_picture(pred_scores, image_index, person_index, threshold, bbox, image)

        if im_crop is None:
            continue

        im_crop = im_crop.resize(size, Image.ANTIALIAS)
        im_crop = tf.keras.preprocessing.image.img_to_array(im_crop)
        im_crop = im_crop / 255
        im_crop = np.expand_dims(im_crop, axis=0)
        im_crop = [im_crop]

        # use the loaded CNN to get a prediciton
        prediction = CNN_model.predict(im_crop)[0][0]

        # store the bbox and the prediction
        cropped_person_for_single_image.append([bbox, prediction])

    return cropped_person_for_single_image


def main():
    hyper_params_faster_rcnn = train_utils.get_hyper_params("mobilenet_v2")
    img_size = hyper_params_faster_rcnn["img_size"]
    # loop over img_paths with batch_size
    steps = math.ceil(len(img_paths) / batch_size)

    for step in range(steps):
        # store the part of img_paths we want to use in this global_batch in img_path_for_one_batch
        start_global = step * batch_size
        end_global = start_global + batch_size
        img_path_for_one_batch = img_paths[start_global:end_global]

        # prepair data
        dataset = tf.data.Dataset.from_generator(lambda: data_utils.custom_data_generator(
            img_path_for_one_batch, img_size, img_size), data_types, data_shapes)
        dataset = dataset.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)

        # predict bboxes with faster-rcnn
        pred_bboxes, _, pred_scores = frcnn_model.predict(dataset, steps=1, verbose=1)

        cropped_person_for_all_images_of_one_batch = []
        array_of_all_images = []

        # loop over batches in terms of faster rcnn predicition
        for image_data in dataset:

            imgs, _, _ = image_data
            img_size = imgs.shape[1]

            # loop over single images
            for i, img in enumerate(imgs):
                cropped_person_for_single_image = get_bbox_and_labels_from_one_image(img, i, img_size, pred_bboxes,
                                                                                     pred_scores, array_of_all_images)

                cropped_person_for_all_images_of_one_batch.append(cropped_person_for_single_image)

        # draw and save the results
        drawing_utils.draw_all_pictures_of_one_batch(step, batch_size, output_dir, array_of_all_images,
                                                     cropped_person_for_all_images_of_one_batch)

    print('executed program successfully')


if __name__ == "__main__":
    main()
