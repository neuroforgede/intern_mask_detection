import os
import math
import tensorflow as tf
from init_faster_rcnn import init_faster_rcnn
from utils import data_utils, train_utils, bbox_utils

# hyperparameters
from utils.bbox_utils import crop_person_from_picture

hyper_params = train_utils.get_hyper_params_for_prediction()
threshold = hyper_params["threshold"]  # filter the prediction of the R-CNN by this threshold
backbone = hyper_params["backbone"]
batch_size = hyper_params["batch_size"]

# custom input and output
input_path = os.path.join(os.getcwd(), 'data/input_for_cropping')
output_dir = os.path.join(os.getcwd(), 'data/output_from_cropping')

data_utils.delete_files_in(output_dir)
img_paths = data_utils.get_custom_imgs(input_path)

data_types = data_utils.get_data_types()
data_shapes = data_utils.get_data_shapes()
padding_values = data_utils.get_padding_values()
frcnn_model = init_faster_rcnn(backbone)


def crop_all_persons_from_one_picture(img, image_index, img_size, pred_bboxes, pred_scores, step):
    denormalized_bboxes = bbox_utils.denormalize_bboxes(pred_bboxes[image_index], img_size, img_size)
    image = tf.keras.preprocessing.image.array_to_img(img)

    # loop over persons for one image
    for person_index, bbox in enumerate(denormalized_bboxes):
        im_crop = crop_person_from_picture(pred_scores, image_index, person_index, threshold, bbox, image)

        if im_crop is None:
            continue

        image_path = os.path.join(output_dir, 'picture:_' + str(step * batch_size + image_index) + '-Person:_' + str(
            person_index) + '.jpg')
        im_crop.save(image_path, 'JPEG')
        print(image_path)


def main():
    hyper_params_faster_rcnn = train_utils.get_hyper_params("mobilenet_v2")
    img_size = hyper_params_faster_rcnn["img_size"]
    # loop over img_paths with batch_size
    steps = math.ceil(len(img_paths) / batch_size)

    for step in range(steps):
        # store the part of img_paths we want to use in this batch in img_path_for_one_batch
        start_global = step * batch_size
        end_global = start_global + batch_size
        img_path_for_one_batch = img_paths[start_global:end_global]

        # prepair data
        dataset = tf.data.Dataset.from_generator(lambda: data_utils.custom_data_generator(
            img_path_for_one_batch, img_size, img_size), data_types, data_shapes)
        dataset = dataset.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)

        # predict bboxes with faster-rcnn
        pred_bboxes, _, pred_scores = frcnn_model.predict(dataset, steps=1, verbose=1)

        for image_data in dataset:

            imgs, _, _ = image_data
            img_size = imgs.shape[1]

            # loop over images
            for i, img in enumerate(imgs):
                crop_all_persons_from_one_picture(img, i, img_size, pred_bboxes, pred_scores, step)


    print('executed program successfully')


if __name__ == "__main__":
    main()
