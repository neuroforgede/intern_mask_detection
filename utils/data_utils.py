import os
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np
import shutil


def get_label_id_for_label(label_name, info):
    for label_id, label in enumerate(info.features["labels"].names):
        if label_name == label:
            return label_id
    print('no label with name {} found'.format(label_name))
    return None


def delete_files_in(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    print('deleted everything in {}'.format(path))
    

def filter_by_label_id(data_set, label_id): 
    """Filter for dataset. Passes just entries that involve persons
    inputs: 
            train_data 
    outputs:
            tensorflow bool vector
    """
    labels = data_set['labels']
    right_label_or_not = (labels == label_id)
    right_label_or_not = tf.dtypes.cast(right_label_or_not, tf.int32)
    right_label_or_not = tf.reduce_sum(right_label_or_not)
    right_label_or_not = tf.dtypes.cast(right_label_or_not, tf.bool)
    return right_label_or_not


def preprocessing(image_data, final_height, final_width, label_id, apply_augmentation=False, evaluate=False):
    """Image resizing operation handled before batch operations
       and discard information on all labels except label with label_id
    inputs:
        image_data = tensorflow dataset image_data
        final_height = final image height after resizing
        final_width = final image width after resizing
    outputs:
        img = (final_height, final_width, channels)
        gt_boxes = (gt_box_size, [y1, x1, y2, x2])
        gt_labels = (gt_box_size)
    """
    img = image_data["image"]
    gt_boxes = image_data["objects"]["bbox"]
    gt_labels = tf.cast(image_data["objects"]["label"] + 1, tf.int32) # add 1 for background

    # delete gt_boxe and gt_label entrys that do not belong to label_id
    person_or_not = gt_labels == (label_id + 1) # + 1 since the lable background is added
    gt_boxes = gt_boxes[person_or_not]
    gt_labels = gt_labels[person_or_not]
    gt_labels = gt_labels - label_id # since just one lable is used it is identified with 1

    if evaluate:
        not_diff = tf.logical_not(image_data["objects"]["is_difficult"])
        gt_boxes = gt_boxes[not_diff]
        gt_labels = gt_labels[not_diff]
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (final_height, final_width))
    if apply_augmentation:
        img, gt_boxes = randomly_apply_operation(flip_horizontally, img, gt_boxes)
    return img, gt_boxes, gt_labels


def get_random_bool():
    """Generating random boolean.
    outputs:
        random boolean 0d tensor
    """
    return tf.greater(tf.random.uniform((), dtype=tf.float32), 0.5)


def randomly_apply_operation(operation, img, gt_boxes):
    """Randomly applying given method to image and ground truth boxes.
    inputs:
        operation = callable method
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    outputs:
        modified_or_not_img = (final_height, final_width, depth)
        modified_or_not_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    """
    return tf.cond(
        get_random_bool(),
        lambda: operation(img, gt_boxes),
        lambda: (img, gt_boxes)
    )


def flip_horizontally(img, gt_boxes):
    """Flip image horizontally and adjust the ground truth boxes.
    inputs:
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    outputs:
        modified_img = (height, width, depth)
        modified_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    """
    flipped_img = tf.image.flip_left_right(img)
    flipped_gt_boxes = tf.stack([gt_boxes[..., 0],
                                1.0 - gt_boxes[..., 3],
                                gt_boxes[..., 2],
                                1.0 - gt_boxes[..., 1]], -1)
    return flipped_img, flipped_gt_boxes


def get_dataset(name, split, data_dir="~/tensorflow_datasets"):
    """Get tensorflow dataset split and info.
    inputs:
        name = name of the dataset, voc/2007, voc/2012, etc.
        split = data split string, should be one of ["train", "validation", "test"]
        data_dir = read/write path for tensorflow datasets
    outputs:
        dataset = tensorflow dataset split
        info = tensorflow dataset info
    """
    assert split in ["train", "train+validation", "validation", "test"]

    dataset, info = tfds.load(name, split=split, data_dir=data_dir, with_info=True)
    return dataset, info


def get_total_item_size(dataset):
    """Get total item size for given split.
    inputs:
        dataset
    outputs:
        total_item_size = number of total items
    """
    total_items = 0
    for element in dataset:
        total_items += 1
    return total_items


def get_labels(info):
    """Get label names list.
    inputs:
        info = tensorflow dataset info
    outputs:
        labels = [labels list]
    """
    return info.features["labels"].names


def get_custom_imgs(custom_image_path):
    """Generating a list of images for given path.
    inputs:
        custom_image_path = folder of the custom images
    outputs:
        custom image list = [path1, path2]
    """
    img_paths = []
    for path, dir, filenames in os.walk(custom_image_path):
        for filename in filenames:
            img_paths.append(os.path.join(path, filename))
        break
    return img_paths


def custom_data_generator(img_paths, final_height, final_width):
    """Yielding custom entities as dataset.
    inputs:
        img_paths = custom image paths
        final_height = final image height after resizing
        final_width = final image width after resizing
    outputs:
        img = (final_height, final_width, depth)
        dummy_gt_boxes = (None, None)
        dummy_gt_labels = (None, )
    """
    for img_path in img_paths:
        image = Image.open(img_path)
        resized_image = image.resize((final_width, final_height), Image.ANTIALIAS)  # Image.LANCZOS
        img = np.array(resized_image)
        img = tf.image.convert_image_dtype(img, tf.float32)
        yield img, tf.constant([[]], dtype=tf.float32), tf.constant([], dtype=tf.int32)


def get_data_types():
    """Generating data types for tensorflow datasets.
    outputs:
        data types = output data types for (images, ground truth boxes, ground truth labels)
    """
    return tf.float32, tf.float32, tf.int32


def get_data_shapes():
    """Generating data shapes for tensorflow datasets.
    outputs:
        data shapes = output data shapes for (images, ground truth boxes, ground truth labels)
    """
    return [None, None, None], [None, None], [None, ]


def get_padding_values():
    """Generating padding values for missing values in batch for tensorflow datasets.
    outputs:
        padding values = padding values with dtypes for (images, ground truth boxes, ground truth labels)
    """
    return tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32)
