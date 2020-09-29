import os
import tensorflow as tf
from datetime import datetime


def get_log_path(model_type, backbone="vgg16", custom_postfix=""):
    """Generating log path from model_type value for tensorboard.
    inputs:
        model_type = "cnn", "faster_rcnn"
        backbone = "vgg16", "mobilenet_v2"
        custom_postfix = any custom string for log folder name
    outputs:
        log_path = tensorboard log path, for example: "logs/rpn_mobilenet_v2/{date}"
    """
    if model_type == 'cnn':
        return "logs/{}{}/{}".format(model_type, custom_postfix, datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        return "logs/{}_{}{}/{}".format(model_type, backbone, custom_postfix, datetime.now().strftime("%Y%m%d-%H%M%S"))


def get_model_path(model_type, backbone="vgg16"):
    """Generating model path from model_type value for save/load model weights.
    inputs:
        model_type = "rpn", "faster_rcnn"
        backbone = "vgg16", "mobilenet_v2"
    outputs:
        model_path = os model path, for example: "trained/rpn_vgg16_model_weights.h5"
    """
    main_path = os.path.join(os.getcwd(),'trained')

    if not os.path.exists(main_path):
        print('make main_path')
        os.makedirs(main_path)
        
    if model_type == 'cnn':
        return os.path.join(main_path, "{}_model_weights.h5".format(model_type))
    else:
        return os.path.join(main_path, "{}_{}_model_weights.h5".format(model_type, backbone))


def is_valid_backbone(backbone):
    """Handling control of given backbone is valid or not.
    inputs:
        backbone = given string from command line
    """
    assert backbone in ["vgg16", "mobilenet_v2"]


def handle_gpu_compatibility():
    """Handling of GPU issues for cuDNN initialize error and memory issues."""
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(e)
