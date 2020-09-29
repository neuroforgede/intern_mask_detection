from models import faster_rcnn
from utils import io_utils, train_utils, bbox_utils


def init_faster_rcnn(backbone):
    # prepair and load faster-RCNN
    frcnn_model_path = io_utils.get_model_path("faster_rcnn", backbone)
    if backbone == "mobilenet_v2":
        from models.rpn_mobilenet_v2 import get_model as get_rpn_model
    else:
        from models.rpn_vgg16 import get_model as get_rpn_model

    hyper_params = train_utils.get_hyper_params("mobilenet_v2")
    hyper_params["total_labels"] = 2

    anchors = bbox_utils.generate_anchors(hyper_params)
    rpn_model, feature_extractor = get_rpn_model(hyper_params)
    frcnn_model = faster_rcnn.get_model(feature_extractor, rpn_model, anchors, hyper_params, mode="inference")
    frcnn_model.load_weights(frcnn_model_path)
    return frcnn_model
