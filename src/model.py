import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_fast_rcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # 기존 분류 헤드 수정
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
