import torchvision
import torch.nn as nn

"""
Fast R-CNN 모델 정의
"""

def get_fast_rcnn_model(num_classes=2):
    # Faster R-CNN을 불러와 Fast R-CNN으로 변형
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # RoI 헤드 부분만 수정 (num_classes 설정)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    return model