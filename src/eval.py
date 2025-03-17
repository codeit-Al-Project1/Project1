import torch
import cv2
import torchvision.transforms as T
from model import get_fast_rcnn_model

def predict(image_path, model_path="models/fast_rcnn.pth", num_classes=74, device="cuda"):
    model = get_fast_rcnn_model(num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    image = cv2.imread(image_path)
    transform = T.ToTensor()
    image_tensor = transform(image).to(device)

    with torch.no_grad():
        predictions = model([image_tensor])

    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"ID: {label}, {score:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite("outputs/detection_result.jpg", image)

if __name__ == "__main__":
    predict("data/sample.jpg")