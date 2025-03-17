import torch
import torch.optim as optim
import torch.nn as nn
from dataset import get_dataloader
from model import get_fast_rcnn_model

"""
**Fast R-CNN에서 SGD를 쓰는 이유**

일반화 성능이 더 좋음 (과적합 방지)
메모리 효율적 (대규모 데이터에 적합)
Momentum을 추가하면 안정적 (빠른 수렴)
즉, Fast R-CNN에서는 빠르게 최적화하는 것보다, 일반화가 잘 되면서도 안정적인 학습이 더 중요하므로 SGD를 선택
"""
def train(json_path, img_dir, num_classes=74, num_epochs=5, lr=0.005, device="cuda"):
    dataloader = get_dataloader(json_path, img_dir)
    model = get_fast_rcnn_model(num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "models/fast_rcnn.pth")

if __name__ == "__main__":
    train(json_path="data/merged_annotations.json", img_dir="data/train_images")
