import torch
import torch.optim as optim
from model import get_fast_rcnn_model
from dataset import get_dataloader

def train(num_epochs=5, lr=0.005, device="cuda"):
    dataloader = get_dataloader()
    model = get_fast_rcnn_model(num_classes=2).to(device)
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

if __name__ == "__main__":
    train()
