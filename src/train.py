import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train(num_epochs=50, model=None, learning_rate=0.001, train_loader=None): 

    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 학습
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for train_images, cleaned_images in train_loader:  # paired_loader에서 가져오기
            # 데이터를 GPU로 전송
            train_images = train_images.to(device)
            cleaned_images = cleaned_images.to(device)

            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # 모델 예측 및 손실 계산
            outputs = model(train_images)
            loss = loss_fn(outputs, cleaned_images)

            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 손실 누적
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")