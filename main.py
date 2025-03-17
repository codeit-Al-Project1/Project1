import argparse
import torch
from src import train
from src import predict

"""
학습 실행
python main.py --mode train --epochs 10 --lr 0.001

예측 실행
python main.py --mode predict --image_path data/sample.jpg

"""


def main():
    parser = argparse.ArgumentParser(description="Fast R-CNN Object Detection")
    parser.add_argument("--mode", type=str, choices=["train", "predict"], required=True, help="모드를 선택하세요: train 또는 predict")
    parser.add_argument("--json_path", type=str, default="data/annotations.json", help="어노테이션 JSON 파일 경로")
    parser.add_argument("--img_dir", type=str, default="data/images", help="이미지 폴더 경로")
    parser.add_argument("--model_path", type=str, default="models/fast_rcnn.pth", help="저장된 모델 경로")
    parser.add_argument("--image_path", type=str, help="예측할 이미지 경로 (predict 모드에서 필요)")
    parser.add_argument("--num_classes", type=int, default=74, help="클래스 개수")
    parser.add_argument("--epochs", type=int, default=5, help="학습할 에폭 수")
    parser.add_argument("--lr", type=float, default=0.005, help="학습률")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "train":
        train(args.json_path, args.img_dir, args.num_classes, args.epochs, args.lr, device)
    elif args.mode == "predict":
        if not args.image_path:
            print("predict 모드에서는 --image_path를 지정해야 합니다.")
            return
        predict(args.image_path, args.model_path, args.num_classes, device)

if __name__ == "__main__":
    main()
