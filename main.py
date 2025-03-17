import argparse
from train import train
from eval import predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--image", type=str, help="Path to image for evaluation")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "eval" and args.image:
        predict(args.image)