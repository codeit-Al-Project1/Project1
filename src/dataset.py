import os
import json
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image

class FastRCNNDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        self.img_dir = img_dir
        self.transform = transform if transform else T.ToTensor()
        self.image_id_to_annotations = self._map_annotations()

    def _map_annotations(self):
        """이미지 ID와 어노테이션을 매핑하여 검색 속도를 높임"""
        mapping = {}
        for ann in self.data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in mapping:
                mapping[img_id] = []
            mapping[img_id].append(ann)
        return mapping

    def __getitem__(self, idx):
        img_info = self.data["images"][idx]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        image_id = img_info["id"]
        annotations = self.image_id_to_annotations.get(image_id, [])

        # BBox 및 Label 정보 추출
        boxes, labels = [], []
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])  # (x1, y1, x2, y2) 형식
            labels.append(ann["category_id"])

        # Tensor 변환
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        return image, {"boxes": boxes, "labels": labels}

    def __len__(self):
        return len(self.data["images"])

def get_dataloader(json_path, img_dir, batch_size=4):
    dataset = FastRCNNDataset(json_path, img_dir)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
