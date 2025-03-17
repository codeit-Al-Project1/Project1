import torch
import torchvision.transforms as T
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torchvision.models.detection.transform import GeneralizedRCNNTransform

class FastRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.dataset = dset.VOCDetection(root, year="2007", image_set="train", download=True)
        self.transform = transform if transform else T.ToTensor()

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        image = self.transform(image)

        boxes = torch.tensor([obj["bndbox"] for obj in target["annotation"]["object"]], dtype=torch.float32)
        labels = torch.tensor([1] * len(boxes), dtype=torch.int64)  # 예제에서는 모든 객체를 같은 클래스로 설정

        return image, {"boxes": boxes, "labels": labels}

    def __len__(self):
        return len(self.dataset)

def get_dataloader(root="data/", batch_size=4):
    dataset = FastRCNNDataset(root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
