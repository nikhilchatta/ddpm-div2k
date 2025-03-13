import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from glob import glob

class DIV2KDataset(Dataset):
    def __init__(self, root, image_size=128):
        self.image_paths = glob(os.path.join(root, "*.png"))
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))

        # ✅ Manual ToTensor and Normalize (bypass torchvision internal error)
        image = (torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
                 .float().view(image.size[1], image.size[0], 3).permute(2, 0, 1)) / 255.0
        image = image * 2 - 1  # Normalize to [-1, 1]

        return image, 0

def get_div2k_dataloader(root, image_size=128, batch_size=16, shuffle=True):
    dataset = DIV2KDataset(root, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
