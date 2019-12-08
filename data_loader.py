from PIL import Image
from torch.utils.data import Dataset


class KittiLoader(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.images = sorted(root_dir.files('*.png'))
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        left_image = Image.open(self.images[idx])
        if self.transform:
            left_image = self.transform(left_image)
        return left_image
