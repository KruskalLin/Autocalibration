import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class KittiLoader(Dataset):
    def __init__(self, root_dir, mode, transform=None, n=3):
        left_dir = os.path.join(root_dir, 'image_02/')
        self.left_paths = sorted([os.path.join(left_dir, fname) for fname\
                           in os.listdir(left_dir)])
        if mode == 'train':
            right_dir = os.path.join(root_dir, 'image_03/')
            self.right_paths = sorted([os.path.join(right_dir, fname) for fname\
                                in os.listdir(right_dir)])
            assert len(self.right_paths) == len(self.left_paths)
            transform_left_dir = os.path.join(root_dir, 'transform_02/')
            self.transform_left_paths = sorted([os.path.join(transform_left_dir, fname) for fname \
                                       in os.listdir(transform_left_dir)])
            assert len(self.transform_left_paths) == n * len(self.left_paths)
            transform_right_dir = os.path.join(root_dir, 'transform_03/')
            self.transform_right_paths = sorted([os.path.join(transform_right_dir, fname) for fname \
                                              in os.listdir(transform_right_dir)])
            assert len(self.transform_right_paths) == n * len(self.right_paths)
        else:
            left_gt_dir = os.path.join(root_dir, 'image_gt_02/')
            self.left_gt_paths = sorted([os.path.join(left_gt_dir, fname) for fname \
                                       in os.listdir(left_gt_dir)])
            assert len(self.left_gt_paths) == len(self.left_paths)
        self.transform = transform
        self.mode = mode
        self.n = n


    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_image = Image.open(self.left_paths[idx])
        if self.mode == 'train':
            right_image = Image.open(self.right_paths[idx])
            transform_left_images = [Image.open(self.transform_left_paths[self.n * idx + i]) for i in range(self.n)]
            transform_right_images = [Image.open(self.transform_right_paths[self.n * idx + i]) for i in range(self.n)]
            sample = {'left_image': left_image, 'right_image': right_image,
                      'transform_left_images': transform_left_images,
                      'transform_right_images': transform_right_images}

            if self.transform:
                sample = self.transform(sample)
                return sample
            else:
                return sample
        else:
            left_gt_image = Image.open(self.left_gt_paths[idx])
            sample = {'left_image': left_image, 'left_gt_image': left_gt_image}
            if self.transform:
                sample = self.transform(sample)
            return sample
