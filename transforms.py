import torch
import torchvision.transforms as transforms
import numpy as np


def image_transforms(mode='train', size=(256, 512)):
    if mode == 'train':
        data_transform = transforms.Compose([
            ResizeImage(train=True, size=size),
            ToTensor(train=True),
        ])
        return data_transform
    elif mode == 'val' or mode == 'test':
        data_transform = transforms.Compose([
            ResizeImage(train=False, size=size),
            ToTensor(train=False),
        ])
        return data_transform
    else:
        print('Wrong mode')


class ResizeImage(object):
    def __init__(self, train=True, size=(256, 512)):
        self.train = train
        self.transform = transforms.Resize(size)

    def __call__(self, sample):
        if self.train:
            left_image = sample['left_image']
            right_image = sample['right_image']
            transform_left_images = sample['transform_left_images']
            transform_right_images = sample['transform_right_images']

            new_right_image = self.transform(right_image)
            new_left_image = self.transform(left_image)
            new_transform_left_images = [self.transform(img) for img in transform_left_images]
            new_transform_right_images = [self.transform(img) for img in transform_right_images]
            sample = {'left_image': new_left_image, 'right_image': new_right_image,
                      'transform_left_images': new_transform_left_images,
                      'transform_right_images': new_transform_right_images}
        else:
            left_image = sample['left_image']
            left_gt_image = sample['left_gt_image']
            new_left_image = self.transform(left_image)
            new_left_gt_image = self.transform(left_gt_image)
            sample = {'left_image': new_left_image, 'left_gt_image': new_left_gt_image}
        return sample


class ToTensor(object):
    def __init__(self, train):
        self.train = train
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        if self.train:
            left_image = sample['left_image']
            right_image = sample['right_image']
            transform_left_images = sample['transform_left_images']
            transform_right_images = sample['transform_right_images']

            new_right_image = self.transform(right_image)
            new_left_image = self.transform(left_image)
            new_transform_left_images = [self.transform(img) for img in transform_left_images]
            new_transform_right_images = [self.transform(img) for img in transform_right_images]
            sample = {'left_image': new_left_image, 'right_image': new_right_image,
                      'transform_left_images': new_transform_left_images,
                      'transform_right_images': new_transform_right_images}
        else:
            left_image = sample['left_image']
            left_gt_image = sample['left_gt_image']
            new_left_image = self.transform(left_image)
            new_left_gt_image = self.transform(left_gt_image)
            sample = {'left_image': new_left_image, 'left_gt_image': new_left_gt_image}
        return sample