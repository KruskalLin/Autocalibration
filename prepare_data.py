from path import Path
from PIL import Image
from imageio import imread
import random
import numpy as np
from tqdm import tqdm
from shutil import copyfile

dirname = './utils/filenames/'
kitti_trains = open(dirname + 'kitti_train_files.txt').readlines()
kitti_vals = open(dirname + 'kitti_val_files.txt').readlines()
kitti_tests = open(dirname + 'kitti_test_files.txt').readlines()
eigen_trains = open(dirname + 'eigen_train_files.txt').readlines()
eigen_vals = open(dirname + 'eigen_val_files.txt').readlines()
eigen_tests = open(dirname + 'eigen_test_files.txt').readlines()
kitti_depth_trains = open(dirname + 'kitti_depth_train.txt').readlines()
kitti_depth_vals = open(dirname + 'kitti_depth_val.txt').readlines()

def prepare_data(root, output):
    root = Path(root)
    output = Path(output)
    image_paths = [[root / p.strip().split(' ')[0], root / p.strip().split(' ')[1]] for p in kitti_tests]
    for i, p in tqdm(enumerate(image_paths)):
        copyfile(p[0], output / 'image_02' / ('{:010d}'.format(i) + '.png'))
        copyfile(p[1], output / 'image_03' / ('{:010d}'.format(i) + '.png'))
        copyfile(p[0].parent.parent.parent / 'velodyne_points' / 'data' / (p[0].stem + '.bin'), output / 'velodyne_points' / ('{:010d}'.format(i) + '.bin'))

def prepare_depth_data(root, output, gt):
    root = Path(root)
    output = Path(output)
    gt = Path(gt)
    image_left_paths = []
    image_right_paths = []
    for dir in gt.dirs():
        image_left_paths.extend(sorted(((dir.dirs()[0].dirs()[0]) / 'image_02').files()))
        image_right_paths.extend(sorted(((dir.dirs()[0].dirs()[0]) / 'image_03').files()))
    for i, p in tqdm(enumerate(image_left_paths)):
        copyfile(p, output / 'image_gt_02' / ('{:010d}'.format(i) + '.png'))
        image_p = root / p.parent.parent.parent.parent.stem[:10] / p.parent.parent.parent.parent.stem / 'image_02' / 'data' / (p.stem + '.png')
        copyfile(image_p, output / 'image_02' / ('{:010d}'.format(i) + '.png'))

    for i, p in tqdm(enumerate(image_right_paths)):
        copyfile(p, output / 'image_gt_03' / ('{:010d}'.format(i) + '.png'))
        image_p = root / p.parent.parent.parent.parent.stem[
                         :10] / p.parent.parent.parent.parent.stem / 'image_03' / 'data' / (p.stem + '.png')
        copyfile(image_p, output / 'image_03' / ('{:010d}'.format(i) + '.png'))


def select_data(root, output):
    data_dir = Path(root) / 'whole_data'
    output = Path(output)
    images = data_dir.files('*.png')
    sample_images = random.sample(images, 20)
    for img in sample_images:
        img.copy(output / img.name)
        velodyne_name = img.stem + '.bin'
        velodyne_path = data_dir / velodyne_name
        velodyne_path.copy(output / velodyne_name)

    cam2cam = Path(root) / 'calib_cam_to_cam.txt'
    velo2cam = Path(root) / 'calib_velo_to_cam.txt'
    cam2cam.copy(output / 'calib_cam_to_cam.txt')
    velo2cam.copy(output / 'calib_velo_to_cam.txt')


if __name__ == '__main__':
    root = '/data/data/'
    kitti_train_output = '/data/data/kitti_depth_val'
    gt = '/data/data/kitti_depth/val'
    eigen_train_output = '/data/data/eigen_test'
    output = './selected_data'
    # prepare_data(root, kitti_train_output)
    prepare_depth_data(root, kitti_train_output, gt)
    # select_data(root, output)
