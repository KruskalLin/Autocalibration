from path import Path
from PIL import Image
from imageio import imread
import random
import numpy as np
from tqdm import tqdm

def prepare(root):
    data_dirs = Path(root).dirs()
    image_paths = []
    data_dirs = sorted(data_dirs, key=lambda x: x.stem)
    for dir in data_dirs:
        if dir.stem == 'whole_data':
            continue
        image_dir = dir / 'image_02' / 'data'
        image_paths.extend(image_dir.files())

    new_dir = Path(root) / 'whole_data'
    new_dir.makedirs_p()
    for i, p in tqdm(enumerate(image_paths)):
        img = Image.fromarray(imread(p))
        img.save(new_dir / ('{:010d}'.format(i) + '.png'))
        img_name = p.stem
        velo = np.fromfile(p.parent.parent.parent / 'velodyne_points' / 'data' / (img_name + '.bin'), dtype=np.float32)
        velo.tofile(new_dir / ('{:010d}'.format(i) + '.bin'))


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
    root = '/data/data/2011_09_26'
    output = './selected_data'
    prepare(root)
    select_data(root, output)
