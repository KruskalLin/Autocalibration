import numpy as np
from PIL import Image
from path import Path
from tqdm import tqdm


def census_transform(img, csize=(3, 3)):
    img = img[:, :, 0]
    height = img.shape[0]
    width = img.shape[1]
    y_offset = int(csize[0] / 2)
    x_offset = int(csize[1] / 2)
    census_values = np.zeros(shape=(height - 2 * y_offset, width - 2 * x_offset), dtype=np.uint64)
    center_pixels = img[y_offset:height - y_offset, x_offset:width - x_offset]
    offsets = [(u, v) for v in range(y_offset) for u in range(x_offset) if not u == (x_offset + y_offset) / 2 == v]
    for u, v in offsets:
        census_values = (census_values << 1) | (img[v:v + height - 2 * y_offset, u:u + width - 2 * x_offset] > center_pixels)
    return np.pad(census_values, ((y_offset, y_offset), (x_offset, x_offset)), 'constant')

def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)
    return np.array((data - min) / (max - min), dtype=np.float64)


root = '.'

left_dir = Path(root + '/image_02')

for f in tqdm(left_dir.files()):
    left_image = np.array(Image.open(f))
    cl = minmaxscaler(census_transform(left_image, (3, 3)))
    cl_img_2 = np.array(cl * 255, dtype=np.uint8)
    cl = minmaxscaler(census_transform(left_image, (5, 5)))
    cl_img_1 = np.array(cl * 255, dtype=np.uint8)
    cl = minmaxscaler(census_transform(left_image, (7, 7)))
    cl_img_0 = np.array(cl * 255, dtype=np.uint8)

    Image.fromarray(cl_img_2).save(root + '/transform_02/' + f.stem + '_2.png')
    Image.fromarray(cl_img_1).save(root + '/transform_02/' + f.stem + '_1.png')
    Image.fromarray(cl_img_0).save(root + '/transform_02/' + f.stem + '_0.png')



left_dir = Path(root + '/image_03')

for f in tqdm(left_dir.files()):
    left_image = np.array(Image.open(f))
    cl = minmaxscaler(census_transform(left_image, (3, 3)))
    cl_img_2 = np.array(cl * 255, dtype=np.uint8)
    cl = minmaxscaler(census_transform(left_image, (5, 5)))
    cl_img_1 = np.array(cl * 255, dtype=np.uint8)
    cl = minmaxscaler(census_transform(left_image, (7, 7)))
    cl_img_0 = np.array(cl * 255, dtype=np.uint8)

    Image.fromarray(cl_img_2).save(root + '/transform_03/' + f.stem + '_2.png')
    Image.fromarray(cl_img_1).save(root + '/transform_03/' + f.stem + '_1.png')
    Image.fromarray(cl_img_0).save(root + '/transform_03/' + f.stem + '_0.png')