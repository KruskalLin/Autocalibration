from path import Path
from skimage.metrics import *
from skimage.feature import *
from color_utils import array2color
from calib_utils import *
from PIL import Image
from imageio import imread
from collections import Counter
from lidar_interpolate_toolbox import *
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe, anneal, rand
import argparse

global root, img_height, img_width, id, visualize, img, velo, P_rect, R_rect, pre_depth, vis


def return_arguments():
    parser = argparse.ArgumentParser(description='Find Extrinsic Parameters')

    parser.add_argument('--root',
                        default='selected_data',
                        help='path to the sampled data.'
                             'It should contain pairs of images and point clouds and the result of depth estimator'
                        )
    parser.add_argument('--id', help='id of the image and point cloud')
    parser.add_argument('--img_height', type=int, help='input height',
                        default=256)
    parser.add_argument('--img_width', type=int, help='input width',
                        default=512)
    parser.add_argument('--visualize', type=bool, default=False, help='visualize data using visdom')
    parser.add_argument('--optim', default='anneal', help='mode: bayesian or anneal (default: anneal)')
    args = parser.parse_args()
    return args


def generate_depth_map(velo, Tr, R_rect, P_rect, depth_size_ratio=1, depth_scale=4, img_height=256, img_width=512,
                       refine=False, choose_closest=False):
    # compute projection matrix velodyne->image plane

    velo2cam = np.vstack((Tr, np.array([0, 0, 0, 1.0])))

    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = R_rect.reshape(3, 3)

    P_rect[0] /= depth_size_ratio
    P_rect[1] /= depth_size_ratio
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, -1:]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1

    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < img_width / depth_size_ratio)
    val_inds = val_inds & (velo_pts_im[:, 1] < img_height / depth_size_ratio)
    velo_pts_im = velo_pts_im[val_inds, :]
    # project to image
    depth = np.zeros((img_height // depth_size_ratio, img_width // depth_size_ratio)).astype(
        np.float32)
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    if choose_closest:
        def sub2ind(matrixSize, rowSub, colSub):
            m, n = matrixSize
            return rowSub * (n - 1) + colSub - 1

        inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
        dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
        for dd in dupe_inds:
            pts = np.where(inds == dd)[0]
            x_loc = int(velo_pts_im[pts[0], 0])
            y_loc = int(velo_pts_im[pts[0], 1])
            depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()

    depth[depth < 0] = 0
    interpolated_lidar = interpolator2d(depth)
    if interpolated_lidar.max() == interpolated_lidar.min():
        return interpolated_lidar
    interpolated_lidar = np.floor((interpolated_lidar - interpolated_lidar.min()) / (interpolated_lidar.max() - interpolated_lidar.min()) * 255) * depth_scale
    interpolated_lidar[interpolated_lidar > 255.] = 255
    if refine:
        depth = np.floor((depth - depth.min()) / (depth.max() - depth.min()) * 255) * depth_scale
        depth[depth > 255.] = 255
        interpolated_lidar = sd_filter(img, depth)
    return interpolated_lidar


def init_matrix(a, b, c, p, q, r, d=None):
    if d == None:
        theta = np.asarray([a, b, c])
        R = euler2rot(theta)
    else:
        q = np.asarray([a, b, c, d])
        R = quat2rot(q)
    Tr = np.column_stack((R, np.asarray([p, q, r])))
    return Tr


def compute_loss(a, b, c, p, q, r, d=None, grad=False):
    depth = generate_depth_map(velo, init_matrix(a, b, c, p, q, r, d), R_rect, P_rect, img_height=img_height,
                               img_width=img_width)
    depth = cv2.GaussianBlur(depth, (13, 13), 0)
    if visualize:
        vis.images(
            array2color(depth, max_value=None,
                        colormap='magma'), win='pst_depth', opts=dict(title='pst_depth'))
        vis.images(
            array2color(pre_depth, max_value=None,
                        colormap='magma'), win='pre_depth', opts=dict(title='pre_depth'))
    loss1 = - np.sum(depth[int(depth.shape[0] / 2.5):, :] < 0.1) / (img_width * img_height)
    loss2 = structural_similarity(depth[int(depth.shape[0] / 2.5):, :], pre_depth[int(depth.shape[0] / 2.5):, :],
                                  win_size=27)
    grad_depth = cv2.Sobel(depth[int(depth.shape[0] / 2.5):, :], cv2.CV_16S, 1, 0)
    grad_pre = cv2.Sobel(pre_depth[int(pre_depth.shape[0] / 2.5):, :], cv2.CV_16S, 1, 0)
    if visualize:
        vis.images(
            array2color(grad_depth, max_value=None,
                        colormap='magma'), win='grad', opts=dict(title='grad'))
        vis.images(
            array2color(grad_pre, max_value=None,
                        colormap='magma'), win='grad_pre', opts=dict(title='grad_pre'))
    loss3 = structural_similarity(grad_pre, grad_depth)

    fd1 = hog(pre_depth[int(depth.shape[0] / 2.5):, :], pixels_per_cell=(7, 7), cells_per_block=(3, 3),
              orientations=9)
    fd2 = hog(depth[int(depth.shape[0] / 2.5):, :], pixels_per_cell=(7, 7), cells_per_block=(3, 3),
              orientations=9)
    loss4 = - np.linalg.norm(fd1 - fd2)

    if visualize:
        print("the loss is " + str(loss1) + ", " + str(loss2) + ", " + str(loss3) + ", " + str(loss4))
    if grad:
        return 200 * loss1 + 50 * loss2 + loss3 + 5 * loss4
    else:
        return 200 * loss1 + 50 * loss2 + 5 * loss4


def hyperopt_train(params):
    loss = compute_loss(**params)
    return {'loss': -loss, 'status': STATUS_OK}


def main():
    global root, img_height, img_width, id, visualize, img, velo, P_rect, R_rect, pre_depth, vis
    args = return_arguments()
    root = args.root
    img_height = args.img_height
    img_width = args.img_width
    id = int(args.id)
    visualize = args.visualize

    root = Path(root)
    img_file = root / '{:010d}'.format(id) + '.png'
    img = Image.fromarray(imread(img_file))
    zoom_y = img_height / img.size[1]
    zoom_x = img_width / img.size[0]
    img = img.resize((img_width, img_height))

    if visualize:
        import visdom
        vis = visdom.Visdom()
        vis.images(np.array(img).transpose((2, 0, 1)), win='img', opts=dict(title='img'))

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    calib_file = root / 'calib_cam_to_cam.txt'
    filedata = read_raw_calib_file(calib_file)
    P_rect = np.reshape(filedata['P_rect_02'], (3, 4))
    R_rect = filedata['R_rect_02'].reshape(3, 3)

    P_rect[0] *= zoom_x
    P_rect[1] *= zoom_y

    velo_name = '{:010d}'.format(id) + '.bin'
    velo = np.fromfile(root / velo_name, dtype=np.float32).reshape(-1, 4)
    velo[:, 3] = 1
    velo = velo[velo[:, 0] >= 0, :]

    pre_depth = np.load(root / 'disparities_pp.npy')[id] + 1
    pre_depth = 1. / pre_depth
    pre_depth = np.floor((pre_depth - pre_depth.min()) / (pre_depth.max() - pre_depth.min()) * 255) + 1
    pre_depth[pre_depth > 255] = 255
    pre_depth = np.floor((pre_depth - pre_depth.min()) / (pre_depth.max() - pre_depth.min()) * 200) + 55
    pre_depth = cv2.GaussianBlur(pre_depth, (13, 13), 0)

    fspace = {'a': hp.uniform('a', -2, 2), 'b': hp.uniform('b', -2, 2), 'c': hp.uniform('c', -2, 2),
              'p': 0, 'q': 0, 'r': 0}

    print("train rot vec...")
    trials = Trials()
    if args.optim == 'bayesian':
        best = fmin(hyperopt_train, fspace, algo=tpe.suggest, max_evals=500, trials=trials)
    elif args.optim == 'anneal':
        best = fmin(hyperopt_train, fspace, algo=anneal.suggest, max_evals=500, trials=trials)
    else:
        best = fmin(hyperopt_train, fspace, algo=rand.suggest, max_evals=500, trials=trials)

    print("Current best rot vec estimation:")
    print(best)
    if visualize:
        vis.line(trials.losses(), win='rot_losses')
        vis.surf(list(zip(trials.vals['a'], trials.vals['b'])), win='rot_vals')

    if (np.abs(best['b']) - 1.57) < 0.05:
        print("Gimbal Lock!")

    trans = {'p': 0, 'q': 0, 'r': 0}
    rot_vec = best
    rot_vec.update(trans)
    compute_loss(**rot_vec)

    print("train trains vec...")
    rot_vec.update({'p': hp.uniform('p', -0.5, 0.), 'q': hp.uniform('q', -0.5, 0.), 'r': hp.uniform('r', -0.5, 0.)})
    trials = Trials()
    if args.optim == 'bayesian':
        trans_vec = fmin(hyperopt_train, rot_vec, algo=tpe.suggest, max_evals=100, trials=trials)
    elif args.optim == 'anneal':
        trans_vec = fmin(hyperopt_train, rot_vec, algo=anneal.suggest, max_evals=100, trials=trials)
    else:
        trans_vec = fmin(hyperopt_train, rot_vec, algo=rand.suggest, max_evals=100, trials=trials)

    print("Current best trans estimation:")
    print(trans_vec)
    if visualize:
        vis.line(trials.losses(), win='trans_losses')
        vis.surf(list(zip(trials.vals['p'], trials.vals['q'])), win='trans_vals')

    best.update(trans_vec)

    print("Current best estimation:")
    print(best)
    compute_loss(**best)

if __name__ == '__main__':
    main()
