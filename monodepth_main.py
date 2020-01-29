import argparse
import time
import torch
import numpy as np
import torch.optim as optim
from tqdm import tqdm
# custom modules
from color_utils import tensor2array
from loss import MonodepthLoss
from model_utils import get_model, to_device, prepare_dataloader
import visdom

vis = visdom.Visdom()


def return_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Monodepth')

    parser.add_argument('--data_dir',
                        help='path to the dataset folder. \
                        It should contain subfolders with following structure:\
                        "image_02" for left images and \
                        "image_03" for right images'
                        )
    parser.add_argument('--val_data_dir',
                        help='path to the validation dataset folder. \
                            It should contain subfolders with following structure:\
                            "image_02" for left images and \
                            "image_03" for right images'
                        )
    parser.add_argument('--model_path', help='path to the trained model')
    parser.add_argument('--output_directory',
                        help='where save dispairities\
                        for tested images'
                        )
    parser.add_argument('--input_height', type=int, help='input height',
                        default=256)
    parser.add_argument('--input_width', type=int, help='input width',
                        default=512)
    parser.add_argument('--model', default='resnet18_md',
                        help='encoder architecture: ' +
                             'resnet18_md or resnet50_md ' + '(default: resnet18)'
                             + 'or torchvision version of any resnet model'
                        )
    parser.add_argument('--pretrained', default=False,
                        help='Use weights of pretrained model'
                        )
    parser.add_argument('--mode', default='train',
                        help='mode: train or test (default: train)')
    parser.add_argument('--epochs', default=100,
                        help='number of total epochs to run')
    parser.add_argument('--learning_rate', default=1e-2,
                        help='initial learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', default=4,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--adjust_lr', default=True,
                        help='apply learning rate decay or not\
                        (default: True)'
                        )
    parser.add_argument('--device',
                        default='cuda:0',
                        help='choose cpu or cuda:0 device"'
                        )
    parser.add_argument('--input_channels', default=3,
                        help='Number of channels in input tensor')
    parser.add_argument('--num_workers', default=4,
                        help='Number of workers in dataloader')
    parser.add_argument('--use_multiple_gpu', default=False)
    args = parser.parse_args()
    return args


def adjust_learning_rate(optimizer, epoch, learning_rate):

    if learning_rate > 1e-6:
        lr = learning_rate / 2
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def post_process_disparity(disp):
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

class Model:

    def __init__(self, args):
        self.args = args

        # Set up model
        self.device = args.device
        self.model = get_model(args.model, input_channels=args.input_channels, pretrained=args.pretrained)
        self.model = self.model.to(self.device)
        if args.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)

        if args.mode == 'train':
            self.loss_function = MonodepthLoss(
                n=4,
                SSIM_w=0.,
                disp_gradient_w=0., lr_w=0.).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=args.learning_rate)
            self.val_n_img, self.val_loader = prepare_dataloader(args.val_data_dir, 'val',
                                                                 args.batch_size,
                                                                 (args.input_height, args.input_width),
                                                                 args.num_workers)
        else:
            self.model.load_state_dict(torch.load(args.model_path))
            args.augment_parameters = None
            args.do_augmentation = False
            args.batch_size = 1

        # Load data
        self.output_directory = args.output_directory
        self.input_height = args.input_height
        self.input_width = args.input_width

        self.n_img, self.loader = prepare_dataloader(args.data_dir, args.mode, args.batch_size,
                                                     (args.input_height, args.input_width),
                                                     args.num_workers)

        if 'cuda' in self.device:
            torch.cuda.synchronize()

    def train(self):
        losses = []
        val_losses = []
        best_loss = float('Inf')
        best_val_loss = float('Inf')

        for epoch in tqdm(range(self.args.epochs), desc="epochs"):
            if self.args.adjust_lr:
                adjust_learning_rate(self.optimizer, epoch,
                                     self.args.learning_rate)
            c_time = time.time()
            running_loss = 0.0
            self.model.train()
            for data in tqdm(self.loader, desc="training"):
                # Load data
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']
                transform_left_images = data['transform_left_images']
                transform_right_images = data['transform_right_images']

                # One optimization iteration
                self.optimizer.zero_grad()
                disps = self.model(left)
                loss = self.loss_function(disps, [left, right, transform_left_images, transform_right_images])
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

                vis.images(self.loss_function.left[0], win='left[0]', opts=dict(title='left[0]'))
                vis.images(self.loss_function.right[0], win='right[0]', opts=dict(title='right[0]'))
                vis.images(
                    tensor2array(self.loss_function.disp_left_est[0][0, :, :, :], max_value=None,
                                 colormap='magma'), win='disp_left_est[0]', opts=dict(title='disp_left_est[0]'))
                vis.images(self.loss_function.left_est[0][0], win='left_est[0]', opts=dict(title='left_est[0]'))
                vis.images(
                    tensor2array(self.loss_function.disp_right_est[0][0, :, :, :], max_value=None,
                                 colormap='magma'), win='disp_right_est[0]', opts=dict(title='disp_right_est[0]'))
                vis.images(self.loss_function.right_est[0][0], win='right_est[0]', opts=dict(title='right_est[0]'))
                running_loss += loss.item()

            running_val_loss = 0.0
            self.model.eval()
            for data in self.val_loader:
                data = to_device(data, self.device)
                left = data['left_image']
                left_gt_image = data['left_gt_image']
                disps = self.model(left)
                vis.images(left[0], win='val_left[0]', opts=dict(title='val_left[0]'))
                vis.images(
                    tensor2array(left_gt_image[0], max_value=None,
                                 colormap='magma'), win='val_gt[0]', opts=dict(title='gt[0]'))
                # loss = self.loss_function(disps, [left, right])
                # val_losses.append(loss.item())
                # running_val_loss += loss.item()

            # Estimate loss per image
            running_loss /= self.n_img / self.args.batch_size
            running_val_loss /= self.val_n_img / self.args.batch_size
            print(
                'Epoch:',
                epoch + 1,
                'train_loss:',
                running_loss,
                'val_loss:',
                running_val_loss,
                'time:',
                round(time.time() - c_time, 3),
                's',
            )
            # self.save(self.args.model_path[:-4] + '_last.pth')
            # if running_val_loss < best_val_loss:
            #     self.save(self.args.model_path[:-4] + '_cpt.pth')
            #     best_val_loss = running_val_loss
            #     print('Model_saved')

        # print('Finished Training. Best loss:', best_loss)
        # self.save(self.args.model_path)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def test(self):
        self.model.eval()
        disparities = np.zeros((self.n_img,
                                self.input_height, self.input_width),
                               dtype=np.float32)
        disparities_pp = np.zeros((self.n_img,
                                   self.input_height, self.input_width),
                                  dtype=np.float32)
        with torch.no_grad():
            for (i, data) in enumerate(self.loader):
                # Get the inputs
                data = to_device(data, self.device)
                left = data.squeeze()
                # Do a forward pass
                disps = self.model(left)
                disp = disps[0][:, 0, :, :].unsqueeze(1)
                disparities[i] = disp[0].squeeze().cpu().numpy()
                disparities_pp[i] = \
                    post_process_disparity(disps[0][:, 0, :, :] \
                                           .cpu().numpy())

        np.save(self.output_directory + '/disparities.npy', disparities)
        np.save(self.output_directory + '/disparities_pp.npy',
                disparities_pp)
        print('Finished Testing')


def main():
    args = return_arguments()
    if args.mode == 'train':
        model = Model(args)
        model.train()
    elif args.mode == 'test':
        model_test = Model(args)
        model_test.test()


if __name__ == '__main__':
    main()
