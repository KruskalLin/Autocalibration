import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MonodepthLoss(nn.modules.Module):
    def __init__(self, n=4, SSIM_w=0.85, disp_gradient_w=1.0, lr_w=1.0):
        super(MonodepthLoss, self).__init__()
        self.SSIM_w = SSIM_w
        self.disp_gradient_w = disp_gradient_w
        self.lr_w = lr_w
        self.n = n

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = img.size()
        h = s[2]
        w = s[3]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(nn.functional.interpolate(img,
                                                         size=[nh, nw], mode='bilinear',
                                                         align_corners=True))
        return scaled_imgs

    def gradient_x(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def gradient_y(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    def apply_disparity(self, img, disp):
        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                                                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                                                     width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear',
                               padding_mode='zeros')

        return output

    def generate_image_left(self, img, disp):
        return self.apply_disparity(img, -disp)

    def generate_image_right(self, img, disp):
        return self.apply_disparity(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d
        ssim = torch.clamp((1 - SSIM) / 2, 0, 1)
        return ssim

    def disp_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1,
                                           keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1,
                                           keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i]
                        for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i]
                        for i in range(self.n)]

        return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i])
                for i in range(self.n)]


    def forward(self, input, target):
        """
        Args:
            input [disp1, disp2, disp3, disp4]
            target [left, right]

        Return:
            (float): The loss
        """
        left, right, transform_left, transform_right = target
        left_pyramid = self.scale_pyramid(left, self.n)
        right_pyramid = self.scale_pyramid(right, self.n)
        self.left = left
        self.right = right

        transform_left_pyramids = [self.scale_pyramid(img, self.n) for img in transform_left]
        transform_right_pyramids = [self.scale_pyramid(img, self.n) for img in transform_right]

        # Prepare disparities
        disp_left_est = [d[:, 0, :, :].unsqueeze(1) for d in input]
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in input]

        disp_left_pyramid = self.scale_pyramid(disp_left_est[0], self.n)
        disp_right_pyramid = self.scale_pyramid(disp_right_est[0], self.n)

        scale_left_loss = [torch.mean(torch.abs(disp_left_pyramid[i] - disp_left_est[i])) for i in range(self.n)]
        scale_right_loss = [torch.mean(torch.abs(disp_right_pyramid[i] - disp_right_est[i])) for i in range(self.n)]
        scale_loss = sum(scale_left_loss + scale_right_loss)

        self.disp_left_est = disp_left_est
        self.disp_right_est = disp_right_est

        left_image_gradients_x = [self.gradient_x(img) for img in left_pyramid]
        left_image_gradients_y = [self.gradient_y(img) for img in left_pyramid]
        right_image_gradients_x = [self.gradient_x(img) for img in right_pyramid]
        right_image_gradients_y = [self.gradient_y(img) for img in right_pyramid]

        # Generate images
        left_est = [self.generate_image_left(right_pyramid[i], disp_left_est[i]) for i in range(self.n)]
        right_est = [self.generate_image_right(left_pyramid[i], disp_right_est[i]) for i in range(self.n)]
        self.left_est = left_est
        self.right_est = right_est

        transform_left_est = [[self.generate_image_left(pyramid[i], disp_left_est[i]) for i in range(self.n)] for pyramid in transform_right_pyramids]
        transform_right_est = [[self.generate_image_right(pyramid[i], disp_right_est[i]) for i in range(self.n)] for pyramid in transform_left_pyramids]

        left_image_gradients_x_est = [self.generate_image_left(right_image_gradients_x[i], disp_left_est[i]) for i in range(self.n)]
        left_image_gradients_y_est = [self.generate_image_left(right_image_gradients_y[i], disp_left_est[i]) for i in range(self.n)]
        right_image_gradients_x_est = [self.generate_image_left(left_image_gradients_x[i], disp_right_est[i]) for i in range(self.n)]
        right_image_gradients_y_est = [self.generate_image_left(left_image_gradients_y[i], disp_right_est[i]) for i in range(self.n)]

        # # L-R Consistency
        right_left_disp = [self.generate_image_left(disp_right_est[i],
                                                    disp_left_est[i]) for i in range(self.n)]
        left_right_disp = [self.generate_image_right(disp_left_est[i],
                                                     disp_right_est[i]) for i in range(self.n)]

        # Disparities smoothness
        disp_left_smoothness = self.disp_smoothness(disp_left_est, left_pyramid)
        disp_right_smoothness = self.disp_smoothness(disp_right_est, right_pyramid)

        # MAE
        l1_left = [torch.mean(torch.abs(left_est[i] - left_pyramid[i]))
                   for i in range(self.n)]
        l1_right = [torch.mean(torch.abs(right_est[i] - right_pyramid[i]))
                              for i in range(self.n)]

        l1_left_pool1 = [torch.mean(torch.abs(nn.AvgPool2d(5)(left_est[i]) - nn.AvgPool2d(5)(left_pyramid[i])))
                   for i in range(self.n)]
        l1_right_pool1 = [torch.mean(torch.abs(nn.AvgPool2d(5)(right_est[i]) - nn.AvgPool2d(5)(right_pyramid[i])))
                    for i in range(self.n)]

        l1_left_pool2 = [torch.mean(torch.abs(nn.AvgPool2d(11)(left_est[i]) - nn.AvgPool2d(11)(left_pyramid[i])))
                        for i in range(self.n)]
        l1_right_pool2 = [torch.mean(torch.abs(nn.AvgPool2d(11)(right_est[i]) - nn.AvgPool2d(11)(right_pyramid[i])))
                         for i in range(self.n)]

        l1_transform_left = [[torch.mean(torch.abs(transform_left_est[j][i] - transform_left_pyramids[j][i]))
                   for i in range(self.n)] for j in range(len(transform_left_pyramids))]
        l1_transform_right = [[torch.mean(torch.abs(transform_right_est[j][i] - transform_right_pyramids[j][i]))
                              for i in range(self.n)] for j in range(len(transform_right_pyramids))]

        l1_left_image_gradients_x = [torch.mean((left_image_gradients_x_est[i] - left_image_gradients_x[i]) ** 2)
                              for i in range(self.n)]
        l1_left_image_gradients_y = [torch.mean((left_image_gradients_y_est[i] - left_image_gradients_y[i]) ** 2)
                                     for i in range(self.n)]
        l1_right_image_gradients_x = [torch.mean((right_image_gradients_x_est[i] - right_image_gradients_x[i]) ** 2)
                                     for i in range(self.n)]
        l1_right_image_gradients_y = [torch.mean((right_image_gradients_y_est[i] - right_image_gradients_y[i]) ** 2)
                                     for i in range(self.n)]

        # SSIM
        ssim_left = [torch.mean(self.SSIM(left_est[i],
                                          left_pyramid[i])) for i in range(self.n)]
        ssim_right = [torch.mean(self.SSIM(right_est[i],
                                           right_pyramid[i])) for i in range(self.n)]

        image_loss_left = [self.SSIM_w * ssim_left[i]
                           + (1 - self.SSIM_w) * l1_left[i]
                           for i in range(self.n)]
        image_loss_right = [self.SSIM_w * ssim_right[i]
                            + (1 - self.SSIM_w) * l1_right[i]
                            for i in range(self.n)]
        # image_loss = sum(image_loss_left + image_loss_right + l1_transform_left + l1_transform_right)
        image_loss = sum(image_loss_left + image_loss_right) + \
                     sum(l1_left_pool1) + sum(l1_right_pool1) + sum(l1_left_pool2) + sum(l1_right_pool2)
                     # sum(np.sum(l1_transform_left, axis=1)) + sum(np.sum(l1_transform_right, axis=1))
                     # 2 * sum(l1_left_image_gradients_x) + 2 * sum(l1_left_image_gradients_y) + \
                     # 2 * sum(l1_right_image_gradients_x) + 2 * sum(l1_right_image_gradients_y)

        # L-R Consistency
        lr_left_loss = [torch.mean(torch.abs(right_left_disp[i] - disp_left_est[i]))
                        for i in range(self.n)]
        lr_right_loss = [torch.mean(torch.abs(left_right_disp[i] - disp_right_est[i]))
                         for i in range(self.n)]
        lr_loss = sum(lr_left_loss + lr_right_loss)

        # Disparities smoothness
        disp_left_loss = [torch.mean(torch.abs(disp_left_smoothness[i])) / 2 ** i for i in range(self.n)]
        disp_right_loss = [torch.mean(torch.abs(disp_right_smoothness[i])) / 2 ** i for i in range(self.n)]
        disp_gradient_loss = sum(disp_left_loss + disp_right_loss)

        loss = image_loss + self.disp_gradient_w * disp_gradient_loss + self.lr_w * lr_loss + scale_loss
        self.image_loss = image_loss
        self.disp_gradient_loss = disp_gradient_loss
        self.lr_loss = lr_loss
        return loss
