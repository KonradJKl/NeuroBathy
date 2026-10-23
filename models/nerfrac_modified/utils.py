# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# TINY_NUMBER and colorize_np come from NeRF++ (BSD-2-Clause, Copyright (c) 2020,
# the NeRF++ authors), see LICENSE-BSD-NERFPP.
#
# Modified by Konrad Kloska, 2026: SSIM computed with OpenCV (ssim,
# ssim_single_channel), following BasicSR's calculate_ssim; unused helpers removed.
# A copy of the Apache license is in LICENSE-APACHE.

import math

import cv2
import lpips
import matplotlib
import numpy as np

TINY_NUMBER = 1e-6  # float32 only has 7 decimal digits precision


def colorize_np(x, cmap_name="jet", range=None):
    """An [H, W] array as [H, W, 3] RGB in [0, 1]; range defaults to the 1st
    percentile and the maximum of x."""
    if range is not None:
        vmin, vmax = range
    else:
        vmin, vmax = np.percentile(x, (1, 100))
        vmax += TINY_NUMBER

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin)

    cmap = matplotlib.colormaps[cmap_name]
    return cmap(x)[:, :, :3]


class util_of_lpips:
    def __init__(self, net, use_gpu=False):
        self.loss_fn = lpips.LPIPS(net=net)
        self.use_gpu = use_gpu
        if use_gpu:
            self.loss_fn.cuda()

    def calc_lpips(self, img0, img1):
        # uint8 RGB -> tensors in [-1, 1]
        img0 = lpips.im2tensor(img0)
        img1 = lpips.im2tensor(img1)

        if self.use_gpu:
            img0 = img0.cuda()
            img1 = img1.cuda()
        dist01 = self.loss_fn.forward(img0, img1)
        return dist01


def psnr(img1, img2):
    """Mean of the per-channel PSNR of two uint8 [H, W, 3] images (each channel capped
    at 100 dB), not the PSNR over all channels."""
    img1 = img1.transpose(2, 0, 1)
    img2 = img2.transpose(2, 0, 1)

    res1B = psnr_1(img1[0], img2[0])
    res1G = psnr_1(img1[1], img2[1])
    res1R = psnr_1(img1[2], img2[2])

    psnr = (res1B + res1G + res1R) / 3

    return psnr


def psnr_1(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1e-10:
        return 100
    psnr_1 = 20 * math.log10(255 / math.sqrt(mse))
    return psnr_1


def ssim(img1, img2):
    """Mean of the per-channel SSIM of two [H, W, 3] images in the 0-255 range."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    return np.mean([ssim_single_channel(img1[..., i], img2[..., i]) for i in range(3)])


def ssim_single_channel(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    # drop the 5 px border, so that only full 11x11 windows count
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()
