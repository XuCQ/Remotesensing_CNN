import numpy as np
import random
import matplotlib.pyplot as plt
import platform
import cv2
from sklearn import preprocessing


# 图像白化，图像翻转
class data_augmentation(object):
    def __init__(self, images, flip=False, noise=False, noise_mean=0, noise_std=0.01, normalization=False):
        self.images = images
        self.flip = flip
        self.normalization = normalization
        self.noise = noise
        self.noise_mena = noise_mean
        self.noise_std = noise_std
        # 图像翻转
        if self.flip:
            self.image_flip()
        # 添加噪声
        if self.noise:
            self.image_noise()
        # 标准化
        if self.normalization:
            self.image_normalization()
        return self.images

    def image_flip(self):
        for i in range(self.images.shape[0]):
            old_image = self.images[i, :, :, , :]
            if np.random.random() < 0.5:
                new_image = cv2.flip(old_image, 1)
            else:
                new_image = old_image
            self.images[i, :, :, :] = new_image

    def image_noise(self):
        for i in range(self.images.shape[0]):
            old_image = self.images[i, :, :, :]
            new_image = old_image
            if np.random.random() < 0.5:
                for i in range(old_image[0]):
                    for j in range(old_image[1]):
                        for k in range(old_image[2]):
                            new_image[i][j][k] += random.gauss(self.mean, self.std)
                self.images[i, :, :, :] = new_image

    def image_normalization(self):
        self.images = preprocessing.scale(self.images)
