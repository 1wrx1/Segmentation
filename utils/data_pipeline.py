"""
Updated: 2021.06.10
New features:
(1) Add more data augmentation for experiments.
"""
import os
import cv2
import math
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from skimage import img_as_ubyte
from skimage.color import gray2rgb
from skimage.exposure import equalize_adapthist, rescale_intensity, adjust_gamma
from PIL import Image


__all__ = ['SegmentationDataset', 'CLAHE', 'Rescale', 'RandomCrop', 'CenterCrop', 'Pad', 'CropOrPad',
           'RandomHorizontalFlip', 'RandomVerticalFlip', 'Rotate', 'RandomGamma', 'RandomAffine',
           'RandomElasticDeformation', 'ToTensor']


class SegmentationDataset(Dataset):
    def __init__(self, image_root, mask_root, subject_list, transform=None, vis=False):
        self.image_root = image_root
        self.mask_root = mask_root
        self.transform = transform
        self.subject_list = subject_list
        self.vis = vis
        self.file_list = self.get_file_list()

    def get_file_list(self):
        file_list = []
        for file in os.listdir(self.image_root):
            patient_id = file.split('_')[1]
            if patient_id in self.subject_list:
                file_list.append(file)
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i):
        image_filename = os.path.join(self.image_root, self.file_list[i])
        mask_filename = os.path.join(self.mask_root, self.file_list[i])
        # 分离文件名和扩展名
        mask_filename, _ = os.path.splitext(mask_filename)
        # 更改扩展名
        mask_filename = mask_filename + '.bmp'

        data_id = self.file_list[i].split('.')[0]

        # image = np.load(image_filename)
        image = Image.open(image_filename)
        image = np.array(image)

        if image.shape[2] !=3:
            image = gray2rgb(remap_by_window(image, window_width=80, window_level=1035))
        mask = Image.open(mask_filename)
        mask = np.array(mask)
        mask[mask > 1] = 1

        sample = {'image': image, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)

        if self.vis:
            return sample, data_id
        else:
            return sample


class CLAHE:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) algorithm for image augmentation.
    """
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        img_processed = img_as_ubyte(equalize_adapthist(image))
        return {'image': img_processed, 'mask': mask}


class Rescale:
    """
    Rescale images to a certain shape.
    image shape: (height, width, channels)
    mask shape: (height, width)
    output_size: int or tuple
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # # set preserve_range=True，otherwise image & mask will be transformed into float format
        # image = resize(image, (new_h, new_w), preserve_range=True)
        # mask = resize(mask, (new_h, new_w), preserve_range=True)
        image = cv2.resize(image, (new_h, new_w))
        mask = cv2.resize(mask, (new_h, new_w))

        return {'image': image.astype(np.uint8), 'mask': mask.astype(np.int)}


class RandomCrop:
    """
    Crop a patch randomly of the input image.
    image shape: (height, width, channels)
    mask shape: (height, width)
    output_size: int or tuple
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w, :]
        mask = mask[top: top + new_h, left: left + new_w]

        return {'image': image, 'mask': mask}


class CenterCrop:
    """
    Perform center crop
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        image = image[top: top + new_h, left: left + new_w, :]
        mask = mask[top: top + new_h, left: left + new_w]

        return {'image': image, 'mask': mask}


class Pad:
    """
    Pad the image to a desired shape
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        assert new_h > h, 'Padding desired shape shall be larger than the original.'

        h_diff = new_h - h
        w_diff = new_w - w

        image = np.pad(image, ((h_diff // 2, h_diff // 2), (w_diff // 2, w_diff // 2), (0, 0)), mode='reflect')
        mask = np.pad(mask, ((h_diff // 2, h_diff // 2), (w_diff // 2, w_diff // 2)), mode='reflect')

        return {'image': image, 'mask': mask}


class CropOrPad:
    """
    Perform crop or pad to resize the image to a desired shape.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        # perform center crop
        if new_h <= h:
            sample = CenterCrop(self.output_size)(sample)
        else:
            sample = Pad(self.output_size)(sample)

        return sample


class RandomHorizontalFlip:
    """
    Random horizontal flip.
    image shape: (height, width, channels)
    mask shape: (height, width)
    possibility: possibility for flip
    """

    def __init__(self, possibility=0.5):
        assert isinstance(possibility, (int, float))
        self.possibility = possibility

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if random.random() <= self.possibility:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)

        return {'image': image, 'mask': mask}


class RandomVerticalFlip:
    """
    Random vertical flip.
    image shape: (height, width, channels)
    mask shape: (height, width)
    possibility: possibility for flip
    """
    def __init__(self, possibility=0.5):
        assert isinstance(possibility, (int, float))
        self.possibility = possibility

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if random.random() <= self.possibility:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)

        return {'image': image, 'mask': mask}


class Rotate:
    """
    Random rotation.
    image shape: (height, width, channels)
    mask shape: (height, width)
    possibility: possibility for rotate
    range: range of rotation angles
    """
    def __init__(self, possibility=0.5, range=20):
        self.possibility = possibility
        self.range = range

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        width, height = image.shape[:2]

        if random.random() <= self.possibility:
            angle = np.random.randint(0, self.range)

            center = (width // 2, height // 2)
            # 得到旋转矩阵，第一个参数为旋转中心，第二个参数为旋转角度，第三个参数为旋转之前原图像缩放比例
            M = cv2.getRotationMatrix2D(center, -angle, 1)
            # 进行仿射变换，第一个参数图像，第二个参数是旋转矩阵，第三个参数是变换之后的图像大小
            image = cv2.warpAffine(image, M, (width, height))
            mask = cv2.warpAffine(mask.astype(np.uint8), M, (width, height))

        return {'image': image.astype(np.uint8), 'mask': mask.astype(np.int)}


class RandomGamma:
    """
    Random gamma transform
    """
    def __init__(self, possibility=0.5, range=0.2):
        self.possibility = possibility
        self.range = range

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if random.random() <= self.possibility:
            rand_range = random.uniform(1 - self.range, 1 + self.range)
            image = adjust_gamma(image, rand_range)

        return {'image': image, 'mask': mask}


class RandomAffine:
    """
    Random affine transform: scale/rotate/translation/shear
    https://stackabuse.com/affine-image-transformations-in-python-with-numpy-pillow-and-opencv
    """
    def __init__(self, possibility=0.5, scale=0.3, rotate=20, translation=0.1, shear=0.1):
        self.possibility = possibility
        self.scale_range = scale
        self.rotate_range = rotate
        self.translation_range = translation
        self.shear_range = shear

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask'].astype(np.uint8)
        height, width = image.shape[:2]
        if random.random() <= self.possibility:
            scale = random.uniform(1 - self.scale_range, 1 + self.scale_range)
            new_h, new_w = int(height * scale), int(width * scale)
            rotate = random.uniform(- self.rotate_range * math.pi / 180, self.rotate_range * math.pi / 180)
            trans_x = random.uniform(- self.translation_range * new_w, self.translation_range * new_w)
            trans_y = random.uniform(- self.translation_range * new_h, self.translation_range * new_h)
            shear = random.uniform(- self.shear_range, self.shear_range)
            # move image center to the zero-point
            T_pos = np.array([[1, 0, width],
                              [0, 1, height],
                              [0, 0, 1]])
            # rotate matrix
            T_r = np.array([[math.cos(rotate), math.sin(rotate), 0],
                           [- math.sin(rotate), math.cos(rotate), 0],
                            [0, 0, 1]])
            # translation matrix
            T_t = np.array([[1, 0, trans_x],
                           [0, 1, trans_y],
                            [0, 0, 1]])
            # shear translation matrix
            T_s = np.array([[1, shear, 0],
                            [shear, 1, 0],
                             [0, 0, 1]])
            # scale translation matrix
            T_sc = np.array([[scale, 0, 0],
                             [0, scale, 0],
                             [0, 0, 1]])
            # move back
            T_neg = np.array([[1, 0, -width/scale],
                              [0, 1, -height/scale],
                              [0, 0, 1]])
            T = T_pos@T_sc@T_t@T_r@T_s@T_neg
            image_t = cv2.warpAffine(image, T[:2, :].astype(np.float32), (new_w, new_h))
            mask_t = cv2.warpAffine(mask, T[:2, :].astype(np.float32), (new_w, new_h))
            sample = {'image': image_t, 'mask': mask_t.astype(np.int)}

        return CropOrPad((width, height))(sample)


class RandomElasticDeformation:
    """
    [Simard2003] Simard, Steinkraus and Platt, "Best Practices for Convolutional Neural Networks applied to
    Visual Document Analysis", in Proc. of the International Conference on Document Analysis and Recognition, 2003
    """
    def __init__(self, possibility=0.5, alpha=34, sigma=4):
        self.possibility = possibility
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask'].astype(np.uint8)
        height, width = image.shape[:2]
        # kernel size for Gaussian filtering
        ksize = (self.sigma * 6 + 1, self.sigma * 6 + 1)
        # coordinate shift for elastic transform
        dx = cv2.GaussianBlur(np.random.rand(height, width) * 2 - 1, ksize, self.sigma) * self.alpha
        dy = cv2.GaussianBlur(np.random.rand(height, width) * 2 - 1, ksize, self.sigma) * self.alpha

        # original coordinate
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # new coordinate
        map_x = np.float32(x + dx)
        map_y = np.float32(y + dy)

        # map the vanilla coordinates to the transformed
        image_t = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        mask_t = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        return {'image': image_t, 'mask': mask_t.astype(np.int)}


class ToTensor(object):
    """
    Convert numpy array into torch.tensor
    """
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = np.transpose(image, (2, 0, 1)) / 255.0

        return {'image': torch.from_numpy(np.ascontiguousarray(image).astype(np.float32)),
                'mask': torch.from_numpy(np.ascontiguousarray(mask))}


def remap_by_window(float_data, window_width, window_level):
    """
    CT window transform
    """
    low = int(window_level - window_width // 2)
    high = int(window_level + window_width // 2)
    output = rescale_intensity(float_data, in_range=(low, high), out_range=np.uint8).astype(np.uint8)
    return output
