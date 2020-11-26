import os

import cv2
from PIL import Image
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import RandomRotation


class ImageFolderWithPaths(ImageFolder):

    def __init__(self, custom_labels=None, **kwargs):
        super().__init__(**kwargs)
        self.custom_labels = custom_labels

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        if self.custom_labels:
            original_tuple = (original_tuple[0], self.custom_labels[path])
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class SingleFolderDataset(Dataset):
    """
    Creating a dataset from a directory with no subfolders. Each image can be repeated multiple times, for domains with
    small amount of images.
    """
    def __init__(self, root, transform=None, num_repeats=1):
        self.root = root
        self.transform = transform
        im_list = os.listdir(root)
        repeated_im_list = []
        for im in im_list:
            for _ in range(num_repeats):
                repeated_im_list.append(im)
        self.im_list = repeated_im_list

    def __getitem__(self, index):
        path = os.path.join(self.root, self.im_list[index])
        im = Image.open(path)
        if self.transform:
            im = self.transform(im)
        return im, path

    def __len__(self):
        return len(self.im_list)


class EncoderDataSetWrapper(object):

    def __init__(self, config):
        self.batch_size = config['modalities_encoder']['batch_size']
        self.num_workers = config['data']['num_workers']
        self.color_jitter_strength = config['modalities_encoder']['color_jitter_strength']
        self.input_shape = (config['data']['crop_image_height'], config['data']['crop_image_width'], 3)
        self.augmentations = config['modalities_encoder']['augmentations']
        self.root = config['data']['train_root']

    def get_data_loader(self):
        data_augment = self._get_augmentations_transform()
        transform = AugDataTransform(data_augment)
        train_dataset = ImageFolderWithPaths(root=self.root, transform=transform)
        sampler = SubsetRandomSampler(range(len(train_dataset)))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)
        return train_loader

    def _get_augmentations_transform(self):
        augmentations = []
        if "crop" in self.augmentations:
            augmentations.append(transforms.RandomResizedCrop(size=self.input_shape[0]))
        if "horizontal_flip" in self.augmentations:
            augmentations.append(transforms.RandomHorizontalFlip())
        if "shuffle" in self.augmentations:
            augmentations.append(transforms.RandomApply([ShufflePatches(8)], p=0.6))
        if "color_jitter" in self.augmentations:
            color_jitter = transforms.ColorJitter(0.8 * self.color_jitter_strength, 0.8 * self.color_jitter_strength,
                                                  0.8 * self.color_jitter_strength, 0.2 * self.color_jitter_strength)
            augmentations.append(transforms.RandomApply([color_jitter], p=0.8))
        if "gray_scale" in self.augmentations:
            augmentations.append(transforms.RandomGrayscale(p=0.2))
        if "blur" in self.augmentations:
            augmentations.append(GaussianBlur(kernel_size=int(0.1 * self.input_shape[0] - 1)))
        augmentations.append(transforms.ToTensor())
        transform = transforms.Compose(augmentations)
        return transform


class AugDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


class ShufflePatches(object):
    def __init__(self, sqrt_num_patches, only_horizontal=False, rotate=False):
        self.sqrt_num_patches = sqrt_num_patches
        self.only_horizontal = only_horizontal
        self.rotate = rotate

    def __call__(self, sample):
        a = sample.height / (2 * self.sqrt_num_patches)
        tiles = [[None for _ in range(self.sqrt_num_patches)] for _ in range(self.sqrt_num_patches)]
        for n in range((self.sqrt_num_patches) ** 2):
            i = int(n / self.sqrt_num_patches)
            j = n % self.sqrt_num_patches
            c = [a * i * 2 + a, a * j * 2 + a]
            tile = sample.crop((c[1]-a,c[0]-a,c[1]+a+1,c[0]+a+1))
            tiles[i][j] = tile
        if self.only_horizontal:
            order = np.array([np.random.permutation(self.sqrt_num_patches) for _ in range(self.sqrt_num_patches)])
            order += (np.arange(self.sqrt_num_patches).reshape(1, -1) * self.sqrt_num_patches).T
        else:
            order = np.random.permutation(self.sqrt_num_patches ** 2).reshape(self.sqrt_num_patches, self.sqrt_num_patches)
        result = Image.new('RGB', sample.size)
        for i in range(self.sqrt_num_patches):
            for j in range(self.sqrt_num_patches):
                tile = tiles[i][j]
                rotation_angle = np.random.choice((0, 90, 180, 270)) if self.rotate else 0
                tile = RandomRotation((rotation_angle, rotation_angle))(tile)
                result.paste(tile, ((order[i,j] % self.sqrt_num_patches) * int(a * 2), (order[i,j] // self.sqrt_num_patches) * int(a * 2)))
        return result
