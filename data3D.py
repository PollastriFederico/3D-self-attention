import os

from torchvision import transforms
import torch

from torch.utils.data import DataLoader
import numpy as np
import numpy
import imgaug as ia
import random
import matplotlib.pyplot as plt

from scipy.ndimage import rotate
import yaml_segmentation_dataset as public_prostata_dataset
import transforms3D as s_transforms
from losses import get_criterion


class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, angle_spectrum=8, mode='nearest', order=0, execution_probability=0.5, **kwargs):
        self.angle_spectrum = angle_spectrum
        self.mode = mode
        self.order = order
        self.execution_probability = execution_probability

    def __call__(self, image, gt):
        if np.random.uniform(0, 1) < self.execution_probability:
            angle = np.random.randint(-self.angle_spectrum, self.angle_spectrum)
            image = rotate(image, angle, axes=(1, 2), reshape=False, order=self.order, mode=self.mode)
            gt = rotate(gt, angle, axes=(1, 2), reshape=False, order=self.order, mode=self.mode)
            return image, gt
        return image, gt


class CenterCrop:

    def __init__(self, target_shape):
        self.target_shape = target_shape

    def __call__(self, image, gt=None):
        z_offset = image.shape[-3] - self.target_shape[0]
        y_offset = image.shape[-2] - self.target_shape[1]
        x_offset = image.shape[-1] - self.target_shape[2]
        z_offset = int(np.floor(z_offset / 2.)), image.shape[-3] - int(np.ceil(z_offset / 2.))
        y_offset = int(np.floor(y_offset / 2.)), image.shape[-2] - int(np.ceil(y_offset / 2.))
        x_offset = int(np.floor(x_offset / 2.)), image.shape[-1] - int(np.ceil(x_offset / 2.))

        crop_img = image[..., z_offset[0]:z_offset[1], y_offset[0]:y_offset[1], x_offset[0]:x_offset[1]]
        if gt is not None:
            assert image.shape == gt.shape
            gt = gt[..., z_offset[0]:z_offset[1], y_offset[0]:y_offset[1], x_offset[0]:x_offset[1]]
            return crop_img, gt
        return crop_img


class RandomCrop:

    def __init__(self, target_shape):
        self.target_shape = target_shape

    def __call__(self, image, gt=None):
        Z_start = random.randint(0, image.shape[1] - self.target_shape[0])
        Y_start = random.randint(0, image.shape[2] - self.target_shape[1])
        X_start = random.randint(0, image.shape[3] - self.target_shape[2])

        crop_img = image[:, Z_start:Z_start + self.target_shape[0], Y_start:Y_start + self.target_shape[1],
                   X_start:X_start + self.target_shape[2]]
        if gt is not None:
            assert image.shape == gt.shape
            gt = gt[:, Z_start:Z_start + self.target_shape[0], Y_start:Y_start + self.target_shape[1],
                 X_start:X_start + self.target_shape[2]]
            return crop_img, gt
        return crop_img


class OrderedCrop:

    def __init__(self, target_shape):
        self.target_shape = target_shape

    def __call__(self, image, crop_start, gt=None):
        Z_start, Y_start, X_start = crop_start

        crop_img = image[..., Z_start:Z_start + self.target_shape[0], Y_start:Y_start + self.target_shape[1],
                   X_start:X_start + self.target_shape[2]]
        if gt is not None:
            assert image.shape == gt.shape
            gt = gt[..., Z_start:Z_start + self.target_shape[0], Y_start:Y_start + self.target_shape[1],
                 X_start:X_start + self.target_shape[2]]
            return crop_img, gt
        return crop_img


class OrderedPadder:
    def __init__(self, output_shape, crop_size):
        self.crop_size = crop_size
        self.pred_mask = torch.zeros(output_shape, device='cuda')
        self.voters_mask = torch.zeros_like(self.pred_mask, device='cuda')

    def __call__(self, image, crop_start):
        self.pred_mask[..., crop_start[0]:crop_start[0] + self.crop_size[0],
        crop_start[1]:crop_start[1] + self.crop_size[1],
        crop_start[2]:crop_start[2] + self.crop_size[2]] += image

        self.voters_mask[..., crop_start[0]:crop_start[0] + self.crop_size[0],
        crop_start[1]:crop_start[1] + self.crop_size[1],
        crop_start[2]:crop_start[2] + self.crop_size[2]] += 1.0

    def get_prediction(self):
        return (self.pred_mask / self.voters_mask)


class CenterPad:
    def __init__(self, final_shape):
        self.size = final_shape

    def __call__(self, image):
        z_offset = self.size[0] - image.shape[-3]
        y_offset = self.size[1] - image.shape[-2]
        x_offset = self.size[2] - image.shape[-1]

        z_offset = int(np.floor(z_offset / 2.)), int(np.ceil(z_offset / 2.))
        y_offset = int(np.floor(y_offset / 2.)), int(np.ceil(y_offset / 2.))
        x_offset = int(np.floor(x_offset / 2.)), int(np.ceil(x_offset / 2.))

        return torch.nn.functional.pad(image, [x_offset[0], x_offset[1], y_offset[0], y_offset[1], z_offset[0], z_offset[1]])


class RandomHorizontalFlip:

    def __init__(self, execution_probability=0.5):
        self.execution_probability = execution_probability

    def __call__(self, image, gt):
        if np.random.uniform(0, 1) < self.execution_probability:
            image = np.flip(image, axis=2)
            gt = np.flip(gt, axis=2)
            return image, gt
        return image, gt


# Cutout Data Augmentation Class. Can be used in the torchvision compose pipeline
class CutOut(object):
    """Randomly mask out one or more patches from an image.
    Args:
    n_holes (int): Number of patches to cut out of each image.
    length (int): The length (in pixels) of each square patch.

    Return:
     -image with mask and original image
    """

    def __init__(self, n_holes, length):
        # if isinstance(n_holes, int):
        #     self.n_holes = n_holes
        #     self.is_list = False
        # else:
        #     self.is_list = True
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        if not (self.length[0] == 0 and len(self.length) == 1):
            # if self.is_list:
            t_n_holes = random.choice(self.n_holes)
            # else:
            #     t_n_holes = self.n_holes

            h = img.size(1)
            w = img.size(2)

            mask = numpy.ones((h, w), numpy.float32)

            for n in range(t_n_holes):
                t_length = random.choice(self.length)
                y = random.randint(0, h - 1)
                x = random.randint(0, w - 1)

                y1 = numpy.clip(y - t_length // 2, 0, h)
                y2 = numpy.clip(y + t_length // 2, 0, h)
                x1 = numpy.clip(x - t_length // 2, 0, w)
                x2 = numpy.clip(x + t_length // 2, 0, w)

                mask[y1: y2, x1: x2] = 0.

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            return img * mask
        else:
            return img


class ImgAugTransform:
    def __init__(self, config_code, size_w=512, size_h=512):
        self.w = size_w
        self.h = size_h
        self.config = config_code

        # sometimes = lambda aug: ia.augmenters.Sometimes(0.5, aug)
        sometimes = lambda aug: ia.augmenters.Sometimes(1, aug)

        # FIRST "BIT" OF CONFIG CODE IS THE PADDING MODE
        cc = max(self.config, 0)
        if cc % 2:
            self.mode = 'constant'
            cc -= 1
        else:
            self.mode = 'reflect'
        self.possible_aug_list = [
            None,  # dummy for padding mode                                                         # 1
            None,  # placeholder for future inclusion                                               # 2
            sometimes(ia.augmenters.AdditivePoissonNoise((0, 10), per_channel=True)),  # 4
            sometimes(ia.augmenters.Dropout((0, 0.02), per_channel=False)),  # 8
            sometimes(ia.augmenters.GaussianBlur((0, 0.8))),  # 16
            sometimes(ia.augmenters.AddToHueAndSaturation((-20, 10))),  # 32
            sometimes(ia.augmenters.GammaContrast((0.75, 1.25))),  # 64
            None,  # placeholder for future inclusion                                               # 128
            None,  # placeholder for future inclusion                                               # 256
            sometimes(ia.augmenters.PiecewiseAffine((0, 0.02))),  # 512
            sometimes(ia.augmenters.Affine(shear=(-3, 3), mode=self.mode)),  # 1024
            # sometimes(ia.augmenters.CropToFixedSize(width=self.h, height=self.h)),                  # 2048
            ia.augmenters.CropToFixedSize(width=self.h, height=self.h),  # 2048
        ]
        self.aug_list = [
            ia.augmenters.Fliplr(0.5),
            # ia.augmenters.Flipud(0.5),
        ]

        # FIRST "BIT" OF CONFIG CODE IS THE PADDING MODE, FIRST LOOP IS A DUMMY
        for i in range(len(self.possible_aug_list)):
            if cc % 2:
                self.aug_list.append(self.possible_aug_list[i])
            cc = cc // 2
            if not cc:
                break

        self.aug_list.append(ia.augmenters.Affine(rotate=(-10, 10), mode=self.mode))
        self.aug = ia.augmenters.Sequential(self.aug_list)
        if self.config >= 0:
            print(self.mode)
            for a in self.aug_list:
                print(a.name)

    def __call__(self, img, grnd):
        self.aug.reseed(random.randint(1, 10000))

        plot(img)
        plot(grnd)

        img = ia.augmenters.Resize({"width": self.w, "height": self.h}).augment_image(img)
        grnd = ia.augmenters.Resize({"width": self.w, "height": self.h},
                                    interpolation='nearest').augment_image(grnd)
        plot(img)
        plot(grnd)

        if self.config != -1:
            grnd = np.expand_dims(grnd, axis=-1)
            grnd = np.expand_dims(grnd, axis=0)
            img, grnd = self.aug(image=img, segmentation_maps=grnd)
            grnd = np.squeeze(grnd)
            plot(img)
            plot(grnd)

        # return np.ascontiguousarray(img), np.ascontiguousarray(grnd)
        return img, grnd


def get_dataset(dname='prostata', interp_size=224, crop_size=(32, 128, 128), crop_type='center', net='Unet'):
    if crop_type == 'center':
        train_crop_fn = test_crop_fn = CenterCrop(crop_size)
    elif crop_type == 'random':
        train_crop_fn = RandomCrop(crop_size)
        test_crop_fn = CenterPad((64, interp_size, interp_size))  # TODO: hardcoded value Z + 4
        # test_crop_fn = lambda x: x
    else:
        raise ValueError('wrong type for crop type function')

    training_transforms = s_transforms.Compose([
        RandomHorizontalFlip(execution_probability=1),
        RandomRotate(),
        s_transforms.DCMToTensorNoGrnd(),
        s_transforms.Interpolate((interp_size, interp_size)),
        train_crop_fn,
        s_transforms.Normalize(0.2296, 0.1369),
        # CutOut(*cutout_params)
    ])

    test_transforms = s_transforms.Compose([
        # inference_imgaug_transforms,
        s_transforms.DCMToTensorNoGrnd(),
        s_transforms.Normalize(0.2296, 0.1369),

    ])

    image_transform = transforms.Compose([
        s_transforms.Interpolate((interp_size, interp_size)),
        test_crop_fn
    ])

    valid_transforms = test_transforms

    n_channels = 3
    classes = dname.split('_')[2:]
    dataset = public_prostata_dataset.YAMLSegmentationDataset_3D(transform=training_transforms, split=['training'],
                                                                 n_channels=n_channels, classes=classes)
    val_dataset = public_prostata_dataset.YAMLSegmentationDataset_3D(transform=valid_transforms,
                                                                     input_image_transform=image_transform,
                                                                     split=['validation'], n_channels=n_channels, classes=classes)
    test_dataset = public_prostata_dataset.YAMLSegmentationDataset_3D(transform=test_transforms,
                                                                      input_image_transform=image_transform,
                                                                      split=['test'], n_channels=n_channels, classes=classes)

    return dataset, val_dataset, test_dataset


def compute_weights(dataset, interp_size, crop_size, num_classes):
    l = [im['gt'].flatten() for im in dataset.imgs]
    result_w = []
    for k in range(1, num_classes + 1):
        positives_l = [np.sum(dcm == k) / (dcm.size / interp_size / interp_size / 60) for dcm in l]
        # count_l = [np.size(dcm) for dcm in l]
        count_l = crop_size[0] * crop_size[1] * crop_size[2] * len(l)

        stats = np.sum(positives_l) / np.sum(count_l)
        result_w.append(1.0 / stats)
    return result_w



def find_stats(data_loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _, _ in data_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    print("\ntraining")
    print("mean: " + str(mean) + " | std: " + str(std))


def plot(img):
    return
    plt.figure()
    plt.imshow(img)
    plt.show(block=False)


if __name__ == '__main__':
    dataset, _, _ = get_dataset(
        dname='public_prostata_Target',
        interp_size=256,
        crop_size=(56, 144, 144)
    )
    loader = DataLoader(dataset,
               batch_size=8,
               shuffle=True,
               num_workers=0,
               drop_last=True,
               pin_memory=True)
    for x in loader:
        continue
