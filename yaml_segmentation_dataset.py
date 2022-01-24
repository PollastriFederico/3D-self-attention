import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import yaml
import os.path
import numpy as np
from yaml import CLoader as Loader
from glob import glob
import torch.utils.data as data

'''
mean: tensor([0.2296, 0.2296, 0.2296]) | std: tensor([0.1369, 0.1369, 0.1369])
'''

class YAMLSegmentationDataset_2D(data.Dataset):

    def __init__(self, dataset='/MYDIR/prostate.yml',
                 transform=None, input_image_transform=None, mask_transform=None,
                 split=['training'], classes=['Surface'], shape=(256, 256)):

        self.dataset = dataset
        self.transform = transform
        self.input_image_transform = input_image_transform
        self.mask_transform = mask_transform
        self.classes = classes if classes else ['Surface']
        self.shape = shape
        self.imgs = []
        self.masks = {}

        data_root = os.path.dirname(dataset)

        with open(self.dataset, 'r') as stream:
            try:
                d = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)

        for s in split:
            for i in d['split'][s]:
                img = {'img': os.path.join(data_root, d['images'][i]['location']).replace('\\', '/')}
                surface_mask = 0
                target_mask = 0
                for k, v in d['images'][i]['values'].items():
                    unix_path = os.path.join(data_root, d['images'][i]['values'][k]).replace('\\', '/')
                    if 'Surface' in self.classes and k == 'Surface':
                        surface_mask = np.load(unix_path)
                    if 'Target' in self.classes and 'Target' in k:
                        target_mask += np.load(unix_path)

                target_mask = np.clip(target_mask, 0, 1) * len(self.classes)
                img['gt'] = np.clip(surface_mask + target_mask, 0, 2)
                if img['gt'].size == 1 and img['gt'] == 0:
                    img['gt'] = np.zeros((60, *shape), dtype=np.uint8)
                self.imgs.append(img)

    def __getitem__(self, index):
        volume_idx = index // 60
        slice_idx = index % 60

        img_slices = glob(os.path.join(self.imgs[volume_idx]['img'], '*.dcm'))
        img_slice_name = img_slices[slice_idx]
        image = pydicom.dcmread(img_slice_name)
        image = apply_voi_lut(arr=image.pixel_array, ds=image)
        image = np.expand_dims(image, -1).astype(np.float32)
        image = np.repeat(image, 3, -1)

        try:
            mask = self.imgs[volume_idx]['gt'][slice_idx]
        except Exception as e:
            print(e)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        if self.input_image_transform is not None:
            image = self.input_image_transform(image)

        if self.mask_transform is not None:
            mask = self.transform(mask)

        return image, mask, img_slice_name

    def __len__(self):
        return 60 * len(self.imgs)


class YAMLSegmentationDataset_3D(data.Dataset):

    def __init__(self, dataset='/MYDIR/prostate.yml',
                 transform=None, input_image_transform=None, mask_transform=None,
                 split=['training'], classes=['Surface'], shape=(256, 256), n_channels=1):

        self.dataset = dataset
        self.transform = transform
        self.input_image_transform = input_image_transform
        self.mask_transform = mask_transform
        self.classes = classes if classes else ['Surface']
        self.shape = shape
        self.n_channels = n_channels
        self.imgs = []
        self.masks = {}
        self.counter = 0

        data_root = os.path.dirname(dataset)

        with open(self.dataset, 'r') as stream:
            try:
                d = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)

        for s in split:
            for i in d['split'][s]:
                img = {'img': os.path.join(data_root, d['images'][i]['location']).replace('\\', '/')}
                surface_mask = 0
                target_mask = 0
                for k, v in d['images'][i]['values'].items():
                    unix_path = os.path.join(data_root, d['images'][i]['values'][k]).replace('\\', '/')
                    if 'Surface' in self.classes and k == 'Surface':
                        surface_mask = np.load(unix_path)
                    if 'Target' in self.classes and 'Target' in k:
                        target_mask += np.load(unix_path)

                target_mask = np.clip(target_mask, 0, 1) * len(self.classes)

                img['gt'] = np.clip(surface_mask + target_mask, 0, 2)
                if img['gt'].size == 1 and img['gt'] == 0:
                    img['gt'] = np.zeros((60, *shape), dtype=np.uint8)
                self.imgs.append(img)

    def __getitem__(self, index):

        img_slices = glob(os.path.join(self.imgs[index]['img'], '*.dcm'))
        vol = [pydicom.dcmread(img_slice_name) for img_slice_name in img_slices]
        z_flip_flag = vol[1].ImagePositionPatient[-1] - vol[0].ImagePositionPatient[-1] < 0  # is this dicom acquired in the opposite direction (along Z)?
        vol = [apply_voi_lut(arr=i.pixel_array, ds=i) for i in vol]

        vol_name = self.imgs[index]['img']

        vol = np.asarray(vol).astype(np.float32)
        mask = self.imgs[index]['gt']

        if z_flip_flag:
            self.counter += 1
            vol = np.flip(vol, axis=0)
            mask = np.flip(mask, axis=0)

        if self.transform is not None:
            vol, mask = self.transform(vol, mask)

        if self.input_image_transform is not None:
            vol = self.input_image_transform(vol)

        if self.mask_transform is not None:
            mask = self.transform(mask)

        if self.n_channels != 1:
            vol = vol.repeat(self.n_channels, 1, 1, 1)

        return vol, mask, vol_name

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    yaml_f = '/MYDIR/prostate.yml'
    training_dataset = YAMLSegmentationDataset_2D(dataset=yaml_f, split=['training'])
    data_loader = DataLoader(dataset=training_dataset,
                             batch_size=2,
                             shuffle=False,
                             num_workers=0,
                             drop_last=True,
                             pin_memory=True)

    for data in data_loader:
        im, mask, name = data
        pass
