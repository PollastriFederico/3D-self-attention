import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from glob import glob
from voxelize.voxelize import voxelize
import os
from scipy import ndimage
from matplotlib import pyplot as plt
import pathlib

DICOM_max = 4095.0


def prosta_slice(gt, data, id=0):
    save_dir = f'results/patient{str(id)}'
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    coords = np.argwhere(gt)
    data = np.stack([data, data, data], axis=-1)
    data[coords[:, 0], coords[:, 1], coords[:, 2]] = (0, 1, 0)
    for i in range(data.shape[0]):
        img = plt.imshow(data[i])
        img.set_cmap('hot')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{i}.png'), bbox_inches='tight', transparent=True)


def voxelization(dcm_path, stl, fill_holes=True):
    filenames = glob(os.path.join(dcm_path, '*.dcm'))
    datasets = [pydicom.dcmread(filename) for filename in filenames]
    volume = np.stack([apply_voi_lut(arr=images.pixel_array, ds=images) for images in datasets])
    x = np.array(datasets[0].ImageOrientationPatient[0:3])
    y = np.array(datasets[0].ImageOrientationPatient[3:6])
    z = (np.array(datasets[0].ImagePositionPatient) - np.array(datasets[-1].ImagePositionPatient)) / (
            1 - len(datasets))
    scale = np.append(np.array(datasets[0].PixelSpacing), (1, 1))

    t_matrix = np.column_stack((x, y, z, np.array(datasets[0].ImagePositionPatient)))
    t_matrix = t_matrix * scale
    t_matrix = np.vstack((t_matrix, [0, 0, 0, 1]))
    inv_t_matrix = np.linalg.inv(t_matrix)

    bounds = np.array(volume.shape) - 1
    prostata = np.zeros((*volume.shape, 4), dtype=np.uint8)
    for x, y, z in voxelize(stl, inv_t_matrix):
        try:
            prostata[z, y, x] = (0, 1, 0, 1)
        except IndexError as e:
            prostata[min(z, bounds[0]), min(y, bounds[1]), min(x, bounds[2])] = (0, 1, 0, 1)

    if fill_holes:
        prostata[ndimage.binary_fill_holes(prostata)] = 1
    # volume = volume.astype(np.float) / DICOM_max
    volume = volume.astype(np.float) / volume.max()
    return prostata, volume
