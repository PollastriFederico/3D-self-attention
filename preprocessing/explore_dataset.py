import pydicom
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import os
import yaml
from yaml import CLoader
from preprocessing.utils import voxelization, prosta_slice
from random import shuffle


def look_for_img(dir):
    # Prostate-MRI-US-Biopsy-0001\\06-28-2009-MRI PROSTATE W WO CONTRAST-51743\\11.000000-t2spcrstaxial oblProstate-90221
    files = glob(f'{dir}/*/*/*/1-01.dcm')

    null_rotation = np.array([1, 0, 0, 0, 1, 0])
    max_rot = 0
    max_f = "null"
    for f in files:
        img = pydicom.dcmread(f)
        rotation = np.abs(np.array(img.ImageOrientationPatient) - null_rotation)
        rotation = np.sum(rotation)

        if rotation > max_rot and rotation < 1:
            max_rot = rotation
            max_f = f

    print(f'il file {max_f} ha rotazione {max_rot}')


def count_slices(dir):
    """
    check the number of slices distribution over the original prostate dataset
    :param dir: path to the original prostate dataset
    :return:
    """
    h_list = []
    vols_list = []
    patients = glob(f'{dir}/*/')

    for p in patients:
        volumes_counter = 0
        subdirs = glob(f'{p}/*/*')
        for d in subdirs:
            files = glob(f'{d}/*.dcm')
            if len(files) > 1:
                h_list.append(len(files))
                volumes_counter += 1
        vols_list.append(volumes_counter)
    plt.hist(vols_list, bins=100)
    plt.show()
    print(f'I found {len([v for v in vols_list if v != 0])} patients with at least 1 biopsy,')
    print(f'out of {len(vols_list)} patients,')
    print(f'for a total of {sum(vols_list)} biopsies')
    plt.hist(h_list, bins=100)
    plt.show()


def count_rotated(dir):
    """
    count rotation of the patients in the original dataset
    :param dir: path to the original dataset
    :return:
    """
    files = glob(f'{dir}/*/*/*/1-01.dcm')
    null_rotation = np.array([1, 0, 0, 0, 1, 0])
    counter = 0
    for f in files:
        img = pydicom.dcmread(f)
        rotation = np.abs(np.array(img.ImageOrientationPatient) - null_rotation)
        rotation = np.sum(rotation)
        if rotation > 0.02:
            counter += 1
    print(f'I counted {counter} rotated images')


def fetch_stl(dcm, type='ProstateSurface'):
    stl_dir = 'E:\Prostata\public_dataset\\STLs\\'
    patient_dirname = os.path.basename(os.path.dirname(os.path.dirname(dcm)))
    uid = patient_dirname[-4:]
    if f'Prostate-MRI-US-Biopsy-{uid}' != patient_dirname:
        print(f'ERROR with {patient_dirname}')
    correct = dcm[-5:]
    f_names = glob(f'{stl_dir}Prostate-MRI-US-Biopsy-{uid}-{type}*{correct}.STL')
    return f_names


def get_instance_stl(dir, yamldir):
    dict = {}
    values_dict = {}
    surface = fetch_stl(dir, 'ProstateSurface')
    targets = fetch_stl(dir, 'Target')
    dict['location'] = dir.replace(yamldir, '')
    values_dict['Surface'] = surface[0].replace(yamldir, '')
    for i, t in enumerate(targets, start=1):
        values_dict[f'Target{i}'] = t.replace(yamldir, '')
    dict['values'] = values_dict
    # dict['values'] = [d[44:] for d in surface + targets]
    return dict


def check_matrix(dcm_path):
    """
    control the rotation error of a patient
    :param dcm_path: path to the patient DICOM files
    :return:
    """
    filenames = glob(os.path.join(dcm_path, '*.dcm'))
    datasets = [pydicom.dcmread(filename) for filename in filenames]
    x = np.array(datasets[0].ImageOrientationPatient[0:3])
    y = np.array(datasets[0].ImageOrientationPatient[3:6])
    our_z = np.cross(x, y) * datasets[0].SliceThickness
    new_z = (np.array(datasets[0].ImagePositionPatient) - np.array(datasets[-1].ImagePositionPatient)) / (
            1 - len(datasets))
    return np.sum(our_z - new_z)


def write_yaml_wstl(dir, yamldir):
    """
    use this function to create a clean yaml file from the original prostate dataset
    :param dir: path to the original prostate dataset
    :param yamldir: path where you want to save the yaml dir
    :return:
    """
    yaml_dict = {}
    imgs_list = []
    patients = glob(f'{dir}/*/')
    exclude = [r'Prostate-MRI-US-Biopsy-0711\01-13-2007-MRI PROSTATE W ENDORECTAL COIL-05630\8.000000-t2spcrstaxial oblProstate-58736']
    for p in patients:
        subdirs = glob(f'{p}/*/*')
        for d in subdirs:
            if d in exclude:
                continue
            files = glob(f'{d}/*.dcm')
            if len(files) == 60:
                imgs_list.append(get_instance_stl(d, yamldir))
                # pass
    yaml_dict['name'] = 'Prostate-MRI-US-Biopsy'
    yaml_dict['images'] = imgs_list
    with open(os.path.join(yamldir, 'prostate.yml'), 'w') as file:
        yaml.dump(yaml_dict, file)


def read_yaml(f):
    with open(f, 'r') as stream:
        try:
            d = yaml.load(stream, Loader=CLoader)
        except Exception as exc:
            print(exc)
    return d


def create_numpy(yaml_dir):
    """
    load meshes from the final yaml, voxelize them and save them as numpy file
    :param yaml_dir: final yaml dir
    :return:
    """
    dict = read_yaml(yaml_dir)
    base_path = os.path.dirname(yaml_dir)

    for i, im in enumerate(dict['images']):
        uid = int(im['values']['Surface'][28:32])
        print(f'UID: {uid}')
        err = check_matrix(os.path.join(base_path, im['location']).replace("\\","/"))
        if err < 2.9 or err > 3.1:
            print('this patient seems to be too rotated: ', im['location'])
        for name, label in im['values'].items():
            gt, _ = voxelization(os.path.join(base_path, im['location']).replace("\\","/"), os.path.join(base_path, label).replace("\\","/"))
            np.save(os.path.join(base_path, 'npys', os.path.basename(label)[:-4].replace("\\","/") + '.npy'), gt)


def check_dataset_matrixes(f):
    dict = read_yaml(f)
    base_path = os.path.dirname(f)
    counter = 0
    for i, im in enumerate(dict['images']):
        full_path = os.path.join(base_path, im['location'])
        err = check_matrix(full_path)
        if err > 1e-10:
            print(f'\nImage {full_path} -- counter: {i} -- contains an error of {err}\n')
            counter += 1
        if i % 100 == 0:
            print(f'{i} images checked')

    print(f'I found {counter} images with an error')


def yaml_stl2npy(f):
    """
    convert the yaml file once you have created the numpy volumes.
    .stl extension in the yaml must be replaced by .npy
    :param f: path to the yaml
    :return:
    """
    dict = read_yaml(f)
    imgs = dict['images']
    new_imgs = []
    for im in imgs:
        new_img = {'location': im['location'], 'values': {}}
        for k, v in im['values'].items():
            new_img['values'][k] = v.replace('STLs\\', 'npys\\').replace('.STL', '.npy')
        new_imgs.append(new_img)
    dict['images'] = new_imgs

    with open(f.replace('.yml', '_stl2npy.yml'), 'w') as file:
        yaml.dump(dict, file)

def yaml_npy2stl(f):
    """
    create the stl yaml file from the numpy yaml file
    :param f: path to the yaml
    :return:
    """
    dict = read_yaml(f)
    imgs = dict['images']
    new_imgs = []
    for im in imgs:
        new_img = {'location': im['location'], 'values': {}}
        for k, v in im['values'].items():
            new_img['values'][k] = v.replace('npys\\', 'STLs\\').replace('.npy', '.STL')
        new_imgs.append(new_img)
    dict['images'] = new_imgs

    with open(f.replace('.yml', '_stl.yml'), 'w') as file:
        yaml.dump(dict, file)


def plot(img):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show(block=False)


if __name__ == '__main__':

    # path to the yaml with STL dir
    yaml_dir = '/MYDIR/stl.yml'

    # generate numpy file from STL meshes
    create_numpy(yaml_dir)

    # write a new yaml with npy extension instead of .stl
    yaml_stl2npy(yaml_dir)

    # statistics about the number of slices per patient on the dataset
    # count_slices(dir='/path/to/dataset')

