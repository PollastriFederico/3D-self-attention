import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import time
import numpy as np
import cv2

import torch
from torch import nn
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from torch.backends import cudnn
from torch.utils.data import DataLoader

from deep_learning.model.Model import Model
from deep_learning.utils import jaccard, count_pixels_multiclass_gpu

import data3D as public_prostata_data

segm_root = '/MYDIR/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SegmentationNet:
    def __init__(self, opt):
        self.num_epochs = opt.epochs
        self.lossname = opt.loss
        self.optimizer_name = opt.optimizer
        self.learning_rate = opt.learning_rate
        self.interp_size = opt.interp_size

        self.crop_size = tuple(opt.crop_size)
        self.crop_type = opt.crop_type
        self.batch_size = opt.batch_size
        self.eval_batch_size = 1
        self.n_workers = opt.workers
        self.thresh = opt.thresh
        self.plot_flag = opt.plot_flag
        self.num_classes = max(1, len(opt.dataset.split('_')[2:]))
        self.net_name = opt.network
        self.ckpt_name = opt.ckpt_name
        self.dname = opt.dataset

        self.outf = os.path.join(segm_root, opt.network, opt.split, opt.outf)
        self.outm = os.path.join(segm_root, opt.network, opt.split, opt.outm)

        exp_name = '3D' + self.net_name + '_' + self.dname + '_' + self.lossname + '_' + str(self.learning_rate)

        self.job_id = opt.job_id
        self.experiment_fullname = exp_name + '_' + self.job_id

        self.n = Model(num_classes=self.num_classes, reduction_factor=8)

        self.n.to(DEVICE)

        self.criterion = None

        self.dataset, self.val_dataset, self.test_dataset = public_prostata_data.get_dataset(dname=opt.dataset,
                                                                                             interp_size=self.interp_size,
                                                                                             crop_size=self.crop_size,
                                                                                             crop_type=self.crop_type,
                                                                                             net=self.net_name
                                                                                             )
        weights = public_prostata_data.compute_weights(self.dataset, self.interp_size, self.crop_size, self.num_classes)
        self.criterion = public_prostata_data.get_criterion(self.lossname, self.dname, weights)

        if self.criterion is None:
            if self.num_classes == 1:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                print("no criterion because of wrong dataset param")

        if self.optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.n.parameters()),
                                              lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.n.parameters()),
                                             lr=self.learning_rate)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', verbose=True,
                                                                        patience=7)

        self.data_loader = DataLoader(self.dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.n_workers,
                                      drop_last=True,
                                      pin_memory=True)

        self.eval_data_loader = DataLoader(self.val_dataset,
                                           batch_size=self.eval_batch_size,
                                           shuffle=False,
                                           num_workers=self.n_workers,
                                           drop_last=False,
                                           pin_memory=True)

        self.test_data_loader = DataLoader(self.test_dataset,
                                           batch_size=self.eval_batch_size,
                                           shuffle=False,
                                           num_workers=self.n_workers,
                                           drop_last=False,
                                           pin_memory=True)

        if not os.path.exists(self.outf):
            os.makedirs(self.outf)
        if not os.path.exists(self.outm):
            os.makedirs(self.outm)

        self.total = len(self.data_loader)
        print(len(self.data_loader))
        print(len(self.eval_data_loader))

    def train(self, epochs=None):
        if opt.SRV:
            try:
                self.writer = SummaryWriter(log_dir=os.path.join(segm_root, 'runs', self.experiment_fullname),
                                            purge_step=opt.load_epoch)
            except:
                print("COULD NOT CREATE TENSORBOARD WRITER")
        if epochs:
            self.num_epochs = epochs
        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch)
            print("TEST:", end=" ")
            slice_iou, slice_dice, vol_iou, vol_dice = self.eval(d_loader=self.test_data_loader)
            self.writer.add_scalar('Test/Slice_IoU', slice_iou, epoch)
            self.writer.add_scalar('Test/Slice_DICE', slice_dice, epoch)
            self.writer.add_scalar('Test/Vol_IoU', vol_iou, epoch)
            self.writer.add_scalar('Test/Vol_DICE', vol_dice, epoch)
            print("VALIDATION:", end=" ")
            slice_iou, slice_dice, vol_iou, vol_dice = self.eval(d_loader=self.eval_data_loader)
            self.writer.add_scalar('Validation/Slice_IoU', slice_iou, epoch)
            self.writer.add_scalar('Validation/Slice_DICE', slice_dice, epoch)
            self.writer.add_scalar('Validation/Vol_IoU', vol_iou, epoch)
            self.writer.add_scalar('Validation/Vol_DICE', vol_dice, epoch)
            if self.optimizer_name == 'SGD':
                self.scheduler.step(slice_iou)
                if self.learning_rate // self.optimizer.param_groups[0]['lr'] >= 10 ** 4:
                    print("Training process will be stopped now due to the low learning rate reached")
                    self.save()
                    return

            if epoch % 10 == 9:
                self.save()
                # self.test()

    def train_one_epoch(self, epoch):
        self.n.train()
        losses = []
        real_start_time = time.time()
        for i, (x, target, _) in enumerate(self.data_loader):

            # compute output
            x = x.to(DEVICE)
            if self.num_classes > 1 or self.lossname == 'focal':
                target = target.to(DEVICE, dtype=torch.long)
            else:
                target = target.to(DEVICE, dtype=torch.float)
            output = self.n(x)
            loss = self.criterion(output, target)
            losses.append(loss.item())
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print('\nEpoch: ' + str(epoch) + ' | loss: ' + str(np.mean(losses)) + ' | time: ' + str(
            time.time() - real_start_time))
        self.writer.add_scalar('Loss/Train', np.mean(losses), epoch)

    def eval(self, d_loader=None):
        if not d_loader:
            d_loader = self.eval_data_loader
        if 'public_prostata' in self.dname:
            return self.eval_imgwise(d_loader)
        else:
            raise NotImplementedError("Can not find an evaluation function for this dataset")

    def eval_imgwise(self, d_loader):
        if self.crop_type == 'center':
            post_process = public_prostata_data.CenterPad((60, self.interp_size, self.interp_size))
        elif self.crop_type == 'random':
            post_process = public_prostata_data.CenterCrop((60, self.interp_size, self.interp_size))
        else:
            raise ValueError('wrong type for crop type function')

        with torch.no_grad():
            self.n.eval()
            slice_iou = []
            vol_iou = []
            slice_dice = []
            vol_dice = []
            start_time = time.time()
            sigm = nn.Sigmoid()
            for i, (x, target, name) in enumerate(d_loader):
                # compute output
                x = x.to(DEVICE)
                target = target.to(DEVICE)
                output = self.n(x)
                output = sigm(output)

                output = post_process(output)
                output = nn.functional.interpolate(output[0], size=target.shape[-2:], mode='bilinear')

                iou, dice = count_pixels_multiclass_gpu(label_pred=output.unsqueeze(0), label_gt=target, num_classes=self.num_classes, threshold=0.5)
                vol_iou.append(iou.cpu().numpy())
                vol_dice.append(dice.cpu().numpy())
                iou, dice = count_pixels_multiclass_gpu(label_pred=output.unsqueeze(0), label_gt=target, num_classes=self.num_classes, threshold=0.5,
                                                        dims_to_keep=(2,))
                slice_iou.append(iou.cpu().numpy())
                slice_dice.append(dice.cpu().numpy())

                if self.plot_flag:
                    prediction = output.squeeze_().cpu().numpy()
                    gt = target.squeeze_().cpu().numpy()
                    x = post_process(x)
                    x = nn.functional.interpolate(x[0], size=target.shape[-2:], mode='bilinear')  # 3, Z, H, W
                    self.plot_contours(x, prediction, gt, name)

            print('Slices IoU: ' + str(np.mean(slice_iou)) + ' - Slices DICE: ' + str(np.mean(slice_dice)) +
                  ' Volume Iou: ' + str(np.mean(vol_iou)) + ' - Volume DICE: ' + str(np.mean(vol_dice)) + ' | time: ' + str(time.time() - start_time))
            return np.mean(slice_iou), np.mean(slice_dice), np.mean(vol_iou), np.mean(vol_dice)

    def test(self):
        with torch.no_grad():

            self.n.eval()
            acc = []
            start_time = time.time()
            for i, (x, target) in enumerate(self.test_data_loader):
                # compute output
                x = x.to(DEVICE)
                target = target.to(DEVICE)
                gt = target.squeeze_().cpu().numpy()
                output = self.n(x)
                output = nn.functional.interpolate(output, size=target.shape, mode='bilinear')
                prediction = output.squeeze_(1).squeeze_().cpu().numpy()
                acc.append(jaccard(prediction, gt, self.num_classes))

            print('Test set = Acc: ' + str(np.mean(acc)) + ' | time: ' + str(time.time() - start_time))
        if self.plot_flag:
            ffname = opt.outd + 'UNet_accuracies.txt'
            with open(ffname, 'a') as f:
                f.write(str(np.mean(acc)) + '\n')

    def save(self):
        try:
            torch.save(self.n.state_dict(),
                       os.path.join(self.outm, self.experiment_fullname + self.ckpt_name + '_net.pth'))
            torch.save(self.optimizer.state_dict(),
                       os.path.join(self.outm, self.experiment_fullname + self.ckpt_name + '_opt.pth'))
            print("model weights successfully saved")
        except Exception:
            print("Error during Saving")

    def load(self):
        try:
            self.n.load_state_dict(
                torch.load(os.path.join(self.outm, self.experiment_fullname + self.ckpt_name + '_net.pth')))
            self.optimizer.load_state_dict(
                torch.load(os.path.join(self.outm, self.experiment_fullname + self.ckpt_name + '_opt.pth')))
            print('Saved weights loaded successfully!')
        except:
            raise ValueError("could not load checkpoint file")

    def find_stats(self):
        mean = 0.
        std = 0.
        nb_samples = 0.
        for data, _ in self.data_loader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples
        print("\ntraining")
        print("mean: " + str(mean) + " | std: " + str(std))

        mean = 0.
        std = 0.
        nb_samples = 0.
        for data, _ in self.eval_data_loader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples
        print("\nvalidation")
        print("mean: " + str(mean) + " | std: " + str(std))

        mean = 0.
        std = 0.
        nb_samples = 0.
        for data, _ in self.test_data_loader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples
        print("\ntest")
        print("mean: " + str(mean) + " | std: " + str(std))

    def plot_output(self, d_loader=None):
        savedir = os.path.join(self.outf, self.net_name + '_' + self.dname + '_' + self.lossname + '_' +
                               str(self.learning_rate) + self.ckpt_name)
        sigm = nn.Sigmoid()
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        if not d_loader:
            d_loader = self.eval_data_loader
        with torch.no_grad():
            self.n.eval()

            for i, (x, target, name) in enumerate(d_loader):
                # compute output
                x = x.to(DEVICE)
                output = self.n(x)
                output = sigm(output)
                output = output.squeeze().cpu().numpy() * 255.0
                target = target.squeeze().cpu().numpy() * 255.0

                for slice in range(output.shape[0]):
                    pred = Image.fromarray(output[slice].astype(np.uint8))
                    gt = Image.fromarray(target[slice].astype(np.uint8))
                    uid = name[slice].find('Prostate-MRI-US-Biopsy-') + len('Prostate-MRI-US-Biopsy-')
                    uid = name[slice][uid:uid + 4]
                    uid += '-' + os.path.dirname(name[slice])[-5:]
                    img_name = uid + '-' + os.path.basename(name[slice])
                    pred.save(os.path.join(savedir, img_name.replace('.dcm', '_pred.png')))
                    gt.save(os.path.join(savedir, img_name.replace('.dcm', '_gt.png')))

    def plot_contours(self, x, output, target, name):
        savedir = os.path.join(self.outf, self.experiment_fullname + self.ckpt_name, 'contours')
        output = output * 255.0
        target = target * 255.0
        for slice in range(output.shape[0]):
            uid = name[0].find('Prostate-MRI-US-Biopsy-') + len('Prostate-MRI-US-Biopsy-')
            uid = name[0][uid:uid + 4]
            patient_savedir = os.path.join(savedir, uid)
            if not os.path.exists(patient_savedir):
                os.makedirs(patient_savedir)
            uid = name[0][-5:]
            img_name = uid + '-' + str(slice) + '.png'

            x.squeeze_()  # fix if volume is RGB
            if x.shape[0] == 3 and x.ndim == 4:
                x = x[0]

            img = x.squeeze()[slice].cpu().numpy()
            img = np.expand_dims(img, -1)
            img = np.repeat(img, 3, -1)
            img = img * 0.1369 + 0.2296
            img *= 255
            # img = np.swapaxes(img, 0, 1)
            _, output_thresh = cv2.threshold(output[slice].astype(np.uint8), 127, 255, 0)
            output_contours, _ = cv2.findContours(output_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            _, gt_thresh = cv2.threshold(target[slice].astype(np.uint8), 127, 255, 0)
            gt_contours, _ = cv2.findContours(gt_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            new_img = img.astype(np.uint8)
            if output_contours:
                new_img = cv2.drawContours(new_img.copy(), output_contours, -1, (0, 0, 255), thickness=1)
            if gt_contours:
                new_img = cv2.drawContours(new_img.copy(), gt_contours, -1, (0, 255, 0), thickness=1)
            cv2.imwrite(os.path.join(patient_savedir, img_name), new_img)


def parser3D():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='public_prostata')
    parser.add_argument('--network', default='Model')
    parser.add_argument('--split', default='prostate', help='dataset split')
    parser.add_argument('--outd', default=segm_root + 'Results', help='directory to save results')
    parser.add_argument('--outf', default=segm_root + 'Images', help='folder to save synthetic images')
    parser.add_argument('--outl', default=segm_root + 'Losses', help='folder to save Losses')
    parser.add_argument('--outm', default=segm_root + 'Models', help='folder to save models')
    parser.add_argument('--load_epoch', type=int, default=0, help='load pretrained models')

    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='list of batch sizes during the training')
    parser.add_argument('--interp_size', type=int, default=256,
                        help='the height / width of the input image after scaling')
    parser.add_argument('--crop_size', nargs='+', type=int, default=[56, 144, 144],
                        help='the height / width of the input image after central cropping')
    parser.add_argument('--crop_type', type=str, default='center', help='type of crop: center, random, ...')
    parser.add_argument('--loss', type=str, default="BCE_jaccard", help='Name of the loss to employ')
    parser.add_argument('--optimizer', type=str, default="SGD", help='Name of the optimizer to employ')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='number of data loading workers')
    parser.add_argument('--thresh', type=float, default=0.5, help='number of data loading workers')
    parser.add_argument('--epochs', type=int, default=201, help='number of data loading workers')
    parser.add_argument('--ckpt_name', default='')
    parser.add_argument('--SRV', action='store_true', help='is training on remote server')
    parser.add_argument('--plot_flag', action='store_true', help='prostaslice flag')
    parser.add_argument('--job_id', type=str, default='', help='job ID')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':

    cudnn.benchmark = True

    opt = parser3D()
    print(opt)

    n = SegmentationNet(opt=opt)

    # EVALUATING NETWORK
    if opt.epochs == 0:
        n.load()
        n.eval(d_loader=n.test_data_loader)

    # TRAINING NETWORK
    else:
        if opt.load_epoch:
            n.load()
            n.learning_rate = opt.learning_rate / 10 ** 3
            n.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, n.n.parameters()), n.learning_rate)
            n.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(n.optimizer, mode='max', verbose=True, patience=7)
        n.train(opt.epochs - opt.load_epoch)
        # n.eval(plot_flag=False)
        # n.test()
