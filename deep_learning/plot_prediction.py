from main import parser3D, SegmentationNet as SegmentationNet3D

from torch.backends import cudnn


segm_root = '/MYDIR/'

def slice_3D():
    opt = parser3D()
    print(opt)
    n = SegmentationNet3D(opt)
    n.load()
    n.eval(d_loader=n.test_data_loader)


if __name__ == '__main__':
    cudnn.benchmark = True
    slice_3D()





