__author__ = 'Wang Yunlong'

import numpy as np
import os
from argparse import ArgumentParser
import skimage.io as io
from skimage.transform import resize
from skimage import color
import time
import scipy.io as sio
import h5py
import math


def opts_parser():
    usage = "Generate Training and Test Datasets for Pseudo-4D CNN.\n" \
            "Error Code\n" \
            "10: This folder has Not n^2 images!\n" \
            "11: 'Length is larger than angularsize!'\n" \
            "12: 'Not RGB input!'\n" \
            "13: 'No such folder!'"
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '-D', '--path', type=str, default=None, dest='path',
        help='Loading 4D LF from this path: (default: %(default)s)')
    parser.add_argument(
        '-t', '--total_num', type=int, default=100, dest='total_num',
        help='Total numer of samples in this path: (default: %(default)s)')
    parser.add_argument(
        '-n', '--sample_num', type=int, default=25, dest='sample_num',
        help='Number of Samples in use: (default: %(default)s)')
    parser.add_argument(
        '-e', '--ext', type=str, default='png', dest='ext',
        help='Format of view images: (default: %(default)s)')
    parser.add_argument(
        '-l', '--length', type=int, default=9, dest='length',
        help='Length of 3D LF: (default: %(default)s)')
    parser.add_argument(
        '-f', '--factor', type=int, default=2, dest='factor',
        help='Angular Upsampling factor: (default: %(default)s)')
    parser.add_argument(
        '-p', '--patch_size', type=int, default=48, dest='patch_size',
        help='Patch Size: (default: %(default)s)')
    parser.add_argument(
        '-b', '--batch_size', type=int, default=64, dest='batch_size',
        help='Batch Size: (default: %(default)s)')
    parser.add_argument(
        '-s', '--stride', type=int, default=36, dest='stride',
        help='Length of 3D LF: (default: %(default)s)')
    parser.add_argument(
        '-r', '--ratio', type=float, default=0.8, dest='ratio',
        help='Ratio for splitting train and test datasets: (default: %(default)s)')
    parser.add_argument(
        '-k', '--blur_kernel', type=int, default=-1, dest='blur_kernel_size',
        help='Blur Kernel: (default: %(default)s)')

    return parser


def up_scale_len(length, factor):

    border = length % factor

    if border == 0:
        lr_len = length // factor
        hr_len = lr_len * factor
    elif border == 1:
        lr_len = length // factor + 1
        hr_len = (lr_len - 1) * factor + 1
    else:
        raise ValueError('Length {} factor {} border {} can not deal with.'.format(length, factor, border))
    return border, lr_len, hr_len


def get_init_len(lr_len, factor, border):
    if border == 0:
        hr_len = lr_len * factor
    else:
        hr_len = (lr_len - 1) * factor + 1

    return hr_len


def generateMain(path, total_num, sample_num, ext, length, factor, patch_size, stride, ratio, kernel_size):

    single_img_data = io.ImageCollection(path+'/*.'+ext)
    if single_img_data is None:
        raise ValueError('No Reference Image.')
    [height, width, channel] = single_img_data[0].shape
    x_len = len(range(0, height-patch_size, stride))
    y_len = len(range(0, width-patch_size, stride))

    sample_order = np.random.randint(total_num-1, size=sample_num)

    border_len, lr_len, hr_len = up_scale_len(length, factor)
    print('Ground Truth: %d*%d; Input sparse views: %d*%d' % (hr_len, hr_len, lr_len, lr_len))
    total_count = sample_num * x_len * y_len

    print('Total Samples Number: %d' % total_count)

    count = 0

    data_shape = (patch_size, patch_size, lr_len, lr_len, total_count)
    label_shape = (patch_size, patch_size, hr_len, hr_len, total_count)
    data = np.zeros(data_shape).astype(np.float32)
    label = np.zeros(label_shape).astype(np.float32)

    blur_kernel_file = 'BlurKernel.mat'
    if kernel_size > 1 and os.path.isfile(blur_kernel_file):
        blur_kernel = sio.loadmat(blur_kernel_file)['kernel']
        print(blur_kernel.shape)
        blur_kernel = resize(blur_kernel.astype(np.float32), [3, kernel_size], order=3)
        blur_kernel = blur_kernel / blur_kernel.sum()
        print('Blur Kernel')
        print(blur_kernel)
    else:
        blur_kernel = np.asarray([[0.0, 0.0, 0.0],
                                  [0.1214, 0.7572, 0.1214],
                                  [0.0, 0.0, 0.0]], dtype=np.float32)
        print('Blur Kernel')
        print(blur_kernel)

    for ss in range(sample_num):
        sample_index = sample_order[ss] + 1
        path_str = path + '/%.3d/*.' % sample_index + ext
        print('--------------------')
        print('Loading %s files from %s' % (ext, path + '/%.3d' % sample_index))
        img_data = io.ImageCollection(path_str)
        # print('Done.')
        # print(len(img_data))
        # print img_data[3].shape
        N = int(math.sqrt(len(img_data)))
        if not(N**2 == len(img_data)):
            print('Exit. This folder has not n*n images')
            os._exit(10)
        lf_shape = (N, N, height, width, channel)
        print('LF shape: %s' % str(lf_shape))
        border = int((N-length)/2)
        if border < 0:
            print('Exit. Length is larger than angularsize!')
            os._exit(11)
        [height, width, channel] = img_data[0].shape
        if not(channel == 3):
            print('Exit. Not RGB input!')
            os._exit(12)
        all_gt = np.zeros((height, width, length, length)).astype(np.float32)
        # all_lr_tmp = np.zeros((height,width,channel,int((length-1)/factor)+1))
        # print('all_lr_tmp shape: '+str(all_lr_tmp.shape))
        # all_lr = np.zeros((height,width,d_len,d_len)).astype(config.floatX)
        # lf_shape = (N,N,height,width,channel)
        # lf = np.zeros(lf_shape)
        # # save_path = './DATA/train/001/Coll/'
        print('Generating %d * %d patches' % (patch_size, patch_size))
        t0 = time.time()
        for i in range(border, N-border, 1):
            for j in range(border, N-border, 1):
                indx = j + i*N
                ori_subimg = color.rgb2ycbcr(np.uint8(img_data[indx]))[:, :, 0]/255.0
                # Blur the sub-aperture images with pre-defined blur kernels
                # all_gt[:, :, i-border, j-border] = convolve(ori_subimg, blur_kernel)
                all_gt[:, :, i-border, j-border] = ori_subimg

        all_lr = all_gt[:, :, 0:length:factor, 0:length:factor]

        for x in range(0, height-patch_size, stride):
            for y in range(0, width-patch_size, stride):
                label[:, :, :, :, count] = all_gt[x:x+patch_size, y:y+patch_size, :, :]
                data[:, :, :, :, count] = all_lr[x:x+patch_size, y:y+patch_size, :, :]
                count += 1

        print("Elapsed time: %.2f sec" % (time.time() - t0))

    print('--------------------')
    print('======Data Augmentation======')
    data = np.transpose(data, [4, 2, 3, 0, 1])
    label = np.transpose(label, [4, 2, 3, 0, 1])

    print('--------------------')
    print('Permuting Angular Dimensions')
    data = np.concatenate((data[:(count//2)], np.transpose(data[(count//2):], (0, 2, 1, 3, 4))), axis=0)
    label = np.concatenate((label[:(count//2)], np.transpose(label[(count//2):], (0, 2, 1, 3, 4))), axis=0)

    print('--------------------')
    print('Shuffling Along First Axis')
    order = np.random.permutation(count)
    data = data[order, :, :, :, :]
    label = label[order, :, :, :, :]
    # data = np.random.shuffle(data)
    # label = np.random.shuffle(label)

    train_data = data[:int(ratio*count)]
    train_label = label[:int(ratio*count)]

    print('--------------------')
    print('Train DATA Size: '+str(train_data.shape))
    print('Train LABEL Size: '+str(train_label.shape))

    test_data = data[int(ratio*count):]
    test_label = label[int(ratio*count):]

    print('--------------------')
    print('Test DATA Size: '+str(test_data.shape))
    print('Test LABEL Size: '+str(test_label.shape))

    return train_data, train_label, test_data, test_label

if __name__ == '__main__':

    parser = opts_parser()
    args = parser.parse_args()

    path = args.path
    total_num = args.total_num
    sample_num = args.sample_num
    ext = args.ext
    length = args.length
    factor = args.factor
    patch_size = args.patch_size
    stride = args.stride
    ratio = args.ratio
    blur_kernel_size = args.blur_kernel_size

    global_channel = 1

    print('--------------------')
    print('Summary')
    print('Path: %s' % path)
    print('Total Num: %d' % total_num)
    print('Sample Num: %d' % sample_num)
    print('Format of Images: %s' % ext)
    print('Length: %d' % length)
    print('Upsampling Factor: %d' % factor)
    print('Patch Size: %d' % patch_size)
    print('Stride: %d' % stride)
    print('Ratio: %f' %ratio)
    print('Blur Kernel Size: %s' %blur_kernel_size)

    if not(os.path.exists(path)):
        print('Exit. No such folder!')
        os._exit(13)

    t0 = time.time()
    [train_data, train_label, valid_data, valid_label] = generateMain(path=path, total_num=total_num,
                                                                      sample_num=sample_num, ext=ext, length=length,
                                                                      factor=factor, patch_size=patch_size,
                                                                      stride=stride, ratio=ratio,
                                                                      kernel_size=blur_kernel_size)

    data_filename = 'P4DCNN_Train_c%d_s%d_l%d_f%d_p%d.hdf5' % (global_channel, sample_num, length, factor, patch_size)
    print('--------------------')
    print('Saving to %s file' % data_filename)
    border_len, lr_len, hr_len = up_scale_len(length, factor)
    print('Ground Truth: %d*%d views; Input: %d*%d views; Border: %d view' % (hr_len, hr_len, lr_len, lr_len, border_len))

    f = h5py.File(data_filename, 'w')
    train_group = f.create_group("Train")
    valid_group = f.create_group("Valid")
    # label_chunksize = (64, patch_size, patch_size, hr_len, hr_len)
    # data_chunksize = (64, patch_size, patch_size, lr_len, lr_len)
    label_chunksize = (64, hr_len, hr_len, patch_size, patch_size)
    data_chunksize = (64, lr_len, lr_len, patch_size, patch_size)
    train_data_des = f.create_dataset("train_data", data=train_data, chunks=data_chunksize, compression="gzip")
    train_label_des = f.create_dataset("train_label", data=train_label, chunks=label_chunksize, compression="gzip")
    valid_data_des = f.create_dataset("valid_data", data=valid_data, chunks=data_chunksize, compression="gzip")
    valid_label_des = f.create_dataset("valid_label", data=valid_label, chunks=label_chunksize, compression="gzip")
    train_group["data"] = train_data_des
    train_group["label"] = train_label_des
    valid_group["data"] = valid_data_des
    valid_group["label"] = valid_label_des
    print("Total Elapsed time: %.2f sec" % (time.time() - t0))



