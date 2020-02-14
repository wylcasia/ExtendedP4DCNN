from __future__ import print_function

import os
import gc
import time
import cv2
from datetime import datetime
import numpy as np
from argparse import ArgumentTypeError, ArgumentParser
import skimage.io as io
import scipy.io as sio
from skimage import color, img_as_float, transform
from math import sqrt
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()


UPSAMPLING_FACTOR = 3
HEIGHT = 376
WIDTH = 541
CHANNEL = 1
LR_ANGULAR_SIZE = 3
SR_ANGULAR_SIZE = 7


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Unsupported value encountered.')

def opts_parser():

    usage = "Bicubic 4D upsample"
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '-D', '--path', type=str, default=None, dest='path',
        help='Loading 4D training and validation LF from this path: (default: %(default)s)')
    parser.add_argument(
        '-S', '--save_path', type=str, default=None, dest='save_path',
        help='Save Upsampled LF to this path: (default: %(default)s)')
    parser.add_argument(
        '-e', '--ext', type=str, default='png', dest='ext',
        help='Format of view images: (default: %(default)s)')
    parser.add_argument(
        '--adjust_tone', type=float, default=0.0, dest='adjust_tone',
        help='Image Filename: (default: %(default)s)')
    parser.add_argument(
        '-f', '--factor', type=int, default=2, dest='factor',
        help='Angular Upsampling factor: (default: %(default)s)')
    parser.add_argument(
        '-c', '--crop_length', type=int, default=7, dest='crop_length',
        help='Crop Length from Initial LF: (default: %(default)s)')
    parser.add_argument(
        '-a', '--angular_size', type=int, default=-1, dest='angular_size',
        help='Angular Resolution of Initial LF: (default: %(default)s)')
    parser.add_argument(
        '--save_results', type=str2bool, default=True, dest='save_results',
        help='Save Results or Not: (default: %(default)s)')

    return parser


def AdjustTone(img, coef, norm_flag=False):

    log.info('--------------')
    log.info('Adjust Tone')

    tic = time.time()
    rgb = np.zeros(img.shape)
    img = np.clip(img, 0.0, 1.0)
    output = img ** (1/coef)
    output = color.rgb2hsv(output)
    output[:, :, 1] = output[:, :, 1] * coef
    output = color.hsv2rgb(output)
    if norm_flag:
        r = output[:, :, 0]
        g = output[:, :, 1]
        b = output[:, :, 2]
        rgb[:, :, 0] = (r-r.min())/(r.max()-r.min())
        rgb[:, :, 1] = (g-g.min())/(g.max()-g.min())
        rgb[:, :, 2] = (b-b.min())/(b.max()-b.min())
    else:
        rgb = output

    log.info('IN Range: %.2f-%.2f' % (img.min(), img.max()))
    log.info('OUT Range: %.2f-%.2f' % (output.min(), output.max()))
    log.info("Elapsed time: %.2f sec" % (time.time() - tic))
    log.info('--------------')

    return rgb


def FolderTo4DLF(path, ext, length):
    path_str = path+'/*.'+ext
    log.info('-'*40)
    log.info('Loading %s files from %s' % (ext, path))
    img_data = io.ImageCollection(path_str)
    if len(img_data) == 0:
        raise IOError('No .%s file in this folder' % ext)
    # print(len(img_data))
    # print img_data[3].shape
    N = int(sqrt(len(img_data)))
    if not(N**2 == len(img_data)):
        raise ValueError('This folder does not have n^2 images!')
    if len(img_data[0].shape) == 3:
        [height, width, channel] = img_data[0].shape
        print(channel, ' Channels, RGB input.')
    elif len(img_data[0].shape) == 2:
        [height, width] = img_data[0].shape
        print('1 Channels, Grayscale input.')
        channel = 1
    else:
        raise ValueError('Not 1 or 3 channels')

    lf_shape = (N, N, height, width, 3)
    log.info('Initial LF shape: '+str(lf_shape))
    border = int((N-length)/2)
    if border < 0:
        raise ValueError('Border {0} < 0'.format(border))
    out_lf_shape = (length, length, height, width, 3)
    log.info('Output LF shape: '+str(out_lf_shape))
    lf = np.zeros(out_lf_shape).astype(np.float32)
    # save_path = './DATA/train/001/Coll/'
    for i in range(border, N-border, 1):
        for j in range(border, N-border, 1):
            indx = j + i*N
            if channel == 3:
                im = color.rgb2ycbcr(np.uint8(img_data[indx]))
            else:
                gray_img = np.uint8(img_data[indx])
                rgb_im = np.stack([gray_img, gray_img, gray_img], axis=2)
                im = color.rgb2ycbcr(rgb_im)

            lf[i-border, j-border, :, :, 0] = im[:, :, 0]/255.0
            lf[i-border, j-border, :, :, 1:3] = im[:, :, 1:3]
            # io.imsave(save_path+str(indx)+'.png',img_data[indx])
    log.info('LF Range:')
    log.info('Channel 1 [%.2f %.2f]' % (lf[:, :, 0, :, :].max(), lf[:, :, 0, :, :].min()))
    log.info('Channel 2 [%.2f %.2f]' % (lf[:, :, 1, :, :].max(), lf[:, :, 1, :, :].min()))
    log.info('Channel 3 [%.2f %.2f]' % (lf[:, :, 2, :, :].max(), lf[:, :, 2, :, :].min()))
    log.info('--------------------')
    return lf


def ImgTo4DLF(filename, unum, vnum, length, adjust_tone, save_sub_flag=False):

    if save_sub_flag:
        subaperture_path = os.path.splitext(filename)[0]+'_GT/'
        if not(os.path.exists(subaperture_path)):
            os.mkdir(subaperture_path)

    rgb_uint8 = io.imread(filename)
    rgb = np.asarray(img_as_float(rgb_uint8))
    log.info('Image Shape: %s' % str(rgb.shape))

    height = int(rgb.shape[0]/vnum)
    width = int(rgb.shape[1]/unum)
    channel = rgb.shape[2]

    if channel > 3:
        log.info('  Bands/Channels >3 Convert to RGB')
        rgb = rgb[:, :, 0:3]
        channel = 3

    if adjust_tone > 0.0:
        rgb = AdjustTone(rgb, adjust_tone)

    lf_shape = (vnum, unum, height, width, channel)
    lf = np.zeros(lf_shape).astype(np.float32)
    log.info('Initial LF shape: '+str(lf_shape))
    for i in range(vnum):
        for j in range(unum):
            im = rgb[i::vnum, j::unum, :]
            if save_sub_flag:
                subaperture_name = subaperture_path+'View_%d_%d.png' % (i+1, j+1)
                io.imsave(subaperture_name, im)
            lf[i, j, :, :, :] = color.rgb2ycbcr(im)
            lf[i, j, :, :, 0] = lf[i, j, :, :, 0]/255.0

    if unum % 2 == 0:
        border = (unum-length) // 2 + 1
        u_start_indx = int(border)
        u_stop_indx = int(unum - border + 1)
        v_start_indx = int(border)
        v_stop_indx = int(vnum - border + 1)
    else:
        border = (unum-length) // 2
        u_start_indx = int(border)
        u_stop_indx = int(unum - border)
        v_start_indx = int(border)
        v_stop_indx = int(vnum - border)

    if border < 0:
        raise ValueError('Border {0} < 0'.format(border))

    out_lf = lf[v_start_indx:v_stop_indx, u_start_indx:u_stop_indx, :, :, :]
    log.info('Output LF shape: '+str(out_lf.shape))

    log.info('LF Range:')
    log.info('Channel 1 [%.2f %.2f]' % (out_lf[:, :, :, :, 0].max(), out_lf[:, :, :, :, 0].min()))
    log.info('Channel 2 [%.2f %.2f]' % (out_lf[:, :, :, :, 1].max(), out_lf[:, :, :, :, 1].min()))
    log.info('Channel 3 [%.2f %.2f]' % (out_lf[:, :, :, :, 2].max(), out_lf[:, :, :, :, 2].min()))
    log.info('--------------------')
    return out_lf


def del_files(path, ext):
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(ext):
                os.remove(os.path.join(root, name))

def getSceneNameFromPath(path, ext, angular_size=-1):
    sceneNamelist = []
    if angular_size > 1:
        for root, dirs, files in os.walk(path):
            # print(files)
            for name in files:
                print(name)
                if name.endswith(ext):
                    sceneName = os.path.splitext(name)[0]
                    sceneNamelist.append(sceneName)
    else:
        for root, dirs, files in os.walk(path):
            # print(files)
            for scene_dir in dirs:
                sceneNamelist.append(scene_dir)

    sceneNamelist.sort()
    print(len(sceneNamelist), ' scenes in total, ', sceneNamelist)
    return tuple(sceneNamelist)

def pad_lf(in_lf):
    if len(in_lf.shape) == 4:
        (in_u, in_v, in_h, in_w) = in_lf.shape
        out_lf = np.zeros((in_u, in_v, HEIGHT, WIDTH), dtype=np.float32)
        out_lf[:, :, :in_h, :in_w] = in_lf
    else:
        (in_u, in_v, in_h, in_w, in_c) = in_lf.shape
        out_lf = np.zeros((in_u, in_v, HEIGHT, WIDTH, in_c), dtype=np.float32)
        out_lf[:, :, :in_h, :in_w, :] = in_lf

    return out_lf

def Upsample4D(
    path=None,
    save_path=None,
    ext='png',
    angular_size=-1,
    crop_length=7,
    factor=3,
    adjust_tone=0.0,
    save_results=False
):
    options = locals().copy()

    if path is not None:
        log.info('='*40)
        if not os.path.exists(path):
            raise IOError('No such folder: {}'.format(path))
        # if save_path is None:
        #     save_path = path+'_up4d/'
        # if not os.path.exists(save_path):
        #     log.warning('No such path for saving Our results, creating dir {}'
        #                 .format(save_path))
        #     os.mkdir(save_path)
        save_path = path + '_up4d'
        os.makedirs(save_path, exist_ok=True)

        sceneNameTuple = getSceneNameFromPath(path, ext, angular_size)

        if len(sceneNameTuple) == 0:
            raise IOError('Not any .%s file found in %s' % (ext, path))
        else:
            scene_num = len(sceneNameTuple)
    else:
        raise NameError('No folder given.')

    log_file = os.path.join(save_path, 'test_%s.log'
                            % datetime.now().strftime("%Y%m%d%H%M"))
    fh = logging.FileHandler(log_file)
    log.addHandler(fh)

    total_PSNR = []
    total_SSIM = []
    total_Elapsedtime = []

    performacne_index_file = os.path.join(save_path, 'performance_stat.mat')

    # log.info('-'*40)
    # log.info('-'*40)
    # log.info('...Loading Pre-trained Model'+model_path)
    # model = p4dcnn_model(lr_angular_size=crop_length//factor + 1,
    #                      height=HEIGHT,
    #                      width=WIDTH,
    #                      channel=1,
    #                      hr_angular_size=crop_length,
    #                      dilation_rate=dilation_rate)
    # model.load_weights(model_path)
    # log.info('Model Successfully Loaded in.')
    # log.info('-'*40)
    # log.info('-'*40)

    for indx, scene in enumerate(sceneNameTuple):
        log.info('='*20+'['+str(indx+1)+'/'+str(len(sceneNameTuple))+'] '+str(scene)+'='*20)
        if save_results:
            our_save_path = os.path.join(save_path, scene + '_U4D')
            # GT_save_path = os.path.join(save_path, scene + '_GT')
            if os.path.isdir(our_save_path):
                log.info('-'*20)
                del_files(our_save_path, 'png')
                log.warning('Ours Save Path %s exists, delete all .png files' % our_save_path)
            else:
                os.mkdir(our_save_path)

            # if os.path.isdir(GT_save_path):
            #     del_files(GT_save_path, 'png')
            #     log.info('GT path %s exists, delete all .png files' % GT_save_path)
            # else:
            #     os.mkdir(GT_save_path)

        tic = time.time()

        if angular_size > 1:
            img_filename = os.path.join(path, scene+'.'+ext)
            lf = ImgTo4DLF(filename=img_filename, vnum=angular_size, unum=angular_size, length=crop_length,
                           adjust_tone=adjust_tone, save_sub_flag=False)
        else:
            scene_folder = os.path.join(path, scene)
            lf = FolderTo4DLF(scene_folder, ext, crop_length)

        (g_u, g_v, g_h, g_w, g_c) = lf.shape

        gt_lf = lf

        # if g_h != HEIGHT or g_w != WIDTH:
        #     log.warning('!!! Input LF h-w [%d %d] not required [%d %d] padding zeros to border !!!' %
        #                 (lf.shape[2], lf.shape[3], HEIGHT, WIDTH))
        #     log.info('='*40)
        #     gt_lf = pad_lf(lf)
        # else:
        #     gt_lf = lf

        input_sparse_lf = gt_lf[0:crop_length:factor,
                                0:crop_length:factor, :, :, 0]
        
        (s_u, s_v, s_h, s_w) = input_sparse_lf.shape

        log.info("Elapsed time: %.2f sec" % (time.time() - tic))

        log.info('-'*40)
        log.info('...Upsampling 4D Light fields')
        tic = time.time()
        intermediate_lf = np.zeros((g_u, s_v, g_h, s_w), dtype=np.float32)
        out_dense_lf = np.zeros((g_u, g_v, g_h, g_w), dtype=np.float32)
        log.info('First Along x-s Epi')
        for v in range(s_v):
            for w in range(s_w):
                uh_epi = input_sparse_lf[:,v,:,w]
                uh_epi = cv2.resize(uh_epi, (g_h, g_u), interpolation=cv2.INTER_LINEAR)
                intermediate_lf[:,v,:,w] = uh_epi
        log.info('Then Along y-t Epi')
        for u in range(g_u):
            for h in range(g_h):
                vw_epi = intermediate_lf[u,:,h,:]
                vw_epi = cv2.resize(vw_epi, (g_w, g_v), interpolation=cv2.INTER_LINEAR)
                out_dense_lf[u,:,h,:] = vw_epi
     
        s_res = out_dense_lf.shape[0]
        t_res = out_dense_lf.shape[1]
        x_res = out_dense_lf.shape[2]
        y_res = out_dense_lf.shape[3]
        # channel = out_dense_lf.shape[4]
        processed_time = time.time() - tic
        log.info("Elapsed time: %.2f sec" % processed_time)

        PSNR = []
        SSIM = []

        log.info('-'*40)
        log.info('Evaluation......')

        for s_n in range(s_res):
            for t_n in range(t_res):
                gt_img = lf[s_n, t_n, :, :, 0]
                # print('GT range: %.2f-%.2f' %(gt_img.min(),gt_img.max()))
                view_img = np.clip(out_dense_lf[s_n, t_n, :, :], gt_img.min(), gt_img.max())
                # if not(u % factor == 0 and v % factor == 0):
                # this_test_loss = np.sqrt(mse(view_img,gt_img))
                # this_PSNR = 20*math.log(1.0/this_test_loss,10)

                if s_n % factor == 0 and t_n % factor == 0:
                    log.info('[o] View %.2d_%.2d: Original view not included' % (s_n+1, t_n+1))
                else:
                    this_PSNR = psnr(view_img, gt_img)
                    this_SSIM = ssim(view_img, gt_img)
                    PSNR.append(this_PSNR)
                    SSIM.append(this_SSIM)
                    log.info('[n] View %.2d_%.2d: PSNR: %.2fdB SSIM: %.4f' % (s_n+1, t_n+1, this_PSNR, this_SSIM))


                if save_results:
                    filename = os.path.join(our_save_path, 'View_'+str(s_n+1)+'_'+str(t_n+1)+'.png')
                    # GTname = os.path.join(GT_save_path, 'View_'+str(s_n+1)+'_'+str(t_n+1)+'.png')
                    out_img = np.zeros((x_res, y_res, 3))
                    # gt_out_img = np.zeros((x_res, y_res, 3))

                    out_img[:, :, 0] = np.clip(view_img*255.0, 16.0, 235.0)
                    # gt_out_img[:, :, 0] = np.clip(gt_img*255.0, 16.0, 235.0)
                    # print('Max: %.2f Min: %.2f' %(out_img[:,:,0].max(),out_img[:,:,0].min()))
                    out_img[:, :, 1:3] = lf[s_n, t_n, :, :, 1:3]
                    # gt_out_img[:, :, 1:3] = lf[s_n, t_n, :, :, 1:3]
                    # print('Max: %.2f Min: %.2f' %(out_img[:,:,1].max(),out_img[:,:,1].min()))

                    out_img = color.ycbcr2rgb(out_img)
                    out_img = np.clip(out_img, 0.0, 1.0)
                    out_img = np.uint8(out_img*255.0)

                    # gt_out_img = color.ycbcr2rgb(gt_out_img)
                    # gt_out_img = np.clip(gt_out_img, 0.0, 1.0)
                    # gt_out_img = np.uint8(gt_out_img*255.0)

                    io.imsave(filename, out_img)
                    # io.imsave(GTname, gt_out_img)

        log.info('='*40)
        total_PSNR.append(np.mean(np.array(PSNR)))
        total_SSIM.append(np.mean(np.array(SSIM)))
        total_Elapsedtime.append(processed_time)
        log.info("Average PSNR: %.2f dB\nSSIM: %.4f\nElapsed time: %.2f sec" % (np.mean(np.array(PSNR)),
                                                                                np.mean(np.array(SSIM)), processed_time))
        gc.collect()
        log.info('='*40)

    log.info('='*6+'Average Performance on %d scenes' % scene_num+'='*6)
    log.info('PSNR: %.2f dB' % np.mean(np.array(total_PSNR)))
    log.info('SSIM: %.4f' % np.mean(np.array(total_SSIM)))
    log.info('Elapsed Time: %.2f sec' % np.mean(np.array(total_Elapsedtime)))
    log.info('='*40)

    embeded = dict(NAME=sceneNameTuple, PSNR=np.array(total_PSNR), SSIM=np.array(total_SSIM),
                   TIME=np.array(total_Elapsedtime))
    sio.savemat(performacne_index_file, embeded)


if __name__ == '__main__':

    parser = opts_parser()
    args = parser.parse_args()

    path = args.path
    save_path = args.save_path
    ext = args.ext
    factor = args.factor
    crop_length = args.crop_length
    adjust_tone = args.adjust_tone
    save_results = args.save_results
    angular_size = args.angular_size

    Upsample4D(path=path,
    ext=ext,
    factor=factor,
    adjust_tone=adjust_tone,
    save_path=save_path,
    crop_length=crop_length,
    angular_size=angular_size,
    save_results=save_results)







