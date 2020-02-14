from __future__ import print_function

import os
from os.path import isfile
import h5py
import numpy as np
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from P4DCNN_model import p4dcnn_3L
import keras.backend as K
import pickle

MODEL_CONFIG = '3L_SPS_ESP'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
SAMPLE_NUM = 30
BATCH_SIZE = 16
# HEIGHT = 48
# WIDTH = 48
PATCH_SIZE = 48
UPSAMPLING_FACTOR = 4
CHANNEL = 1
L2_REG = 1e-4
SEED = 25536
MOMENTUM_VAL = 0.9
LR_ANGULAR_SIZE = 3
SR_ANGULAR_SIZE = 9
LR_BASE = 1e-4
LR_POWER = 0.9
MAX_EPOCHS = 100
LOSS_LAMDA = 10.0
DILATION_RATE = 1
# SPS loss
ORI_VIEW_LAMADA = 0.1
FIRST_STAGE_LAMADA = 1.0
SECOND_STAGE_LAMADA = 2.0

sobelFilter = K.variable([[[[1.,  1.]], [[0.,  2.]], [[-1.,  1.]]],
                          [[[2.,  0.]], [[0.,  0.]], [[-2.,  0.]]],
                          [[[1., -1.]], [[0., -2.]], [[-1., -1.]]]])


def gen_view_prior_coet(batchSize, lrAngReso, hrAngReso):
    if lrAngReso > 0 and lrAngReso < hrAngReso:
        npCoetArr = np.zeros(
            [batchSize, hrAngReso, hrAngReso, 1], dtype=np.float32)
        upFactor = (hrAngReso - 1) // (lrAngReso - 1)
        for s in range(hrAngReso):
            for t in range(hrAngReso):
                if s % upFactor == 0 and t % upFactor == 0:
                    npCoetArr[:, s, t] = ORI_VIEW_LAMADA
                elif s % upFactor == 0 and t % upFactor != 0:
                    npCoetArr[:, s, t] = FIRST_STAGE_LAMADA
                elif s % upFactor != 0 and t % upFactor == 0:
                    npCoetArr[:, s, t] = FIRST_STAGE_LAMADA
                else:
                    npCoetArr[:, s, t] = SECOND_STAGE_LAMADA
    else:
        raise IOError('Input not > 1 or lr < hr.')

    # print('Prior Sensitive Loss Coet Shape: ')
    # print(npCoetArr.shape)

    return K.constant(npCoetArr)


def load_data(path):
    """the data is scalaed in [0 1]"""

    f = h5py.File(path, 'r')
    # lr_data = np.asarray(f.get('train_data')[:], dtype=np.float32).transpose((0, 3, 4, 1, 2))
    # hr_data = np.asarray(f.get('train_label')[:], dtype=np.float32).transpose((0, 3, 4, 1, 2))
    # v_lr_data = np.asarray(f.get('valid_data')[:], dtype=np.float32).transpose((0, 3, 4, 1, 2))
    # v_hr_data = np.asarray(f.get('valid_label')[:], dtype=np.float32).transpose((0, 3, 4, 1, 2))

    lr_data = np.asarray(f.get('train_data')[:], dtype=np.float32)
    hr_data = np.asarray(f.get('train_label')[:], dtype=np.float32)
    v_lr_data = np.asarray(f.get('valid_data')[:], dtype=np.float32)
    v_hr_data = np.asarray(f.get('valid_label')[:], dtype=np.float32)

    nrof_train_samples = lr_data.shape[0]
    tvs = nrof_train_samples // BATCH_SIZE * BATCH_SIZE

    nrof_valid_samples = v_lr_data.shape[0]
    vvs = nrof_valid_samples // BATCH_SIZE * BATCH_SIZE

    print('='*30)
    print('Reading LF data from ', path)
    print('Train data Size', lr_data.shape,
          ' Range: ', lr_data.max(), lr_data.min())
    print('Train label Size', hr_data.shape,
          ' Range: ', hr_data.max(), hr_data.min())
    print('Validation data Size', v_lr_data.shape,
          ' Range: ', v_lr_data.max(), v_lr_data.min())
    print('Validation label size', v_hr_data.shape,
          ' Range: ', v_lr_data.max(), v_lr_data.min())
    print('='*30)

    return lr_data[:tvs, :, :, :, :, np.newaxis], \
        hr_data[:tvs, :, :, :, :, np.newaxis], \
        v_lr_data[:vvs, :, :, :, :, np.newaxis], \
        v_hr_data[:vvs, :, :, :, :, np.newaxis]


def expandedSobel(inputTensor):

    #this considers data_format = 'channels_last'
    inputChannels = K.reshape(K.ones_like(
        inputTensor[0, 0, 0, :]), (1, 1, -1, 1))
    #if you're using 'channels_first', use inputTensor[0,:,0,0] above

    return sobelFilter * inputChannels


def SPS_loss(y_true, y_pred):

    # Reshape lf data to [batch_size, s, t, x*y*channel]
    y_true_r = K.reshape(
        y_true, [BATCH_SIZE, SR_ANGULAR_SIZE, SR_ANGULAR_SIZE, -1])
    y_pred_r = K.reshape(
        y_pred, [BATCH_SIZE, SR_ANGULAR_SIZE, SR_ANGULAR_SIZE, -1])
    view_prior_coet = gen_view_prior_coet(
        BATCH_SIZE, LR_ANGULAR_SIZE, SR_ANGULAR_SIZE)

    return K.mean(view_prior_coet * K.square(y_pred_r - y_true_r))


def ESP_loss(y_true, y_pred):

    # Permute and Reshape
    y_true_r = K.reshape(K.permute_dimensions(y_true, [0, 1, 3, 2, 4, 5]), [
                         BATCH_SIZE, SR_ANGULAR_SIZE, PATCH_SIZE, -1])
    y_pred_r = K.reshape(K.permute_dimensions(y_pred, [0, 1, 3, 2, 4, 5]), [
                         BATCH_SIZE, SR_ANGULAR_SIZE, PATCH_SIZE, -1])

    filt = expandedSobel(y_true_r)
    sobelTrue = K.depthwise_conv2d(y_true_r, filt)
    sobelPred = K.depthwise_conv2d(y_pred_r, filt)

    return K.mean(K.abs(sobelPred - sobelTrue))


def SPS_ESP_loss(y_true, y_pred):

    spsLoss = SPS_loss(y_true, y_pred)
    espLoss = ESP_loss(y_true, y_pred)

    return LOSS_LAMDA * espLoss + spsLoss


def MAE_ESP_loss(y_true, y_pred):

    spsLoss = SPS_loss(y_true, y_pred)

    return LOSS_LAMDA * spsLoss + K.mean(K.abs(y_pred - y_true))


def psnr_metric(y_true, y_pred):

    y_true_r = K.round(y_true * 255.0)
    y_pred_r = K.round(y_pred * 255.0)

    err = K.mean(K.square(y_true_r - y_pred_r))

    return 20.0 * K.log(K.sqrt(err)) / K.log(K.constant(10.0))


def lr_scheduler(epoch):

    mode = 'power_decay'

    if mode is 'power_decay':
        # original lr scheduler
        lr = LR_BASE * ((1 - float(epoch)/MAX_EPOCHS) ** LR_POWER)
    if mode is 'exp_decay':
        # exponential decay
        lr = (float(LR_BASE) ** float(LR_POWER)) ** float(epoch+1)
    # adam default lr
    if mode is 'adam':
        lr = 0.001

    if mode is 'progressive_drops':
        # drops as progression proceeds, good for sgd
        if epoch > 0.9 * MAX_EPOCHS:
            lr = 0.0001
        elif epoch > 0.75 * MAX_EPOCHS:
            lr = 0.001
        elif epoch > 0.5 * MAX_EPOCHS:
            lr = 0.01
        else:
            lr = 0.1
    #
    print('--- learning rate: %f' % lr)
    return lr


scheduler = LearningRateScheduler(lr_scheduler)

model = p4dcnn_3L(lr_angular_size=LR_ANGULAR_SIZE,
                  height=PATCH_SIZE,
                  width=PATCH_SIZE,
                  channel=CHANNEL,
                  hr_angular_size=SR_ANGULAR_SIZE,
                  dilation_rate=DILATION_RATE,
                  use_residual=True)
# print(model.summary())

load_pre_model = True
pre_model_file = './model/{}_x{}.hdf5'.format(MODEL_CONFIG, UPSAMPLING_FACTOR)
save_model_file = './model/{}_x{}.hdf5'.format(MODEL_CONFIG, UPSAMPLING_FACTOR)

checkpointer = ModelCheckpoint(
    filepath=save_model_file, verbose=1, save_best_only=True)

# tensorboarad_monitor = TensorBoard(log_dir='./p4d_logs', histogram_freq=0, write_graph=True,
#                                    write_images=False, embeddings_freq=0,
#                                    embeddings_layer_names=None, embeddings_metadata=None)

reduce_lr_on_plateau = ReduceLROnPlateau(patience=5)

early_stopping = EarlyStopping(monitor='val_loss', patience=20)

if load_pre_model and isfile(pre_model_file):
    print('Loading Pre-trained model from %s...' % pre_model_file)
    model.load_weights(pre_model_file, by_name=True)
    print('Done.')

dataset_path = 'P4DCNN_Train_c%d_s%d_l%d_f%d_p%d.hdf5' % (
    CHANNEL, SAMPLE_NUM, SR_ANGULAR_SIZE, UPSAMPLING_FACTOR, PATCH_SIZE)
# train_set_x, train_set_y, valid_set_x, valid_set_y = load_data(dataset_path)
train_set_x, train_set_y, valid_set_x, valid_set_y = load_data(dataset_path)

# DFC_model_parallel = multi_gpu_model(DFC_model, gpus=4)
optimizer = SGD(lr=LR_BASE, momentum=0.99)

# optimizer = RMSprop(lr=LR_BASE)

# EPI structure preserving loss
print('='*50)
print('------- {} -------'.format(MODEL_CONFIG))
print('='*50)

model.compile(optimizer=optimizer,
              loss=SPS_ESP_loss,
              metrics=['mae'])

training_historty = model.fit(x=train_set_x,
                              y=train_set_y,
                              batch_size=BATCH_SIZE,
                              epochs=MAX_EPOCHS,
                              callbacks=[checkpointer],
                              validation_data=(valid_set_x, valid_set_y))

with open('./model/history.pkl', 'wb') as his_file:
    pickle.dump(training_historty.history, his_file)
    his_file.close()
