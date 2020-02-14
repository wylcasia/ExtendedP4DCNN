from __future__ import print_function

from keras.models import Model
from keras.utils import plot_model
from keras.layers import Input, Conv3D, Conv2DTranspose, TimeDistributed, Permute, Add, Reshape
from keras import initializers, regularizers


def p4dcnn_6L(lr_angular_size=3, height=48, width=48, channel=1, hr_angular_size=7,
                 L2_REG=1e-4, dilation_rate=3, use_residual=True):

    sparse_lf_input = Input((lr_angular_size, lr_angular_size, height, width, channel))

    x = Permute((1, 3, 2, 4, 5))(sparse_lf_input)

    x = Reshape((-1, lr_angular_size, width, channel))(x)

    bilinear_x = TimeDistributed(Conv2DTranspose(channel, (5, 5), strides=(hr_angular_size // (lr_angular_size - 1), 1),
                                                 padding='same', output_padding=(0, 0)))(x)

    # print(bilinear_x.shape)

    # bilinear_x = TimeDistributed(Conv2D(filters=channel,
    #                                     kernel_size=(3, 3),
    #                                     padding='same',
    #                                     activation='relu',
    #                                     # dilation_rate=(3, 3),
    #                                     kernel_regularizer=regularizers.l2(L2_REG)))(bilinear_x)

    bilinear_x = Reshape([lr_angular_size, height, hr_angular_size, width, channel])(bilinear_x)

    bilinear_x = Permute((1, 3, 2, 4, 5))(bilinear_x)

    x = TimeDistributed(Conv3D(filters=64,
                               kernel_size=(3, 5, 5),
                               padding='same',
                               activation='relu',
                               dilation_rate=dilation_rate,
                               kernel_regularizer=regularizers.l2(L2_REG)),
                        name='conv3d_s1_TD')(bilinear_x)

    x = TimeDistributed(Conv3D(filters=16,
                               kernel_size=(3, 3, 3),
                               padding='same',
                               activation='relu',
                               dilation_rate=2,
                               kernel_regularizer=regularizers.l2(L2_REG)),
                        name='conv3d_s2_TD')(x)

    x = TimeDistributed(Conv3D(filters=16,
                               kernel_size=(3, 3, 3),
                               padding='same',
                               activation='relu',
                               dilation_rate=dilation_rate,
                               kernel_regularizer=regularizers.l2(L2_REG)),
                        name='conv3d_s3_TD')(x)

    x = TimeDistributed(Conv3D(filters=16,
                               kernel_size=(3, 3, 3),
                               padding='same',
                               activation='relu',
                               dilation_rate=dilation_rate,
                               kernel_regularizer=regularizers.l2(L2_REG)),
                        name='conv3d_s4_TD')(x)

    x = TimeDistributed(Conv3D(filters=64,
                               kernel_size=(3, 3, 3),
                               padding='same',
                               activation='relu',
                               dilation_rate=dilation_rate,
                               kernel_regularizer=regularizers.l2(L2_REG)),
                        name='conv3d_s5_TD')(x)

    x = TimeDistributed(Conv3D(filters=channel,
                               kernel_size=(3, 9, 9),
                               padding='same',
                               activation='linear',
                               dilation_rate=dilation_rate,
                               kernel_regularizer=regularizers.l2(L2_REG)),
                        name='conv3d_s6_TD')(x)

    if use_residual:
        x = Add(name='Add_U')([x, bilinear_x])

    x = Permute((2, 4, 1, 3, 5))(x)

    x = Reshape((-1, lr_angular_size, height, channel))(x)

    bilinear_x = TimeDistributed(Conv2DTranspose(channel, (5, 5), strides=(hr_angular_size // (lr_angular_size - 1), 1),
                                                 padding='same', output_padding=(0, 0)))(x)

    # bilinear_x = TimeDistributed(Conv2D(filters=channel,
    #                                     kernel_size=(3, 3),
    #                                     padding='same',
    #                                     activation='relu',
    #                                     # dilation_rate=(3, 3),
    #                                     kernel_regularizer=regularizers.l2(L2_REG)))(bilinear_x)

    bilinear_x = Reshape([hr_angular_size, width, hr_angular_size, height, channel])(bilinear_x)

    bilinear_x = Permute((1, 3, 4, 2, 5))(bilinear_x)

    x = TimeDistributed(Conv3D(filters=64,
                               kernel_size=(3, 5, 5),
                               padding='same',
                               activation='relu',
                               dilation_rate=dilation_rate,
                               kernel_regularizer=regularizers.l2(L2_REG)),
                        name='conv3d_s7_TD')(bilinear_x)

    x = TimeDistributed(Conv3D(filters=16,
                               kernel_size=(3, 3, 3),
                               padding='same',
                               activation='relu',
                               dilation_rate=dilation_rate,
                               kernel_regularizer=regularizers.l2(L2_REG)),
                        name='conv3d_s8_TD')(x)

    x = TimeDistributed(Conv3D(filters=16,
                               kernel_size=(3, 3, 3),
                               padding='same',
                               activation='relu',
                               dilation_rate=dilation_rate,
                               kernel_regularizer=regularizers.l2(L2_REG)),
                        name='conv3d_s9_TD')(x)

    x = TimeDistributed(Conv3D(filters=16,
                               kernel_size=(3, 3, 3),
                               padding='same',
                               activation='relu',
                               dilation_rate=dilation_rate,
                               kernel_regularizer=regularizers.l2(L2_REG)),
                        name='conv3d_s10_TD')(x)

    x = TimeDistributed(Conv3D(filters=64,
                               kernel_size=(3, 3, 3),
                               padding='same',
                               activation='relu',
                               dilation_rate=dilation_rate,
                               kernel_regularizer=regularizers.l2(L2_REG)),
                        name='conv3d_s11_TD')(x)

    x = TimeDistributed(Conv3D(filters=channel,
                               kernel_size=(3, 9, 9),
                               padding='same',
                               activation='linear',
                               dilation_rate=dilation_rate,
                               kernel_regularizer=regularizers.l2(L2_REG)),
                        name='conv3d_s12_TD')(x)

    if use_residual:
        x = Add(name='Add_V')([x, bilinear_x])

    dense_lf_output = Permute((2, 1, 3, 4, 5))(x)

    model = Model(inputs=sparse_lf_input, outputs=dense_lf_output)

    # plot_model(p4dcnn_model, to_file='p4dcnn_model.png', show_shapes=True, show_layer_names=True)

    print(model.summary())

    return model


def p4dcnn_3L(lr_angular_size=3, height=48, width=48, channel=1, hr_angular_size=7,
                 L2_REG=1e-4, dilation_rate=3, use_residual=True):

    sparse_lf_input = Input((lr_angular_size, lr_angular_size, height, width, channel))

    x = Permute((1, 3, 2, 4, 5))(sparse_lf_input)

    x = Reshape((-1, lr_angular_size, width, channel))(x)

    bilinear_x = TimeDistributed(Conv2DTranspose(channel, (5, 5), strides=(hr_angular_size // (lr_angular_size - 1), 1),
                                                 padding='same', output_padding=(0, 0)))(x)

    # print(bilinear_x.shape)

    # bilinear_x = TimeDistributed(Conv2D(filters=channel,
    #                                     kernel_size=(3, 3),
    #                                     padding='same',
    #                                     activation='relu',
    #                                     # dilation_rate=(3, 3),
    #                                     kernel_regularizer=regularizers.l2(L2_REG)))(bilinear_x)

    bilinear_x = Reshape([lr_angular_size, height, hr_angular_size, width, channel])(bilinear_x)

    bilinear_x = Permute((1, 3, 2, 4, 5))(bilinear_x)

    x = TimeDistributed(Conv3D(filters=64,
                               kernel_size=(3, 5, 5),
                               padding='same',
                               activation='relu',
                               dilation_rate=dilation_rate,
                               kernel_regularizer=regularizers.l2(L2_REG)),
                        name='conv3d_s1_R')(bilinear_x)

    x = TimeDistributed(Conv3D(filters=32,
                               kernel_size=(3, 1, 1),
                               padding='same',
                               activation='relu',
                               dilation_rate=2,
                               kernel_regularizer=regularizers.l2(L2_REG)),
                        name='conv3d_s2_R')(x)

    x = TimeDistributed(Conv3D(filters=channel,
                               kernel_size=(3, 9, 9),
                               padding='same',
                               activation='linear',
                               dilation_rate=dilation_rate,
                               kernel_regularizer=regularizers.l2(L2_REG)),
                        name='conv3d_s3_R')(x)

    if use_residual:
        x = Add(name='Add_U')([x, bilinear_x])

    x = Permute((2, 4, 1, 3, 5))(x)

    x = Reshape((-1, lr_angular_size, height, channel))(x)

    bilinear_x = TimeDistributed(Conv2DTranspose(channel, (5, 5), strides=(hr_angular_size // (lr_angular_size - 1), 1),
                                                 padding='same', output_padding=(0, 0)))(x)

    # bilinear_x = TimeDistributed(Conv2D(filters=channel,
    #                                     kernel_size=(3, 3),
    #                                     padding='same',
    #                                     activation='relu',
    #                                     # dilation_rate=(3, 3),
    #                                     kernel_regularizer=regularizers.l2(L2_REG)))(bilinear_x)

    bilinear_x = Reshape([hr_angular_size, width, hr_angular_size, height, channel])(bilinear_x)

    bilinear_x = Permute((1, 3, 4, 2, 5))(bilinear_x)

    x = TimeDistributed(Conv3D(filters=64,
                               kernel_size=(3, 5, 5),
                               padding='same',
                               activation='relu',
                               dilation_rate=dilation_rate,
                               kernel_regularizer=regularizers.l2(L2_REG)),
                        name='conv3d_s1_C')(bilinear_x)

    x = TimeDistributed(Conv3D(filters=32,
                               kernel_size=(3, 1, 1),
                               padding='same',
                               activation='relu',
                               dilation_rate=2,
                               kernel_regularizer=regularizers.l2(L2_REG)),
                        name='conv3d_s2_C')(x)

    x = TimeDistributed(Conv3D(filters=channel,
                               kernel_size=(3, 9, 9),
                               padding='same',
                               activation='linear',
                               dilation_rate=dilation_rate,
                               kernel_regularizer=regularizers.l2(L2_REG)),
                        name='conv3d_s6_C')(x)

    if use_residual:
        x = Add(name='Add_V')([x, bilinear_x])

    dense_lf_output = Permute((2, 1, 3, 4, 5))(x)

    model = Model(inputs=sparse_lf_input, outputs=dense_lf_output)

    # plot_model(p4dcnn_model, to_file='p4dcnn_model.png', show_shapes=True, show_layer_names=True)

    print(model.summary())

    return model












