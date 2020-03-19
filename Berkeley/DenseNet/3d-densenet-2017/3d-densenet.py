from keras.models import Model
from keras.layers import Input, merge, ZeroPadding3D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution3D
from keras.layers.pooling import AveragePooling3D, GlobalAveragePooling3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
import keras.backend as K

from custom_layers import Scale

## Note: Based off of https://github.com/flyyufelix/DenseNet-Keras
## My comments are all ## (as opposed to single #)

def DenseNet(nb_dense_block=3, growth_rate=24, nb_filter=64, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, classes=1000, weights_path=None):
    '''Instantiate the DenseNet 121 architecture,
        # Arguments
            nb_dense_block: number of dense blocks to add to end
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters
            reduction: reduction factor of transition blocks.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            classes: optional number of classes to classify images
            weights_path: path to pre-trained weights
        # Returns
            A Keras model instance.
    '''
    ## nb_dense_block = 3 from paper
    ## k=24 from paper
    ## I'm not sure what nb_filter should be
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    ## Change the size of images here. Note, I think the 3 represents RGB layers? So a 3d spatio-temopral densenet should be (width, height, num_frames)
    ## Paper has images scaled to 100Hx100W, 16 frames
    # Handle Dimension Ordering for different backends
    global concat_axis
    concat_axis = 3
    img_input = Input(shape=(100, 100, 3, 16), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    ## The paper says x4 for each layer, 3 total
    nb_layers = [4,4,4] 

    ## Note: subsample = strides
    ## Convolution-3D: 7x7x7 conv, stride=2
    x = ZeroPadding3D((3, 3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution3D(nb_filter, 7, 7, 7, subsample=(2, 2, 2), name='conv1', bias=False)(x)

    ## Pooling-3D: 3x3x3 avg pool, stride=2
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding3D((1, 1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), name='pool1')(x)

    ## Add dense blocks 1, 2
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        ## Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    ## Add last dense block: 3 (since we don't have another transition after)
    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    ## Classification Layer
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
    
    ## For our paper, we want a 7x7x4 avg pool instead of a global average pool, assumed to be strides = 1 b/c not specified
    ## x = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x = AveragePooling3D((7, 7, 4), strides=(1,1,1), name=pool_name_base)(x)

    ## Fully connected (dense) 2D softmax
    ## Note: the original 2d densenet paper does a 1000D softmax, but I think this should be 2d since the pooling is no longer global.
    x = Dense(classes, name='fc6')(x)
    x = Activation('softmax', name='prob')(x)

    model = Model(img_input, x, name='densenet')

    if weights_path is not None:
      model.load_weights(weights_path)

    return model


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    ## 1x1x1 Convolution (Bottleneck layer)
    ## Note: I'm not really sure what this 4 is. This isn't the number of layers (see nb_layers above)
    inter_channel = nb_filter * 4  
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Convolution3D(inter_channel, 1, 1, 1, name=conv_name_base+'_x1', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    ## 3x3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding3D((1, 1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Convolution3D(nb_filter, 3, 3, 3, name=conv_name_base+'_x2', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage) 

    ## 1x1x1 convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Convolution3D(int(nb_filter * compression), 1, 1, 1, name=conv_name_base, bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    ## 2x2x2 avg pooling
    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter

