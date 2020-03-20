'''DenseNet and DenseNet-FCN models for Keras.

DenseNet is a network architecture where each layer is directly connected
to every other layer in a feed-forward fashion (within each dense block).
For each layer, the feature maps of all preceding layers are treated as
separate inputs whereas its own feature maps are passed on as inputs to
all subsequent layers. This connectivity pattern yields state-of-the-art
accuracies on CIFAR10/100 (with or without data augmentation) and SVHN.
On the large scale ILSVRC 2012 (ImageNet) dataset, DenseNet achieves a
similar accuracy as ResNet, but using less than half the amount of
parameters and roughly half the number of FLOPs.
'''

def DenseNet3D(input_shape=None,
               depth=40,
               nb_dense_block=3,
               growth_rate=12,
               nb_filter=-1,
               nb_layers_per_block=1,
               bottleneck=False,
               reduction=0.0,
               dropout_rate=0.0,
               weight_decay=1e-4,
               subsample_initial_block=False,
               include_top=True,
               input_tensor=None,
               pooling=None,
               classes=10,
               activation='softmax',
               transition_pooling='avg'):
  
    '''Instantiate the DenseNet architecture.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 224, 3)` (with `channels_last` dim ordering)
            or `(3, 224, 224, 224)` (with `channels_first` dim ordering).
            It should have exactly 4 inputs channels,
            and width and height should be no smaller than 8.
            E.g. `(224, 224, 224, 3)` would be one valid value.
        depth: number or layers in the DenseNet
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters. -1 indicates initial
            number of filters will default to 2 * growth_rate
        nb_layers_per_block: number of layers in each dense block.
            Can be a -1, positive integer or a list.
            If -1, calculates nb_layer_per_block from the network depth.
            If positive integer, a set number of layers per dense block.
            If list, nb_layer is used as provided. Note that list size must
            be nb_dense_block
        bottleneck: flag to add bottleneck blocks in between dense blocks
        reduction: reduction factor of transition blocks.
            Note : reduction value is inverted to compute compression.
        dropout_rate: dropout rate
        weight_decay: weight decay rate
        subsample_initial_block: Changes model type to suit different datasets.
            Should be set to True for ImageNet, and False for CIFAR datasets.
            When set to True, the initial convolution will be strided and
            adds a MaxPooling3D before the initial dense block.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True.
        activation: Type of activation at the top layer. Can be one of
            'softmax' or 'sigmoid'. Note that if sigmoid is used,
             classes must be 1.
        transition_pooling: `avg` for avg pooling (default), `max` for max pooling,
            None for no pooling during scale transition blocks. Please note that this
            default differs from the DenseNetFCN paper in accordance with the DenseNet
            paper.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid input shape.
    '''

    if activation not in ['softmax', 'sigmoid']:
        raise ValueError('activation must be one of "softmax" or "sigmoid"')

    if activation == 'sigmoid' and classes != 1:
        raise ValueError('sigmoid activation can only be used when classes = 1')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = __create_dense_net(classes, img_input, include_top, depth, nb_dense_block,
                           growth_rate, nb_filter, nb_layers_per_block, bottleneck,
                           reduction, dropout_rate, weight_decay,
                           subsample_initial_block, pooling, activation,
                           transition_pooling)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='densenet')

    return model

def DenseNet3D_121(input_shape=None,
                          bottleneck=True,
                          reduction=0.5,
                          dropout_rate=0.0,
                          weight_decay=1e-4,
                          include_top=False,
                          input_tensor=None,
                          pooling=None,
                          classes=10,
                          activation='softmax'):
    return DenseNet3D(input_shape, depth=121, nb_dense_block=3, growth_rate=24,
                      nb_filter=64, nb_layers_per_block=[4,4,4],
                      bottleneck=bottleneck, reduction=reduction,
                      dropout_rate=dropout_rate, weight_decay=weight_decay,
                      subsample_initial_block=True, include_top=include_top,
                      input_tensor=input_tensor,
                      pooling=pooling, classes=classes, activation=activation)
#nb filter was 64
#nb layers per block was 6,12,24,16
#nb dense blocks was 4
def name_or_none(prefix, name):
    return prefix + name if (prefix is not None and name is not None) else None

def __conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None,
                 weight_decay=1e-4, block_prefix=None):
    '''
    Adds a convolution layer (with batch normalization and relu),
    and optionally a bottleneck layer.

    # Arguments
        ip: Input tensor
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        bottleneck: if True, adds a bottleneck convolution block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        block_prefix: str, for unique layer naming

     # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.

    # Returns
        output tensor of block
    '''
    with K.name_scope('ConvBlock'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                               name=name_or_none(block_prefix, '_bn'))(ip)
        x = Activation('relu')(x)

        if bottleneck:
            inter_channel = nb_filter * 4

            x = Conv3D(inter_channel, (1, 1, 1), kernel_initializer='he_normal',
                       padding='same', use_bias=False,
                       kernel_regularizer=l2(weight_decay),
                       name=name_or_none(block_prefix, '_bottleneck_Conv3D'))(x)
            x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                                   name=name_or_none(block_prefix, '_bottleneck_bn'))(x)
            x = Activation('relu')(x)

        x = Conv3D(nb_filter, (3, 3, 3), kernel_initializer='he_normal', padding='same',
                   use_bias=False, name=name_or_none(block_prefix, '_Conv3D'))(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

    return x


def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False,
                  dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True,
                  return_concat_list=False, block_prefix=None):
    '''
    Build a dense_block where the output of each conv_block is fed
    to subsequent ones

    # Arguments
        x: input keras tensor
        nb_layers: the number of conv_blocks to append to the model
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        growth_rate: growth rate of the dense block
        bottleneck: if True, adds a bottleneck convolution block to
            each conv_block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: if True, allows number of filters to grow
        return_concat_list: set to True to return the list of
            feature maps along with the actual output
        block_prefix: str, for block unique naming

    # Return
        If return_concat_list is True, returns a list of the output
        keras tensor, the number of filters and a list of all the
        dense blocks added to the keras tensor

        If return_concat_list is False, returns a list of the output
        keras tensor and the number of filters
    '''
    with K.name_scope('DenseBlock'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
        x_list = [x]
        for i in range(nb_layers):
            cb = __conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay,
                              block_prefix=name_or_none(block_prefix, '_%i' % i))
            x_list.append(cb)
            x = concatenate([x, cb], axis=concat_axis)
            if grow_nb_filters:
                nb_filter += growth_rate
        if return_concat_list:
            return x, nb_filter, x_list
        else:
            return x, nb_filter


def __transition_block(ip, nb_filter, compression=1.0, weight_decay=1e-4,
                       block_prefix=None, transition_pooling='max'):
    '''
    Adds a pointwise convolution layer (with batch normalization and relu),
    and an average pooling layer. The number of output convolution filters
    can be reduced by appropriately reducing the compression parameter.

    # Arguments
        ip: input keras tensor
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        compression: calculated as 1 - reduction. Reduces the number
            of feature maps in the transition block.
        weight_decay: weight decay factor
        block_prefix: str, for block unique naming

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, nb_filter * compression, rows / 2, cols / 2)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows / 2, cols / 2, nb_filter * compression)`
        if data_format='channels_last'.

    # Returns
        a keras tensor
    '''
    with K.name_scope('Transition'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                               name=name_or_none(block_prefix, '_bn'))(ip)
        x = Activation('relu')(x)
        x = Conv3D(int(nb_filter * compression), (1, 1, 1), kernel_initializer='he_normal',
                   padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),
                   name=name_or_none(block_prefix, '_Conv3D'))(x)
        if transition_pooling == 'avg':
            x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)
        elif transition_pooling == 'max':
            x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(x)

        return x


def __transition_up_block(ip, nb_filters, type='deconv', weight_decay=1E-4,
                          block_prefix=None):
    '''Adds an upsampling block. Upsampling operation relies on the the type parameter.

    # Arguments
        ip: input keras tensor
        nb_filters: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        type: can be 'upsampling', 'subpixel', 'deconv'. Determines
            type of upsampling performed
        weight_decay: weight decay factor
        block_prefix: str, for block unique naming

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, rows * 2, cols * 2)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows * 2, cols * 2, nb_filter)` if data_format='channels_last'.

    # Returns
        a keras tensor
    '''
    with K.name_scope('TransitionUp'):

        if type == 'upsampling':
            x = UpSampling3D(name=name_or_none(block_prefix, '_upsampling'))(ip)
        elif type == 'subpixel':
            x = Conv3D(nb_filters, (3, 3, 3), activation='relu', padding='same',
                       kernel_regularizer=l2(weight_decay), use_bias=False,
                       kernel_initializer='he_normal',
                       name=name_or_none(block_prefix, '_Conv3D'))(ip)
            x = SubPixelUpscaling(scale_factor=2,
                                  name=name_or_none(block_prefix, '_subpixel'))(x)
            x = Conv3D(nb_filters, (3, 3, 3), activation='relu', padding='same',
                       kernel_regularizer=l2(weight_decay), use_bias=False,
                       kernel_initializer='he_normal',
                       name=name_or_none(block_prefix, '_Conv3D'))(x)
        else:
            x = Conv3DTranspose(nb_filters, (3, 3, 3), activation='relu', padding='same',
                                strides=(2, 2, 2), kernel_initializer='he_normal',
                                kernel_regularizer=l2(weight_decay),
                                name=name_or_none(block_prefix, '_Conv3DT'))(ip)
        return x


def __create_dense_net(nb_classes, img_input, include_top, depth=40, nb_dense_block=3,
                       growth_rate=12, nb_filter=-1, nb_layers_per_block=-1,
                       bottleneck=False, reduction=0.0, dropout_rate=None,
                       weight_decay=1e-4, subsample_initial_block=False, pooling=None,
                       activation='softmax', transition_pooling='avg'):
    ''' Build the DenseNet model

    # Arguments
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        depth: number or layers
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters. Default -1 indicates initial number
            of filters is 2 * growth_rate
        nb_layers_per_block: number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_block from the depth of the network.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
        bottleneck: add bottleneck blocks
        reduction: reduction factor of transition blocks. Note : reduction value is
            inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay rate
        subsample_initial_block: Changes model type to suit different datasets.
            Should be set to True for ImageNet, and False for CIFAR datasets.
            When set to True, the initial convolution will be strided and
            adds a MaxPooling3D before the initial dense block.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        activation: Type of activation at the top layer. Can be one of 'softmax' or
            'sigmoid'. Note that if sigmoid is used, classes must be 1.
        transition_pooling: `avg` for avg pooling (default), `max` for max pooling,
            None for no pooling during scale transition blocks. Please note that this
            default differs from the DenseNetFCN paper in accordance with the DenseNet
            paper.

    # Returns
        a keras tensor

    # Raises
        ValueError: in case of invalid argument for `reduction`
            or `nb_dense_block`
    '''
    with K.name_scope('DenseNet'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        if reduction != 0.0:
            if not (reduction <= 1.0 and reduction > 0.0):
                raise ValueError('`reduction` value must lie between 0.0 and 1.0')

        # layers in each dense block
        if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
            nb_layers = list(nb_layers_per_block)  # Convert tuple to list

            if len(nb_layers) != nb_dense_block:
                raise ValueError('If `nb_dense_block` is a list, its length must match '
                                 'the number of layers provided by `nb_layers`.')

            final_nb_layer = nb_layers[-1]
            nb_layers = nb_layers[:-1]
        else:
            if nb_layers_per_block == -1:
                assert (depth - 4) % 3 == 0, ('Depth must be 3 N + 4 '
                                              'if nb_layers_per_block == -1')
                count = int((depth - 4) / 3)

                if bottleneck:
                    count = count // 2

                nb_layers = [count for _ in range(nb_dense_block)]
                final_nb_layer = count
            else:
                final_nb_layer = nb_layers_per_block
                nb_layers = [nb_layers_per_block] * nb_dense_block

        # compute initial nb_filter if -1, else accept users initial nb_filter
        if nb_filter <= 0:
            nb_filter = 2 * growth_rate

        # compute compression factor
        compression = 1.0 - reduction

        # Initial convolution
        if subsample_initial_block:
            initial_kernel = (7, 7, 7)
            initial_strides = (2, 2, 2)
        else:
            initial_kernel = (3, 3, 3)
            initial_strides = (1, 1, 1)

        x = Conv3D(nb_filter, initial_kernel, kernel_initializer='he_normal',
                   padding='same', name='initial_Conv3D', strides=initial_strides,
                   use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

        if subsample_initial_block:
            x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                                   name='initial_bn')(x)
            x = Activation('relu')(x)
            x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)

        # Add dense blocks
        for block_idx in range(nb_dense_block - 1):
            x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter,
                                         growth_rate, bottleneck=bottleneck,
                                         dropout_rate=dropout_rate,
                                         weight_decay=weight_decay,
                                         block_prefix='dense_%i' % block_idx)
            # add transition_block
            x = __transition_block(x, nb_filter, compression=compression,
                                   weight_decay=weight_decay,
                                   block_prefix='tr_%i' % block_idx,
                                   transition_pooling=transition_pooling)
            nb_filter = int(nb_filter * compression)
        # The last dense_block does not have a transition_block
        #x = Flatten()(x)
        x, nb_filter = __dense_block(x, final_nb_layer, nb_filter, growth_rate,
                                     bottleneck=bottleneck, dropout_rate=dropout_rate,
                                     weight_decay=weight_decay,
                                     block_prefix='dense_%i' % (nb_dense_block - 1))
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5, name='final_bn')(x)
        x = Activation('softmax')(x)
        x = AveragePooling3D((7, 7, 4), strides=(1,1,1),padding='same')(x)
        '''
        if include_top:
            if pooling == 'avg':
                #x = GlobalAveragePooling3D(padding='same')(x)
                x = AveragePooling3D((7, 7, 4), strides=(1,1,1), name=pool_name_base,padding='same')(x)
            elif pooling == 'max':
                #x = GlobalMaxPooling3D(padding='same')(x)
                x = MaxPooling3D((7, 7, 4), strides=(1,1,1), name=pool_name_base,padding='same')(x)
            x = Dense(nb_classes, activation=activation)(x)
        else:
            if pooling == 'avg':
                #x = GlobalAveragePooling3D(padding='same')(x)
                x = AveragePooling3D((7, 7, 4), strides=(1,1,1), name=pool_name_base,padding='same')(x)
            elif pooling == 'max':
                #x = GlobalMaxPooling3D(padding='same')(x)
                x = MaxPooling3D((7, 7, 4), strides=(1,1,1), name=pool_name_base,padding='same')(x)
        '''
        x = Flatten()(x)
        x = Dense(nb_classes, activation='softmax')(x)
        return x