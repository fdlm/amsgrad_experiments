from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.init import *


def logreg(in_shape, n_classes):
    net = InputLayer(shape=(None,) + in_shape, name='Input')
    net = DenseLayer(net, num_units=n_classes, nonlinearity=softmax,
                     name='Output', b=Constant(0.))
    return net


def small_vgg(in_shape, n_classes):
    """ Compile net architecture """
    nonlin = rectify

    def init_conv():
        return HeNormal('relu')

    def conv_bn(in_layer, num_filters, filter_size, nonlinearity, pad):
        in_layer = Conv2DLayer(in_layer, num_filters=num_filters,
                               filter_size=filter_size,
                               nonlinearity=nonlinearity, pad=pad, name='conv',
                               W=init_conv())
        in_layer = batch_norm(in_layer)
        return in_layer

    net1 = InputLayer(shape=(None, in_shape[0], in_shape[1], in_shape[2]), name='Input')

    # number of filters
    nf0 = 32
    pad = 'same'
    net1 = conv_bn(net1, num_filters=nf0, filter_size=3, nonlinearity=nonlin,
                   pad=pad)
    net1 = conv_bn(net1, num_filters=nf0, filter_size=3, nonlinearity=nonlin,
                   pad=pad)
    net1 = MaxPool2DLayer(net1, pool_size=2, stride=2, name='pool1')
    net1 = DropoutLayer(net1, p=0.25)

    net1 = conv_bn(net1, num_filters=nf0*2, filter_size=3, nonlinearity=nonlin,
                   pad=pad)
    net1 = conv_bn(net1, num_filters=nf0*2, filter_size=3, nonlinearity=nonlin,
                   pad=pad)
    net1 = MaxPool2DLayer(net1, pool_size=2, stride=2, name='pool2')
    net1 = DropoutLayer(net1, p=0.25)

    net1 = conv_bn(net1, num_filters=nf0*4, filter_size=3, nonlinearity=nonlin,
                   pad=pad)
    net1 = conv_bn(net1, num_filters=nf0*4, filter_size=3, nonlinearity=nonlin,
                   pad=pad)
    net1 = MaxPool2DLayer(net1, pool_size=2, stride=2, name='pool2')
    net1 = DropoutLayer(net1, p=0.25)

    net1 = conv_bn(net1, num_filters=512, filter_size=3, nonlinearity=nonlin,
                   pad='valid')
    net1 = DropoutLayer(net1, p=0.5)
    net1 = conv_bn(net1, num_filters=512, filter_size=1, nonlinearity=nonlin,
                   pad='valid')
    net1 = DropoutLayer(net1, p=0.5)

    net1 = conv_bn(net1, num_filters=n_classes, filter_size=1,
                   nonlinearity=nonlin, pad=pad)
    net1 = GlobalPoolLayer(net1)
    net1 = FlattenLayer(net1)
    net1 = NonlinearityLayer(net1, nonlinearity=softmax)

    return net1


def vgg(in_shape, n_classes):
    nonlin = rectify

    def init_conv():
        return HeNormal('relu')

    l_in = InputLayer(shape=(None, in_shape[0], in_shape[1], in_shape[2]),
                      name='Input')

    net = Conv2DLayer(l_in, num_filters=64, filter_size=3, pad=1,
                      W=init_conv(), nonlinearity=nonlin, name='Conv')
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=64, filter_size=3, pad=1, W=init_conv(),
                      nonlinearity=nonlin, name='Conv')
    net = batch_norm(net)
    net = MaxPool2DLayer(net, pool_size=2, name='Pool')
    net = DropoutLayer(net, p=0.25, name='Dropout')

    net = Conv2DLayer(net, num_filters=128, filter_size=3, pad=1,
                      W=init_conv(), nonlinearity=nonlin, name='Conv')
    net = Conv2DLayer(net, num_filters=128, filter_size=3, pad=1,
                      W=init_conv(), nonlinearity=nonlin, name='Conv')
    net = batch_norm(net)
    net = MaxPool2DLayer(net, pool_size=2, name='Pool')
    net = DropoutLayer(net, p=0.25, name='Dropout')

    net = Conv2DLayer(net, num_filters=256, filter_size=3, pad=1,
                      W=init_conv(), nonlinearity=nonlin, name='Conv')
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=256, filter_size=3, pad=1,
                      W=init_conv(), nonlinearity=nonlin, name='Conv')
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=256, filter_size=3, pad=1,
                      W=init_conv(), nonlinearity=nonlin, name='Conv')
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=256, filter_size=3, pad=1,
                      W=init_conv(), nonlinearity=nonlin, name='Conv')
    net = batch_norm(net)
    net = MaxPool2DLayer(net, pool_size=2, name='Pool')
    net = DropoutLayer(net, p=0.25, name='Dropout')

    net = Conv2DLayer(net, num_filters=1024, filter_size=3, pad=0,
                      W=init_conv(), nonlinearity=nonlin, name='Conv')
    net = batch_norm(net)
    net = DropoutLayer(net, p=0.5, name='Dropout')
    net = Conv2DLayer(net, num_filters=1024, filter_size=1, pad=0,
                      W=init_conv(), nonlinearity=nonlin, name='Conv')
    net = batch_norm(net)
    net = DropoutLayer(net, p=0.5, name='Dropout')

    net = Conv2DLayer(net, num_filters=n_classes, filter_size=1, W=init_conv(),
                      nonlinearity=nonlin, name='Conv')
    net = batch_norm(net)
    net = GlobalPoolLayer(net)
    net = DenseLayer(net, num_units=n_classes, nonlinearity=softmax)
    return net


def cifarnet(in_shape, n_classes):
    nonlin = rectify

    def init_weights():
        return HeNormal('relu')

    init_bias_const = 0.

    net = InputLayer(shape=(None, in_shape[0], in_shape[1], in_shape[2]),
                     name='Input')
    net = Conv2DLayer(net, num_filters=64, filter_size=6, pad='valid',
                      W=init_weights(), b=Constant(init_bias_const),
                      nonlinearity=nonlin, name='Conv')
    net = LocalResponseNormalization2DLayer(net)
    net = MaxPool2DLayer(net, pool_size=2)
    net = Conv2DLayer(net, num_filters=64, filter_size=6, pad='valid',
                      W=init_weights(), b=Constant(init_bias_const),
                      nonlinearity=nonlin, name='Conv')
    net = LocalResponseNormalization2DLayer(net)
    net = MaxPool2DLayer(net, pool_size=2)
    net = DenseLayer(net, num_units=384, W=init_weights(), b=Constant(0.))
    net = DropoutLayer(net, p=0.5, name='Dropout')
    net = DenseLayer(net, num_units=192, W=init_weights(), b=Constant(0.))
    net = DenseLayer(net, num_units=n_classes, nonlinearity=softmax,
                     W=init_weights(), b=Constant(0.))

    return net

